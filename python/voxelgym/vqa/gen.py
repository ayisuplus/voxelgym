"""VQA dataset generator: oracle expert episodes -> per-task NPZ tensors +
one shared JSONL manifest.

    python -m voxelgym.vqa.gen --tasks <csv> --episodes N --out data/vqa

Per task, `episodes` oracle episodes are run (circuit tasks: circuit_door,
plate_door, logic_probe get 2x — the circuit-only families need the volume
to reach >=300 samples/family). Samples are emitted at tick 8 (settle
state) and every 15 ticks. Each applicable family emits one QA per sample;
a family that crashes or is inapplicable is simply absent from that sample.
Failed episodes (expert miss / death) are skipped silently.

Determinism contract: rerunning with identical args must produce a
byte-identical manifest.jsonl (asserted by tests; the sim, the render, and
the per-sample rng = default_rng(seed*1000003 + tick) are all
deterministic, and rows are appended in fixed task/seed/tick/family order).

NPZ keys per task: id (S,) str + stacked tensors: rgb u8 (128,128,3),
depth f16 (128,128), normals f32 (128,128,3), seg u16 (128,128),
lidar_range f32 (16,256), voxels u16 (21,11,21), pose f32 (6,),
inventory u16 (36,2). seg is a label SOURCE only — never a model input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

from ..env import VoxelGymEnv
from ..experts import make_expert
from ..tasks import make_task
from .families import FAMILIES, Ctx

DEFAULT_TASKS = (
    "navigate_to_target", "collect_log", "smelt_iron", "circuit_door",
    "plate_door", "logic_probe", "collapse_judge",
)
CIRCUIT_TASKS = frozenset({"circuit_door", "plate_door", "logic_probe"})

LIDAR = {"channels": 16, "azimuth": 256, "min_elev": -20, "max_elev": 10, "max_range": 48}

# door cells are fixed per task scenario (anchors: tasks/probes.py
# CircuitDoor.scenario and PlateDoor.scenario both place the door at
# (10, 5, 0))
DOOR_CELLS = {"circuit_door": (10, 5, 0), "plate_door": (10, 5, 0)}

TENSOR_KEYS = ("rgb", "depth", "normals", "seg", "lidar_range", "voxels", "pose", "inventory")


def _ctx_for(task_name: str, task) -> Ctx:
    """Per-episode family anchors from the (already reset) task object."""
    if task_name == "logic_probe":
        lamp = task.lamp
        return Ctx(
            task=task_name,
            target=(lamp[0] + 0.5, float(lamp[1]), lamp[2] + 0.5),
            lamp=lamp,
            lever_a=task.LEVER_A,
            lever_b=task.LEVER_B,
        )
    t = getattr(task, "target", None) or getattr(task, "TARGET", None)
    return Ctx(
        task=task_name,
        target=tuple(float(v) for v in t) if t is not None else None,
        door=DOOR_CELLS.get(task_name),
    )


def _sample_rows(task_name, seed, tick, world, obs, ctx, crashes: Counter):
    rng = np.random.default_rng(seed * 1000003 + tick)
    rows = []
    for fam in FAMILIES:
        if task_name not in fam.tasks:
            continue
        try:
            out = fam.emit(world, obs, ctx, rng)
        except Exception:
            crashes[fam.name] += 1  # absent from this sample; never fatal
            continue
        if out is None:
            continue
        q_en, q_zh, ans = out
        rows.append({
            "id": f"{task_name}/{seed}/{tick}",
            "task": task_name,
            "seed": seed,
            "tick": tick,
            "family": fam.name,
            "q_en": q_en,
            "q_zh": q_zh,
            "answer": ans,
            "needs": sorted(fam.needs),
        })
    return rows


def _gen_episode(task_name: str, seed: int, crashes: Counter):
    """Run one oracle episode; return (tensors_by_key, manifest_rows) or
    None when the expert failed (truncated or died)."""
    task = make_task(task_name)
    env = VoxelGymEnv(task=task, seed=seed, render=1, lidar=LIDAR)
    env.reset(seed=seed)
    ctx = _ctx_for(task_name, task)
    expert = make_expert(task_name, task, seed=seed)
    samples = []  # (sample_id, obs)
    rows = []
    while True:
        a = expert.act(env.world)
        env.world.take_swap()  # keep expert inventory events flowing (as run_episode)
        obs, _r, term, trunc, _ = env.step(a)
        tick = env.world.tick()
        if tick == 8 or tick % 15 == 0:
            sid = f"{task_name}/{seed}/{tick}"
            samples.append((sid, {k: obs[k] for k in TENSOR_KEYS}))
            rows.extend(_sample_rows(task_name, seed, tick, env.world, obs, ctx, crashes))
        if term or trunc:
            break
    success = bool(term) and not env.world.dead()
    env.close()
    if not success:
        return None
    return samples, rows


def run(task_names, episodes: int, out_dir: str) -> Counter:
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    crashes: Counter = Counter()
    family_counts: Counter = Counter()
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for task_name in task_names:
            n_eps = episodes * 2 if task_name in CIRCUIT_TASKS else episodes
            all_samples: list = []
            n_ok = 0
            for i in range(n_eps):
                seed = i
                res = _gen_episode(task_name, seed, crashes)
                if res is None:
                    print(f"  {task_name} ep {i} seed={seed}: expert failed, skipped", flush=True)
                    continue
                samples, rows = res
                n_ok += 1
                all_samples.extend(samples)
                for row in rows:
                    mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    family_counts[row["family"]] += 1
                print(f"  {task_name} ep {i} seed={seed}: OK, {len(samples)} samples", flush=True)
            ids = [sid for sid, _ in all_samples]
            arrays = {
                k: np.stack([obs[k] for _, obs in all_samples]) for k in TENSOR_KEYS
            }
            np.savez_compressed(os.path.join(out_dir, f"{task_name}.npz"), id=np.array(ids), **arrays)
            print(f"{task_name}: {n_ok}/{n_eps} episodes, {len(all_samples)} samples", flush=True)
    if crashes:
        print(f"family emit crashes (samples skipped, never fatal): {dict(crashes)}", flush=True)
    print("per-family QA counts:", dict(sorted(family_counts.items())), flush=True)
    return family_counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                    help="comma-separated task names")
    ap.add_argument("--episodes", type=int, default=12,
                    help="episodes per task (circuit tasks get 2x)")
    ap.add_argument("--out", default="data/vqa")
    args = ap.parse_args(argv)
    run([t.strip() for t in args.tasks.split(",") if t.strip()], args.episodes, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
