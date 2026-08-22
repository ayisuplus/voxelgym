"""Replay & verify: rebuild the world from the sidecar seed + task scenario,
re-apply the recorded action sequence, assert checkpoint and final hashes.

Usage: python -m voxelgym.replay <shard.parquet> --verify
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pyarrow.parquet as pq

from .env import ACTION_KEYS


def _load(parquet_path: str):
    sidecar_path = parquet_path[: -len(".parquet")] + ".json"
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    table = pq.read_table(parquet_path)
    return sidecar, table.to_pylist()


def verify(parquet_path: str, verbose: bool = True) -> bool:
    import voxelgym_rs as rs

    sidecar, rows = _load(parquet_path)
    task_name = sidecar["task"]
    seed = sidecar["seed"]

    # Rebuild the exact reset path: task scenario + on_reset are deterministic
    # in the episode seed.
    from .tasks import make_task
    from .env import VoxelGymEnv

    task = make_task(task_name)
    env = VoxelGymEnv(task=task, preset=task.preset, seed=seed)
    env.reset(seed=seed)
    world = env.world

    scratch = rs.PyWorld(0, "void")
    checked_ckpts = 0
    for row in rows:
        if row.get("swap"):
            world.swap_to_hotbar(row["swap"])
        action = tuple(int(row[k]) for k in ACTION_KEYS)
        world.step(action)
        ckpt = row["world_ckpt"]
        if ckpt is not None:
            scratch.restore(ckpt)
            live = world.hash()
            snap = scratch.hash()
            if live != snap:
                print(f"CHECKPOINT MISMATCH at tick {row['tick']}: live={live:016x} ckpt={snap:016x}")
                return False
            checked_ckpts += 1
    final = world.hash()
    ok = final == sidecar["final_hash"]
    if verbose:
        print(f"task={task_name} seed={seed} steps={len(rows)} ckpts_checked={checked_ckpts}")
        print(f"final hash: {final:016x} vs sidecar {sidecar['final_hash']:016x}")
        print("PASS" if ok else "FAIL")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if not args.verify:
        print("nothing to do (pass --verify)")
        return 2
    return 0 if verify(args.parquet) else 1


if __name__ == "__main__":
    sys.exit(main())
