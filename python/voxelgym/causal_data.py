"""Parallel local production of Dataset Manifest v1 and Training Pack v1."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
import hashlib
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

import numpy as np

from .config import ResearchConfig
from .training_pack import (
    assign_split,
    build_training_pack,
    bundle_sha256,
    write_dataset_manifest,
)


def benchmark_worker_counts(
    candidates: Iterable[int],
    *,
    trials: int = 3,
    max_memory_fraction: float = 0.70,
    steps_per_worker: int = 256,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure native simulator throughput using Windows-safe spawned workers."""

    results: list[dict[str, Any]] = []
    logical_cpus = os.cpu_count() or 1
    for requested in candidates:
        workers = min(int(requested), logical_cpus)
        throughputs: list[float] = []
        memory_fractions: list[float] = []
        for trial in range(int(trials)):
            started = time.perf_counter()
            jobs = [
                (
                    trial * 1_000_000 + index,
                    steps_per_worker,
                    dict(environment or {}),
                )
                for index in range(workers)
            ]
            with _pool(workers) as executor:
                completed = sum(executor.map(_benchmark_worker, jobs))
                # Spawned workers remain alive here, so system memory includes
                # their retained native/Python allocations instead of the
                # post-shutdown baseline.
                memory_fractions.append(_memory_fraction())
            elapsed = max(time.perf_counter() - started, 1e-9)
            throughputs.append(completed / elapsed)
        results.append(
            {
                "requested_workers": int(requested),
                "workers": workers,
                "median_steps_per_second": statistics.median(throughputs),
                "peak_observed_memory_fraction": max(memory_fractions),
            }
        )
    eligible = [
        result
        for result in results
        if result["peak_observed_memory_fraction"] < max_memory_fraction
    ]
    if not eligible:
        raise RuntimeError(
            f"all worker candidates exceeded the {max_memory_fraction:.0%} memory gate"
        )
    selected = max(eligible, key=lambda result: result["median_steps_per_second"])
    return {
        "selected_workers": selected["workers"],
        "results": results,
        "environment": dict(environment or {}),
    }


def build_causal_dataset(
    config: ResearchConfig,
    *,
    benchmark_workers: bool = True,
    build_pack: bool = True,
) -> dict[str, Any]:
    """Generate authoritative bundles, write a manifest, then derive a pack."""

    config.validate()
    root = Path(config.dataset.root).resolve()
    bundles_dir = root / "bundles"
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"{manifest_path} already exists; choose a new dataset.root or use pack-only"
        )
    bundles_dir.mkdir(parents=True, exist_ok=True)
    (root / "resolved_config.json").write_text(
        json.dumps(config.as_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if benchmark_workers:
        benchmark = benchmark_worker_counts(
            config.generation.worker_candidates,
            trials=config.generation.benchmark_trials,
            max_memory_fraction=config.generation.max_memory_fraction,
            environment={
                "render_every": config.environment.render_every,
                "lidar": config.environment.lidar,
                "spacetime": config.environment.spacetime,
                "scale": config.environment.scale,
                "dt_numerator": config.environment.dt_numerator,
                "dt_denominator": config.environment.dt_denominator,
                "physics": config.environment.physics,
            },
        )
        workers = int(benchmark["selected_workers"])
    else:
        workers = int(config.generation.workers)
        benchmark = {
            "selected_workers": workers,
            "results": [],
            "skipped": True,
        }
    (root / "worker-benchmark.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    from .tasks import task_names

    tasks = tuple(config.dataset.tasks) or tuple(task_names())
    target_bytes = int(config.dataset.target_gib * (1 << 30))
    max_sources = config.dataset.max_episodes
    sources: list[dict[str, Any]] = []
    cycle = 0
    total_bytes = 0
    while total_bytes < target_bytes and (max_sources is None or len(sources) < max_sources):
        cycles_in_batch = max(1, math.ceil(workers / 9))
        jobs: list[dict[str, Any]] = []
        for _ in range(cycles_in_batch):
            jobs.extend(_generation_cycle(config, tasks, cycle, bundles_dir))
            cycle += 1
        if max_sources is not None:
            remaining = max_sources - len(sources)
            jobs = _truncate_jobs(jobs, remaining)
        if not jobs:
            break
        with _pool(workers) as executor:
            futures = [executor.submit(_generate_job, job) for job in jobs]
            for future in as_completed(futures):
                generated = future.result()
                for source in generated:
                    source["path"] = Path(source["path"]).resolve().relative_to(root).as_posix()
                    sources.append(source)
                    total_bytes += int(source["bytes"])

    sources.sort(key=lambda item: (item["seed"], item["pair_role"] or "", item["path"]))
    quality = _verify_replay_sample(sources, root, config.generation.replay_sample_rate)
    manifest_config = config.as_dict()
    manifest_config["generation"]["selected_workers"] = workers
    manifest_config["quality"] = quality
    write_dataset_manifest(manifest_path, config=manifest_config, sources=sources)

    pack_manifest: Path | None = None
    if build_pack:
        pack_manifest = build_training_pack(
            manifest_path,
            root / "pack",
            segment_steps=config.dataset.segment_steps,
            window_steps=config.dataset.window_steps,
            shard_bytes=int(config.dataset.shard_gib * (1 << 30)),
        )
    return {
        "dataset_manifest": str(manifest_path),
        "training_pack_manifest": None if pack_manifest is None else str(pack_manifest),
        "sources": len(sources),
        "bytes": total_bytes,
        "selected_workers": workers,
        "quality": quality,
    }


def build_pack_only(config: ResearchConfig) -> Path:
    root = Path(config.dataset.root).resolve()
    return build_training_pack(
        root / "manifest.json",
        root / "pack",
        segment_steps=config.dataset.segment_steps,
        window_steps=config.dataset.window_steps,
        shard_bytes=int(config.dataset.shard_gib * (1 << 30)),
    )


def _generation_cycle(
    config: ResearchConfig,
    tasks: tuple[str, ...],
    cycle: int,
    bundles_dir: Path,
) -> list[dict[str, Any]]:
    """Nine jobs produce ten trajectories: 5 expert, 3 mixed, 2 paired."""

    policies: tuple[tuple[str, float | None], ...] = (
        ("oracle_expert", 0.0),
        ("oracle_expert", 0.0),
        ("epsilon_mixed", config.dataset.epsilon_values[0 % len(config.dataset.epsilon_values)]),
        ("oracle_expert", 0.0),
        ("paired_intervention", 0.15),
        ("epsilon_mixed", config.dataset.epsilon_values[1 % len(config.dataset.epsilon_values)]),
        ("oracle_expert", 0.0),
        ("epsilon_mixed", config.dataset.epsilon_values[2 % len(config.dataset.epsilon_values)]),
        ("oracle_expert", 0.0),
    )
    jobs: list[dict[str, Any]] = []
    randomized_offsets = _randomized_job_offsets(
        config.generation.domain_randomization_fraction, cycle
    )
    for offset, (policy, epsilon) in enumerate(policies):
        source_index = cycle * 10 + (offset if offset < 5 else offset + 1)
        seed = config.generation.seed0 + source_index
        task = tasks[source_index % len(tasks)]
        profile = _ood_profile(config, source_index, cycle)
        domain_randomized = offset in randomized_offsets
        physics = _episode_physics(config, seed, randomized=domain_randomized)
        physics = {**(physics or {}), **dict(profile.get("physics", {}))} or None
        job = {
            "kind": "pair" if policy == "paired_intervention" else "single",
            "task": task,
            "seed": seed,
            "out_dir": str(bundles_dir),
            "render_every": config.environment.render_every,
            "lidar": config.environment.lidar,
            "scale": float(profile.get("scale", config.environment.scale)),
            "dt_numerator": int(
                profile.get("dt_numerator", config.environment.dt_numerator)
            ),
            "dt_denominator": int(
                profile.get("dt_denominator", config.environment.dt_denominator)
            ),
            "physics": physics,
            "trace_level": config.generation.trace_level,
            "policy": policy,
            "epsilon": epsilon,
            "counterfactual_steps": config.generation.counterfactual_steps,
            "rollout_steps": config.generation.rollout_steps,
            "domain_randomized": domain_randomized,
            "train_fraction": config.dataset.train_fraction,
            "validation_fraction": config.dataset.validation_fraction,
            "split_override": config.dataset.split_override,
            "domain": profile.get("name"),
            "intervention_rotation": cycle % 5,
        }
        jobs.append(job)
    return jobs


def _ood_profile(
    config: ResearchConfig, source_index: int, cycle: int
) -> dict[str, Any]:
    profiles = config.generation.ood_profiles
    if not profiles:
        return {}
    return dict(profiles[(source_index + cycle) % len(profiles)])


def _randomized_job_offsets(fraction: float, cycle: int) -> set[int]:
    target_sources = round(10 * fraction)
    job_sizes = (1, 1, 1, 1, 2, 1, 1, 1, 1)
    choices = [
        selected
        for count in range(len(job_sizes) + 1)
        for selected in itertools.combinations(range(len(job_sizes)), count)
        if sum(job_sizes[offset] for offset in selected) == target_sources
    ]
    if not choices:
        possible_totals = {
            sum(job_sizes[offset] for offset in selected)
            for count in range(len(job_sizes) + 1)
            for selected in itertools.combinations(range(len(job_sizes)), count)
        }
        nearest = min(possible_totals, key=lambda total: abs(total - target_sources))
        return _randomized_job_offsets(nearest / 10.0, cycle)
    return set(choices[cycle % len(choices)])


def _episode_physics(
    config: ResearchConfig, seed: int, *, randomized: bool
) -> dict[str, float] | None:
    base = dict(config.environment.physics or {})
    rng = np.random.default_rng(seed + 7_919)
    if config.generation.physics_ranges and randomized:
        for key, bounds in sorted(config.generation.physics_ranges.items()):
            base[key] = float(rng.uniform(float(bounds[0]), float(bounds[1])))
    return base or None


def _truncate_jobs(jobs: list[dict[str, Any]], remaining_sources: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used = 0
    for job in jobs:
        size = 2 if job["kind"] == "pair" else 1
        if used + size > remaining_sources:
            continue
        selected.append(job)
        used += size
        if used == remaining_sources:
            break
    return selected


def _generate_job(job: dict[str, Any]) -> list[dict[str, Any]]:
    if job["kind"] == "pair":
        return _generate_pair(job)
    return [_generate_single(job)]


def _generate_single(job: dict[str, Any]) -> dict[str, Any]:
    from .experts import run_episode

    success, steps, final_hash, path = run_episode(
        job["task"],
        job["seed"],
        record_dir=job["out_dir"],
        render=job["render_every"],
        epsilon=float(job["epsilon"]),
        scale=job["scale"],
        record_format=2,
        trace_level=job["trace_level"],
        dt_numerator=job["dt_numerator"],
        dt_denominator=job["dt_denominator"],
        physics=job["physics"],
        lidar=job["lidar"],
        max_steps=job["rollout_steps"],
    )
    return _source_record(
        Path(path),
        job,
        steps=steps,
        final_hash=final_hash,
        success=success,
    )


def _generate_pair(job: dict[str, Any]) -> list[dict[str, Any]]:
    from .env import ACTION_KEYS, VoxelGymEnv, random_action
    from .experts import make_expert
    from .recorder import CausalRecorder
    from .tasks import make_task

    task = make_task(job["task"])
    control = VoxelGymEnv(
        task=task,
        seed=job["seed"],
        render=job["render_every"],
        lidar=job["lidar"],
        scale=job["scale"],
        dt_numerator=job["dt_numerator"],
        dt_denominator=job["dt_denominator"],
        spacetime=True,
        physics=job["physics"],
    )
    initial, _ = control.reset(seed=job["seed"])
    pair_boundary_tick = int(control.world.tick())
    try:
        expert = make_expert(job["task"], task, seed=job["seed"])
    except (AttributeError, KeyError, NotImplementedError):
        expert = None
    treatment = control.fork()
    intervention_specs, intervention_kind = _select_pair_intervention(
        control, int(job.get("intervention_rotation", job["seed"] % 5))
    )
    pair_id = hashlib.sha256(
        f"{job['task']}:{job['seed']}:{intervention_kind}".encode("utf-8")
    ).hexdigest()[:20]
    common = {
        "trace_level": job["trace_level"],
        "scale": job["scale"],
        "dt_numerator": job["dt_numerator"],
        "dt_denominator": job["dt_denominator"],
        "render_every": job["render_every"],
        "lidar": job["lidar"],
        "spacetime": True,
    }
    control_recorder = CausalRecorder(
        job["out_dir"],
        job["task"],
        job["seed"],
        branch_id=1,
        stem=f"{job['task']}_seed{job['seed']}_{pair_id}_control.vxbundle",
        metadata={
            "pair_id": pair_id,
            "pair_role": "control",
            "pair_intervention_kind": intervention_kind,
            "behavior_epsilon": 0.15,
        },
        **common,
    )
    treatment_recorder = CausalRecorder(
        job["out_dir"],
        job["task"],
        job["seed"],
        branch_id=2,
        stem=f"{job['task']}_seed{job['seed']}_{pair_id}_treatment.vxbundle",
        metadata={
            "pair_id": pair_id,
            "pair_role": "treatment",
            "pair_intervention_kind": intervention_kind,
            "behavior_epsilon": 0.15,
        },
        **common,
    )
    control_recorder.start(control, initial)
    treatment_recorder.start(
        treatment, {key: np.array(value, copy=True) for key, value in initial.items()}
    )
    rng = np.random.default_rng(job["seed"] + 999_983)
    reward_control = 0.0
    reward_treatment = 0.0
    steps = 0
    for step in range(int(job["counterfactual_steps"])):
        # Choose once from the factual expert and replay exactly on both
        # branches.  Inventory UI moves are restored into explicit controls,
        # matching the authoritative expert recorder's semantics.
        decision_snapshot = control.snapshot()
        swap = 0
        if expert is None:
            action = random_action(rng, zero=("place", "craft"))
        else:
            try:
                action = expert.act(control.world)
                swap = control.world.take_swap()
            except (AttributeError, KeyError, NotImplementedError):
                control.restore(decision_snapshot)
                expert = None
                action = random_action(rng, zero=("place", "craft"))
        shared_interventions: tuple[dict[str, Any], ...] = ()
        if swap:
            control.restore(decision_snapshot)
            shared_interventions = (
                {"kind": "swap_to_hotbar", "item": int(swap)},
            )
        if rng.random() < 0.15:
            action = random_action(rng, zero=("place", "craft"))
        control_obs, control_reward, control_term, control_trunc, control_info = (
            control.step_traced(
                action,
                trace_level=job["trace_level"],
                branch_id=1,
                interventions=shared_interventions,
            )
        )
        interventions = shared_interventions + (intervention_specs if step == 0 else ())
        treatment_obs, treatment_reward, treatment_term, treatment_trunc, treatment_info = (
            treatment.step_traced(
                action,
                trace_level=job["trace_level"],
                branch_id=2,
                interventions=interventions,
            )
        )
        packed_action = tuple(int(action[key]) for key in ACTION_KEYS)
        control_recorder.log(
            control,
            packed_action,
            control_reward,
            control_term,
            control_trunc,
            control_info,
            control_obs,
        )
        treatment_recorder.log(
            treatment,
            packed_action,
            treatment_reward,
            treatment_term,
            treatment_trunc,
            treatment_info,
            treatment_obs,
        )
        reward_control += float(control_reward)
        reward_treatment += float(treatment_reward)
        steps += 1
        if control_term or control_trunc or treatment_term or treatment_trunc:
            break

    control_hash = int(control.world.hash())
    treatment_hash = int(treatment.world.hash())
    changed = control_hash != treatment_hash
    reward_delta = reward_treatment - reward_control
    for recorder in (control_recorder, treatment_recorder):
        recorder.writer.metadata.update(
            {
                "pair_outcome_changed": changed,
                "pair_reward_delta": reward_delta,
                "pair_steps": steps,
                "pair_intervention_kind": intervention_kind,
                "behavior_epsilon": 0.15,
            }
        )
    control_path = control_recorder.save(control_hash)
    treatment_path = treatment_recorder.save(treatment_hash)
    control.close()
    treatment.close()
    control_source = _source_record(
        control_path,
        job,
        steps=steps,
        final_hash=control_hash,
        success=False,
        pair_id=pair_id,
        pair_role="control",
        pair_outcome_changed=changed,
        pair_reward_delta=reward_delta,
        branch_id=1,
        pair_intervention_kind=intervention_kind,
        pair_boundary_tick=pair_boundary_tick,
    )
    treatment_source = _source_record(
        treatment_path,
        job,
        steps=steps,
        final_hash=treatment_hash,
        success=False,
        pair_id=pair_id,
        pair_role="treatment",
        pair_outcome_changed=changed,
        pair_reward_delta=reward_delta,
        branch_id=2,
        pair_intervention_kind=intervention_kind,
        pair_boundary_tick=pair_boundary_tick,
    )
    return [control_source, treatment_source]


def _select_pair_intervention(
    env, rotation: int
) -> tuple[tuple[dict[str, Any], ...], str]:
    """Choose an effective canonical intervention by deterministic rotation."""

    from . import ids

    state = env.world.oracle_state()
    px, py, pz = (float(value) for value in state["position_cells"])
    at = (math.floor(px), math.floor(py) - 1, math.floor(pz))
    current_cell = int(env.world.get_block(*at))
    replacement = ids.STONE if (current_cell & 0xFFF) == ids.AIR else ids.AIR
    inventory = np.asarray(env.world.obs_inventory())
    swappable = [
        int(item)
        for item, count in inventory
        if int(item) != 0 and int(count) > 0
    ]
    swap_item = swappable[0] if swappable else ids.STONE
    candidates: tuple[tuple[str, tuple[dict[str, Any], ...]], ...] = (
        (
            "set_cell",
            ({"kind": "set_cell", "at": list(at), "cell": int(replacement)},),
        ),
        (
            "teleport_agent",
            ({"kind": "teleport_agent", "position": [px + 1.0, py, pz]},),
        ),
        (
            "set_agent_velocity",
            ({"kind": "set_agent_velocity", "velocity": [0.5, 0.5, 0.0]},),
        ),
        (
            "give_item",
            ({"kind": "give_item", "item": int(ids.STONE), "count": 1},),
        ),
        (
            "swap_to_hotbar",
            ({"kind": "swap_to_hotbar", "item": int(swap_item)},),
        ),
    )
    for offset in range(len(candidates)):
        kind, specs = candidates[(int(rotation) + offset) % len(candidates)]
        probe = env.fork()
        before = int(probe.world.hash())
        try:
            for spec in specs:
                probe.world.apply_intervention(spec, trace_level="off")
            effective = int(probe.world.hash()) != before
        except (TypeError, ValueError, RuntimeError):
            effective = False
        finally:
            probe.close()
        if effective:
            return specs, kind
    raise RuntimeError("no effective canonical intervention at the pair boundary")


def _source_record(
    path: Path,
    job: dict[str, Any],
    *,
    steps: int,
    final_hash: int,
    success: bool,
    pair_id: str | None = None,
    pair_role: str | None = None,
    pair_outcome_changed: bool = False,
    pair_reward_delta: float = 0.0,
    branch_id: int = 0,
    pair_intervention_kind: str | None = None,
    pair_boundary_tick: int | None = None,
) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    return {
        "path": str(path),
        "sha256": bundle_sha256(path),
        "bytes": _directory_size(path),
        "task": job["task"],
        "seed": int(job["seed"]),
        "split": job.get("split_override")
        or assign_split(
            job["task"],
            int(job["seed"]),
            float(job["train_fraction"]),
            float(job["validation_fraction"]),
        ),
        "policy": job["policy"],
        "epsilon": (
            0.15 if job.get("policy") == "paired_intervention" else float(job["epsilon"] or 0.0)
        ),
        "success": bool(success),
        "steps": int(steps),
        "final_hash": int(final_hash),
        "physics_config": metadata.get("physics_config"),
        "physics": metadata.get("physics"),
        "scale": metadata.get("scale"),
        "clock": metadata.get("clock"),
        "sensor_profile": metadata.get("sensor_profile"),
        "pair_id": pair_id,
        "pair_role": pair_role,
        "pair_outcome_changed": bool(pair_outcome_changed),
        "pair_reward_delta": float(pair_reward_delta),
        "pair_intervention_kind": pair_intervention_kind,
        "pair_boundary_tick": pair_boundary_tick,
        "branch_id": int(branch_id),
        "domain_randomized": bool(job.get("domain_randomized", False)),
        "domain": job.get("domain"),
    }


def _verify_replay_sample(
    sources: list[dict[str, Any]], root: Path, sample_rate: float
) -> dict[str, Any]:
    from .replay import verify

    if not sources or sample_rate <= 0:
        return {"sampled": 0, "passed": 0, "rate": sample_rate}
    count = max(1, math.ceil(len(sources) * sample_rate))
    ranked = sorted(
        sources,
        key=lambda source: hashlib.sha256(source["path"].encode("utf-8")).digest(),
    )[:count]
    failures = [source["path"] for source in ranked if not verify(str(root / source["path"]), False)]
    if failures:
        raise RuntimeError(f"replay quality gate failed: {failures}")
    return {"sampled": count, "passed": count, "rate": sample_rate}


def _pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=max(1, int(workers)),
        mp_context=multiprocessing.get_context("spawn"),
    )


def _benchmark_worker(job: tuple[int, int, dict[str, Any]]) -> int:
    from .env import VoxelGymEnv

    seed, steps, profile = job
    if not profile:
        import voxelgym_rs as rs

        world = rs.PyWorld(seed, "default")
        action = (1, 0, 0, seed % 24, 4, 0, 0, 0, 0, 0)
        for _ in range(steps):
            world.step(action)
        return steps

    env = VoxelGymEnv(
        preset="default",
        seed=seed,
        render=int(profile["render_every"]),
        lidar=profile.get("lidar"),
        spacetime=bool(profile.get("spacetime", True)),
        scale=float(profile.get("scale", 1.0)),
        dt_numerator=int(profile.get("dt_numerator", 1)),
        dt_denominator=int(profile.get("dt_denominator", 20)),
        physics=profile.get("physics"),
    )
    env.reset(seed=seed)
    action = {
        "move": 1,
        "jump": 0,
        "sneak": 0,
        "yaw": seed % 24,
        "pitch": 4,
        "mine": 0,
        "place": 0,
        "use": 0,
        "hotbar": 0,
        "craft": 0,
    }
    try:
        for _ in range(steps):
            env.step(action)
    finally:
        env.close()
    return steps


def _memory_fraction() -> float:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return 1.0 - status.available_physical / status.total_physical
    if hasattr(os, "sysconf"):
        try:
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            return 1.0 - available / total
        except (OSError, ValueError):
            pass
    return 0.0


def _directory_size(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


__all__ = [
    "benchmark_worker_counts",
    "build_causal_dataset",
    "build_pack_only",
]
