"""Replay & verify: rebuild the world from the sidecar seed + task scenario,
re-apply the recorded action sequence, assert checkpoint and final hashes.

Usage: python -m voxelgym.replay <shard.parquet> --verify
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import pyarrow.parquet as pq

from .env import ACTION_KEYS
from .episode_bundle import EpisodeBundleReader
from .interventions import canonical_interventions
from .recorder import decode_observation, encode_oracle_view
from .task_state import EnvSnapshot, encode_task_fields


def _load(parquet_path: str):
    sidecar_path = parquet_path[: -len(".parquet")] + ".json"
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    table = pq.read_table(parquet_path)
    return sidecar, table.to_pylist()


def _checkpoint_hash(world, snapshot: bytes) -> int:
    """Use the identity contract that was current when a checkpoint was written."""

    version = None
    if len(snapshot) >= 8 and snapshot[:4] == b"VXG1":
        version = int.from_bytes(snapshot[4:8], "little")
    if version is not None and version < 8 and hasattr(world, "legacy_hash_v7"):
        return int(world.legacy_hash_v7())
    return int(world.hash())


def _verify_v1(parquet_path: str, verbose: bool = True) -> bool:
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
            # v5-v7 snapshots predate clock/semantic/dirty hash identity.
            # Compare both sides under that historical contract instead of
            # accidentally hashing restored legacy state as a new v8 world.
            live = _checkpoint_hash(world, ckpt)
            snap = _checkpoint_hash(scratch, ckpt)
            if live != snap:
                print(f"CHECKPOINT MISMATCH at tick {row['tick']}: live={live:016x} ckpt={snap:016x}")
                return False
            checked_ckpts += 1
    final = world.hash()
    expected = sidecar["final_hash"]
    hash_mode = "v8"
    ok = final == expected
    if not ok and hasattr(world, "legacy_hash_v7"):
        legacy = world.legacy_hash_v7()
        if legacy == expected:
            ok = True
            final = legacy
            hash_mode = "legacy-v7"
    if verbose:
        print(f"task={task_name} seed={seed} steps={len(rows)} ckpts_checked={checked_ckpts}")
        print(f"final hash ({hash_mode}): {final:016x} vs sidecar {expected:016x}")
        print("PASS" if ok else "FAIL")
    return ok


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _snapshot_equal(left: EnvSnapshot, right: EnvSnapshot) -> bool:
    if (
        left.world_snapshot != right.world_snapshot
        or left.task_state != right.task_state
        or _json(encode_task_fields(left.np_random_state))
        != _json(encode_task_fields(right.np_random_state))
        or left.episode_seed != right.episode_seed
        or left.terminated != right.terminated
        or left.truncated != right.truncated
        or left.last_reward != right.last_reward
        or left.render_sample_tick != right.render_sample_tick
        or left.lidar_sample_tick != right.lidar_sample_tick
        or left.previous_pose != right.previous_pose
        or left.render_every != right.render_every
        or left.lidar_config != right.lidar_config
        or left.spacetime != right.spacetime
        or left.last_trace != right.last_trace
        or left.native_trace_state != right.native_trace_state
        or left.native_intervention_cursor != right.native_intervention_cursor
        or left.intervention_cursor != right.intervention_cursor
    ):
        return False
    for first, second in (
        (left.last_frames, right.last_frames),
        (left.last_scan, right.last_scan),
    ):
        if (first is None) != (second is None):
            return False
        if first is not None and any(
            not np.array_equal(a, b) for a, b in zip(first, second, strict=True)
        ):
            return False
    return True


def _verify_v2(bundle_path: str, verbose: bool = True) -> bool:
    from .env import VoxelGymEnv
    from .tasks import make_task

    reader = EpisodeBundleReader(bundle_path)
    try:
        reader.validate()
    except ValueError as exc:
        if verbose:
            print(f"BUNDLE INVALID: {exc}")
        return False
    manifest = reader.manifest
    metadata = manifest.get("metadata", {})
    clock = metadata.get("clock", {})
    task = make_task(manifest["task"])
    env = VoxelGymEnv(
        task=task,
        preset=task.preset,
        seed=int(manifest["seed"]),
        render=int(metadata.get("render_every", 0)),
        lidar=metadata.get("lidar"),
        scale=float(metadata.get("scale", 1.0)),
        dt_numerator=int(clock.get("dt_numerator", 1)),
        dt_denominator=int(clock.get("dt_denominator", 20)),
        spacetime=bool(metadata.get("spacetime", False)),
    )
    checkpoint_rows_list = reader.checkpoints.to_pylist()
    initial_checkpoint = next(
        row for row in checkpoint_rows_list if row["boundary"] == "initial"
    )
    env.restore(EnvSnapshot.from_bytes(initial_checkpoint["env_snapshot"]))
    event_rows: dict[int, list[dict]] = {}
    for event in reader.events.to_pylist():
        event_rows.setdefault(event["transition_id"], []).append(event)
    delta_rows: dict[int, list[dict]] = {}
    for delta in reader.deltas.to_pylist():
        delta_rows.setdefault(delta["transition_id"], []).append(delta)
    checkpoint_rows = {
        row["transition_id"]: row
        for row in checkpoint_rows_list
        if row["boundary"] != "initial"
    }

    def mismatch(message: str) -> bool:
        if verbose:
            print(f"REPLAY MISMATCH: {message}")
        return False

    for row in reader.transitions.to_pylist():
        if encode_oracle_view(env.oracle_view()) != row["oracle_state_before"]:
            return mismatch(f"before oracle state at transition {row['transition_id']}")
        action = {key: int(row[key]) for key in ACTION_KEYS}
        try:
            interventions, external_intervention_count = (
                reader.transition_interventions(row)
            )
        except ValueError as exc:
            return mismatch(
                f"interventions at transition {row['transition_id']}: {exc}"
            )
        if not interventions and row.get("swap"):
            # Read-only compatibility for early v2 rows that represented this
            # one intervention only through the legacy swap column.
            interventions = canonical_interventions(
                ({"kind": "swap_to_hotbar", "item": int(row["swap"])},)
            )
            external_intervention_count = 1
        obs, reward, terminated, truncated, info = env.step_traced(
            action,
            trace_level=manifest.get("trace_level", "full"),
            branch_id=int(row["branch_id"]),
            interventions=interventions[:external_intervention_count],
        )
        try:
            replayed_interventions = canonical_interventions(
                info.get("interventions", ())
            )
        except ValueError as exc:
            return mismatch(
                f"runtime interventions at transition {row['transition_id']}: {exc}"
            )
        if replayed_interventions != interventions:
            return mismatch(
                f"interventions at transition {row['transition_id']}"
            )
        oracle = env.oracle_view()
        trace = oracle.get("trace")
        if trace is None:
            return mismatch(f"missing oracle trace at transition {row['transition_id']}")
        if (
            trace["clock_before"]["tick"] != row["tick_before"]
            or trace["clock_after"]["tick"] != row["tick_after"]
        ):
            return mismatch(f"clock at transition {row['transition_id']}")
        if row["before_hash"] is not None and trace["before_hash"] != row["before_hash"]:
            return mismatch(f"before hash at transition {row['transition_id']}")
        if row["after_hash"] is not None and trace["after_hash"] != row["after_hash"]:
            return mismatch(f"after hash at transition {row['transition_id']}")
        if not np.isclose(reward, row["reward"], rtol=0.0, atol=1e-6):
            return mismatch(f"reward at transition {row['transition_id']}")
        if (terminated, truncated) != (row["terminated"], row["truncated"]):
            return mismatch(f"termination at transition {row['transition_id']}")
        if info["reward_outcome"]["termination_reason"] != row["termination_reason"]:
            return mismatch(f"termination reason at transition {row['transition_id']}")
        if _json(info["reward_outcome"]["components"]) != row["reward_components"]:
            return mismatch(f"reward components at transition {row['transition_id']}")
        stored_obs = decode_observation(row["agent_observation"])
        if stored_obs.keys() != obs.keys() or any(
            not np.array_equal(stored_obs[key], obs[key]) for key in stored_obs
        ):
            return mismatch(f"agent observation at transition {row['transition_id']}")
        try:
            stored_oracle = json.loads(bytes(row["oracle_state"]).decode("utf-8"))
            stored_oracle_payload = _json(stored_oracle).encode("utf-8")
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return mismatch(f"oracle state at transition {row['transition_id']}")
        if stored_oracle_payload != encode_oracle_view(oracle):
            return mismatch(f"oracle state at transition {row['transition_id']}")

        expected_events = event_rows.get(row["transition_id"], [])
        actual_events = trace["events"]
        if len(expected_events) != len(actual_events):
            return mismatch(f"event count at transition {row['transition_id']}")
        for expected_event, actual_event in zip(expected_events, actual_events, strict=True):
            location = actual_event.get("location")
            expected_location = (
                None
                if expected_event["position_x"] is None
                else (
                    expected_event["position_x"],
                    expected_event["position_y"],
                    expected_event["position_z"],
                )
            )
            if (
                expected_event["id"] != actual_event["id"]
                or expected_event["tick"] != actual_event["tick"]
                or expected_event["phase"] != actual_event["phase"]
                or expected_event["kind"] != actual_event["kind"]
                or expected_event["mechanism"] != actual_event["mechanism"]
                or expected_event["parent_ids"] != actual_event["parent_ids"]
                or expected_location != location
                or expected_event["actor"]
                != (None if actual_event.get("actor") is None else _json(actual_event["actor"]))
                or expected_event["target"]
                != (None if actual_event.get("target") is None else _json(actual_event["target"]))
                or expected_event["root_cause"] != _json(actual_event["root_cause"])
            ):
                return mismatch(f"event {actual_event['id']}")

        expected_deltas = delta_rows.get(row["transition_id"], [])
        actual_deltas = trace["deltas"]
        if len(expected_deltas) != len(actual_deltas):
            return mismatch(f"delta count at transition {row['transition_id']}")
        for expected_delta, actual_delta in zip(expected_deltas, actual_deltas, strict=True):
            if (
                expected_delta["event_id"] != actual_delta["event_id"]
                or expected_delta["subject"] != _json(actual_delta["subject"])
                or expected_delta["field"] != actual_delta["field_or_cell"]
                or expected_delta["before"] != _json(actual_delta["before"])
                or expected_delta["after"] != _json(actual_delta["after"])
            ):
                return mismatch(f"delta for event {actual_delta['event_id']}")

        checkpoint = checkpoint_rows.get(row["transition_id"])
        if checkpoint is not None:
            current = env.snapshot()
            stored = EnvSnapshot.from_bytes(checkpoint["env_snapshot"])
            if checkpoint["world_snapshot"] != current.world_snapshot:
                return mismatch(f"world checkpoint at transition {row['transition_id']}")
            if not _snapshot_equal(stored, current):
                return mismatch(f"environment checkpoint at transition {row['transition_id']}")

    final = env.world.hash()
    ok = final == manifest["final_hash"]
    if verbose:
        print(
            f"task={manifest['task']} seed={manifest['seed']} "
            f"steps={reader.transitions.num_rows} format=v2"
        )
        print(f"final hash: {final:016x} vs manifest {manifest['final_hash']:016x}")
        print("PASS" if ok else "FAIL")
    return ok


def verify(episode_path: str, verbose: bool = True) -> bool:
    """Auto-detect and verify a legacy shard or Episode Bundle v2."""

    if Path(episode_path).suffix == ".parquet":
        return _verify_v1(episode_path, verbose=verbose)
    return _verify_v2(episode_path, verbose=verbose)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if not args.verify:
        print("nothing to do (pass --verify)")
        return 2
    return 0 if verify(args.episode) else 1


if __name__ == "__main__":
    sys.exit(main())
