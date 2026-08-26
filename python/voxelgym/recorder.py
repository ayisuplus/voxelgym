"""Episode recorder: one Parquet shard per episode + sidecar JSON.

Columns: tick u32, 10 action u8 columns, reward f32, done bool,
voxel_win binary, inv binary, rgb/depth/seg binary (nullable, M4),
world_ckpt binary (full snapshot every 600 ticks and on the final row).
Binary columns are zstd-compressed by the parquet writer.
"""

from __future__ import annotations

import json
import io
import os
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .env import ACTION_KEYS
from .episode_bundle import EpisodeBoundary, EpisodeBundleWriter, TransitionRecord

CKPT_EVERY = 600

SCHEMA = pa.schema(
    [
        ("tick", pa.uint32()),
        *[(k, pa.uint8()) for k in ACTION_KEYS],
        # inventory-management event (oracle experts only): item id pulled
        # into the hotbar this tick, 0 = none. Part of the behavior trace —
        # replay applies it before the action. Not a world-state input.
        ("swap", pa.uint16()),
        ("reward", pa.float32()),
        ("done", pa.bool_()),
        ("voxel_win", pa.binary()),
        ("inv", pa.binary()),
        ("rgb", pa.binary()),
        ("depth", pa.binary()),
        ("seg", pa.binary()),
        ("world_ckpt", pa.binary()),
    ]
)


def code_version() -> str:
    try:
        import subprocess

        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class Recorder:
    """Streams rows to a ParquetWriter in FLUSH_EVERY-row groups — episode
    RAM stays flat instead of growing with horizon (a rendered 10k-tick
    episode is >1 GB of binary columns if buffered whole)."""

    FLUSH_EVERY = 1000

    def __init__(self, out_dir: str, task: str, seed: int, render: bool = False):
        self.out_dir = out_dir
        self.task = task
        self.seed = seed
        self.render = render
        self.rows: list[dict] = []
        self._writer: pq.ParquetWriter | None = None
        self._n = 0
        os.makedirs(out_dir, exist_ok=True)
        stem = f"{task}_seed{seed}_{int(time.time() * 1000)}"
        self._stem = stem
        self._pq_path = os.path.join(out_dir, stem + ".parquet")

    def log(self, world, action: tuple, reward: float, done: bool, frames=None, swap: int = 0):
        tick = world.tick()
        ckpt = None
        if tick % CKPT_EVERY == 0 or done:
            ckpt = bytes(world.snapshot())
        row = {
            "tick": tick,
            **{k: int(a) for k, a in zip(ACTION_KEYS, action)},
            "swap": int(swap),
            "reward": float(reward),
            "done": bool(done),
            "voxel_win": world.obs_voxels_bytes(),
            "inv": world.obs_inventory_bytes(),
            "rgb": None,
            "depth": None,
            "seg": None,
            "world_ckpt": ckpt,
        }
        if frames is not None:
            rgb, depth, seg = frames
            row["rgb"] = rgb.tobytes()
            row["depth"] = depth.tobytes()
            row["seg"] = seg.tobytes()
        self.rows.append(row)
        if len(self.rows) >= self.FLUSH_EVERY:
            self._flush()

    def _flush(self):
        if not self.rows:
            return
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._pq_path, SCHEMA, compression="zstd")
        self._writer.write_table(pa.Table.from_pylist(self.rows, schema=SCHEMA))
        self._n += len(self.rows)
        self.rows.clear()

    def save(self, final_hash: int) -> str:
        self._flush()
        if self._writer is None:
            # zero-row episode: still emit a valid empty shard
            pq.write_table(SCHEMA.empty_table(), self._pq_path, compression="zstd")
        else:
            self._writer.close()
            self._writer = None
        sidecar = {
            "task": self.task,
            "seed": self.seed,
            "code_version": code_version(),
            "final_hash": final_hash,
            "steps": self._n,
            "parquet": os.path.basename(self._pq_path),
        }
        with open(os.path.join(self.out_dir, self._stem + ".json"), "w") as f:
            json.dump(sidecar, f, indent=2)
        return self._pq_path


def encode_observation(observation: dict[str, np.ndarray]) -> bytes:
    """Portable, pickle-free encoding for the complete agent-visible view."""

    output = io.BytesIO()
    np.savez_compressed(
        output,
        **{key: np.asarray(value) for key, value in sorted(observation.items())},
    )
    return output.getvalue()


def decode_observation(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def encode_oracle_view(oracle: dict[str, Any]) -> bytes:
    """Canonical encoding of the privileged view stored in Bundle v2.

    The full World Snapshot belongs in the checkpoint table. Keeping it out of
    this payload makes per-transition oracle verification inexpensive while
    retaining every other value exposed to training consumers.
    """

    payload = dict(oracle)
    payload.pop("world_snapshot", None)
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


class CausalRecorder:
    """Episode Bundle v2 recorder for traced Gym transitions."""

    def __init__(
        self,
        out_dir: str,
        task: str,
        seed: int,
        *,
        trace_level: str = "full",
        branch_id: int = 0,
        scale: float = 1.0,
        dt_numerator: int = 1,
        dt_denominator: int = 20,
        render_every: int = 0,
        lidar: dict[str, Any] | None = None,
        spacetime: bool | None = None,
        checkpoint_every: int = CKPT_EVERY,
    ):
        self.trace_level = trace_level
        self.branch_id = int(branch_id)
        self._configured_spacetime = (
            None if spacetime is None else bool(spacetime)
        )
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.writer = EpisodeBundleWriter(
            out_dir,
            task=task,
            seed=seed,
            trace_level=trace_level,
            branch_id=branch_id,
            metadata={
                "code_version": code_version(),
                "scale": float(scale),
                "render_every": int(render_every),
                "lidar": lidar,
                "spacetime": False if spacetime is None else bool(spacetime),
                "clock": {
                    "dt_numerator": int(dt_numerator),
                    "dt_denominator": int(dt_denominator),
                },
            },
        )
        self._env = None
        self._expected_tick: int | None = None
        self._transition_count = 0
        self._agent_observation_before: bytes | None = None
        self._oracle_state_before: bytes | None = None

    def start(self, env, observation: dict[str, np.ndarray]) -> None:
        """Bind the recorder to a complete pre-transition branch boundary."""

        if self._env is not None:
            raise RuntimeError("causal recorder is already started")
        env_spacetime = bool(getattr(env, "_spacetime", False))
        if (
            self._configured_spacetime is not None
            and env_spacetime != self._configured_spacetime
        ):
            raise ValueError(
                "recorder spacetime setting does not match the environment"
            )
        oracle = env.oracle_view()
        snapshot = env.snapshot()
        tick = int(env.world.tick())
        if int(oracle["clock"]["tick"]) != tick:
            raise ValueError("oracle boundary tick does not match the environment")
        agent_payload = encode_observation(observation)
        oracle_payload = encode_oracle_view(oracle)
        external_parent_ids = getattr(
            env.world, "trace_external_parent_ids", None
        )
        self.writer.set_external_parent_ids(
            ()
            if not callable(external_parent_ids)
            else external_parent_ids()
        )
        self.writer.set_initial_boundary(
            EpisodeBoundary(
                tick=tick,
                world_snapshot=snapshot.world_snapshot,
                env_snapshot=snapshot.to_bytes(),
                agent_observation=agent_payload,
                oracle_state=oracle_payload,
            )
        )
        self.writer.metadata.update(
            {
                "scale": float(oracle["scale"]),
                "render_every": int(getattr(env, "_render_every", 0)),
                "lidar": getattr(env, "_lidar", None),
                "spacetime": env_spacetime,
                "clock": {
                    "dt_numerator": int(oracle["clock"]["dt_numerator"]),
                    "dt_denominator": int(oracle["clock"]["dt_denominator"]),
                },
            }
        )
        self._env = env
        self._expected_tick = tick
        self._agent_observation_before = agent_payload
        self._oracle_state_before = oracle_payload

    def log(
        self,
        env,
        action: tuple[int, ...],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        observation: dict[str, np.ndarray],
        *,
        swap: int = 0,
    ) -> None:
        if env is not self._env:
            raise ValueError("causal recorder must log the environment passed to start()")
        oracle = env.oracle_view()
        trace = oracle.get("trace")
        if trace is None:
            raise ValueError("Episode Bundle v2 requires an oracle trace from step_traced()")
        env_spacetime = bool(getattr(env, "_spacetime", False))
        if (
            self._configured_spacetime is not None
            and env_spacetime != self._configured_spacetime
        ):
            raise ValueError(
                "recorder spacetime setting does not match the environment"
            )
        # ``log`` has the authoritative environment configuration, so callers
        # that omitted the constructor hint still produce an exact manifest.
        self.writer.metadata["spacetime"] = env_spacetime
        tick_before = int(trace["clock_before"]["tick"])
        tick_after = int(trace["clock_after"]["tick"])
        if tick_before != self._expected_tick:
            raise ValueError(
                f"recorded transition starts at tick {tick_before}, "
                f"expected {self._expected_tick}"
            )
        transition_id = ((self.branch_id & 0xFFFFFFFF) << 32) | tick_after
        done = bool(terminated or truncated)
        world_checkpoint = None
        env_checkpoint = None
        if tick_after % self.checkpoint_every == 0 or done:
            snapshot = env.snapshot()
            world_checkpoint = snapshot.world_snapshot
            env_checkpoint = snapshot.to_bytes()
        oracle_payload = encode_oracle_view(oracle)
        reward_outcome = info["reward_outcome"]
        if "interventions" in info:
            interventions = tuple(info["interventions"])
            external_intervention_count = info.get(
                "external_intervention_count", 0
            )
        elif swap:
            # Transitional compatibility for callers predating the explicit
            # intervention payload in Env.step_traced() info.
            interventions = ({"kind": "swap_to_hotbar", "item": int(swap)},)
            external_intervention_count = 1
        else:
            interventions = ()
            external_intervention_count = 0
        if (
            isinstance(external_intervention_count, bool)
            or not isinstance(external_intervention_count, int)
            or not 0 <= external_intervention_count <= len(interventions)
        ):
            raise ValueError(
                "external_intervention_count must index the intervention sequence"
            )
        if swap and not any(
            spec.get("kind") == "swap_to_hotbar"
            and int(spec.get("item", -1)) == int(swap)
            for spec in interventions[:external_intervention_count]
        ):
            raise ValueError(
                "legacy swap side channel does not match the transition interventions"
            )
        frames = (
            observation.get("rgb"),
            observation.get("depth"),
            observation.get("seg"),
        )
        self.writer.log(
            TransitionRecord(
                transition_id=transition_id,
                branch_id=self.branch_id,
                tick_before=tick_before,
                tick_after=tick_after,
                before_hash=trace.get("before_hash"),
                after_hash=trace.get("after_hash"),
                action=action,
                swap=int(swap),
                interventions=interventions,
                external_intervention_count=external_intervention_count,
                reward=float(reward),
                reward_components=dict(reward_outcome.get("components", {})),
                terminated=bool(terminated),
                truncated=bool(truncated),
                termination_reason=reward_outcome.get("termination_reason"),
                sensor_ticks=dict(info.get("sensor_sample_ticks", {})),
                agent_observation_before=self._agent_observation_before,
                oracle_state_before=self._oracle_state_before,
                agent_observation=encode_observation(observation),
                oracle_state=oracle_payload,
                rgb=None if frames[0] is None else frames[0].tobytes(),
                depth=None if frames[1] is None else frames[1].tobytes(),
                seg=None if frames[2] is None else frames[2].tobytes(),
                events=list(trace.get("events", ())),
                deltas=list(trace.get("deltas", ())),
                checkpoint=world_checkpoint,
                env_checkpoint=env_checkpoint,
            )
        )
        self._expected_tick = tick_after
        self._transition_count += 1
        self._agent_observation_before = encode_observation(observation)
        self._oracle_state_before = oracle_payload

    def save(self, final_hash: int):
        if self._env is None or self._transition_count == 0:
            raise RuntimeError("causal recorder has no recorded transitions")
        if self._env.world.tick() != self._expected_tick:
            raise ValueError("environment advanced after the last recorded transition")
        live_hash = int(self._env.world.hash())
        if live_hash != int(final_hash):
            raise ValueError(
                f"final hash {int(final_hash):016x} does not match live environment "
                f"{live_hash:016x}"
            )
        snapshot = self._env.snapshot()
        self.writer.set_final_boundary(
            EpisodeBoundary(
                tick=int(self._expected_tick),
                world_snapshot=snapshot.world_snapshot,
                env_snapshot=snapshot.to_bytes(),
                agent_observation=self._agent_observation_before,
                oracle_state=self._oracle_state_before,
            )
        )
        return self.writer.save(final_hash=final_hash)
