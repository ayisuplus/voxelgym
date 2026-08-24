"""Sharded vector env: num_shards processes, each holding a PyWorldBatch of
several worlds stepped with rayon inside Rust (GIL released).

Protocol per worker pipe:
  ("step", np.ndarray (k,10) uint8) -> (obs dict of stacked arrays, dead flags)
  ("hashes", None) -> list[int]
  ("close", None)
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any

import numpy as np


def _worker(pipe, specs: list[tuple[int, str]]):
    import time

    import voxelgym_rs as rs

    batch = rs.PyWorldBatch(specs)
    while True:
        cmd, payload = pipe.recv()
        if cmd == "step":
            deads = batch.step_batch_np(payload)
            obs = {
                "voxels": batch.obs_voxels_batch(),
                "inventory": batch.obs_inventory_batch(),
                "pose": batch.obs_pose_batch(),
                "raycast": batch.obs_raycast_batch(),
            }
            pipe.send((obs, np.asarray(deads, dtype=bool)))
        elif cmd == "bench":
            # payload: (steps, size, 10) uint8 — loop inside the worker so the
            # measurement is sim-only (no per-step obs IPC).
            start = time.perf_counter()
            for row in payload:
                batch.step_batch_np(row)
            pipe.send(time.perf_counter() - start)
        elif cmd == "hashes":
            pipe.send(batch.hashes())
        elif cmd == "close":
            pipe.close()
            return


class ShardedVectorEnv:
    """num_envs worlds split across num_shards worker processes."""

    def __init__(
        self,
        num_envs: int,
        num_shards: int | None = None,
        preset: str = "default",
        seed: int = 0,
    ):
        self.num_envs = num_envs
        self.num_shards = num_shards or (os.cpu_count() or 1)
        shards = min(self.num_shards, num_envs)
        self.num_shards = shards
        base = num_envs // shards
        rem = num_envs % shards
        self._sizes = [base + (1 if i < rem else 0) for i in range(shards)]
        self._pipes = []
        self._procs = []
        ctx = mp.get_context("spawn")
        first = 0
        for size in self._sizes:
            specs = [(seed + first + j, preset) for j in range(size)]
            first += size
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, specs), daemon=True)
            p.start()
            self._pipes.append(parent)
            self._procs.append(p)
        self._closed = False

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """actions: (num_envs, 10) integer array. Returns stacked obs + deads."""
        assert actions.shape == (self.num_envs, 10)
        actions = np.ascontiguousarray(actions, dtype=np.uint8)
        off = 0
        for pipe, size in zip(self._pipes, self._sizes):
            pipe.send(("step", actions[off : off + size]))
            off += size
        obs_parts: dict[str, list[np.ndarray]] = {}
        deads = []
        for pipe in self._pipes:
            obs, dead = pipe.recv()
            deads.append(dead)
            for k, v in obs.items():
                obs_parts.setdefault(k, []).append(v)
        return {k: np.concatenate(v) for k, v in obs_parts.items()}, np.concatenate(deads)

    def hashes(self) -> list[int]:
        for pipe in self._pipes:
            pipe.send(("hashes", None))
        out: list[int] = []
        for pipe in self._pipes:
            out.extend(pipe.recv())
        return out

    def close(self):
        if self._closed:
            return
        self._closed = True
        for pipe in self._pipes:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc: Any):
        self.close()
