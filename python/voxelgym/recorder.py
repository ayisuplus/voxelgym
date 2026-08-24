"""Episode recorder: one Parquet shard per episode + sidecar JSON.

Columns: tick u32, 10 action u8 columns, reward f32, done bool,
voxel_win binary, inv binary, rgb/depth/seg binary (nullable, M4),
world_ckpt binary (full snapshot every 600 ticks and on the final row).
Binary columns are zstd-compressed by the parquet writer.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .env import ACTION_KEYS

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
