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
    def __init__(self, out_dir: str, task: str, seed: int, render: bool = False):
        self.out_dir = out_dir
        self.task = task
        self.seed = seed
        self.render = render
        self.rows: list[dict] = []
        os.makedirs(out_dir, exist_ok=True)

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
            "voxel_win": world.obs_voxels().tobytes(),
            "inv": world.obs_inventory().tobytes(),
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

    def save(self, final_hash: int) -> str:
        table = pa.Table.from_pylist(self.rows, schema=SCHEMA)
        stem = f"{self.task}_seed{self.seed}_{int(time.time() * 1000)}"
        pq_path = os.path.join(self.out_dir, stem + ".parquet")
        pq.write_table(table, pq_path, compression="zstd")
        sidecar = {
            "task": self.task,
            "seed": self.seed,
            "code_version": code_version(),
            "final_hash": final_hash,
            "steps": len(self.rows),
            "parquet": os.path.basename(pq_path),
        }
        with open(os.path.join(self.out_dir, stem + ".json"), "w") as f:
            json.dump(sidecar, f, indent=2)
        return pq_path
