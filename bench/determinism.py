"""Determinism check: same seed + same action sequence, run twice, compare
final world hashes (xxh3 over canonical serialization).

Usage: python bench/determinism.py [--seed 42] [--ticks 20000]
Exit code 0 iff hashes match.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np

import voxelgym_rs as rs


def run(seed: int, ticks: int, action_seed: int) -> int:
    rng = np.random.default_rng(action_seed)
    w = rs.PyWorld(seed, "default")
    for _ in range(ticks):
        a = (
            int(rng.integers(0, 5)),
            int(rng.integers(0, 2)),
            int(rng.integers(0, 2)),
            int(rng.integers(0, 24)),
            int(rng.integers(0, 9)),
            int(rng.random() < 0.1),
            0,
            0,
            int(rng.integers(0, 9)),
            0,
        )
        w.step(a)
    return w.hash()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ticks", type=int, default=20_000)
    args = ap.parse_args()

    h1 = run(args.seed, args.ticks, action_seed=999)
    h2 = run(args.seed, args.ticks, action_seed=999)
    print(f"run1 hash: {h1:016x}")
    print(f"run2 hash: {h2:016x}")
    if h1 != h2:
        print("FAIL: determinism violated")
        return 1
    print("PASS: deterministic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
