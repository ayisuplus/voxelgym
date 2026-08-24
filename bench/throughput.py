"""Throughput benchmark: single-env steps/s and 64-env aggregate steps/s.

Usage: python bench/throughput.py [--envs 64] [--ticks 100000] [--trials 3]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np

import voxelgym_rs as rs
from voxelgym.vec import ShardedVectorEnv


def gen_actions(rng: np.random.Generator, n: int) -> np.ndarray:
    a = np.zeros((n, 10), dtype=np.uint8)
    a[:, 0] = rng.integers(0, 5, n)
    a[:, 1] = rng.integers(0, 2, n)
    a[:, 3] = rng.integers(0, 24, n)
    a[:, 4] = rng.integers(0, 9, n)
    # sparse mine/place to exercise action paths without churning chunks hard
    a[rng.random(n) < 0.05, 5] = 1
    return a


def bench_single(ticks: int, trials: int) -> float:
    rates = []
    for t in range(trials):
        w = rs.PyWorld(1000 + t, "default")
        acts = gen_actions(np.random.default_rng(7 + t), ticks)
        start = time.perf_counter()
        for row in acts:
            w.step(tuple(int(v) for v in row))
            w.obs_pose()  # minimal obs touch per step
        dt = time.perf_counter() - start
        rates.append(ticks / dt)
    return statistics.median(rates)


def bench_vec(envs: int, ticks: int, trials: int) -> float:
    """Aggregate world-ticks/s with `envs` worlds across cpu_count shards.

    Each worker loops internally (sim-only); per-step obs IPC would move
    ~633 KB/step for 64 envs and dominate the measurement.
    """
    rates = []
    shards = os.cpu_count() or 1
    for t in range(trials):
        vec = ShardedVectorEnv(num_envs=envs, num_shards=shards, preset="default", seed=2000 + t)
        acts = gen_actions(np.random.default_rng(17 + t), ticks)
        start = time.perf_counter()
        off = 0
        for pipe, size in zip(vec._pipes, vec._sizes):
            pipe.send(("bench", np.ascontiguousarray(np.tile(acts[:, None, :], (1, size, 1)))))
            off += size
        for pipe in vec._pipes:
            pipe.recv()
        dt = time.perf_counter() - start
        vec.close()
        rates.append(ticks * envs / dt)
        print(f"  trial {t}: {ticks * envs / dt:,.0f} world-ticks/s", flush=True)
    return statistics.median(rates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--ticks", type=int, default=100_000)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    single = bench_single(args.ticks, args.trials)
    print(f"single env: {single:,.0f} steps/s (target >= 5,000)")

    if args.envs > 1:
        agg = bench_vec(args.envs, args.ticks, args.trials)
        print(f"{args.envs} envs aggregate: {agg:,.0f} steps/s (target >= 100,000)")

    ok_single = single >= 5000
    print("single:", "PASS" if ok_single else "BELOW TARGET")
    return 0 if ok_single else 1


if __name__ == "__main__":
    sys.exit(main())
