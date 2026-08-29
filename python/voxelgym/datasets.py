"""Action-conditioned video dataset: export (via scripted experts + Recorder)
and a torch loader.

CLI:
  python -m voxelgym.datasets export --task collect_log --episodes 50 --render 1 --out data/v1
  python -m voxelgym.datasets baseline --data data/v1
  python -m voxelgym.datasets build-causal --config experiments/causal-pilot.toml
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

from .env import ACTION_KEYS
from .episode_bundle import EpisodeBundleReader

FRAME = 128


def export(task: str, episodes: int, out_dir: str, render: int = 1, seed0: int = 0, epsilon: float = 0.0,
           scale: float = 1.0, format_version: int = 1,
           physics: dict[str, float] | None = None):
    from .experts import run_episode

    os.makedirs(out_dir, exist_ok=True)
    wins = 0
    for i in range(episodes):
        kwargs = {
            "record_dir": out_dir,
            "render": bool(render),
            "epsilon": epsilon,
            "scale": scale,
        }
        if physics is not None:
            kwargs["physics"] = physics
        if format_version != 1:
            kwargs["record_format"] = format_version
        ok, steps, h, path = run_episode(task, seed0 + i, **kwargs)
        wins += ok
        print(f"  ep {i}: {'OK' if ok else 'fail'} {steps} ticks -> {os.path.basename(path or '')}", flush=True)
    print(f"exported {episodes} episodes of {task} to {out_dir} (expert success {wins}/{episodes}, scale={scale})")


def _decode_frames(rows, key, shape, dtype):
    if rows[0][key] is None:
        return None
    return np.stack([np.frombuffer(row[key], dtype=dtype).reshape(shape) for row in rows])


class VoxelSequenceDataset:
    """torch Dataset over recorded episodes: 16-frame slices ->
    (rgb, action, depth, seg). Decodes lazily per episode with a one-shard
    cache (episodes are ~20-60 MB decoded).
    """

    def __init__(self, data_dir: str, seq_len: int = 16, split: str = "train", test_frac: float = 0.1):
        import torch  # noqa: F401  (torch Dataset protocol without hard dep at import)
        from torch.utils.data import Dataset  # noqa: F401
        legacy = glob.glob(os.path.join(data_dir, "*.parquet"))
        bundles = glob.glob(os.path.join(data_dir, "*.vxbundle"))
        shards = sorted(legacy + bundles)
        assert shards, f"no parquet shards or Episode Bundles in {data_dir}"
        n_test = max(1, int(round(len(shards) * test_frac)))
        if split == "test":
            self.shards = shards[:n_test]
        else:
            self.shards = shards[n_test:] if len(shards) > n_test else shards
        self.seq_len = seq_len
        self._lengths: list[int] = []
        for s in self.shards:
            n = (
                EpisodeBundleReader(s).transitions.num_rows
                if os.path.isdir(s)
                else pq.read_metadata(s).num_rows
            )
            self._lengths.append(max(0, n - seq_len + 1))
        self._cum = np.cumsum([0] + self._lengths)
        self._cache_idx = -1
        self._cache = None

    def __len__(self):
        return int(self._cum[-1])

    def _load(self, shard_idx: int):
        if self._cache_idx == shard_idx:
            return self._cache
        path = self.shards[shard_idx]
        table = (
            EpisodeBundleReader(path).transitions
            if os.path.isdir(path)
            else pq.read_table(path)
        )
        rows = table.to_pylist()
        rgb = _decode_frames(rows, "rgb", (FRAME, FRAME, 3), np.uint8)
        if rgb is None:
            raise ValueError(f"{self.shards[shard_idx]} has no rgb frames (export without render?)")
        depth = _decode_frames(rows, "depth", (FRAME, FRAME), np.float16)
        seg = _decode_frames(rows, "seg", (FRAME, FRAME), np.uint16)
        actions = np.stack([[r[k] for k in ACTION_KEYS] for r in rows]).astype(np.uint8)
        self._cache = (rgb, actions, depth, seg)
        self._cache_idx = shard_idx
        return self._cache

    def __getitem__(self, i: int):
        import torch

        shard = int(np.searchsorted(self._cum, i, side="right") - 1)
        start = i - int(self._cum[shard])
        rgb, actions, depth, seg = self._load(shard)
        s = slice(start, start + self.seq_len)
        return (
            torch.from_numpy(rgb[s].copy()),                    # (16,128,128,3) u8
            torch.from_numpy(actions[s].copy()),                # (16,10) u8
            torch.from_numpy(depth[s].astype(np.float32).copy()),  # (16,128,128)
            torch.from_numpy(seg[s].copy()),                    # (16,128,128) u16
        )


def baseline(data: str, steps: int = 50_000, batch: int = 32, seq_len: int = 16, lr: float = 3e-4,
             limit_steps: int | None = None, channels: str = "rgb",
             device: str = "auto", dtype: str = "bf16"):
    """RSSM-lite latent-prediction baseline vs copy-last-latent. See
    baseline.py for the model. Prints the acceptance ratio."""
    from .baseline import run_baseline

    return run_baseline(data, steps=steps, batch=batch, seq_len=seq_len, lr=lr,
                        limit_steps=limit_steps, channels=channels, device=device, dtype=dtype)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--task", required=True)
    e.add_argument("--episodes", type=int, default=50)
    e.add_argument("--render", type=int, default=1)
    e.add_argument("--seed0", type=int, default=0)
    e.add_argument("--epsilon", type=float, default=0.0, help="uniform-random action mixture")
    e.add_argument("--scale", type=float, default=1.0, help="cells per meter (2.0 = 0.5 m cells)")
    e.add_argument("--out", required=True)
    e.add_argument("--format", type=int, choices=(1, 2), default=2)
    b = sub.add_parser("baseline")
    b.add_argument("--data", required=True)
    b.add_argument("--steps", type=int, default=50_000)
    b.add_argument("--batch", type=int, default=32)
    b.add_argument("--seq-len", type=int, default=16)
    b.add_argument("--limit-steps", type=int, default=None, help="dev override for a quick run")
    b.add_argument("--channels", choices=["rgb", "rgbd"], default="rgb",
                   help="ablation axis: rgbd adds the metric-depth channel")
    b.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    b.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    c = sub.add_parser("build-causal")
    c.add_argument("--config", required=True, help="TOML experiment configuration")
    c.add_argument("--skip-worker-benchmark", action="store_true")
    c.add_argument("--pack-only", action="store_true")
    c.add_argument("--no-pack", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "export":
        export(args.task, args.episodes, args.out, render=args.render, seed0=args.seed0,
               epsilon=args.epsilon, scale=args.scale, format_version=args.format)
        return 0
    if args.cmd == "baseline":
        ratio = baseline(args.data, steps=args.steps, batch=args.batch, seq_len=args.seq_len,
                         limit_steps=args.limit_steps, channels=args.channels,
                         device=args.device, dtype=args.dtype)
        print(f"acceptance: model/copy MSE ratio = {ratio:.3f} (need < 0.9)")
        return 0 if ratio < 0.9 else 1
    if args.cmd == "build-causal":
        if args.pack_only and args.no_pack:
            ap.error("--pack-only and --no-pack are mutually exclusive")
        from .causal_data import build_causal_dataset, build_pack_only
        from .config import ResearchConfig

        config = ResearchConfig.from_toml(args.config)
        if args.pack_only:
            result = {"training_pack_manifest": str(build_pack_only(config))}
        else:
            result = build_causal_dataset(
                config,
                benchmark_workers=not args.skip_worker_benchmark,
                build_pack=not args.no_pack,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
