"""Resolution-transfer probe (the "割圆术" experiment): does dynamics learned
at 1 m cells transfer to 0.5 m cells?

Design: train the RSSM-lite baseline on the scale-1 navigate dataset, then
evaluate the FROZEN model on (a) the scale-1 test split and (b) the scale-2
dataset. Both scores are latent-MSE ratios vs the copy-last baseline, which
is self-normalizing per dataset — so the two ratios are directly comparable.

Usage: python bench/resolution_transfer.py [--steps 4000]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from voxelgym.baseline import run_baseline

    print("=== train on scale-1 (1 m cells), eval on scale-1 test + scale-2 set ===")
    run_baseline(
        "data/nav_s1",
        steps=args.steps,
        batch=args.batch,
        seq_len=16,
        lr=3e-4,
        limit_steps=None,
        channels="rgb",
        transfer_data="data/nav_s2",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
