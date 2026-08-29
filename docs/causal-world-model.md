# Causal world-model platform

VoxelGym's research loop is local-first: CPU workers generate authoritative
Episode Bundle v2 trajectories, Training Pack v1 streams their derived tensors,
and one NVIDIA GPU trains and evaluates offline world models.  It downloads no
datasets, pretrained models, or weights.

## Data and visibility contract

`VoxelGymEnv(physics=...)` records requested and effective physics, scale,
rational clock, and sensor profile through reset info, Oracle View, bundle
metadata, manifests, and resolved run configuration.  Omitting `physics`
retains the historical tick-for-tick trajectory.

Each Pack window is 65 boundary Agent Views joined by 64 transitions.  State
modalities are RGB, depth, normals, LiDAR range, local voxels, pose, and
inventory.  Agent actions and canonical external interventions are separate
declared control inputs.  During training, a random observed prefix is kept and
the following 16 state ticks are masked; the action/intervention stream remains
visible through prediction horizons 1/4/8/16.

Segmentation, rewards, termination, events, deltas, time-to-event, typed causal
edges, hashes, snapshots, and task truth never become inputs.  Render and LiDAR
sample IDs make all ticks sharing a cached sensor frame one masking group.
[ADR 0003](adr/0003-observed-state-and-declared-control.md) fixes this boundary.

Dataset Manifest v1 checksum-binds task/seed splits, policy mix, physics,
clock, scale, sensors, and pair identity to immutable bundles.  Training Pack
v1 stores memory-mapped Zstd Parquet segments and a build-time Eval Suite.  The
suite selects at most 64 episode-balanced windows per task/domain, never more
than one window per episode; pair branches use the same boundary.  Its
fingerprint is independent of run seed.

## Production

Start with `experiments/causal-pilot.toml`.  The builder benchmarks 8/16/24
spawned workers three times and selects the highest median throughput below 70%
system-memory use.  A deterministic supercycle yields exactly 50% oracle
expert, 30% epsilon-mixed expert (0.05/0.15/0.30 equally), and 20% aligned
intervention branches.  Pair actions use the shared epsilon-0.15 expert stream;
intervention kinds rotate with deterministic invalid-candidate fallback.

```powershell
python -m voxelgym.datasets build-causal --config experiments/causal-pilot.toml
python -m voxelgym.datasets build-causal --config experiments/causal-pilot.toml --pack-only
```

The 500 GiB build assigns 30% of trajectories to training-domain
randomization.  The independent OOD Pack contains scale 2, 40 Hz, gravity
0.5x/1.5x, and altered fluid periods, all in the test split.

## Five-arm ladder

| arm | config | fixed objective |
|---|---|---|
| RSSM | `experiments/rssm-pilot.toml` | EMA BYOL next-latent + RGB reconstruction |
| Dynamics-T | `experiments/dynamics-transformer-pilot.toml` | latent/depth/seg/reward/terminal |
| Causal-T | `experiments/causal-transformer-pilot.toml` | Dynamics-T + event/delta/time/typed edge |
| CF-T | `experiments/causal-pilot.toml` | Causal-T + paired latent effect/propagation/reward delta |
| Temporal JEPA | `experiments/jepa-pilot.toml` | EMA teacher masked-latent cosine + variance |

The three Transformer arms share the same architecture and initialization and
activate losses cumulatively.  The formal stack uses 12 layers, width 768, 12
heads, MLP 3072, context 64, BF16 autocast, AdamW, activation checkpointing,
microbatch 8, and accumulation 8.  JEPA uses `tau=0.996`; RSSM retains latent
1024 and hidden 512.  Optimizer state stays FP32 and `torch.compile` is not a
v1 dependency.

Checkpoint v2 atomically stores model/EMA, optimizer, scheduler, global step,
Python/NumPy/Torch/CUDA RNG, deterministic sampler/mask state, resolved model
metadata, and Pack identity.  `--seed` overrides only the run seed and the
resolved JSON records the final value.

```powershell
# 1,000-step hardware smoke test
python -m voxelgym.train --config experiments/jepa-pilot.toml --seed 0 --stop-after 1000

# Three-seed 25k comparison; CF-T can continue to its planned 100k total.
python -m voxelgym.train --config experiments/causal-transformer-pilot.toml --seed 1
python -m voxelgym.train --config experiments/causal-pilot.toml --seed 1 --stop-after 25000
```

## Evaluation and gates

Formal evaluation requires a test split and consumes the Pack's recorded Eval
Suite directly; it never falls back to validation or train.  `--run` is
repeatable.  Aggregation rejects mismatched data, suite, step, or model identity
and reports t-95% intervals plus same-seed Causal−Dynamics and CF−Causal paired
differences.

```powershell
python -m voxelgym.evaluate --run runs/<dynamics-s0> --run runs/<causal-s0> `
  --run runs/<cf-s0>
python -m voxelgym.evaluate --run runs/<cf-s0> `
  --pack data/causal-ood/pack/manifest.json
```

Reports include horizon/copy-last ratios, depth AbsRel/RMSE, segmentation mIoU,
reward MAE, terminal/event/delta/typed-edge macro AUPRC/F1, time-to-event tick
MAE, unique-pair propagation AUPRC/balanced accuracy, reward-delta MAE/zero
ratio/sign accuracy, pair latent-effect error, and representation effective
rank.  Every arm also runs deterministic frozen linear probes: AdamW,
`lr=1e-3`, 2,000 steps, validation-selected thresholds locked for test.

The 100 GiB pilot gates the 500 GiB build on 100% replay verification, policy
and intervention distributions, positive three-seed causal increments, RSSM
h1 `<0.9`, other arms h1/h4 `<0.9` and h16 `<1.0`, JEPA effective rank at
least `0.25 × d_model`, GPU utilization at least 80%, loader wait below 10%,
and peak reserved memory below 30 GiB.  All 20 tasks and every OOD domain remain
separate in the report.
