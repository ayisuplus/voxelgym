# voxelgym

**Minecraft's semantics, rebuilt as a deterministic physics instrument — not a game, a lab.**
借 Minecraft 的物理语义，造一台确定性、可消融的物理实验机：给 AI 训练"物理世界理解"(重力、支撑、流体、因果、电路)的可交互数据集发动机。

A from-scratch voxel physics simulator (no Mojang code or assets, Apache-2.0) where the player is an AI: Rust core + PyO3 bindings + gymnasium API + full-truth sensors + recording/replay + a physics-probe task suite + a live web demo.

## Why not just use Minecraft?

| | Minecraft (Java, MineRL/Malmö/MineDojo) | voxelgym |
|---|---|---|
| Throughput | ~55–190 steps/s/instance | **321k steps/s single env, 1.03M on 64 envs** (measured) |
| Determinism | no — entity tick order, GC, JVM jitter | **bit-exact**: seed + action sequence → identical world hash (xxh3), snapshot/restore mid-episode |
| Ground truth | pixels only | per-tick voxel window (block ids+states), depth, segmentation, **surface normals**, LiDAR range image, full world snapshots |
| Physics | fixed, closed, 20 TPS lock | **rule-table driven and ablatable** — gravity, fluid spread, gate delay, fall damage… are config fields; exact rational clocks and a **voxel scale knob** let the same physical world run at different tick rates and spatial resolutions |
| Sensors | screen capture | CPU DDA camera (rgb/depth/seg, exact by construction) + **spinning LiDAR model** (0.4 ms/scan) with seeded noise |
| Legality | EULA forbids redistribution | Apache-2.0, zero Minecraft assets |
| Task solvability | unknown | every probe task validates solvability at generation by branch simulation; oracle experts gated ≥ 0.95 |

The point is not "another Minecraft clone". It is a **dataset engine for interactive physics**: unlimited action-conditioned trajectories with exact action labels and full world truth, where the laws of physics themselves are an experimental variable.

## 60-second demo script

```bash
.venv/Scripts/python.exe web/server.py   # open http://127.0.0.1:8000
```

The page auto-cycles the curriculum (showcase mode). Talking points as it runs:

1. **"Looks like Minecraft. It is not."** — the orange figure is an AI executing a 10-action discrete command each tick; everything you see is computed, not scripted.
2. **Truth channels** — the segmentation and LiDAR panels are exact per-ray ground truth from the same DDA traversal as the physics. Minecraft can never give you this.
3. **Determinism** — `python bench/determinism.py --seed 42 --ticks 20000` prints two identical 64-bit hashes: same actions, same universe, bit for bit.
4. **Speed** — `python bench/throughput.py --envs 64 --ticks 100000`: ~1M world-ticks/s on one desktop CPU. A Java MC stack manages ~190.
5. **Ablatable physics** — the training world is a hypothesis you can edit: change gravity or fluid spread in `PhysicsConfig`, retrain, and ask whether the model learned *physics* or learned *Minecraft*.
6. **The world is a digital circuit simulator** — torches are NOT gates with unit delay, wire joins are wired-OR: RS latches store bits and 3-torch rings oscillate at exactly 2× the stage count (tested). Task `logic_probe` trains input→output reasoning on real gate-level causality.
7. **Every episode is a causal dataset** — Episode Bundle v2 separates transitions, semantic events, exact deltas, and complete environment checkpoints; `replay.py --verify` re-executes actions and checks world/task/reward/sensor/causal equivalence.

## Physical spacetime and causal supervision

World Snapshot v8 stores an immutable reduced-rational simulation clock. The
default remains exactly 20 Hz; changing the tick duration rescales physical
rates and quantizes timers without changing the default trajectory. Spatial
interfaces distinguish voxel cells, continuous world coordinates, metric
coordinates, poses, AABBs, frames, and stable semantic IDs. The Python oracle
exposes collision-consistent reachability, shortest paths, visibility, scene
objects, metric kinematics, and clock/sample-age metadata.

Normal `step()` remains the high-throughput path. `step_traced()` derives a
rooted event DAG and, at full level, exact state deltas and boundary hashes.
Tagged interventions support deterministic fork → intervene → rollout →
compare experiments without Python mutation closures. These traces never drive
physics: World State remains the sole truth source.

```python
from voxelgym import VoxelGymEnv

env = VoxelGymEnv(preset="flat", dt_numerator=1, dt_denominator=40,
                  spacetime=True)
obs, info = env.reset(seed=7)
obs, reward, terminated, truncated, info = env.step_traced(
    {key: 0 for key in env.action_space.spaces}, trace_level="full"
)
checkpoint = env.snapshot().to_bytes()
oracle = env.oracle_view()  # recorder-only; never inserted into policy input
```

## The curriculum (20 tasks, oracle-gated ≥ 0.95)

- **Navigation**: `navigate_to_target`
- **Tech tree** (DreamerV3-style long-horizon): `collect_log → craft_planks → craft_table → craft_wooden_pickaxe → mine_stone → craft_stone_pickaxe → smelt_iron → craft_iron_pickaxe → mine_diamond`
- **Physics judgment**: `collapse_judge`, `firebreak_judge` (predict collapse/fire by branch simulation, answer on a pad)
- **Physics intervention**: `water_routing`, `bridge_over_lava`, `buried_escape`, `tnt_clear`
- **Circuits**: `circuit_door`, `circuit_door_two`, `plate_door`, `logic_probe`

Physics implemented with Minecraft's public constants: entity force model (Y→X→Z clipped collision), fall damage, suffocation, lava/fire, water swimming, fluid cellular automata (7-cell water spread, source formation, water+lava→stone/cobble), loose-block falling, TNT explosions, and gate-level circuits with unit delay. Terrain: five biomes (ocean/plains/desert/hills/volcanic) + caves + stratified ores, all seeded position hashes — bit-exact per platform.

## Verification

- Rust workspace tests cover the physics truth tables, circuit memory and timing, snapshot replay, DDA renderer/LiDAR goldens, PyO3 bindings, and batch interfaces.
- The complete pytest suite covers the env contract, recorder/replay hashes, render and LiDAR goldens, task/expert behavior, data pipelines, VQA, web protocols, and no-clip properties.
- Oracle experts: 13 gated tasks ≥ 0.95 success over 20 episodes; `mine_diamond` tracked (0.80) without a hard gate.
- Determinism: 20k-tick double run, identical final hash.

## Measured results (this repo runs them)

**Sensor ablation — does metric depth help dynamics learning?** RSSM-lite (BYOL latent prediction), same data/seed/steps, only the input channels differ:

| data | rgb ratio | rgbd ratio | rgbd model MSE vs rgb |
|---|---|---|---|
| `collect_log` (pure expert) | 6.31 | 5.80 | −9.8% |
| `collect_log` ε=0.1 (motion-rich) | 0.777 | **0.772** | −6.6% |

Depth gives a small, consistent gain — and the near-static dataset row shows why data diversity matters (both arms fail the <0.9 gate when footage barely moves; the ε-mixed data passes it).

**Resolution transfer (the 割圆术 experiment)**: a world model trained at 1 m cells, evaluated frozen on the same curriculum at 0.5 m cells (`scale=2`):

| eval set | ratio |
|---|---|
| scale-1 test | 1.342 |
| scale-2 (0.5 m) | 1.423 (**+6%**) |

Learned dynamics largely survive the 2× refinement — the finer-cut world does not break the representation. Caveat: `navigate_to_target` footage is locomotion-only (copy-last is strong, both ratios > 1); interaction-rich transfer data is the follow-up.

## Quickstart

```bash
# Rust stable + Python 3.11
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e "python[dev]"
maturin develop --release -m crates/voxel-py/Cargo.toml   # or scripts/build_dev.bat
cargo test --workspace --release && pytest python/tests -q

python -m voxelgym.experts --task smelt_iron --episodes 20   # oracle gate
python -m voxelgym.experts --task collapse_judge --episodes 1 --record data/v2 --format 2
python -m voxelgym.replay data/v2/<episode>.vxbundle --verify
python bench/determinism.py --seed 42 --ticks 20000
python bench/throughput.py --envs 64 --ticks 100000
python web/server.py                                          # live demo
```

## Causal world-model research loop

The local-first research path keeps Episode Bundle v2 and World Snapshot v8 as
the replay authority. It derives a checksum-bound Dataset Manifest v1 and
memory-mapped Training Pack v1. Each sample aligns 65 boundary Agent Views with
64 action/intervention transitions and masks future states, so long-horizon
predictions receive declared controls without seeing their outcomes. Oracle
events, deltas, hashes, snapshots, and task truth remain labels only.

The model ladder is RSSM, an apples-to-apples Dynamics/Causal/Counterfactual
Transformer ablation, and a Temporal JEPA baseline. The three ~100M Transformer
arms share architecture and initialization; only their active supervision
changes. A build-time, run-seed-independent Eval Suite makes task/domain and
paired comparisons identical across arms.

```powershell
# 100 GiB pilot: benchmarks 8/16/24 generators, produces bundles + pack.
python -m voxelgym.datasets build-causal --config experiments/causal-pilot.toml

# Native-Windows eager BF16 training; metrics, TensorBoard and atomic resume
# points are written below runs/.
python -m voxelgym.train --config experiments/causal-pilot.toml
python -m voxelgym.evaluate --run runs/<run-directory>

# Other equal-compute arms; --seed is written into the resolved config.
python -m voxelgym.train --config experiments/rssm-pilot.toml --seed 0
python -m voxelgym.train --config experiments/dynamics-transformer-pilot.toml --seed 0
python -m voxelgym.train --config experiments/causal-transformer-pilot.toml --seed 0
python -m voxelgym.train --config experiments/jepa-pilot.toml --seed 0

# Generate the independent scale/clock/gravity/fluid OOD pack, then evaluate
# the same frozen checkpoint on it.
python -m voxelgym.datasets build-causal --config experiments/causal-ood.toml
python -m voxelgym.evaluate --run runs/<run-directory> `
  --pack data/causal-ood/pack/manifest.json
```

Formal evaluation refuses a missing test split and never falls back to training
data. Repeating `--run` reports t-95% intervals and matched-seed
Causal−Dynamics / Counterfactual−Causal differences. It also reports frozen
linear probes, typed-edge and pair metrics, reconstruction quality, and JEPA
effective rank.

The 500 GiB configuration fixes the source mix at 50% oracle expert, 30%
epsilon-mixed expert, and 20% paired interventions, with exactly 30% of source
trajectories domain-randomized per ten-trajectory production cycle. Do not run
it until the pilot replay, distribution, utilization, memory, causal-increment,
and short-horizon gates pass. See [the causal platform guide](docs/causal-world-model.md).

## Testing and coverage

CI keeps the stable correctness suites and independently gates production line
coverage at 80% for the Rust workspace, Python plus the web server, and browser
`app.js`. All three reports include branch coverage without using it as a gate;
HTML and machine-readable reports are retained as GitHub Actions artifacts even
when a threshold fails. See [docs/testing.md](docs/testing.md) for the exact local
commands, coverage scope, and artifact layout.

## Layout

```
crates/voxel-core   # sim: registry, chunks, worldgen, entity, fluids, loose, fire, tnt, circuits
crates/voxel-view   # CPU DDA renderer + LiDAR (shared traversal = exact truth)
crates/voxel-py     # PyO3 bindings (snapshot/hash, batch vec envs, sensors)
python/voxelgym     # gymnasium env, tasks, oracle experts, recorder/replay, datasets
web/                # live demo server (WebSocket frame stream + controls)
bench/              # throughput + determinism benchmarks
```

## Honest boundaries

- 1 m voxels and discrete ticks: this is an **algorithm curriculum** tool, not a deployment-grade robotics sim (no cm-accurate dynamics, no contact friction cones).
- Bit-exactness is per-platform (float transcendentals in noise paths); datasets carry full snapshots every 600 ticks for cross-machine branching.
- Not the full Minecraft feature set (no redstone pistons/comparators, no mobs, no nether) — the gate-level circuit basis (torch/repeater) is computationally complete by design, not by imitation.

## License

Apache-2.0. No Minecraft code, assets, or trademarks.
