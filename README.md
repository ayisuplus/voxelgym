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
| Physics | fixed, closed, 20 TPS lock | **rule-table driven and ablatable** — gravity, fluid spread, gate delay, fall damage… are config fields; **voxel scale knob**: `scale=2` runs the same physical world at 0.5 m cells |
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
7. **Every episode is a dataset shard** — Parquet with actions, voxel windows, frames, and world checkpoints every 600 ticks; `replay.py --verify` re-executes the action log and asserts the final hash.

## The curriculum (20 tasks, oracle-gated ≥ 0.95)

- **Navigation**: `navigate_to_target`
- **Tech tree** (DreamerV3-style long-horizon): `collect_log → craft_planks → craft_table → craft_wooden_pickaxe → mine_stone → craft_stone_pickaxe → smelt_iron → craft_iron_pickaxe → mine_diamond`
- **Physics judgment**: `collapse_judge`, `firebreak_judge` (predict collapse/fire by branch simulation, answer on a pad)
- **Physics intervention**: `water_routing`, `bridge_over_lava`, `buried_escape`, `tnt_clear`
- **Circuits**: `circuit_door`, `circuit_door_two`, `plate_door`, `logic_probe`

Physics implemented with Minecraft's public constants: entity force model (Y→X→Z clipped collision), fall damage, suffocation, lava/fire, water swimming, fluid cellular automata (7-cell water spread, source formation, water+lava→stone/cobble), loose-block falling, TNT explosions, and gate-level circuits with unit delay. Terrain: five biomes (ocean/plains/desert/hills/volcanic) + caves + stratified ores, all seeded position hashes — bit-exact per platform.

## Verification

- 78 Rust tests (`voxel-core`) + 6 renderer/LiDAR golden tests (`voxel-view`): fluid truth tables, 15-cell power decay, NOT/NOR gate tables, RS latch memory, ring-oscillator period, snapshot replay under oscillation, DDA vs brute force, scale-2 worldgen/physics/snapshot.
- 53 pytest: env contract, recorder/replay hash match, render golden (seg/depth/normals bit-exact vs analytic DDA), LiDAR golden range, no-clip property tests (5 seeds × 3000 ticks, terminal-velocity fall onto a 1-cell platform).
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
pip install maturin pytest gymnasium numpy pyarrow fastapi "uvicorn[standard]"
# equivalently: pip install -e "python[dev]"
maturin develop --release -m crates/voxel-py/Cargo.toml   # or scripts/build_dev.bat
cargo test --workspace && pytest python/tests -q          # also what .github/workflows/ci.yml runs

python -m voxelgym.experts --task smelt_iron --episodes 20   # oracle gate
python bench/determinism.py --seed 42 --ticks 20000
python bench/throughput.py --envs 64 --ticks 100000
python web/server.py                                          # live demo
```

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
