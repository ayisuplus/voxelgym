# Repository context map

This is the canonical entry point for durable domain context in this multi-component repository. Read the repository-wide intent in [README.md](README.md), then use the table below to locate the component that owns a change.

| Area | Owned scope | Primary entry points |
| --- | --- | --- |
| `crates/voxel-core` | Deterministic world state, physics, terrain, blocks, entities, fluids, fire, TNT, circuits, inventory, and recipes | [Voxel Core context](crates/voxel-core/CONTEXT.md), `crates/voxel-core/src/lib.rs`, modules under `crates/voxel-core/src/` |
| `crates/voxel-view` | CPU DDA rendering and LiDAR truth sensors | `crates/voxel-view/src/lib.rs`, `crates/voxel-view/src/lidar.rs` |
| `crates/voxel-py` | PyO3 boundary between the Rust engine and Python | `crates/voxel-py/src/lib.rs` |
| `python/voxelgym` | Gymnasium environment, tasks, experts, datasets, recording/replay, vector environments, and VQA tooling | `python/voxelgym/__init__.py`, package modules, `python/tests/` |
| `web` | FastAPI/WebSocket demo server and browser client | `web/server.py`, `web/static/` |
| `bench` | Determinism, throughput, and experiment benchmarks | scripts under `bench/` |
| `data` | Repository-owned data inputs and small artifacts | files under `data/` |

Cross-component decisions belong in [docs/adr/](docs/adr/). If an area grows enough to need its own durable context document, add `<area>/CONTEXT.md` and register it in this map; do not create an unregistered context file.
