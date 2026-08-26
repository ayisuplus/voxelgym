---
status: accepted
---

# Evolve World Snapshots with explicit versions

World Snapshots are a cross-context persistence contract used by Voxel Core, the PyO3 adapter, and Recorder/Replay. New snapshots use version 8, include every state field that can affect future simulation in both restoration and hashing, persist the reduced `ClockConfig`, and reject trailing bytes for known versions. Versions 5 through 7 remain readable: versions 5 and 6 restore a falling block's omitted accumulated fall distance to a deterministic, conservative zero, and all three legacy versions restore with the historical `1/20`-second tick because they did not encode a clock. An old v5/v6 snapshot taken mid-fall therefore cannot promise exact future equivalence. We use explicit version bumps instead of silently changing a known layout because datasets embed Checkpoints and the layouts must never be ambiguous.
