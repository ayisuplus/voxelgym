---
status: accepted
---

# Evolve World Snapshots with explicit versions

World Snapshots are a cross-context persistence contract used by Voxel Core, the PyO3 adapter, and Recorder/Replay. New snapshots use version 7, include every state field that can affect future simulation in both restoration and hashing, and reject trailing bytes for known versions. Versions 5 and 6 remain readable on a best-effort basis; because their writers never stored a falling block's accumulated fall distance, restoration uses a deterministic, conservative value of zero and cannot reproduce the exact future of an old mid-fall snapshot. We chose an explicit version bump instead of silently changing version 6 because datasets embed Checkpoints and the two layouts must not be ambiguous.
