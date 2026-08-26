# VoxelGym Python

VoxelGym Python owns the training-environment boundary above Voxel Core: task state, policy observations, privileged supervision, exact episode continuation, and episode storage. Core time, spatial, transition, event, and delta vocabulary is defined in [Voxel Core context](../../crates/voxel-core/CONTEXT.md). The authority boundary is fixed by [ADR 0002](../../docs/adr/0002-world-state-and-derived-causal-supervision.md).

## Episode state

**Env Snapshot**:
A complete Python episode checkpoint used for exact continuation. It contains World Snapshot bytes, versioned task state, the Gym `np_random` state, episode seed, terminated/truncated bookkeeping, the most recent structured reward outcome, render and LiDAR caches, the tick at which each cached sensor value was sampled, the serializable-intervention cursor, and optional native trace bookkeeping when the loaded binding exposes that capability. Restore does not rerun scenario generation or task reset hooks. A World Snapshot alone is insufficient when task, RNG, reward, intervention identity, trace continuity, or sensor scheduling can affect subsequent Gym outputs.
_Avoid_: World Snapshot, observation snapshot, dataset checkpoint

Task state is JSON-safe and versioned through `state_dict()` / `load_state_dict()`. It covers every mutable task field. Reward evaluation is a pure function of the post-transition world, the pre-reward task state, and a read-only view of that transition's events: it returns a `RewardOutcome` containing total, named components, termination reason, selectively matched numeric `evidence_event_ids`, separate semantic `evidence_labels`, and explicit task-state updates. Labels are task assertions and never masquerade as World Event IDs. `Env.step()` commits updates only after evaluation while retaining the Gymnasium scalar reward API. Reward evaluation must not mutate either World State or Task State; task-induced world changes belong to the explicit intervention phase.

## Visibility boundary

**Agent View**:
Exactly the observations made available to the policy by the configured Gym observation space at a transition boundary. The current base view contains the local voxel window, inventory, pose, and raycast, with optional RGB, depth, segmentation, normals, and LiDAR. Cached sensors remain observations from their recorded sample ticks. Agent View never implicitly includes full snapshots, hashes, hidden task truth, causal traces, or `PyWorld.oracle_state()`.
_Avoid_: every value available to Python, recorder row

**Oracle View**:
Privileged, read-only supervision derived from the same boundary World State plus explicitly serialized task and episode truth. It may include frame-explicit metric kinematics, complete state or snapshot references, spatial relations, scene structure, World Events, State Deltas, hashes, task truth, and reward evidence. It is available to recorders, evaluators, dataset builders, and explicitly identified oracle experts; it must not enter Agent View unless a dataset or experiment declares that modality as policy input.
_Avoid_: a second simulation state, hidden mutation channel

Agent View and Oracle View can contain different projections, but both must identify the same branch, transition boundary, spatial frame, and sensor sample time. Oracle computation must not perturb World State, RNG, scheduling, snapshot bytes, or hashes.

## Episode storage

**Episode Bundle v2**:
An immutable `.vxbundle` directory for one recorded episode or branch. It contains:

- `manifest.json`: format/version, task, seed, branch, trace level, step count, final hash, the exact external causal parents used from the initial Trace State, observation configuration (including whether the spacetime view was enabled), and table filenames.
- `transitions.parquet`: stable transition ID, branch, before/after ticks and hashes, action fields, scalar and component rewards, termination flags/reason, sensor sample ticks, and optional Agent/Oracle View payloads.
- `events.parquet`: World Events joined by transition ID and stable event ID.
- `deltas.parquet`: State Deltas joined by transition and event ID.
- `checkpoints.parquet`: optional World Snapshot and serialized Env Snapshot payloads at transition boundaries.

Transition IDs are unique within a bundle. Every transition matches the manifest branch, advances exactly one tick, chains contiguously in table order, chains adjacent hashes whenever both are present, carries Agent and Oracle payloads, and never claims a future sensor sample. Each transition also stores the canonical ordered intervention specs and the boundary between caller-supplied and task-generated specs; replay injects only the former, regenerates the latter from restored Task State, and requires the combined sequence to match exactly. The legacy `swap` field remains a read-only compatibility side channel, not the source of truth. Event IDs are unique, every event references an existing transition, parent IDs reference existing non-future events on the same branch, and the causal graph is acyclic. A recorder starting from an already-running branch reads the allowed ancestry from the initial native Trace State; the manifest then stores exactly the subset actually referenced as `external_parent_ids`. Missing parents are never auto-promoted into boundary ancestors. Every delta references an existing event and transition. Per-transition event/delta counts bind the trace tables to their owning transitions. `TraceLevel::Off` requires absent hashes and empty event/delta tables; `Events` requires absent hashes and an empty delta table; `Full` requires both boundary hashes, exactly one action root, and exactly one matching World `tick_before → tick_after` delta per non-empty transition. A no-op action may have no additional state deltas, but the tick delta is never optional.

World-model transition windows expose the first boundary's Agent/Oracle payloads only as inputs and the last boundary's payloads as explicit prediction targets. Factual/counterfactual pairs require identical pre-state views, identical action sequences, and exactly one explicitly recorded intervention side; differing actions are never mislabeled as an intervention effect.

The original v1 single-Parquet episode shard remains read-only input. Readers detect v1 or v2 and tag rows with their source version; they never rewrite a v1 shard as v2 implicitly. Episode Bundle tables are training artifacts derived from state and transitions, not inputs to Voxel Core physics.
