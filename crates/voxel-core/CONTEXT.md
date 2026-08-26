# Voxel Core

Voxel Core defines the deterministic voxel world whose state can be recorded, restored, branched, and advanced under configurable physics.

The truth boundary for causal supervision is fixed by [ADR 0002](../../docs/adr/0002-world-state-and-derived-causal-supervision.md).

**World State**:
The complete canonical simulator state at one Simulation Tick boundary whose values can affect a future physical transition. It includes the exact clock and spatial configuration, deterministic RNG state, world cells, actors, dynamic entities, inventories, and pending physical work. World Snapshot restoration and world hashing define its persistence and identity boundaries. Trace records, oracle projections, render caches, and other recomputable observations are not World State.
_Avoid_: observation dictionary, causal trace, episode row

## Time and transitions

**Simulation Tick**:
The ordinal of a simulation boundary, equal to the number of completed world transitions. Tick 0 is the initial boundary; a transition observed with `clock_before.tick == t` completes with `clock_after.tick == t + 1`. `ClockConfig` stores the immutable episode duration of one tick as a reduced positive rational number of seconds, defaulting to `1/20`. Elapsed time is `tick * seconds_per_tick`; sensor sample time and data age are always computed from boundary ticks, never wall-clock time.
_Avoid_: frame number, render frame, elapsed wall time

**Transition**:
One deterministic training step from boundary `t` to boundary `t + 1`. Across the Python and Rust layers its logical order is: (1) the atomic batch of explicit external and task interventions, if any; (2) agent action; (3) entity integration; (4) scheduled block work; (5) environmental propagation in the fixed fluid, fire, circuit, then TNT order; (6) item logic; and (7) boundary observation and reward evaluation. Inputs are validated before the intervention phase, and a rejected intervention batch leaves the common branch point unchanged. Trace phases retain the finer mechanism names. An observation belongs to the boundary at which it was sampled; a reused sensor value carries its original sample tick.
_Avoid_: using tick to mean both a boundary and the transition after it

**Intervention**:
A typed, serializable exogenous mutation applied explicitly at a transition boundary, outside the agent action vocabulary. Current core forms are setting a cell, teleporting the agent, setting agent velocity, giving an item, and swapping an existing item into the selected hotbar slot. An intervention changes World State and therefore can change snapshots and hashes. Its corresponding event and delta are derived records of that input, not the mechanism that performs the mutation. Counterfactual branches must fork one common pre-state, apply the treatment only to its designated branch, and run equal-length action sequences.
_Avoid_: arbitrary mutation closure, hidden task-side world edit

## Space

**Spatial Frame**:
A stable `FrameId` plus an origin, axis convention, orientation convention, and units. `FrameId::WORLD` is 0. `CellCoord` is a discrete voxel address; `WorldPos` is continuous and measured in cells; `MetricPos` is continuous and measured in meters. `SpatialScale` is the reduced cells-per-meter ratio that converts between them. Cell ownership uses mathematical floor, including at negative coordinates. World yaw 0 faces +Z and yaw 90 faces -X; negative pitch looks upward. Any non-world frame must have an explicit transform before values are compared or stored together.
_Avoid_: unlabelled `(x, y, z)`, treating cell and meter values as interchangeable

Static blocks are identified by `CellCoord`. Dynamic entities, semantic structures, and semantic regions use their distinct stable ID types; equal integer payloads across ID types do not identify the same object. Reachability uses the live agent AABB and collision oracle for local motion and executes the real action/physics transition on an isolated snapshot branch for drop landings, including configured time, fall damage, and death.

Legacy `ScenarioSpec` regions are inclusive scale-1 meter volumes at every public `World::new*` constructor. Construction resolves them exactly once into the world's cell frame, preserving the represented volume at higher cell densities (including across negative coordinates). Each distinct legacy `(Region, cell)` receives domain-separated, non-zero `RegionId` and `StructureId` values derived from canonical content; these IDs do not depend on scenario ordering or cell density. Explicit `SemanticRegionSpec` construction supports several uniquely identified regions sharing one non-zero `StructureId`. `World::semantic_regions()` exposes the resolved metadata read-only. Restored snapshots already contain resolved cell-space regions and are never scaled again. `World::scale()` and `World::physics()` are read-only episode metadata; canonical physics overrides are supplied by consuming `World::with_physics` during construction.

## Causal supervision

**World Event**:
An immutable semantic observation that something happened during a transition. It has a deterministic ID scoped by branch, tick, phase, and ordinal; semantic kind; actor and target references; optional location; mechanism; zero or more causal parent IDs; and one root cause (`Action`, `Intervention`, `Exogenous`, or an explicitly named `Periodic` mechanism). Parent links form a DAG and may only point within the current or an earlier transition. Scheduler lineage is carried across tick boundaries so delayed effects remain descendants of the input that scheduled them. Events describe transitions but are not World State and are never read by physics.
_Avoid_: scheduler command, mutable event entity, source of simulation truth

**State Delta**:
An exact before/after value change attributed to one World Event. It names the affected subject and field or cell; floating-point values retain their IEEE bit patterns. At full trace level, every canonical state field that changed across the transition is represented, including RNG, actor/entity state, inventories, schedulers, mechanism sets, furnaces, and cells. Deltas do not include a self-referential world-hash pseudo-field and do not replace a World Snapshot. A delta explains recorded change; it never applies that change during ordinary stepping.
_Avoid_: patch to replay, authoritative state update

`TraceLevel::Off` emits neither events nor deltas and uses the original hot path. `Events` emits semantic events without exact values or hashes. `Full` additionally emits boundary hashes and exact deltas. All levels must finish with identical World State and hash for the same pre-state and action.

## Persistence language

**World Snapshot**:
The canonical, versioned byte representation of all restorable world state that can affect future simulation.
_Avoid_: Checkpoint, save file

**Checkpoint**:
A World Snapshot captured at a particular tick and embedded in a dataset for branching or replay verification.
_Avoid_: Snapshot format, save file

World Snapshot v8 persists `ClockConfig`, stable semantic region/structure metadata, and the ordered dirty-cell queue. The semantic and dirty records form a little-endian trailer after the historical snapshot body. v5-v7 restore with the historical `1/20` default, content-derived semantic IDs, and an empty dirty queue. The v8 world hash includes this trailer; `legacy_hash_v7` excludes it. Python episode persistence composes this contract as an Env Snapshot; it does not redefine core state.
