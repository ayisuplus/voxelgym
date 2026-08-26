---
status: accepted
---

# Keep World State authoritative and causal supervision derived

## Context

World-model training needs semantic events, exact changes, reward evidence, counterfactual branches, and privileged oracle data. If those artifacts participate in scheduling or physical decisions, enabling tracing can change the universe being measured. Treating a trace as restorable state would also create competing truths across Voxel Core, Python task state, datasets, and replay.

## Decision

World State is the sole source of physical truth. The transition function may consume only the prior World State, immutable world configuration, deterministic RNG state, the agent action, and explicitly declared interventions. World Snapshot and world hash remain the canonical persistence and identity contracts for that state.

World Events, State Deltas, causal parent links, reward evidence, scene relations, and Oracle View values are derived supervision. They may observe or compare boundary states, but physics, scheduling, collision, task interventions, and replay must never read them to decide the next World State. `TraceLevel::Off`, `Events`, and `Full` must produce the same post-transition World State and hash.

An Intervention is different from its trace: the typed intervention is an explicit input that mutates World State at a declared boundary; the `intervention_applied` event and associated delta merely describe it. Counterfactual evaluation forks one canonical pre-state before applying a treatment.

Python Env Snapshot composes World Snapshot with causal-trace continuation state, task state, Gym RNG, episode bookkeeping, and sensor caches so the complete training interaction can continue exactly, including deterministic event IDs and cross-tick parent links. Agent View contains only the declared policy observation. Oracle View and causal labels remain privileged unless an experiment explicitly promotes a named oracle modality into the policy input.

Episode Bundle v2 stores transitions, derived events, derived deltas, and checkpoints in separately joinable tables. Legacy v1 episodes remain readable without implicit conversion. Dataset storage does not become an execution log that drives the simulator.

## Consequences

- Tracing can be disabled on the hot path without changing physical behavior, and traced/untraced equivalence is testable by snapshot and hash.
- Causal datasets can be regenerated, extended, or corrected without redefining historical World State.
- Event vocabularies can evolve, but a `Full` trace must account for every canonical boundary-state change. Consumers still must not use deltas as a replacement replay patch.
- Full tracing pays additional comparison, hashing, and storage cost, so collection must choose its trace level explicitly.
- New state that can affect future simulation requires a World Snapshot version change; new derived labels require an Episode Bundle or trace-schema evolution instead.
- Reward and oracle code must remain read-only with respect to World State. Reward calculation is also read-only with respect to Task State and returns explicit updates for the environment to commit. Task-driven world mutations must move to an explicit intervention or simulation phase.
