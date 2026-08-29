---
status: accepted
---

# Predict from observed state and declared control

## Context

A long-horizon world model cannot answer an action-conditioned question if it
sees only the first action.  Conversely, a counterfactual predictor is invalid
if treatment-side observations recorded after an intervention are available to
the predictor.  External interventions also cannot be treated as ordinary
agent actions: they are privileged environment controls with different
authority and a separately auditable origin.

## Decision

A Training Pack window contains 65 boundary Agent Views and the 64 transitions
between them.  Every transition supplies an agent action and a separately
encoded, canonical external-intervention declaration.  During masked
prediction the model may see the observed prefix and all declared controls up
to each requested horizon, but target-side Agent Views remain masked.

The intervention declaration is an allowed control input.  Its resulting
event, delta, hash, snapshot, Oracle View, and hidden task truth remain
privileged targets and cannot enter the model-input schema.  Agent actions and
external interventions use distinct encoders so their authority is never
conflated.

A factual/counterfactual example begins at one shared branch boundary.  Control
and treatment receive the same agent action sequence and differ only in the
declared intervention.  Counterfactual predictions are computed from that
shared pre-state and the two control streams; recorded post-intervention
observations are targets only.

Causal supervision is a typed, lag-bucketed edge vocabulary of the form
`parent_kind -> child_kind @ lag_bucket`.  It describes event-kind relations,
not exact event-identity recovery.  Pair propagation excludes the direct
`intervention_applied` trace and requires a downstream event/state,
reward, or terminal difference.

## Consequences

- Horizon 1/4/8/16 predictions answer a well-formed conditional query with the
  controls needed to determine each future.
- Changing a post-intervention target observation cannot change a
  counterfactual prediction; changing an action or declared intervention can.
- Training Pack v1 can add the derived controls and labels without changing
  Episode Bundle v2, which remains the replay authority.
- Dataset and model tests must enforce 65-state/64-transition alignment,
  sensor-cache group masking, typed-edge vocabulary identity, and the
  privileged-input deny-list.
