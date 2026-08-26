"""Training adapters and deterministic smoke baselines for world models.

The adapter treats Episode Bundle tables as immutable source data.  It turns
them into small, typed supervision records without making any assumptions
about a particular ML framework, so NumPy, PyTorch, and JAX consumers can all
share the same split and target semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from itertools import combinations
from math import isfinite
from pathlib import Path
from typing import Iterable

import numpy as np

from .env import ACTION_KEYS
from .episode_bundle import EpisodeBundleReader


@dataclass(frozen=True)
class TransitionSample:
    """A transition window conditioned only on its pre-step input views."""

    episode_index: int
    task: str
    seed: int | None
    branch_id: int
    transition_ids: tuple[int, ...]
    horizon: int
    tick_before: int
    tick_after: int
    before_hash: int | None
    after_hash: int | None
    actions: tuple[tuple[int, ...], ...]
    reward: float
    terminated: bool
    truncated: bool
    agent_observation_before: bytes | None
    oracle_state_before: bytes | None
    agent_observation_target: bytes | None
    oracle_state_target: bytes | None
    intervention_event_ids: tuple[int, ...]

    @property
    def agent_observation(self) -> bytes | None:
        """Backward-compatible alias for the agent input at ``tick_before``."""

        return self.agent_observation_before

    @property
    def oracle_state(self) -> bytes | None:
        """Backward-compatible alias for the oracle input at ``tick_before``."""

        return self.oracle_state_before


@dataclass(frozen=True)
class EventSample:
    """One semantic event conditioned on the transition that produced it."""

    episode_index: int
    transition_id: int
    branch_id: int
    before_hash: int | None
    action: tuple[int, ...]
    event_id: int
    tick: int
    order_index: int
    phase: str
    kind: str
    actor: str | None
    target: str | None
    position: tuple[int, int, int] | None
    mechanism: str
    parent_ids: tuple[int, ...]
    root_cause: str


@dataclass(frozen=True)
class SpatialSample:
    """Metric displacement in an explicit coordinate frame and scale."""

    episode_index: int
    transition_id: int
    event_id: int
    subject: str
    frame_id: int
    scale: float
    meters_per_cell: float
    coordinate_unit: str
    before: tuple[float, float, float]
    after: tuple[float, float, float]
    displacement: tuple[float, float, float]
    relation: str


@dataclass(frozen=True)
class DeltaSample:
    """A field/cell localization target linked to its explaining event."""

    episode_index: int
    transition_id: int
    event_id: int
    subject: str
    field: str
    before: object
    after: object


@dataclass(frozen=True)
class TemporalOrderSample:
    episode_index: int
    first_event_id: int
    second_event_id: int
    first_kind: str
    second_kind: str
    tick_delta: int
    first_precedes_second: bool


@dataclass(frozen=True)
class TimeToEventSample:
    episode_index: int
    anchor_event_id: int
    target_event_id: int
    target_kind: str
    ticks_until_event: int


@dataclass(frozen=True)
class CausalParentSample:
    episode_index: int
    candidate_parent_id: int
    child_event_id: int
    parent_kind: str
    child_kind: str
    tick_delta: int
    is_parent: bool


@dataclass(frozen=True)
class FactualCounterfactualPair:
    """Two different branch outcomes sharing an identical pre-state."""

    factual: TransitionSample
    counterfactual: TransitionSample
    outcome_changed: bool
    reward_delta: float


@dataclass(frozen=True)
class WorldModelSupervision:
    single_step: tuple[TransitionSample, ...]
    multi_step: tuple[TransitionSample, ...]
    events: tuple[EventSample, ...]
    deltas: tuple[DeltaSample, ...]
    spatial: tuple[SpatialSample, ...]
    temporal_order: tuple[TemporalOrderSample, ...]
    time_to_event: tuple[TimeToEventSample, ...]
    causal_parent: tuple[CausalParentSample, ...]
    counterfactual: tuple[FactualCounterfactualPair, ...]


@dataclass(frozen=True)
class BaselineResult:
    """Machine-readable metrics and per-sample deterministic predictions."""

    report: dict[str, dict[str, int | float | str | None]]
    predictions: dict[str, tuple[object, ...]]


@dataclass(frozen=True)
class _Transition:
    episode_index: int
    task: str
    seed: int | None
    branch_id: int
    transition_id: int
    tick_before: int
    tick_after: int
    before_hash: int | None
    after_hash: int | None
    action: tuple[int, ...]
    reward: float
    terminated: bool
    truncated: bool
    agent_observation_before: bytes | None
    oracle_state_before: bytes | None
    agent_observation_after: bytes | None
    oracle_state_after: bytes | None
    intervention_event_ids: tuple[int, ...]


@dataclass(frozen=True)
class _SpatialContext:
    frame_id: int
    scale: float

    @property
    def meters_per_cell(self) -> float:
        return 1.0 / self.scale


class WorldModelAdapter:
    """Build deterministic training targets from v1 or v2 episode readers."""

    def __init__(
        self,
        sources: (
            str
            | Path
            | EpisodeBundleReader
            | Iterable[str | Path | EpisodeBundleReader]
        ),
    ) -> None:
        if isinstance(sources, (str, Path, EpisodeBundleReader)):
            sources = (sources,)
        self._readers: tuple[EpisodeBundleReader, ...] = tuple(
            source if isinstance(source, EpisodeBundleReader) else EpisodeBundleReader(source)
            for source in sources
        )
        transitions: list[_Transition] = []
        spatial_contexts: list[_SpatialContext] = []
        for episode_index, reader in enumerate(self._readers):
            reader.validate()
            spatial_contexts.append(_manifest_spatial_context(reader.manifest))
            task = str(reader.manifest.get("task", "unknown"))
            raw_seed = reader.manifest.get("seed")
            seed = None if raw_seed is None else int(raw_seed)
            rows = reader.transitions.to_pylist()
            intervention_events: dict[int, list[int]] = {}
            for event in reader.events.to_pylist():
                if (
                    event.get("phase") == "intervention"
                    and event.get("kind") == "intervention_applied"
                ):
                    intervention_events.setdefault(
                        int(event["transition_id"]), []
                    ).append(int(event["id"]))
            for row_index, row in enumerate(rows):
                tick_before = int(row.get("tick_before", row.get("tick", row_index)))
                transition_id = int(row.get("transition_id", row_index))
                transitions.append(
                    _Transition(
                        episode_index=episode_index,
                        task=task,
                        seed=seed,
                        branch_id=int(row.get("branch_id", reader.manifest.get("branch_id", 0))),
                        transition_id=transition_id,
                        tick_before=tick_before,
                        tick_after=int(row.get("tick_after", tick_before + 1)),
                        before_hash=_optional_int(row.get("before_hash")),
                        after_hash=_optional_int(row.get("after_hash", row.get("hash"))),
                        action=tuple(int(row.get(key, 0) or 0) for key in ACTION_KEYS),
                        reward=float(row.get("reward", 0.0) or 0.0),
                        terminated=bool(row.get("terminated", row.get("done", False))),
                        truncated=bool(row.get("truncated", False)),
                        agent_observation_before=row.get("agent_observation_before"),
                        oracle_state_before=row.get("oracle_state_before"),
                        agent_observation_after=row.get("agent_observation"),
                        oracle_state_after=row.get("oracle_state"),
                        intervention_event_ids=tuple(
                            intervention_events.get(transition_id, ())
                        ),
                    )
                )
        self._transitions = tuple(
            sorted(
                transitions,
                key=lambda item: (
                    item.episode_index,
                    item.branch_id,
                    item.tick_before,
                    item.transition_id,
                ),
            )
        )
        self._spatial_contexts = tuple(spatial_contexts)
        transitions_by_id = {
            (item.episode_index, item.transition_id): item for item in self._transitions
        }
        events: list[EventSample] = []
        delta_rows: list[tuple[int, dict]] = []
        for episode_index, reader in enumerate(self._readers):
            for order_index, row in enumerate(reader.events.to_pylist()):
                transition = transitions_by_id[(episode_index, int(row["transition_id"]))]
                position = _position(row)
                events.append(
                    EventSample(
                        episode_index=episode_index,
                        transition_id=transition.transition_id,
                        branch_id=transition.branch_id,
                        before_hash=transition.before_hash,
                        action=transition.action,
                        event_id=int(row["id"]),
                        tick=int(row["tick"]),
                        order_index=order_index,
                        phase=str(row["phase"]),
                        kind=str(row["kind"]),
                        actor=row.get("actor"),
                        target=row.get("target"),
                        position=position,
                        mechanism=str(row["mechanism"]),
                        parent_ids=tuple(int(value) for value in row.get("parent_ids", ())),
                        root_cause=str(row["root_cause"]),
                    )
                )
            delta_rows.extend((episode_index, row) for row in reader.deltas.to_pylist())
        self._events = tuple(events)
        self._delta_rows = tuple(delta_rows)

    def build_supervision(self, *, max_horizon: int = 3) -> WorldModelSupervision:
        if max_horizon < 1:
            raise ValueError("max_horizon must be at least one")
        single_step = self._windows(1)
        multi_step = tuple(
            sample
            for horizon in range(2, max_horizon + 1)
            for sample in self._windows(horizon)
        )
        temporal_order, time_to_event = self._temporal_samples()
        return WorldModelSupervision(
            single_step=single_step,
            multi_step=multi_step,
            events=self._events,
            deltas=self._delta_samples(),
            spatial=self._spatial_samples(),
            temporal_order=temporal_order,
            time_to_event=time_to_event,
            causal_parent=self._causal_parent_samples(),
            counterfactual=self._counterfactual_pairs(single_step),
        )

    def _windows(self, horizon: int) -> tuple[TransitionSample, ...]:
        grouped: dict[tuple[int, int], list[_Transition]] = {}
        for transition in self._transitions:
            grouped.setdefault((transition.episode_index, transition.branch_id), []).append(
                transition
            )

        samples: list[TransitionSample] = []
        for key in sorted(grouped):
            transitions = grouped[key]
            for start in range(0, len(transitions) - horizon + 1):
                window = transitions[start : start + horizon]
                if not _is_contiguous(window):
                    continue
                first, last = window[0], window[-1]
                samples.append(
                    TransitionSample(
                        episode_index=first.episode_index,
                        task=first.task,
                        seed=first.seed,
                        branch_id=first.branch_id,
                        transition_ids=tuple(item.transition_id for item in window),
                        horizon=horizon,
                        tick_before=first.tick_before,
                        tick_after=last.tick_after,
                        before_hash=first.before_hash,
                        after_hash=last.after_hash,
                        actions=tuple(item.action for item in window),
                        reward=float(sum(item.reward for item in window)),
                        terminated=any(item.terminated for item in window),
                        truncated=any(item.truncated for item in window),
                        agent_observation_before=first.agent_observation_before,
                        oracle_state_before=first.oracle_state_before,
                        agent_observation_target=last.agent_observation_after,
                        oracle_state_target=last.oracle_state_after,
                        intervention_event_ids=tuple(
                            event_id
                            for item in window
                            for event_id in item.intervention_event_ids
                        ),
                    )
                )
        return tuple(samples)

    def _spatial_samples(self) -> tuple[SpatialSample, ...]:
        samples: list[SpatialSample] = []
        for episode_index, row in self._delta_rows:
            if str(row.get("field", "")).lower() not in {
                "position",
                "pos",
                "agent.position",
            }:
                continue
            before = _vector3(row.get("before"))
            after = _vector3(row.get("after"))
            if before is None or after is None:
                continue
            context = self._spatial_contexts[episode_index]
            before = tuple(value * context.meters_per_cell for value in before)
            after = tuple(value * context.meters_per_cell for value in after)
            displacement = tuple(after[index] - before[index] for index in range(3))
            samples.append(
                SpatialSample(
                    episode_index=episode_index,
                    transition_id=int(row["transition_id"]),
                    event_id=int(row["event_id"]),
                    subject=str(row["subject"]),
                    frame_id=context.frame_id,
                    scale=context.scale,
                    meters_per_cell=context.meters_per_cell,
                    coordinate_unit="meters",
                    before=before,
                    after=after,
                    displacement=displacement,
                    relation=_spatial_relation(displacement),
                )
            )
        return tuple(samples)

    def _delta_samples(self) -> tuple[DeltaSample, ...]:
        return tuple(
            DeltaSample(
                episode_index=episode_index,
                transition_id=int(row["transition_id"]),
                event_id=int(row["event_id"]),
                subject=str(row["subject"]),
                field=str(row["field"]),
                before=_json_value(row.get("before")),
                after=_json_value(row.get("after")),
            )
            for episode_index, row in self._delta_rows
        )

    def _event_sequences(self) -> tuple[tuple[EventSample, ...], ...]:
        grouped: dict[tuple[int, int], list[EventSample]] = {}
        for event in self._events:
            grouped.setdefault((event.episode_index, event.branch_id), []).append(event)
        return tuple(
            tuple(sorted(grouped[key], key=lambda event: (event.tick, event.order_index)))
            for key in sorted(grouped)
        )

    def _temporal_samples(
        self,
    ) -> tuple[tuple[TemporalOrderSample, ...], tuple[TimeToEventSample, ...]]:
        order_samples: list[TemporalOrderSample] = []
        time_samples: list[TimeToEventSample] = []
        for events in self._event_sequences():
            for first, second in zip(events, events[1:]):
                delta = second.tick - first.tick
                order_samples.extend(
                    (
                        TemporalOrderSample(
                            episode_index=first.episode_index,
                            first_event_id=first.event_id,
                            second_event_id=second.event_id,
                            first_kind=first.kind,
                            second_kind=second.kind,
                            tick_delta=delta,
                            first_precedes_second=True,
                        ),
                        TemporalOrderSample(
                            episode_index=first.episode_index,
                            first_event_id=second.event_id,
                            second_event_id=first.event_id,
                            first_kind=second.kind,
                            second_kind=first.kind,
                            tick_delta=-delta,
                            first_precedes_second=False,
                        ),
                    )
                )
                time_samples.append(
                    TimeToEventSample(
                        episode_index=first.episode_index,
                        anchor_event_id=first.event_id,
                        target_event_id=second.event_id,
                        target_kind=second.kind,
                        ticks_until_event=delta,
                    )
                )
        return tuple(order_samples), tuple(time_samples)

    def _causal_parent_samples(self) -> tuple[CausalParentSample, ...]:
        samples: list[CausalParentSample] = []
        for events in self._event_sequences():
            by_id = {event.event_id: event for event in events}
            for index, child in enumerate(events):
                parent_ids = set(child.parent_ids)
                for parent_id in child.parent_ids:
                    parent = by_id.get(parent_id)
                    if parent is None:
                        # A branch can begin with scheduled work whose causal
                        # ancestor lives before this bundle.  Its kind/tick are
                        # unavailable, so do not fabricate a positive sample.
                        continue
                    samples.append(_causal_sample(parent, child, True))
                negative = next(
                    (
                        event
                        for event in reversed(events[:index])
                        if event.event_id not in parent_ids
                    ),
                    None,
                )
                if negative is not None:
                    samples.append(_causal_sample(negative, child, False))
        return tuple(samples)

    @staticmethod
    def _counterfactual_pairs(
        single_step: tuple[TransitionSample, ...],
    ) -> tuple[FactualCounterfactualPair, ...]:
        grouped: dict[tuple[str, int | None, int, int], list[TransitionSample]] = {}
        for sample in single_step:
            if sample.before_hash is None:
                continue
            key = (sample.task, sample.seed, sample.tick_before, sample.before_hash)
            grouped.setdefault(key, []).append(sample)

        pairs: list[FactualCounterfactualPair] = []
        for key in sorted(grouped, key=repr):
            candidates = sorted(
                grouped[key],
                key=lambda sample: (
                    sample.branch_id,
                    sample.episode_index,
                    sample.transition_ids,
                ),
            )
            for factual, counterfactual in combinations(candidates, 2):
                if factual.branch_id == counterfactual.branch_id:
                    continue
                if factual.actions != counterfactual.actions:
                    continue
                if (
                    factual.agent_observation_before
                    != counterfactual.agent_observation_before
                    or factual.oracle_state_before
                    != counterfactual.oracle_state_before
                ):
                    continue
                factual_intervened = bool(factual.intervention_event_ids)
                counterfactual_intervened = bool(
                    counterfactual.intervention_event_ids
                )
                if factual_intervened == counterfactual_intervened:
                    continue
                if factual_intervened:
                    factual, counterfactual = counterfactual, factual
                if (
                    factual.after_hash == counterfactual.after_hash
                    and factual.reward == counterfactual.reward
                    and factual.terminated == counterfactual.terminated
                    and factual.truncated == counterfactual.truncated
                ):
                    continue
                pairs.append(
                    FactualCounterfactualPair(
                        factual=factual,
                        counterfactual=counterfactual,
                        outcome_changed=(
                            factual.after_hash != counterfactual.after_hash
                            or factual.reward != counterfactual.reward
                            or factual.terminated != counterfactual.terminated
                            or factual.truncated != counterfactual.truncated
                        ),
                        reward_delta=counterfactual.reward - factual.reward,
                    )
                )
        return tuple(pairs)


def _is_contiguous(window: list[_Transition]) -> bool:
    for before, after in zip(window, window[1:]):
        if before.tick_after != after.tick_before:
            return False
        if (
            before.after_hash is not None
            and after.before_hash is not None
            and before.after_hash != after.before_hash
        ):
            return False
    return True


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


def _manifest_spatial_context(manifest: dict) -> _SpatialContext:
    metadata = manifest.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("episode manifest metadata must be a mapping")

    raw_scale = metadata.get("scale", 1.0)
    if isinstance(raw_scale, bool) or not isinstance(raw_scale, (int, float)):
        raise ValueError("episode manifest metadata scale must be a positive number")
    scale = float(raw_scale)
    if not isfinite(scale) or scale <= 0.0:
        raise ValueError("episode manifest metadata scale must be a positive number")

    raw_frame_id = metadata.get("frame_id", 0)
    if (
        isinstance(raw_frame_id, bool)
        or not isinstance(raw_frame_id, int)
        or not 0 <= raw_frame_id <= (1 << 64) - 1
    ):
        raise ValueError("episode manifest metadata frame_id must be a uint64")
    return _SpatialContext(frame_id=raw_frame_id, scale=scale)


def _position(row: dict) -> tuple[int, int, int] | None:
    values = (row.get("position_x"), row.get("position_y"), row.get("position_z"))
    if any(value is None for value in values):
        return None
    return tuple(int(value) for value in values)


def _vector3(value) -> tuple[float, float, float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, dict) and value.get("kind") == "vec3_bits":
        value = value.get("value")
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    result = tuple(float(item) for item in value)
    if not all(float("-inf") < item < float("inf") for item in result):
        return None
    return result


def _json_value(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _spatial_relation(displacement: tuple[float, float, float]) -> str:
    dx, dy, dz = displacement
    epsilon = 1e-9
    distance_squared = dx * dx + dy * dy + dz * dz
    if distance_squared <= epsilon * epsilon:
        return "same"
    if abs(dx) <= epsilon and abs(dz) <= epsilon:
        return "above" if dy > 0.0 else "below"
    if distance_squared <= 1.0 + epsilon:
        return "adjacent"
    return "displaced"


def _causal_sample(
    parent: EventSample,
    child: EventSample,
    is_parent: bool,
) -> CausalParentSample:
    return CausalParentSample(
        episode_index=child.episode_index,
        candidate_parent_id=parent.event_id,
        child_event_id=child.event_id,
        parent_kind=parent.kind,
        child_kind=child.kind,
        tick_delta=child.tick - parent.tick,
        is_parent=is_parent,
    )


def run_world_model_baseline(supervision: WorldModelSupervision) -> BaselineResult:
    """Run tiny copy/mean/majority baselines for every supervision head.

    This is intentionally a smoke baseline, not a competitive model.  Its
    contract is that every head can be collated and scored without optional ML
    dependencies, and that identical input produces byte-for-byte equivalent
    Python values.
    """

    report: dict[str, dict[str, int | float | str | None]] = {}
    predictions: dict[str, tuple[object, ...]] = {}

    for name, samples in (
        ("single_step", supervision.single_step),
        ("multi_step", supervision.multi_step),
    ):
        rewards = np.asarray([sample.reward for sample in samples], dtype=np.float64)
        reward_prediction = float(rewards.mean()) if rewards.size else 0.0
        terminal_prediction = _majority([sample.terminated for sample in samples], False)
        predictions[name] = tuple(
            {
                "after_hash": sample.before_hash,
                "agent_observation": sample.agent_observation_before,
                "oracle_state": sample.oracle_state_before,
                "reward": reward_prediction,
                "terminated": terminal_prediction,
            }
            for sample in samples
        )
        state_scores = [
            (
                sample.agent_observation_before
                == sample.agent_observation_target
                and sample.oracle_state_before == sample.oracle_state_target
            )
            for sample in samples
            if sample.agent_observation_before is not None
            and sample.agent_observation_target is not None
            and sample.oracle_state_before is not None
            and sample.oracle_state_target is not None
        ]
        state_accuracy = (
            None
            if not state_scores
            else float(np.asarray(state_scores, dtype=np.float64).mean())
        )
        report[name] = _head_report(
            len(samples), "copy_state_exact_match", state_accuracy
        )

    event_kind = _majority([sample.kind for sample in supervision.events], "none")
    predictions["event"] = tuple(event_kind for _ in supervision.events)
    report["event"] = _head_report(
        len(supervision.events),
        "kind_accuracy",
        _accuracy([sample.kind for sample in supervision.events], event_kind),
    )

    delta_field = _majority([sample.field for sample in supervision.deltas], "none")
    predictions["delta"] = tuple(delta_field for _ in supervision.deltas)
    report["delta"] = _head_report(
        len(supervision.deltas),
        "field_accuracy",
        _accuracy([sample.field for sample in supervision.deltas], delta_field),
    )

    spatial_vectors = np.asarray(
        [sample.displacement for sample in supervision.spatial], dtype=np.float64
    )
    if spatial_vectors.size:
        mean_displacement = tuple(float(value) for value in spatial_vectors.mean(axis=0))
        displacement_mse = float(
            np.square(spatial_vectors - np.asarray(mean_displacement)).mean()
        )
    else:
        mean_displacement = (0.0, 0.0, 0.0)
        displacement_mse = None
    relation = _majority([sample.relation for sample in supervision.spatial], "same")
    predictions["spatial"] = tuple(
        {"displacement": mean_displacement, "relation": relation}
        for _ in supervision.spatial
    )
    report["spatial"] = _head_report(
        len(supervision.spatial), "displacement_mse", displacement_mse
    )

    temporal_prediction = _majority(
        [sample.first_precedes_second for sample in supervision.temporal_order], False
    )
    predictions["temporal_order"] = tuple(
        temporal_prediction for _ in supervision.temporal_order
    )
    report["temporal_order"] = _head_report(
        len(supervision.temporal_order),
        "order_accuracy",
        _accuracy(
            [sample.first_precedes_second for sample in supervision.temporal_order],
            temporal_prediction,
        ),
    )

    time_means = _means_by_key(
        (
            (sample.target_kind, float(sample.ticks_until_event))
            for sample in supervision.time_to_event
        )
    )
    time_predictions = tuple(
        time_means[sample.target_kind] for sample in supervision.time_to_event
    )
    predictions["time_to_event"] = time_predictions
    time_actual = np.asarray(
        [sample.ticks_until_event for sample in supervision.time_to_event], dtype=np.float64
    )
    time_predicted = np.asarray(time_predictions, dtype=np.float64)
    time_mae = (
        None
        if not time_actual.size
        else float(np.abs(time_actual - time_predicted).mean())
    )
    report["time_to_event"] = _head_report(
        len(supervision.time_to_event), "ticks_mae", time_mae
    )

    causal_prediction = _majority(
        [sample.is_parent for sample in supervision.causal_parent], False
    )
    predictions["causal_parent"] = tuple(
        causal_prediction for _ in supervision.causal_parent
    )
    report["causal_parent"] = _head_report(
        len(supervision.causal_parent),
        "parent_accuracy",
        _accuracy(
            [sample.is_parent for sample in supervision.causal_parent],
            causal_prediction,
        ),
    )

    changed_prediction = _majority(
        [sample.outcome_changed for sample in supervision.counterfactual], False
    )
    reward_deltas = np.asarray(
        [sample.reward_delta for sample in supervision.counterfactual], dtype=np.float64
    )
    delta_prediction = float(reward_deltas.mean()) if reward_deltas.size else 0.0
    predictions["counterfactual"] = tuple(
        {"outcome_changed": changed_prediction, "reward_delta": delta_prediction}
        for _ in supervision.counterfactual
    )
    delta_mae = (
        None
        if not reward_deltas.size
        else float(np.abs(reward_deltas - delta_prediction).mean())
    )
    report["counterfactual"] = _head_report(
        len(supervision.counterfactual), "reward_delta_mae", delta_mae
    )

    return BaselineResult(report=report, predictions=predictions)


def _head_report(
    samples: int, metric: str, value: float | None
) -> dict[str, int | float | str | None]:
    return {"samples": samples, "metric": metric, "value": value}


def _majority(values: Iterable, default):
    counts: dict[object, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return default
    return min(counts, key=lambda value: (-counts[value], repr(value)))


def _accuracy(actual: list, prediction) -> float | None:
    if not actual:
        return None
    return float(np.mean(np.asarray([value == prediction for value in actual], dtype=np.float64)))


def _means_by_key(items: Iterable[tuple[str, float]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for key, value in items:
        grouped.setdefault(key, []).append(value)
    return {
        key: float(np.asarray(grouped[key], dtype=np.float64).mean())
        for key in sorted(grouped)
    }
