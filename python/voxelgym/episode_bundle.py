"""Versioned episode storage for transition, event, delta, and checkpoint data.

Version 1 is the original single-Parquet recorder format and is opened read
only. Version 2 is a directory whose typed tables share stable transition and
event identifiers. Keeping the tables separate lets high-throughput consumers
skip causal detail they do not need while preserving exact supervision for
world-model training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import operator
from pathlib import Path
import time
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from .env import ACTION_KEYS
from .interventions import canonical_interventions
from .task_state import EnvSnapshot


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalize_external_parent_ids(parent_ids: Iterable[int]) -> tuple[int, ...]:
    normalized: set[int] = set()
    for raw_parent in parent_ids:
        if isinstance(raw_parent, bool):
            raise ValueError("external parent IDs must be unsigned 64-bit integers")
        try:
            parent = operator.index(raw_parent)
        except TypeError as exc:
            raise ValueError(
                "external parent IDs must be unsigned 64-bit integers"
            ) from exc
        if not 0 <= parent <= (1 << 64) - 1:
            raise ValueError("external parent IDs must be unsigned 64-bit integers")
        normalized.add(parent)
    return tuple(sorted(normalized))


def _encode_interventions(specs: Iterable[dict[str, Any]]) -> str:
    return _json(list(canonical_interventions(specs)))


def _decode_interventions(payload: Any, *, label: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(payload, str):
        raise ValueError(f"{label} interventions must be canonical JSON")
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} interventions must be canonical JSON") from exc
    if not isinstance(decoded, list):
        raise ValueError(f"{label} interventions must be a JSON list")
    try:
        normalized = canonical_interventions(decoded)
    except ValueError as exc:
        raise ValueError(f"{label} has invalid intervention specs: {exc}") from exc
    if payload != _json(list(normalized)):
        raise ValueError(f"{label} interventions are not canonically encoded")
    return normalized


@dataclass
class TransitionRecord:
    transition_id: int
    branch_id: int
    tick_before: int
    tick_after: int
    before_hash: int | None
    after_hash: int | None
    action: tuple[int, ...]
    reward: float
    swap: int = 0
    interventions: tuple[dict[str, Any], ...] = ()
    external_intervention_count: int | None = None
    reward_components: dict[str, float] = field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False
    termination_reason: str | None = None
    sensor_ticks: dict[str, int] = field(default_factory=dict)
    agent_observation_before: bytes | None = None
    oracle_state_before: bytes | None = None
    agent_observation: bytes | None = None
    oracle_state: bytes | None = None
    rgb: bytes | None = None
    depth: bytes | None = None
    seg: bytes | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    deltas: list[dict[str, Any]] = field(default_factory=list)
    checkpoint: bytes | None = None
    env_checkpoint: bytes | None = None

    def __post_init__(self):
        if len(self.action) != len(ACTION_KEYS):
            raise ValueError(f"action must have {len(ACTION_KEYS)} fields")
        if self.tick_after <= self.tick_before:
            raise ValueError("tick_after must be greater than tick_before")
        self.interventions = canonical_interventions(self.interventions)
        if not self.interventions and self.swap:
            # Compatibility for older v2 callers that only supplied the
            # inventory-management side channel.
            self.interventions = canonical_interventions(
                ({"kind": "swap_to_hotbar", "item": self.swap},)
            )
        if self.external_intervention_count is None:
            self.external_intervention_count = len(self.interventions)
        if (
            isinstance(self.external_intervention_count, bool)
            or not isinstance(self.external_intervention_count, int)
            or not 0 <= self.external_intervention_count <= len(self.interventions)
        ):
            raise ValueError(
                "external_intervention_count must index the intervention sequence"
            )


@dataclass(frozen=True, slots=True)
class EpisodeBoundary:
    """Complete environment state at one recorded episode boundary."""

    tick: int
    world_snapshot: bytes
    env_snapshot: bytes
    agent_observation: bytes
    oracle_state: bytes

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise ValueError("episode boundary tick must be non-negative")
        if not all(
            (
                self.world_snapshot,
                self.env_snapshot,
                self.agent_observation,
                self.oracle_state,
            )
        ):
            raise ValueError("episode boundary requires complete state and view payloads")


TRANSITION_SCHEMA = pa.schema(
    [
        ("transition_id", pa.uint64()),
        ("branch_id", pa.uint64()),
        ("tick_before", pa.uint64()),
        ("tick_after", pa.uint64()),
        ("before_hash", pa.uint64()),
        ("after_hash", pa.uint64()),
        *[(key, pa.uint8()) for key in ACTION_KEYS],
        ("swap", pa.uint16()),
        ("interventions", pa.string()),
        ("external_intervention_count", pa.uint32()),
        ("event_count", pa.uint32()),
        ("delta_count", pa.uint32()),
        ("reward", pa.float32()),
        ("reward_components", pa.string()),
        ("terminated", pa.bool_()),
        ("truncated", pa.bool_()),
        ("termination_reason", pa.string()),
        ("sensor_ticks", pa.string()),
        ("agent_observation_before", pa.binary()),
        ("oracle_state_before", pa.binary()),
        ("agent_observation", pa.binary()),
        ("oracle_state", pa.binary()),
        ("rgb", pa.binary()),
        ("depth", pa.binary()),
        ("seg", pa.binary()),
    ]
)

EVENT_SCHEMA = pa.schema(
    [
        ("transition_id", pa.uint64()),
        ("id", pa.uint64()),
        ("tick", pa.uint64()),
        ("phase", pa.string()),
        ("kind", pa.string()),
        ("actor", pa.string()),
        ("target", pa.string()),
        ("position_x", pa.int32()),
        ("position_y", pa.int32()),
        ("position_z", pa.int32()),
        ("mechanism", pa.string()),
        ("parent_ids", pa.list_(pa.uint64())),
        ("root_cause", pa.string()),
    ]
)

DELTA_SCHEMA = pa.schema(
    [
        ("transition_id", pa.uint64()),
        ("event_id", pa.uint64()),
        ("subject", pa.string()),
        ("field", pa.string()),
        ("before", pa.string()),
        ("after", pa.string()),
    ]
)

CHECKPOINT_SCHEMA = pa.schema(
    [
        ("boundary", pa.string()),
        ("transition_id", pa.uint64()),
        ("tick", pa.uint64()),
        ("world_snapshot", pa.binary()),
        ("env_snapshot", pa.binary()),
        ("agent_observation", pa.binary()),
        ("oracle_state", pa.binary()),
    ]
)


class EpisodeBundleWriter:
    """Accumulate one v2 episode and atomically publish its five files."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        task: str,
        seed: int,
        trace_level: str = "full",
        branch_id: int = 0,
        stem: str | None = None,
        metadata: dict[str, Any] | None = None,
        external_parent_ids: Iterable[int] = (),
    ):
        if trace_level not in {"off", "events", "full"}:
            raise ValueError("trace_level must be off, events, or full")
        self.out_dir = Path(out_dir)
        self.task = task
        self.seed = int(seed)
        self.trace_level = trace_level
        self.branch_id = int(branch_id)
        self.metadata = dict(metadata or {})
        self.external_parent_ids = _normalize_external_parent_ids(
            external_parent_ids
        )
        self.stem = stem or f"{task}_seed{seed}_{int(time.time() * 1000)}.vxbundle"
        if not self.stem.endswith(".vxbundle"):
            self.stem += ".vxbundle"
        self._records: list[TransitionRecord] = []
        self._initial_boundary: EpisodeBoundary | None = None
        self._final_boundary: EpisodeBoundary | None = None
        self._saved = False

    def set_external_parent_ids(self, parent_ids: Iterable[int]) -> None:
        """Declare lineage already present at the initial bundle boundary."""

        if self._saved or self._records:
            raise RuntimeError(
                "external parent IDs must be set before recording transitions"
            )
        self.external_parent_ids = _normalize_external_parent_ids(parent_ids)

    def log(self, record: TransitionRecord) -> None:
        if self._saved:
            raise RuntimeError("episode bundle is already saved")
        self._records.append(record)

    def set_initial_boundary(self, boundary: EpisodeBoundary) -> None:
        if self._saved:
            raise RuntimeError("episode bundle is already saved")
        if self._records:
            raise RuntimeError("initial boundary must be set before transitions")
        if self._initial_boundary is not None:
            raise RuntimeError("initial boundary is already set")
        self._initial_boundary = boundary

    def set_final_boundary(self, boundary: EpisodeBoundary) -> None:
        if self._saved:
            raise RuntimeError("episode bundle is already saved")
        if not self._records:
            raise RuntimeError("final boundary requires at least one transition")
        if self._final_boundary is not None:
            raise RuntimeError("final boundary is already set")
        self._final_boundary = boundary

    def save(self, *, final_hash: int) -> Path:
        if self._saved:
            raise RuntimeError("episode bundle is already saved")
        ids: set[int] = set()
        for record in self._records:
            if record.transition_id in ids:
                raise ValueError(f"duplicate transition_id {record.transition_id}")
            ids.add(record.transition_id)
        if self._records:
            if self._initial_boundary is None:
                raise ValueError("non-empty Episode Bundle requires an initial boundary")
            if self._final_boundary is None:
                raise ValueError("non-empty Episode Bundle requires a final boundary")

        event_ids = {
            int(event["id"])
            for record in self._records
            for event in record.events
        }
        referenced_before_bundle = {
            int(parent)
            for record in self._records
            for event in record.events
            for parent in event.get("parent_ids", ())
            if int(parent) not in event_ids
        }
        undeclared = referenced_before_bundle - set(self.external_parent_ids)
        if undeclared:
            raise ValueError(
                "event parents are neither recorded nor declared at the "
                f"initial boundary: {sorted(undeclared)}"
            )
        used_external_parent_ids = sorted(referenced_before_bundle)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        bundle = self.out_dir / self.stem
        if bundle.exists():
            raise FileExistsError(bundle)
        bundle.mkdir()

        transitions, events, deltas, checkpoints = self._tables()
        pq.write_table(transitions, bundle / "transitions.parquet", compression="zstd")
        pq.write_table(events, bundle / "events.parquet", compression="zstd")
        pq.write_table(deltas, bundle / "deltas.parquet", compression="zstd")
        pq.write_table(checkpoints, bundle / "checkpoints.parquet", compression="zstd")
        manifest = {
            "format": "voxelgym.episode",
            "format_version": 2,
            "checkpoint_schema_version": 2,
            "task": self.task,
            "seed": self.seed,
            "branch_id": self.branch_id,
            "trace_level": self.trace_level,
            "steps": len(self._records),
            "final_hash": int(final_hash),
            # A bundle may begin at an arbitrary EnvSnapshot branch point.
            # Native TraceState can therefore carry scheduler lineage whose
            # ancestor event happened before the recorded interval.  Declare
            # those boundary ancestors instead of pretending they occurred in
            # the first transition or discarding the causal link.
            "external_parent_ids": used_external_parent_ids,
            "files": {
                "transitions": "transitions.parquet",
                "events": "events.parquet",
                "deltas": "deltas.parquet",
                "checkpoints": "checkpoints.parquet",
            },
            "metadata": self.metadata,
        }
        (bundle / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
        self._saved = True
        return bundle

    def _tables(self) -> tuple[pa.Table, pa.Table, pa.Table, pa.Table]:
        transition_rows: list[dict[str, Any]] = []
        event_rows: list[dict[str, Any]] = []
        delta_rows: list[dict[str, Any]] = []
        checkpoint_rows: list[dict[str, Any]] = []
        if self._records and self._initial_boundary is not None:
            checkpoint_rows.append(
                {
                    "boundary": "initial",
                    "transition_id": self._records[0].transition_id,
                    "tick": self._initial_boundary.tick,
                    "world_snapshot": self._initial_boundary.world_snapshot,
                    "env_snapshot": self._initial_boundary.env_snapshot,
                    "agent_observation": self._initial_boundary.agent_observation,
                    "oracle_state": self._initial_boundary.oracle_state,
                }
            )
        for index, record in enumerate(self._records):
            transition_rows.append(
                {
                    "transition_id": record.transition_id,
                    "branch_id": record.branch_id,
                    "tick_before": record.tick_before,
                    "tick_after": record.tick_after,
                    "before_hash": record.before_hash,
                    "after_hash": record.after_hash,
                    **{key: int(value) for key, value in zip(ACTION_KEYS, record.action)},
                    "swap": int(record.swap),
                    "interventions": _encode_interventions(record.interventions),
                    "external_intervention_count": int(
                        record.external_intervention_count or 0
                    ),
                    "event_count": len(record.events),
                    "delta_count": len(record.deltas),
                    "reward": record.reward,
                    "reward_components": _json(record.reward_components),
                    "terminated": record.terminated,
                    "truncated": record.truncated,
                    "termination_reason": record.termination_reason,
                    "sensor_ticks": _json(record.sensor_ticks),
                    "agent_observation_before": record.agent_observation_before,
                    "oracle_state_before": record.oracle_state_before,
                    "agent_observation": record.agent_observation,
                    "oracle_state": record.oracle_state,
                    "rgb": record.rgb,
                    "depth": record.depth,
                    "seg": record.seg,
                }
            )
            for event in record.events:
                position = event.get("position", event.get("location"))
                event_rows.append(
                    {
                        "transition_id": record.transition_id,
                        "id": int(event["id"]),
                        "tick": int(event.get("tick", record.tick_before)),
                        "phase": str(event["phase"]),
                        "kind": str(event["kind"]),
                        "actor": None if event.get("actor") is None else _json(event["actor"]),
                        "target": None if event.get("target") is None else _json(event["target"]),
                        "position_x": None if position is None else int(position[0]),
                        "position_y": None if position is None else int(position[1]),
                        "position_z": None if position is None else int(position[2]),
                        "mechanism": str(event["mechanism"]),
                        "parent_ids": [int(value) for value in event.get("parent_ids", ())],
                        "root_cause": _json(event["root_cause"]),
                    }
                )
            for delta in record.deltas:
                delta_rows.append(
                    {
                        "transition_id": record.transition_id,
                        "event_id": int(delta["event_id"]),
                        "subject": _json(delta["subject"]),
                        "field": str(delta.get("field", delta.get("field_or_cell"))),
                        "before": _json(delta.get("before")),
                        "after": _json(delta.get("after")),
                    }
                )
            is_last = index == len(self._records) - 1
            if (
                not is_last
                and (record.checkpoint is not None or record.env_checkpoint is not None)
            ):
                checkpoint_rows.append(
                    {
                        "boundary": "checkpoint",
                        "transition_id": record.transition_id,
                        "tick": record.tick_after,
                        "world_snapshot": record.checkpoint,
                        "env_snapshot": record.env_checkpoint,
                        "agent_observation": record.agent_observation,
                        "oracle_state": record.oracle_state,
                    }
                )
        if self._records and self._final_boundary is not None:
            checkpoint_rows.append(
                {
                    "boundary": "final",
                    "transition_id": self._records[-1].transition_id,
                    "tick": self._final_boundary.tick,
                    "world_snapshot": self._final_boundary.world_snapshot,
                    "env_snapshot": self._final_boundary.env_snapshot,
                    "agent_observation": self._final_boundary.agent_observation,
                    "oracle_state": self._final_boundary.oracle_state,
                }
            )
        return (
            pa.Table.from_pylist(transition_rows, schema=TRANSITION_SCHEMA),
            pa.Table.from_pylist(event_rows, schema=EVENT_SCHEMA),
            pa.Table.from_pylist(delta_rows, schema=DELTA_SCHEMA),
            pa.Table.from_pylist(checkpoint_rows, schema=CHECKPOINT_SCHEMA),
        )


def detect_episode_format(path: str | Path) -> int:
    path = Path(path)
    if path.is_file() and path.suffix == ".parquet":
        return 1
    manifest_path = path / "manifest.json"
    if path.is_dir() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") == "voxelgym.episode" and manifest.get("format_version") == 2:
            return 2
    raise ValueError(f"unrecognized episode path: {path}")


class EpisodeBundleReader:
    """Read v2 bundles or adapt an original v1 Parquet shard read-only."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.format_version = detect_episode_format(self.path)
        if self.format_version == 1:
            self.transitions = pq.read_table(self.path)
            sidecar = self.path.with_suffix(".json")
            self.manifest = (
                json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
            )
            self.manifest = {"format_version": 1, **self.manifest}
            self.events = pa.Table.from_pylist([], schema=EVENT_SCHEMA)
            self.deltas = pa.Table.from_pylist([], schema=DELTA_SCHEMA)
            self.checkpoints = pa.Table.from_pylist([], schema=CHECKPOINT_SCHEMA)
            return

        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        files = self.manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("malformed Episode Bundle v2 manifest: missing files mapping")
        required = ("transitions", "events", "deltas", "checkpoints")
        if any(not isinstance(files.get(name), str) for name in required):
            raise ValueError("malformed Episode Bundle v2 manifest: incomplete files mapping")
        try:
            self.transitions = pq.read_table(self.path / files["transitions"])
            self.events = pq.read_table(self.path / files["events"])
            self.deltas = pq.read_table(self.path / files["deltas"])
            self.checkpoints = pq.read_table(self.path / files["checkpoints"])
        except Exception as exc:
            raise ValueError(f"failed to read Episode Bundle v2 tables: {exc}") from exc

    def validate(self) -> None:
        if self.format_version == 1:
            return
        if self.manifest.get("checkpoint_schema_version") != 2:
            raise ValueError("unsupported Episode Bundle checkpoint schema version")
        trace_level = self.manifest.get("trace_level")
        if trace_level not in {"off", "events", "full"}:
            raise ValueError("manifest trace_level must be off, events, or full")
        for name, table, schema in (
            ("transitions", self.transitions, TRANSITION_SCHEMA),
            ("events", self.events, EVENT_SCHEMA),
            ("deltas", self.deltas, DELTA_SCHEMA),
            ("checkpoints", self.checkpoints, CHECKPOINT_SCHEMA),
        ):
            if not table.schema.equals(schema, check_metadata=False):
                raise ValueError(
                    f"{name} schema mismatch: expected {schema}, got {table.schema}"
                )
        transition_rows = self.transitions.to_pylist()
        if trace_level == "full":
            if any(
                row["before_hash"] is None or row["after_hash"] is None
                for row in transition_rows
            ):
                raise ValueError(
                    "full trace transitions require before_hash and after_hash"
                )
        elif any(
            row["before_hash"] is not None or row["after_hash"] is not None
            for row in transition_rows
        ):
            raise ValueError(
                f"{trace_level} trace transitions must not contain boundary hashes"
            )
        if trace_level == "off" and self.events.num_rows:
            raise ValueError("off trace bundles must not contain events")
        if trace_level in {"off", "events"} and self.deltas.num_rows:
            raise ValueError(f"{trace_level} trace bundles must not contain deltas")
        manifest_steps = self.manifest.get("steps")
        if isinstance(manifest_steps, bool) or not isinstance(manifest_steps, int):
            raise ValueError("manifest steps must be an integer")
        if manifest_steps != len(transition_rows):
            raise ValueError(
                f"manifest steps {manifest_steps} does not match "
                f"{len(transition_rows)} transitions"
            )
        manifest_final_hash = self.manifest.get("final_hash")
        if isinstance(manifest_final_hash, bool) or not isinstance(manifest_final_hash, int):
            raise ValueError("manifest final_hash must be an integer")
        external_parent_ids = self.manifest.get("external_parent_ids", [])
        if (
            not isinstance(external_parent_ids, list)
            or any(
                isinstance(parent, bool) or not isinstance(parent, int) or parent < 0
                for parent in external_parent_ids
            )
            or external_parent_ids != sorted(set(external_parent_ids))
        ):
            raise ValueError(
                "manifest external_parent_ids must be sorted unique unsigned integers"
            )
        external_parent_set = set(external_parent_ids)
        transition_ids = [row["transition_id"] for row in transition_rows]
        if len(set(transition_ids)) != len(transition_ids):
            raise ValueError("duplicate transition IDs in bundle")
        transition_set = set(transition_ids)
        transition_by_id = {row["transition_id"]: row for row in transition_rows}
        interventions_by_transition: dict[int, tuple[dict[str, Any], ...]] = {}
        transition_order = {
            row["transition_id"]: index for index, row in enumerate(transition_rows)
        }
        manifest_branch = self.manifest.get("branch_id")
        if isinstance(manifest_branch, bool) or not isinstance(manifest_branch, int):
            raise ValueError("manifest branch_id must be an integer")
        previous_by_branch: dict[int, dict[str, Any]] = {}
        for transition in transition_rows:
            transition_label = f"transition {transition['transition_id']}"
            interventions = _decode_interventions(
                transition["interventions"], label=transition_label
            )
            external_count = transition["external_intervention_count"]
            if not 0 <= external_count <= len(interventions):
                raise ValueError(
                    f"{transition_label} external_intervention_count exceeds its specs"
                )
            interventions_by_transition[transition["transition_id"]] = interventions
            branch_id = transition["branch_id"]
            if branch_id != manifest_branch:
                raise ValueError(
                    f"transition {transition['transition_id']} does not match manifest branch"
                )
            if transition["tick_after"] != transition["tick_before"] + 1:
                raise ValueError(
                    f"transition {transition['transition_id']} ticks are not contiguous"
                )
            previous = previous_by_branch.get(branch_id)
            if previous is not None:
                if transition["tick_before"] != previous["tick_after"]:
                    raise ValueError(
                        f"branch {branch_id} transition ticks are not contiguous"
                    )
                if (
                    previous["after_hash"] is not None
                    and transition["before_hash"] is not None
                    and previous["after_hash"] != transition["before_hash"]
                ):
                    raise ValueError(
                        f"branch {branch_id} transition hash chain is broken"
                    )
            previous_by_branch[branch_id] = transition
            for field in (
                "agent_observation_before",
                "oracle_state_before",
                "agent_observation",
                "oracle_state",
            ):
                if transition[field] is None:
                    raise ValueError(
                        f"transition {transition['transition_id']} is missing {field}"
                    )
            before_oracle = self._decode_oracle(
                transition["oracle_state_before"],
                f"transition {transition['transition_id']} before oracle",
            )
            after_oracle = self._decode_oracle(
                transition["oracle_state"],
                f"transition {transition['transition_id']} oracle",
            )
            self._validate_oracle_boundary(
                before_oracle,
                tick=transition["tick_before"],
                # Inventory UI swaps are serialized interventions inside the
                # transition, so the stored pre-state hash remains the common
                # branch boundary before every input is applied.
                expected_hash=transition["before_hash"],
                label=f"transition {transition['transition_id']} before oracle",
            )
            self._validate_oracle_boundary(
                after_oracle,
                tick=transition["tick_after"],
                expected_hash=transition["after_hash"],
                label=f"transition {transition['transition_id']} oracle",
            )
            if previous is not None:
                if previous["agent_observation"] != transition["agent_observation_before"]:
                    raise ValueError(
                        f"transition {transition['transition_id']} agent-view boundary is broken"
                    )
                if previous["oracle_state"] != transition["oracle_state_before"]:
                    raise ValueError(
                        f"transition {transition['transition_id']} oracle-view boundary is broken"
                    )
            try:
                sensor_ticks = json.loads(transition["sensor_ticks"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"transition {transition['transition_id']} has invalid sensor_ticks"
                ) from exc
            if not isinstance(sensor_ticks, dict):
                raise ValueError(
                    f"transition {transition['transition_id']} sensor_ticks must be a mapping"
                )
            for sensor, sample_tick in sensor_ticks.items():
                if sample_tick is None:
                    continue
                if isinstance(sample_tick, bool) or not isinstance(sample_tick, int):
                    raise ValueError(
                        f"transition {transition['transition_id']} sensor tick "
                        f"for {sensor!r} must be an integer or null"
                    )
                if sample_tick > transition["tick_after"]:
                    raise ValueError(
                        f"transition {transition['transition_id']} has future sensor tick "
                        f"for {sensor!r}"
                    )

        events = self.events.to_pylist()
        event_by_id: dict[int, dict[str, Any]] = {}
        event_order: dict[int, int] = {}
        events_by_transition: dict[int, list[dict[str, Any]]] = {
            transition_id: [] for transition_id in transition_ids
        }
        for index, event in enumerate(events):
            if event["transition_id"] not in transition_set:
                raise ValueError(f"event {event['id']} references unknown transition")
            if event["id"] in event_by_id:
                raise ValueError(f"duplicate event_id {event['id']}")
            transition = transition_by_id[event["transition_id"]]
            if event["tick"] != transition["tick_before"]:
                raise ValueError(
                    f"event {event['id']} tick does not match its transition boundary"
                )
            event_by_id[event["id"]] = event
            event_order[event["id"]] = index
            events_by_transition[event["transition_id"]].append(event)

        deltas = self.deltas.to_pylist()
        deltas_by_transition: dict[int, list[dict[str, Any]]] = {
            transition_id: [] for transition_id in transition_ids
        }
        for delta in deltas:
            if delta["transition_id"] in deltas_by_transition:
                deltas_by_transition[delta["transition_id"]].append(delta)

        for transition in transition_rows:
            transition_id = transition["transition_id"]
            actual_events = events_by_transition[transition_id]
            actual_deltas = deltas_by_transition[transition_id]
            if transition["event_count"] != len(actual_events):
                raise ValueError(
                    f"transition {transition_id} event_count does not match events table"
                )
            if transition["delta_count"] != len(actual_deltas):
                raise ValueError(
                    f"transition {transition_id} delta_count does not match deltas table"
                )
            interventions = interventions_by_transition[transition_id]
            if trace_level != "off":
                intervention_events = [
                    event
                    for event in actual_events
                    if event["phase"] == "intervention"
                    and event["kind"] == "intervention_applied"
                ]
                if len(intervention_events) != len(interventions):
                    raise ValueError(
                        f"transition {transition_id} intervention specs do not match "
                        "its intervention events"
                    )
        referenced_parent_ids = {
            parent for event in events for parent in event["parent_ids"]
        }
        for event in events:
            for parent in event["parent_ids"]:
                if parent not in event_by_id and parent not in external_parent_set:
                    raise ValueError(f"unknown parent event {parent}")
        expected_external_parents = referenced_parent_ids - event_by_id.keys()
        if external_parent_set != expected_external_parents:
            raise ValueError(
                "manifest external_parent_ids does not match boundary ancestry"
            )
        self._validate_acyclic(event_by_id)
        for event in events:
            for parent in event["parent_ids"]:
                if parent in external_parent_set:
                    continue
                parent_event = event_by_id[parent]
                parent_transition = transition_by_id[parent_event["transition_id"]]
                child_transition = transition_by_id[event["transition_id"]]
                if parent_transition["branch_id"] != child_transition["branch_id"]:
                    raise ValueError(f"event {event['id']} has a parent from another branch")
                parent_key = (
                    parent_transition["tick_before"],
                    transition_order[parent_event["transition_id"]],
                    event_order[parent],
                )
                child_key = (
                    child_transition["tick_before"],
                    transition_order[event["transition_id"]],
                    event_order[event["id"]],
                )
                if parent_key >= child_key:
                    raise ValueError(f"event {event['id']} depends on a future transition")

        # Run graph diagnostics before the completeness check so a malformed
        # non-empty graph reports its causal defect.  An erased (or replaced)
        # Full trace still cannot pass: every physical transition owns one
        # deterministic action root, even when it has no state deltas.
        action_root_by_transition: dict[int, dict[str, Any]] = {}
        if trace_level == "full":
            for transition_id, transition_events in events_by_transition.items():
                action_roots = [
                    event
                    for event in transition_events
                    if event["phase"] == "agent_action"
                    and event["kind"] == "action_applied"
                    and not event["parent_ids"]
                ]
                if len(action_roots) != 1:
                    raise ValueError(
                        f"full trace transition {transition_id} requires exactly one "
                        "action root event"
                    )
                action_root_by_transition[transition_id] = action_roots[0]

        for delta in deltas:
            if delta["transition_id"] not in transition_set:
                raise ValueError(f"delta references unknown transition {delta['transition_id']}")
            if delta["event_id"] not in event_by_id:
                raise ValueError(f"delta references unknown event {delta['event_id']}")
            if event_by_id[delta["event_id"]]["transition_id"] != delta["transition_id"]:
                raise ValueError("delta transition does not match its event transition")

        if trace_level == "full":
            for transition in transition_rows:
                transition_id = transition["transition_id"]
                tick_deltas = [
                    delta
                    for delta in deltas_by_transition[transition_id]
                    if delta["field"] == "tick"
                ]
                if len(tick_deltas) != 1:
                    raise ValueError(
                        f"full trace transition {transition_id} requires exactly one "
                        "world tick delta"
                    )
                tick_delta = tick_deltas[0]
                action_root = action_root_by_transition[transition_id]
                if (
                    tick_delta["event_id"] != action_root["id"]
                    or tick_delta["subject"] != _json({"kind": "world"})
                    or tick_delta["before"] != _json(transition["tick_before"])
                    or tick_delta["after"] != _json(transition["tick_after"])
                ):
                    raise ValueError(
                        f"full trace transition {transition_id} has an invalid "
                        "world tick delta"
                    )

        checkpoint_rows = self.checkpoints.to_pylist()
        boundary_keys: set[tuple[str, int]] = set()
        initial_rows: list[dict[str, Any]] = []
        final_rows: list[dict[str, Any]] = []
        for checkpoint in checkpoint_rows:
            if checkpoint["transition_id"] not in transition_set:
                raise ValueError(
                    f"checkpoint references unknown transition {checkpoint['transition_id']}"
                )
            transition = transition_by_id[checkpoint["transition_id"]]
            boundary = checkpoint["boundary"]
            if boundary not in {"initial", "checkpoint", "final"}:
                raise ValueError(f"unknown checkpoint boundary {boundary!r}")
            key = (boundary, checkpoint["transition_id"])
            if key in boundary_keys:
                raise ValueError(
                    f"duplicate {boundary} boundary for transition {checkpoint['transition_id']}"
                )
            boundary_keys.add(key)
            expected_tick = (
                transition["tick_before"]
                if boundary == "initial"
                else transition["tick_after"]
            )
            if checkpoint["tick"] != expected_tick:
                raise ValueError(
                    f"{boundary} checkpoint tick does not match transition boundary"
                )
            if any(
                checkpoint[field] is None
                for field in (
                    "world_snapshot",
                    "env_snapshot",
                    "agent_observation",
                    "oracle_state",
                )
            ):
                raise ValueError(f"{boundary} boundary is missing a complete snapshot")
            try:
                env_snapshot = EnvSnapshot.from_bytes(checkpoint["env_snapshot"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid {boundary} EnvSnapshot: {exc}") from exc
            if env_snapshot.world_snapshot != checkpoint["world_snapshot"]:
                raise ValueError(
                    f"{boundary} EnvSnapshot world does not match world_snapshot"
                )
            expected_agent = (
                transition["agent_observation_before"]
                if boundary == "initial"
                else transition["agent_observation"]
            )
            expected_oracle = (
                transition["oracle_state_before"]
                if boundary == "initial"
                else transition["oracle_state"]
            )
            if checkpoint["agent_observation"] != expected_agent:
                raise ValueError(f"{boundary} agent view does not match transition boundary")
            if checkpoint["oracle_state"] != expected_oracle:
                raise ValueError(f"{boundary} oracle view does not match transition boundary")
            if boundary == "initial":
                initial_rows.append(checkpoint)
            elif boundary == "final":
                final_rows.append(checkpoint)

        if transition_rows:
            first = transition_rows[0]
            last = transition_rows[-1]
            if len(initial_rows) != 1:
                raise ValueError("non-empty bundle requires exactly one initial boundary")
            if len(final_rows) != 1:
                raise ValueError("non-empty bundle requires exactly one final boundary")
            if initial_rows[0]["transition_id"] != first["transition_id"]:
                raise ValueError("initial boundary does not reference the first transition")
            if final_rows[0]["transition_id"] != last["transition_id"]:
                raise ValueError("final boundary does not reference the last transition")
            if last["after_hash"] is not None and last["after_hash"] != manifest_final_hash:
                raise ValueError("manifest final_hash does not match final transition")
            final_oracle = self._decode_oracle(
                last["oracle_state"], "final transition oracle"
            )
            if final_oracle.get("world_hash") != manifest_final_hash:
                raise ValueError("manifest final_hash does not match final oracle boundary")
        elif checkpoint_rows:
            raise ValueError("empty bundle cannot contain checkpoints")

    @staticmethod
    def _decode_oracle(payload: bytes, label: str) -> dict[str, Any]:
        try:
            value = json.loads(bytes(payload).decode("utf-8"))
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} is not canonical JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a mapping")
        return value

    @staticmethod
    def transition_interventions(
        transition: dict[str, Any],
    ) -> tuple[tuple[dict[str, Any], ...], int]:
        """Decode the canonical inputs and external/task split for replay."""

        specs = _decode_interventions(
            transition.get("interventions"),
            label=f"transition {transition.get('transition_id')}",
        )
        external_count = transition.get("external_intervention_count")
        if (
            isinstance(external_count, bool)
            or not isinstance(external_count, int)
            or not 0 <= external_count <= len(specs)
        ):
            raise ValueError(
                "transition external_intervention_count exceeds its specs"
            )
        return specs, external_count

    @staticmethod
    def _validate_oracle_boundary(
        oracle: dict[str, Any],
        *,
        tick: int,
        expected_hash: int | None,
        label: str,
    ) -> None:
        clock = oracle.get("clock")
        if not isinstance(clock, dict) or clock.get("tick") != tick:
            raise ValueError(f"{label} tick does not match transition boundary")
        world_hash = oracle.get("world_hash")
        if isinstance(world_hash, bool) or not isinstance(world_hash, int):
            raise ValueError(f"{label} is missing an integer world_hash")
        if expected_hash is not None and world_hash != expected_hash:
            raise ValueError(f"{label} hash does not match transition boundary")

    @staticmethod
    def _validate_acyclic(event_by_id: dict[int, dict[str, Any]]) -> None:
        visiting: set[int] = set()
        visited: set[int] = set()

        def visit(event_id: int) -> None:
            if event_id in visited:
                return
            if event_id in visiting:
                raise ValueError(f"causal event graph contains a cycle at {event_id}")
            visiting.add(event_id)
            for parent in event_by_id[event_id]["parent_ids"]:
                # A branch bundle may begin after the root event that caused
                # pending scheduler work.  Such parents are declared exactly
                # in manifest.external_parent_ids and form boundary roots for
                # this bundle's local DAG.
                if parent in event_by_id:
                    visit(parent)
            visiting.remove(event_id)
            visited.add(event_id)

        for event_id in event_by_id:
            visit(event_id)


def iter_transition_records(paths: Iterable[str | Path]):
    """Yield version-tagged transition rows without rewriting legacy data."""

    for path in paths:
        reader = EpisodeBundleReader(path)
        for row in reader.transitions.to_pylist():
            yield reader.format_version, row
