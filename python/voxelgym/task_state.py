"""Serializable Python-side episode state and structured reward contracts.

The Rust world snapshot remains the source of truth for physics.  The types
here capture the Python state that is required to resume the same Gym episode
and annotate scalar rewards without changing the Gymnasium API.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import io
import json
from typing import Any, Protocol, runtime_checkable

import numpy as np


TASK_STATE_VERSION = 1
ENV_SNAPSHOT_VERSION = 1
_ENV_SNAPSHOT_MAGIC = b"VXENV1\0"


@dataclass(frozen=True, slots=True)
class TaskState:
    """Versioned, JSON-native state of one task state machine."""

    task_type: str
    task_name: str
    fields: dict[str, Any]
    schema_version: int = TASK_STATE_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "task_name": self.task_name,
            "fields": deepcopy(self.fields),
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "TaskState":
        fields = state.get("fields")
        if not isinstance(fields, dict):
            raise ValueError("task state fields must be a mapping")
        return cls(
            schema_version=int(state.get("schema_version", -1)),
            task_type=str(state.get("task_type", "")),
            task_name=str(state.get("task_name", "")),
            fields=deepcopy(fields),
        )


@dataclass(frozen=True, slots=True)
class RewardOutcome:
    """Structured explanation accompanying Gymnasium's scalar reward."""

    total: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    terminated: bool = False
    termination_reason: str | None = None
    evidence_event_ids: tuple[int, ...] = ()
    evidence_labels: tuple[str, ...] = ()
    task_state_updates: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(
            isinstance(event_id, bool)
            or not isinstance(event_id, int)
            or event_id < 0
            for event_id in self.evidence_event_ids
        ):
            raise TypeError("reward evidence event IDs must be non-negative integers")
        if not all(isinstance(label, str) for label in self.evidence_labels):
            raise TypeError("reward evidence labels must be strings")
        if not all(isinstance(key, str) for key in self.task_state_updates):
            raise TypeError("reward task-state update keys must be strings")
        try:
            normalized = json.loads(
                json.dumps(self.task_state_updates, allow_nan=False)
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("reward task-state updates must be JSON-safe") from exc
        object.__setattr__(self, "task_state_updates", normalized)

    def as_info(self) -> dict[str, Any]:
        return {
            "total": float(self.total),
            "components": {key: float(value) for key, value in self.components.items()},
            "termination_reason": self.termination_reason,
            "evidence_event_ids": list(self.evidence_event_ids),
            "evidence_labels": list(self.evidence_labels),
            "task_state_updates": deepcopy(self.task_state_updates),
        }


@dataclass(frozen=True, slots=True)
class EnvSnapshot:
    """Complete Python episode checkpoint layered over a World Snapshot."""

    world_snapshot: bytes
    task_state: dict[str, Any] | None
    np_random_state: dict[str, Any]
    episode_seed: int
    terminated: bool
    truncated: bool
    last_reward: RewardOutcome
    last_frames: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
    last_scan: tuple[np.ndarray, np.ndarray, np.ndarray] | None
    render_sample_tick: int | None
    lidar_sample_tick: int | None
    previous_pose: tuple[float, ...] | None
    render_every: int
    lidar_config: dict[str, Any] | None
    spacetime: bool
    last_trace: dict[str, Any] | None = None
    native_trace_state: bytes | None = None
    native_intervention_cursor: int = 0
    intervention_cursor: int = 0

    def __post_init__(self) -> None:
        if self.intervention_cursor < 0 or self.native_intervention_cursor < 0:
            raise ValueError("intervention cursor must be non-negative")

    def to_bytes(self) -> bytes:
        """Encode the checkpoint without pickle or Python object arrays."""

        arrays: dict[str, np.ndarray] = {
            "world": np.frombuffer(self.world_snapshot, dtype=np.uint8),
        }
        native_trace_state_name = None
        if self.native_trace_state is not None:
            native_trace_state_name = "native_trace_state"
            arrays[native_trace_state_name] = np.frombuffer(
                self.native_trace_state, dtype=np.uint8
            )
        frame_names: list[str] = []
        if self.last_frames is not None:
            for index, value in enumerate(self.last_frames):
                name = f"frame_{index}"
                arrays[name] = np.asarray(value)
                frame_names.append(name)
        scan_names: list[str] = []
        if self.last_scan is not None:
            for index, value in enumerate(self.last_scan):
                name = f"scan_{index}"
                arrays[name] = np.asarray(value)
                scan_names.append(name)
        metadata = {
            "version": ENV_SNAPSHOT_VERSION,
            "task_state": _encode(self.task_state),
            "np_random_state": _encode(self.np_random_state),
            "episode_seed": int(self.episode_seed),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "last_reward": {
                "total": float(self.last_reward.total),
                "components": dict(self.last_reward.components),
                "terminated": bool(self.last_reward.terminated),
                "termination_reason": self.last_reward.termination_reason,
                "evidence_event_ids": list(
                    self.last_reward.evidence_event_ids
                ),
                "evidence_labels": list(self.last_reward.evidence_labels),
                "task_state_updates": _encode(
                    self.last_reward.task_state_updates
                ),
            },
            "frame_names": frame_names,
            "scan_names": scan_names,
            "render_sample_tick": self.render_sample_tick,
            "lidar_sample_tick": self.lidar_sample_tick,
            "previous_pose": self.previous_pose,
            "render_every": int(self.render_every),
            "lidar_config": _encode(self.lidar_config),
            "spacetime": bool(self.spacetime),
            "last_trace": _encode(self.last_trace),
            "native_trace_state_name": native_trace_state_name,
            "native_intervention_cursor": int(self.native_intervention_cursor),
            "intervention_cursor": int(self.intervention_cursor),
        }
        metadata_bytes = json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        arrays["metadata"] = np.frombuffer(metadata_bytes, dtype=np.uint8)
        output = io.BytesIO()
        np.savez_compressed(output, **arrays)
        return _ENV_SNAPSHOT_MAGIC + output.getvalue()

    @classmethod
    def from_bytes(cls, payload: bytes | bytearray | memoryview) -> "EnvSnapshot":
        """Decode a versioned checkpoint produced by :meth:`to_bytes`."""

        raw = bytes(payload)
        if not raw.startswith(_ENV_SNAPSHOT_MAGIC):
            raise ValueError("invalid EnvSnapshot magic")
        try:
            with np.load(io.BytesIO(raw[len(_ENV_SNAPSHOT_MAGIC) :]), allow_pickle=False) as archive:
                metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
                if metadata.get("version") != ENV_SNAPSHOT_VERSION:
                    raise ValueError(
                        f"unsupported EnvSnapshot version {metadata.get('version')!r}"
                    )
                reward = metadata["last_reward"]
                reward_updates = _decode(reward.get("task_state_updates", {}))
                if not isinstance(reward_updates, dict):
                    raise ValueError(
                        "EnvSnapshot reward task-state updates are not a mapping"
                    )
                frames = tuple(
                    np.array(archive[name], copy=True) for name in metadata["frame_names"]
                )
                scans = tuple(
                    np.array(archive[name], copy=True) for name in metadata["scan_names"]
                )
                task_state = _decode(metadata["task_state"])
                random_state = _decode(metadata["np_random_state"])
                last_trace = _decode(metadata.get("last_trace"))
                if task_state is not None and not isinstance(task_state, dict):
                    raise ValueError("EnvSnapshot task state is not a mapping")
                if not isinstance(random_state, dict):
                    raise ValueError("EnvSnapshot random state is not a mapping")
                if last_trace is not None and not isinstance(last_trace, dict):
                    raise ValueError("EnvSnapshot last trace is not a mapping")
                trace_state_name = metadata.get("native_trace_state_name")
                if trace_state_name is not None and not isinstance(trace_state_name, str):
                    raise ValueError("EnvSnapshot native trace state name is invalid")
                intervention_cursor = int(metadata.get("intervention_cursor", 0))
                native_intervention_cursor = int(
                    metadata.get("native_intervention_cursor", 0)
                )
                if intervention_cursor < 0 or native_intervention_cursor < 0:
                    raise ValueError("intervention cursor must be non-negative")
                return cls(
                    world_snapshot=archive["world"].tobytes(),
                    task_state=task_state,
                    np_random_state=random_state,
                    episode_seed=int(metadata["episode_seed"]),
                    terminated=bool(metadata["terminated"]),
                    truncated=bool(metadata["truncated"]),
                    last_reward=RewardOutcome(
                        total=float(reward["total"]),
                        components={
                            str(key): float(value)
                            for key, value in reward["components"].items()
                        },
                        terminated=bool(reward["terminated"]),
                        termination_reason=reward["termination_reason"],
                        evidence_event_ids=tuple(
                            int(value)
                            for value in reward.get("evidence_event_ids", ())
                        ),
                        evidence_labels=tuple(
                            str(value)
                            for value in reward.get("evidence_labels", ())
                        ),
                        task_state_updates=reward_updates,
                    ),
                    last_frames=frames or None,
                    last_scan=scans or None,
                    render_sample_tick=metadata["render_sample_tick"],
                    lidar_sample_tick=metadata["lidar_sample_tick"],
                    previous_pose=(
                        None
                        if metadata.get("previous_pose") is None
                        else tuple(float(value) for value in metadata["previous_pose"])
                    ),
                    render_every=int(metadata["render_every"]),
                    lidar_config=_decode(metadata["lidar_config"]),
                    spacetime=bool(metadata["spacetime"]),
                    last_trace=last_trace,
                    native_trace_state=(
                        None
                        if trace_state_name is None
                        else archive[trace_state_name].tobytes()
                    ),
                    native_intervention_cursor=native_intervention_cursor,
                    intervention_cursor=intervention_cursor,
                )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"invalid EnvSnapshot payload: {exc}") from exc


@runtime_checkable
class StatefulTask(Protocol):
    """Task persistence seam used by :class:`VoxelGymEnv`."""

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


def task_type_name(task: object) -> str:
    cls = type(task)
    return f"{cls.__module__}.{cls.__qualname__}"


def encode_task_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Encode arbitrary built-in task fields into a JSON-safe typed tree."""

    return _encode(fields)


def decode_task_fields(fields: dict[str, Any]) -> dict[str, Any]:
    decoded = _decode(fields)
    if not isinstance(decoded, dict):
        raise ValueError("task fields must decode to a mapping")
    return decoded


def clone_sensor_cache(cache):
    if cache is None:
        return None
    return tuple(np.array(value, copy=True) for value in cache)


def clone_random_state(state: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(state)


def _encode(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "$type": "ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "items": value.reshape(-1).tolist(),
        }
    if isinstance(value, tuple):
        return {"$type": "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, list):
        return {"$type": "list", "items": [_encode(item) for item in value]}
    if isinstance(value, set):
        items = [_encode(item) for item in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return {"$type": "set", "items": items}
    if isinstance(value, dict):
        items = [[_encode(key), _encode(item)] for key, item in value.items()]
        items.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True, separators=(",", ":")))
        return {"$type": "dict", "items": items}
    raise TypeError(f"unsupported task state value: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if not isinstance(value, dict) or "$type" not in value:
        return value
    kind = value.get("$type")
    items = value.get("items")
    if not isinstance(items, list):
        raise ValueError("typed task state items must be a list")
    if kind == "tuple":
        return tuple(_decode(item) for item in items)
    if kind == "list":
        return [_decode(item) for item in items]
    if kind == "set":
        return {_decode(item) for item in items}
    if kind == "dict":
        return {_decode(key): _decode(item) for key, item in items}
    if kind == "ndarray":
        array = np.asarray(items, dtype=value["dtype"])
        return array.reshape(tuple(int(size) for size in value["shape"]))
    raise ValueError(f"unknown task state value type: {kind!r}")
