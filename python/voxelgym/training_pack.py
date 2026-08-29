"""Derived, streamable Training Pack v1 built from Episode Bundle v2.

The pack is never authoritative simulation state. It contains Agent View
tensors and oracle-derived labels, but never snapshots, hashes, or raw Oracle
View payloads in the model input schema.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .env import ACTION_KEYS
from .episode_bundle import EpisodeBundleReader
from .recorder import decode_observation


PACK_FORMAT = "voxelgym.training-pack"
PACK_VERSION = 1
DATASET_FORMAT = "voxelgym.dataset"
DATASET_VERSION = 1
PAIR_HORIZONS = (1, 4, 8, 16)
INTERVENTION_KINDS = (
    "set_cell",
    "teleport_agent",
    "set_agent_velocity",
    "give_item",
    "swap_to_hotbar",
)
_INTERVENTION_INDEX = {kind: index for index, kind in enumerate(INTERVENTION_KINDS)}

MODEL_INPUT_FIELDS = (
    "rgb",
    "depth",
    "normals",
    "lidar_range",
    "voxels",
    "pose",
    "inventory",
    "action",
    "intervention_kind",
    "intervention_params",
)
MODEL_LABEL_FIELDS = (
    "seg",
    "reward",
    "terminal",
    "event_kinds",
    "delta_fields",
    "causal_edges",
    "time_to_event",
    "counterfactual_propagated",
    "counterfactual_reward_delta",
)
FORBIDDEN_MODEL_INPUTS = {
    "events",
    "deltas",
    "hash",
    "world_hash",
    "snapshot",
    "world_snapshot",
    "oracle",
    "oracle_view",
    "oracle_state",
    "task_state",
    "task_truth",
}


def validate_model_input_schema(fields: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(str(field) for field in fields)
    unknown = set(normalized) - set(MODEL_INPUT_FIELDS)
    forbidden = {
        field
        for field in normalized
        if field in FORBIDDEN_MODEL_INPUTS
        or any(token in field.lower() for token in ("snapshot", "hash", "oracle", "event", "delta"))
    }
    if unknown or forbidden:
        raise ValueError(
            f"model input schema contains unknown/privileged fields: {sorted(unknown | forbidden)}"
        )
    return normalized


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_sha256(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = child.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(child)))
    return digest.hexdigest()


def assign_split(task: str, seed: int, train_fraction: float, validation_fraction: float) -> str:
    # Split identity is the episode seed alone.  Keeping the task argument in
    # the public helper preserves existing call sites while ensuring the same
    # seed can never cross splits if it appears under more than one task.
    del task
    digest = hashlib.sha256(str(seed).encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "little") / 2**64
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "test"


def write_dataset_manifest(
    path: str | Path,
    *,
    config: dict[str, Any],
    sources: list[dict[str, Any]],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": DATASET_FORMAT,
        "format_version": DATASET_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "sources": sorted(sources, key=lambda item: (item["task"], item["seed"], item["path"])),
    }
    canonical = _canonical_json(payload)
    payload["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    destination.write_text(_pretty_json(payload), encoding="utf-8")
    return destination


def read_dataset_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != DATASET_FORMAT or payload.get("format_version") != DATASET_VERSION:
        raise ValueError("unsupported Dataset Manifest")
    expected = payload.get("fingerprint")
    unsigned = dict(payload)
    unsigned.pop("fingerprint", None)
    actual = hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
    if expected != actual:
        raise ValueError("Dataset Manifest fingerprint mismatch")
    return payload


def build_training_pack(
    dataset_manifest: str | Path,
    output_dir: str | Path,
    *,
    segment_steps: int = 256,
    window_steps: int = 64,
    shard_bytes: int = 1 << 30,
) -> Path:
    dataset_path = Path(dataset_manifest)
    dataset = read_dataset_manifest(dataset_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validate_model_input_schema(MODEL_INPUT_FIELDS)
    pair_targets = _derive_pair_targets(dataset_path, dataset["sources"], PAIR_HORIZONS)

    rows: list[dict[str, Any]] = []
    row_bytes = 0
    shard_index = 0
    files: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    event_vocab: set[str] = set()
    delta_vocab: set[str] = set()
    edge_vocab: set[str] = set()
    observation_schema: dict[str, dict[str, Any]] | None = None

    def flush() -> None:
        nonlocal rows, row_bytes, shard_index
        if not rows:
            return
        name = f"pack-{shard_index:05d}.parquet"
        path = output / name
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path, compression="zstd", row_group_size=1)
        first_segment = len(segments) - len(rows)
        for offset, row in enumerate(rows):
            segments[first_segment + offset].update({"file": name, "row_group": offset})
        files.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "rows": len(rows),
            }
        )
        shard_index += 1
        rows = []
        row_bytes = 0

    for source_index, source in enumerate(dataset["sources"]):
        bundle = (dataset_path.parent / source["path"]).resolve()
        if bundle_sha256(bundle) != source["sha256"]:
            raise ValueError(f"source bundle checksum mismatch: {source['path']}")
        reader = EpisodeBundleReader(bundle)
        reader.validate()
        transition_rows = reader.transitions.to_pylist()
        event_rows = reader.events.to_pylist()
        events_by_transition = _group_rows(event_rows, "transition_id")
        event_by_id = {int(event["id"]): event for event in event_rows}
        deltas_by_transition = _group_rows(reader.deltas.to_pylist(), "transition_id")
        for start in range(0, len(transition_rows), segment_steps):
            chunk = transition_rows[start : start + segment_steps]
            if len(chunk) < 2:
                continue
            packed, schema = _pack_segment(
                source_index,
                source,
                chunk,
                events_by_transition,
                deltas_by_transition,
                event_by_id,
                _boundary_sensor_ticks(transition_rows, start, chunk[0]),
                pair_targets.get(str(source.get("pair_id"))),
            )
            if observation_schema is None:
                observation_schema = schema
            elif observation_schema != schema:
                raise ValueError(
                    f"observation schema mismatch in {source['path']}: {schema} != {observation_schema}"
                )
            segment_entry = {
                "source_index": source_index,
                "length": len(chunk),
                "split": source["split"],
                "start_tick": int(chunk[0]["tick_before"]),
                "policy": str(source.get("policy", "unknown")),
                "epsilon": float(source.get("epsilon", 0.0)),
                "pair_id": source.get("pair_id"),
                "pair_role": source.get("pair_role"),
                "pair_boundary_tick": packed.get("pair_boundary_tick"),
            }
            segments.append(segment_entry)
            rows.append(packed)
            row_bytes += _estimate_row_bytes(packed)
            for values in json.loads(packed["event_kinds"]):
                event_vocab.update(values)
            for values in json.loads(packed["delta_fields"]):
                delta_vocab.update(values)
            for values in json.loads(packed["causal_edges"]):
                edge_vocab.update(values)
            if row_bytes >= shard_bytes:
                flush()
    flush()
    if observation_schema is None:
        raise ValueError("Dataset Manifest contains no packable transitions")

    evaluation_suite = _build_evaluation_suite(
        dataset["sources"], segments, window_steps=int(window_steps)
    )
    manifest = {
        "format": PACK_FORMAT,
        "format_version": PACK_VERSION,
        # Derivation metadata is stable for a fixed Dataset Manifest so the
        # entire Training Pack fingerprint can be reproduced byte-for-byte.
        "created_at": dataset["created_at"],
        "dataset_manifest": Path(
            os.path.relpath(dataset_path.resolve(), output.resolve())
        ).as_posix(),
        "dataset_fingerprint": dataset["fingerprint"],
        "segment_steps": int(segment_steps),
        "window_steps": int(window_steps),
        "state_steps": int(window_steps) + 1,
        "transition_steps": int(window_steps),
        "input_fields": list(MODEL_INPUT_FIELDS),
        "label_fields": list(MODEL_LABEL_FIELDS),
        "intervention_kinds": list(INTERVENTION_KINDS),
        "intervention_parameter_width": 4,
        "pair_horizons": list(PAIR_HORIZONS),
        "observation_schema": observation_schema,
        "event_vocab": sorted(event_vocab),
        "delta_vocab": sorted(delta_vocab),
        "edge_vocab": sorted(edge_vocab),
        "pair_quality": _pair_quality(dataset["sources"], pair_targets),
        "evaluation_suite": evaluation_suite,
        "files": files,
        "segments": segments,
    }
    canonical = _canonical_json(manifest)
    manifest["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest_path = output / "manifest.json"
    manifest_path.write_text(_pretty_json(manifest), encoding="utf-8")
    return manifest_path


def _pack_segment(
    source_index: int,
    source: dict[str, Any],
    rows: list[dict[str, Any]],
    events_by_transition: dict[int, list[dict[str, Any]]],
    deltas_by_transition: dict[int, list[dict[str, Any]]],
    event_by_id: dict[int, dict[str, Any]],
    boundary_sensor_ticks: dict[str, int | None],
    pair_targets: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    # A transition consumes one boundary and produces the next.  Keeping the
    # initial `before` payload makes T controls align with exactly T+1 states.
    observations = [decode_observation(bytes(rows[0]["agent_observation_before"]))]
    observations.extend(
        decode_observation(bytes(row["agent_observation"])) for row in rows
    )
    schema = {
        key: {"shape": list(value.shape), "dtype": value.dtype.str}
        for key, value in sorted(observations[0].items())
        if key in set(MODEL_INPUT_FIELDS) | {"seg", "lidar_intensity", "lidar_seg", "raycast"}
    }
    for observation in observations[1:]:
        current = {
            key: {"shape": list(value.shape), "dtype": value.dtype.str}
            for key, value in sorted(observation.items())
            if key in schema
        }
        if current != schema:
            raise ValueError("observation tensors change shape or dtype within a segment")

    sensor_ticks = [boundary_sensor_ticks]
    sensor_ticks.extend(json.loads(row["sensor_ticks"]) for row in rows)
    render_index, render_sample_id, render_frames = _deduplicate_sensor(
        observations,
        sensor_ticks,
        "render",
        ("rgb", "depth", "normals", "seg"),
    )
    lidar_index, lidar_sample_id, lidar_frames = _deduplicate_sensor(
        observations,
        sensor_ticks,
        "lidar",
        ("lidar_range", "lidar_intensity", "lidar_seg"),
    )
    actions = np.asarray([[row[key] for key in ACTION_KEYS] for row in rows], dtype=np.uint8)
    intervention_kind, intervention_params = _encode_external_interventions(rows)
    event_kinds: list[list[str]] = []
    delta_fields: list[list[str]] = []
    causal_edges: list[list[str]] = []
    causal_parent = np.zeros(len(rows), dtype=np.uint8)  # compatibility alias
    has_non_action_event = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        transition_id = int(row["transition_id"])
        events = events_by_transition.get(transition_id, [])
        deltas = deltas_by_transition.get(transition_id, [])
        kinds = sorted({str(event["kind"]) for event in events})
        fields = sorted({str(delta["field"]) for delta in deltas})
        edges = _typed_edges(events, event_by_id)
        event_kinds.append(kinds)
        delta_fields.append(fields)
        causal_edges.append(edges)
        causal_parent[index] = bool(edges)
        has_non_action_event[index] = any(kind != "action" for kind in kinds)
    time_to_event = _time_to_next_event(has_non_action_event)

    packed: dict[str, Any] = {
        "source_index": int(source_index),
        "task": str(source["task"]),
        "seed": int(source["seed"]),
        "split": str(source["split"]),
        "policy": str(source.get("policy", "unknown")),
        "epsilon": float(source.get("epsilon", 0.0)),
        "pair_id": source.get("pair_id"),
        "pair_role": source.get("pair_role"),
        "branch_id": int(source.get("branch_id", 0)),
        "start_tick": int(rows[0]["tick_before"]),
        "length": len(rows),
        "actions": actions.tobytes(),
        "intervention_kind": intervention_kind.tobytes(),
        "intervention_params": intervention_params.tobytes(),
        "rewards": np.asarray([row["reward"] for row in rows], dtype=np.float32).tobytes(),
        "terminated": np.asarray([row["terminated"] for row in rows], dtype=np.uint8).tobytes(),
        "truncated": np.asarray([row["truncated"] for row in rows], dtype=np.uint8).tobytes(),
        "voxels": _stack_bytes(observations, "voxels"),
        "inventory": _stack_bytes(observations, "inventory"),
        "pose": _stack_bytes(observations, "pose"),
        "raycast": _stack_bytes(observations, "raycast"),
        "render_index": render_index.tobytes(),
        "lidar_index": lidar_index.tobytes(),
        "render_sample_id": render_sample_id.tobytes(),
        "lidar_sample_id": lidar_sample_id.tobytes(),
        "event_kinds": _canonical_json(event_kinds),
        "delta_fields": _canonical_json(delta_fields),
        "causal_edges": _canonical_json(causal_edges),
        "causal_parent": causal_parent.tobytes(),
        "time_to_event": time_to_event.tobytes(),
        "pair_boundary_tick": None if pair_targets is None else int(pair_targets["boundary_tick"]),
        "counterfactual_valid": (
            np.zeros(len(PAIR_HORIZONS), dtype=np.uint8).tobytes()
            if pair_targets is None
            else np.asarray(pair_targets["valid"], dtype=np.uint8).tobytes()
        ),
        "counterfactual_propagated": (
            np.zeros(len(PAIR_HORIZONS), dtype=np.uint8).tobytes()
            if pair_targets is None
            else np.asarray(pair_targets["propagated"], dtype=np.uint8).tobytes()
        ),
        "counterfactual_reward_delta": (
            np.zeros(len(PAIR_HORIZONS), dtype=np.float32).tobytes()
            if pair_targets is None
            else np.asarray(pair_targets["reward_delta"], dtype=np.float32).tobytes()
        ),
    }
    for key, value in render_frames.items():
        packed[f"{key}_frames"] = value
    for key, value in lidar_frames.items():
        packed[f"{key}_frames"] = value
    return packed, schema


def _boundary_sensor_ticks(
    transitions: list[dict[str, Any]], start: int, first: dict[str, Any]
) -> dict[str, int | None]:
    if start > 0:
        return dict(json.loads(transitions[start - 1]["sensor_ticks"]))
    observation = decode_observation(bytes(first["agent_observation_before"]))
    tick = int(first["tick_before"])
    return {
        "render": tick if "rgb" in observation else None,
        "lidar": tick if "lidar_range" in observation else None,
    }


def _encode_external_interventions(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    kinds = np.zeros((len(rows), len(INTERVENTION_KINDS)), dtype=np.float32)
    params = np.zeros((len(rows), len(INTERVENTION_KINDS), 4), dtype=np.float32)
    for step, row in enumerate(rows):
        specs, external_count = EpisodeBundleReader.transition_interventions(row)
        for spec in specs[:external_count]:
            kind = str(spec["kind"])
            slot = _INTERVENTION_INDEX[kind]
            kinds[step, slot] += 1.0
            if kind == "set_cell":
                value = (*spec["at"], spec["cell"])
            elif kind == "teleport_agent":
                value = (*spec["position"], 0.0)
            elif kind == "set_agent_velocity":
                value = (*spec["velocity"], 0.0)
            elif kind == "give_item":
                value = (spec["item"], spec["count"], 0.0, 0.0)
            else:
                value = (spec["item"], 0.0, 0.0, 0.0)
            params[step, slot] += np.asarray(value, dtype=np.float32)
    counts = kinds[..., None]
    np.divide(params, counts, out=params, where=counts > 0)
    return kinds, params


def _typed_edges(
    events: list[dict[str, Any]], event_by_id: dict[int, dict[str, Any]]
) -> list[str]:
    labels: set[str] = set()
    for child in events:
        child_kind = str(child["kind"])
        child_tick = int(child["tick"])
        for raw_parent in child.get("parent_ids", ()):
            parent = event_by_id.get(int(raw_parent))
            if parent is None:
                labels.add(f"external->{child_kind}@boundary")
                continue
            lag = max(0, child_tick - int(parent["tick"]))
            labels.add(
                f"{parent['kind']}->{child_kind}@{_lag_bucket(lag)}"
            )
    return sorted(labels)


def _lag_bucket(lag: int) -> str:
    if lag == 0:
        return "lag0"
    if lag == 1:
        return "lag1"
    if lag <= 3:
        return "lag2-3"
    if lag <= 7:
        return "lag4-7"
    if lag <= 15:
        return "lag8-15"
    return "lag16+"


def _derive_pair_targets(
    dataset_path: Path,
    sources: list[dict[str, Any]],
    horizons: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for source in sources:
        pair_id = source.get("pair_id")
        role = source.get("pair_role")
        if pair_id is not None and role is not None:
            grouped.setdefault(str(pair_id), {})[str(role)] = source

    targets: dict[str, dict[str, Any]] = {}
    for pair_id, roles in grouped.items():
        if set(roles) != {"control", "treatment"}:
            raise ValueError(f"pair {pair_id!r} must contain control and treatment sources")
        readers = {
            role: EpisodeBundleReader((dataset_path.parent / source["path"]).resolve())
            for role, source in roles.items()
        }
        for reader in readers.values():
            reader.validate()
        control = readers["control"].transitions.to_pylist()
        treatment = readers["treatment"].transitions.to_pylist()
        if not control or not treatment:
            raise ValueError(f"pair {pair_id!r} has an empty branch")
        if int(control[0]["tick_before"]) != int(treatment[0]["tick_before"]):
            raise ValueError(f"pair {pair_id!r} does not share a branch boundary")
        common_steps = min(len(control), len(treatment))
        for step in range(common_steps):
            control_action = tuple(int(control[step][key]) for key in ACTION_KEYS)
            treatment_action = tuple(int(treatment[step][key]) for key in ACTION_KEYS)
            if control_action != treatment_action:
                raise ValueError(f"pair {pair_id!r} has mismatched actions at step {step}")

        propagated_by_step = _pair_propagation_steps(
            readers["control"], readers["treatment"], control, treatment
        )
        reward_differences = np.asarray(
            [
                float(treatment[index]["reward"]) - float(control[index]["reward"])
                for index in range(common_steps)
            ],
            dtype=np.float64,
        )
        cumulative_reward = np.cumsum(reward_differences)
        valid: list[bool] = []
        propagated: list[bool] = []
        reward_delta: list[float] = []
        for horizon in horizons:
            available = horizon <= common_steps
            valid.append(available)
            propagated.append(
                bool(available and np.any(propagated_by_step[:horizon]))
            )
            reward_delta.append(
                float(cumulative_reward[horizon - 1]) if available else 0.0
            )
        targets[pair_id] = {
            "boundary_tick": int(control[0]["tick_before"]),
            "valid": valid,
            "propagated": propagated,
            "reward_delta": reward_delta,
        }
    return targets


def _pair_propagation_steps(
    control_reader: EpisodeBundleReader,
    treatment_reader: EpisodeBundleReader,
    control: list[dict[str, Any]],
    treatment: list[dict[str, Any]],
) -> np.ndarray:
    common_steps = min(len(control), len(treatment))
    propagated = np.zeros(common_steps, dtype=bool)
    control_specs, control_external = EpisodeBundleReader.transition_interventions(control[0])
    treatment_specs, treatment_external = EpisodeBundleReader.transition_interventions(
        treatment[0]
    )
    del control_specs, treatment_specs
    added_external = max(0, treatment_external - control_external)
    event_rows = treatment_reader.events.to_pylist()
    by_transition = _group_rows(event_rows, "transition_id")
    first_events = [
        event
        for event in by_transition.get(int(treatment[0]["transition_id"]), ())
        if event["phase"] == "intervention" and event["kind"] == "intervention_applied"
    ]
    direct = {
        int(event["id"])
        for event in first_events[
            control_external : control_external + added_external
        ]
    }
    event_by_id = {int(event["id"]): event for event in event_rows}

    memo: dict[int, bool] = {}

    def descends(event_id: int) -> bool:
        if event_id in direct:
            return True
        if event_id in memo:
            return memo[event_id]
        event = event_by_id[event_id]
        memo[event_id] = any(
            int(parent) in direct
            or (int(parent) in event_by_id and descends(int(parent)))
            for parent in event.get("parent_ids", ())
        )
        return memo[event_id]

    transition_index = {
        int(row["transition_id"]): index
        for index, row in enumerate(treatment[:common_steps])
    }
    for event in event_rows:
        event_id = int(event["id"])
        index = transition_index.get(int(event["transition_id"]))
        if index is not None and event_id not in direct and direct and descends(event_id):
            propagated[index] = True

    control_deltas = _group_rows(control_reader.deltas.to_pylist(), "transition_id")
    treatment_deltas = _group_rows(treatment_reader.deltas.to_pylist(), "transition_id")

    def signature(delta: dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(delta["subject"]),
            str(delta["field"]),
            str(delta["before"]),
            str(delta["after"]),
        )

    for index in range(common_steps):
        control_signature = {
            signature(delta)
            for delta in control_deltas.get(int(control[index]["transition_id"]), ())
        }
        treatment_signature = {
            signature(delta)
            for delta in treatment_deltas.get(int(treatment[index]["transition_id"]), ())
            if int(delta["event_id"]) not in direct
        }
        if control_signature != treatment_signature:
            propagated[index] = True

    for index in range(common_steps):
        if (
            float(control[index]["reward"]) != float(treatment[index]["reward"])
            or bool(control[index]["terminated"]) != bool(treatment[index]["terminated"])
            or bool(control[index]["truncated"]) != bool(treatment[index]["truncated"])
        ):
            propagated[index] = True
    return propagated


def _build_evaluation_suite(
    sources: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    *,
    window_steps: int,
) -> dict[str, Any]:
    by_source: dict[int, list[dict[str, Any]]] = {}
    for segment in segments:
        if segment["split"] == "test" and int(segment["length"]) >= window_steps:
            by_source.setdefault(int(segment["source_index"]), []).append(segment)

    units: dict[str, list[int]] = {}
    for source_index in by_source:
        source = sources[source_index]
        unit = (
            f"pair:{source['pair_id']}"
            if source.get("pair_id") is not None
            else f"source:{source_index}"
        )
        units.setdefault(unit, []).append(source_index)

    candidates: list[dict[str, Any]] = []
    for unit, source_indices in sorted(units.items()):
        common_ticks: set[int] | None = None
        segments_by_source: dict[int, dict[int, dict[str, Any]]] = {}
        for source_index in source_indices:
            tick_map = {
                int(segment["start_tick"]): segment
                for segment in by_source[source_index]
            }
            segments_by_source[source_index] = tick_map
            common_ticks = set(tick_map) if common_ticks is None else common_ticks & set(tick_map)
        if not common_ticks:
            continue
        digest = hashlib.sha256(unit.encode("utf-8")).digest()
        selected_segments: dict[int, dict[str, Any]] = {}
        offsets: dict[int, int] = {}
        if unit.startswith("pair:"):
            declared_boundaries = {
                int(sources[index]["pair_boundary_tick"])
                for index in source_indices
                if sources[index].get("pair_boundary_tick") is not None
            }
            if len(declared_boundaries) > 1:
                raise ValueError(f"{unit} branches declare different pair boundaries")
            boundary_tick = (
                next(iter(declared_boundaries))
                if declared_boundaries
                else min(common_ticks)
            )
            for source_index in source_indices:
                eligible = [
                    segment
                    for segment in by_source[source_index]
                    if int(segment["start_tick"]) <= boundary_tick
                    and boundary_tick - int(segment["start_tick"])
                    <= int(segment["length"]) - window_steps
                ]
                if not eligible:
                    selected_segments = {}
                    break
                segment = min(eligible, key=lambda item: int(item["start_tick"]))
                selected_segments[source_index] = segment
                offsets[source_index] = boundary_tick - int(segment["start_tick"])
            if not selected_segments:
                continue
        else:
            ticks = sorted(common_ticks)
            tick = ticks[int.from_bytes(digest[:8], "little") % len(ticks)]
            max_start = min(
                int(segments_by_source[index][tick]["length"]) - window_steps
                for index in source_indices
            )
            offset = int.from_bytes(digest[8:16], "little") % (max_start + 1)
            for source_index in source_indices:
                selected_segments[source_index] = segments_by_source[source_index][tick]
                offsets[source_index] = offset
        source = sources[source_indices[0]]
        candidates.append(
            {
                "unit": unit,
                "task": str(source["task"]),
                "domain": _source_domain_label(source),
                "rank": hashlib.sha256((unit + ":eval").encode("utf-8")).hexdigest(),
                "source_indices": sorted(source_indices),
                "segments": selected_segments,
                "offsets": offsets,
            }
        )

    selected: list[dict[str, Any]] = []
    counts: dict[tuple[str, str], int] = {}
    for candidate in sorted(candidates, key=lambda item: item["rank"]):
        key = (candidate["task"], candidate["domain"])
        entry_count = len(candidate["source_indices"])
        if counts.get(key, 0) + entry_count > 64:
            continue
        counts[key] = counts.get(key, 0) + entry_count
        for source_index in candidate["source_indices"]:
            segment = candidate["segments"][source_index]
            source = sources[source_index]
            selected.append(
                {
                    "source_index": source_index,
                    "file": str(segment["file"]),
                    "row_group": int(segment["row_group"]),
                    "start": int(candidate["offsets"][source_index]),
                    "task": candidate["task"],
                    "domain": candidate["domain"],
                    "pair_id": source.get("pair_id"),
                    "pair_role": source.get("pair_role"),
                }
            )
    suite: dict[str, Any] = {
        "version": 1,
        "split": "test",
        "max_per_task_domain": 64,
        "entries": selected,
    }
    suite["fingerprint"] = hashlib.sha256(_canonical_json(suite).encode("utf-8")).hexdigest()
    return suite


def _pair_quality(
    sources: list[dict[str, Any]], pair_targets: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    intervention_by_pair: dict[str, str] = {}
    for source in sources:
        if source.get("pair_id") and source.get("pair_intervention_kind"):
            intervention_by_pair[str(source["pair_id"])] = str(
                source["pair_intervention_kind"]
            )
    kind_counts: dict[str, int] = {}
    for kind in intervention_by_pair.values():
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    pair_count = len(pair_targets)
    kind_fractions = {
        kind: count / max(pair_count, 1) for kind, count in sorted(kind_counts.items())
    }
    horizon_counts: dict[str, dict[str, int]] = {}
    propagated_total = 0
    valid_total = 0
    for horizon_index, horizon in enumerate(PAIR_HORIZONS):
        valid = [
            bool(target["valid"][horizon_index]) for target in pair_targets.values()
        ]
        propagated = [
            bool(target["propagated"][horizon_index])
            for target in pair_targets.values()
            if target["valid"][horizon_index]
        ]
        horizon_counts[str(horizon)] = {
            "valid": sum(valid),
            "propagated": sum(propagated),
            "not_propagated": len(propagated) - sum(propagated),
        }
        propagated_total += sum(propagated)
        valid_total += len(propagated)
    propagated_fraction = propagated_total / max(valid_total, 1)
    not_propagated_fraction = (valid_total - propagated_total) / max(valid_total, 1)
    intervention_gate = sum(value >= 0.10 for value in kind_fractions.values()) >= 4
    propagation_gate = (
        valid_total > 0
        and propagated_fraction >= 0.20
        and not_propagated_fraction >= 0.20
    )
    return {
        "pairs": pair_count,
        "intervention_kind_counts": dict(sorted(kind_counts.items())),
        "intervention_kind_fractions": kind_fractions,
        "horizons": horizon_counts,
        "propagated_fraction": propagated_fraction,
        "not_propagated_fraction": not_propagated_fraction,
        "gates": {
            "four_intervention_kinds_ge_0_10": intervention_gate,
            "propagation_classes_ge_0_20": propagation_gate,
            "pilot_distribution_ready": intervention_gate and propagation_gate,
        },
    }


def _source_domain_label(source: dict[str, Any]) -> str:
    if source.get("domain"):
        return str(source["domain"])
    labels: list[str] = []
    scale = float(source.get("scale", 1.0))
    if scale != 1.0:
        labels.append(f"scale={scale:g}")
    clock = source.get("clock") or {}
    numerator = int(clock.get("dt_numerator", 1))
    denominator = int(clock.get("dt_denominator", 20))
    if (numerator, denominator) != (1, 20):
        labels.append(f"clock={denominator / numerator:g}hz")
    physics = source.get("physics_config") or {}
    if "gravity" in physics:
        labels.append("gravity")
    if "water_period" in physics or "lava_period" in physics:
        labels.append("fluid-period")
    return "+".join(labels) or "in-domain"


def _deduplicate_sensor(
    observations: list[dict[str, np.ndarray]],
    ticks: list[dict[str, Any]],
    sensor: str,
    fields: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, list[bytes] | None]]:
    if not all(field in observations[0] for field in fields):
        return (
            np.full(len(observations), -1, dtype=np.int16),
            np.full(len(observations), -1, dtype=np.int64),
            {field: None for field in fields},
        )
    slot_by_tick: dict[int, int] = {}
    indices = np.empty(len(observations), dtype=np.int16)
    sample_ids = np.full(len(observations), -1, dtype=np.int64)
    frames: dict[str, list[bytes]] = {field: [] for field in fields}
    for step, (observation, sample_ticks) in enumerate(zip(observations, ticks, strict=True)):
        sample_tick = sample_ticks.get(sensor)
        if sample_tick is None:
            indices[step] = -1
            continue
        sample_tick = int(sample_tick)
        sample_ids[step] = sample_tick
        slot = slot_by_tick.get(sample_tick)
        if slot is None:
            slot = len(slot_by_tick)
            if slot > np.iinfo(np.int16).max:
                raise ValueError("too many unique sensor samples in one segment")
            slot_by_tick[sample_tick] = slot
            for field in fields:
                frames[field].append(np.ascontiguousarray(observation[field]).tobytes())
        indices[step] = slot
    return indices, sample_ids, frames


def _time_to_next_event(mask: np.ndarray) -> np.ndarray:
    result = np.full(len(mask), len(mask) + 1, dtype=np.float32)
    next_index: int | None = None
    for index in range(len(mask) - 1, -1, -1):
        if mask[index]:
            next_index = index
        if next_index is not None:
            result[index] = float(next_index - index)
    return result


def _stack_bytes(observations: list[dict[str, np.ndarray]], key: str) -> bytes:
    return np.ascontiguousarray(np.stack([observation[key] for observation in observations])).tobytes()


def _group_rows(rows: list[dict[str, Any]], key: str) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row[key]), []).append(row)
    return grouped


def _estimate_row_bytes(row: dict[str, Any]) -> int:
    size = 0
    for value in row.values():
        if isinstance(value, (bytes, bytearray, str)):
            size += len(value)
        elif isinstance(value, list):
            size += sum(len(item) if isinstance(item, (bytes, bytearray, str)) else 16 for item in value)
        else:
            size += 16
    return size


@dataclass(frozen=True, slots=True)
class SegmentReference:
    file: str
    row_group: int
    length: int
    split: str
    source_index: int
    start_tick: int


class TrainingPackDataset:
    """Map-style, memory-mapped windows over Training Pack v1."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str = "train",
        context: int | None = None,
        cache_files: int = 8,
    ):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != PACK_FORMAT or self.manifest.get("format_version") != PACK_VERSION:
            raise ValueError("unsupported Training Pack")
        validate_model_input_schema(self.manifest.get("input_fields", ()))
        expected_fingerprint = self.manifest.get("fingerprint")
        unsigned = dict(self.manifest)
        unsigned.pop("fingerprint", None)
        actual_fingerprint = hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
        if expected_fingerprint != actual_fingerprint:
            raise ValueError("Training Pack fingerprint mismatch")
        dataset_manifest_path = Path(self.manifest["dataset_manifest"])
        if not dataset_manifest_path.is_absolute():
            dataset_manifest_path = (self.root / dataset_manifest_path).resolve()
        self.dataset_manifest = read_dataset_manifest(dataset_manifest_path)
        self.dataset_sources = tuple(self.dataset_manifest["sources"])
        self.context = int(context or self.manifest["window_steps"])
        self.schema = dict(self.manifest["observation_schema"])
        self.event_vocab = tuple(self.manifest.get("event_vocab", ()))
        self.delta_vocab = tuple(self.manifest.get("delta_vocab", ()))
        self.edge_vocab = tuple(self.manifest.get("edge_vocab", ()))
        self.pair_horizons = tuple(int(value) for value in self.manifest.get("pair_horizons", PAIR_HORIZONS))
        self._event_index = {value: index for index, value in enumerate(self.event_vocab)}
        self._delta_index = {value: index for index, value in enumerate(self.delta_vocab)}
        self._edge_index = {value: index for index, value in enumerate(self.edge_vocab)}
        self.references = [
            SegmentReference(
                file=str(item["file"]),
                row_group=int(item["row_group"]),
                length=int(item["length"]),
                split=str(item["split"]),
                source_index=int(item["source_index"]),
                start_tick=int(item["start_tick"]),
            )
            for item in self.manifest["segments"]
            if item["split"] == split and int(item["length"]) >= self.context
        ]
        if not self.references:
            raise ValueError(f"Training Pack has no {split!r} windows")
        self._window_counts = np.asarray(
            [reference.length - self.context + 1 for reference in self.references], dtype=np.int64
        )
        self._cumulative = np.cumsum(np.concatenate(([0], self._window_counts)))
        self._ranges_by_source: dict[int, list[tuple[int, int, int]]] = {}
        for reference_index, reference in enumerate(self.references):
            self._ranges_by_source.setdefault(reference.source_index, []).append(
                (
                    int(self._cumulative[reference_index]),
                    int(self._window_counts[reference_index]),
                    reference.start_tick,
                )
            )
        self._cache_limit = max(1, int(cache_files))
        self._parquet_cache: OrderedDict[str, pq.ParquetFile] = OrderedDict()
        self._row_cache: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        segment_index = int(np.searchsorted(self._cumulative, index, side="right") - 1)
        start = int(index - self._cumulative[segment_index])
        reference = self.references[segment_index]
        row = self._read_row(reference)
        return self._decode_window(row, start)

    def _read_row(self, reference: SegmentReference) -> dict[str, Any]:
        row_key = (reference.file, reference.row_group)
        cached_row = self._row_cache.get(row_key)
        if cached_row is not None:
            self._row_cache.move_to_end(row_key)
            return cached_row
        parquet = self._parquet_cache.get(reference.file)
        if parquet is None:
            parquet = pq.ParquetFile(self.root / reference.file, memory_map=True)
            self._parquet_cache[reference.file] = parquet
            while len(self._parquet_cache) > self._cache_limit:
                self._parquet_cache.popitem(last=False)
        else:
            self._parquet_cache.move_to_end(reference.file)
        rows = parquet.read_row_group(reference.row_group).to_pylist()
        if len(rows) != 1:
            raise ValueError("Training Pack row groups must contain exactly one segment")
        row = rows[0]
        self._row_cache[row_key] = row
        while len(self._row_cache) > self._cache_limit:
            self._row_cache.popitem(last=False)
        return row

    def _decode_window(self, row: dict[str, Any], start: int) -> dict[str, Any]:
        import torch

        length = int(row["length"])
        transition_stop = start + self.context
        state_stop = transition_stop + 1
        output: dict[str, Any] = {}
        for field in ("voxels", "inventory", "pose", "raycast"):
            spec = self.schema[field]
            array = np.frombuffer(row[field], dtype=np.dtype(spec["dtype"])).reshape(
                length + 1, *spec["shape"]
            )[start:state_stop]
            output[field] = torch.from_numpy(array.copy())
        actions = np.frombuffer(row["actions"], dtype=np.uint8).reshape(length, len(ACTION_KEYS))
        output["action"] = torch.from_numpy(actions[start:transition_stop].copy())
        intervention_kind = np.frombuffer(row["intervention_kind"], dtype=np.float32).reshape(
            length, len(INTERVENTION_KINDS)
        )[start:transition_stop]
        intervention_params = np.frombuffer(row["intervention_params"], dtype=np.float32).reshape(
            length, len(INTERVENTION_KINDS), 4
        )[start:transition_stop]
        output["intervention_kind"] = torch.from_numpy(intervention_kind.copy())
        output["intervention_params"] = torch.from_numpy(intervention_params.copy())
        self._decode_sensor_window(output, row, "render", start, state_stop)
        self._decode_sensor_window(output, row, "lidar", start, state_stop)

        rewards = np.frombuffer(row["rewards"], dtype=np.float32)[start:transition_stop]
        terminated = np.frombuffer(row["terminated"], dtype=np.uint8)[start:transition_stop]
        truncated = np.frombuffer(row["truncated"], dtype=np.uint8)[start:transition_stop]
        causal = np.frombuffer(row["causal_parent"], dtype=np.uint8)[start:transition_stop]
        time_to_event = np.frombuffer(row["time_to_event"], dtype=np.float32)[start:transition_stop]
        output["reward"] = torch.from_numpy(rewards.copy())
        output["terminal"] = torch.from_numpy(np.maximum(terminated, truncated).copy())
        output["causal_parent"] = torch.from_numpy(causal.copy())
        output["time_to_event"] = torch.from_numpy(time_to_event.copy())
        output["event_kinds"] = torch.from_numpy(
            self._multi_hot(json.loads(row["event_kinds"])[start:transition_stop], self._event_index)
        )
        output["delta_fields"] = torch.from_numpy(
            self._multi_hot(json.loads(row["delta_fields"])[start:transition_stop], self._delta_index)
        )
        output["causal_edges"] = torch.from_numpy(
            self._multi_hot(json.loads(row["causal_edges"])[start:transition_stop], self._edge_index)
        )
        valid = np.frombuffer(row["counterfactual_valid"], dtype=np.uint8).astype(bool)
        is_boundary = row.get("pair_boundary_tick") is not None and (
            int(row["start_tick"]) + start == int(row["pair_boundary_tick"])
        )
        mask = (
            valid
            & bool(is_boundary)
            & (np.asarray(self.pair_horizons, dtype=np.int64) <= self.context)
        )
        propagated = np.frombuffer(row["counterfactual_propagated"], dtype=np.uint8)
        reward_delta = np.frombuffer(row["counterfactual_reward_delta"], dtype=np.float32)
        output["counterfactual_mask"] = torch.from_numpy(mask.copy())
        output["counterfactual_propagated"] = torch.from_numpy(propagated.astype(np.float32))
        output["counterfactual_reward_delta"] = torch.from_numpy(reward_delta.copy())
        # Deprecated scalar aliases keep old analysis scripts readable while
        # all objective profiles use the typed/per-horizon targets above.
        output["counterfactual_diverged"] = torch.tensor(
            float(bool(np.any(propagated[mask]))), dtype=torch.float32
        )
        output.update(
            {
                "task": str(row["task"]),
                "seed": int(row["seed"]),
                "split": str(row["split"]),
                "policy": str(row["policy"]),
                "epsilon": float(row["epsilon"]),
                "source_index": int(row["source_index"]),
                "pair_id": "" if row.get("pair_id") is None else str(row["pair_id"]),
                "pair_role": "" if row.get("pair_role") is None else str(row["pair_role"]),
                "start_tick": int(row["start_tick"]) + start,
                "domain": self._source_domain(int(row["source_index"])),
            }
        )
        return output

    def _source_domain(self, source_index: int) -> str:
        return _source_domain_label(self.dataset_sources[source_index])

    def _decode_sensor_window(
        self, output: dict[str, Any], row: dict[str, Any], sensor: str, start: int, stop: int
    ) -> None:
        import torch

        length = int(row["length"])
        indices = np.frombuffer(row[f"{sensor}_index"], dtype=np.int16).reshape(length + 1)[start:stop]
        sample_ids = np.frombuffer(row[f"{sensor}_sample_id"], dtype=np.int64).reshape(
            length + 1
        )[start:stop]
        output[f"{sensor}_sample_id"] = torch.from_numpy(sample_ids.copy())
        fields = (
            ("rgb", "depth", "normals", "seg")
            if sensor == "render"
            else ("lidar_range", "lidar_intensity", "lidar_seg")
        )
        for field in fields:
            frames = row.get(f"{field}_frames")
            if frames is None:
                continue
            spec = self.schema[field]
            zero = np.zeros(spec["shape"], dtype=np.dtype(spec["dtype"]))
            decoded = [
                zero
                if slot < 0
                else np.frombuffer(frames[int(slot)], dtype=np.dtype(spec["dtype"])).reshape(spec["shape"])
                for slot in indices
            ]
            output[field] = torch.from_numpy(np.stack(decoded).copy())

    @staticmethod
    def _multi_hot(values: list[list[str]], index: dict[str, int]) -> np.ndarray:
        result = np.zeros((len(values), len(index)), dtype=np.float32)
        for row, labels in enumerate(values):
            for label in labels:
                if label in index:
                    result[row, index[label]] = 1.0
        return result

    def sample_source(self, source_index: int, rng: np.random.Generator) -> int:
        ranges = self._ranges_by_source.get(int(source_index), ())
        total = sum(count for _, count, _ in ranges)
        if total <= 0:
            raise ValueError(f"source {source_index} has no windows")
        offset = int(rng.integers(0, total))
        for first, count, _ in ranges:
            if offset < count:
                return first + offset
            offset -= count
        raise AssertionError("unreachable source window offset")

    def source_boundary_index(self, source_index: int, boundary_tick: int) -> int:
        for first, count, start_tick in self._ranges_by_source.get(int(source_index), ()):
            offset = int(boundary_tick) - start_tick
            if 0 <= offset < count:
                return first + offset
        raise ValueError(
            f"source {source_index} does not contain boundary tick {boundary_tick}"
        )


class DeterministicBatchSampler:
    """Stateless, source-balanced supercycles safe across prefetch/resume.

    A ten-sample cycle contains five expert, one sample from each epsilon
    stratum, and the aligned control/treatment branches of one pair.  Sources
    are selected before windows, so long episodes cannot dominate training.
    """

    def __init__(
        self,
        dataset: TrainingPackDataset | int,
        batch_size: int,
        *,
        seed: int,
        start_batch: int,
        total_batches: int,
    ):
        self.dataset = dataset if isinstance(dataset, TrainingPackDataset) else None
        self.dataset_size = len(dataset) if self.dataset is not None else int(dataset)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.start_batch = int(start_batch)
        self.total_batches = int(total_batches)
        self._expert_sources: tuple[int, ...] = ()
        self._mixed_sources: tuple[tuple[int, ...], ...] = ()
        self._pairs: tuple[tuple[int, int, int], ...] = ()
        if self.dataset is not None:
            eligible = set(self.dataset._ranges_by_source)
            expert: list[int] = []
            mixed: dict[float, list[int]] = {}
            pairs: dict[str, dict[str, int]] = {}
            for source_index, source in enumerate(self.dataset.dataset_sources):
                if source_index not in eligible:
                    continue
                policy = str(source.get("policy", "unknown"))
                if policy == "oracle_expert":
                    expert.append(source_index)
                elif policy == "epsilon_mixed":
                    mixed.setdefault(round(float(source.get("epsilon", 0.0)), 6), []).append(
                        source_index
                    )
                elif policy == "paired_intervention" and source.get("pair_id"):
                    pairs.setdefault(str(source["pair_id"]), {})[
                        str(source.get("pair_role", ""))
                    ] = source_index
            complete_pairs: list[tuple[int, int, int]] = []
            for roles in pairs.values():
                if set(roles) != {"control", "treatment"}:
                    continue
                control = roles["control"]
                treatment = roles["treatment"]
                control_ticks = {
                    tick for _, _, tick in self.dataset._ranges_by_source[control]
                }
                treatment_ticks = {
                    tick for _, _, tick in self.dataset._ranges_by_source[treatment]
                }
                common = sorted(control_ticks & treatment_ticks)
                if common:
                    complete_pairs.append((control, treatment, common[0]))
            self._expert_sources = tuple(expert)
            self._mixed_sources = tuple(
                tuple(mixed[epsilon]) for epsilon in sorted(mixed)
            )
            self._pairs = tuple(complete_pairs)
        self._balanced = bool(
            self._expert_sources and len(self._mixed_sources) == 3 and self._pairs
        )

    def __iter__(self) -> Iterator[list[int]]:
        for batch_number in range(self.start_batch, self.total_batches):
            if not self._balanced:
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, batch_number]))
                yield rng.integers(0, self.dataset_size, size=self.batch_size).tolist()
                continue
            first_ordinal = batch_number * self.batch_size
            yield [
                self._balanced_index(first_ordinal + offset)
                for offset in range(self.batch_size)
            ]

    def __len__(self) -> int:
        return max(0, self.total_batches - self.start_batch)

    def _balanced_index(self, ordinal: int) -> int:
        assert self.dataset is not None
        position = ordinal % 10
        cycle = ordinal // 10
        if position < 5:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, cycle, position])
            )
            source = self._expert_sources[int(rng.integers(len(self._expert_sources)))]
            return self.dataset.sample_source(source, rng)
        if position < 8:
            epsilon_slot = position - 5
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, cycle, 100 + epsilon_slot])
            )
            sources = self._mixed_sources[epsilon_slot]
            source = sources[int(rng.integers(len(sources)))]
            return self.dataset.sample_source(source, rng)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, cycle, 200]))
        control, treatment, boundary_tick = self._pairs[
            int(rng.integers(len(self._pairs)))
        ]
        source = control if position == 8 else treatment
        return self.dataset.source_boundary_index(source, boundary_tick)


def make_training_loader(
    dataset: TrainingPackDataset,
    *,
    batch_size: int,
    seed: int,
    start_batch: int,
    total_batches: int,
    workers: int = 8,
    prefetch_factor: int = 4,
):
    import torch
    from torch.utils.data import DataLoader

    sampler = DeterministicBatchSampler(
        dataset,
        batch_size,
        seed=seed,
        start_batch=start_batch,
        total_batches=total_batches,
    )
    kwargs: dict[str, Any] = {
        "batch_sampler": sampler,
        "num_workers": workers,
        "pin_memory": True,
        # DataLoader otherwise consumes the process-global Torch RNG when an
        # iterator is created. A private generator keeps resume bit-exact.
        "generator": torch.Generator().manual_seed(int(seed) + 4_294_967),
    }
    if workers:
        kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": prefetch_factor,
            }
        )
    return DataLoader(dataset, **kwargs)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


__all__ = [
    "DATASET_FORMAT",
    "DATASET_VERSION",
    "DeterministicBatchSampler",
    "FORBIDDEN_MODEL_INPUTS",
    "INTERVENTION_KINDS",
    "MODEL_INPUT_FIELDS",
    "MODEL_LABEL_FIELDS",
    "PACK_FORMAT",
    "PACK_VERSION",
    "PAIR_HORIZONS",
    "TrainingPackDataset",
    "assign_split",
    "build_training_pack",
    "bundle_sha256",
    "file_sha256",
    "make_training_loader",
    "read_dataset_manifest",
    "validate_model_input_schema",
    "write_dataset_manifest",
]
