"""Fixed-suite factual, counterfactual, probe, and performance evaluation."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import ResearchConfig
from .models import build_model
from .probes import (
    ProbeProtocol,
    ProbeTask,
    binary_classification_metrics,
    fit_frozen_linear_probes,
    regression_metrics,
)
from .train import _move_batch, _resolve_device
from .training_pack import TrainingPackDataset


EVALUATION_FORMAT = "voxelgym.evaluation"
EVALUATION_VERSION = 2
EVAL_SUITE_VERSION = 1
PROBE_TASKS = ("event", "delta", "typed_edge", "reward", "terminal", "paired_effect")


@torch.no_grad()
def evaluate_run(
    run_dir: str | Path,
    *,
    device_name: str | None = None,
    pack_path_override: str | Path | None = None,
    probe_steps: int = 2_000,
) -> dict[str, Any]:
    """Evaluate one v2 checkpoint on a run-seed-independent test suite."""

    run = Path(run_dir).resolve()
    config = ResearchConfig.from_dict(
        json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    )
    model_metadata_path = run / "model.json"
    model_metadata = (
        json.loads(model_metadata_path.read_text(encoding="utf-8"))
        if model_metadata_path.exists()
        else {}
    )
    data_record = json.loads((run / "data.json").read_text(encoding="utf-8"))
    training_pack_path = _resolve_recorded_path(data_record["training_pack_manifest"])
    training_manifest = json.loads(training_pack_path.read_text(encoding="utf-8"))
    event_vocab = tuple(training_manifest.get("event_vocab", ()))
    delta_vocab = tuple(training_manifest.get("delta_vocab", ()))
    edge_vocab = tuple(training_manifest.get("edge_vocab", ()))
    evaluation_pack_path = _resolve_recorded_path(
        pack_path_override or data_record["training_pack_manifest"]
    )
    suite = fixed_evaluation_suite(
        evaluation_pack_path, context=config.model.context, split="test"
    )

    device = _resolve_device(device_name or config.training.device)
    model = _build_model(config, event_vocab, delta_vocab, edge_vocab).to(device)
    checkpoint_path = _latest_checkpoint(run)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "voxelgym.checkpoint" or checkpoint.get("format_version") != 2:
        raise ValueError("formal evaluation requires voxelgym.checkpoint v2")
    if checkpoint.get("pack_fingerprint") != training_manifest.get("fingerprint"):
        raise ValueError("checkpoint Training Pack identity does not match run metadata")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    loader = _suite_loader(
        suite,
        batch_size=config.training.microbatch,
        pin_memory=device.type == "cuda",
    )
    horizon_totals: dict[int, list[float]] = {}
    task_horizons: dict[str, dict[int, list[float]]] = {}
    domain_horizons: dict[str, dict[int, list[float]]] = {}
    classification = _BinaryAccumulator()
    reward = _RegressionAccumulator()
    time_to_event = _RegressionAccumulator()
    depth = _DepthAccumulator()
    segmentation = _SegmentationAccumulator(classes=64)
    counterfactual = _UniquePairAccumulator(config.model.horizons)
    representation = _EffectiveRankAccumulator()

    for batch in loader:
        tasks = list(batch["task"])
        domains = list(batch["domain"])
        _remap_supervision(batch, suite.base, event_vocab, delta_vocab, edge_vocab)
        moved = _move_batch(batch, device)
        with _autocast(config, device):
            output = model(moved)
        active = set(output.get("active_losses", ()))
        _accumulate_horizons(
            config.model.kind,
            output,
            config.model.horizons,
            horizon_totals,
            tasks,
            domains,
            task_horizons,
            domain_horizons,
        )
        transition_mask = _transition_mask(output, moved)
        if "event" in active:
            classification.add("event", output.get("event"), moved.get("event_kinds"), transition_mask)
        if "delta" in active:
            classification.add("delta", output.get("delta"), moved.get("delta_fields"), transition_mask)
        if "causal_edge" in active:
            classification.add(
                "typed_edge",
                output.get("causal_edge", output.get("causal_edges")),
                moved.get("causal_edges"),
                transition_mask,
            )
        if "terminal" in active:
            classification.add("terminal", output.get("terminal"), moved.get("terminal"), transition_mask)
        if "reward" in active:
            reward.add(output.get("reward"), moved.get("reward"), transition_mask)
        if "time_to_event" in active:
            time_to_event.add(
                output.get("time_to_event"), moved.get("time_to_event"), transition_mask, log_target=True
            )
        if "depth" in active:
            depth.add(output, moved)
        if "seg" in active:
            segmentation.add(output, moved)
        if "counterfactual_latent_effect" in active:
            counterfactual.add(output, moved, batch)
        if output.get("probe_latent") is not None:
            representation.add(output["probe_latent"], transition_mask)

    objective = str(getattr(config.model, "objective", None) or _default_objective(config.model.kind))
    arm = _arm_name(config.model.kind, objective)
    model_record = config.as_dict()["model"]
    report: dict[str, Any] = {
        "format": EVALUATION_FORMAT,
        "format_version": EVALUATION_VERSION,
        "run": str(run),
        "run_seed": config.run.seed,
        "arm": arm,
        "model_kind": config.model.kind,
        "objective": objective,
        "active_losses": list(getattr(model, "active_losses", ())),
        "model_metadata": model_metadata,
        "model_fingerprint": _fingerprint(model_record),
        "architecture_fingerprint": _architecture_fingerprint(model_record),
        "checkpoint": str(checkpoint_path),
        "checkpoint_format_version": 2,
        "training_pack": str(training_pack_path),
        "training_pack_fingerprint": training_manifest["fingerprint"],
        "evaluation_pack": str(evaluation_pack_path),
        "dataset_fingerprint": suite.base.manifest["dataset_fingerprint"],
        "evaluation_pack_fingerprint": suite.base.manifest["fingerprint"],
        "evaluation_suite_fingerprint": suite.fingerprint,
        "evaluation_suite": {
            "format_version": EVAL_SUITE_VERSION,
            "split": "test",
            "windows": len(suite.dataset),
            "max_windows_per_task_domain": suite.max_per_task_domain,
        },
        "global_step": int(checkpoint["global_step"]),
        "split": "test",
        "windows": len(suite.dataset),
        "horizons": _finish_horizons(horizon_totals),
        "tasks": {
            task: _finish_horizons(values)
            for task, values in sorted(task_horizons.items())
        },
        "domains": {
            domain: _finish_horizons(values)
            for domain, values in sorted(domain_horizons.items())
        },
        "classification": classification.finish(),
        "reconstruction": {
            "depth": depth.finish(),
            "segmentation": segmentation.finish(),
        },
        "reward": reward.finish(),
        "time_to_event": time_to_event.finish(tick_space=True),
        "counterfactual": counterfactual.finish(),
        "representation": representation.finish(),
        "unseen_labels": {
            "events": sorted(set(suite.base.event_vocab) - set(event_vocab)),
            "deltas": sorted(set(suite.base.delta_vocab) - set(delta_vocab)),
            "edges": sorted(set(getattr(suite.base, "edge_vocab", ())) - set(edge_vocab)),
        },
        "performance": _performance_report(run),
    }
    report["frozen_probes"] = _evaluate_frozen_probes(
        model,
        training_pack_path,
        suite,
        config,
        event_vocab,
        delta_vocab,
        edge_vocab,
        device,
        steps=probe_steps,
    )
    report["acceptance"] = _acceptance(
        config.model.kind, horizon_totals, report["representation"]
    )

    evaluation_name = (
        "evaluation.json"
        if pack_path_override is None
        else f"evaluation-{suite.base.manifest['dataset_fingerprint'][:12]}.json"
    )
    (run / evaluation_name).write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


@dataclass(frozen=True, slots=True)
class _SuiteEntry:
    index: int
    source_index: int
    task: str
    domain: str
    seed: int
    pair_id: str
    pair_role: str
    start_tick: int

    def record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "source_index": self.source_index,
            "task": self.task,
            "domain": self.domain,
            "seed": self.seed,
            "pair_id": self.pair_id,
            "pair_role": self.pair_role,
            "start_tick": self.start_tick,
        }


class _IndexedDataset(Dataset):
    def __init__(self, base: TrainingPackDataset, entries: Iterable[_SuiteEntry]):
        self.base = base
        self.entries = tuple(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.base[self.entries[index].index]


@dataclass(frozen=True, slots=True)
class FixedEvaluationSuite:
    base: TrainingPackDataset
    dataset: _IndexedDataset
    entries: tuple[_SuiteEntry, ...]
    fingerprint: str
    max_per_task_domain: int


class _PairPreservingBatchSampler:
    def __init__(self, entries: tuple[_SuiteEntry, ...], batch_size: int):
        if batch_size <= 0:
            raise ValueError("evaluation batch size must be positive")
        units: list[list[int]] = []
        index = 0
        while index < len(entries):
            entry = entries[index]
            if not entry.pair_id:
                units.append([index])
                index += 1
                continue
            identity = (entry.pair_id, entry.start_tick)
            stop = index + 1
            while stop < len(entries) and (
                entries[stop].pair_id,
                entries[stop].start_tick,
            ) == identity:
                stop += 1
            units.append(list(range(index, stop)))
            index = stop
        self.batches: list[list[int]] = []
        current: list[int] = []
        for unit in units:
            if current and len(current) + len(unit) > batch_size:
                self.batches.append(current)
                current = []
            current.extend(unit)
            if len(current) >= batch_size:
                self.batches.append(current)
                current = []
        if current:
            self.batches.append(current)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def _suite_loader(
    suite: FixedEvaluationSuite,
    *,
    batch_size: int,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        suite.dataset,
        batch_sampler=_PairPreservingBatchSampler(suite.entries, batch_size),
        num_workers=0,
        pin_memory=pin_memory,
    )


def fixed_evaluation_suite(
    manifest_path: str | Path,
    *,
    context: int,
    split: str = "test",
    max_per_task_domain: int = 64,
) -> FixedEvaluationSuite:
    """Derive the canonical one-window-per-episode suite without a run seed."""

    base = TrainingPackDataset(manifest_path, split=split, context=context)
    recorded = base.manifest.get("evaluation_suite")
    if split == "test" and isinstance(recorded, dict):
        return _consume_recorded_suite(base, recorded, context=context)
    candidates: list[_SuiteEntry] = []
    for reference_index, reference in enumerate(base.references):
        row = base._read_row(reference)
        source_index = int(row["source_index"])
        pair_id = "" if row.get("pair_id") is None else str(row["pair_id"])
        pair_role = "" if row.get("pair_role") is None else str(row["pair_role"])
        window_count = int(base._window_counts[reference_index])
        shared_identity = pair_id or f"source:{source_index}"
        boundary_tick = row.get("pair_boundary_tick") if pair_id else None
        if boundary_tick is not None:
            offset = int(boundary_tick) - int(row["start_tick"])
            if not 0 <= offset < window_count:
                continue
        else:
            offset = int.from_bytes(
                hashlib.sha256(
                    f"{base.manifest['fingerprint']}:{split}:{shared_identity}:{row['start_tick']}".encode()
                ).digest()[:8],
                "little",
            ) % window_count
        candidates.append(
            _SuiteEntry(
                index=int(base._cumulative[reference_index]) + offset,
                source_index=source_index,
                task=str(row["task"]),
                domain=base._source_domain(source_index),
                seed=int(row["seed"]),
                pair_id=pair_id,
                pair_role=pair_role,
                start_tick=int(row["start_tick"]) + offset,
            )
        )

    by_episode: dict[tuple[int, str], _SuiteEntry] = {}
    for entry in candidates:
        key = (entry.source_index, entry.pair_role)
        current = by_episode.get(key)
        if current is None or _episode_choice_key(entry) < _episode_choice_key(current):
            by_episode[key] = entry
    unpaired: list[tuple[_SuiteEntry, ...]] = []
    paired: dict[tuple[str, int], list[_SuiteEntry]] = {}
    for entry in by_episode.values():
        if entry.pair_id:
            paired.setdefault((entry.pair_id, entry.start_tick), []).append(entry)
        else:
            unpaired.append((entry,))
    units = unpaired + [
        tuple(sorted(entries, key=lambda item: (item.pair_role != "control", item.source_index)))
        for entries in paired.values()
        if {entry.pair_role for entry in entries} >= {"control", "treatment"}
    ]
    grouped: dict[tuple[str, str], list[tuple[_SuiteEntry, ...]]] = {}
    for unit in units:
        grouped.setdefault((unit[0].task, unit[0].domain), []).append(unit)
    selected_units: list[tuple[_SuiteEntry, ...]] = []
    for _group, group_units in sorted(grouped.items()):
        used = 0
        for unit in sorted(group_units, key=_unit_sort_key):
            if used + len(unit) > max_per_task_domain:
                continue
            selected_units.append(unit)
            used += len(unit)
    selected_units.sort(key=lambda unit: (len(unit) != 2, _unit_sort_key(unit)))
    entries = tuple(entry for unit in selected_units for entry in unit)
    if not entries:
        raise ValueError(f"Training Pack has no fixed {split!r} evaluation windows")
    payload = {
        "format": "voxelgym.eval-suite",
        "format_version": EVAL_SUITE_VERSION,
        "pack_fingerprint": base.manifest["fingerprint"],
        "split": split,
        "context": int(context),
        "max_per_task_domain": int(max_per_task_domain),
        "entries": [entry.record() for entry in entries],
    }
    fingerprint = _fingerprint(payload)
    dataset = _IndexedDataset(base, entries)
    return FixedEvaluationSuite(base, dataset, entries, fingerprint, max_per_task_domain)


def _consume_recorded_suite(
    base: TrainingPackDataset,
    recorded: dict[str, Any],
    *,
    context: int,
) -> FixedEvaluationSuite:
    if recorded.get("version") != EVAL_SUITE_VERSION or recorded.get("split") != "test":
        raise ValueError("unsupported Training Pack evaluation_suite")
    if int(context) != int(base.manifest["window_steps"]):
        raise ValueError("formal Eval Suite requires model.context == pack window_steps")
    expected = recorded.get("fingerprint")
    unsigned = dict(recorded)
    unsigned.pop("fingerprint", None)
    if expected != _fingerprint(unsigned):
        raise ValueError("Training Pack Eval Suite fingerprint mismatch")
    reference_index = {
        (reference.file, reference.row_group): index
        for index, reference in enumerate(base.references)
    }
    entries: list[_SuiteEntry] = []
    for item in recorded.get("entries", ()):
        key = (str(item["file"]), int(item["row_group"]))
        if key not in reference_index:
            raise ValueError(f"Eval Suite references a missing test segment: {key}")
        index = reference_index[key]
        reference = base.references[index]
        start = int(item["start"])
        if not 0 <= start < int(base._window_counts[index]):
            raise ValueError("Eval Suite window offset is outside its segment")
        source_index = int(item["source_index"])
        source = base.dataset_sources[source_index]
        task = str(item["task"])
        domain = str(item["domain"])
        if task != str(source["task"]) or domain != base._source_domain(source_index):
            raise ValueError("Eval Suite task/domain metadata does not match its source")
        entries.append(
            _SuiteEntry(
                index=int(base._cumulative[index]) + start,
                source_index=source_index,
                task=task,
                domain=domain,
                seed=int(source["seed"]),
                pair_id="" if item.get("pair_id") is None else str(item["pair_id"]),
                pair_role="" if item.get("pair_role") is None else str(item["pair_role"]),
                start_tick=reference.start_tick + start,
            )
        )
    if not entries:
        raise ValueError("Training Pack Eval Suite contains no test windows")
    values = tuple(entries)
    return FixedEvaluationSuite(
        base,
        _IndexedDataset(base, values),
        values,
        str(expected),
        int(recorded.get("max_per_task_domain", 64)),
    )


class _BinaryAccumulator:
    def __init__(self) -> None:
        self.values: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}

    def add(
        self,
        name: str,
        prediction: torch.Tensor | None,
        target: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> None:
        if prediction is None or target is None or target.numel() == 0:
            return
        predicted, expected = _align_prediction_target(prediction.detach(), target.detach())
        predicted = predicted.reshape(-1, predicted.shape[-1] if predicted.ndim > 2 else 1)
        expected = expected.reshape_as(predicted)
        if mask is not None:
            aligned_mask = _align_mask(mask, prediction.shape[:2]).reshape(-1)
            predicted = predicted[aligned_mask]
            expected = expected[aligned_mask]
        if predicted.numel() == 0:
            return
        scores, labels = self.values.setdefault(name, ([], []))
        scores.append(predicted.float().cpu())
        labels.append(expected.float().cpu())

    def finish(self) -> dict[str, Any]:
        return {
            name: binary_classification_metrics(torch.cat(scores), torch.cat(labels))
            for name, (scores, labels) in sorted(self.values.items())
        }


class _RegressionAccumulator:
    def __init__(self) -> None:
        self.prediction: list[torch.Tensor] = []
        self.target: list[torch.Tensor] = []

    def add(
        self,
        prediction: torch.Tensor | None,
        target: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        *,
        log_target: bool = False,
    ) -> None:
        if prediction is None or target is None or target.numel() == 0:
            return
        predicted, expected = _align_prediction_target(prediction.detach(), target.detach())
        if log_target:
            predicted = torch.expm1(predicted.float()).clamp_min(0.0)
        if mask is not None:
            aligned_mask = _align_mask(mask, prediction.shape[:2])
            predicted = predicted[aligned_mask]
            expected = expected[aligned_mask]
        self.prediction.append(predicted.float().reshape(-1, 1).cpu())
        self.target.append(expected.float().reshape(-1, 1).cpu())

    def finish(self, *, tick_space: bool = False) -> dict[str, Any]:
        if not self.prediction:
            return {"samples": 0, "mae": None}
        result = regression_metrics(torch.cat(self.prediction), torch.cat(self.target))
        if tick_space:
            result["tick_mae"] = result.pop("mae")
        return result


class _DepthAccumulator:
    def __init__(self) -> None:
        self.absolute_relative = 0.0
        self.square = 0.0
        self.count = 0

    def add(self, output: dict[str, Any], batch: dict[str, Any]) -> None:
        prediction = output.get("depth")
        target = batch.get("depth")
        indices = output.get("reconstruction_indices")
        if prediction is None or target is None:
            return
        state_target = (
            target[:, 1:]
            if batch.get("action") is not None
            and target.shape[1] == batch["action"].shape[1] + 1
            else target
        )
        expected = _reconstruction_target(
            state_target.float(), prediction.shape[1], indices
        )
        expected = F.interpolate(
            expected.flatten(0, 1).unsqueeze(1), prediction.shape[-2:], mode="nearest"
        ).reshape_as(prediction)
        predicted = torch.expm1(prediction.float() * 5.0).clamp_min(0.0)
        valid = torch.isfinite(expected) & torch.isfinite(predicted) & (expected > 0)
        transition_mask = output.get("transition_mask")
        if transition_mask is not None:
            reconstruction_mask = (
                transition_mask.index_select(1, indices.long())
                if indices is not None
                else transition_mask[:, : prediction.shape[1]]
            )
            valid &= reconstruction_mask[..., None, None]
        difference = predicted[valid] - expected[valid]
        self.absolute_relative += float(
            (difference.abs() / expected[valid].clamp_min(1e-3)).sum()
        )
        self.square += float(torch.square(difference).sum())
        self.count += int(difference.numel())

    def finish(self) -> dict[str, Any]:
        return {
            "samples": self.count,
            "abs_rel": self.absolute_relative / self.count if self.count else None,
            "rmse": math.sqrt(self.square / self.count) if self.count else None,
        }


class _SegmentationAccumulator:
    def __init__(self, *, classes: int):
        self.intersection = torch.zeros(classes, dtype=torch.int64)
        self.union = torch.zeros(classes, dtype=torch.int64)

    def add(self, output: dict[str, Any], batch: dict[str, Any]) -> None:
        prediction = output.get("seg")
        target = batch.get("seg")
        indices = output.get("reconstruction_indices")
        if prediction is None or target is None:
            return
        state_target = (
            target[:, 1:]
            if batch.get("action") is not None
            and target.shape[1] == batch["action"].shape[1] + 1
            else target
        )
        expected = _reconstruction_target(
            state_target.long(), prediction.shape[1], indices
        )
        expected = torch.bitwise_and(expected, 0xFFF).clamp_max(prediction.shape[2] - 1)
        expected = F.interpolate(
            expected.flatten(0, 1).unsqueeze(1).float(),
            prediction.shape[-2:],
            mode="nearest",
        ).squeeze(1).long()
        predicted = prediction.argmax(dim=2).flatten(0, 1)
        transition_mask = output.get("transition_mask")
        if transition_mask is not None:
            reconstruction_mask = (
                transition_mask.index_select(1, indices.long())
                if indices is not None
                else transition_mask[:, : prediction.shape[1]]
            ).flatten()
            predicted = predicted[reconstruction_mask]
            expected = expected[reconstruction_mask]
        for label in range(len(self.intersection)):
            predicted_label = predicted == label
            expected_label = expected == label
            self.intersection[label] += (predicted_label & expected_label).sum().cpu()
            self.union[label] += (predicted_label | expected_label).sum().cpu()

    def finish(self) -> dict[str, Any]:
        present = self.union > 0
        values = self.intersection[present].double() / self.union[present].double()
        return {
            "classes_present": int(present.sum()),
            "miou": float(values.mean()) if values.numel() else None,
        }


class _EffectiveRankAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.total: torch.Tensor | None = None
        self.cross: torch.Tensor | None = None

    def add(self, value: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        features = value.detach()
        if mask is not None:
            features = features[_align_mask(mask, value.shape[:2])]
        features = features.reshape(-1, features.shape[-1]).double().cpu()
        if not features.numel():
            return
        if self.total is None:
            self.total = torch.zeros(features.shape[1], dtype=torch.float64)
            self.cross = torch.zeros(features.shape[1], features.shape[1], dtype=torch.float64)
        self.count += features.shape[0]
        self.total += features.sum(dim=0)
        assert self.cross is not None
        self.cross += features.T @ features

    def finish(self) -> dict[str, Any]:
        if self.count < 2 or self.total is None or self.cross is None:
            return {
                "samples": self.count,
                "dimension": 0,
                "effective_rank": None,
                "ratio": None,
            }
        covariance = (
            self.cross - torch.outer(self.total, self.total) / self.count
        ) / (self.count - 1)
        singular = torch.sqrt(torch.linalg.eigvalsh(covariance).clamp_min(0.0))
        probability = singular / singular.sum().clamp_min(1e-30)
        rank = float(
            torch.exp(-(probability * probability.clamp_min(1e-30).log()).sum())
        )
        return {
            "samples": self.count,
            "dimension": int(self.total.numel()),
            "effective_rank": rank,
            "ratio": rank / self.total.numel(),
        }


class _UniquePairAccumulator:
    def __init__(self, horizons: Iterable[int]):
        self.horizons = tuple(int(value) for value in horizons)
        self.records: dict[tuple[str, int, int], dict[str, list[Any]]] = {}

    def add(
        self,
        output: dict[str, Any],
        moved: dict[str, Any],
        original: dict[str, Any],
    ) -> None:
        pair_indices = output.get("cf_pair_indices")
        if pair_indices is None or pair_indices.numel() == 0:
            return
        propagation = output.get("counterfactual_propagation")
        reward_delta = output.get("counterfactual_reward_delta")
        latent_effect = output.get("counterfactual_latent_effect")
        target_propagation = _first_present(
            moved,
            "counterfactual_propagated",
            "counterfactual_propgated",
            "counterfactual_diverged",
        )
        target_reward = moved.get("counterfactual_reward_delta")
        valid_horizons = moved.get("counterfactual_mask")
        pair_ids = list(original.get("pair_id", ()))
        starts = original.get("start_tick")
        target_latent = output.get("target_latent")
        visible_prefix = output.get("visible_prefix")
        for pair_number, (control, treatment) in enumerate(
            pair_indices.detach().cpu().tolist()
        ):
            pair_id = str(pair_ids[treatment] or pair_ids[control])
            start_tick = int(starts[treatment]) if starts is not None else 0
            for horizon_index, horizon in enumerate(self.horizons):
                if (
                    valid_horizons is not None
                    and not bool(valid_horizons[treatment, horizon_index])
                ):
                    continue
                record = self.records.setdefault(
                    (pair_id, start_tick, horizon),
                    {
                        "propagation": [],
                        "propagation_target": [],
                        "reward": [],
                        "reward_target": [],
                        "latent": [],
                        "latent_target": [],
                    },
                )
                if propagation is not None and target_propagation is not None:
                    record["propagation"].append(
                        float(propagation[pair_number, horizon_index].detach())
                    )
                    record["propagation_target"].append(
                        float(target_propagation[treatment, horizon_index].detach())
                    )
                if reward_delta is not None and target_reward is not None:
                    record["reward"].append(
                        float(reward_delta[pair_number, horizon_index].detach())
                    )
                    record["reward_target"].append(
                        float(target_reward[treatment, horizon_index].detach())
                    )
                if (
                    latent_effect is not None
                    and target_latent is not None
                    and visible_prefix is not None
                ):
                    boundary = int(visible_prefix[treatment].detach())
                    target_index = min(
                        max(boundary - 1 + horizon, 0), target_latent.shape[1] - 1
                    )
                    actual = (
                        target_latent[treatment, target_index]
                        - target_latent[control, target_index]
                    )
                    record["latent"].append(
                        latent_effect[pair_number, horizon_index].detach().float().cpu()
                    )
                    record["latent_target"].append(actual.detach().float().cpu())

    def finish(self) -> dict[str, Any]:
        unique_pairs = {(pair_id, start) for pair_id, start, _ in self.records}
        by_horizon: dict[str, Any] = {}
        for horizon in self.horizons:
            records = [value for key, value in self.records.items() if key[2] == horizon]
            report: dict[str, Any] = {"unique_pairs": len(records)}
            propagation_prediction = [
                statistics.mean(item["propagation"])
                for item in records
                if item["propagation"]
            ]
            propagation_target = [
                statistics.mean(item["propagation_target"])
                for item in records
                if item["propagation_target"]
            ]
            if propagation_prediction:
                report["propagation"] = binary_classification_metrics(
                    torch.tensor(propagation_prediction),
                    torch.tensor(propagation_target),
                )
            reward_prediction = [
                statistics.mean(item["reward"])
                for item in records
                if item["reward"]
            ]
            reward_target = [
                statistics.mean(item["reward_target"])
                for item in records
                if item["reward_target"]
            ]
            if reward_prediction:
                report["reward_delta"] = regression_metrics(
                    torch.tensor(reward_prediction), torch.tensor(reward_target)
                )
            latent_mse: list[float] = []
            latent_cosine: list[float] = []
            for item in records:
                if not item["latent"]:
                    continue
                predicted = torch.stack(item["latent"]).mean(dim=0)
                expected = torch.stack(item["latent_target"]).mean(dim=0)
                latent_mse.append(float(torch.square(predicted - expected).mean()))
                latent_cosine.append(
                    float(F.cosine_similarity(predicted[None], expected[None]))
                )
            if latent_mse:
                report["latent_effect"] = {
                    "mse": statistics.mean(latent_mse),
                    "cosine_similarity": statistics.mean(latent_cosine),
                }
            by_horizon[str(horizon)] = report
        return {"unique_pairs": len(unique_pairs), "horizons": by_horizon}


def _evaluate_frozen_probes(
    model: torch.nn.Module,
    training_pack_path: Path,
    test_suite: FixedEvaluationSuite,
    config: ResearchConfig,
    event_vocab: tuple[str, ...],
    delta_vocab: tuple[str, ...],
    edge_vocab: tuple[str, ...],
    device: torch.device,
    *,
    steps: int,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("probe_steps must be positive")
    try:
        train_suite = fixed_evaluation_suite(
            training_pack_path, context=config.model.context, split="train"
        )
        validation_suite = fixed_evaluation_suite(
            training_pack_path, context=config.model.context, split="validation"
        )
    except ValueError as error:
        # Tiny CI packs may intentionally omit a probe split.  The formal test
        # report is still valid, but it must say that probes were not run rather
        # than training a threshold on test or falling back to train.
        return {
            "status": "unavailable",
            "reason": str(error),
            "protocol": {
                "optimizer": "AdamW",
                "steps": steps,
                "learning_rate": 1e-3,
                "threshold_selection": "validation-max-f1",
                "test_thresholds": "locked",
            },
        }
    splits = {
        "train": _collect_probe_tasks(
            model, train_suite, config, event_vocab, delta_vocab, edge_vocab, device
        ),
        "validation": _collect_probe_tasks(
            model,
            validation_suite,
            config,
            event_vocab,
            delta_vocab,
            edge_vocab,
            device,
        ),
        "test": _collect_probe_tasks(
            model, test_suite, config, event_vocab, delta_vocab, edge_vocab, device
        ),
    }
    common = set.intersection(*(set(value) for value in splits.values()))
    missing = sorted(set(PROBE_TASKS) - common)
    if missing:
        return {
            "status": "unavailable",
            "reason": f"frozen-probe suite is missing targets: {missing}",
            "suite_fingerprints": {
                "train": train_suite.fingerprint,
                "validation": validation_suite.fingerprint,
                "test": test_suite.fingerprint,
            },
        }
    selected = tuple(name for name in PROBE_TASKS if name in common)
    report = fit_frozen_linear_probes(
        {name: splits["train"][name] for name in selected},
        {name: splits["validation"][name] for name in selected},
        {name: splits["test"][name] for name in selected},
        protocol=ProbeProtocol(steps=steps, learning_rate=1e-3, seed=0),
        device=device,
    )
    report["suite_fingerprints"] = {
        "train": train_suite.fingerprint,
        "validation": validation_suite.fingerprint,
        "test": test_suite.fingerprint,
    }
    return report


@torch.no_grad()
def _collect_probe_tasks(
    model: torch.nn.Module,
    suite: FixedEvaluationSuite,
    config: ResearchConfig,
    event_vocab: tuple[str, ...],
    delta_vocab: tuple[str, ...],
    edge_vocab: tuple[str, ...],
    device: torch.device,
) -> dict[str, ProbeTask]:
    collected: dict[str, tuple[list[torch.Tensor], list[torch.Tensor], str]] = {}
    loader = _suite_loader(suite, batch_size=config.training.microbatch)
    seen_pairs: set[tuple[str, int, int]] = set()
    for batch in loader:
        _remap_supervision(batch, suite.base, event_vocab, delta_vocab, edge_vocab)
        moved = _move_batch(batch, device)
        with _autocast(config, device):
            output = model(moved)
        features = output.get("probe_latent")
        if features is None:
            raise ValueError("model does not expose the required frozen probe_latent")
        mask = _align_mask(_transition_mask(output, moved), features.shape[:2])
        for name, key, kind in (
            ("event", "event_kinds", "binary"),
            ("delta", "delta_fields", "binary"),
            ("typed_edge", "causal_edges", "binary"),
            ("reward", "reward", "regression"),
            ("terminal", "terminal", "binary"),
        ):
            target = moved.get(key)
            if target is None or target.numel() == 0:
                continue
            length = min(features.shape[1], target.shape[1])
            feature_value = features[:, :length]
            target_value = target[:, :length]
            aligned_mask = _align_mask(mask, feature_value.shape[:2])
            _append_probe(
                collected,
                name,
                feature_value[aligned_mask],
                target_value[aligned_mask],
                kind,
            )

        pair_output = output
        if (
            getattr(config.model, "objective", "dynamics") != "counterfactual"
            and "visible_prefix" in inspect.signature(model.forward).parameters
        ):
            # Pair probes are causal queries from the common branch boundary.
            # Re-run only their representation with every later state masked;
            # declared actions/interventions remain visible to the backbone.
            with _autocast(config, device):
                pair_output = model(moved, visible_prefix=1)
        pair_indices = pair_output.get("cf_pair_indices")
        if pair_indices is None:
            pair_indices = _pair_indices_from_batch(batch, moved["action"].device)
        pair_features = pair_output.get("cf_probe_latent")
        if pair_features is None and pair_indices is not None and pair_indices.numel():
            selected = pair_output.get("selected_horizon_latent")
            if selected is not None:
                pair_features = torch.stack(
                    [selected[treatment] - selected[control] for control, treatment in pair_indices]
                )
            else:
                # RSSM's first transition representation consumes only the
                # shared boundary plus the first declared control.  Later GRU
                # states are deliberately excluded because they encode target
                # observations in the recurrent baseline.
                pair_features = torch.stack(
                    [
                        (features[treatment, 0] - features[control, 0])
                        .unsqueeze(0)
                        .expand(len(config.model.horizons), -1)
                        for control, treatment in pair_indices
                    ]
                )
        pair_target = _first_present(
            moved,
            "counterfactual_propagated",
            "counterfactual_propgated",
            "counterfactual_diverged",
        )
        pair_valid = moved.get("counterfactual_mask")
        if (
            pair_features is not None
            and pair_indices is not None
            and pair_target is not None
        ):
            pair_ids = list(batch.get("pair_id", ()))
            starts = batch.get("start_tick")
            for pair_number, (_control, treatment) in enumerate(
                pair_indices.detach().cpu().tolist()
            ):
                for horizon_index, horizon in enumerate(config.model.horizons):
                    if (
                        pair_valid is not None
                        and not bool(pair_valid[treatment, horizon_index])
                    ):
                        continue
                    identity = (
                        str(pair_ids[treatment]),
                        int(starts[treatment]),
                        int(horizon),
                    )
                    if identity in seen_pairs:
                        continue
                    seen_pairs.add(identity)
                    _append_probe(
                        collected,
                        "paired_effect",
                        pair_features[pair_number, horizon_index][None],
                        pair_target[treatment, horizon_index].reshape(1, 1),
                        "binary",
                    )
    return {
        name: ProbeTask(torch.cat(features), torch.cat(targets), kind)  # type: ignore[arg-type]
        for name, (features, targets, kind) in collected.items()
        if features and targets
    }


def _append_probe(
    destination: dict[str, tuple[list[torch.Tensor], list[torch.Tensor], str]],
    name: str,
    features: torch.Tensor,
    targets: torch.Tensor,
    kind: str,
) -> None:
    if features.numel() == 0 or targets.numel() == 0:
        return
    feature_list, target_list, existing_kind = destination.setdefault(
        name, ([], [], kind)
    )
    if existing_kind != kind:
        raise ValueError(f"probe kind changed for {name}")
    feature_list.append(
        features.detach().float().cpu().reshape(features.shape[0], -1)
    )
    target_list.append(targets.detach().float().cpu().reshape(targets.shape[0], -1))


def _accumulate_horizons(
    kind: str,
    output: dict[str, Any],
    horizons: tuple[int, ...],
    totals: dict[int, list[float]],
    tasks: list[str],
    domains: list[str],
    task_totals: dict[str, dict[int, list[float]]],
    domain_totals: dict[str, dict[int, list[float]]],
) -> None:
    target = output["target_latent"].detach()
    if kind == "jepa":
        batch_index = torch.arange(target.shape[0], device=target.device)
        copied = F.normalize(
            target[batch_index, output["visible_prefix"] - 1], dim=-1
        )
        valid = output["selected_horizon_mask"].bool()
        for horizon_index, horizon in enumerate(horizons):
            prediction = output["jepa_prediction"][:, horizon_index : horizon_index + 1]
            expected = output["jepa_target"][:, horizon_index : horizon_index + 1]
            copy_value = copied[:, None]
            mask = valid[:, horizon_index : horizon_index + 1]
            _add_errors(totals, horizon, prediction, copy_value, expected, mask)
            for index, task in enumerate(tasks):
                _add_errors(
                    task_totals.setdefault(task, {}), horizon,
                    prediction[index : index + 1], copy_value[index : index + 1],
                    expected[index : index + 1], mask[index : index + 1],
                )
            for index, domain in enumerate(domains):
                _add_errors(
                    domain_totals.setdefault(domain, {}), horizon,
                    prediction[index : index + 1], copy_value[index : index + 1],
                    expected[index : index + 1], mask[index : index + 1],
                )
        return
    if kind == "rssm":
        predictions = {1: output.get("next_latent", output.get("latent"))}
    else:
        predictions = {
            int(key): value for key, value in output["horizon_latent"].items()
        }
    for horizon in ((1,) if kind == "rssm" else horizons):
        prediction = predictions[horizon]
        expected = target[:, horizon : horizon + prediction.shape[1]]
        copied = target[:, : prediction.shape[1]]
        horizon_masks = output.get("horizon_mask", {})
        mask = horizon_masks.get(horizon) if isinstance(horizon_masks, dict) else None
        _add_errors(totals, horizon, prediction, copied, expected, mask)
        for index, task in enumerate(tasks):
            _add_errors(
                task_totals.setdefault(task, {}),
                horizon,
                prediction[index : index + 1],
                copied[index : index + 1],
                expected[index : index + 1],
                None if mask is None else mask[index : index + 1],
            )
        for index, domain in enumerate(domains):
            _add_errors(
                domain_totals.setdefault(domain, {}),
                horizon,
                prediction[index : index + 1],
                copied[index : index + 1],
                expected[index : index + 1],
                None if mask is None else mask[index : index + 1],
            )


def _add_errors(
    totals: dict[int, list[float]],
    horizon: int,
    prediction: torch.Tensor,
    copied: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> None:
    model_error = torch.square(prediction - target)
    copy_error = torch.square(copied - target)
    if mask is not None:
        valid = mask.bool()
        while valid.ndim < model_error.ndim:
            valid = valid.unsqueeze(-1)
        valid = valid.expand_as(model_error)
        model_error = model_error[valid]
        copy_error = copy_error[valid]
    entry = totals.setdefault(horizon, [0.0, 0.0, 0.0])
    entry[0] += float(model_error.sum())
    entry[1] += float(copy_error.sum())
    entry[2] += float(model_error.numel())


def _finish_horizons(values: dict[int, list[float]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon, (model_sum, copy_sum, count) in sorted(values.items()):
        model_mse = model_sum / max(count, 1)
        copy_mse = copy_sum / max(count, 1)
        result[str(horizon)] = {
            "samples": int(count),
            "model_mse": model_mse,
            "copy_last_mse": copy_mse,
            "model_to_copy_ratio": model_mse / max(copy_mse, 1e-12),
        }
    return result


def _acceptance(
    kind: str,
    horizons: dict[int, list[float]],
    rank: dict[str, Any],
) -> dict[str, Any]:
    ratios = {
        horizon: value[0] / max(value[1], 1e-12)
        for horizon, value in horizons.items()
    }
    if kind == "rssm":
        return {"rssm_h1_ratio_lt_0_9": ratios.get(1, math.inf) < 0.9}
    result = {
        "h1_ratio_lt_0_9": ratios.get(1, math.inf) < 0.9,
        "h4_ratio_lt_0_9": ratios.get(4, math.inf) < 0.9,
        "h16_ratio_lt_1_0": ratios.get(16, math.inf) < 1.0,
    }
    if kind == "jepa":
        result["effective_rank_ge_quarter_dimension"] = (
            rank.get("effective_rank") is not None
            and rank["effective_rank"] >= 0.25 * rank["dimension"]
        )
    return result


def aggregate_runs(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate arms only after validating shared experimental identity."""

    if not reports:
        raise ValueError("at least one evaluation report is required")
    identity_fields = (
        "dataset_fingerprint",
        "evaluation_pack_fingerprint",
        "evaluation_suite_fingerprint",
        "training_pack_fingerprint",
        "global_step",
    )
    identity = {field: reports[0].get(field) for field in identity_fields}
    for report in reports[1:]:
        mismatched = [
            field for field in identity_fields if report.get(field) != identity[field]
        ]
        if mismatched:
            raise ValueError(f"cross-run comparison identity mismatch: {mismatched}")
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        by_arm.setdefault(report["arm"], []).append(report)
    for arm, arm_values in by_arm.items():
        seeds = [int(report["run_seed"]) for report in arm_values]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate run seed for arm {arm!r}")
        if len({report["model_fingerprint"] for report in arm_values}) != 1:
            raise ValueError(f"model identity differs within arm {arm!r}")
    transformer_arms = [
        arm for arm in ("dynamics-t", "causal-t", "cf-t") if arm in by_arm
    ]
    if len(transformer_arms) > 1:
        fingerprints = {
            report["architecture_fingerprint"]
            for arm in transformer_arms
            for report in by_arm[arm]
        }
        if len(fingerprints) != 1:
            raise ValueError("Transformer arms do not share an architecture identity")

    arm_reports = {
        arm: _aggregate_arm(values) for arm, values in sorted(by_arm.items())
    }
    paired: dict[str, Any] = {}
    for left, right, label in (
        ("dynamics-t", "causal-t", "causal_minus_dynamics"),
        ("causal-t", "cf-t", "cf_minus_causal"),
    ):
        if left in by_arm and right in by_arm:
            paired[label] = _paired_arm_difference(by_arm[left], by_arm[right])
    result = {
        "format": "voxelgym.evaluation-aggregate",
        "format_version": 2,
        "comparison_identity": identity,
        "arms": arm_reports,
        "paired_differences": paired,
    }
    if len(arm_reports) == 1:
        only = next(iter(arm_reports.values()))
        result.update(
            {
                "runs": only["runs"],
                "run_seeds": only["run_seeds"],
                "metrics": only["metrics"],
            }
        )
    return result


def _aggregate_arm(reports: list[dict[str, Any]]) -> dict[str, Any]:
    flattened: dict[str, list[float]] = {}
    for report in reports:
        for prefix in (
            "horizons",
            "classification",
            "reconstruction",
            "reward",
            "time_to_event",
            "counterfactual",
            "representation",
            "frozen_probes",
            "tasks",
            "domains",
        ):
            for key, value in _flatten_numeric(
                report.get(prefix, {}), prefix
            ).items():
                flattened.setdefault(key, []).append(value)
    return {
        "runs": len(reports),
        "run_seeds": sorted(int(report["run_seed"]) for report in reports),
        "metrics": {
            key: _confidence_interval(values)
            for key, values in sorted(flattened.items())
            if len(values) == len(reports)
        },
    }


def _paired_arm_difference(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any]:
    left_by_seed = {int(report["run_seed"]): report for report in left}
    right_by_seed = {int(report["run_seed"]): report for report in right}
    if set(left_by_seed) != set(right_by_seed):
        raise ValueError("paired arm comparison requires identical run seeds")
    differences: dict[str, list[float]] = {}
    for seed in sorted(left_by_seed):
        left_values = _flatten_numeric(left_by_seed[seed], "")
        right_values = _flatten_numeric(right_by_seed[seed], "")
        for key in set(left_values) & set(right_values):
            if key.endswith("global_step") or key.endswith("run_seed"):
                continue
            differences.setdefault(key, []).append(
                right_values[key] - left_values[key]
            )
    return {
        "seeds": sorted(left_by_seed),
        "metrics": {
            key: _confidence_interval(values)
            for key, values in sorted(differences.items())
            if len(values) == len(left_by_seed)
        },
    }


def _performance_report(run: Path) -> dict[str, Any]:
    metrics_path = run / "metrics.jsonl"
    if not metrics_path.exists():
        return {"samples": 0}
    utilization: list[float] = []
    wait_fraction: list[float] = []
    peak_memory = 0.0
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("split") != "train":
            continue
        metrics = row["metrics"]
        if "gpu_utilization" in metrics:
            utilization.append(float(metrics["gpu_utilization"]))
        if "data_wait_fraction" in metrics:
            wait_fraction.append(float(metrics["data_wait_fraction"]))
        peak_memory = max(
            peak_memory, float(metrics.get("cuda_reserved_gib", 0.0))
        )
    return {
        "samples": len(wait_fraction),
        "median_gpu_utilization": (
            statistics.median(utilization) if utilization else None
        ),
        "median_data_wait_fraction": (
            statistics.median(wait_fraction) if wait_fraction else None
        ),
        "peak_cuda_reserved_gib": peak_memory or None,
        "gpu_utilization_ge_0_8": (
            statistics.median(utilization) >= 0.8 if utilization else None
        ),
        "data_wait_lt_0_1": (
            statistics.median(wait_fraction) < 0.1 if wait_fraction else None
        ),
        "memory_lt_30_gib": peak_memory < 30.0 if peak_memory else None,
    }


def _build_model(
    config: ResearchConfig,
    event_vocab: tuple[str, ...],
    delta_vocab: tuple[str, ...],
    edge_vocab: tuple[str, ...],
) -> torch.nn.Module:
    kwargs = {
        "event_classes": len(event_vocab),
        "delta_classes": len(delta_vocab),
    }
    if "edge_classes" in inspect.signature(build_model).parameters:
        kwargs["edge_classes"] = len(edge_vocab)
    return build_model(config.model, **kwargs)


def _remap_supervision(
    batch: dict[str, Any],
    dataset: TrainingPackDataset,
    event_vocab: tuple[str, ...],
    delta_vocab: tuple[str, ...],
    edge_vocab: tuple[str, ...],
) -> None:
    if "event_kinds" in batch:
        batch["event_kinds"] = _remap_multihot(
            batch["event_kinds"], dataset.event_vocab, event_vocab
        )
    if "delta_fields" in batch:
        batch["delta_fields"] = _remap_multihot(
            batch["delta_fields"], dataset.delta_vocab, delta_vocab
        )
    if "causal_edges" in batch:
        batch["causal_edges"] = _remap_multihot(
            batch["causal_edges"],
            tuple(getattr(dataset, "edge_vocab", ())),
            edge_vocab,
        )


def _remap_multihot(
    value: torch.Tensor,
    source: tuple[str, ...],
    target: tuple[str, ...],
) -> torch.Tensor:
    if source == target:
        return value
    result = value.new_zeros(*value.shape[:-1], len(target))
    source_index = {label: index for index, label in enumerate(source)}
    for target_index, label in enumerate(target):
        if label in source_index:
            result[..., target_index] = value[..., source_index[label]]
    return result


def _transition_mask(
    output: dict[str, Any], batch: dict[str, Any]
) -> torch.Tensor:
    mask = output.get("transition_mask", batch.get("transition_mask"))
    if mask is not None:
        return mask.bool()
    length = batch["action"].shape[1]
    return torch.ones(
        batch["action"].shape[0],
        length,
        dtype=torch.bool,
        device=batch["action"].device,
    )


def _align_prediction_target(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.ndim == 1:
        prediction = prediction.unsqueeze(-1)
    if target.ndim == 1:
        target = target.unsqueeze(-1)
    if prediction.ndim >= 2 and target.ndim >= 2:
        length = min(prediction.shape[1], target.shape[1])
        prediction = prediction[:, :length]
        target = target[:, :length]
    if prediction.shape != target.shape:
        if prediction.numel() != target.numel():
            raise ValueError(
                f"metric prediction/target shapes differ: {prediction.shape} vs {target.shape}"
            )
        target = target.reshape_as(prediction)
    return prediction, target


def _align_mask(mask: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    value = mask.bool()
    if value.ndim > 2:
        value = value.reshape(*value.shape[:2], -1).all(dim=-1)
    return value[: shape[0], : shape[1]]


def _reconstruction_target(
    target: torch.Tensor,
    count: int,
    indices: torch.Tensor | None,
) -> torch.Tensor:
    if indices is not None:
        return target.index_select(1, indices.to(target.device).long())
    if target.shape[1] == count:
        return target
    stride = max(1, target.shape[1] // count)
    return target[:, ::stride][:, :count]


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def _pair_indices_from_batch(
    batch: dict[str, Any], device: torch.device
) -> torch.Tensor | None:
    pair_ids = list(batch.get("pair_id", ()))
    roles = list(batch.get("pair_role", ()))
    starts = batch.get("start_tick")
    grouped: dict[tuple[str, int], dict[str, int]] = {}
    for index, (pair_id, role) in enumerate(zip(pair_ids, roles, strict=True)):
        if not pair_id or role not in {"control", "treatment"}:
            continue
        grouped.setdefault((str(pair_id), int(starts[index])), {})[str(role)] = index
    values = [
        (roles_by_name["control"], roles_by_name["treatment"])
        for _identity, roles_by_name in sorted(grouped.items())
        if set(roles_by_name) >= {"control", "treatment"}
    ]
    if not values:
        return None
    return torch.tensor(values, dtype=torch.long, device=device)


def _suite_sort_key(entry: _SuiteEntry) -> bytes:
    return hashlib.sha256(
        f"{entry.task}:{entry.domain}:{entry.seed}:{entry.pair_id}:{entry.source_index}:{entry.start_tick}".encode()
    ).digest()


def _episode_choice_key(entry: _SuiteEntry) -> bytes:
    identity = (
        f"{entry.task}:{entry.domain}:{entry.pair_id}:{entry.start_tick}"
        if entry.pair_id
        else f"{entry.task}:{entry.domain}:{entry.seed}:{entry.source_index}:{entry.start_tick}"
    )
    return hashlib.sha256(identity.encode()).digest()


def _unit_sort_key(unit: tuple[_SuiteEntry, ...]) -> bytes:
    return hashlib.sha256(
        b"".join(_suite_sort_key(entry) for entry in unit)
    ).digest()


def _resolve_recorded_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else path.resolve()


def _latest_checkpoint(run: Path) -> Path:
    latest = run / "checkpoints" / "latest.json"
    if latest.exists():
        return run / "checkpoints" / json.loads(
            latest.read_text(encoding="utf-8")
        )["checkpoint"]
    checkpoints = sorted((run / "checkpoints").glob("step-*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint found in {run}")
    return checkpoints[-1]


def _autocast(config: ResearchConfig, device: torch.device):
    if config.training.dtype == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _arm_name(kind: str, objective: str) -> str:
    if kind == "rssm":
        return "rssm"
    if kind == "jepa":
        return "temporal-jepa"
    return {
        "dynamics": "dynamics-t",
        "causal": "causal-t",
        "counterfactual": "cf-t",
    }[objective]


def _default_objective(kind: str) -> str:
    return "counterfactual" if kind == "transformer" else "dynamics"


def _architecture_fingerprint(model: dict[str, Any]) -> str:
    value = dict(model)
    value.pop("kind", None)
    value.pop("objective", None)
    return _fingerprint(value)


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _flatten_numeric(value: Any, prefix: str) -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            output.update(_flatten_numeric(child, child_prefix))
    elif (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        excluded = (
            "/samples",
            "/rows",
            "/classes",
            "/dimension",
            "/global_step",
            "/run_seed",
        )
        if not prefix.endswith(excluded):
            output[prefix] = float(value)
    return output


def _confidence_interval(values: list[float]) -> dict[str, Any]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return {"mean": mean, "ci95": None, "values": values}
    critical = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
    }.get(len(values), 1.96)
    half_width = critical * statistics.stdev(values) / math.sqrt(len(values))
    return {
        "mean": mean,
        "ci95": [mean - half_width, mean + half_width],
        "values": values,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, dest="runs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument(
        "--pack", default=None, help="optional OOD Training Pack manifest"
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=2_000,
        help="frozen linear-probe AdamW steps (shorten only for CPU smoke tests)",
    )
    args = parser.parse_args(argv)
    reports = [
        evaluate_run(
            run,
            device_name=args.device,
            pack_path_override=args.pack,
            probe_steps=args.probe_steps,
        )
        for run in args.runs
    ]
    output = reports[0] if len(reports) == 1 else aggregate_runs(reports)
    print(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FixedEvaluationSuite",
    "aggregate_runs",
    "evaluate_run",
    "fixed_evaluation_suite",
]
