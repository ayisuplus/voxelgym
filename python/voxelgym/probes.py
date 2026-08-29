"""Deterministic frozen linear probes for world-model representations.

The world model is never updated by this module.  Callers provide one feature
matrix and target matrix per probe task for the train, validation, and test
splits.  Validation selects classification thresholds; those thresholds are
then locked for the test report.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Literal, Mapping

import torch
from torch import nn
import torch.nn.functional as F


ProbeKind = Literal["binary", "regression"]


@dataclass(frozen=True, slots=True)
class ProbeTask:
    """A frozen feature matrix and labels for one probe target."""

    features: torch.Tensor
    targets: torch.Tensor
    kind: ProbeKind


@dataclass(frozen=True, slots=True)
class ProbeProtocol:
    """The fixed protocol used by formal reports and shortened CPU tests."""

    steps: int = 2_000
    learning_rate: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 0.0
    seed: int = 0

    def validate(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0 or self.learning_rate <= 0:
            raise ValueError("probe steps, batch size, and learning rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("probe weight decay must be non-negative")


def fit_frozen_linear_probes(
    train: Mapping[str, ProbeTask],
    validation: Mapping[str, ProbeTask],
    test: Mapping[str, ProbeTask],
    *,
    protocol: ProbeProtocol | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Fit deterministic linear heads without back-propagating into features.

    Probe names and kinds must agree across all three splits.  Binary probes
    use BCE and report macro AUPRC/F1.  Regression probes use smooth-L1 for
    fitting and report MAE against the zero-change baseline.
    """

    resolved = protocol or ProbeProtocol()
    resolved.validate()
    names = tuple(sorted(train))
    if not names:
        return {"protocol": _protocol_record(resolved), "tasks": {}}
    if set(validation) != set(names) or set(test) != set(names):
        raise ValueError("probe task names must match across train/validation/test")

    probe_device = torch.device(device)
    reports: dict[str, Any] = {}
    for name in names:
        train_task = _validated_task(name, train[name])
        validation_task = _validated_task(name, validation[name])
        test_task = _validated_task(name, test[name])
        if {train_task.kind, validation_task.kind, test_task.kind} != {train_task.kind}:
            raise ValueError(f"probe kind differs across splits for {name!r}")
        feature_dim = int(train_task.features.shape[1])
        target_dim = int(train_task.targets.shape[1])
        for split, task in (("validation", validation_task), ("test", test_task)):
            if task.features.shape[1] != feature_dim or task.targets.shape[1] != target_dim:
                raise ValueError(f"{split} probe dimensions differ for {name!r}")

        head = _fit_head(name, train_task, resolved, probe_device)
        with torch.no_grad():
            validation_prediction = head(
                validation_task.features.detach().to(probe_device, dtype=torch.float32)
            ).cpu()
            test_prediction = head(
                test_task.features.detach().to(probe_device, dtype=torch.float32)
            ).cpu()
        if train_task.kind == "binary":
            thresholds = select_f1_thresholds(
                validation_prediction, validation_task.targets.detach().cpu()
            )
            reports[name] = {
                "kind": "binary",
                "feature_dim": feature_dim,
                "classes": target_dim,
                "thresholds": thresholds.tolist(),
                "validation": binary_classification_metrics(
                    validation_prediction,
                    validation_task.targets.detach().cpu(),
                    thresholds=thresholds,
                ),
                "test": binary_classification_metrics(
                    test_prediction,
                    test_task.targets.detach().cpu(),
                    thresholds=thresholds,
                ),
            }
        else:
            reports[name] = {
                "kind": "regression",
                "feature_dim": feature_dim,
                "targets": target_dim,
                "validation": regression_metrics(
                    validation_prediction, validation_task.targets.detach().cpu()
                ),
                "test": regression_metrics(
                    test_prediction, test_task.targets.detach().cpu()
                ),
            }
    return {"protocol": _protocol_record(resolved), "tasks": reports}


def binary_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    thresholds: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Return per-class and macro AP/F1 plus transparent trivial baselines."""

    scores, truth = _binary_matrices(logits, targets)
    probabilities = torch.sigmoid(scores)
    if thresholds is None:
        thresholds = torch.full((truth.shape[1],), 0.5, dtype=torch.float64)
    thresholds = thresholds.detach().cpu().to(torch.float64).reshape(-1)
    if thresholds.numel() != truth.shape[1]:
        raise ValueError("classification threshold count does not match classes")

    per_class: list[dict[str, Any]] = []
    for column in range(truth.shape[1]):
        label = truth[:, column]
        score = probabilities[:, column]
        predicted = score >= thresholds[column]
        positives = int(label.sum())
        negatives = int(label.numel() - positives)
        tp = int((predicted & label).sum())
        fp = int((predicted & ~label).sum())
        fn = int((~predicted & label).sum())
        f1 = _f1(tp, fp, fn)
        ap = _average_precision(score, label) if positives else None
        true_positive_rate = tp / positives if positives else None
        true_negative = int((~predicted & ~label).sum())
        true_negative_rate = true_negative / negatives if negatives else None
        balanced_accuracy = (
            (true_positive_rate + true_negative_rate) / 2
            if true_positive_rate is not None and true_negative_rate is not None
            else None
        )
        majority_value = positives >= negatives
        majority_tp = positives if majority_value else 0
        majority_fp = negatives if majority_value else 0
        majority_fn = 0 if majority_value else positives
        per_class.append(
            {
                "samples": int(label.numel()),
                "positives": positives,
                "prevalence": positives / max(int(label.numel()), 1),
                "auprc": ap,
                "f1": f1,
                "balanced_accuracy": balanced_accuracy,
                "threshold": float(thresholds[column]),
                "majority_f1": _f1(majority_tp, majority_fp, majority_fn),
                "zero_change_f1": 0.0 if positives else 1.0,
            }
        )
    return {
        "rows": int(truth.shape[0]),
        "classes": int(truth.shape[1]),
        "macro_auprc": _mean_present(item["auprc"] for item in per_class),
        "macro_f1": _mean_present(item["f1"] for item in per_class),
        "macro_balanced_accuracy": _mean_present(
            item["balanced_accuracy"] for item in per_class
        ),
        "majority_macro_f1": _mean_present(item["majority_f1"] for item in per_class),
        "zero_change_macro_f1": _mean_present(
            item["zero_change_f1"] for item in per_class
        ),
        "prevalence_macro_auprc": _mean_present(
            item["prevalence"] for item in per_class if item["positives"]
        ),
        "per_class": per_class,
    }


def select_f1_thresholds(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Select one threshold per class on validation, with 0.5 as tie-breaker."""

    scores, truth = _binary_matrices(logits, targets)
    probabilities = torch.sigmoid(scores)
    candidates = torch.linspace(0.01, 0.99, 99, dtype=torch.float64)
    candidates = candidates[torch.argsort(torch.abs(candidates - 0.5), stable=True)]
    selected: list[float] = []
    for column in range(truth.shape[1]):
        label = truth[:, column]
        best_threshold = 0.5
        best_f1 = -1.0
        for candidate in candidates:
            predicted = probabilities[:, column] >= candidate
            value = _f1(
                int((predicted & label).sum()),
                int((predicted & ~label).sum()),
                int((~predicted & label).sum()),
            )
            if value > best_f1:
                best_f1 = value
                best_threshold = float(candidate)
        selected.append(best_threshold)
    return torch.tensor(selected, dtype=torch.float64)


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    predicted = _matrix(prediction.detach().cpu().to(torch.float64))
    truth = _matrix(target.detach().cpu().to(torch.float64))
    if predicted.shape != truth.shape:
        raise ValueError("regression prediction and target shapes differ")
    absolute = torch.abs(predicted - truth)
    zero = torch.abs(truth)
    sign = torch.sign(predicted) == torch.sign(truth)
    return {
        "samples": int(truth.shape[0]),
        "targets": int(truth.shape[1]),
        "mae": float(absolute.mean()),
        "zero_change_mae": float(zero.mean()),
        "mae_to_zero_ratio": float(absolute.mean() / zero.mean().clamp_min(1e-12)),
        "sign_accuracy": float(sign.to(torch.float64).mean()),
    }


def effective_rank(features: torch.Tensor) -> dict[str, Any]:
    """Entropy effective rank of centered representation singular values."""

    value = _matrix(features.detach().cpu().to(torch.float64))
    if value.shape[0] < 2 or value.shape[1] == 0:
        return {"samples": int(value.shape[0]), "dimension": int(value.shape[1]), "effective_rank": None, "ratio": None}
    centered = value - value.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(value.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    singular = torch.sqrt(eigenvalues)
    total = singular.sum()
    if not bool(total > 0):
        rank = 0.0
    else:
        probability = singular / total
        entropy = -(probability * probability.clamp_min(1e-30).log()).sum()
        rank = float(torch.exp(entropy))
    return {
        "samples": int(value.shape[0]),
        "dimension": int(value.shape[1]),
        "effective_rank": rank,
        "ratio": rank / value.shape[1],
    }


def _fit_head(
    name: str,
    task: ProbeTask,
    protocol: ProbeProtocol,
    device: torch.device,
) -> nn.Linear:
    features = task.features.detach().to(device, dtype=torch.float32)
    targets = task.targets.detach().to(device, dtype=torch.float32)
    seed = protocol.seed + int.from_bytes(
        hashlib.sha256(name.encode("utf-8")).digest()[:4], "little"
    )
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        head = nn.Linear(features.shape[1], targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    # Evaluation entrypoints commonly run under torch.no_grad(); only the
    # ephemeral linear head needs autograd.
    with torch.enable_grad():
        for _ in range(protocol.steps):
            indices = torch.randint(
                0,
                features.shape[0],
                (min(protocol.batch_size, features.shape[0]),),
                generator=generator,
            ).to(device)
            prediction = head(features.index_select(0, indices))
            expected = targets.index_select(0, indices)
            loss = (
                F.binary_cross_entropy_with_logits(prediction, expected)
                if task.kind == "binary"
                else F.smooth_l1_loss(prediction, expected)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return head.eval()


def _validated_task(name: str, task: ProbeTask) -> ProbeTask:
    if task.kind not in {"binary", "regression"}:
        raise ValueError(f"unsupported probe kind for {name!r}: {task.kind!r}")
    features = _matrix(task.features)
    targets = _matrix(task.targets)
    if features.shape[0] != targets.shape[0] or features.shape[0] == 0:
        raise ValueError(f"probe {name!r} must contain aligned non-empty rows")
    if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
        raise ValueError(f"probe {name!r} contains non-finite values")
    if task.kind == "binary" and not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError(f"binary probe {name!r} targets must be zero or one")
    return ProbeTask(features=features, targets=targets, kind=task.kind)


def _binary_matrices(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scores = _matrix(logits.detach().cpu().to(torch.float64))
    truth_value = _matrix(targets.detach().cpu())
    if scores.shape != truth_value.shape:
        raise ValueError("binary prediction and target shapes differ")
    truth = truth_value.bool()
    return scores, truth


def _average_precision(scores: torch.Tensor, truth: torch.Tensor) -> float:
    order = torch.argsort(scores, descending=True, stable=True)
    ordered_truth = truth[order].to(torch.float64)
    ordered_scores = scores[order]
    cumulative_true = ordered_truth.cumsum(0)
    cumulative_false = (1.0 - ordered_truth).cumsum(0)
    # Precision-recall changes only after a complete tied-score group.  Grouping
    # ties makes AP independent of the input order and matches threshold-based
    # PR integration.
    group_end = torch.ones_like(ordered_truth, dtype=torch.bool)
    group_end[:-1] = ordered_scores[:-1] != ordered_scores[1:]
    true_at_threshold = cumulative_true[group_end]
    false_at_threshold = cumulative_false[group_end]
    recall = true_at_threshold / ordered_truth.sum().clamp_min(1.0)
    precision = true_at_threshold / (true_at_threshold + false_at_threshold)
    previous_recall = torch.cat((recall.new_zeros(1), recall[:-1]))
    return float(((recall - previous_recall) * precision).sum())


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def _matrix(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(-1, 1) if value.ndim == 1 else value.reshape(value.shape[0], -1)


def _mean_present(values: Any) -> float | None:
    present = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(present) / len(present) if present else None


def _protocol_record(protocol: ProbeProtocol) -> dict[str, Any]:
    return {
        "optimizer": "AdamW",
        "steps": protocol.steps,
        "learning_rate": protocol.learning_rate,
        "batch_size": protocol.batch_size,
        "weight_decay": protocol.weight_decay,
        "seed": protocol.seed,
        "threshold_selection": "validation-max-f1",
        "test_thresholds": "locked",
    }


__all__ = [
    "ProbeProtocol",
    "ProbeTask",
    "binary_classification_metrics",
    "effective_rank",
    "fit_frozen_linear_probes",
    "regression_metrics",
    "select_f1_thresholds",
]
