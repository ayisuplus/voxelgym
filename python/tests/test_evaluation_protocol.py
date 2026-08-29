from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from voxelgym.evaluate import (
    _PairPreservingBatchSampler,
    _SuiteEntry,
    _UniquePairAccumulator,
    aggregate_runs,
)
from voxelgym.probes import (
    ProbeProtocol,
    ProbeTask,
    binary_classification_metrics,
    effective_rank,
    fit_frozen_linear_probes,
)


def test_binary_metrics_are_macro_threshold_based_and_tie_stable():
    logits = torch.tensor([[3.0, 0.0], [-3.0, 0.0], [2.0, 0.0], [-2.0, 0.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    report = binary_classification_metrics(logits, target)
    assert report["per_class"][0]["auprc"] == 1.0
    # All second-class scores are tied, so AP is prevalence and cannot depend
    # on the incoming row order.
    assert report["per_class"][1]["auprc"] == 0.5
    reversed_report = binary_classification_metrics(logits.flip(0), target.flip(0))
    assert reversed_report["per_class"][1]["auprc"] == 0.5
    assert "macro_balanced_accuracy" in report


def test_frozen_probe_uses_validation_thresholds_and_never_grads_features():
    feature = torch.tensor(
        [[-2.0], [-1.0], [1.0], [2.0]], requires_grad=True
    )
    binary = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    regression = feature.detach() * 2.0
    split = {
        "event": ProbeTask(feature, binary, "binary"),
        "reward": ProbeTask(feature, regression, "regression"),
    }
    with torch.no_grad():
        report = fit_frozen_linear_probes(
            split,
            split,
            split,
            protocol=ProbeProtocol(steps=80, learning_rate=0.05, batch_size=4),
        )
    event = report["tasks"]["event"]
    assert event["test"]["per_class"][0]["threshold"] == event["thresholds"][0]
    assert event["test"]["macro_auprc"] == 1.0
    assert report["tasks"]["reward"]["test"]["mae"] < 0.25
    assert feature.grad is None


def test_effective_rank_detects_full_and_collapsed_representations():
    full = torch.cat((torch.eye(4), -torch.eye(4)))
    full_report = effective_rank(full)
    assert full_report["effective_rank"] == pytest.approx(4.0)
    collapsed = torch.ones(8, 4)
    collapsed_report = effective_rank(collapsed)
    assert collapsed_report["effective_rank"] == 0.0


def _report(arm: str, seed: int, ratio: float) -> dict:
    return {
        "format": "voxelgym.evaluation",
        "format_version": 2,
        "arm": arm,
        "run_seed": seed,
        "model_fingerprint": f"model-{arm}",
        "architecture_fingerprint": "shared-transformer",
        "dataset_fingerprint": "dataset",
        "evaluation_pack_fingerprint": "pack",
        "evaluation_suite_fingerprint": "suite",
        "training_pack_fingerprint": "training-pack",
        "global_step": 25_000,
        "horizons": {
            "1": {
                "samples": 10,
                "model_mse": ratio,
                "copy_last_mse": 1.0,
                "model_to_copy_ratio": ratio,
            }
        },
    }


def test_aggregate_validates_identity_and_reports_seed_paired_differences():
    reports = []
    for seed in (0, 1, 2):
        reports.extend(
            (
                _report("dynamics-t", seed, 0.9 - seed * 0.01),
                _report("causal-t", seed, 0.8 - seed * 0.01),
                _report("cf-t", seed, 0.7 - seed * 0.01),
            )
        )
    aggregate = aggregate_runs(reports)
    causal = aggregate["paired_differences"]["causal_minus_dynamics"]
    difference = causal["metrics"]["horizons/1/model_to_copy_ratio"]
    assert difference["mean"] == pytest.approx(-0.1)
    assert difference["ci95"][0] == pytest.approx(-0.1)
    assert set(aggregate["arms"]) == {"dynamics-t", "causal-t", "cf-t"}

    mismatched = deepcopy(reports)
    mismatched[-1]["evaluation_suite_fingerprint"] = "different"
    with pytest.raises(ValueError, match="comparison identity mismatch"):
        aggregate_runs(mismatched)


def test_counterfactual_metrics_count_each_pair_boundary_once():
    accumulator = _UniquePairAccumulator((1, 4))
    target_latent = torch.zeros(2, 6, 3)
    target_latent[1, 1] = 1.0
    target_latent[1, 4] = 2.0
    output = {
        "cf_pair_indices": torch.tensor([[0, 1]]),
        "counterfactual_propagation": torch.tensor([[4.0, -4.0]]),
        "counterfactual_reward_delta": torch.tensor([[2.0, -1.0]]),
        "counterfactual_latent_effect": torch.tensor(
            [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]
        ),
        "target_latent": target_latent,
        "visible_prefix": torch.tensor([1, 1]),
    }
    moved = {
        "counterfactual_propagated": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "counterfactual_reward_delta": torch.tensor([[2.0, -1.0], [2.0, -1.0]]),
        "counterfactual_mask": torch.ones(2, 2, dtype=torch.bool),
    }
    original = {
        "pair_id": ["pair-1", "pair-1"],
        "start_tick": torch.tensor([12, 12]),
    }
    accumulator.add(output, moved, original)
    accumulator.add(output, moved, original)
    report = accumulator.finish()
    assert report["unique_pairs"] == 1
    assert report["horizons"]["1"]["unique_pairs"] == 1
    assert report["horizons"]["1"]["reward_delta"]["mae"] == 0.0
    assert report["horizons"]["4"]["latent_effect"]["mse"] == 0.0


def test_eval_batches_never_split_aligned_pair_branches():
    def entry(index: int, pair: str = "", role: str = "") -> _SuiteEntry:
        return _SuiteEntry(index, index, "task", "domain", index, pair, role, 12)

    entries = (
        entry(0),
        entry(1, "p1", "control"),
        entry(2, "p1", "treatment"),
        entry(3),
        entry(4),
    )
    batches = list(_PairPreservingBatchSampler(entries, batch_size=2))
    assert [1, 2] in batches
    assert all(not ({1, 2} & set(batch)) or {1, 2} <= set(batch) for batch in batches)
