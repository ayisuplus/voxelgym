from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from voxelgym.config import ModelConfig, ResearchConfig
from voxelgym.models import (
    RSSMLitePack,
    TemporalJEPA,
    build_model,
    model_parameter_counts,
)


def _config(kind: str, objective: str | None = None) -> ModelConfig:
    return ModelConfig(
        kind=kind,
        objective=objective,
        latent=16,
        hidden=8,
        d_model=16,
        layers=1,
        heads=4,
        mlp_dim=32,
        context=4,
        horizons=(1,),
        mask_steps=1,
        intervention_features=4,
        modalities=("rgb", "action"),
    )


def _batch(*, paired: bool = False) -> dict[str, object]:
    torch.manual_seed(11)
    batch_size, transitions, states = 2, 4, 5
    batch: dict[str, object] = {
        "rgb": torch.randint(
            0, 256, (batch_size, states, 12, 12, 3), dtype=torch.uint8
        ),
        "action": torch.zeros(batch_size, transitions, 10, dtype=torch.long),
        "intervention_kind": torch.zeros(batch_size, transitions, 5),
        "intervention_params": torch.zeros(batch_size, transitions, 5, 4),
        "counterfactual_mask": torch.zeros(batch_size, 1, dtype=torch.bool),
        "pair_id": ["", ""],
        "pair_role": ["", ""],
        "start_tick": torch.tensor([0, 0]),
    }
    if paired:
        batch["counterfactual_mask"] = torch.ones(batch_size, 1, dtype=torch.bool)
        batch["pair_id"] = ["pair-1", "pair-1"]
        batch["pair_role"] = ["control", "treatment"]
        batch["intervention_kind"][1, 0, 0] = 1.0
        batch["intervention_params"][1, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    return batch


def test_model_config_resolves_legacy_objective_and_rejects_invalid_combinations():
    assert ModelConfig().objective == "counterfactual"
    assert ModelConfig(kind="rssm").objective == "dynamics"
    assert ModelConfig(kind="jepa").objective == "dynamics"
    assert ResearchConfig.from_dict({"model": {"kind": "transformer"}}).model.objective == "counterfactual"
    with pytest.raises(ValueError, match="only supports"):
        ModelConfig(kind="jepa", objective="causal").validate()
    with pytest.raises(ValueError, match="even microbatch"):
        ResearchConfig.from_dict(
            {
                "model": {
                    "kind": "transformer",
                    "objective": "counterfactual",
                },
                "training": {"microbatch": 1},
            }
        )
    with pytest.raises(ValueError, match="fixed at 50/30/20"):
        ResearchConfig.from_dict(
            {
                "dataset": {
                    "expert_fraction": 0.4,
                    "mixed_fraction": 0.4,
                    "paired_fraction": 0.2,
                }
            }
        )


def test_three_transformer_objectives_have_identical_initial_parameters():
    states = []
    for objective in ("dynamics", "causal", "counterfactual"):
        torch.manual_seed(19)
        model = build_model(
            _config("transformer", objective),
            event_classes=3,
            delta_classes=2,
            edge_classes=4,
        )
        states.append(model.state_dict())
    assert states[0].keys() == states[1].keys() == states[2].keys()
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key])
        assert torch.equal(states[0][key], states[2][key])


@pytest.mark.parametrize(
    ("objective", "required", "inactive"),
    [
        ("dynamics", "latent", "event"),
        ("causal", "causal_edge", "counterfactual_propagation"),
        ("counterfactual", "counterfactual_propagation", None),
    ],
)
def test_transformer_arm_forward_contract_and_backward(objective, required, inactive):
    model = build_model(
        _config("transformer", objective),
        event_classes=3,
        delta_classes=2,
        edge_classes=4,
    )
    output = model(_batch(paired=True), visible_prefix=1)
    assert required in output["active_losses"]
    if inactive is not None:
        assert inactive not in output["active_losses"]
    assert output["state_latent"].shape == (2, 5, 16)
    assert output["control_latent"].shape == (2, 4, 16)
    assert output["probe_latent"].shape == (2, 4, 16)
    assert output["horizon_latent"][1].shape == (2, 4, 16)
    assert output["reward"].shape == (2, 4)
    assert output["causal_edge"].shape == (2, 4, 4)
    assert output["depth"].shape == (2, 1, 32, 32)
    output["selected_horizon_latent"].square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.temporal.parameters())


def test_masked_prediction_ignores_future_views_but_uses_action_and_intervention():
    torch.manual_seed(23)
    model = build_model(
        _config("transformer", "counterfactual"),
        event_classes=1,
        delta_classes=1,
        edge_classes=1,
    ).eval()
    original = _batch()
    changed_future = deepcopy(original)
    changed_future["rgb"][:, 1:] = 255 - changed_future["rgb"][:, 1:]
    with torch.no_grad():
        baseline = model(original, visible_prefix=1)["selected_horizon_latent"]
        no_leak = model(changed_future, visible_prefix=1)["selected_horizon_latent"]
    assert torch.equal(baseline, no_leak)

    changed_action = deepcopy(original)
    changed_action["action"][:, 0, 0] = 1
    changed_intervention = deepcopy(original)
    changed_intervention["intervention_kind"][:, 0, 2] = 1.0
    changed_intervention["intervention_params"][:, 0, 2, 0] = 2.0
    with torch.no_grad():
        action_prediction = model(changed_action, visible_prefix=1)["selected_horizon_latent"]
        intervention_prediction = model(changed_intervention, visible_prefix=1)["selected_horizon_latent"]
    assert not torch.allclose(baseline, action_prediction)
    assert not torch.allclose(baseline, intervention_prediction)


def test_named_horizon_uses_only_last_visible_boundary():
    model = build_model(
        _config("transformer", "dynamics"),
        event_classes=1,
        delta_classes=1,
    )
    output = model(_batch(), visible_prefix=torch.tensor([2, 3]))
    mask = output["horizon_mask"][1]
    assert mask.sum(dim=1).tolist() == [1, 1]
    assert mask[0].nonzero().item() == 1
    assert mask[1].nonzero().item() == 2


def test_cached_sensor_frame_is_hidden_as_one_group():
    torch.manual_seed(29)
    model = build_model(
        _config("transformer", "dynamics"),
        event_classes=1,
        delta_classes=1,
    ).eval()
    batch = _batch()
    batch["render_sample_id"] = torch.tensor(
        [[0, 1, 1, 2, 3], [0, 1, 1, 2, 3]]
    )
    changed = deepcopy(batch)
    changed["rgb"][:, 1].fill_(255)
    with torch.no_grad():
        first = model(batch, visible_prefix=2)
        second = model(changed, visible_prefix=2)
    assert not first["sensor_group_visibility"]["rgb"][:, 1].any()
    assert torch.equal(
        first["selected_horizon_latent"], second["selected_horizon_latent"]
    )


def test_counterfactual_heads_use_paired_predictions_and_never_post_branch_views():
    torch.manual_seed(29)
    model = build_model(
        _config("transformer", "counterfactual"),
        event_classes=1,
        delta_classes=1,
        edge_classes=1,
    ).eval()
    batch = _batch(paired=True)
    changed = deepcopy(batch)
    changed["rgb"][:, 1:] = torch.randint_like(changed["rgb"][:, 1:], 0, 256)
    with torch.no_grad():
        first = model(batch, visible_prefix=torch.tensor([4, 4]))
        second = model(changed, visible_prefix=torch.tensor([4, 4]))
    assert first["visible_prefix"].tolist() == [1, 1]
    assert first["cf_pair_indices"].tolist() == [[0, 1]]
    assert first["counterfactual_latent_effect"].shape == (1, 1, 16)
    assert first["counterfactual_propagation"].shape == (1, 1)
    assert torch.equal(
        first["counterfactual_latent_effect"],
        second["counterfactual_latent_effect"],
    )


def test_rssm_uses_sixty_five_state_style_next_boundary_contract():
    model = build_model(_config("rssm"), event_classes=0, delta_classes=0)
    assert isinstance(model, RSSMLitePack)
    output = model(_batch())
    assert output["state_latent"].shape == (2, 5, 16)
    assert output["next_latent"].shape == output["next_target_latent"].shape == (2, 4, 16)
    assert output["probe_latent"].shape == (2, 4, 8)
    loss = F.mse_loss(output["next_latent"], output["next_target_latent"])
    loss.backward()
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_jepa_has_state_only_frozen_teacher_and_ema_update():
    model = build_model(_config("jepa"), event_classes=0, delta_classes=0)
    assert isinstance(model, TemporalJEPA)
    output = model(_batch(), visible_prefix=1)
    assert output["jepa_prediction"].shape == output["jepa_target"].shape == (2, 1, 16)
    loss = 1.0 - F.cosine_similarity(output["jepa_prediction"], output["jepa_target"], dim=-1).mean()
    loss = loss + 0.1 * output["jepa_variance_loss"]
    loss.backward()
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    before = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.online_encoder.parameters()).add_(1.0)
    model.ema_update()
    after = next(model.target_encoder.parameters()).detach()
    assert not torch.equal(before, after)


@pytest.mark.ml
def test_formal_transformer_and_jepa_parameter_budgets():
    transformer = build_model(
        ModelConfig(), event_classes=64, delta_classes=32, edge_classes=128
    )
    jepa = build_model(
        ModelConfig(kind="jepa"), event_classes=64, delta_classes=32, edge_classes=128
    )
    transformer_counts = model_parameter_counts(transformer)
    jepa_counts = model_parameter_counts(jepa)
    assert 100_000_000 <= transformer_counts["trainable"] <= 110_000_000
    assert 95_000_000 <= jepa_counts["trainable"] <= 105_000_000
    assert jepa_counts["checkpoint"] > jepa_counts["trainable"]
