"""Action-conditioned world models for Training Pack v1.

Agent View tensors use ``[batch, states, ...]`` (normally 65); ``action`` and
declared intervention tensors use ``[batch, transitions, ...]`` (normally 64).
Prediction paths never read supervision fields. Future views are masked while
the actions and declared interventions leading to them remain visible.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


ACTION_CARDINALITIES = (5, 2, 2, 24, 9, 2, 2, 2, 9, 8)
INTERVENTION_KINDS = (
    "set_cell",
    "teleport_agent",
    "set_agent_velocity",
    "give_item",
    "swap_to_hotbar",
)

TRANSFORMER_LOSSES: dict[str, tuple[str, ...]] = {
    "dynamics": ("latent", "depth", "seg", "reward", "terminal"),
    "causal": (
        "latent", "depth", "seg", "reward", "terminal", "event", "delta",
        "time_to_event", "causal_edge",
    ),
    "counterfactual": (
        "latent", "depth", "seg", "reward", "terminal", "event", "delta",
        "time_to_event", "causal_edge", "counterfactual_latent_effect",
        "counterfactual_propagation", "counterfactual_reward_delta",
    ),
}


class SharedImageEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(F.interpolate(value, size=(32, 32), mode="bilinear", align_corners=False))


class LidarEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class VoxelEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(4096, 16)
        self.net = nn.Sequential(
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(64, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(torch.bitwise_and(value.long(), 0xFFF))
        return self.net(embedded.permute(0, 4, 1, 2, 3))


class ActionEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        width = max(8, output_dim // len(ACTION_CARDINALITIES))
        self.embeddings = nn.ModuleList(nn.Embedding(size, width) for size in ACTION_CARDINALITIES)
        self.projection = nn.Linear(width * len(ACTION_CARDINALITIES), output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != len(ACTION_CARDINALITIES):
            raise ValueError(f"action must have {len(ACTION_CARDINALITIES)} fields")
        parts = [embedding(value[:, index].long()) for index, embedding in enumerate(self.embeddings)]
        return self.projection(torch.cat(parts, dim=-1))


class InterventionEncoder(nn.Module):
    """Encode the five declared external-intervention kinds, separately from action."""

    def __init__(self, output_dim: int, parameter_features: int):
        super().__init__()
        self.parameter_features = int(parameter_features)
        self.kind = nn.Parameter(torch.empty(len(INTERVENTION_KINDS), output_dim))
        self.none = nn.Parameter(torch.zeros(output_dim))
        self.parameter_projection = nn.Sequential(
            nn.Linear(parameter_features, output_dim), nn.SiLU(), nn.Linear(output_dim, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)
        nn.init.normal_(self.kind, std=0.02)

    def forward(self, batch: dict[str, Any], *, batch_size: int, transitions: int, device: torch.device) -> torch.Tensor:
        kind_value = batch.get("intervention_kind")
        parameter_value = batch.get("intervention_params")
        if kind_value is None:
            dense = batch.get("intervention")
            if dense is None:
                dense = batch.get("external_intervention")
            if dense is not None:
                kind_value, parameter_value = self._from_dense(dense)
        if kind_value is None:
            return self.none.view(1, 1, -1).expand(batch_size, transitions, -1)
        if not torch.is_tensor(kind_value) or kind_value.shape[:2] != (batch_size, transitions):
            raise ValueError("intervention_kind must align with action [B,T]")

        weights = kind_value.to(device=device, dtype=self.kind.dtype)
        if weights.ndim == 2:
            weights = F.one_hot(
                weights.long().clamp(0, len(INTERVENTION_KINDS)),
                num_classes=len(INTERVENTION_KINDS) + 1,
            )[..., 1:].to(self.kind.dtype)
        if weights.ndim != 3 or weights.shape[-1] != len(INTERVENTION_KINDS):
            raise ValueError("intervention_kind must have shape [B,T,5]")

        if parameter_value is None:
            parameters = weights.new_zeros(batch_size, transitions, len(INTERVENTION_KINDS), self.parameter_features)
        else:
            if not torch.is_tensor(parameter_value):
                raise TypeError("intervention_params must be a tensor")
            parameters = parameter_value.to(device=device, dtype=self.kind.dtype)
            if parameters.ndim == 3:
                parameters = parameters.unsqueeze(-2).expand(-1, -1, len(INTERVENTION_KINDS), -1)
            if parameters.shape[:3] != (batch_size, transitions, len(INTERVENTION_KINDS)):
                raise ValueError("intervention_params must have shape [B,T,5,P]")
            parameters = _pad_or_trim(parameters, self.parameter_features)

        nonnegative = weights.clamp_min(0.0)
        total = nonnegative.sum(dim=-1, keepdim=True)
        tokens = self.kind.view(1, 1, len(INTERVENTION_KINDS), -1) + self.parameter_projection(parameters)
        encoded = (tokens * nonnegative.unsqueeze(-1)).sum(dim=-2) / total.clamp_min(1.0)
        encoded = torch.where(total > 0, encoded, self.none.view(1, 1, -1))
        return self.norm(encoded)

    @staticmethod
    def _from_dense(value: Any) -> tuple[Any, Any]:
        if not torch.is_tensor(value) or value.ndim < 3:
            raise TypeError("intervention must be a [B,T,F] tensor")
        return value[..., 0], value[..., 1:]


class InventoryEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.item = nn.Embedding(65536, 32)
        self.projection = nn.Sequential(nn.Linear(33, output_dim), nn.SiLU())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        items = self.item(value[..., 0].long().clamp(0, 65535))
        counts = torch.log1p(value[..., 1].float().clamp_min(0.0)).unsqueeze(-1) / 8.0
        return self.projection(torch.cat((items, counts), dim=-1)).mean(dim=1)


class MultimodalEncoder(nn.Module):
    """Encode Agent View state only; controls are excluded even from legacy lists."""

    def __init__(self, output_dim: int, modalities: tuple[str, ...]):
        super().__init__()
        self.output_dim = int(output_dim)
        self.modalities = tuple(name for name in modalities if name not in {"action", "intervention", "external_intervention"})
        if not self.modalities:
            raise ValueError("MultimodalEncoder needs at least one state modality")
        self.image = SharedImageEncoder(output_dim)
        self.lidar = LidarEncoder(output_dim)
        self.voxel = VoxelEncoder(output_dim)
        self.pose = nn.Sequential(nn.Linear(6, output_dim), nn.SiLU(), nn.Linear(output_dim, output_dim))
        self.inventory = InventoryEncoder(output_dim)
        self.missing = nn.ParameterDict({name: nn.Parameter(torch.zeros(output_dim)) for name in self.modalities})
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * len(self.modalities), output_dim), nn.LayerNorm(output_dim), nn.SiLU()
        )

    def forward(
        self,
        batch: dict[str, Any],
        *,
        modality_visibility: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        reference = next((batch[name] for name in self.modalities if torch.is_tensor(batch.get(name))), None)
        if reference is None:
            raise ValueError("batch contains none of the configured state modalities")
        batch_size, states = reference.shape[:2]
        flat_count = batch_size * states
        encoded: list[torch.Tensor] = []
        for modality in self.modalities:
            tensor = batch.get(modality)
            if tensor is None:
                value = self.missing[modality].view(1, -1).expand(flat_count, -1)
            else:
                if not torch.is_tensor(tensor) or tensor.shape[:2] != (batch_size, states):
                    raise ValueError(f"state modality {modality!r} must align as [B,S,...]")
                if modality == "rgb":
                    image = tensor.reshape(flat_count, *tensor.shape[2:]).permute(0, 3, 1, 2).float() / 255.0
                    value = self.image(image)
                elif modality == "depth":
                    scalar = tensor.reshape(flat_count, 1, *tensor.shape[2:]).float()
                    value = self.image((torch.log1p(scalar.clamp_min(0.0)) / 5.0).expand(-1, 3, -1, -1))
                elif modality == "normals":
                    image = tensor.reshape(flat_count, *tensor.shape[2:]).permute(0, 3, 1, 2).float()
                    value = self.image(image)
                elif modality == "lidar_range":
                    lidar = tensor.reshape(flat_count, 1, *tensor.shape[2:]).float()
                    value = self.lidar(torch.log1p(lidar.clamp_min(0.0)) / 5.0)
                elif modality == "voxels":
                    value = self.voxel(tensor.reshape(flat_count, *tensor.shape[2:]))
                elif modality == "pose":
                    value = self.pose(tensor.reshape(flat_count, -1).float())
                elif modality == "inventory":
                    value = self.inventory(tensor.reshape(flat_count, *tensor.shape[2:]))
                else:
                    raise ValueError(f"unsupported model state modality {modality!r}")
                if modality_visibility is not None and modality in modality_visibility:
                    visible = modality_visibility[modality]
                    if visible.shape != (batch_size, states):
                        raise ValueError(
                            f"visibility for {modality!r} must have shape [B,S]"
                        )
                    missing = self.missing[modality].view(1, -1).expand(flat_count, -1)
                    value = torch.where(
                        visible.to(device=value.device, dtype=torch.bool).reshape(
                            flat_count, 1
                        ),
                        value,
                        missing,
                    )
            encoded.append(value)
        return self.fusion(torch.cat(encoded, dim=-1)).reshape(batch_size, states, self.output_dim)


class ControlEncoder(nn.Module):
    """Keep agent actions and external interventions as distinct authorities."""

    def __init__(self, output_dim: int, intervention_features: int):
        super().__init__()
        self.output_dim = int(output_dim)
        self.action = ActionEncoder(output_dim)
        self.intervention = InterventionEncoder(output_dim, intervention_features)
        self.fusion = nn.Sequential(nn.Linear(output_dim * 2, output_dim), nn.LayerNorm(output_dim), nn.SiLU())

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        action = batch.get("action")
        if not torch.is_tensor(action) or action.ndim != 3:
            raise ValueError("batch action must have shape [B,T,10]")
        batch_size, transitions = action.shape[:2]
        action_token = self.action(action.reshape(-1, action.shape[-1])).reshape(batch_size, transitions, self.output_dim)
        intervention_token = self.intervention(batch, batch_size=batch_size, transitions=transitions, device=action.device)
        return self.fusion(torch.cat((action_token, intervention_token), dim=-1))


class SpatialDecoder(nn.Module):
    def __init__(self, input_dim: int, output_channels: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 128 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 96, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(64, output_channels, 4, stride=2, padding=1),
        )
        self.output_channels = output_channels

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape[:-1]
        decoded = self.net(self.fc(value).reshape(-1, 128, 4, 4))
        return decoded.reshape(*shape, self.output_channels, 32, 32)


class PredictionHeads(nn.Module):
    def __init__(self, hidden_dim: int, event_classes: int, delta_classes: int, edge_classes: int):
        super().__init__()
        self.depth = SpatialDecoder(hidden_dim, 1)
        self.seg = SpatialDecoder(hidden_dim, 64)
        self.reward = nn.Linear(hidden_dim, 1)
        self.terminal = nn.Linear(hidden_dim, 1)
        self.event = nn.Linear(hidden_dim, event_classes) if event_classes else None
        self.delta = nn.Linear(hidden_dim, delta_classes) if delta_classes else None
        self.time_to_event = nn.Linear(hidden_dim, 1)
        self.causal_edge = nn.Linear(hidden_dim, edge_classes) if edge_classes else None

    def forward(self, hidden: torch.Tensor, *, reconstruction_stride: int = 4) -> dict[str, torch.Tensor]:
        indices = torch.arange(0, hidden.shape[1], reconstruction_stride, device=hidden.device)
        reconstruction = hidden.index_select(1, indices)
        return {
            "depth": self.depth(reconstruction).squeeze(2),
            "seg": self.seg(reconstruction),
            "reconstruction_indices": indices,
            "reward": self.reward(hidden).squeeze(-1),
            "terminal": self.terminal(hidden).squeeze(-1),
            "event": self.event(hidden) if self.event is not None else hidden.new_empty(*hidden.shape[:-1], 0),
            "delta": self.delta(hidden) if self.delta is not None else hidden.new_empty(*hidden.shape[:-1], 0),
            "time_to_event": self.time_to_event(hidden).squeeze(-1),
            "causal_edge": self.causal_edge(hidden) if self.causal_edge is not None else hidden.new_empty(*hidden.shape[:-1], 0),
        }


class TemporalBackbone(nn.Module):
    """Causal state/control interleaving with masked future state tokens."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(config.d_model))
        self.state_type = nn.Parameter(torch.zeros(config.d_model))
        self.control_type = nn.Parameter(torch.zeros(config.d_model))
        self.position = nn.Parameter(torch.zeros(1, config.context * 2 + 1, config.d_model))
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=config.d_model, nhead=config.heads, dim_feedforward=config.mlp_dim,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(config.layers)
        )
        self.norm = nn.LayerNorm(config.d_model)
        for parameter in (self.mask_token, self.state_type, self.control_type, self.position):
            nn.init.normal_(parameter, std=0.02)

    def forward(self, state: torch.Tensor, control: torch.Tensor, visible_prefix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, states, width = state.shape
        transitions = control.shape[1]
        if control.shape[0] != batch_size or control.shape[2] != width:
            raise ValueError("state and control encodings must share batch and width")
        if transitions not in {states - 1, states}:
            raise ValueError("Training Pack windows require S=T+1 (legacy S=T is compatible)")
        indices = torch.arange(states, device=state.device)
        state_mask = indices.view(1, -1) >= visible_prefix.view(-1, 1)
        masked_state = torch.where(state_mask.unsqueeze(-1), self.mask_token.view(1, 1, -1), state)
        masked_state = masked_state + self.state_type.view(1, 1, -1)
        typed_control = control + self.control_type.view(1, 1, -1)
        paired = torch.stack((masked_state[:, :transitions], typed_control), dim=2)
        sequence = paired.reshape(batch_size, transitions * 2, width)
        if states > transitions:
            sequence = torch.cat((sequence, masked_state[:, -1:]), dim=1)
        if sequence.shape[1] > self.position.shape[1]:
            raise ValueError("encoded sequence exceeds model.context capacity")
        sequence = sequence + self.position[:, : sequence.shape[1]]
        causal_mask = torch.triu(torch.ones(sequence.shape[1], sequence.shape[1], device=sequence.device, dtype=torch.bool), diagonal=1)
        hidden = sequence
        for layer in self.layers:
            if self.training and torch.is_grad_enabled():
                hidden = checkpoint(
                    lambda value, block=layer: block(value, src_mask=causal_mask, is_causal=True),
                    hidden, use_reentrant=False, preserve_rng_state=True,
                )
            else:
                hidden = layer(hidden, src_mask=causal_mask, is_causal=True)
        hidden = self.norm(hidden)
        return hidden[:, 0::2], hidden[:, 1::2], state_mask


class CausalTransformer(nn.Module):
    """One fixed architecture shared by Dynamics-T, Causal-T, and CF-T."""

    def __init__(self, config: ModelConfig, event_classes: int, delta_classes: int, edge_classes: int = 1):
        super().__init__()
        self.config = config
        self.objective = str(config.objective)
        self.active_losses = TRANSFORMER_LOSSES[self.objective]
        self.state_encoder = MultimodalEncoder(config.d_model, config.state_modalities)
        self.control_encoder = ControlEncoder(config.d_model, config.intervention_features)
        self.temporal = TemporalBackbone(config)
        self.horizon_heads = nn.ModuleDict({str(h): nn.Linear(config.d_model, config.d_model) for h in config.horizons})
        self.heads = PredictionHeads(config.d_model, event_classes, delta_classes, edge_classes)
        cf_hidden = max(32, config.d_model // 2)
        self.cf_propagation = nn.Sequential(nn.Linear(config.d_model, cf_hidden), nn.GELU(), nn.Linear(cf_hidden, 1))
        self.cf_reward_delta = nn.Sequential(nn.Linear(config.d_model, cf_hidden), nn.GELU(), nn.Linear(cf_hidden, 1))

    @property
    def encoder(self) -> MultimodalEncoder:
        """Compatibility alias without duplicating the module in checkpoints."""

        return self.state_encoder

    def forward(self, batch: dict[str, Any], *, visible_prefix: int | torch.Tensor | None = None) -> dict[str, Any]:
        target_latent = self.state_encoder(batch)
        control_latent = self.control_encoder(batch)
        prefix = _resolve_visible_prefix(target_latent, self.config, visible_prefix, training=self.training)
        if self.objective == "counterfactual":
            prefix = _counterfactual_prefix(batch, prefix)
        visibility = _sensor_group_visibility(batch, prefix, self.config.state_modalities)
        state_latent = (
            target_latent
            if visibility is None
            else self.state_encoder(batch, modality_visibility=visibility)
        )
        state_hidden, control_hidden, state_mask = self.temporal(state_latent, control_latent, prefix)
        horizon = _horizon_predictions(state_hidden, self.horizon_heads, self.config.horizons, prefix)
        transition_hidden = _transition_hidden(state_hidden, control_hidden, control_latent.shape[1])
        output: dict[str, Any] = self.heads(transition_hidden)
        output.update({
            "kind": "transformer", "objective": self.objective, "active_losses": self.active_losses,
            "state_latent": state_latent, "target_latent": target_latent.detach(),
            "control_latent": control_latent, "probe_latent": transition_hidden,
            "visible_prefix": prefix, "state_mask": state_mask,
            "sensor_group_visibility": visibility or {},
            "transition_mask": _transition_prediction_mask(prefix, states=state_latent.shape[1], transitions=control_latent.shape[1]),
            **horizon,
        })
        output.update(self._counterfactual_outputs(batch, horizon["selected_horizon_latent"], state_latent.device))
        if output["causal_edge"].shape[-1] == 1:
            output["causal_parent"] = output["causal_edge"].squeeze(-1)
        return output

    def predict_counterfactual(self, control_hidden: torch.Tensor, treatment_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        if control_hidden.shape != treatment_hidden.shape:
            raise ValueError("counterfactual branches must have aligned hidden tensors")
        effect = treatment_hidden - control_hidden
        return {
            "counterfactual_latent_effect": effect, "cf_probe_latent": effect,
            "counterfactual_propagation": self.cf_propagation(effect).squeeze(-1),
            "counterfactual_reward_delta": self.cf_reward_delta(effect).squeeze(-1),
        }

    def _counterfactual_outputs(self, batch: dict[str, Any], selected: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
        indices = _match_counterfactual_pairs(batch, device) if self.objective == "counterfactual" else torch.empty((0, 2), dtype=torch.long, device=device)
        if indices.numel():
            control = selected.index_select(0, indices[:, 0])
            treatment = selected.index_select(0, indices[:, 1])
        else:
            control = selected.new_empty((0, selected.shape[1], selected.shape[2]))
            treatment = control
        result = self.predict_counterfactual(control, treatment)
        result["cf_pair_indices"] = indices
        return result


class RSSMLitePack(nn.Module):
    """1024-latent/512-hidden BYOL RSSM over S boundary states and T controls."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.objective = "dynamics"
        self.active_losses = ("latent", "rgb")
        self.online_encoder = MultimodalEncoder(config.latent, config.state_modalities)
        self.target_encoder = deepcopy(self.online_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        self.control_encoder = ControlEncoder(config.latent, config.intervention_features)
        self.transition_input = nn.Sequential(nn.Linear(config.latent * 2, config.latent), nn.LayerNorm(config.latent), nn.SiLU())
        self.gru = nn.GRU(config.latent, config.hidden, batch_first=True)
        self.predictor = nn.Sequential(nn.Linear(config.hidden, config.latent), nn.LayerNorm(config.latent))
        self.decoder = SpatialDecoder(config.latent, 3)
        self.ema_init()

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        state = F.normalize(self.online_encoder(batch), dim=-1)
        control = self.control_encoder(batch)
        transitions = min(control.shape[1], state.shape[1] - 1)
        if transitions <= 0:
            raise ValueError("RSSM needs at least two states and one control")
        gru_input = self.transition_input(torch.cat((state[:, :transitions], control[:, :transitions]), dim=-1))
        hidden, _ = self.gru(gru_input)
        prediction = F.normalize(self.predictor(hidden), dim=-1)
        with torch.no_grad():
            target = F.normalize(self.target_encoder(batch), dim=-1)
        reconstruction_indices = torch.arange(0, state.shape[1], 4, device=state.device)
        return {
            "kind": "rssm", "objective": self.objective, "active_losses": self.active_losses,
            "latent": prediction, "next_latent": prediction, "state_latent": state,
            "target_latent": target, "next_target_latent": target[:, 1 : transitions + 1],
            "control_latent": control, "probe_latent": hidden,
            "transition_mask": torch.ones(prediction.shape[:2], dtype=torch.bool, device=prediction.device),
            "rgb_reconstruction": torch.sigmoid(self.decoder(state.index_select(1, reconstruction_indices))),
            "reconstruction_indices": reconstruction_indices,
        }

    @torch.no_grad()
    def ema_init(self) -> None:
        for target, online in zip(self.target_encoder.parameters(), self.online_encoder.parameters(), strict=True):
            target.copy_(online)

    @torch.no_grad()
    def ema_update(self, tau: float = 0.99) -> None:
        for target, online in zip(self.target_encoder.parameters(), self.online_encoder.parameters(), strict=True):
            target.mul_(tau).add_(online.detach(), alpha=1.0 - tau)


class TemporalJEPA(nn.Module):
    """Action/intervention-conditioned masked latent predictor with EMA targets."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.objective = "dynamics"
        self.active_losses = ("jepa", "variance")
        self.online_encoder = MultimodalEncoder(config.d_model, config.state_modalities)
        self.target_encoder = deepcopy(self.online_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        self.control_encoder = ControlEncoder(config.d_model, config.intervention_features)
        self.temporal = TemporalBackbone(config)
        self.horizon_heads = nn.ModuleDict({str(h): nn.Linear(config.d_model, config.d_model) for h in config.horizons})
        self.ema_init()

    def forward(self, batch: dict[str, Any], *, visible_prefix: int | torch.Tensor | None = None) -> dict[str, Any]:
        state = self.online_encoder(batch)
        control = self.control_encoder(batch)
        prefix = _resolve_visible_prefix(state, self.config, visible_prefix, training=self.training)
        visibility = _sensor_group_visibility(batch, prefix, self.config.state_modalities)
        if visibility is not None:
            state = self.online_encoder(batch, modality_visibility=visibility)
        state_hidden, control_hidden, state_mask = self.temporal(state, control, prefix)
        horizon = _horizon_predictions(state_hidden, self.horizon_heads, self.config.horizons, prefix)
        with torch.no_grad():
            target = self.target_encoder(batch)
        target_selected = F.normalize(_selected_targets(target, prefix, self.config.horizons), dim=-1)
        prediction = F.normalize(horizon["selected_horizon_latent"], dim=-1)
        feature_std = prediction.float().std(dim=(0, 1), unbiased=False)
        transition_hidden = _transition_hidden(state_hidden, control_hidden, control.shape[1])
        return {
            "kind": "jepa", "objective": self.objective, "active_losses": self.active_losses,
            "state_latent": state, "target_latent": target, "control_latent": control,
            "probe_latent": transition_hidden, "visible_prefix": prefix, "state_mask": state_mask,
            "sensor_group_visibility": visibility or {},
            "transition_mask": _transition_prediction_mask(prefix, states=state.shape[1], transitions=control.shape[1]),
            **horizon, "jepa_prediction": prediction, "jepa_target": target_selected,
            "jepa_feature_std": feature_std,
            "jepa_variance": prediction.float().var(dim=(0, 1), unbiased=False).mean(),
            "jepa_variance_loss": F.relu(1.0 - feature_std).mean(),
            "jepa_effective_rank": _effective_rank(prediction),
        }

    @torch.no_grad()
    def ema_init(self) -> None:
        for target, online in zip(self.target_encoder.parameters(), self.online_encoder.parameters(), strict=True):
            target.copy_(online)

    @torch.no_grad()
    def ema_update(self, tau: float | None = None) -> None:
        momentum = self.config.jepa_tau if tau is None else float(tau)
        for target, online in zip(self.target_encoder.parameters(), self.online_encoder.parameters(), strict=True):
            target.mul_(momentum).add_(online.detach(), alpha=1.0 - momentum)


def build_model(config: ModelConfig, *, event_classes: int, delta_classes: int, edge_classes: int = 1) -> nn.Module:
    config.validate()
    if config.kind == "transformer":
        return CausalTransformer(config, event_classes, delta_classes, edge_classes=edge_classes)
    if config.kind == "jepa":
        return TemporalJEPA(config)
    return RSSMLitePack(config)


def parameter_count(model: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if not trainable_only or parameter.requires_grad)


def model_parameter_counts(model: nn.Module) -> dict[str, int]:
    return {"trainable": parameter_count(model, trainable_only=True), "checkpoint": parameter_count(model)}


def _pad_or_trim(value: torch.Tensor, features: int) -> torch.Tensor:
    if value.shape[-1] == features:
        return value
    if value.shape[-1] > features:
        return value[..., :features]
    return F.pad(value, (0, features - value.shape[-1]))


def _resolve_visible_prefix(state: torch.Tensor, config: ModelConfig, visible_prefix: int | torch.Tensor | None, *, training: bool) -> torch.Tensor:
    batch_size, states = state.shape[:2]
    if states < 2:
        raise ValueError("masked temporal models need at least two boundary states")
    if visible_prefix is None:
        masked = min(states - 1, max(config.mask_steps, max(config.horizons)))
        latest = max(1, states - masked)
        if training and latest > 1:
            return torch.randint(1, latest + 1, (batch_size,), device=state.device)
        return torch.full((batch_size,), latest, dtype=torch.long, device=state.device)
    if torch.is_tensor(visible_prefix):
        prefix = visible_prefix.to(device=state.device, dtype=torch.long)
        if prefix.ndim == 0:
            prefix = prefix.expand(batch_size)
        if prefix.shape != (batch_size,):
            raise ValueError("visible_prefix tensor must have shape [B]")
    else:
        prefix = torch.full((batch_size,), int(visible_prefix), dtype=torch.long, device=state.device)
    if bool(((prefix < 1) | (prefix > states)).any()):
        raise ValueError("visible_prefix must be between 1 and the number of states")
    return prefix


def _counterfactual_prefix(batch: dict[str, Any], prefix: torch.Tensor) -> torch.Tensor:
    paired = batch.get("counterfactual_mask")
    if not torch.is_tensor(paired):
        return prefix
    paired = paired.to(device=prefix.device, dtype=torch.bool)
    if paired.shape[0] != prefix.shape[0]:
        raise ValueError("counterfactual_mask must have leading shape [B]")
    paired = paired.reshape(prefix.shape[0], -1).any(dim=1)
    return torch.where(paired, torch.ones_like(prefix), prefix)


def _horizon_predictions(state_hidden: torch.Tensor, heads: nn.ModuleDict, horizons: Iterable[int], prefix: torch.Tensor) -> dict[str, Any]:
    batch_size, states, width = state_hidden.shape
    predictions: dict[int, torch.Tensor] = {}
    masks: dict[int, torch.Tensor] = {}
    selected: list[torch.Tensor] = []
    selected_mask: list[torch.Tensor] = []
    for horizon_value in horizons:
        horizon = int(horizon_value)
        prediction = heads[str(horizon)](state_hidden[:, horizon:])
        source = torch.arange(states - horizon, device=state_hidden.device)
        # Every named horizon is measured from the last visible boundary.
        # Merely crossing the mask boundary would mix shorter predictions into
        # (for example) the h=16 loss and make long-horizon metrics optimistic.
        valid = source.view(1, -1) == (prefix - 1).view(-1, 1)
        predictions[horizon] = prediction
        masks[horizon] = valid
        source_index = (prefix - 1).clamp_max(states - horizon - 1)
        selected.append(prediction.gather(1, source_index.view(batch_size, 1, 1).expand(-1, 1, width)).squeeze(1))
        selected_mask.append(prefix + horizon <= states)
    return {
        "horizon_latent": predictions, "horizon_mask": masks,
        "selected_horizon_latent": torch.stack(selected, dim=1),
        "selected_horizon_mask": torch.stack(selected_mask, dim=1),
    }


def _selected_targets(target: torch.Tensor, prefix: torch.Tensor, horizons: Sequence[int]) -> torch.Tensor:
    batch_size, states, width = target.shape
    values = []
    for horizon in horizons:
        index = (prefix - 1 + int(horizon)).clamp_max(states - 1)
        values.append(target.gather(1, index.view(batch_size, 1, 1).expand(-1, 1, width)).squeeze(1))
    return torch.stack(values, dim=1)


def _transition_hidden(state_hidden: torch.Tensor, control_hidden: torch.Tensor, transitions: int) -> torch.Tensor:
    return state_hidden[:, 1:] if state_hidden.shape[1] == transitions + 1 else control_hidden[:, :transitions]


def _transition_prediction_mask(prefix: torch.Tensor, *, states: int, transitions: int) -> torch.Tensor:
    target_state = torch.arange(1, states, device=prefix.device) if states == transitions + 1 else torch.arange(transitions, device=prefix.device)
    return target_state.view(1, -1) >= prefix.view(-1, 1)


def _sensor_group_visibility(
    batch: dict[str, Any],
    prefix: torch.Tensor,
    modalities: Sequence[str],
) -> dict[str, torch.Tensor] | None:
    """Hide cached sensor values shared by visible states and masked targets."""

    groups = {
        "render_sample_id": ("rgb", "depth", "normals"),
        "lidar_sample_id": ("lidar_range",),
    }
    configured = set(modalities)
    result: dict[str, torch.Tensor] = {}
    for sample_key, group_modalities in groups.items():
        sample_ids = batch.get(sample_key)
        if not torch.is_tensor(sample_ids):
            continue
        sample_ids = sample_ids.to(device=prefix.device, dtype=torch.long)
        if sample_ids.ndim != 2 or sample_ids.shape[0] != prefix.shape[0]:
            raise ValueError(f"{sample_key} must have shape [B,S]")
        positions = torch.arange(sample_ids.shape[1], device=prefix.device)
        target = positions.view(1, -1) >= prefix.view(-1, 1)
        valid_target = target & (sample_ids >= 0)
        same_sample = sample_ids.unsqueeze(2) == sample_ids.unsqueeze(1)
        leaks_target = (same_sample & valid_target.unsqueeze(1)).any(dim=2)
        visibility = ~(leaks_target & ~target & (sample_ids >= 0))
        if bool((~visibility).any()):
            for modality in group_modalities:
                if modality in configured:
                    result[modality] = visibility
    return result or None


def _match_counterfactual_pairs(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    pair_ids = _metadata_list(batch.get("pair_id"))
    roles = _metadata_list(batch.get("pair_role")) or _metadata_list(batch.get("role"))
    starts = _metadata_list(batch.get("start_tick"))
    if not pair_ids or len(pair_ids) != len(roles):
        return torch.empty((0, 2), dtype=torch.long, device=device)
    if not starts:
        starts = [0] * len(pair_ids)
    groups: dict[tuple[str, str], dict[str, int]] = {}
    for index, (pair_id, role, start) in enumerate(zip(pair_ids, roles, starts, strict=True)):
        pair_id = str(pair_id)
        if not pair_id:
            continue
        normalized = {"factual": "control", "counterfactual": "treatment"}.get(str(role).lower(), str(role).lower())
        if normalized in {"control", "treatment"}:
            groups.setdefault((pair_id, str(start)), {})[normalized] = index
    pairs = [(members["control"], members["treatment"]) for _, members in sorted(groups.items()) if {"control", "treatment"} <= set(members)]
    return torch.tensor(pairs, dtype=torch.long, device=device) if pairs else torch.empty((0, 2), dtype=torch.long, device=device)


def _metadata_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


@torch.no_grad()
def _effective_rank(value: torch.Tensor) -> torch.Tensor:
    flattened = value.detach().float().reshape(-1, value.shape[-1])
    if flattened.shape[0] < 2:
        return value.new_tensor(1.0, dtype=torch.float32)
    centered = flattened - flattened.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    probabilities = singular / singular.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return entropy.exp().to(device=value.device)


__all__ = [
    "CausalTransformer", "ControlEncoder", "INTERVENTION_KINDS",
    "MultimodalEncoder", "RSSMLitePack", "TemporalJEPA", "TRANSFORMER_LOSSES",
    "build_model", "model_parameter_counts", "parameter_count",
]
