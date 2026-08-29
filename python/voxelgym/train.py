"""Local BF16/eager trainer with deterministic sampler and exact resume."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .config import ResearchConfig
from .experiment import (
    RunLogger,
    atomic_torch_save,
    capture_rng_state,
    create_run_directory,
    restore_rng_state,
)
from .models import RSSMLitePack, TemporalJEPA, build_model, parameter_count
from .training_pack import TrainingPackDataset, make_training_loader


def train(config: ResearchConfig, *, stop_after_step: int | None = None) -> Path:
    config.validate()
    _seed_everything(config.run.seed, deterministic=config.training.deterministic)
    device = _resolve_device(config.training.device)
    use_bf16 = config.training.dtype == "bf16"
    if use_bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support BF16")

    pack_manifest_path = Path(config.dataset.root).resolve() / "pack" / "manifest.json"
    training_dataset = TrainingPackDataset(
        pack_manifest_path, split="train", context=config.model.context
    )
    pack_fingerprint = str(training_dataset.manifest["fingerprint"])
    model = build_model(
        config.model,
        event_classes=len(training_dataset.event_vocab),
        delta_classes=len(training_dataset.delta_vocab),
        edge_classes=len(training_dataset.edge_vocab),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _learning_rate_schedule(config.training.steps, config.training.warmup_fraction),
    )

    global_step = 0
    start_batch = 0
    resume_path = config.training.resume
    if resume_path:
        checkpoint_path = Path(resume_path).resolve()
        run_dir = checkpoint_path.parent.parent
        checkpoint_payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint_payload.get("format_version") != 2:
            raise ValueError("only checkpoint format v2 can be resumed")
        if checkpoint_payload["training_identity"] != _training_identity(config):
            raise ValueError("resume config does not match the checkpoint config")
        if checkpoint_payload["pack_fingerprint"] != pack_fingerprint:
            raise ValueError("resume Training Pack does not match the checkpoint")
        if checkpoint_payload.get("model_metadata") != _model_metadata(model, config):
            raise ValueError("resume model metadata does not match the checkpoint")
        model.load_state_dict(checkpoint_payload["model"])
        optimizer.load_state_dict(checkpoint_payload["optimizer"])
        _optimizer_to(optimizer, device)
        scheduler.load_state_dict(checkpoint_payload["scheduler"])
        global_step = int(checkpoint_payload["global_step"])
        start_batch = int(checkpoint_payload["sampler_batch"])
        expected_batch = global_step * config.training.gradient_accumulation
        if start_batch != expected_batch:
            raise ValueError("checkpoint was not saved at an optimizer-step boundary")
        if int(
            checkpoint_payload.get("mask_state", {}).get(
                "next_sampler_batch", -1
            )
        ) != start_batch:
            raise ValueError("checkpoint mask state does not match sampler state")
        restore_rng_state(checkpoint_payload["rng"])
    else:
        run_dir = create_run_directory(config)

    data_record = {
        "training_pack_manifest": str(pack_manifest_path),
        "training_pack_fingerprint": pack_fingerprint,
        "dataset_manifest": training_dataset.manifest["dataset_manifest"],
        "dataset_manifest_fingerprint": training_dataset.manifest["dataset_fingerprint"],
    }
    (run_dir / "data.json").write_text(
        json.dumps(data_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    model_record = {
        **_model_metadata(model, config),
        "device": str(device),
        "dtype": config.training.dtype,
        "eager": True,
    }
    (run_dir / "model.json").write_text(
        json.dumps(model_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    total_batches = config.training.steps * config.training.gradient_accumulation
    loader = make_training_loader(
        training_dataset,
        batch_size=config.training.microbatch,
        seed=config.run.seed,
        start_batch=start_batch,
        total_batches=total_batches,
        workers=config.training.loader_workers,
        prefetch_factor=config.training.prefetch_factor,
    )
    logger = RunLogger(run_dir)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated: dict[str, float] = {}
    accumulated_microbatches = 0
    last_iteration_end = time.perf_counter()
    step_started = last_iteration_end
    last_checkpoint: Path | None = None
    try:
        for sampler_batch, batch in enumerate(loader, start=start_batch):
            data_wait_seconds = time.perf_counter() - last_iteration_end
            batch = _move_batch(batch, device)
            visible_prefix = deterministic_visible_prefixes(
                batch,
                seed=config.run.seed,
                sampler_batch=sampler_batch,
                horizons=config.model.horizons,
            )
            with _autocast(device, use_bf16):
                output = (
                    model(batch)
                    if isinstance(model, RSSMLitePack)
                    else model(batch, visible_prefix=visible_prefix)
                )
                loss, metrics = world_model_loss(
                    config.model.kind,
                    output,
                    batch,
                    config.model.horizons,
                    config.model.objective,
                )
                scaled_loss = loss / config.training.gradient_accumulation
            scaled_loss.backward()
            accumulated_microbatches += 1
            for key, value in metrics.items():
                accumulated[key] = accumulated.get(key, 0.0) + float(value)
            accumulated["data_wait_seconds"] = (
                accumulated.get("data_wait_seconds", 0.0) + data_wait_seconds
            )
            if accumulated_microbatches < config.training.gradient_accumulation:
                last_iteration_end = time.perf_counter()
                continue

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, RSSMLitePack):
                model.ema_update()
            elif isinstance(model, TemporalJEPA):
                model.ema_update()
            global_step += 1
            elapsed = time.perf_counter() - step_started
            train_metrics = {
                key: value / accumulated_microbatches
                for key, value in accumulated.items()
            }
            train_metrics.update(
                {
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "step_seconds": elapsed,
                    "data_wait_fraction": min(
                        1.0, train_metrics["data_wait_seconds"] / max(elapsed, 1e-9)
                    ),
                }
            )
            train_metrics.update(_cuda_metrics(device))
            if global_step % config.training.log_every == 0 or global_step == 1:
                logger.log(global_step, train_metrics, split="train")

            accumulated = {}
            accumulated_microbatches = 0
            step_started = time.perf_counter()

            if global_step % config.training.evaluate_every == 0:
                evaluation = evaluate_model(
                    model,
                    config,
                    pack_manifest_path,
                    device=device,
                    use_bf16=use_bf16,
                )
                logger.log(global_step, evaluation, split="validation")
                model.train()

            if global_step % config.training.checkpoint_every == 0:
                last_checkpoint = _save_checkpoint(
                    run_dir,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    pack_fingerprint,
                    global_step,
                )
            if stop_after_step is not None and global_step >= stop_after_step:
                if last_checkpoint is None or _checkpoint_step(last_checkpoint) != global_step:
                    last_checkpoint = _save_checkpoint(
                        run_dir,
                        model,
                        optimizer,
                        scheduler,
                        config,
                        pack_fingerprint,
                        global_step,
                    )
                break
            last_iteration_end = time.perf_counter()

        expected_step = (
            min(stop_after_step, config.training.steps)
            if stop_after_step is not None
            else config.training.steps
        )
        if global_step != expected_step:
            raise RuntimeError(
                f"training stopped at step {global_step}, expected {expected_step}"
            )
        if last_checkpoint is None or _checkpoint_step(last_checkpoint) != global_step:
            last_checkpoint = _save_checkpoint(
                run_dir,
                model,
                optimizer,
                scheduler,
                config,
                pack_fingerprint,
                global_step,
            )
    finally:
        logger.close()
    return run_dir


def world_model_loss(
    kind: str,
    output: dict[str, Any],
    batch: dict[str, Any],
    horizons: Iterable[int],
    objective: str | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    horizons = tuple(int(value) for value in horizons)
    if kind == "rssm":
        target = output["next_target_latent"]
        model_mse = F.mse_loss(output["next_latent"], target)
        copy_mse = F.mse_loss(
            output["target_latent"][:, : target.shape[1]], target
        )
        reconstruction_indices = output["reconstruction_indices"]
        rgb_target = _resize_rgb(
            batch["rgb"].index_select(1, reconstruction_indices), 32
        )
        reconstruction = F.mse_loss(output["rgb_reconstruction"], rgb_target)
        total = model_mse + 0.5 * reconstruction
        metrics = {
            "loss": float(total.detach()),
            "latent_mse_h1": float(model_mse.detach()),
            "copy_mse_h1": float(copy_mse.detach()),
            "copy_ratio_h1": float((model_mse / copy_mse.clamp_min(1e-12)).detach()),
            "rgb_reconstruction": float(reconstruction.detach()),
        }
        return total, metrics

    if kind == "jepa":
        valid = output["selected_horizon_mask"].bool()
        cosine = (output["jepa_prediction"] * output["jepa_target"]).sum(dim=-1)
        predictive = _masked_mean(2.0 - 2.0 * cosine, valid)
        variance = output["jepa_variance_loss"]
        total = predictive + 0.10 * variance
        metrics = {
            "loss": float(total.detach()),
            "jepa": float(predictive.detach()),
            "variance": float(variance.detach()),
            "effective_rank": float(output["jepa_effective_rank"].detach()),
        }
        target = output["target_latent"]
        prefix = output["visible_prefix"]
        for index, horizon in enumerate(horizons):
            selected = valid[:, index]
            prediction = output["jepa_prediction"][:, index]
            expected = output["jepa_target"][:, index]
            copied = F.normalize(
                _gather_states(target, prefix - 1), dim=-1
            )
            model_mse = _masked_vector_mse(prediction, expected, selected)
            copy_mse = _masked_vector_mse(copied, expected, selected)
            metrics[f"latent_mse_h{horizon}"] = float(model_mse.detach())
            metrics[f"copy_mse_h{horizon}"] = float(copy_mse.detach())
            metrics[f"copy_ratio_h{horizon}"] = float(
                (model_mse / copy_mse.clamp_min(1e-12)).detach()
            )
        return total, metrics

    if kind != "transformer":
        raise ValueError(f"unsupported world-model kind {kind!r}")

    active = set(output.get("active_losses", ()))
    resolved_objective = objective or str(output.get("objective", "counterfactual"))
    target_latent = output["target_latent"]
    latent_losses: list[torch.Tensor] = []
    metrics: dict[str, float] = {}
    for horizon in horizons:
        prediction = output["horizon_latent"][horizon]
        target = target_latent[:, horizon:]
        mask = output["horizon_mask"][horizon].bool()
        model_mse = _masked_vector_mse(prediction, target, mask)
        copy_mse = _masked_vector_mse(target_latent[:, :-horizon], target, mask)
        latent_losses.append(model_mse)
        metrics[f"latent_mse_h{horizon}"] = float(model_mse.detach())
        metrics[f"copy_mse_h{horizon}"] = float(copy_mse.detach())
        metrics[f"copy_ratio_h{horizon}"] = float(
            (model_mse / copy_mse.clamp_min(1e-12)).detach()
        )

    transition_mask = output["transition_mask"].bool()
    reconstruction_indices = output["reconstruction_indices"]
    reconstruction_mask = transition_mask.index_select(1, reconstruction_indices)
    depth_states = batch["depth"][:, 1:].index_select(1, reconstruction_indices)
    depth_target = _resize_scalar(depth_states.float(), 32)
    depth_target = torch.log1p(depth_target.clamp_min(0.0)) / 5.0
    depth_loss = _masked_smooth_l1(output["depth"], depth_target, reconstruction_mask)
    seg_states = batch["seg"][:, 1:].index_select(1, reconstruction_indices)
    seg_target = torch.bitwise_and(seg_states.long(), 0xFFF)
    seg_target = _resize_labels(seg_target, 32).clamp_max(63)
    seg_loss = _masked_cross_entropy(output["seg"], seg_target, reconstruction_mask)
    reward_loss = _masked_smooth_l1(
        output["reward"], batch["reward"].float(), transition_mask
    )
    terminal_loss = _masked_binary_loss(
        output["terminal"], batch["terminal"], transition_mask
    )
    time_target = torch.log1p(batch["time_to_event"].float().clamp_min(0.0))
    time_loss = _masked_smooth_l1(
        output["time_to_event"], time_target, transition_mask
    )
    event_loss = _optional_multilabel_loss(
        output.get("event"), batch["event_kinds"], transition_mask
    )
    delta_loss = _optional_multilabel_loss(
        output.get("delta"), batch["delta_fields"], transition_mask
    )
    edge_loss = _optional_multilabel_loss(
        output.get("causal_edge"), batch["causal_edges"], transition_mask
    )

    latent_loss = torch.stack(latent_losses).mean()
    components: dict[str, tuple[torch.Tensor, float]] = {
        "latent": (latent_loss, 1.0),
        "depth": (depth_loss, 0.25),
        "seg": (seg_loss, 0.10),
        "reward": (reward_loss, 0.25),
        "terminal": (terminal_loss, 0.10),
        "event": (event_loss, 0.10),
        "delta": (delta_loss, 0.10),
        "time_to_event": (time_loss, 0.10),
        "causal_edge": (edge_loss, 0.10),
    }

    cf_metrics = _counterfactual_losses(output, batch, target_latent, horizons)
    components.update(
        {
            "counterfactual_latent_effect": (cf_metrics["latent"], 0.25),
            "counterfactual_propagation": (cf_metrics["propagation"], 0.10),
            "counterfactual_reward_delta": (cf_metrics["reward_delta"], 0.10),
        }
    )
    total = latent_loss.new_zeros(())
    for name, (value, weight) in components.items():
        if name in active:
            total = total + weight * value
            metrics[name] = float(value.detach())
    metrics["loss"] = float(total.detach())
    metrics["objective_id"] = float(
        {"dynamics": 0, "causal": 1, "counterfactual": 2}[resolved_objective]
    )
    return total, metrics


def deterministic_visible_prefixes(
    batch: dict[str, Any],
    *,
    seed: int,
    sampler_batch: int,
    horizons: Iterable[int],
) -> torch.Tensor:
    """Choose reproducible rollout anchors without consuming process RNG state.

    Paired counterfactual rows always start at their common branch boundary.
    Ordinary rows use a stable per-source digest, so all model arms see the
    same masked future and resume needs only the sampler batch index.
    """

    transition_steps = int(batch["action"].shape[1])
    maximum_horizon = max(int(value) for value in horizons)
    latest = transition_steps - maximum_horizon
    if latest < 0:
        raise ValueError("batch is shorter than the largest prediction horizon")
    earliest = min(maximum_horizon, latest)
    pair_mask = batch.get("pair_horizon_mask")
    if pair_mask is None:
        pair_mask = batch.get("counterfactual_mask")
    source_indices = batch.get("source_index")
    start_ticks = batch.get("start_tick")
    pair_ids = batch.get("pair_id")
    batch_size = int(batch["action"].shape[0])
    prefixes: list[int] = []
    for index in range(batch_size):
        is_pair_query = bool(
            pair_mask is not None
            and torch.is_tensor(pair_mask)
            and pair_mask[index].bool().any()
        )
        if is_pair_query:
            # Model prefixes count visible boundary states.  One means the
            # shared pre-intervention state is visible and every post-state is
            # masked.
            prefixes.append(1)
            continue
        source = (
            int(source_indices[index])
            if torch.is_tensor(source_indices)
            else index
        )
        tick = int(start_ticks[index]) if torch.is_tensor(start_ticks) else 0
        pair_id = str(pair_ids[index]) if pair_ids is not None else ""
        digest = hashlib.sha256(
            f"{seed}:{sampler_batch}:{source}:{tick}:{pair_id}".encode("utf-8")
        ).digest()
        span = latest - earliest + 1
        prefixes.append(earliest + int.from_bytes(digest[:8], "little") % span)
    return torch.tensor(prefixes, dtype=torch.long, device=batch["action"].device)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    config: ResearchConfig,
    pack_manifest_path: str | Path,
    *,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, float]:
    dataset = _evaluation_dataset(pack_manifest_path, config.model.context)
    batches = min(
        config.training.evaluation_batches,
        max(1, math.ceil(len(dataset) / config.training.microbatch)),
    )
    loader = make_training_loader(
        dataset,
        batch_size=config.training.microbatch,
        seed=config.run.seed + 17,
        start_batch=0,
        total_batches=batches,
        workers=0,
        prefetch_factor=config.training.prefetch_factor,
    )
    aggregate: dict[str, float] = {}
    task_losses: dict[str, list[float]] = {}
    model.eval()
    for sampler_batch, batch in enumerate(loader):
        task_names = list(batch["task"])
        moved = _move_batch(batch, device)
        visible_prefix = deterministic_visible_prefixes(
            moved,
            seed=config.run.seed + 17,
            sampler_batch=sampler_batch,
            horizons=config.model.horizons,
        )
        with _autocast(device, use_bf16):
            output = (
                model(moved)
                if isinstance(model, RSSMLitePack)
                else model(moved, visible_prefix=visible_prefix)
            )
            _loss, metrics = world_model_loss(
                config.model.kind,
                output,
                moved,
                config.model.horizons,
                config.model.objective,
            )
        for key, value in metrics.items():
            aggregate[key] = aggregate.get(key, 0.0) + float(value)
        for task_name in task_names:
            task_losses.setdefault(task_name, []).append(metrics["loss"])
    count = max(1, len(loader))
    result = {key: value / count for key, value in aggregate.items()}
    result.update(
        {
            f"task/{task_name}/loss": float(np.mean(values))
            for task_name, values in sorted(task_losses.items())
        }
    )
    return result


def _evaluation_dataset(path: str | Path, context: int) -> TrainingPackDataset:
    for split in ("validation", "test", "train"):
        try:
            return TrainingPackDataset(path, split=split, context=context)
        except ValueError:
            continue
    raise ValueError("Training Pack contains no evaluable windows")


def _save_checkpoint(
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: ResearchConfig,
    pack_fingerprint: str,
    global_step: int,
) -> Path:
    path = run_dir / "checkpoints" / f"step-{global_step:08d}.pt"
    atomic_torch_save(
        {
            "format": "voxelgym.checkpoint",
            "format_version": 2,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": int(global_step),
            "sampler_batch": int(global_step * config.training.gradient_accumulation),
            "mask_state": {
                "next_sampler_batch": int(
                    global_step * config.training.gradient_accumulation
                )
            },
            "rng": capture_rng_state(),
            "training_identity": _training_identity(config),
            "pack_fingerprint": pack_fingerprint,
            "model_metadata": _model_metadata(model, config),
        },
        path,
    )
    temporary = run_dir / "checkpoints" / f"latest.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps({"checkpoint": path.name, "global_step": global_step}) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, run_dir / "checkpoints" / "latest.json")
    return path


def _training_identity(config: ResearchConfig) -> str:
    payload = config.as_dict()
    payload["training"]["resume"] = None
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _model_metadata(
    model: torch.nn.Module, config: ResearchConfig
) -> dict[str, Any]:
    return {
        "kind": config.model.kind,
        "objective": config.model.objective,
        "active_losses": list(getattr(model, "active_losses", ())),
        "trainable_parameters": parameter_count(model, trainable_only=True),
        "checkpoint_parameters": parameter_count(model),
    }


def _learning_rate_schedule(total_steps: int, warmup_fraction: float):
    warmup_steps = int(total_steps * warmup_fraction)

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return schedule


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device=cuda but CUDA is unavailable")
    return torch.device(requested)


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _seed_everything(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _resize_rgb(value: torch.Tensor, size: int) -> torch.Tensor:
    batch, time_steps = value.shape[:2]
    tensor = value.reshape(-1, *value.shape[2:]).permute(0, 3, 1, 2).float() / 255.0
    tensor = F.interpolate(tensor, size=(size, size), mode="area")
    return tensor.reshape(batch, time_steps, 3, size, size)


def _resize_scalar(value: torch.Tensor, size: int) -> torch.Tensor:
    batch, time_steps = value.shape[:2]
    tensor = F.interpolate(
        value.reshape(-1, 1, *value.shape[2:]),
        size=(size, size),
        mode="area",
    )
    return tensor.reshape(batch, time_steps, size, size)


def _resize_labels(value: torch.Tensor, size: int) -> torch.Tensor:
    batch, time_steps = value.shape[:2]
    tensor = F.interpolate(
        value.reshape(-1, 1, *value.shape[2:]).float(),
        size=(size, size),
        mode="nearest",
    )
    return tensor.reshape(batch, time_steps, size, size).long()


def _optional_multilabel_loss(
    prediction: torch.Tensor | None,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction is None or target.shape[-1] == 0:
        reference = target if prediction is None else prediction
        return reference.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(
        prediction, target.float(), reduction="none"
    ).mean(dim=-1)
    return _masked_mean(loss, mask)


def _masked_binary_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        prediction, target.float(), reduction="none"
    )
    return _masked_mean(loss, mask)


def _masked_smooth_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, target.float(), reduction="none")
    return _masked_mean(loss, mask)


def _masked_vector_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    loss = torch.square(prediction - target).mean(dim=-1)
    return _masked_mean(loss, mask)


def _masked_cross_entropy(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    return F.cross_entropy(prediction[mask], target[mask])


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.bool()
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(value)
    if not bool(expanded.any()):
        return value.sum() * 0.0
    return value.masked_select(expanded).mean()


def _gather_states(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_size, _states, width = value.shape
    return value.gather(
        1, indices.view(batch_size, 1, 1).expand(-1, 1, width)
    ).squeeze(1)


def _selected_latent_targets(
    target: torch.Tensor, prefix: torch.Tensor, horizons: tuple[int, ...]
) -> torch.Tensor:
    states = target.shape[1]
    selected = [
        _gather_states(target, (prefix - 1 + horizon).clamp_max(states - 1))
        for horizon in horizons
    ]
    return torch.stack(selected, dim=1)


def _counterfactual_losses(
    output: dict[str, Any],
    batch: dict[str, Any],
    target_latent: torch.Tensor,
    horizons: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    prediction = output["counterfactual_latent_effect"]
    pairs = output["cf_pair_indices"]
    zero = prediction.sum() * 0.0
    if pairs.numel() == 0:
        return {"latent": zero, "propagation": zero, "reward_delta": zero}

    control = pairs[:, 0]
    treatment = pairs[:, 1]
    targets = _selected_latent_targets(
        target_latent, output["visible_prefix"], horizons
    )
    target_effect = targets.index_select(0, treatment) - targets.index_select(0, control)
    selected_valid = output["selected_horizon_mask"].bool()
    valid = selected_valid.index_select(0, control) & selected_valid.index_select(
        0, treatment
    )

    stored_mask = batch["counterfactual_mask"].bool()
    stored_propagation = batch["counterfactual_propagated"].float()
    stored_reward = batch["counterfactual_reward_delta"].float()
    columns: list[int] = []
    canonical_horizons = (1, 4, 8, 16)
    for horizon in horizons:
        if horizon not in canonical_horizons:
            return {"latent": zero, "propagation": zero, "reward_delta": zero}
        columns.append(canonical_horizons.index(horizon))
    column_index = torch.tensor(columns, device=stored_mask.device, dtype=torch.long)
    stored_mask = stored_mask.index_select(1, column_index)
    stored_propagation = stored_propagation.index_select(1, column_index)
    stored_reward = stored_reward.index_select(1, column_index)
    valid = valid & stored_mask.index_select(0, control) & stored_mask.index_select(
        0, treatment
    )

    latent = _masked_vector_mse(prediction, target_effect, valid)
    propagation = _masked_binary_loss(
        output["counterfactual_propagation"],
        stored_propagation.index_select(0, treatment),
        valid,
    )
    reward_delta = _masked_smooth_l1(
        output["counterfactual_reward_delta"],
        stored_reward.index_select(0, treatment),
        valid,
    )
    return {
        "latent": latent,
        "propagation": propagation,
        "reward_delta": reward_delta,
    }


def _cuda_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    metrics = {
        "cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / (1 << 30),
        "cuda_reserved_gib": torch.cuda.max_memory_reserved(device) / (1 << 30),
    }
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            text=True,
            timeout=2,
            stderr=subprocess.DEVNULL,
        )
        metrics["gpu_utilization"] = float(output.strip().splitlines()[0]) / 100.0
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return metrics


def _checkpoint_step(path: Path) -> int:
    return int(path.stem.rsplit("-", 1)[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    config = ResearchConfig.from_toml(args.config)
    if args.seed is not None:
        config = replace(config, run=replace(config.run, seed=args.seed))
    run_dir = train(config, stop_after_step=args.stop_after)
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate_model", "train", "world_model_loss"]
