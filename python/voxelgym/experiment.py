"""Local-first experiment directories, metrics, and exact checkpoints."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any

import numpy as np

from .config import ResearchConfig


def create_run_directory(config: ResearchConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path(config.run.output_dir)
    run = base / f"{stamp}-{config.run.name}-s{config.run.seed}"
    suffix = 1
    while run.exists():
        run = base / f"{stamp}-{config.run.name}-s{config.run.seed}-{suffix}"
        suffix += 1
    (run / "checkpoints").mkdir(parents=True)
    _write_json(run / "resolved_config.json", config.as_dict())
    _write_json(run / "environment.json", environment_record(config))
    return run


def environment_record(config: ResearchConfig) -> dict[str, Any]:
    record: dict[str, Any] = {
        "config_fingerprint": config.fingerprint(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pid": os.getpid(),
        "formats": {
            "world_snapshot": 8,
            "episode_bundle": 2,
            "dataset_manifest": 1,
            "training_pack": 1,
            "checkpoint": 2,
        },
    }
    try:
        record["voxelgym"] = version("voxelgym")
    except PackageNotFoundError:
        record["voxelgym"] = "editable-source"
    with suppress(Exception):
        record["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        record["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    with suppress(Exception):
        import torch

        record.update(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
                "bf16_supported": (
                    torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
                ),
            }
        )
    return record


@dataclass(slots=True)
class RunLogger:
    run_dir: Path
    _metrics_path: Path = field(init=False, repr=False)
    _writer: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._writer = None
        with suppress(ImportError):
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))

    def log(self, step: int, metrics: dict[str, float], *, split: str) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": int(step),
            "split": split,
            "metrics": {key: float(value) for key, value in sorted(metrics.items())},
        }
        with self._metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        if self._writer is not None:
            for key, value in metrics.items():
                self._writer.add_scalar(f"{split}/{key}", value, step)
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


def capture_rng_state() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "RunLogger",
    "atomic_torch_save",
    "capture_rng_state",
    "create_run_directory",
    "environment_record",
    "restore_rng_state",
]
