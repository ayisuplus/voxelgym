"""TOML configuration for causal-dataset and world-model experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import tomllib
from typing import Any, TypeVar


T = TypeVar("T")


@dataclass(slots=True)
class EnvironmentConfig:
    scale: float = 1.0
    dt_numerator: int = 1
    dt_denominator: int = 20
    render_every: int = 1
    spacetime: bool = True
    lidar: dict[str, Any] | None = field(
        default_factory=lambda: {
            "channels": 16,
            "azimuth": 256,
            "min_elev": -20.0,
            "max_elev": 10.0,
            "max_range": 48.0,
            "every": 1,
        }
    )
    physics: dict[str, float] | None = None

    def validate(self) -> None:
        if self.scale < 1 or not float(self.scale).is_integer():
            raise ValueError("environment.scale must be an integer-valued number >= 1")
        if self.dt_numerator <= 0 or self.dt_denominator <= 0:
            raise ValueError("environment clock numerator and denominator must be positive")
        if self.render_every <= 0:
            raise ValueError("environment.render_every must be positive for training data")
        if not self.spacetime:
            raise ValueError("environment.spacetime must be true for causal data")


@dataclass(slots=True)
class DatasetConfig:
    root: str = "data/causal"
    tasks: tuple[str, ...] = ()
    target_gib: float = 100.0
    shard_gib: float = 1.0
    segment_steps: int = 256
    window_steps: int = 64
    max_episodes: int | None = None
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    expert_fraction: float = 0.5
    mixed_fraction: float = 0.3
    paired_fraction: float = 0.2
    epsilon_values: tuple[float, ...] = (0.05, 0.15, 0.30)
    split_override: str | None = None

    def validate(self) -> None:
        if self.target_gib <= 0 or self.shard_gib <= 0:
            raise ValueError("dataset target and shard sizes must be positive")
        if self.segment_steps < self.window_steps or self.window_steps < 2:
            raise ValueError("dataset segment_steps must be >= window_steps >= 2")
        fractions = self.expert_fraction + self.mixed_fraction + self.paired_fraction
        if abs(fractions - 1.0) > 1e-9:
            raise ValueError("dataset policy fractions must sum to 1")
        if any(
            abs(actual - expected) > 1e-9
            for actual, expected in zip(
                (self.expert_fraction, self.mixed_fraction, self.paired_fraction),
                (0.50, 0.30, 0.20),
                strict=True,
            )
        ):
            raise ValueError("dataset policy fractions are fixed at 50/30/20")
        if not 0 < self.train_fraction < 1:
            raise ValueError("dataset.train_fraction must be in (0, 1)")
        if not 0 <= self.validation_fraction < 1 - self.train_fraction:
            raise ValueError("dataset.validation_fraction leaves no test split")
        if not self.epsilon_values or any(not 0 <= value <= 1 for value in self.epsilon_values):
            raise ValueError("dataset.epsilon_values must contain probabilities")
        if tuple(float(value) for value in self.epsilon_values) != (0.05, 0.15, 0.30):
            raise ValueError("dataset.epsilon_values are fixed at 0.05/0.15/0.30")
        if self.max_episodes is not None and (
            self.max_episodes <= 0 or self.max_episodes % 10
        ):
            raise ValueError("dataset.max_episodes must be a positive multiple of 10")
        if self.split_override not in {None, "train", "validation", "test"}:
            raise ValueError("dataset.split_override must be train, validation, test, or omitted")


@dataclass(slots=True)
class GenerationConfig:
    workers: int = 16
    worker_candidates: tuple[int, ...] = (8, 16, 24)
    benchmark_trials: int = 3
    max_memory_fraction: float = 0.70
    seed0: int = 0
    trace_level: str = "full"
    replay_sample_rate: float = 0.01
    counterfactual_steps: int = 256
    rollout_steps: int = 256
    domain_randomization_fraction: float = 0.0
    physics_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    ood_profiles: tuple[dict[str, Any], ...] = ()

    def validate(self) -> None:
        if self.workers <= 0 or any(value <= 0 for value in self.worker_candidates):
            raise ValueError("generation worker counts must be positive")
        if self.benchmark_trials <= 0:
            raise ValueError("generation.benchmark_trials must be positive")
        if not 0 < self.max_memory_fraction <= 1:
            raise ValueError("generation.max_memory_fraction must be in (0, 1]")
        if self.trace_level not in {"off", "events", "full"}:
            raise ValueError("generation.trace_level must be off, events, or full")
        if not 0 <= self.replay_sample_rate <= 1:
            raise ValueError("generation.replay_sample_rate must be in [0, 1]")
        if self.counterfactual_steps < 2:
            raise ValueError("generation.counterfactual_steps must be at least 2")
        if self.rollout_steps < 2:
            raise ValueError("generation.rollout_steps must be at least 2")
        if not 0 <= self.domain_randomization_fraction <= 1:
            raise ValueError("generation.domain_randomization_fraction must be in [0, 1]")
        for key, bounds in self.physics_ranges.items():
            if len(bounds) != 2 or float(bounds[0]) > float(bounds[1]):
                raise ValueError(f"generation.physics_ranges[{key!r}] must be [low, high]")
        for profile in self.ood_profiles:
            allowed = {"name", "scale", "dt_numerator", "dt_denominator", "physics"}
            unknown = set(profile) - allowed
            if unknown or not isinstance(profile.get("name"), str):
                raise ValueError(f"invalid generation.ood_profiles entry: {profile}")
            if "physics" in profile and not isinstance(profile["physics"], dict):
                raise ValueError("OOD profile physics must be a mapping")


@dataclass(slots=True)
class ModelConfig:
    kind: str = "transformer"
    objective: str | None = None
    latent: int = 1024
    hidden: int = 512
    d_model: int = 768
    layers: int = 12
    heads: int = 12
    mlp_dim: int = 3072
    context: int = 64
    horizons: tuple[int, ...] = (1, 4, 8, 16)
    mask_steps: int = 16
    jepa_tau: float = 0.996
    intervention_features: int = 4
    modalities: tuple[str, ...] = (
        "rgb",
        "depth",
        "normals",
        "lidar_range",
        "voxels",
        "pose",
        "inventory",
        "action",
    )

    def __post_init__(self) -> None:
        # ``objective`` did not exist in the first experimental configs.  A
        # missing value must therefore preserve the old full-transformer
        # behaviour while giving the two baselines their natural objective.
        if self.objective is None:
            self.objective = (
                "counterfactual" if self.kind == "transformer" else "dynamics"
            )

    @property
    def state_modalities(self) -> tuple[str, ...]:
        """Modalities that describe an Agent View, excluding controls."""

        return tuple(
            name
            for name in self.modalities
            if name not in {"action", "intervention", "external_intervention"}
        )

    def validate(self) -> None:
        if self.kind not in {"rssm", "transformer", "jepa"}:
            raise ValueError("model.kind must be rssm, transformer, or jepa")
        if self.objective not in {"dynamics", "causal", "counterfactual"}:
            raise ValueError(
                "model.objective must be dynamics, causal, or counterfactual"
            )
        if self.kind != "transformer" and self.objective != "dynamics":
            raise ValueError(
                f"model.kind={self.kind!r} only supports objective='dynamics'"
            )
        if self.context < 2 or not self.horizons or max(self.horizons) >= self.context:
            raise ValueError("model horizons must be positive and smaller than context")
        if any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("model horizons must be positive and smaller than context")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("model horizons must not contain duplicates")
        if self.d_model % self.heads:
            raise ValueError("model.d_model must be divisible by model.heads")
        if min(
            self.latent,
            self.hidden,
            self.d_model,
            self.layers,
            self.heads,
            self.mlp_dim,
            self.mask_steps,
            self.intervention_features,
        ) <= 0:
            raise ValueError("model dimensions and mask_steps must be positive")
        if not 0.0 <= self.jepa_tau < 1.0:
            raise ValueError("model.jepa_tau must be in [0, 1)")
        if not self.state_modalities:
            raise ValueError("model.modalities must include at least one state modality")


@dataclass(slots=True)
class TrainingConfig:
    steps: int = 100_000
    microbatch: int = 8
    gradient_accumulation: int = 8
    loader_workers: int = 8
    prefetch_factor: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_fraction: float = 0.02
    gradient_clip: float = 1.0
    checkpoint_every: int = 1000
    evaluate_every: int = 2000
    log_every: int = 50
    evaluation_batches: int = 32
    device: str = "auto"
    dtype: str = "bf16"
    resume: str | None = None
    deterministic: bool = False

    def validate(self) -> None:
        positive = (
            self.steps,
            self.microbatch,
            self.gradient_accumulation,
            self.prefetch_factor,
            self.checkpoint_every,
            self.evaluate_every,
            self.log_every,
            self.evaluation_batches,
        )
        if any(value <= 0 for value in positive) or self.loader_workers < 0:
            raise ValueError("training counts must be positive (loader_workers may be zero)")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("training.device must be auto, cpu, or cuda")
        if self.dtype not in {"fp32", "bf16"}:
            raise ValueError("training.dtype must be fp32 or bf16")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("training.warmup_fraction must be in [0, 1)")


@dataclass(slots=True)
class RunConfig:
    output_dir: str = "runs"
    name: str = "causal-transformer"
    seed: int = 0


@dataclass(slots=True)
class ResearchConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    run: RunConfig = field(default_factory=RunConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> "ResearchConfig":
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ResearchConfig":
        allowed = {"environment", "dataset", "generation", "model", "training", "run"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown top-level config sections: {sorted(unknown)}")
        config = cls(
            environment=_section(EnvironmentConfig, raw.get("environment", {})),
            dataset=_section(DatasetConfig, raw.get("dataset", {})),
            generation=_section(GenerationConfig, raw.get("generation", {})),
            model=_section(ModelConfig, raw.get("model", {})),
            training=_section(TrainingConfig, raw.get("training", {})),
            run=_section(RunConfig, raw.get("run", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        self.environment.validate()
        self.dataset.validate()
        self.generation.validate()
        self.model.validate()
        self.training.validate()
        if self.model.context != self.dataset.window_steps:
            raise ValueError("model.context must equal dataset.window_steps")
        if (
            self.model.kind == "transformer"
            and self.model.objective == "counterfactual"
            and self.training.microbatch % 2
        ):
            raise ValueError(
                "counterfactual training requires an even microbatch so paired "
                "branches stay in the same batch"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _section(section_type: type[T], values: Any) -> T:
    if not isinstance(values, dict):
        raise TypeError(f"{section_type.__name__} section must be a mapping")
    fields = section_type.__dataclass_fields__
    unknown = set(values) - set(fields)
    if unknown:
        raise ValueError(f"unknown {section_type.__name__} fields: {sorted(unknown)}")
    normalized = dict(values)
    for name in (
        "tasks",
        "epsilon_values",
        "worker_candidates",
        "horizons",
        "modalities",
        "ood_profiles",
    ):
        if name in normalized:
            normalized[name] = tuple(normalized[name])
    if "physics_ranges" in normalized:
        normalized["physics_ranges"] = {
            str(key): tuple(value) for key, value in normalized["physics_ranges"].items()
        }
    return section_type(**normalized)


__all__ = [
    "DatasetConfig",
    "EnvironmentConfig",
    "GenerationConfig",
    "ModelConfig",
    "ResearchConfig",
    "RunConfig",
    "TrainingConfig",
]
