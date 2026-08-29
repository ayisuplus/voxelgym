from __future__ import annotations

from dataclasses import replace
import gc
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from voxelgym.config import (
    DatasetConfig,
    EnvironmentConfig,
    GenerationConfig,
    ModelConfig,
    ResearchConfig,
    RunConfig,
    TrainingConfig,
)
from voxelgym.causal_data import (
    _generate_pair,
    _generation_cycle,
    benchmark_worker_counts,
)
from voxelgym.env import ACTION_KEYS, VoxelGymEnv
from voxelgym.episode_bundle import EpisodeBundleReader
from voxelgym.evaluate import evaluate_run
from voxelgym.models import build_model, parameter_count
from voxelgym.recorder import CausalRecorder
from voxelgym.tasks.base import Task
from voxelgym.train import train, world_model_loss
from voxelgym.training_pack import (
    FORBIDDEN_MODEL_INPUTS,
    TrainingPackDataset,
    build_training_pack,
    bundle_sha256,
    make_training_loader,
    validate_model_input_schema,
    write_dataset_manifest,
)


IDLE = {
    "move": 0,
    "jump": 0,
    "sneak": 0,
    "yaw": 0,
    "pitch": 4,
    "mine": 0,
    "place": 0,
    "use": 0,
    "hotbar": 0,
    "craft": 0,
}


@pytest.fixture(scope="module")
def tiny_pack(tmp_path_factory):
    root = tmp_path_factory.mktemp("causal-pack")
    bundle_root = root / "bundles"
    task_name = "collapse_judge"
    lidar = {
        "channels": 2,
        "azimuth": 8,
        "min_elev": -10.0,
        "max_elev": 5.0,
        "max_range": 16.0,
        "every": 2,
    }
    env = VoxelGymEnv(
        task=None,
        preset="void",
        seed=11,
        render=2,
        lidar=lidar,
        spacetime=True,
        physics={"gravity": 0.08},
    )
    observation, info = env.reset(seed=11)
    assert info["physics_config"] == {"gravity": 0.08}
    assert info["sensor_profile"]["render_every"] == 2
    recorder = CausalRecorder(
        str(bundle_root),
        task_name,
        11,
        render_every=2,
        lidar=lidar,
        spacetime=True,
        checkpoint_every=2,
        stem="tiny.vxbundle",
    )
    recorder.start(env, observation)
    for _ in range(4):
        observation, reward, terminated, truncated, step_info = env.step_traced(
            IDLE, trace_level="full"
        )
        recorder.log(
            env,
            tuple(IDLE[key] for key in ACTION_KEYS),
            reward,
            terminated,
            truncated,
            step_info,
            observation,
        )
    bundle = recorder.save(env.world.hash())
    env.close()
    source = {
        "path": bundle.relative_to(root).as_posix(),
        "sha256": bundle_sha256(bundle),
        "bytes": sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file()),
        "task": task_name,
        "seed": 11,
        "split": "train",
        "policy": "oracle_expert",
        "epsilon": 0.0,
        "physics_config": {"gravity": 0.08},
        "physics": info["physics"],
        "scale": 1.0,
        "clock": {"dt_numerator": 1, "dt_denominator": 20},
        "sensor_profile": info["sensor_profile"],
        "pair_id": None,
        "pair_role": None,
        "branch_id": 0,
    }
    test_source = dict(source)
    test_source["split"] = "test"
    dataset_manifest = write_dataset_manifest(
        root / "manifest.json",
        config={"fixture": True},
        sources=[source, test_source],
    )
    pack_manifest = build_training_pack(
        dataset_manifest,
        root / "pack",
        segment_steps=4,
        window_steps=2,
        shard_bytes=1 << 20,
    )
    return root, pack_manifest


def test_default_physics_argument_preserves_world_hashes():
    first = VoxelGymEnv(seed=7, preset="void")
    second = VoxelGymEnv(seed=7, preset="void", physics=None)
    first.reset(seed=7)
    second.reset(seed=7)
    for _ in range(8):
        first.step(IDLE)
        second.step(IDLE)
        assert first.world.hash() == second.world.hash()
    first.close()
    second.close()


def test_spawned_worker_benchmark_selects_an_eligible_candidate():
    result = benchmark_worker_counts(
        (1,), trials=1, max_memory_fraction=1.0, steps_per_worker=2
    )
    assert result["selected_workers"] == 1
    assert result["results"][0]["median_steps_per_second"] > 0
    representative = benchmark_worker_counts(
        (1,),
        trials=1,
        max_memory_fraction=1.0,
        steps_per_worker=1,
        environment={
            "render_every": 1,
            "lidar": {
                "channels": 2,
                "azimuth": 8,
                "min_elev": -10.0,
                "max_elev": 10.0,
                "max_range": 8.0,
                "every": 1,
            },
            "spacetime": True,
            "scale": 1.0,
            "dt_numerator": 1,
            "dt_denominator": 20,
            "physics": None,
        },
    )
    assert representative["environment"]["render_every"] == 1


def test_production_configs_and_cycle_encode_fixed_mixes(tmp_path):
    repository = Path(__file__).parents[2]
    pilot = ResearchConfig.from_toml(repository / "experiments" / "causal-pilot.toml")
    large = ResearchConfig.from_toml(repository / "experiments" / "causal-500g.toml")
    ood = ResearchConfig.from_toml(repository / "experiments" / "causal-ood.toml")
    assert pilot.dataset.target_gib == 100.0
    assert large.dataset.target_gib == 500.0
    assert large.generation.domain_randomization_fraction == 0.30
    assert ood.dataset.split_override == "test"
    assert len(ood.generation.ood_profiles) == 5

    jobs = _generation_cycle(large, ("collapse_judge",), 0, tmp_path)
    source_counts = {
        policy: sum(
            2 if job["kind"] == "pair" else 1
            for job in jobs
            if job["policy"] == policy
        )
        for policy in ("oracle_expert", "epsilon_mixed", "paired_intervention")
    }
    assert source_counts == {
        "oracle_expert": 5,
        "epsilon_mixed": 3,
        "paired_intervention": 2,
    }
    assert sum(2 if job["kind"] == "pair" else 1 for job in jobs if job["domain_randomized"]) == 3


def test_training_pack_deduplicates_sensors_and_blocks_oracle_leakage(tiny_pack):
    _root, manifest_path = tiny_pack
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_model_input_schema(manifest["input_fields"])
    assert not (set(manifest["input_fields"]) & FORBIDDEN_MODEL_INPUTS)
    parquet = pq.ParquetFile(manifest_path.parent / manifest["files"][0]["path"])
    row = parquet.read_row_group(0).to_pylist()[0]
    assert len(row["rgb_frames"]) < row["length"]
    assert len(row["lidar_range_frames"]) < row["length"]

    dataset = TrainingPackDataset(manifest_path, split="train", context=2)
    sample = dataset[0]
    assert sample["rgb"].shape == (3, 128, 128, 3)
    assert sample["lidar_range"].shape == (3, 2, 8)
    assert sample["action"].shape == (2, len(ACTION_KEYS))
    assert sample["domain"] == "gravity"
    assert len(dataset._row_cache) == 1
    assert not any(key in sample for key in ("events", "deltas", "hash", "snapshot", "oracle"))


def test_training_pack_rebuild_is_deterministic(tiny_pack):
    root, first_path = tiny_pack
    rebuilt_path = build_training_pack(
        root / "manifest.json",
        root / "pack-rebuilt",
        segment_steps=4,
        window_steps=2,
        shard_bytes=1 << 20,
    )
    first = json.loads(first_path.read_text(encoding="utf-8"))
    rebuilt = json.loads(rebuilt_path.read_text(encoding="utf-8"))
    assert first["fingerprint"] == rebuilt["fingerprint"]
    assert [item["sha256"] for item in first["files"]] == [
        item["sha256"] for item in rebuilt["files"]
    ]


@pytest.mark.skipif(os.name != "nt", reason="Windows spawned-worker contract")
def test_windows_streaming_loader_uses_spawned_workers(tiny_pack):
    _root, manifest_path = tiny_pack
    dataset = TrainingPackDataset(manifest_path, split="train", context=2)
    loader = make_training_loader(
        dataset,
        batch_size=2,
        seed=11,
        start_batch=0,
        total_batches=1,
        workers=2,
        prefetch_factor=2,
    )
    iterator = iter(loader)
    batch = next(iterator)
    assert batch["action"].shape[:2] == (2, 2)
    del iterator, loader
    gc.collect()


def test_tiny_transformer_forward_covers_all_prediction_heads(tiny_pack):
    _root, manifest_path = tiny_pack
    dataset = TrainingPackDataset(manifest_path, split="train", context=2)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
    config = ModelConfig(
        kind="transformer",
        d_model=32,
        layers=1,
        heads=4,
        mlp_dim=64,
        context=2,
        horizons=(1,),
    )
    model = build_model(
        config,
        event_classes=len(dataset.event_vocab),
        delta_classes=len(dataset.delta_vocab),
        edge_classes=len(dataset.edge_vocab),
    )
    output = model(batch)
    loss, metrics = world_model_loss("transformer", output, batch, (1,))
    assert torch.isfinite(loss)
    assert output["depth"].shape == (1, 1, 32, 32)
    assert output["seg"].shape == (1, 1, 64, 32, 32)
    assert "copy_ratio_h1" in metrics
    assert "causal_edge" in metrics


@pytest.mark.ml
def test_formal_transformer_stays_in_the_100_to_150m_parameter_budget():
    model = build_model(ModelConfig(), event_classes=64, delta_classes=32)
    assert 100_000_000 <= parameter_count(model) <= 150_000_000


def test_counterfactual_generator_uses_one_boundary_and_one_action_sequence(
    tmp_path, monkeypatch
):
    class FastTask(Task):
        name = "tnt_clear"
        preset = "void"
        horizon = 4

    import voxelgym.tasks as tasks

    monkeypatch.setattr(tasks, "make_task", lambda _name: FastTask())
    job = {
        "kind": "pair",
        "task": "tnt_clear",
        "seed": 5,
        "out_dir": str(tmp_path),
        "render_every": 0,
        "lidar": None,
        "scale": 1.0,
        "dt_numerator": 1,
        "dt_denominator": 20,
        "physics": None,
        "trace_level": "full",
        "policy": "paired_intervention",
        "epsilon": None,
        "counterfactual_steps": 2,
        "train_fraction": 0.8,
        "validation_fraction": 0.1,
    }
    sources = _generate_pair(job)
    assert {source["pair_role"] for source in sources} == {"control", "treatment"}
    assert len({source["pair_id"] for source in sources}) == 1
    readers = [EpisodeBundleReader(source["path"]) for source in sources]
    for reader in readers:
        reader.validate()
    control, treatment = [reader.transitions.to_pylist() for reader in readers]
    assert [tuple(row[key] for key in ACTION_KEYS) for row in control] == [
        tuple(row[key] for key in ACTION_KEYS) for row in treatment
    ]
    control_initial = readers[0].checkpoints.to_pylist()[0]
    treatment_initial = readers[1].checkpoints.to_pylist()[0]
    assert control_initial["world_snapshot"] == treatment_initial["world_snapshot"]
    assert control_initial["agent_observation"] == treatment_initial["agent_observation"]
    assert control[0]["external_intervention_count"] == 0
    assert treatment[0]["external_intervention_count"] == 1


def _training_config(
    root: Path,
    pack_root: Path,
    *,
    kind: str = "rssm",
    objective: str | None = None,
) -> ResearchConfig:
    return ResearchConfig(
        environment=EnvironmentConfig(render_every=2),
        dataset=DatasetConfig(
            root=str(pack_root),
            target_gib=0.001,
            shard_gib=0.001,
            segment_steps=4,
            window_steps=2,
        ),
        generation=GenerationConfig(workers=1, worker_candidates=(1,)),
        model=ModelConfig(
            kind=kind,
            objective=objective,
            latent=16,
            hidden=8,
            d_model=16,
            layers=1,
            heads=4,
            mlp_dim=32,
            context=2,
            horizons=(1,),
            mask_steps=1,
            intervention_features=4,
            modalities=("rgb", "action"),
        ),
        training=TrainingConfig(
            steps=2,
            microbatch=2 if objective == "counterfactual" else 1,
            gradient_accumulation=1,
            loader_workers=0,
            prefetch_factor=1,
            checkpoint_every=1,
            evaluate_every=10,
            log_every=1,
            evaluation_batches=1,
            device="cpu",
            dtype="fp32",
            deterministic=True,
        ),
        run=RunConfig(
            output_dir=str(root),
            name=f"resume-{kind}-{objective or 'default'}",
            seed=3,
        ),
    )


def _assert_nested_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_nested_equal(first, second)
    else:
        assert left == right


@pytest.mark.ml
@pytest.mark.parametrize(
    ("kind", "objective"),
    (("rssm", None), ("transformer", "counterfactual"), ("jepa", None)),
)
def test_two_step_training_resume_is_bit_exact_and_evaluable(
    tiny_pack, tmp_path, kind, objective
):
    pack_root, _manifest_path = tiny_pack
    case = f"{kind}-{objective or 'default'}"
    continuous_config = _training_config(
        tmp_path / case / "continuous",
        pack_root,
        kind=kind,
        objective=objective,
    )
    continuous_run = train(continuous_config)

    interrupted_config = _training_config(
        tmp_path / case / "interrupted",
        pack_root,
        kind=kind,
        objective=objective,
    )
    interrupted_run = train(interrupted_config, stop_after_step=1)
    first_checkpoint = interrupted_run / "checkpoints" / "step-00000001.pt"
    resumed_config = replace(
        interrupted_config,
        training=replace(interrupted_config.training, resume=str(first_checkpoint)),
    )
    resumed_run = train(resumed_config)
    assert resumed_run == interrupted_run

    continuous = torch.load(
        continuous_run / "checkpoints" / "step-00000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    resumed = torch.load(
        resumed_run / "checkpoints" / "step-00000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert continuous["global_step"] == resumed["global_step"] == 2
    assert continuous["sampler_batch"] == resumed["sampler_batch"] == 2
    assert continuous["model_metadata"]["objective"] == (
        objective or "dynamics"
    )
    assert continuous["model_metadata"]["checkpoint_parameters"] >= continuous[
        "model_metadata"
    ]["trainable_parameters"]
    for key, value in continuous["model"].items():
        assert torch.equal(value, resumed["model"][key]), key
    _assert_nested_equal(continuous["optimizer"], resumed["optimizer"])
    assert continuous["scheduler"] == resumed["scheduler"]

    continuous_metrics = [
        json.loads(line)
        for line in (continuous_run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["step"] == 2
    ][0]["metrics"]
    resumed_metrics = [
        json.loads(line)
        for line in (resumed_run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["step"] == 2
    ][0]["metrics"]
    deterministic_metrics = set(continuous_metrics) - {
        "data_wait_fraction",
        "data_wait_seconds",
        "step_seconds",
    }
    assert deterministic_metrics
    for key in deterministic_metrics:
        assert continuous_metrics[key] == resumed_metrics[key]

    report = evaluate_run(resumed_run, device_name="cpu")
    assert report["global_step"] == 2
    assert "1" in report["horizons"]
    assert "collapse_judge" in report["tasks"]
