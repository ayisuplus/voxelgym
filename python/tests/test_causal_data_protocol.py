from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from voxelgym.causal_data import _generate_pair, _select_pair_intervention
from voxelgym.env import ACTION_KEYS, VoxelGymEnv
from voxelgym.recorder import CausalRecorder
from voxelgym.training_pack import (
    DeterministicBatchSampler,
    INTERVENTION_KINDS,
    TrainingPackDataset,
    _build_evaluation_suite,
    _pair_quality,
    assign_split,
    build_training_pack,
    bundle_sha256,
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


def test_pack_aligns_t_controls_with_t_plus_one_states_and_sensor_ids(tmp_path):
    env = VoxelGymEnv(preset="void", seed=31, render=2, spacetime=True)
    initial, info = env.reset(seed=31)
    recorder = CausalRecorder(
        str(tmp_path / "bundles"),
        "fixture",
        31,
        render_every=2,
        spacetime=True,
        stem="aligned.vxbundle",
    )
    recorder.start(env, initial)
    for step in range(4):
        interventions = (
            ({"kind": "set_agent_velocity", "velocity": [0.5, 0.0, 0.0]},)
            if step == 0
            else ()
        )
        observation, reward, terminated, truncated, step_info = env.step_traced(
            IDLE, trace_level="full", interventions=interventions
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
        "path": bundle.relative_to(tmp_path).as_posix(),
        "sha256": bundle_sha256(bundle),
        "bytes": sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file()),
        "task": "fixture",
        "seed": 31,
        "split": "test",
        "policy": "oracle_expert",
        "epsilon": 0.0,
        "scale": 1.0,
        "clock": {"dt_numerator": 1, "dt_denominator": 20},
        "physics_config": {},
        "sensor_profile": info["sensor_profile"],
        "pair_id": None,
        "pair_role": None,
        "branch_id": 0,
    }
    dataset_manifest = write_dataset_manifest(
        tmp_path / "manifest.json", config={"fixture": True}, sources=[source]
    )
    pack_manifest = build_training_pack(
        dataset_manifest,
        tmp_path / "pack",
        segment_steps=4,
        window_steps=2,
        shard_bytes=1 << 20,
    )
    dataset = TrainingPackDataset(pack_manifest, split="test", context=2)
    sample = dataset[0]
    assert sample["rgb"].shape == (3, 128, 128, 3)
    assert sample["voxels"].shape[0] == 3
    assert sample["action"].shape == (2, len(ACTION_KEYS))
    assert sample["intervention_kind"].shape == (2, len(INTERVENTION_KINDS))
    assert sample["intervention_params"].shape == (2, len(INTERVENTION_KINDS), 4)
    velocity_slot = INTERVENTION_KINDS.index("set_agent_velocity")
    assert sample["intervention_kind"][0, velocity_slot] == 1
    assert sample["render_sample_id"].tolist() == [0, 0, 2]
    assert sample["causal_edges"].shape == (2, len(dataset.edge_vocab))
    manifest = json.loads(pack_manifest.read_text(encoding="utf-8"))
    assert manifest["state_steps"] == 3
    assert manifest["transition_steps"] == 2
    assert manifest["evaluation_suite"]["entries"][0]["source_index"] == 0


def test_balanced_sampler_uses_exact_source_first_supercycle():
    dataset = object.__new__(TrainingPackDataset)
    dataset._cumulative = np.asarray([0, 10], dtype=np.int64)
    dataset._ranges_by_source = {index: [(index, 1, 0)] for index in range(10)}
    sources = [
        {"policy": "oracle_expert", "epsilon": 0.0, "pair_id": None}
        for _ in range(5)
    ]
    sources.extend(
        {"policy": "epsilon_mixed", "epsilon": epsilon, "pair_id": None}
        for epsilon in (0.05, 0.15, 0.30)
    )
    sources.extend(
        (
            {
                "policy": "paired_intervention",
                "epsilon": 0.15,
                "pair_id": "pair",
                "pair_role": role,
            }
            for role in ("control", "treatment")
        )
    )
    dataset.dataset_sources = tuple(sources)
    sampler = DeterministicBatchSampler(
        dataset, batch_size=10, seed=7, start_batch=0, total_batches=1
    )
    batch = next(iter(sampler))
    selected = [sources[index] for index in batch]
    assert [source["policy"] for source in selected].count("oracle_expert") == 5
    assert [source["epsilon"] for source in selected[5:8]] == [0.05, 0.15, 0.30]
    assert [source["pair_role"] for source in selected[8:]] == ["control", "treatment"]


def test_episode_seed_alone_determines_split():
    assert assign_split("task-a", 91, 0.8, 0.1) == assign_split(
        "task-b", 91, 0.8, 0.1
    )


def test_intervention_rotation_uses_effective_canonical_candidates():
    env = VoxelGymEnv(preset="void", seed=9, spacetime=True)
    env.reset(seed=9)
    selected = [_select_pair_intervention(env, rotation)[1] for rotation in range(4)]
    env.close()
    assert selected == list(INTERVENTION_KINDS[:4])


def test_pair_pack_exposes_aligned_per_horizon_outcomes(tmp_path):
    job = {
        "kind": "pair",
        "task": "tnt_clear",
        "seed": 5,
        "out_dir": str(tmp_path / "bundles"),
        "render_every": 0,
        "lidar": None,
        "scale": 1.0,
        "dt_numerator": 1,
        "dt_denominator": 20,
        "physics": None,
        "trace_level": "full",
        "policy": "paired_intervention",
        "epsilon": 0.15,
        "counterfactual_steps": 4,
        "train_fraction": 0.8,
        "validation_fraction": 0.1,
        "split_override": "test",
        "intervention_rotation": 2,
    }
    sources = _generate_pair(job)
    for source in sources:
        source["path"] = Path(source["path"]).relative_to(tmp_path).as_posix()
    dataset_manifest = write_dataset_manifest(
        tmp_path / "manifest.json", config={"fixture": True}, sources=sources
    )
    pack_manifest = build_training_pack(
        dataset_manifest,
        tmp_path / "pack",
        segment_steps=4,
        window_steps=2,
        shard_bytes=1 << 20,
    )
    dataset = TrainingPackDataset(pack_manifest, split="test", context=2)
    control = dataset[0]
    treatment = dataset[3]
    assert control["pair_id"] == treatment["pair_id"]
    assert {control["pair_role"], treatment["pair_role"]} == {"control", "treatment"}
    assert control["start_tick"] == treatment["start_tick"]
    assert control["counterfactual_mask"].tolist() == [True, False, False, False]
    assert treatment["counterfactual_mask"].tolist() == [True, False, False, False]
    velocity_slot = INTERVENTION_KINDS.index("set_agent_velocity")
    branch_by_role = {control["pair_role"]: control, treatment["pair_role"]: treatment}
    assert branch_by_role["control"]["intervention_kind"][0].sum() == 0
    assert branch_by_role["treatment"]["intervention_kind"][0, velocity_slot] == 1
    manifest = json.loads(pack_manifest.read_text(encoding="utf-8"))
    assert manifest["pair_quality"]["pairs"] == 1
    assert manifest["pair_quality"]["horizons"]["1"]["valid"] == 1


def test_evaluation_suite_caps_final_pair_branch_entries_at_64():
    sources = []
    segments = []
    for pair_index in range(33):
        pair_id = f"pair-{pair_index}"
        for role in ("control", "treatment"):
            source_index = len(sources)
            sources.append(
                {
                    "task": "fixture",
                    "seed": pair_index,
                    "split": "test",
                    "pair_id": pair_id,
                    "pair_role": role,
                    "pair_boundary_tick": 3,
                    "scale": 1.0,
                }
            )
            segments.append(
                {
                    "source_index": source_index,
                    "length": 8,
                    "split": "test",
                    "start_tick": 0,
                    "file": f"pack-{source_index}.parquet",
                    "row_group": 0,
                }
            )
    suite = _build_evaluation_suite(sources, segments, window_steps=2)
    assert len(suite["entries"]) == 64
    roles_by_pair: dict[str, set[str]] = {}
    for entry in suite["entries"]:
        roles_by_pair.setdefault(entry["pair_id"], set()).add(entry["pair_role"])
    assert len(roles_by_pair) == 32
    assert all(roles == {"control", "treatment"} for roles in roles_by_pair.values())
    assert {entry["start"] for entry in suite["entries"]} == {3}


def test_pair_quality_reports_pilot_distribution_gates():
    sources = []
    targets = {}
    for index, kind in enumerate(INTERVENTION_KINDS):
        pair_id = f"pair-{index}"
        sources.append(
            {"pair_id": pair_id, "pair_intervention_kind": kind}
        )
        targets[pair_id] = {
            "valid": [True] * 4,
            "propagated": [index < 2] * 4,
        }
    quality = _pair_quality(sources, targets)
    assert all(value == 0.2 for value in quality["intervention_kind_fractions"].values())
    assert quality["propagated_fraction"] == 0.4
    assert quality["not_propagated_fraction"] == 0.6
    assert quality["gates"]["pilot_distribution_ready"]
