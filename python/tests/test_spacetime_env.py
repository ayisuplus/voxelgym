from __future__ import annotations

import numpy as np
import voxelgym_rs as rs

from voxelgym import ACTION_KEYS, VoxelGymEnv
from voxelgym.tasks.probes import CollapseJudge


def _idle() -> dict[str, int]:
    action = {key: 0 for key in ACTION_KEYS}
    action["pitch"] = 4
    return action


def test_default_observation_contract_stays_unchanged():
    env = VoxelGymEnv(preset="flat", seed=2)
    observation, _ = env.reset()

    assert set(observation) == {"voxels", "inventory", "pose", "raycast"}


def test_opt_in_spacetime_view_reports_clock_egomotion_and_sensor_age():
    env = VoxelGymEnv(
        preset="flat",
        seed=3,
        render=2,
        spacetime=True,
        dt_numerator=1,
        dt_denominator=40,
    )
    first, info = env.reset()

    assert info["clock"]["tick"] == 0
    assert first["clock"].tolist()[:2] == [0.0, 0.0]
    assert first["spatial_meta"].tolist() == [0.0, 1.0, 1.0]
    np.testing.assert_array_equal(first["egomotion"], np.zeros(6, dtype=np.float32))

    second, _, _, _, info = env.step({**_idle(), "move": 1})
    assert info["clock"]["tick"] == 1
    assert info["clock"]["elapsed_seconds"] == 0.025
    assert second["clock"][0] == 1
    assert second["clock"][1] == 0.025
    assert second["sensor_age"][0] == 0.025
    assert second["egomotion"][2] != 0.0
    assert second["local_relations"].shape == (6,)


def test_env_step_traced_exposes_causal_and_oracle_views():
    env = VoxelGymEnv(preset="void", seed=4, spacetime=True)
    env.reset()

    _, _, _, _, info = env.step_traced(_idle(), trace_level="full", branch_id=9)
    before_oracle = bytes(env.world.snapshot())
    oracle = env.oracle_view()

    assert "trace" not in info
    assert oracle["trace"]["clock_before"]["tick"] == 0
    assert oracle["trace"]["clock_after"]["tick"] == 1
    assert oracle["trace"]["after_hash"] == env.world.hash()
    assert oracle["world_hash"] == env.world.hash()
    assert oracle["clock"]["tick"] == 1
    assert isinstance(oracle["world_snapshot"], bytes)
    assert bytes(env.world.snapshot()) == before_oracle


def test_egomotion_uses_the_shortest_rotation_across_heading_wraparound():
    env = VoxelGymEnv(preset="flat", seed=4, spacetime=True)
    env.reset()

    near_wrap, *_ = env.step({**_idle(), "yaw": 23})
    wrapped, *_ = env.step({**_idle(), "yaw": 0})

    assert near_wrap["egomotion"][3] == -15.0
    assert wrapped["egomotion"][3] == 15.0


def test_task_world_changes_are_tagged_interventions_with_reward_evidence():
    task = CollapseJudge()
    env = VoxelGymEnv(task=task, seed=5, spacetime=True)
    env.reset(seed=5)
    env.world.teleport(-4.5, 6.0, -2.5)

    _, _, terminated, _, info = env.step_traced(_idle(), trace_level="full")

    oracle = env.oracle_view()
    intervention_events = [
        event for event in oracle["events"] if event["phase"] == "intervention"
    ]
    assert terminated
    assert intervention_events
    assert all(event["root_cause"]["kind"] == "intervention" for event in intervention_events)
    assert "task:collapse_judge:supports_removed" in info["reward_outcome"][
        "evidence_labels"
    ]
    intervention_ids = {event["id"] for event in intervention_events}
    assert set(info["reward_outcome"]["evidence_event_ids"]) == intervention_ids
    assert all(
        event["kind"] == "intervention_applied"
        for event in intervention_events
        if event["id"] in info["reward_outcome"]["evidence_event_ids"]
    )
    assert oracle["scene_graph"]["nodes"]


def test_oracle_scene_graph_emits_one_node_for_a_compound_structure():
    env = VoxelGymEnv(preset="void", seed=6)
    env.reset()
    env._w = rs.PyWorld(
        6,
        "void",
        semantic_regions=[
            (1, 99, 0, 5, 0, 0, 5, 0, 1),
            (2, 99, 1, 5, 0, 1, 5, 0, 1),
        ],
    )

    node_ids = [node["id"] for node in env.oracle_view()["scene_graph"]["nodes"]]

    assert len(node_ids) == len(set(node_ids))
    assert node_ids.count("structure:99") == 1
    assert {"region:1", "region:2", "structure:99"} <= set(node_ids)


def test_task_scenario_can_define_one_compound_structure_for_gym_oracle():
    from voxelgym.tasks import make_task

    env = VoxelGymEnv(task=make_task("circuit_door_two"), seed=7)
    env.reset(seed=7)

    oracle = env.oracle_view()
    regions = oracle["semantic_regions"]
    node_ids = [node["id"] for node in oracle["scene_graph"]["nodes"]]
    structure_ids = {region["structure_id"] for region in regions}

    assert len(regions) > 2
    assert len(structure_ids) == 1
    assert len(node_ids) == len(set(node_ids))
    assert node_ids.count(f"structure:{next(iter(structure_ids))}") == 1
