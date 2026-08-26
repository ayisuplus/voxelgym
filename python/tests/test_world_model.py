from __future__ import annotations

from dataclasses import replace
import json

from voxelgym.episode_bundle import (
    EpisodeBoundary,
    EpisodeBundleWriter,
    TransitionRecord,
)
from voxelgym.task_state import EnvSnapshot, RewardOutcome
from voxelgym.world_model import WorldModelAdapter, run_world_model_baseline


def _action(move: int) -> tuple[int, ...]:
    return (move, 0, 0, 6, 4, 0, 0, 0, 0, 0)


def _agent_payload(_branch_id: int, tick: int) -> bytes:
    return f"agent-{tick}".encode()


def _oracle_payload(_branch_id: int, tick: int, world_hash: int) -> bytes:
    return json.dumps(
        {
            "clock": {"tick": tick},
            "world_hash": world_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _boundary(
    branch_id: int,
    tick: int,
    world_hash: int,
    label: str,
) -> EpisodeBoundary:
    world_snapshot = f"world-{label}-{branch_id}-{tick}".encode()
    env_snapshot = EnvSnapshot(
        world_snapshot=world_snapshot,
        task_state=None,
        np_random_state={},
        episode_seed=7,
        terminated=False,
        truncated=False,
        last_reward=RewardOutcome(),
        last_frames=None,
        last_scan=None,
        render_sample_tick=None,
        lidar_sample_tick=None,
        previous_pose=None,
        render_every=0,
        lidar_config=None,
        spacetime=False,
    )
    return EpisodeBoundary(
        tick=tick,
        world_snapshot=world_snapshot,
        env_snapshot=env_snapshot.to_bytes(),
        agent_observation=_agent_payload(branch_id, tick),
        oracle_state=_oracle_payload(branch_id, tick, world_hash),
    )


def _record(
    transition_id: int,
    branch_id: int,
    tick: int,
    before_hash: int,
    after_hash: int,
    *,
    move: int,
    reward: float,
    position_cells_per_step: float = 1.0,
    intervention: bool = False,
) -> TransitionRecord:
    event_id = branch_id * 100 + transition_id * 10
    events = []
    if intervention:
        events.append(
            {
                "id": event_id - 1,
                "tick": tick,
                "phase": "intervention",
                "kind": "intervention_applied",
                "actor": "intervention:0",
                "target": "cell:20:20:20",
                "position": (20, 20, 20),
                "mechanism": "set_cell",
                "parent_ids": [],
                "root_cause": f"intervention:{branch_id}:0",
            }
        )
    events.extend(
        [
            {
                "id": event_id,
                "tick": tick,
                "phase": "agent_action",
                "kind": "action_applied",
                "actor": "agent:0",
                "target": "world",
                "position": (tick, 5, branch_id),
                "mechanism": "agent_action",
                "parent_ids": [],
                "root_cause": f"action:{branch_id}:{tick}",
            },
            {
                "id": event_id + 1,
                "tick": tick,
                "phase": "entity_integration",
                "kind": "agent_moved",
                "actor": "agent:0",
                "target": "agent:0",
                "position": (tick + 1, 5, branch_id),
                "mechanism": "agent_motion",
                "parent_ids": [event_id],
                "root_cause": f"action:{branch_id}:{tick}",
            },
        ]
    )
    return TransitionRecord(
        transition_id=transition_id,
        branch_id=branch_id,
        tick_before=tick,
        tick_after=tick + 1,
        before_hash=before_hash,
        after_hash=after_hash,
        action=_action(move),
        reward=reward,
        interventions=(
            ({"kind": "set_cell", "at": [20, 20, 20], "cell": 1},)
            if intervention
            else ()
        ),
        reward_components={"progress": reward},
        terminated=tick == 2,
        agent_observation_before=_agent_payload(branch_id, tick),
        oracle_state_before=_oracle_payload(branch_id, tick, before_hash),
        agent_observation=_agent_payload(branch_id, tick + 1),
        oracle_state=_oracle_payload(branch_id, tick + 1, after_hash),
        events=events,
        deltas=[
            {
                "event_id": event_id,
                "subject": {"kind": "world"},
                "field": "tick",
                "before": tick,
                "after": tick + 1,
            },
            {
                "event_id": event_id + 1,
                "subject": "agent:0",
                "field": "position",
                "before": [
                    float(tick) * position_cells_per_step,
                    5.0 * position_cells_per_step,
                    float(branch_id) * position_cells_per_step,
                ],
                "after": [
                    float(tick + 1) * position_cells_per_step,
                    5.0 * position_cells_per_step,
                    float(branch_id) * position_cells_per_step,
                ],
            },
            {
                "event_id": event_id + 1,
                "subject": "agent:0",
                "field": "velocity",
                "before": [0.0, 0.0, 0.0],
                "after": [0.1, 0.0, 0.0],
            },
        ],
    )


def _bundle(
    tmp_path,
    branch_id: int,
    hashes: tuple[int, ...],
    moves: tuple[int, ...],
    *,
    scale: float = 1.0,
    frame_id: int = 0,
    intervene_first: bool = False,
):
    writer = EpisodeBundleWriter(
        tmp_path,
        task="world_model_probe",
        seed=7,
        branch_id=branch_id,
        stem=f"branch-{branch_id}.vxbundle",
        metadata={"scale": scale, "frame_id": frame_id},
    )
    writer.set_initial_boundary(_boundary(branch_id, 0, hashes[0], "before"))
    for tick, (before_hash, after_hash, move) in enumerate(
        zip(hashes, hashes[1:], moves)
    ):
        writer.log(
            _record(
                branch_id * 10 + tick + 1,
                branch_id,
                tick,
                before_hash,
                after_hash,
                move=move,
                reward=float(move - 1),
                position_cells_per_step=scale,
                intervention=intervene_first and tick == 0,
            )
        )
    writer.set_final_boundary(
        _boundary(branch_id, len(moves), hashes[-1], "after")
    )
    return writer.save(final_hash=hashes[-1])


def test_world_model_adapter_and_baseline_cover_all_supervision_heads(tmp_path):
    factual = _bundle(tmp_path, 0, (10, 20, 30, 40), (1, 2, 1))
    counterfactual = _bundle(
        tmp_path, 1, (10, 21, 31), (1, 1), intervene_first=True
    )

    supervision = WorldModelAdapter([factual, counterfactual]).build_supervision(
        max_horizon=2
    )

    assert len(WorldModelAdapter(factual).build_supervision().single_step) == 3
    assert len(supervision.single_step) == 5
    assert len(supervision.multi_step) == 3
    assert supervision.multi_step[0].transition_ids == (1, 2)
    assert supervision.multi_step[0].after_hash == 30
    assert len(supervision.events) == 11
    assert supervision.events[1].parent_ids == (10,)
    assert len(supervision.deltas) == 15
    assert supervision.deltas[0].field == "tick"
    assert supervision.deltas[1].field == "position"
    assert len(supervision.spatial) == 5
    assert supervision.spatial[0].displacement == (1.0, 0.0, 0.0)
    assert supervision.spatial[0].relation == "adjacent"
    assert len(supervision.temporal_order) == 18
    assert {sample.first_precedes_second for sample in supervision.temporal_order} == {
        False,
        True,
    }
    assert len(supervision.time_to_event) == 9
    assert supervision.time_to_event[0].ticks_until_event == 0
    assert {sample.is_parent for sample in supervision.causal_parent} == {False, True}
    assert len(supervision.counterfactual) == 1
    pair = supervision.counterfactual[0]
    assert pair.factual.before_hash == pair.counterfactual.before_hash == 10
    assert (
        pair.factual.agent_observation_before
        == pair.counterfactual.agent_observation_before
    )
    assert pair.factual.oracle_state_before == pair.counterfactual.oracle_state_before
    assert pair.factual.branch_id == 0
    assert pair.counterfactual.branch_id == 1
    assert not pair.factual.intervention_event_ids
    assert pair.counterfactual.intervention_event_ids
    assert pair.outcome_changed

    baseline = run_world_model_baseline(supervision)
    repeated = run_world_model_baseline(supervision)
    expected_heads = (
        "single_step",
        "multi_step",
        "event",
        "delta",
        "spatial",
        "temporal_order",
        "time_to_event",
        "causal_parent",
        "counterfactual",
    )
    assert tuple(baseline.report) == expected_heads
    assert baseline == repeated
    assert all(
        baseline.report[head]["samples"] == len(baseline.predictions[head]) > 0
        for head in expected_heads
    )
    first_state_prediction = baseline.predictions["single_step"][0]
    first_state_target = supervision.single_step[0]
    assert baseline.report["single_step"]["metric"] == "copy_state_exact_match"
    assert (
        first_state_prediction["agent_observation"]
        == first_state_target.agent_observation_before
    )
    assert (
        first_state_prediction["agent_observation"]
        != first_state_target.agent_observation_target
    )


def test_world_model_temporal_sequences_never_mix_branches(tmp_path):
    first = _bundle(tmp_path, 0, (10, 20), (1,))
    second = _bundle(tmp_path, 1, (10, 21), (2,))
    adapter = WorldModelAdapter([first, second])
    # Exercise the grouping contract independently of storage's one-branch
    # bundle invariant: two branches may still share an episode identity in a
    # future multi-branch reader.
    adapter._events = tuple(
        replace(event, episode_index=0) for event in adapter._events
    )

    sequences = adapter._event_sequences()

    assert len(sequences) == 2
    assert all(len({event.branch_id for event in sequence}) == 1 for sequence in sequences)


def test_transition_windows_use_only_the_first_pre_step_views(tmp_path):
    bundle = _bundle(tmp_path, 2, (10, 20, 30), (1, 2))

    supervision = WorldModelAdapter(bundle).build_supervision(max_horizon=2)

    single = supervision.single_step[0]
    window = supervision.multi_step[0]
    expected_agent = _agent_payload(2, 0)
    expected_oracle = _oracle_payload(2, 0, 10)
    assert single.agent_observation_before == expected_agent
    assert single.oracle_state_before == expected_oracle
    assert window.agent_observation_before == expected_agent
    assert window.oracle_state_before == expected_oracle
    assert window.agent_observation == expected_agent
    assert window.oracle_state == expected_oracle
    assert window.oracle_state != _oracle_payload(2, 2, 30)
    assert single.agent_observation_target == _agent_payload(2, 1)
    assert single.oracle_state_target == _oracle_payload(2, 1, 20)
    assert window.agent_observation_target == _agent_payload(2, 2)
    assert window.oracle_state_target == _oracle_payload(2, 2, 30)
    assert window.agent_observation_before != window.agent_observation_target
    assert window.oracle_state_before != window.oracle_state_target


def test_counterfactual_pairs_require_matching_actions_and_one_intervention(
    tmp_path,
):
    control = _bundle(tmp_path / "control", 0, (10, 20), (1,))
    different_action = _bundle(
        tmp_path / "different",
        1,
        (10, 21),
        (2,),
        intervene_first=True,
    )
    no_intervention = _bundle(
        tmp_path / "untreated", 2, (10, 22), (1,)
    )

    assert not WorldModelAdapter(
        [control, different_action]
    ).build_supervision().counterfactual
    assert not WorldModelAdapter(
        [control, no_intervention]
    ).build_supervision().counterfactual


def test_spatial_supervision_normalizes_manifest_scale_to_meters(tmp_path):
    scale_one = _bundle(
        tmp_path / "scale-one",
        3,
        (10, 20),
        (1,),
        scale=1.0,
        frame_id=7,
    )
    scale_two = _bundle(
        tmp_path / "scale-two",
        3,
        (10, 20),
        (1,),
        scale=2.0,
        frame_id=7,
    )

    one = WorldModelAdapter(scale_one).build_supervision().spatial[0]
    two = WorldModelAdapter(scale_two).build_supervision().spatial[0]

    assert one.before == two.before == (0.0, 5.0, 3.0)
    assert one.displacement == two.displacement == (1.0, 0.0, 0.0)
    assert one.relation == two.relation == "adjacent"
    assert one.frame_id == two.frame_id == 7
    assert one.scale == 1.0
    assert two.scale == 2.0
    assert one.meters_per_cell == 1.0
    assert two.meters_per_cell == 0.5
    assert one.coordinate_unit == two.coordinate_unit == "meters"
