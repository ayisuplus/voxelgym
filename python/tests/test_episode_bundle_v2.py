from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from voxelgym import ACTION_KEYS, VoxelGymEnv
from voxelgym import ids
from voxelgym.episode_bundle import (
    EpisodeBoundary,
    EpisodeBundleReader,
    EpisodeBundleWriter,
    TransitionRecord,
    detect_episode_format,
)
from voxelgym.datasets import FRAME, VoxelSequenceDataset
from voxelgym.recorder import CausalRecorder, decode_observation, encode_observation
from voxelgym.replay import verify
from voxelgym.task_state import EnvSnapshot, RewardOutcome
from voxelgym.tasks import make_task
from voxelgym.tasks.base import Task
from voxelgym.world_model import WorldModelAdapter


def _transition() -> TransitionRecord:
    before_observation = encode_observation(
        {"tick": np.asarray([8], dtype=np.uint64)}
    )
    after_observation = encode_observation(
        {"tick": np.asarray([9], dtype=np.uint64)}
    )
    return TransitionRecord(
        transition_id=11,
        branch_id=3,
        tick_before=8,
        tick_after=9,
        before_hash=101,
        after_hash=202,
        action=(1, 0, 0, 6, 4, 0, 0, 0, 0, 0),
        reward=0.25,
        reward_components={"progress": 0.25},
        terminated=False,
        truncated=False,
        termination_reason=None,
        sensor_ticks={"rgb": 8, "lidar": 9},
        agent_observation_before=before_observation,
        oracle_state_before=_oracle_payload(8, 101),
        agent_observation=after_observation,
        oracle_state=_oracle_payload(9, 202),
        events=[
            {
                "id": 1001,
                "tick": 8,
                "phase": "agent_action",
                "kind": "action_applied",
                "actor": "agent:0",
                "target": "world",
                "position": None,
                "mechanism": "agent_action",
                "parent_ids": [],
                "root_cause": "action:3:8",
            },
            {
                "id": 1002,
                "tick": 8,
                "phase": "entity_integration",
                "kind": "agent_moved",
                "actor": "agent:0",
                "target": "agent:0",
                "position": (1, 5, 2),
                "mechanism": "agent_motion",
                "parent_ids": [1001],
                "root_cause": "action:3:8",
            },
        ],
        deltas=[
            {
                "event_id": 1001,
                "subject": {"kind": "world"},
                "field": "tick",
                "before": 8,
                "after": 9,
            },
            {
                "event_id": 1002,
                "subject": "agent:0",
                "field": "position",
                "before": [0.5, 5.0, 2.5],
                "after": [1.0, 5.0, 2.5],
            }
        ],
        checkpoint=b"world-v8",
        env_checkpoint=b"env-v1",
    )


def _oracle_payload(tick: int, world_hash: int) -> bytes:
    return json.dumps(
        {"clock": {"tick": tick}, "world_hash": world_hash},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _boundary(
    tick: int,
    *,
    world_snapshot: bytes,
    agent_observation: bytes,
    oracle_state: bytes,
) -> EpisodeBoundary:
    snapshot = EnvSnapshot(
        world_snapshot=world_snapshot,
        task_state=None,
        np_random_state=np.random.default_rng(0).bit_generator.state,
        episode_seed=1,
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
        env_snapshot=snapshot.to_bytes(),
        agent_observation=agent_observation,
        oracle_state=oracle_state,
    )


def _write_bundle(
    writer: EpisodeBundleWriter,
    records: list[TransitionRecord],
    *,
    final_hash: int,
):
    first, last = records[0], records[-1]
    writer.set_initial_boundary(
        _boundary(
            first.tick_before,
            world_snapshot=b"initial-world",
            agent_observation=(
                first.agent_observation_before
                or encode_observation({"tick": np.asarray([first.tick_before])})
            ),
            oracle_state=(
                first.oracle_state_before
                or _oracle_payload(first.tick_before, int(first.before_hash or 0))
            ),
        )
    )
    for record in records:
        writer.log(record)
    writer.set_final_boundary(
        _boundary(
            last.tick_after,
            world_snapshot=b"final-world",
            agent_observation=(
                last.agent_observation
                or encode_observation({"tick": np.asarray([last.tick_after])})
            ),
            oracle_state=(
                last.oracle_state
                or _oracle_payload(last.tick_after, int(last.after_hash or final_hash))
            ),
        )
    )
    return writer.save(final_hash=final_hash)


def _record_single_traced_transition(tmp_path, *, interventions=()):
    task = make_task("navigate_to_target")
    env = VoxelGymEnv(task=task, seed=83, spacetime=True)
    initial_observation, _ = env.reset(seed=83)
    recorder = CausalRecorder(
        str(tmp_path), task.name, 83, trace_level="full", checkpoint_every=1
    )
    recorder.start(env, initial_observation)
    action = {key: 0 for key in ACTION_KEYS}
    action["pitch"] = 4
    observation, reward, terminated, truncated, info = env.step_traced(
        action,
        trace_level="full",
        interventions=interventions,
    )
    recorder.log(
        env,
        tuple(action[key] for key in ACTION_KEYS),
        reward,
        terminated,
        truncated,
        info,
        observation,
    )
    bundle = recorder.save(env.world.hash())
    env.close()
    return bundle


class _TaskDrivenIntervention(Task):
    name = "task_driven_intervention_test"
    preset = "void"

    def __init__(self):
        self.pending = True

    def interventions_before_step(self, world, action):
        if not self.pending:
            return []
        self.pending = False
        return [{"kind": "set_cell", "at": [3, 3, 3], "cell": ids.STONE}]


def test_v2_bundle_round_trips_all_transition_tables(tmp_path):
    writer = EpisodeBundleWriter(
        tmp_path,
        task="navigate_to_target",
        seed=7,
        trace_level="full",
        branch_id=3,
    )
    bundle = _write_bundle(writer, [_transition()], final_hash=202)

    assert detect_episode_format(bundle) == 2
    assert {path.name for path in bundle.iterdir()} == {
        "manifest.json",
        "transitions.parquet",
        "events.parquet",
        "deltas.parquet",
        "checkpoints.parquet",
    }
    reader = EpisodeBundleReader(bundle)
    assert reader.manifest["format_version"] == 2
    assert reader.manifest["trace_level"] == "full"
    assert reader.transitions.to_pylist()[0]["sensor_ticks"] == '{"lidar":9,"rgb":8}'
    assert reader.transitions.to_pylist()[0]["event_count"] == 2
    assert reader.transitions.to_pylist()[0]["delta_count"] == 2
    assert [event["id"] for event in reader.events.to_pylist()] == [1001, 1002]
    position_delta = next(
        delta for delta in reader.deltas.to_pylist() if delta["field"] == "position"
    )
    assert position_delta["after"] == "[1.0,5.0,2.5]"
    checkpoints = reader.checkpoints.to_pylist()
    assert [row["boundary"] for row in checkpoints] == ["initial", "final"]
    assert EnvSnapshot.from_bytes(checkpoints[-1]["env_snapshot"]).world_snapshot == b"final-world"
    reader.validate()


def test_bundle_trace_level_contract_accepts_each_valid_shape(tmp_path):
    full_writer = EpisodeBundleWriter(
        tmp_path / "full", task="probe", seed=1, trace_level="full", branch_id=3
    )
    full = _write_bundle(full_writer, [_transition()], final_hash=202)
    EpisodeBundleReader(full).validate()

    events_record = replace(
        _transition(), before_hash=None, after_hash=None, deltas=[]
    )
    events_writer = EpisodeBundleWriter(
        tmp_path / "events",
        task="probe",
        seed=1,
        trace_level="events",
        branch_id=3,
    )
    events = _write_bundle(events_writer, [events_record], final_hash=202)
    EpisodeBundleReader(events).validate()

    off_record = replace(events_record, events=[])
    off_writer = EpisodeBundleWriter(
        tmp_path / "off", task="probe", seed=1, trace_level="off", branch_id=3
    )
    off = _write_bundle(off_writer, [off_record], final_hash=202)
    EpisodeBundleReader(off).validate()


def test_empty_full_bundle_remains_valid_without_trace_rows(tmp_path):
    writer = EpisodeBundleWriter(
        tmp_path, task="probe", seed=1, trace_level="full", branch_id=3
    )

    bundle = writer.save(final_hash=0)

    reader = EpisodeBundleReader(bundle)
    reader.validate()
    assert reader.transitions.num_rows == reader.events.num_rows == reader.deltas.num_rows == 0


@pytest.mark.parametrize("tamper_counts", (False, True))
def test_full_trace_rejects_deleted_event_and_delta_tables(
    tmp_path, tamper_counts
):
    bundle = _record_single_traced_transition(tmp_path)
    events_path = bundle / "events.parquet"
    deltas_path = bundle / "deltas.parquet"
    pq.write_table(pq.read_table(events_path).slice(0, 0), events_path)
    pq.write_table(pq.read_table(deltas_path).slice(0, 0), deltas_path)
    expected = "event_count"
    if tamper_counts:
        transitions_path = bundle / "transitions.parquet"
        transitions = pq.read_table(transitions_path)
        for field in ("event_count", "delta_count"):
            transitions = transitions.set_column(
                transitions.schema.get_field_index(field),
                field,
                pa.array([0], type=pa.uint32()),
            )
        pq.write_table(transitions, transitions_path)
        expected = "action root event"

    with pytest.raises(ValueError, match=expected):
        EpisodeBundleReader(bundle).validate()


def test_full_trace_rejects_deleted_deltas_even_when_count_is_changed_to_zero(
    tmp_path,
):
    bundle = _record_single_traced_transition(tmp_path)
    deltas_path = bundle / "deltas.parquet"
    pq.write_table(pq.read_table(deltas_path).slice(0, 0), deltas_path)
    transitions_path = bundle / "transitions.parquet"
    transitions = pq.read_table(transitions_path)
    transitions = transitions.set_column(
        transitions.schema.get_field_index("delta_count"),
        "delta_count",
        pa.array([0], type=pa.uint32()),
    )
    pq.write_table(transitions, transitions_path)

    with pytest.raises(ValueError, match="world tick delta"):
        EpisodeBundleReader(bundle).validate()


def test_full_trace_allows_only_the_mandatory_tick_delta_for_a_noop(tmp_path):
    writer = EpisodeBundleWriter(
        tmp_path, task="probe", seed=1, trace_level="full", branch_id=3
    )
    bundle = _write_bundle(
        writer, [_chain_record(1, 0, 10, 11)], final_hash=11
    )

    reader = EpisodeBundleReader(bundle)
    reader.validate()
    assert reader.transitions.to_pylist()[0]["event_count"] == 1
    assert reader.transitions.to_pylist()[0]["delta_count"] == 1
    assert reader.deltas.to_pylist()[0] == {
        "transition_id": 1,
        "event_id": 10_001,
        "subject": '{"kind":"world"}',
        "field": "tick",
        "before": "0",
        "after": "1",
    }


@pytest.mark.parametrize(
    ("trace_level", "message"),
    (("events", "boundary hashes"), ("off", "boundary hashes")),
)
def test_bundle_trace_level_rejects_full_payload_under_lower_levels(
    tmp_path, trace_level, message
):
    writer = EpisodeBundleWriter(
        tmp_path / trace_level,
        task="probe",
        seed=1,
        trace_level="full",
        branch_id=3,
    )
    bundle = _write_bundle(writer, [_transition()], final_hash=202)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["trace_level"] = trace_level
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        EpisodeBundleReader(bundle).validate()


def test_bundle_full_trace_requires_boundary_hashes(tmp_path):
    record = replace(_transition(), before_hash=None, after_hash=None, deltas=[])
    writer = EpisodeBundleWriter(
        tmp_path, task="probe", seed=1, trace_level="full", branch_id=3
    )
    bundle = _write_bundle(writer, [record], final_hash=202)

    with pytest.raises(ValueError, match="require before_hash and after_hash"):
        EpisodeBundleReader(bundle).validate()


def test_bundle_validation_rejects_an_undeclared_boundary_parent(tmp_path):
    transition = _transition()
    writer = EpisodeBundleWriter(
        tmp_path, task="probe", seed=1, trace_level="full", branch_id=3
    )
    bundle = _write_bundle(writer, [transition], final_hash=202)

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["external_parent_ids"] == []
    events_path = bundle / "events.parquet"
    events = pq.read_table(events_path)
    parent_ids = pa.array([[], [9999]], type=pa.list_(pa.uint64()))
    events = events.set_column(
        events.schema.get_field_index("parent_ids"), "parent_ids", parent_ids
    )
    pq.write_table(events, events_path)

    with pytest.raises(ValueError, match="unknown parent event 9999"):
        EpisodeBundleReader(bundle).validate()


def test_bundle_writer_rejects_parents_not_present_at_its_initial_boundary(
    tmp_path,
):
    transition = _transition()
    transition.events[1]["parent_ids"] = [9999]
    writer = EpisodeBundleWriter(
        tmp_path, task="probe", seed=1, trace_level="full", branch_id=3
    )

    with pytest.raises(ValueError, match="neither recorded nor declared"):
        _write_bundle(writer, [transition], final_hash=202)


def test_format_detection_keeps_v1_parquet_read_only(tmp_path):
    legacy = tmp_path / "episode.parquet"
    pq.write_table(pa.table({"tick": pa.array([1], type=pa.uint32())}), legacy)
    legacy.with_suffix(".json").write_text(json.dumps({"task": "legacy", "seed": 0}))

    assert detect_episode_format(legacy) == 1
    reader = EpisodeBundleReader(legacy)
    assert reader.format_version == 1
    assert reader.transitions.column("tick").to_pylist() == [1]


def test_bundle_rejects_duplicate_transition_ids(tmp_path):
    writer = EpisodeBundleWriter(tmp_path, task="probe", seed=1, branch_id=3)
    writer.log(_transition())
    writer.log(_transition())

    with pytest.raises(ValueError, match="duplicate transition_id 11"):
        writer.save(final_hash=202)


def test_causal_recorder_persists_agent_oracle_trace_and_env_checkpoint(tmp_path):
    env = VoxelGymEnv(preset="void", seed=19, spacetime=True)
    initial_observation, _ = env.reset(seed=19)
    recorder = CausalRecorder(
        str(tmp_path), "causal_probe", 19, trace_level="full", branch_id=3
    )
    recorder.start(env, initial_observation)
    action = {key: 0 for key in ACTION_KEYS}
    action["pitch"] = 4
    action["move"] = 1
    observation, reward, terminated, truncated, info = env.step_traced(
        action, trace_level="full", branch_id=3
    )
    recorder.log(
        env,
        tuple(action[key] for key in ACTION_KEYS),
        reward,
        terminated,
        truncated,
        info,
        observation,
    )

    bundle = recorder.save(env.world.hash())
    reader = EpisodeBundleReader(bundle)
    reader.validate()
    row = reader.transitions.to_pylist()[0]
    restored_observation = decode_observation(row["agent_observation"])
    checkpoints = reader.checkpoints.to_pylist()

    assert row["branch_id"] == 3
    assert row["before_hash"] == env.oracle_view()["trace"]["before_hash"]
    assert row["after_hash"] == env.world.hash()
    assert set(restored_observation) == set(observation)
    assert all(
        EnvSnapshot.from_bytes(checkpoint["env_snapshot"]).world_snapshot
        == checkpoint["world_snapshot"]
        for checkpoint in checkpoints
    )


def test_causal_recorder_persists_and_replays_canonical_set_cell(tmp_path):
    bundle = _record_single_traced_transition(
        tmp_path,
        interventions=(
            {
                "type": "set_cell",
                "at": (31, np.int64(31), 31),
                "cell": np.int64(ids.STONE),
                "ignored_by_native": "not persisted",
            },
        ),
    )

    reader = EpisodeBundleReader(bundle)
    reader.validate()
    row = reader.transitions.to_pylist()[0]
    assert json.loads(row["interventions"]) == [
        {"kind": "set_cell", "at": [31, 31, 31], "cell": ids.STONE}
    ]
    assert row["external_intervention_count"] == 1
    assert verify(str(bundle), verbose=False)


def test_causal_recorder_round_trips_every_intervention_variant(tmp_path):
    specs = (
        {"kind": "set_cell", "at": [30, 30, 30], "cell": ids.STONE},
        {"kind": "teleport_agent", "position": [0.5, 10, 0.5]},
        {"kind": "set_agent_velocity", "velocity": [0, 0.25, 0]},
        {"kind": "give_item", "item": ids.STONE, "count": 2},
        {"kind": "swap_to_hotbar", "item": ids.STONE},
    )
    bundle = _record_single_traced_transition(tmp_path, interventions=specs)

    reader = EpisodeBundleReader(bundle)
    reader.validate()
    row = reader.transitions.to_pylist()[0]
    persisted = json.loads(row["interventions"])
    assert [spec["kind"] for spec in persisted] == [
        "set_cell",
        "teleport_agent",
        "set_agent_velocity",
        "give_item",
        "swap_to_hotbar",
    ]
    assert persisted[1]["position"] == [0.5, 10.0, 0.5]
    assert persisted[2]["velocity"] == [0.0, 0.25, 0.0]
    assert row["external_intervention_count"] == len(specs)
    assert verify(str(bundle), verbose=False)


def test_causal_recorder_persists_task_driven_interventions_without_reinjecting_them(
    tmp_path, monkeypatch
):
    task = _TaskDrivenIntervention()
    env = VoxelGymEnv(task=task, seed=89, spacetime=True)
    initial_observation, _ = env.reset(seed=89)
    recorder = CausalRecorder(
        str(tmp_path), task.name, 89, trace_level="full", checkpoint_every=1
    )
    recorder.start(env, initial_observation)
    action = {key: 0 for key in ACTION_KEYS}
    action["pitch"] = 4
    observation, reward, terminated, truncated, info = env.step_traced(
        action, trace_level="full"
    )
    assert info["interventions"]
    assert info["external_intervention_count"] == 0
    recorder.log(
        env,
        tuple(action[key] for key in ACTION_KEYS),
        reward,
        terminated,
        truncated,
        info,
        observation,
    )
    bundle = recorder.save(env.world.hash())
    env.close()

    reader = EpisodeBundleReader(bundle)
    reader.validate()
    row = reader.transitions.to_pylist()[0]
    assert json.loads(row["interventions"])
    assert row["external_intervention_count"] == 0
    import voxelgym.tasks as task_module

    monkeypatch.setattr(
        task_module, "make_task", lambda name: _TaskDrivenIntervention()
    )
    assert verify(str(bundle), verbose=False)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not-json", "canonical JSON"),
        ('{"kind":"set_cell"}', "JSON list"),
        (
            '[{"at":[1,2],"cell":1,"kind":"set_cell"}]',
            "invalid intervention specs",
        ),
        (
            '[{"at":[1,2,3],"cell":1,"type":"set_cell"}]',
            "canonically encoded",
        ),
    ],
)
def test_bundle_validation_rejects_malformed_intervention_specs(
    tmp_path, payload, message
):
    bundle = _record_single_traced_transition(tmp_path)
    transitions_path = bundle / "transitions.parquet"
    transitions = pq.read_table(transitions_path)
    transitions = transitions.set_column(
        transitions.schema.get_field_index("interventions"),
        "interventions",
        pa.array([payload], type=pa.string()),
    )
    pq.write_table(transitions, transitions_path)

    with pytest.raises(ValueError, match=message):
        EpisodeBundleReader(bundle).validate()


def test_bundle_validation_rejects_intervention_split_past_end(tmp_path):
    bundle = _record_single_traced_transition(tmp_path)
    transitions_path = bundle / "transitions.parquet"
    transitions = pq.read_table(transitions_path)
    transitions = transitions.set_column(
        transitions.schema.get_field_index("external_intervention_count"),
        "external_intervention_count",
        pa.array([1], type=pa.uint32()),
    )
    pq.write_table(transitions, transitions_path)

    with pytest.raises(ValueError, match="external_intervention_count"):
        EpisodeBundleReader(bundle).validate()


def test_v2_replay_verifies_world_task_events_rewards_and_sensors(tmp_path):
    task = make_task("navigate_to_target")
    env = VoxelGymEnv(task=task, seed=23, spacetime=True)
    initial_observation, _ = env.reset(seed=23)
    recorder = CausalRecorder(
        str(tmp_path),
        task.name,
        23,
        trace_level="full",
        checkpoint_every=1,
    )
    recorder.start(env, initial_observation)
    action = {key: 0 for key in ACTION_KEYS}
    action["pitch"] = 4
    action["move"] = 1
    for _ in range(2):
        observation, reward, terminated, truncated, info = env.step_traced(
            action, trace_level="full"
        )
        recorder.log(
            env,
            tuple(action[key] for key in ACTION_KEYS),
            reward,
            terminated,
            truncated,
            info,
            observation,
        )
    bundle = recorder.save(env.world.hash())

    assert verify(str(bundle), verbose=False)
    supervision = WorldModelAdapter(bundle).build_supervision(max_horizon=2)
    assert supervision.deltas
    assert supervision.spatial


def test_v2_replay_restores_recorded_spacetime_false(tmp_path):
    task = make_task("navigate_to_target")
    env = VoxelGymEnv(task=task, seed=29, spacetime=False)
    initial_observation, _ = env.reset(seed=29)
    recorder = CausalRecorder(
        str(tmp_path), task.name, 29, trace_level="full", checkpoint_every=1
    )
    recorder.start(env, initial_observation)
    action = {key: 0 for key in ACTION_KEYS}
    action["pitch"] = 4
    observation, reward, terminated, truncated, info = env.step_traced(
        action, trace_level="full"
    )
    recorder.log(
        env,
        tuple(action[key] for key in ACTION_KEYS),
        reward,
        terminated,
        truncated,
        info,
        observation,
    )
    bundle = recorder.save(env.world.hash())

    assert EpisodeBundleReader(bundle).manifest["metadata"]["spacetime"] is False
    assert verify(str(bundle), verbose=False)


def test_v2_replay_rejects_tampered_oracle_payload(tmp_path):
    task = make_task("navigate_to_target")
    env = VoxelGymEnv(task=task, seed=31, spacetime=True)
    initial_observation, _ = env.reset(seed=31)
    recorder = CausalRecorder(str(tmp_path), task.name, 31, trace_level="full")
    recorder.start(env, initial_observation)
    action = {key: 0 for key in ACTION_KEYS}
    action["pitch"] = 4
    observation, reward, terminated, truncated, info = env.step_traced(
        action, trace_level="full"
    )
    recorder.log(
        env,
        tuple(action[key] for key in ACTION_KEYS),
        reward,
        terminated,
        truncated,
        info,
        observation,
    )
    bundle = recorder.save(env.world.hash())
    transitions_path = bundle / "transitions.parquet"
    table = pq.read_table(transitions_path)
    tampered = pa.array([b'{"tampered":true}'], type=pa.binary())
    table = table.set_column(
        table.schema.get_field_index("oracle_state"), "oracle_state", tampered
    )
    pq.write_table(table, transitions_path)

    assert not verify(str(bundle), verbose=False)


def test_v2_replay_restores_a_mid_episode_branch_from_its_initial_boundary(tmp_path):
    task = make_task("navigate_to_target")
    source = VoxelGymEnv(task=task, seed=37, spacetime=True)
    observation, _ = source.reset(seed=37)
    action = {key: 0 for key in ACTION_KEYS}
    action["pitch"] = 4
    action["move"] = 1
    observation, _, _, _, _ = source.step_traced(
        action, trace_level="full", branch_id=7
    )
    branch = source.fork()
    branch_hash = branch.world.hash()

    recorder = CausalRecorder(
        str(tmp_path),
        task.name,
        37,
        trace_level="full",
        branch_id=7,
        checkpoint_every=1,
        spacetime=True,
    )
    recorder.start(branch, observation)
    next_observation, reward, terminated, truncated, info = branch.step_traced(
        action, trace_level="full", branch_id=7
    )
    recorder.log(
        branch,
        tuple(action[key] for key in ACTION_KEYS),
        reward,
        terminated,
        truncated,
        info,
        next_observation,
    )
    bundle = recorder.save(branch.world.hash())

    reader = EpisodeBundleReader(bundle)
    reader.validate()
    row = reader.transitions.to_pylist()[0]
    boundaries = reader.checkpoints.to_pylist()
    assert row["tick_before"] == 1
    assert row["before_hash"] == branch_hash
    assert decode_observation(row["agent_observation_before"]).keys() == observation.keys()
    assert [boundary["boundary"] for boundary in boundaries] == ["initial", "final"]
    assert EnvSnapshot.from_bytes(boundaries[0]["env_snapshot"]).world_snapshot == boundaries[0][
        "world_snapshot"
    ]
    assert verify(str(bundle), verbose=False)


def test_recorder_declares_only_trace_lineage_present_at_mid_episode_start(
    tmp_path,
):
    env = VoxelGymEnv(preset="void", seed=41)
    env.reset(seed=41)
    env.world.set_block(5, 5, 5, ids.STONE)
    env.world.set_block(5, 6, 5, ids.SAND)
    idle = tuple(4 if key == "pitch" else 0 for key in ACTION_KEYS)
    env.world.step(idle)
    env.world.apply_intervention(
        {"kind": "set_cell", "at": [5, 5, 5], "cell": 0},
        trace_level="full",
        branch_id=11,
        intervention_id=8,
    )
    scheduled = env.world.step_traced(
        idle, trace_level="full", branch_id=11
    )
    schedule_event = next(
        event
        for event in scheduled["events"]
        if event["kind"] == "block_fall_scheduled"
    )
    assert env.world.trace_external_parent_ids() == [schedule_event["id"]]

    observation = env._obs()
    recorder = CausalRecorder(
        str(tmp_path), "branch_lineage", 41, trace_level="full", branch_id=11
    )
    recorder.start(env, observation)
    action = {key: value for key, value in zip(ACTION_KEYS, idle, strict=True)}
    next_observation, reward, terminated, truncated, info = env.step_traced(
        action, trace_level="full", branch_id=11
    )
    recorder.log(
        env,
        idle,
        reward,
        terminated,
        truncated,
        info,
        next_observation,
    )
    bundle = recorder.save(env.world.hash())

    reader = EpisodeBundleReader(bundle)
    reader.validate()
    assert reader.manifest["external_parent_ids"] == [schedule_event["id"]]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (("steps", 2, "manifest steps"), ("final_hash", 999, "final_hash")),
)
def test_bundle_validation_checks_manifest_against_terminal_boundary(
    tmp_path, field, value, message
):
    writer = EpisodeBundleWriter(tmp_path, task="probe", seed=1, branch_id=3)
    bundle = _write_bundle(
        writer, [_chain_record(1, 0, 10, 11)], final_hash=11
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        EpisodeBundleReader(bundle).validate()


def test_bundle_validation_rejects_unknown_checkpoint_schema_version(tmp_path):
    writer = EpisodeBundleWriter(tmp_path, task="probe", seed=1, branch_id=3)
    bundle = _write_bundle(
        writer, [_chain_record(1, 0, 10, 11)], final_hash=11
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_schema_version"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint schema version"):
        EpisodeBundleReader(bundle).validate()


@pytest.mark.parametrize("missing", ("initial", "final"))
def test_bundle_validation_requires_both_terminal_boundaries(tmp_path, missing):
    writer = EpisodeBundleWriter(tmp_path, task="probe", seed=1, branch_id=3)
    bundle = _write_bundle(
        writer, [_chain_record(1, 0, 10, 11)], final_hash=11
    )
    path = bundle / "checkpoints.parquet"
    table = pq.read_table(path)
    keep = [value != missing for value in table.column("boundary").to_pylist()]
    pq.write_table(table.filter(pa.array(keep)), path)

    with pytest.raises(ValueError, match=missing):
        EpisodeBundleReader(bundle).validate()


def _chain_record(
    transition_id: int,
    tick: int,
    before_hash: int,
    after_hash: int,
    *,
    branch_id: int = 3,
) -> TransitionRecord:
    before_observation = encode_observation(
        {"tick": np.asarray([tick], dtype=np.uint64)}
    )
    after_observation = encode_observation(
        {"tick": np.asarray([tick + 1], dtype=np.uint64)}
    )
    return replace(
        _transition(),
        transition_id=transition_id,
        branch_id=branch_id,
        tick_before=tick,
        tick_after=tick + 1,
        before_hash=before_hash,
        after_hash=after_hash,
        agent_observation_before=before_observation,
        oracle_state_before=_oracle_payload(tick, before_hash),
        agent_observation=after_observation,
        oracle_state=_oracle_payload(tick + 1, after_hash),
        events=[
            {
                "id": 10_000 + transition_id,
                "tick": tick,
                "phase": "agent_action",
                "kind": "action_applied",
                "actor": "agent:0",
                "target": "world",
                "position": None,
                "mechanism": "agent_action",
                "parent_ids": [],
                "root_cause": {
                    "kind": "action",
                    "branch_id": branch_id,
                    "tick": tick,
                },
            }
        ],
        deltas=[
            {
                "event_id": 10_000 + transition_id,
                "subject": {"kind": "world"},
                "field": "tick",
                "before": tick,
                "after": tick + 1,
            }
        ],
        checkpoint=None,
        env_checkpoint=None,
        sensor_ticks={},
    )


def test_bundle_validation_requires_manifest_branch_to_match_rows(tmp_path):
    writer = EpisodeBundleWriter(tmp_path, task="probe", seed=1, branch_id=4)
    record = _chain_record(1, 0, 10, 11, branch_id=3)
    bundle = _write_bundle(writer, [record], final_hash=11)

    with pytest.raises(ValueError, match="manifest branch"):
        EpisodeBundleReader(bundle).validate()


def test_bundle_validation_requires_contiguous_transition_ticks(tmp_path):
    writer = EpisodeBundleWriter(tmp_path, task="probe", seed=1, branch_id=3)
    records = [_chain_record(1, 0, 10, 11), _chain_record(2, 2, 11, 12)]
    bundle = _write_bundle(writer, records, final_hash=12)

    with pytest.raises(ValueError, match="contiguous"):
        EpisodeBundleReader(bundle).validate()


def test_bundle_validation_requires_transition_hash_chain(tmp_path):
    writer = EpisodeBundleWriter(tmp_path, task="probe", seed=1, branch_id=3)
    records = [_chain_record(1, 0, 10, 11), _chain_record(2, 1, 99, 12)]
    bundle = _write_bundle(writer, records, final_hash=12)

    with pytest.raises(ValueError, match="hash chain"):
        EpisodeBundleReader(bundle).validate()


def test_bundle_declares_causal_parents_from_before_an_arbitrary_branch_point(
    tmp_path,
):
    writer = EpisodeBundleWriter(
        tmp_path,
        task="probe",
        seed=1,
        branch_id=3,
        external_parent_ids=[991, 992],
    )
    record = _chain_record(1, 8, 10, 11)
    action_root = record.events[0]
    record.events = [
        action_root,
        {
            "id": 1002,
            "tick": 8,
            "phase": "scheduled",
            "kind": "block_changed",
            "actor": None,
            "target": "cell:0:7:0",
            "position": (0, 7, 0),
            "mechanism": "falling_block",
            "parent_ids": [991],
            "root_cause": {"kind": "intervention", "intervention_id": 4},
        }
    ]
    bundle = _write_bundle(writer, [record], final_hash=11)

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["external_parent_ids"] == [991]
    EpisodeBundleReader(bundle).validate()
    supervision = WorldModelAdapter(bundle).build_supervision()
    inherited = next(event for event in supervision.events if event.parent_ids)
    assert inherited.parent_ids == (991,)
    assert all(not sample.is_parent for sample in supervision.causal_parent)

    manifest["external_parent_ids"] = [991, 992]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="boundary ancestry"):
        EpisodeBundleReader(bundle).validate()


def test_bundle_validation_rejects_future_sensor_ticks(tmp_path):
    writer = EpisodeBundleWriter(tmp_path, task="probe", seed=1, branch_id=3)
    record = _chain_record(1, 0, 10, 11)
    record.sensor_ticks = {"rgb": 2}
    bundle = _write_bundle(writer, [record], final_hash=11)

    with pytest.raises(ValueError, match="future sensor tick"):
        EpisodeBundleReader(bundle).validate()


@pytest.mark.parametrize(
    "field",
    [
        "agent_observation_before",
        "oracle_state_before",
        "agent_observation",
        "oracle_state",
    ],
)
def test_bundle_validation_requires_agent_and_oracle_payloads(tmp_path, field):
    writer = EpisodeBundleWriter(tmp_path, task="probe", seed=1, branch_id=3)
    record = _chain_record(1, 0, 10, 11)
    setattr(record, field, None)
    bundle = _write_bundle(writer, [record], final_hash=11)

    with pytest.raises(ValueError, match=field):
        EpisodeBundleReader(bundle).validate()


def test_video_dataset_consumes_v2_transition_frames(tmp_path):
    writer = EpisodeBundleWriter(tmp_path, task="video_probe", seed=1, branch_id=3)
    transition = _transition()
    transition.rgb = np.full((FRAME, FRAME, 3), 17, dtype=np.uint8).tobytes()
    transition.depth = np.full((FRAME, FRAME), 2.5, dtype=np.float16).tobytes()
    transition.seg = np.full((FRAME, FRAME), 9, dtype=np.uint16).tobytes()
    _write_bundle(writer, [transition], final_hash=transition.after_hash)

    dataset = VoxelSequenceDataset(str(tmp_path), seq_len=1, split="train")
    rgb, actions, depth, seg = dataset[0]

    assert tuple(rgb.shape) == (1, FRAME, FRAME, 3)
    assert int(rgb[0, 0, 0, 0]) == 17
    assert actions.shape == (1, len(ACTION_KEYS))
    assert float(depth[0, 0, 0]) == 2.5
    assert int(seg[0, 0, 0]) == 9
