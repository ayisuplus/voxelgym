from __future__ import annotations

import json
import struct

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import voxelgym_rs as rs
from voxelgym.episode_bundle import (
    EpisodeBoundary,
    EpisodeBundleReader,
    EpisodeBundleWriter,
    TransitionRecord,
    detect_episode_format,
    iter_transition_records,
)
from voxelgym.env import VoxelGymEnv
from voxelgym.recorder import Recorder, encode_observation
from voxelgym.replay import verify
from voxelgym import replay
from voxelgym.tasks import make_task
from voxelgym.task_state import EnvSnapshot, RewardOutcome


IDLE = (0, 0, 0, 0, 4, 0, 0, 0, 0, 0)


def _transition(*, events=(), deltas=()) -> TransitionRecord:
    return TransitionRecord(
        transition_id=1,
        branch_id=0,
        tick_before=0,
        tick_after=1,
        before_hash=10,
        after_hash=11,
        action=IDLE,
        reward=0.0,
        agent_observation_before=_observation(0),
        oracle_state_before=_oracle(0, 10),
        agent_observation=_observation(1),
        oracle_state=_oracle(1, 11),
        events=list(events),
        deltas=list(deltas),
    )


def _observation(tick: int) -> bytes:
    return encode_observation({"tick": np.asarray([tick], dtype=np.uint64)})


def _oracle(tick: int, world_hash: int) -> bytes:
    return json.dumps(
        {"clock": {"tick": tick}, "world_hash": world_hash},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _boundary(tick: int, world: bytes, agent: bytes, oracle: bytes) -> EpisodeBoundary:
    snapshot = EnvSnapshot(
        world_snapshot=world,
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
    return EpisodeBoundary(tick, world, snapshot.to_bytes(), agent, oracle)


def _write_bundle(writer, records, final_hash):
    first, last = records[0], records[-1]
    writer.set_initial_boundary(
        _boundary(
            first.tick_before,
            b"initial-world",
            first.agent_observation_before,
            first.oracle_state_before,
        )
    )
    for record in records:
        writer.log(record)
    writer.set_final_boundary(
        _boundary(
            last.tick_after,
            b"final-world",
            last.agent_observation,
            last.oracle_state,
        )
    )
    return writer.save(final_hash=final_hash)


def _legacy_empty_world_snapshot(v8: bytes, version: int) -> bytes:
    """Build a historical empty-world fixture from the public v8 layout.

    The source fixture deliberately has no scenario or falling entities. That
    makes the only v5-v7 layout differences the absent ClockConfig and, for
    v5, the absent per-chunk ``touched`` byte described by ADR 0001.
    """

    assert v8[:4] == b"VXG1"
    assert struct.unpack_from("<I", v8, 4)[0] == 8
    # The final two zero counts are the v8-only semantic-region and dirty
    # queue trailer for this empty fixture. Historical versions end after the
    # pending-explosion list and must not retain those bytes.
    assert v8[-8:] == b"\0" * 8
    legacy = bytearray(
        v8[:4] + struct.pack("<I", version) + v8[8:16] + v8[32:-8]
    )
    if version != 5:
        return bytes(legacy)

    scenario_count_offset = 4 + 4 + 8 + 8 + 1
    assert struct.unpack_from("<I", legacy, scenario_count_offset)[0] == 0
    physics_bytes = 17 * 8
    chunk_count_offset = scenario_count_offset + 4 + 16 + 16 + physics_bytes
    chunk_count = struct.unpack_from("<I", legacy, chunk_count_offset)[0]
    cursor = chunk_count_offset + 4
    scale_one_chunk_bytes = 16 * 128 * 16 * 2
    for _ in range(chunk_count):
        cursor += 8  # chunk x/z
        del legacy[cursor]  # v5 predates the touched flag
        cursor += scale_one_chunk_bytes
    return bytes(legacy)


def test_snapshot_v8_round_trips_reduced_clock_and_future_transition():
    source = rs.PyWorld(31, "void", dt_numerator=3, dt_denominator=60)
    source.step(IDLE)
    snapshot = bytes(source.snapshot())
    restored = rs.PyWorld(0, "void")

    restored.restore(snapshot)

    assert struct.unpack_from("<I", snapshot, 4)[0] == 8
    assert restored.clock()["dt_numerator"] == 1
    assert restored.clock()["dt_denominator"] == 20
    assert restored.hash() == source.hash()
    restored.step(IDLE)
    source.step(IDLE)
    assert restored.hash() == source.hash()


@pytest.mark.parametrize("version", [5, 6, 7])
def test_legacy_world_snapshots_restore_with_historical_twenty_hertz(version):
    source = rs.PyWorld(32, "void")
    legacy = _legacy_empty_world_snapshot(bytes(source.snapshot()), version)
    restored = rs.PyWorld(0, "void", dt_numerator=1, dt_denominator=10)

    restored.restore(legacy)

    assert restored.clock()["dt_numerator"] == 1
    assert restored.clock()["dt_denominator"] == 20
    assert restored.hash() == source.hash()
    source.step(IDLE)
    restored.step(IDLE)
    assert restored.hash() == source.hash()


@pytest.mark.parametrize(
    ("offset", "value", "message"),
    [
        (16, 0, "tick duration must be positive"),
        (24, 0, "denominator must be non-zero"),
    ],
)
def test_snapshot_v8_rejects_malformed_clock_without_replacing_world(
    offset, value, message
):
    destination = rs.PyWorld(33, "flat")
    before = destination.hash()
    malformed = bytearray(rs.PyWorld(34, "void").snapshot())
    malformed[offset : offset + 8] = int(value).to_bytes(8, "little")

    with pytest.raises(ValueError, match=message):
        destination.restore(bytes(malformed))

    assert destination.hash() == before


def test_snapshot_v8_rejects_clock_truncation_without_replacing_world():
    destination = rs.PyWorld(35, "flat")
    before = destination.hash()
    truncated_inside_clock = bytes(rs.PyWorld(36, "void").snapshot())[:24]

    with pytest.raises(ValueError, match="snapshot truncated"):
        destination.restore(truncated_inside_clock)

    assert destination.hash() == before


def test_episode_bundle_rejects_invalid_or_unsupported_manifest(tmp_path):
    invalid_json = tmp_path / "invalid-json.vxbundle"
    invalid_json.mkdir()
    (invalid_json / "manifest.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        detect_episode_format(invalid_json)

    unsupported = tmp_path / "unsupported.vxbundle"
    unsupported.mkdir()
    (unsupported / "manifest.json").write_text(
        json.dumps({"format": "voxelgym.episode", "format_version": 99}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unrecognized episode path"):
        detect_episode_format(unsupported)


def test_episode_bundle_rejects_manifest_without_table_mapping(tmp_path):
    bundle = tmp_path / "missing-files.vxbundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"format": "voxelgym.episode", "format_version": 2}),
        encoding="utf-8",
    )

    assert detect_episode_format(bundle) == 2
    with pytest.raises(ValueError, match="files mapping"):
        EpisodeBundleReader(bundle)


def test_episode_bundle_validation_rejects_missing_required_schema_column(tmp_path):
    writer = EpisodeBundleWriter(tmp_path, task="probe", seed=1, stem="schema")
    bundle = _write_bundle(writer, [_transition()], 11)
    pq.write_table(pa.table({"tick_before": [0], "tick_after": [1]}), bundle / "transitions.parquet")

    reader = EpisodeBundleReader(bundle)
    with pytest.raises(ValueError, match="schema"):
        reader.validate()


def test_episode_bundle_validation_rejects_incompatible_column_type(tmp_path):
    writer = EpisodeBundleWriter(tmp_path, task="probe", seed=1, stem="wrong-type")
    bundle = _write_bundle(writer, [_transition()], 11)
    table = pq.read_table(bundle / "transitions.parquet")
    bad_id = pa.array(["not-an-integer"], type=pa.string())
    malformed = table.set_column(
        table.schema.get_field_index("transition_id"), "transition_id", bad_id
    )
    pq.write_table(malformed, bundle / "transitions.parquet")

    with pytest.raises(ValueError, match="schema"):
        EpisodeBundleReader(bundle).validate()


def test_episode_bundle_validation_rejects_causal_cycle(tmp_path):
    common = {
        "tick": 0,
        "phase": "agent_action",
        "kind": "state_changed",
        "actor": None,
        "target": "world",
        "position": None,
        "mechanism": "test",
        "root_cause": "action:0:0",
    }
    events = [
        {**common, "id": 101, "parent_ids": [102]},
        {**common, "id": 102, "parent_ids": [101]},
    ]
    writer = EpisodeBundleWriter(tmp_path, task="probe", seed=1, stem="cycle")
    bundle = _write_bundle(writer, [_transition(events=events)], 11)

    with pytest.raises(ValueError, match="causal event graph contains a cycle"):
        EpisodeBundleReader(bundle).validate()


def test_episode_bundle_validation_rejects_parent_from_future_tick(tmp_path):
    common = {
        "phase": "agent_action",
        "kind": "state_changed",
        "actor": None,
        "target": "world",
        "position": None,
        "mechanism": "test",
        "root_cause": "action",
    }
    earlier = _transition(
        events=[{**common, "id": 201, "tick": 0, "parent_ids": [202]}]
    )
    earlier.transition_id = 100
    later = _transition(
        events=[{**common, "id": 202, "tick": 1, "parent_ids": []}]
    )
    later.transition_id = 1
    later.tick_before = 1
    later.tick_after = 2
    later.before_hash = earlier.after_hash
    later.after_hash = 12
    later.agent_observation_before = earlier.agent_observation
    later.oracle_state_before = earlier.oracle_state
    later.agent_observation = _observation(2)
    later.oracle_state = _oracle(2, 12)
    writer = EpisodeBundleWriter(tmp_path, task="probe", seed=1, stem="future")
    bundle = _write_bundle(writer, [earlier, later], 12)

    with pytest.raises(ValueError, match="future transition"):
        EpisodeBundleReader(bundle).validate()


def test_legacy_v1_detection_and_iteration_are_read_only(tmp_path):
    legacy = tmp_path / "legacy.parquet"
    pq.write_table(pa.table({"tick": pa.array([7], type=pa.uint32())}), legacy)
    legacy.with_suffix(".json").write_text(
        json.dumps({"task": "legacy", "seed": 4, "final_hash": 99}),
        encoding="utf-8",
    )
    parquet_before = legacy.read_bytes()
    sidecar_before = legacy.with_suffix(".json").read_bytes()

    assert detect_episode_format(legacy) == 1
    assert list(iter_transition_records([legacy])) == [(1, {"tick": 7})]
    assert legacy.read_bytes() == parquet_before
    assert legacy.with_suffix(".json").read_bytes() == sidecar_before
    assert not list(tmp_path.glob("*.vxbundle"))


def test_legacy_v1_replay_accepts_historical_hash_identity(tmp_path):
    task_name = "navigate_to_target"
    seed = 41
    task = make_task(task_name)
    env = VoxelGymEnv(task=task, preset=task.preset, seed=seed)
    env.reset(seed=seed)
    legacy_hash = env.world.legacy_hash_v7()
    assert legacy_hash != env.world.hash()
    shard = Recorder(str(tmp_path), task_name, seed).save(legacy_hash)

    assert verify(shard, verbose=False) is True


def test_legacy_checkpoint_hash_uses_the_snapshot_version_identity():
    class World:
        def hash(self):
            return 800

        def legacy_hash_v7(self):
            return 700

    v7 = b"VXG1" + struct.pack("<I", 7)
    v8 = b"VXG1" + struct.pack("<I", 8)

    assert replay._checkpoint_hash(World(), v7) == 700
    assert replay._checkpoint_hash(World(), v8) == 800
    assert replay._checkpoint_hash(World(), b"not-a-snapshot") == 800
