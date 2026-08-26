import json

import pytest

import voxelgym_rs as rs


IDLE = (0, 0, 0, 0, 4, 0, 0, 0, 0, 0)


def test_world_exposes_reduced_clock_and_metric_oracle_state():
    world = rs.PyWorld(
        7,
        "void",
        scale=2.0,
        dt_numerator=2,
        dt_denominator=40,
    )

    assert world.clock() == {
        "tick": 0,
        "dt_numerator": 1,
        "dt_denominator": 20,
        "seconds_per_tick": pytest.approx(0.05),
        "elapsed_numerator": 0,
        "elapsed_denominator": 1,
        "elapsed_seconds": pytest.approx(0.0),
    }
    world.teleport(4.0, 8.0, -2.0)
    state = world.oracle_state()
    assert state["frame_id"] == 0
    assert state["scale"] == pytest.approx(2.0)
    assert state["meters_per_cell"] == pytest.approx(0.5)
    assert state["position_cells"] == (4.0, 8.0, -2.0)
    assert state["position_meters"] == (2.0, 4.0, -1.0)
    assert state["velocity_state_cells"] == (0.0, 0.0, 0.0)
    assert state["velocity_meters_per_second"] == (0.0, 0.0, 0.0)
    assert state["clock"]["tick"] == 0
    assert state["entities"][0]["kind"] == "agent"
    assert state["inventory"][0] == (0, 0, 0)


def test_world_rejects_invalid_clock_configuration():
    with pytest.raises(ValueError, match="denominator"):
        rs.PyWorld(1, "void", dt_denominator=0)
    with pytest.raises(ValueError, match="positive"):
        rs.PyWorld(1, "void", dt_numerator=0)


def test_physics_overrides_use_canonical_metric_and_twenty_hertz_units():
    world = rs.PyWorld(
        2,
        "void",
        physics={"gravity": 0.08, "walk_speed": 0.2},
        scale=2.0,
        dt_numerator=1,
        dt_denominator=40,
    )

    # Vertical recurrence parameters remain canonical at 20 Hz after metric
    # scaling; the integrator applies the exact fractional half-step at 40 Hz.
    assert world.get_physics("gravity") == pytest.approx(0.08 * 2.0)
    assert world.get_physics("walk_speed") == pytest.approx(0.2 * 2.0 * 0.5)


def test_oracle_scene_objects_have_stable_semantic_ids_after_restore():
    stone = rs.block_id("stone")
    world = rs.PyWorld(3, "void", [(1, 2, 3, 4, 5, 6, stone)])
    before = world.oracle_state()["semantic_regions"]
    clone = rs.PyWorld(0, "void")
    clone.restore(world.snapshot())

    assert len(before) == 1
    assert before[0]["region_id"] > 0
    assert before[0]["structure_id"] > 0
    assert before[0]["frame_id"] == 0
    assert before[0]["bounds_cells"] == (1, 2, 3, 4, 5, 6)
    assert before[0]["bounds_meters_half_open"] == pytest.approx(
        (1.0, 2.0, 3.0, 5.0, 6.0, 7.0)
    )
    assert before[0]["cell"] == stone
    assert clone.oracle_state()["semantic_regions"] == before


def test_semantic_ids_are_order_and_scale_stable_and_support_compound_structures():
    stone = rs.block_id("stone")
    dirt = rs.block_id("dirt")
    first = (1, 2, 3, 1, 2, 3, stone)
    second = (-4, 5, -6, -2, 7, -5, dirt)
    scale_one = rs.PyWorld(4, "void", [first, second], scale=1.0)
    scale_two = rs.PyWorld(4, "void", [second, first], scale=2.0)

    ids_one = {
        region["cell"]: (region["region_id"], region["structure_id"])
        for region in scale_one.oracle_state()["semantic_regions"]
    }
    ids_two = {
        region["cell"]: (region["region_id"], region["structure_id"])
        for region in scale_two.oracle_state()["semantic_regions"]
    }
    assert ids_two == ids_one

    compound = rs.PyWorld(
        5,
        "void",
        semantic_regions=[
            (101, 900, 0, 4, 0, 1, 5, 1, stone),
            (102, 900, 2, 4, 0, 3, 5, 1, dirt),
        ],
    )
    restored = rs.PyWorld(0, "void")
    restored.restore(compound.snapshot())
    regions = restored.oracle_state()["semantic_regions"]
    assert {region["region_id"] for region in regions} == {101, 102}
    assert {region["structure_id"] for region in regions} == {900}


def test_clock_reports_horizon_and_sensor_sample_age_on_request():
    world = rs.PyWorld(8, "void", dt_numerator=1, dt_denominator=10)
    world.step(IDLE)
    world.step(IDLE)

    clock = world.clock(horizon_tick=5, sample_tick=1)

    assert clock["remaining_ticks"] == 3
    assert clock["remaining_seconds"] == pytest.approx(0.3)
    assert clock["sample_tick"] == 1
    assert (clock["sample_time_numerator"], clock["sample_time_denominator"]) == (
        1,
        10,
    )
    assert clock["sample_time_seconds"] == pytest.approx(0.1)
    assert clock["data_age_ticks"] == 1
    assert clock["data_age_seconds"] == pytest.approx(0.1)


def test_step_trace_is_json_native_and_does_not_change_world_truth():
    control = rs.PyWorld(11, "void")
    off = rs.PyWorld(11, "void")
    events = rs.PyWorld(11, "void")
    full = rs.PyWorld(11, "void")

    control.step(IDLE)
    off_outcome = off.step_traced(IDLE, trace_level="off", branch_id=4)
    event_outcome = events.step_traced(IDLE, trace_level="events", branch_id=4)
    full_outcome = full.step_traced(IDLE, trace_level="full", branch_id=4)

    assert {control.hash(), off.hash(), events.hash(), full.hash()} == {control.hash()}
    assert off_outcome["events"] == [] and off_outcome["deltas"] == []
    assert event_outcome["clock_before"]["tick"] == 0
    assert event_outcome["clock_after"]["tick"] == 1
    assert event_outcome["before_hash"] is None
    assert event_outcome["after_hash"] is None
    assert event_outcome["events"][0] == {
        "id": event_outcome["events"][0]["id"],
        "tick": 0,
        "phase": "agent_action",
        "kind": "action_applied",
        "actor": {"kind": "agent", "id": 0},
        "target": {"kind": "world"},
        "location": None,
        "mechanism": "agent_action",
        "parent_ids": [],
        "root_cause": {"kind": "action", "branch_id": 4, "tick": 0},
    }
    assert full_outcome["before_hash"] is not None
    assert full_outcome["after_hash"] == full.hash()
    assert any(delta["field_or_cell"] == "tick" for delta in full_outcome["deltas"])
    json.dumps(full_outcome)


def test_step_trace_rejects_unknown_level_without_advancing():
    world = rs.PyWorld(12, "void")
    before = world.hash()

    with pytest.raises(ValueError, match="trace level"):
        world.step_traced(IDLE, trace_level="verbose")

    assert world.hash() == before


def test_serializable_intervention_mutates_only_the_fork_and_traces_cause():
    source = rs.PyWorld(13, "void", dt_numerator=1, dt_denominator=10)
    branch = source.fork()
    stone = rs.block_id("stone")
    spec = json.loads(
        json.dumps({"kind": "set_cell", "at": [2, 5, -1], "cell": stone})
    )

    outcome = branch.apply_intervention(
        spec,
        trace_level="full",
        branch_id=8,
        intervention_id=3,
    )

    assert source.get_block(2, 5, -1) != stone
    assert branch.get_block(2, 5, -1) == stone
    assert branch.clock()["dt_denominator"] == 10
    assert outcome["event"]["kind"] == "intervention_applied"
    assert outcome["event"]["root_cause"] == {
        "kind": "intervention",
        "branch_id": 8,
        "intervention_id": 3,
    }
    cell_delta = next(
        delta for delta in outcome["deltas"] if delta["subject"]["kind"] == "cell"
    )
    assert cell_delta["subject"] == {
        "kind": "cell",
        "at": (2, 5, -1),
    }
    json.dumps(outcome)


def test_all_intervention_variants_and_branch_comparison_use_tagged_dicts():
    world = rs.PyWorld(14, "void")
    stone = rs.block_id("stone")

    world.apply_intervention({"type": "teleport_agent", "position": [1.5, 6.0, -2.5]})
    world.apply_intervention({"kind": "set_agent_velocity", "velocity": [0.1, 0.2, 0.3]})
    world.apply_intervention({"kind": "give_item", "item": stone, "count": 2})

    assert world.agent_pos() == (1.5, 6.0, -2.5)
    assert world.oracle_state()["velocity_state_cells"] == pytest.approx((0.1, 0.2, 0.3))
    assert world.count_item(stone) == 2

    select_five = list(IDLE)
    select_five[8] = 5
    world.step(tuple(select_five))
    swap = world.apply_intervention(
        {"kind": "swap_to_hotbar", "item": stone}, trace_level="full"
    )
    assert world.oracle_state()["selected_hotbar"] == 0
    assert any(delta["field_or_cell"] == "selected" for delta in swap["deltas"])

    before = world.hash()
    comparison = world.compare_branches(
        {"kind": "set_cell", "at": [3, 4, 5], "cell": stone},
        [IDLE],
        [IDLE],
    )
    assert comparison["common_before_hash"] == before
    assert comparison["diverged"] is True
    assert comparison["control_after_hash"] != comparison["treatment_after_hash"]


def test_intervention_validation_happens_before_world_mutation():
    world = rs.PyWorld(15, "void")
    before = world.hash()
    before_snapshot = bytes(world.snapshot())

    with pytest.raises(ValueError, match="intervention kind"):
        world.apply_intervention({"kind": "erase_reality"})
    with pytest.raises(ValueError, match="finite"):
        world.apply_intervention(
            {"kind": "teleport_agent", "position": [float("nan"), 0.0, 0.0]}
        )
    with pytest.raises(ValueError, match="unknown block id 4095"):
        world.apply_intervention(
            {"kind": "set_cell", "at": [100, 10, 100], "cell": 65535},
            trace_level="full",
        )
    with pytest.raises(ValueError, match="unknown item id 65535"):
        world.apply_intervention(
            {"kind": "give_item", "item": 65535, "count": 1},
            trace_level="full",
        )
    with pytest.raises(ValueError, match="equal-length"):
        world.compare_branches(
            {"kind": "give_item", "item": 1, "count": 1}, [IDLE], []
        )
    different = list(IDLE)
    different[0] = 1
    with pytest.raises(ValueError, match="identical action sequences"):
        world.compare_branches(
            {"kind": "give_item", "item": 1, "count": 1},
            [IDLE],
            [tuple(different)],
        )

    assert world.hash() == before
    assert bytes(world.snapshot()) == before_snapshot


def test_invalid_direct_cell_and_item_setters_preserve_snapshot_and_trace_allocator():
    world = rs.PyWorld(150, "void")
    control = rs.PyWorld(150, "void")
    before = bytes(world.snapshot())

    with pytest.raises(ValueError, match="unknown block id 4095"):
        world.set_block(100, 10, 100, 65535)
    with pytest.raises(ValueError, match="unknown item id 65535"):
        world.give(65535, 1)

    assert bytes(world.snapshot()) == before
    valid = {"kind": "set_cell", "at": [1, 5, 1], "cell": rs.block_id("stone")}
    actual = world.apply_intervention(valid, trace_level="events", branch_id=6)
    expected = control.apply_intervention(valid, trace_level="events", branch_id=6)
    assert actual["event"]["id"] == expected["event"]["id"]


def test_constructor_rejects_unknown_scenario_and_semantic_cells():
    with pytest.raises(ValueError, match="unknown block id 4095"):
        rs.PyWorld(151, "void", [(0, 0, 0, 0, 0, 0, 65535)])

    with pytest.raises(ValueError, match="unknown block id 4095"):
        rs.PyWorld(
            152,
            "void",
            semantic_regions=[(1, 1, 0, 0, 0, 0, 0, 0, 65535)],
        )


def test_default_intervention_ids_are_unique_within_one_boundary():
    world = rs.PyWorld(15, "void")
    stone = rs.block_id("stone")

    first = world.apply_intervention(
        {"kind": "set_cell", "at": [1, 5, 1], "cell": stone},
        trace_level="events",
        branch_id=2,
    )
    second = world.apply_intervention(
        {"kind": "set_cell", "at": [2, 5, 1], "cell": stone},
        trace_level="events",
        branch_id=2,
    )

    assert first["event"]["id"] != second["event"]["id"]


def test_spatial_relations_and_nearest_are_frame_stable():
    world = rs.PyWorld(16, "void")

    assert world.nearest([(2, 0, 0), (0, 0, 2)], origin=(0.5, 0.5, 0.5)) == (
        0,
        0,
        2,
    )
    assert world.nearest([]) is None
    assert world.within((0.0, 0.0, 0.0), (3.0, 4.0, 0.0), 5.0) is True
    assert world.within((0.0, 0.0, 0.0), (3.0, 4.0, 0.1), 5.0) is False
    assert world.adjacent((0, 0, 0), (0, 1, 0)) is True
    assert world.adjacent((0, 0, 0), (1, 1, 0)) is False
    assert world.above((2, 5, -1), (2, 3, -1)) is True
    assert world.below((2, 3, -1), (2, 5, -1)) is True


def test_vertical_velocity_oracle_uses_canonical_physical_units_at_any_tick_rate():
    hz20 = rs.PyWorld(17, "void")
    hz40 = rs.PyWorld(17, "void", dt_numerator=1, dt_denominator=40)
    for world in (hz20, hz40):
        world.apply_intervention(
            {"kind": "set_agent_velocity", "velocity": [0.0, 0.42, 0.0]},
            trace_level="off",
        )

    assert hz20.oracle_state()["velocity_meters_per_second"][1] == pytest.approx(8.4)
    assert hz40.oracle_state()["velocity_meters_per_second"][1] == pytest.approx(8.4)


def test_scale_two_reachability_anchors_to_the_live_continuous_agent_pose():
    stone = rs.block_id("stone")
    world = rs.PyWorld(
        18,
        "void",
        [
            (0, 0, 0, 5, 0, 0, stone),
            (0, 0, -1, 5, 3, -1, stone),
            (0, 0, 1, 5, 3, 1, stone),
        ],
        scale=2.0,
    )
    world.teleport(1.0, 2.0, 1.0)

    path = world.shortest_path((1, 2, 1), (8, 2, 1))

    assert path is not None
    assert path[0] == (1, 2, 1)
    assert path[-1] == (8, 2, 1)
    assert world.reachable((1, 2, 1), (8, 2, 1)) is True


def test_visibility_uses_world_solidity_and_returns_first_hit_semantics():
    world = rs.PyWorld(17, "void")
    stone = rs.block_id("stone")
    origin = (0.5, 5.5, 0.5)
    target = (3, 5, 0)

    world.set_block(*target, stone)
    assert world.visible(origin, target) is True

    world.set_block(2, 5, 0, stone)
    assert world.visible(origin, target) is False

    door = rs.block_id("door")
    world.set_block(2, 5, 0, door | (1 << 12))
    assert world.visible(origin, target) is True


def test_path_queries_use_agent_collision_shape_and_solid_components():
    world = rs.PyWorld(18, "void")
    stone = rs.block_id("stone")
    for x in range(3):
        world.set_block(x, 4, 0, stone)

    assert world.connected_component((0, 4, 0), max_visited=16) == [
        (0, 4, 0),
        (1, 4, 0),
        (2, 4, 0),
    ]
    assert world.shortest_path((0, 5, 0), (2, 5, 0), max_visited=16) == [
        (0, 5, 0),
        (1, 5, 0),
        (2, 5, 0),
    ]
    assert world.reachable((0, 5, 0), (2, 5, 0), max_visited=16) is True

    # A one-block ledge is traversable by the live jump action, so the
    # spatial graph must expose the same edge instead of treating standable
    # cells as an isolated six-neighbour lattice.
    world.set_block(1, 5, 0, stone)
    assert world.shortest_path((0, 5, 0), (2, 5, 0), max_visited=16) == [
        (0, 5, 0),
        (1, 6, 0),
        (2, 5, 0),
    ]
    assert world.reachable((0, 5, 0), (2, 5, 0), max_visited=16) is True

    # The same jump is rejected when the real body AABB has no headroom.
    world.set_block(1, 7, 0, stone)
    assert world.shortest_path((0, 5, 0), (2, 5, 0), max_visited=16) is None
    assert world.reachable((0, 5, 0), (2, 5, 0), max_visited=16) is False


def test_reachability_reuses_live_motion_for_a_survivable_five_meter_drop():
    world = rs.PyWorld(1, "void")
    stone = rs.block_id("stone")
    world.set_block(0, 9, 0, stone)
    for x in range(1, 10):
        world.set_block(x, 4, 0, stone)
    world.teleport(0.5, 10.0, 0.5)
    start = (0, 10, 0)
    goal = (5, 5, 0)

    path = world.shortest_path(start, goal, max_visited=512)
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    assert world.reachable(start, goal, max_visited=512) is True

    forward_x = (1, 0, 0, 18, 4, 0, 0, 0, 0, 0)
    idle = (0, 0, 0, 18, 4, 0, 0, 0, 0, 0)
    for tick in range(80):
        world.step(forward_x if tick < 25 else idle)
    position = world.agent_pos()
    assert tuple(int(value // 1) for value in position) == goal
    assert world.hp() > 0
    assert world.dead() is False


def test_deep_water_reachability_matches_a_live_move_and_swim_rollout():
    stone = rs.block_id("stone")
    water = rs.block_id("water")
    world = rs.PyWorld(
        21,
        "void",
        [
            (0, 0, 0, 4, 0, 2, stone),
            (0, 1, 0, 4, 4, 2, water),
            (6, 1, 1, 6, 1, 1, stone),
        ],
    )
    world.teleport(0.5, 2.0, 1.5)
    start = (0, 2, 1)
    goal = (1, 2, 1)

    # Neither endpoint has solid support immediately below its feet. They
    # are nevertheless valid controller poses because the feet cells are
    # water, and a held jump supplies the swim-up force while moving.
    assert world.get_block(*start) & 0x0FFF == water
    assert world.get_block(start[0], start[1] - 1, start[2]) & 0x0FFF == water
    assert world.shortest_path(start, goal, max_visited=32) == [start, goal]
    assert world.reachable(start, goal, max_visited=32) is True
    # The finite pool does not confer support on the dry gap at x=5 or make
    # the engine's implicit lower boundary an unbounded route to dry land.
    assert world.shortest_path(start, (6, 2, 1), max_visited=128) is None

    move_and_swim = (1, 1, 0, 18, 4, 0, 0, 0, 0, 0)
    visited = []
    for _ in range(12):
        world.step(move_and_swim)
        visited.append(tuple(int(value // 1) for value in world.agent_pos()))

    assert goal in visited
    assert world.dead() is False


@pytest.mark.parametrize(
    ("start", "goal", "jump", "max_ticks"),
    [
        ((0, 2, 1), (0, 3, 1), 1, 24),
        ((0, 3, 1), (0, 2, 1), 0, 8),
    ],
)
def test_vertical_deep_water_edges_require_a_successful_controller_rollout(
    start, goal, jump, max_ticks
):
    stone = rs.block_id("stone")
    water = rs.block_id("water")
    world = rs.PyWorld(
        22,
        "void",
        [
            (0, 0, 0, 2, 0, 2, stone),
            (0, 1, 0, 2, 5, 2, water),
        ],
    )
    world.teleport(0.5, float(start[1]), 1.5)

    assert world.shortest_path(start, goal, max_visited=32) == [start, goal]
    assert world.reachable(start, goal, max_visited=32) is True

    vertical_swim = (0, jump, 0, 0, 4, 0, 0, 0, 0, 0)
    visited = []
    for _ in range(max_ticks):
        world.step(vertical_swim)
        visited.append(tuple(int(value // 1) for value in world.agent_pos()))

    assert goal in visited
    assert world.dead() is False


def test_bounded_spatial_queries_report_limit_errors():
    world = rs.PyWorld(19, "void")
    stone = rs.block_id("stone")
    world.set_block(0, 0, 0, stone)
    world.set_block(1, 0, 0, stone)

    with pytest.raises(ValueError, match="visit limit"):
        world.connected_component((0, 0, 0), max_visited=1)
    with pytest.raises(ValueError, match="visit limit"):
        world.shortest_path((0, 1, 0), (1, 1, 0), max_visited=1)
