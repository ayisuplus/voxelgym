"""Metric task/expert contracts across the supported spatial scales."""

from __future__ import annotations

import numpy as np
import voxelgym_rs as rs

from voxelgym import ids
from voxelgym import experts
from voxelgym.tasks import NavigateToTarget
from voxelgym.tasks.metric import (
    agent_position_meters,
    metric_any_block_is,
    metric_cell_volume,
    metric_set_block,
    metric_set_cell_interventions,
    teleport_meters,
)
from voxelgym.tasks.probes import BuriedEscape, CollapseJudge


class ScaledWorld:
    """Small public-interface double whose engine coordinates are cells."""

    def __init__(self, scale: int):
        self.scale = scale
        self.position_cells = (0.0, 0.0, 0.0)
        self.blocks: dict[tuple[int, int, int], int] = {}

    def oracle_state(self):
        return {"scale": float(self.scale)}

    def agent_pos(self):
        return self.position_cells

    def teleport(self, x, y, z):
        self.position_cells = (x, y, z)

    def get_block(self, x, y, z):
        return self.blocks.get((x, y, z), ids.AIR)

    def set_block(self, x, y, z, cell):
        self.blocks[(x, y, z)] = cell

    def surface_y(self, x, z):
        return 4 * self.scale + (self.scale - 1)

    def give(self, item, count):
        pass


class ScaledExpertWorld(ScaledWorld):
    def __init__(self, scale: int):
        super().__init__(scale)
        self.position_cells = (0.5 * scale, 5.0 * scale, 0.5 * scale)
        self.last_find_radius = None

    def find_blocks(self, block, radius):
        self.last_find_radius = radius
        # Two refined subcells of the same canonical meter voxel.
        return [(2 * self.scale, 5 * self.scale, 0), (3 * self.scale - 1, 6 * self.scale - 1, 0)]


def test_negative_metric_cell_and_intervention_expand_to_whole_meter_volume():
    world = ScaledWorld(2)

    volume = metric_cell_volume(world, (-2, -1, 3))
    assert len(volume) == 8
    assert set(volume) == {
        (x, y, z)
        for x in (-4, -3)
        for y in (-2, -1)
        for z in (6, 7)
    }

    metric_set_block(world, (-2, -1, 3), ids.STONE)
    assert set(world.blocks) == set(volume)
    assert set(world.blocks.values()) == {ids.STONE}

    specs = metric_set_cell_interventions(world, (-2, -1, 3), ids.AIR)
    assert {tuple(spec["at"]) for spec in specs} == set(volume)
    assert all(spec == {"kind": "set_cell", "at": spec["at"], "cell": 0} for spec in specs)


def test_buried_escape_reset_and_reward_are_metric_equivalent_at_scale_1_and_2():
    outcomes = []
    for scale in (1, 2):
        world = ScaledWorld(scale)
        task = BuriedEscape()
        task.on_reset(world, np.random.default_rng(7))

        assert agent_position_meters(world) == (0.5, 5.0, 0.5)
        assert all(
            world.get_block(*cell) == ids.AIR
            for x in range(-1, 2)
            for z in range(-1, 2)
            for cell in metric_cell_volume(world, (x, 6, z))
        )

        teleport_meters(world, task.PAD)
        outcomes.append(task.reward_outcome(world))

    assert outcomes[0] == outcomes[1]
    assert outcomes[0].total == 1.0
    assert outcomes[0].terminated


def test_collapse_branch_interventions_expand_identically_in_metric_space():
    task = CollapseJudge()
    task.supported = 3

    expanded = []
    for scale in (1, 2):
        world = ScaledWorld(scale)
        specs = task._support_removal_interventions(world)
        expanded.append(specs)
        assert len(specs) == (task.SLAB_LEN - task.supported) * 3 * scale**3

    logical_1 = {tuple(spec["at"]) for spec in expanded[0]}
    logical_2 = {
        tuple(coord // 2 for coord in spec["at"])
        for spec in expanded[1]
    }
    assert logical_2 == logical_1


def test_collapse_truth_scenario_and_reward_match_in_real_scale_1_and_2_worlds():
    results = []
    for scale in (1, 2):
        task = CollapseJudge()
        scenario = task.scenario(np.random.default_rng(9))
        world = rs.PyWorld(101, "void", scenario, scale=float(scale))
        task.on_reset(world, np.random.default_rng(9))

        assert agent_position_meters(world) == (-8.5, 6.0, 0.5)
        assert metric_any_block_is(world, (-5, 6, -3), ids.TORCH)

        answer = (-4.5, 6.0, -2.5) if task.collapses else (-4.5, 6.0, 3.5)
        teleport_meters(world, answer)
        specs = task.interventions_before_step(world, {})
        for spec in specs:
            world.set_block(*spec["at"], spec["cell"])
        outcome = task.reward_outcome(world)
        results.append((task.collapses, outcome.total, outcome.termination_reason))

    assert results[0] == results[1]
    assert results[0][1:] == (1.0, "correct_answer")


def test_expert_navigation_and_block_search_use_meters_at_refined_scale():
    scale_1 = ScaledExpertWorld(1)
    scale_2 = ScaledExpertWorld(2)
    nav_1 = experts.Navigator()
    nav_2 = experts.Navigator()

    assert nav_1.toward(scale_1, 10.5, 0.5) == nav_2.toward(scale_2, 10.5, 0.5)

    op = experts.HarvestOp(ids.STONE, None, radius=7)
    assert op._nearest(scale_2) == (2, 5, 0)
    assert scale_2.last_find_radius == 14


def test_navigation_uses_floor_for_a_negative_metric_spawn_column():
    class MissRng:
        values = iter([0.0, 100.0] * 20)

        def uniform(self, *args):
            return next(self.values)

    world = ScaledWorld(2)
    world.position_cells = (-1.0, 10.0, -1.0)  # (-0.5, 5.0, -0.5) meters
    task = NavigateToTarget()
    starts = []

    def reachable(_world, x, z, r=45):
        starts.append((x, z))
        return {(-1, -1), (-21, -1)}

    task._reachable_columns = reachable
    task.on_reset(world, MissRng())
    assert starts == [(-1, -1)]
