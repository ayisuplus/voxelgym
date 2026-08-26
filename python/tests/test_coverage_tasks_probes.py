"""Behavior coverage for task factories, scenarios, and reward contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from voxelgym import ids
from voxelgym.tasks import AchievementTask, NavigateToTarget, make_task, task_names
from voxelgym.tasks import probes
from voxelgym.tasks.base import Task


class FakeWorld:
    def __init__(self, pos=(0.5, 5.0, 0.5), default=ids.AIR):
        self.pos = pos
        self.default = default
        self.blocks = {}
        self.items = {}
        self.teleports = []
        self.gifts = []
        self.tick_value = 0

    def agent_pos(self):
        return self.pos

    def teleport(self, *pos):
        self.pos = tuple(pos)
        self.teleports.append(tuple(pos))

    def set_block(self, x, y, z, cell):
        self.blocks[(x, y, z)] = cell

    def get_block(self, x, y, z):
        return self.blocks.get((x, y, z), self.default)

    def surface_y(self, x, z):
        return 4

    def give(self, item, count):
        self.items[item] = self.items.get(item, 0) + count
        self.gifts.append((item, count))

    def count_item(self, item):
        return self.items.get(item, 0)

    def tick(self):
        return self.tick_value

    def snapshot(self):
        return b"world"


class FixedRng:
    def __init__(self, *, random=0.0, integers=(), uniforms=()):
        self.random_value = random
        self.integer_values = iter(integers)
        self.uniform_values = iter(uniforms)

    def random(self):
        return self.random_value

    def integers(self, *args):
        return next(self.integer_values)

    def uniform(self, *args):
        return next(self.uniform_values)


def test_base_task_defaults_and_reach_reward():
    task = Task()
    world = FakeWorld()
    assert task.scenario(np.random.default_rng(0)) is None
    assert task.on_reset(world, np.random.default_rng(0)) is None
    assert task.step_reward(world) == (0.0, False)
    assert task.reach_reward(world, (10.0, 5.0, 0.5)) == (0.0, False)
    assert task.reach_reward(world, (1.0, 5.0, 0.5)) == (1.0, True)


def test_navigation_reachability_reset_fallback_and_rewards(monkeypatch):
    world = FakeWorld(pos=(0.5, 5.0, 0.5))
    task = NavigateToTarget()

    # Small height field independently pins uphill/downhill passability.
    heights = {(x, z): 62 for x in range(-1, 2) for z in range(-1, 2)}
    heights[(1, 0)] = 64  # two-cell uphill step is not traversable
    world.surface_y = lambda x, z: heights[(x, z)]
    reachable = task._reachable_columns(world, 0, 0, r=1)
    assert (0, 1) in reachable and (1, 0) not in reachable
    world.surface_y = lambda x, z: 4

    # First sample misses the supplied reachable set, forcing deterministic fallback.
    monkeypatch.setattr(task, "_reachable_columns", lambda *a, **kw: {(0, 0), (20, 0), (25, 0)})
    task.on_reset(world, FixedRng(uniforms=[0.0, 30.0] * 20))
    assert task.target == (25.5, 5.0, 0.5)
    assert len(world.blocks) == 5
    world.pos = (24.5, 5.0, 0.5)
    reward, done = task.step_reward(world)
    assert done and reward > 1.0

    # Directly sampled reachable target uses a negative surface fallback.
    task2 = NavigateToTarget()
    world2 = FakeWorld(pos=(0.0, 7.0, 0.0))
    world2.surface_y = lambda x, z: -1
    monkeypatch.setattr(task2, "_reachable_columns", lambda *a, **kw: {(20, 0), (0, 0)})
    task2.on_reset(world2, FixedRng(uniforms=[0.0, 20.0]))
    assert task2.target == (20.5, 8.0, 0.5)
    world2.pos = (1.0, 7.0, 0.0)
    assert task2.step_reward(world2)[1] is False


def test_achievement_and_task_factories_cover_all_public_names():
    world = FakeWorld()
    for name in task_names():
        task = make_task(name)
        assert task.name == name
    with pytest.raises(KeyError):
        make_task("not-a-task")
    with pytest.raises(AssertionError):
        AchievementTask("not-a-goal")
    assert AchievementTask("mine_diamond").horizon == 24000


def test_probe_scenarios_encode_expected_scene_contracts():
    # Both collapse support layouts.
    held = probes.CollapseJudge()
    held_spec = held.scenario(FixedRng(random=0.1))
    collapsing = probes.CollapseJudge()
    collapsing_spec = collapsing.scenario(FixedRng(random=0.9))
    assert held.supported == 6 and collapsing.supported == 3
    assert any(row[-1] == ids.TORCH for row in collapsing_spec)
    assert len(held_spec) == len(collapsing_spec)

    cases = [
        probes.WaterRouting(), probes.BridgeOverLava(), probes.BuriedEscape(),
        probes.CircuitDoor(False), probes.CircuitDoor(True),
        probes.FirebreakJudge(), probes.PlateDoor(), probes.TntClear(),
    ]
    lengths = {task.name: len(task.scenario(FixedRng(random=0.1))) for task in cases}
    assert lengths["water_routing"] == 9
    assert lengths["bridge_over_lava"] == 57
    assert lengths["buried_escape"] == 28
    assert lengths["circuit_door_two"] > lengths["circuit_door"]
    assert lengths["tnt_clear"] == 34

    # Exercise the no-firebreak branch too.
    no_break = probes.FirebreakJudge()
    no_break.scenario(FixedRng(random=0.9))
    assert no_break.wall_x == 5

    expected_lamps = {0: (3, 5, 1), 1: (4, 6, 1), 2: (3, 6, 1), 3: (6, 6, 1)}
    for template_idx, lamp in expected_lamps.items():
        logic = probes.LogicProbe()
        spec = logic.scenario(FixedRng(integers=[template_idx, 1, 0]))
        assert logic.lamp == lamp and spec[-1][-1] & 0xFFF in (ids.LAMP, ids.WIRE)


def test_collapse_truth_commit_and_single_answer(monkeypatch):
    task = probes.CollapseJudge()
    task.supported = 3
    clone = FakeWorld(default=ids.SAND)
    clone.blocks[(3, task.SLAB_Y, 0)] = ids.AIR
    monkeypatch.setattr(probes, "_branch_sim", lambda world, mutate, ticks: (mutate(clone), clone)[1])
    assert task._truth(FakeWorld())

    world = FakeWorld(pos=(-8.5, 6.0, 0.5))
    task.on_reset(world, np.random.default_rng(0))
    assert task.collapses
    assert task.step_reward(world) == (0.0, False)
    world.pos = (-4.5, 6.0, -2.5)
    assert task.step_reward(world) == (1.0, True)
    assert len(world.blocks) == 9
    assert task.step_reward(world) == (0.0, False)

    clone.blocks.clear()
    clone.default = ids.SAND
    assert not task._truth(FakeWorld())


def test_water_reset_validates_branch_solution_and_reward(monkeypatch):
    task = probes.WaterRouting()
    wet = FakeWorld()
    wet.blocks[task.TARGET] = ids.WATER
    monkeypatch.setattr(probes, "_branch_sim", lambda world, mutate, ticks: (mutate(wet), wet)[1])
    world = FakeWorld()
    task.on_reset(world, np.random.default_rng(0))
    assert world.pos == (3.5, 5.0, 2.5)
    assert task.step_reward(wet) == (1.0, True)
    wet.blocks[task.TARGET] = ids.AIR
    assert task.step_reward(wet) == (0.0, False)
    with pytest.raises(RuntimeError, match="unsolvable"):
        task.on_reset(world, np.random.default_rng(0))


@pytest.mark.parametrize(
    "task, start, target",
    [
        (probes.BridgeOverLava(), (8.5, 5.0, 0.5), probes.BridgeOverLava.PAD),
        (probes.BuriedEscape(), (0.5, 5.0, 0.5), probes.BuriedEscape.PAD),
        (probes.CircuitDoor(False), (5.5, 5.0, 0.5), (12.5, 5.0, 0.5)),
        (probes.CircuitDoor(True), (5.5, 5.0, 0.5), (16.5, 5.0, 0.5)),
        (probes.PlateDoor(), (5.5, 5.0, 0.5), probes.PlateDoor.TARGET),
        (probes.TntClear(), (4.5, 5.0, 0.5), probes.TntClear.TARGET),
    ],
)
def test_reach_style_probe_reset_and_reward(task, start, target):
    world = FakeWorld()
    task.on_reset(world, np.random.default_rng(0))
    assert world.pos == start
    if isinstance(task, probes.BridgeOverLava):
        assert world.gifts == [(ids.PLANKS, 20)]
    if isinstance(task, probes.BuriedEscape):
        assert len(world.blocks) == 9
    assert task.step_reward(world) == (0.0, False)
    world.pos = target
    assert task.step_reward(world) == (1.0, True)


def test_firebreak_truth_answer_and_done_guard(monkeypatch):
    task = probes.FirebreakJudge()
    task.wall_x = 5
    clone = FakeWorld(default=ids.PLANKS)
    monkeypatch.setattr(probes, "_branch_sim", lambda *a, **kw: clone)
    assert not task._truth(FakeWorld())
    clone.blocks[(5, 6, 0)] = ids.AIR
    assert task._truth(FakeWorld())
    world = FakeWorld(pos=(-8.5, 6.0, 0.5))
    task.on_reset(world, np.random.default_rng(0))
    assert task.step_reward(world) == (0.0, False)
    world.pos = (-4.5, 6.0, -2.5)
    assert task.step_reward(world) == (1.0, True)
    assert task.step_reward(world) == (0.0, False)


def test_logic_reset_goal_flip_settle_and_success(monkeypatch):
    task = probes.LogicProbe()
    task.scenario(FixedRng(integers=[0, 0, 0]))
    clone = FakeWorld()
    clone.blocks[task.lamp] = ids.LAMP | (1 << 12)
    monkeypatch.setattr(probes, "_branch_sim", lambda *a, **kw: clone)
    world = FakeWorld()
    task.on_reset(world, FixedRng(integers=[1]))
    assert task.goal == 0, "initially-satisfied sampled goal must be inverted"
    world.blocks[task.lamp] = ids.LAMP
    world.tick_value = 7
    assert task.step_reward(world) == (0.0, False)
    world.tick_value = 8
    assert task.step_reward(world) == (1.0, True)
    world.blocks[task.lamp] = ids.LAMP | (1 << 12)
    assert task.step_reward(world) == (0.0, False)


def test_probe_factory_returns_exact_types():
    expected = {
        "collapse_judge": probes.CollapseJudge,
        "water_routing": probes.WaterRouting,
        "bridge_over_lava": probes.BridgeOverLava,
        "buried_escape": probes.BuriedEscape,
        "circuit_door": probes.CircuitDoor,
        "circuit_door_two": probes.CircuitDoor,
        "firebreak_judge": probes.FirebreakJudge,
        "plate_door": probes.PlateDoor,
        "tnt_clear": probes.TntClear,
        "logic_probe": probes.LogicProbe,
    }
    assert {name: type(probes.make_probe(name)) for name in probes.PROBE_TASKS} == expected
    with pytest.raises(KeyError):
        probes.make_probe("unknown")
