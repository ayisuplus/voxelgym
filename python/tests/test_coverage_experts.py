"""Small deterministic behavior tests for scripted expert stages."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from voxelgym import experts, ids
from voxelgym.env import ACTION_KEYS


class ExpertWorld:
    def __init__(self, pos=(0.5, 5.0, 0.5)):
        self.pos = pos
        self.blocks = {}
        self.default = ids.AIR
        self.surface = 4
        self.items = {}
        self.finds = {}
        self.swap_slot = 2
        self.tick_value = 0
        self.crosshair_value = None
        self.drop_values = []
        self.pose = np.array([*pos, 0.0, 0.0, 1.0], dtype=np.float32)
        self.furnace = (0, False, 0)

    def agent_pos(self):
        return self.pos

    def get_block(self, x, y, z):
        return self.blocks.get((x, y, z), self.default)

    def set_block(self, x, y, z, value):
        self.blocks[(x, y, z)] = value

    def surface_y(self, x, z):
        return self.surface

    def count_item(self, item):
        return self.items.get(item, 0)

    def swap_to_hotbar(self, item):
        self.last_swap = item
        return self.swap_slot

    def find_blocks(self, block, radius):
        return list(self.finds.get(block, []))

    def crosshair(self):
        return self.crosshair_value

    def tick(self):
        return self.tick_value

    def drops_of(self, item):
        return list(self.drop_values)

    def obs_pose(self):
        return self.pose

    def furnace_state(self, *pos):
        return self.furnace


class NavStub:
    def __init__(self):
        self.calls = []
        self.stall = False

    def toward(self, world, tx, tz, allow_descent=False):
        self.calls.append((tx, tz, allow_descent))
        return experts.act(move=1, yaw=7, pitch=4)

    def stalled(self, x, z):
        return self.stall


def test_geometry_actions_and_navigator_recovery():
    assert experts.yaw_bucket_toward(0, 1) == 0
    assert experts.yaw_bucket_toward(1, 0) == 18
    assert experts.fwd_vec(0) == pytest.approx((0.0, 1.0))
    assert experts.aim_at((0, 0, 0), 0, -2, 0) == (0, 8)
    assert experts.aim_at((0, 0, 0), 0, 2, 0) == (0, 0)
    assert experts.aim_at((0, 0, 0), 1, 0, 0)[0] == 18
    assert experts.act(move=2)["move"] == 2 and len(experts.act()) == len(ACTION_KEYS)

    world = ExpertWorld()
    nav = experts.Navigator()
    # Ten identical observations establish a stall; subsequent calls trigger jump then detour.
    actions = [nav.toward(world, 10, 0) for _ in range(25)]
    assert any(a["jump"] for a in actions)
    assert nav._detour > 0

    # A cliff without water starts a detour; water at the feet forces swim-up.
    cliff = ExpertWorld(pos=(0.5, 10.0, 0.5))
    cliff.surface = 1
    cliff_nav = experts.Navigator()
    action = cliff_nav.toward(cliff, 0.5, 10.0)
    assert cliff_nav._detour == 15 and action["yaw"] != 0
    cliff.blocks[(0, 10, 0)] = ids.WATER
    assert cliff_nav.toward(cliff, 0.5, 10.0, allow_descent=True)["jump"] == 1


def test_t1_expert_navigates_then_mines_watchdog_obstacle():
    task = SimpleNamespace(target=(10.5, 5.0, 0.5))
    world = ExpertWorld()
    expert = experts.T1Expert(task)
    expert.nav = NavStub()
    assert expert.act(world)["move"] == 1
    world.tick_value = 200
    # Place a blocking cell in the forward search fan.
    world.blocks[(1, 5, 0)] = ids.STONE
    mined = expert.act(world)
    assert mined["mine"] == 1 and expert._focus is not None
    assert expert.act(world)["mine"] == 1
    world.blocks[expert._focus] = ids.AIR
    expert.act(world)
    assert expert._focus is None


def test_place_op_detects_existing_places_and_handles_geometry():
    sink = []
    world = ExpertWorld()
    op = experts.PlaceOp(ids.CRAFTING_TABLE, sink.append)
    assert not op.done(world)
    world.finds[ids.CRAFTING_TABLE] = [(2, 5, 0)]
    assert op.act(world)["place"] == 0
    assert sink == [(2, 5, 0)] and op.done(world)
    assert op.act(world) == experts.act()

    missing = experts.PlaceOp(ids.FURNACE, sink.append)
    world.finds.clear()
    world.swap_slot = -1
    assert missing.act(world) == experts.act()

    # Valid two-ahead pose produces a direct ground placement.
    world.swap_slot = 1
    world.blocks[(2, 4, 0)] = ids.STONE
    direct = experts.PlaceOp(ids.FURNACE, sink.append)
    a = direct.act(world)
    assert a["place"] == 1 and a["pitch"] == 7

    # Pocket widening keeps mining until the side cell becomes air.
    pocket = experts.PlaceOp(ids.FURNACE, sink.append)
    pocket._attempt = 149
    world.blocks.clear()
    world.default = ids.STONE
    a = pocket.act(world)
    assert a["mine"] == 1 and pocket._digging is not None
    world.blocks[pocket._digging[0]] = ids.AIR
    pocket._attempt = 0
    pocket.act(world)
    assert pocket._digging is None

    # Reposition and generic placement-cycle paths remain valid actions.
    world.default = ids.AIR
    world.blocks.clear()
    reposition = experts.PlaceOp(ids.FURNACE, sink.append)
    reposition._reposition = 2
    assert reposition.act(world)["move"] == 2
    cycle = experts.PlaceOp(ids.FURNACE, sink.append)
    cycle._attempt = 47
    assert cycle.act(world)["place"] == 1 and cycle._reposition == 6


def test_harvest_targeting_mining_and_sweep_helpers():
    world = ExpertWorld()
    op = experts.HarvestOp(ids.STONE, None, drop=ids.COBBLESTONE, at_fn=lambda: (2, 5, 0))
    assert op._nearest(world) is None
    world.blocks[(2, 5, 0)] = ids.STONE
    assert op._nearest(world) == (2, 5, 0)

    scan = experts.HarvestOp(ids.STONE, None, drop=ids.COBBLESTONE)
    assert scan._nearest(world) is None
    world.finds[ids.STONE] = [(8, 8, 0), (3, 4, 0)]
    world.blocks[(0, 7, 0)] = ids.STONE  # underground: prefer below-ish target
    assert scan._nearest(world) == (3, 4, 0)
    assert scan._reachable(world, (2, 5, 0))
    assert not scan._reachable(world, (0, 20, 0))

    scan.focus = (2, 5, 0)
    first = scan._mine_focus(world, 3)
    assert first["mine"] == 1 and len(scan._cal) == 7
    world.crosshair_value = ((2, 5, 0), ids.STONE)
    held = scan._mine_focus(world, 3)
    assert held["yaw"] == first["yaw"] and scan._hold_cell == (2, 5, 0)
    world.crosshair_value = ((1, 5, 0), ids.WATER)
    assert scan._mine_focus(world, 3)["mine"] == 1

    assert experts.HarvestOp._underground(world, *world.pos)
    world.blocks.clear()
    assert not experts.HarvestOp._underground(world, *world.pos)
    world.blocks[(0, 7, 1)] = ids.STONE
    swept = scan._sweep_cell(world, yaw=0, reach=1.0)
    assert swept is not None
    world.blocks.clear()
    assert scan._sweep_cell(world, yaw=0, reach=1.0) is None
    assert scan._burrow_cell(world, 0) == (0, 4, 2)


def test_harvest_watchdog_focus_stalls_drops_and_target_cadence():
    world = ExpertWorld()
    op = experts.HarvestOp(ids.LOG, ids.ITEM_WOODEN_PICKAXE)
    op.nav = NavStub()
    assert op._watchdog(world, 0.5, 0.5) is None
    world.tick_value = 50
    assert op._watchdog(world, 3.5, 0.5) is None
    world.items[ids.LOG] = 1
    assert op._watchdog(world, 3.5, 0.5) is None
    world.tick_value = 300
    assert op._watchdog(world, 3.5, 0.5)["move"] == 1
    world.tick_value += 1
    assert op._watchdog(world, 3.5, 0.5)["move"] == 1

    # Finishing a fluid-refilled focus clears calibration and schedules a short walk.
    op.focus = (1, 5, 0)
    world.blocks[op.focus] = ids.WATER
    assert op._continue_focus(world, 0) is None
    assert op.focus is None and op.walk == 2
    op.focus = (0, 20, 0)
    world.blocks[op.focus] = ids.STONE
    assert op._continue_focus(world, 0) is None and op.focus is None
    op.focus = (2, 5, 0)
    world.blocks[op.focus] = ids.LOG
    assert op._continue_focus(world, 0)["mine"] == 1

    op.nav.stall = True
    op._stall = 3
    assert op._stall_rescue(world, *world.pos, 1)["jump"] == 1
    op._stall = 8
    world.blocks[(0, 6, 1)] = ids.STONE
    assert op._stall_rescue(world, *world.pos, 1)["mine"] == 1
    world.blocks.clear()
    op._stall = 8
    assert op._stall_rescue(world, *world.pos, 1) is None
    assert op.no_drops_until == world.tick() + 100

    op.no_drops_until = world.tick() + 1
    assert op._pursue_drops(world, *world.pos, 2) is None
    world.tick_value += 2
    assert op._pursue_drops(world, *world.pos, 2) is None
    world.drop_values = [(5.0, 5.0, 0.5), (0.6, 5.0, 0.5)]
    near = op._pursue_drops(world, *world.pos, 2)
    assert near["hotbar"] == 2 and near["move"] == 0
    world.drop_values = [(5.0, 5.0, 0.5)]
    assert op._pursue_drops(world, *world.pos, 2)["move"] == 1

    op.target = None
    op.target_tick = -100
    world.finds[ids.LOG] = [(3, 5, 0)]
    world.blocks[(3, 5, 0)] = ids.LOG
    assert op._acquire_target(world, 2) is None and op.target == (3, 5, 0)
    world.blocks[(3, 5, 0)] = ids.AIR
    assert op._acquire_target(world, 2)["hotbar"] == 2 and op.target is None


def test_harvest_descent_explore_approach_and_pipeline():
    world = ExpertWorld()
    op = experts.HarvestOp(ids.DIAMOND_ORE, None, deep=True, mine_level=3)
    op.nav = NavStub()

    # Below-level target latches a staircase; solid column yields a mining action.
    target = (3, 2, 3)
    world.default = ids.STONE
    a = op._descend(world, target, *world.pos, 0)
    assert a["mine"] == 1 and op.desc is not None
    # Clearing the queued cell eventually switches to walking.
    op.focus = None
    op.desc["queue"] = []
    world.default = ids.AIR
    world.blocks[(0, 2, 1)] = ids.STONE  # cave guard stops before a 3-cell void
    assert op._descend(world, target, *world.pos, 0) is None

    # Deep targetless descent starts at the configured mine level.
    op.desc = None
    world.default = ids.STONE
    assert op._descend(world, None, *world.pos, 0)["mine"] == 1
    op.desc = {"yaw": 0, "walk": 2, "queue": [], "target_y": 0, "walk_from_y": 5.0}
    assert op._descend(world, None, *world.pos, 0)["move"] == 1
    op.desc = {"yaw": 0, "walk": 1, "queue": [], "target_y": 10}
    op.deep = False
    assert op._descend(world, None, *world.pos, 0) is None

    # Surface exploration delegates to navigation; underground clears the tunnel.
    world.default = ids.AIR
    world.blocks.clear()
    op.deep = False
    assert op._explore(world, *world.pos, 1)["move"] == 1
    op.explore_yaw = 0
    world.blocks[(0, 7, 0)] = ids.STONE
    world.blocks[(0, 6, 1)] = ids.STONE
    assert op._explore(world, *world.pos, 1)["mine"] == 1
    world.blocks[(0, 6, 1)] = ids.AIR
    world.blocks[(0, 5, 1)] = ids.AIR
    op._open_streak = 0
    assert op._explore(world, *world.pos, 1)["move"] == 1

    # Approach: back off a vertical dead cone, sink toward underwater ore,
    # mine through underground rock, then navigate in open terrain.
    world.blocks.clear()
    assert op._approach(world, (0, 2, 0), *world.pos, 1)["move"] == 2
    world.blocks[(0, 5, 0)] = ids.WATER
    assert op._approach(world, (3, 2, 0), *world.pos, 1)["jump"] == 0
    world.blocks[(0, 5, 0)] = ids.AIR
    world.blocks[(0, 7, 0)] = ids.STONE
    world.blocks[(1, 6, 0)] = ids.STONE
    assert op._approach(world, (5, 5, 0), *world.pos, 1)["mine"] == 1
    world.blocks.clear()
    assert op._approach(world, (5, 5, 0), *world.pos, 1)["move"] == 1

    # The public stage action equips tools and mines an immediately reachable target.
    pipeline = experts.HarvestOp(ids.LOG, ids.ITEM_WOODEN_PICKAXE)
    world.items[ids.ITEM_WOODEN_PICKAXE] = 1
    world.finds[ids.LOG] = [(2, 5, 0)]
    world.blocks[(2, 5, 0)] = ids.LOG
    world.tick_value = 20
    result = pipeline.act(world)
    assert result["mine"] == 1 and result["hotbar"] == world.swap_slot
    pipeline.walk = 1
    pipeline.focus = None
    pipeline.target = None
    pipeline.target_tick = world.tick_value
    world.finds.clear()
    assert pipeline.act(world)["move"] == 1


def test_craft_smelt_and_gather_stage_progression():
    world = ExpertWorld()
    assert experts.CraftOp(5).act(world)["craft"] == 5

    smelt = experts.SmeltOp(lambda: (3, 5, 0), want=1)
    smelt.nav = NavStub()
    assert not smelt.done(world)
    assert smelt.act(world)["move"] == 1
    world.pos = (3.5, 5.0, 0.5)
    world.furnace = (1, True, 2)
    assert smelt.act(world)["use"] == 1
    world.furnace = (0, False, 2)
    world.items[ids.IRON_ORE] = 1
    assert smelt.act(world)["use"] == 1
    world.furnace = (2, False, 2)
    assert smelt.act(world)["use"] == 0
    world.items[ids.ITEM_IRON_INGOT] = 1
    assert smelt.done(world)

    short = experts.GatherExpert("collect_log", seed=1)
    assert len(short.stages) == 10
    assert short.act(world)["mine"] in (0, 1)  # first incomplete stage delegates to HarvestOp
    complete = experts.GatherExpert("mine_diamond", seed=1)
    # Satisfy every guard without exercising implementation-specific op state.
    complete.stages = [experts.Stage(lambda w: True, experts.CraftOp(1))]
    assert complete.act(world) == experts.act()
    assert complete._idx == 1
    assert len(experts.GatherExpert("smelt_iron").stages) < len(experts.GatherExpert("mine_diamond").stages)
    assert len(experts.GatherExpert("craft_iron_pickaxe").stages) < len(experts.GatherExpert("mine_diamond").stages)


def _stub_nav(expert):
    expert.nav = NavStub()
    return expert


def test_probe_experts_emit_actions_for_each_stage_branch(monkeypatch):
    world = ExpertWorld()

    assert _stub_nav(experts.CollapseJudgeExpert(SimpleNamespace(collapses=True))).act(world)["move"] == 1
    assert _stub_nav(experts.FirebreakJudgeExpert(SimpleNamespace(burns=False))).goal[-1] == 3.5
    assert _stub_nav(experts.PlateDoorExpert(SimpleNamespace(TARGET=(2.0, 5.0, 0.0)))).act(world)["move"] == 1

    water = _stub_nav(experts.WaterRoutingExpert(SimpleNamespace()))
    world.pos = (1.5, 4.0, 1.5)
    world.default = ids.STONE
    assert water.act(world)["mine"] == 1
    assert water.act(world)["mine"] == 1
    world.blocks[water.focus] = ids.AIR
    world.default = ids.AIR
    assert water.act(world) == experts.act()

    bridge = _stub_nav(experts.BridgeOverLavaExpert(SimpleNamespace(PAD=(18.5, 5.0, 0.5))))
    world.items[ids.PLANKS] = 2
    world.pos = (10.0, 5.0, 0.5)
    world.default = ids.AIR
    world.blocks[(10, 5, 0)] = ids.STONE
    assert bridge.act(world)["jump"] == 1
    world.blocks.clear(); world.blocks[(10, 4, 0)] = ids.STONE
    assert bridge.act(world)["move"] == 1
    world.blocks.clear(); world.blocks[(10, 3, 0)] = ids.STONE
    assert bridge.act(world)["move"] == 1
    world.blocks.clear(); world.pose[5] = 1.0
    assert bridge.act(world)["place"] == 1
    world.pose[5] = 0.0
    assert bridge.act(world)["move"] == 1
    world.pos = (17.0, 5.0, 0.5)
    assert bridge.act(world)["move"] == 1

    buried = _stub_nav(experts.BuriedEscapeExpert(SimpleNamespace(PAD=(5.5, 5.0, 5.5))))
    world.pos = (0.5, 5.0, 0.5); world.blocks.clear(); world.default = ids.AIR
    world.blocks[(0, 6, 0)] = ids.SAND
    assert buried.act(world)["mine"] == 1
    assert buried.act(world)["mine"] == 1
    world.blocks[(0, 6, 0)] = ids.AIR
    assert buried.act(world)["move"] == 1
    world.pos = (5.5, 5.0, 5.5)
    assert buried.act(world) == experts.act()

    circuit = _stub_nav(experts.CircuitDoorExpert(SimpleNamespace(two=False, target=(12.5, 5.0, 0.5))))
    world.pos = (0.5, 5.0, 0.5); world.blocks[(8, 5, 0)] = ids.LEVER
    assert circuit.act(world)["move"] == 1
    world.pos = (8.0, 5.0, 0.5)
    assert circuit.act(world)["use"] == 1
    world.blocks[(8, 5, 0)] = ids.LEVER | (1 << 12)
    assert circuit.act(world)["move"] == 1

    tnt = _stub_nav(experts.TntClearExpert(SimpleNamespace(TARGET=(14.5, 5.0, 0.5))))
    world.pos = (0.5, 5.0, 0.5); world.blocks[(7, 6, 0)] = ids.LEVER
    assert tnt.act(world)["move"] == 1
    world.pos = (7.0, 5.0, 0.5)
    assert tnt.act(world)["use"] == 1
    world.blocks[(7, 6, 0)] = ids.LEVER | (1 << 12)
    world.blocks[(10, 5, 0)] = ids.DIRT; world.blocks[(11, 5, 0)] = ids.DIRT
    assert tnt.act(world)["move"] == 1
    world.pos = (5.5, 5.0, 2.5)
    assert tnt.act(world) == experts.act()
    world.blocks[(10, 5, 0)] = world.blocks[(11, 5, 0)] = ids.AIR
    assert tnt.act(world)["move"] == 1

    task = SimpleNamespace(
        LEVER_A=(0, 5, 0), LEVER_B=(0, 5, 2), lamp=(3, 5, 1),
        goal=1, template="or",
    )
    logic = _stub_nav(experts.LogicProbeExpert(task))
    monkeypatch.setattr(logic, "_truth_table", lambda w: {(0, 0): 0, (1, 0): 1})
    world.blocks[task.LEVER_A] = ids.LEVER
    world.blocks[task.LEVER_B] = ids.LEVER
    world.blocks[task.lamp] = ids.LAMP
    world.pos = (0.5, 5.0, 0.5)
    assert logic.act(world)["use"] == 1
    assert logic.cool == logic.SETTLE
    assert logic.act(world) == experts.act()
    logic.cool = 0; world.tick_value = 8; world.blocks[task.lamp] = ids.LAMP | (1 << 12)
    assert logic.act(world) == experts.act()


def test_expert_factory_and_episode_orchestration(monkeypatch, tmp_path):
    names = {
        "navigate_to_target": experts.T1Expert,
        "collapse_judge": experts.CollapseJudgeExpert,
        "water_routing": experts.WaterRoutingExpert,
        "bridge_over_lava": experts.BridgeOverLavaExpert,
        "buried_escape": experts.BuriedEscapeExpert,
        "circuit_door": experts.CircuitDoorExpert,
        "firebreak_judge": experts.FirebreakJudgeExpert,
        "plate_door": experts.PlateDoorExpert,
        "tnt_clear": experts.TntClearExpert,
        "logic_probe": experts.LogicProbeExpert,
        "collect_log": experts.GatherExpert,
    }
    tasks = {
        "navigate_to_target": SimpleNamespace(target=(1, 5, 1)),
        "collapse_judge": SimpleNamespace(collapses=True),
        "water_routing": SimpleNamespace(),
        "bridge_over_lava": SimpleNamespace(PAD=(1, 5, 1)),
        "buried_escape": SimpleNamespace(PAD=(1, 5, 1)),
        "circuit_door": SimpleNamespace(two=False, target=(1, 5, 1)),
        "firebreak_judge": SimpleNamespace(burns=True),
        "plate_door": SimpleNamespace(TARGET=(1, 5, 1)),
        "tnt_clear": SimpleNamespace(TARGET=(1, 5, 1)),
        "logic_probe": SimpleNamespace(),
        "collect_log": SimpleNamespace(),
    }
    assert {name: type(experts.make_expert(name, tasks[name])) for name in names} == names

    class EpisodeWorld:
        def take_swap(self): return 4
        def dead(self): return False
        def hash(self): return 123

    class Env:
        def __init__(self, **kwargs):
            self.world = EpisodeWorld(); self.i = 0; self.closed = False
        def reset(self, **kwargs): return None
        def step(self, action):
            self.i += 1
            obs = {"rgb": np.zeros((1, 1, 3)), "depth": np.zeros((1, 1)), "seg": np.zeros((1, 1))}
            return obs, 1.0, self.i == 2, False, {}
        def close(self): self.closed = True

    class Expert:
        def act(self, world): return experts.act(move=1)

    logs = []
    class Recorder:
        def __init__(self, *a, **kw): pass
        def log(self, *a, **kw): logs.append((a, kw))
        def save(self, h): return str(tmp_path / "episode.parquet")

    monkeypatch.setattr(experts, "make_task", lambda name: SimpleNamespace(preset="flat"))
    monkeypatch.setattr(experts, "VoxelGymEnv", Env)
    monkeypatch.setattr(experts, "make_expert", lambda *a, **kw: Expert())
    monkeypatch.setattr("voxelgym.recorder.Recorder", Recorder)
    monkeypatch.setattr(experts, "random_action", lambda rng, zero: experts.act(yaw=3))
    result = experts.run_episode("collect_log", 2, str(tmp_path), render=True, epsilon=1.0, scale=2.0)
    assert result == (True, 2, 123, str(tmp_path / "episode.parquet"))
    assert len(logs) == 2 and logs[0][1]["swap"] == 4


def test_experts_cli_success_threshold(monkeypatch, capsys):
    outcomes = iter([(True, 1, 1, None), (False, 2, 2, None)])
    monkeypatch.setattr(experts, "task_names", lambda: ["collect_log"])
    monkeypatch.setattr(experts, "run_episode", lambda *a, **kw: next(outcomes))
    assert experts.main(["--task", "collect_log", "--episodes", "2"]) == 1
    assert "success 1/2" in capsys.readouterr().out
    monkeypatch.setattr(experts, "run_episode", lambda *a, **kw: (True, 1, 1, None))
    assert experts.main(["--task", "collect_log", "--episodes", "1"]) == 0
