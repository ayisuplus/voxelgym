"""T3 physics-probe tasks + T4 circuit tasks (M3).

All probes are programmatic scenes (void/flat + ScenarioSpec). Tasks that
need solvability/truth guarantees compute them at reset by branch-simulating
a cloned world (the sim is the oracle), per plan.
"""

from __future__ import annotations

import math

import numpy as np

from .. import ids
from .base import Task


def _reg(x0, y0, z0, x1, y1, z1, cell):
    return (x0, y0, z0, x1, y1, z1, cell)


def _branch_sim(world, mutate=None, ticks: int = 300):
    """Clone the world, optionally mutate, simulate, return the clone."""
    import voxelgym_rs as rs

    scratch = rs.PyWorld(0, "void")
    scratch.restore(world.snapshot())
    if mutate is not None:
        mutate(scratch)
    for _ in range(ticks):
        scratch.step((0, 0, 0, 0, 4, 0, 0, 0, 0, 0))
    return scratch


class CollapseJudge(Task):
    """T3: sand slab with a hidden support ratio; agent walks onto the
    'collapses' pad (z<0) or the 'holds' pad (z>0). Truth comes from a
    branch simulation that removes the tail supports. The real world applies
    the same removal when the agent commits, so the outcome is witnessed.
    """

    name = "collapse_judge"
    preset = "void"
    horizon = 1200

    SLAB_X0 = 0
    SLAB_LEN = 6
    SLAB_Z = (0, 2)
    SLAB_Y = 8

    def scenario(self, rng: np.random.Generator):
        # supported column count: 6 (all -> holds) or 3 (half -> collapses)
        self.supported = 6 if rng.random() < 0.5 else 3
        spec = [
            _reg(-10, 5, -10, 10, 5, 10, ids.STONE),  # ground
        ]
        for i in range(self.SLAB_LEN):
            x = self.SLAB_X0 + i
            for z in range(self.SLAB_Z[0], self.SLAB_Z[1] + 1):
                spec.append(_reg(x, self.SLAB_Y, z, x, self.SLAB_Y, z, ids.SAND))
                if i < self.supported:
                    spec.append(_reg(x, self.SLAB_Y - 1, z, x, self.SLAB_Y - 1, z, ids.STONE))
                else:
                    # torches hold the tail until the agent commits
                    spec.append(_reg(x, self.SLAB_Y - 1, z, x, self.SLAB_Y - 1, z, ids.TORCH))
        # answer pads: collapse (z=-3), holds (z=+3), marked with torches
        spec.append(_reg(-5, 6, -3, -5, 6, -3, ids.TORCH))
        spec.append(_reg(-5, 6, 3, -5, 6, 3, ids.TORCH))
        return spec

    def _truth(self, world) -> bool:
        """True iff the slab collapses when tail supports are removed."""

        def remove(mut_world):
            for i in range(self.supported, self.SLAB_LEN):
                x = self.SLAB_X0 + i
                for z in range(self.SLAB_Z[0], self.SLAB_Z[1] + 1):
                    mut_world.set_block(x, self.SLAB_Y - 1, z, 0)

        clone = _branch_sim(world, remove, ticks=40)
        # collapsed iff any falling entity spawned or any slab cell left y=SLAB_Y
        for i in range(self.supported, self.SLAB_LEN):
            x = self.SLAB_X0 + i
            for z in range(self.SLAB_Z[0], self.SLAB_Z[1] + 1):
                if clone.get_block(x, self.SLAB_Y, z) & 0xFFF != ids.SAND:
                    return True
        return False

    def on_reset(self, world, rng: np.random.Generator):
        self.collapses = self._truth(world)
        self._done = False
        world.teleport(-8.5, 6.0, 0.5)  # between the pads, facing the slab

    def step_reward(self, world):
        if self._done:
            return 0.0, False
        x, y, z = world.agent_pos()
        on_collapse = abs(x - (-4.5)) <= 1.5 and abs(z - (-2.5)) <= 1.5
        on_holds = abs(x - (-4.5)) <= 1.5 and abs(z - 3.5) <= 1.5
        if not (on_collapse or on_holds):
            return 0.0, False
        answered_collapse = on_collapse
        # commit: drop the tail supports in the real world
        for i in range(self.supported, self.SLAB_LEN):
            sx = self.SLAB_X0 + i
            for sz in range(self.SLAB_Z[0], self.SLAB_Z[1] + 1):
                world.set_block(sx, self.SLAB_Y - 1, sz, 0)
        self._done = True
        correct = answered_collapse == self.collapses
        return (1.0 if correct else 0.0), True


class WaterRouting(Task):
    """T3: route water from a walled basin to a capped target cell by digging
    a channel. 600-tick budget. Generation-time solvability: branch sim."""

    name = "water_routing"
    preset = "flat"
    horizon = 600

    # basin at origin; trench along z=0 from x=1..6; target (7,4,0) capped
    DIG_CELLS = [(1, 7, 0)] + [(x, 4, 0) for x in range(1, 7)]
    TARGET = (7, 4, 0)

    def scenario(self, rng: np.random.Generator):
        spec = [
            # basin: stone floor under source, dirt walls at y=7
            _reg(0, 6, 0, 0, 6, 0, ids.STONE),
            _reg(-1, 7, 0, -1, 7, 0, ids.DIRT),
            _reg(1, 7, 0, 1, 7, 0, ids.DIRT),
            _reg(0, 7, -1, 0, 7, -1, ids.DIRT),
            _reg(0, 7, 1, 0, 7, 1, ids.DIRT),
            _reg(0, 7, 0, 0, 7, 0, ids.WATER),
            # target cell is OPEN (water must be able to occupy it), capped
            # with stone so only the lateral trench route reaches it
            _reg(7, 4, 0, 7, 4, 0, 0),
            _reg(7, 5, 0, 7, 5, 0, ids.STONE),
            _reg(7, 6, 0, 7, 6, 0, ids.TORCH),  # marker
        ]
        return spec

    def on_reset(self, world, rng: np.random.Generator):
        # solvability validation at generation: dig the channel in a clone,
        # water must reach the target
        def solve(w):
            for c in self.DIG_CELLS:
                w.set_block(*c, 0)

        clone = _branch_sim(world, solve, ticks=400)
        wet = clone.get_block(*self.TARGET) & 0xFFF == ids.WATER
        if not wet:
            raise RuntimeError("water_routing scene unsolvable (generation bug)")
        world.teleport(3.5, 5.0, 2.5)

    def step_reward(self, world):
        if world.get_block(*self.TARGET) & 0xFFF == ids.WATER:
            return 1.0, True
        return 0.0, False


class BridgeOverLava(Task):
    """T3: 20 planks to cross a 6-wide lava trench to the far pad."""

    name = "bridge_over_lava"
    preset = "flat"
    horizon = 1200

    LAVA_X = (10, 15)
    PAD = (18.5, 5.0, 0.5)

    def scenario(self, rng: np.random.Generator):
        spec = []
        for x in range(self.LAVA_X[0], self.LAVA_X[1] + 1):
            for z in range(-3, 4):
                spec.append(_reg(x, 4, z, x, 4, z, ids.LAVA))
        # banks keep the lava contained
        for z in range(-3, 4):
            spec.append(_reg(self.LAVA_X[0] - 1, 4, z, self.LAVA_X[0] - 1, 4, z, ids.STONE))
            spec.append(_reg(self.LAVA_X[1] + 1, 4, z, self.LAVA_X[1] + 1, 4, z, ids.STONE))
        spec.append(_reg(18, 5, 0, 18, 5, 0, ids.TORCH))
        return spec

    def on_reset(self, world, rng: np.random.Generator):
        world.give(ids.PLANKS, 20)
        world.teleport(8.5, 5.0, 0.5)

    def step_reward(self, world):
        x, y, z = world.agent_pos()
        if math.hypot(x - self.PAD[0], z - self.PAD[2]) <= 1.5 and y >= 4.5:
            return 1.0, True
        return 0.0, False


class BuriedEscape(Task):
    """T3: a 2-layer sand slab drops on the agent; dig out and reach the pad
    within 400 ticks (suffocation clock is real: 1 half-heart per 20 ticks).
    """

    name = "buried_escape"
    preset = "flat"
    horizon = 400

    PAD = (5.5, 5.0, 5.5)

    def scenario(self, rng: np.random.Generator):
        spec = []
        for x in range(-1, 2):
            for z in range(-1, 2):
                spec.append(_reg(x, 6, z, x, 6, z, ids.TORCH))  # support layer
                spec.append(_reg(x, 7, z, x, 7, z, ids.SAND))
                spec.append(_reg(x, 8, z, x, 8, z, ids.SAND))
        spec.append(_reg(5, 5, 5, 5, 5, 5, ids.TORCH))  # pad marker
        return spec

    def on_reset(self, world, rng: np.random.Generator):
        world.teleport(0.5, 5.0, 0.5)
        # trigger: remove the support layer -> both sand layers cascade
        for x in range(-1, 2):
            for z in range(-1, 2):
                world.set_block(x, 6, z, 0)

    def step_reward(self, world):
        return self.reach_reward(world, self.PAD)


class CircuitDoor(Task):
    """T4: lever -> wire -> door -> target. `two` variant chains a second
    lever+door reachable only through the first (order enforced spatially).
    """

    preset = "flat"
    horizon = 600

    def __init__(self, two: bool = False):
        self.two = two
        self.name = "circuit_door_two" if two else "circuit_door"
        self.target = (16.5, 5.0, 0.5) if two else (12.5, 5.0, 0.5)

    def scenario(self, rng: np.random.Generator):
        spec = []
        # wall 1 at x=10 with a 2-high doorway: door at y=5, air at y=6
        for z in range(-2, 3):
            for y in (5, 6, 7):
                if z == 0 and y in (5, 6):
                    continue
                spec.append(_reg(10, y, z, 10, y, z, ids.STONE))
        spec.append(_reg(10, 5, 0, 10, 5, 0, ids.DOOR))
        spec.append(_reg(8, 5, 0, 8, 5, 0, ids.LEVER))
        spec.append(_reg(9, 5, 0, 9, 5, 0, ids.WIRE))
        if self.two:
            for z in range(-2, 3):
                for y in (5, 6, 7):
                    if z == 0 and y in (5, 6):
                        continue
                    spec.append(_reg(14, y, z, 14, y, z, ids.STONE))
            spec.append(_reg(14, 5, 0, 14, 5, 0, ids.DOOR))
            spec.append(_reg(12, 5, 0, 12, 5, 0, ids.LEVER))
            spec.append(_reg(13, 5, 0, 13, 5, 0, ids.WIRE))
        spec.append(_reg(int(self.target[0]), 5, 0, int(self.target[0]), 5, 0, ids.TORCH))
        return spec

    def on_reset(self, world, rng: np.random.Generator):
        world.teleport(5.5, 5.0, 0.5)

    def step_reward(self, world):
        return self.reach_reward(world, self.target)


class FirebreakJudge(Task):
    """M3.5: a lava source beside a plank wall; the scene either has a stone
    firebreak or not. Agent answers 'burns down' (pad z<0) or 'survives'
    (pad z>0). Truth from a 400-tick branch simulation."""

    name = "firebreak_judge"
    preset = "void"
    horizon = 1200

    def scenario(self, rng: np.random.Generator):
        self.has_break = bool(rng.random() < 0.5)
        spec = [
            _reg(-10, 5, -10, 10, 5, 10, ids.STONE),   # ground
            _reg(4, 6, 0, 4, 6, 0, ids.LAVA),          # lava source
        ]
        if self.has_break:
            for y in (6, 7, 8):
                spec.append(_reg(5, y, 0, 5, y, 0, ids.STONE))
            wall_x = 6
        else:
            wall_x = 5
        self.wall_x = wall_x
        for y in (6, 7, 8):
            spec.append(_reg(wall_x, y, 0, wall_x, y, 0, ids.PLANKS))
        # pads: "burns" z=-3, "survives" z=+3
        spec.append(_reg(-5, 6, -3, -5, 6, -3, ids.TORCH))
        spec.append(_reg(-5, 6, 3, -5, 6, 3, ids.TORCH))
        return spec

    def _truth(self, world) -> bool:
        """True iff the wall burns."""
        clone = _branch_sim(world, None, ticks=400)
        remaining = sum(
            1 for y in (6, 7, 8)
            if clone.get_block(self.wall_x, y, 0) & 0xFFF == ids.PLANKS
        )
        return remaining < 3

    def on_reset(self, world, rng: np.random.Generator):
        self.burns = self._truth(world)
        self._done = False
        world.teleport(-8.5, 6.0, 0.5)

    def step_reward(self, world):
        if self._done:
            return 0.0, False
        x, y, z = world.agent_pos()
        on_burns = abs(x - (-4.5)) <= 1.5 and abs(z - (-2.5)) <= 1.5
        on_survives = abs(x - (-4.5)) <= 1.5 and abs(z - 3.5) <= 1.5
        if not (on_burns or on_survives):
            return 0.0, False
        self._done = True
        correct = on_burns == self.burns
        return (1.0 if correct else 0.0), True


class PlateDoor(Task):
    """M3.5: pressure plate on the path opens the door. Walk through."""

    name = "plate_door"
    preset = "flat"
    horizon = 600
    TARGET = (12.5, 5.0, 0.5)

    def scenario(self, rng: np.random.Generator):
        spec = []
        for z in range(-2, 3):
            for y in (5, 6, 7):
                if z == 0 and y in (5, 6):
                    continue
                spec.append(_reg(10, y, z, 10, y, z, ids.STONE))
        spec.append(_reg(10, 5, 0, 10, 5, 0, ids.DOOR))
        # plates on BOTH sides (MC pattern: the far plate holds the door open
        # while the agent clears the doorway)
        spec.append(_reg(9, 5, 0, 9, 5, 0, ids.PRESSURE_PLATE))
        spec.append(_reg(11, 5, 0, 11, 5, 0, ids.PRESSURE_PLATE))
        # wire mesh: (9,5,1) beside plate A; (10,5,1) beside the door;
        # (11,5,1) beside plate B — all chained
        spec.append(_reg(9, 5, 1, 9, 5, 1, ids.WIRE))
        spec.append(_reg(10, 5, 1, 10, 5, 1, ids.WIRE))
        spec.append(_reg(11, 5, 1, 11, 5, 1, ids.WIRE))
        spec.append(_reg(12, 5, 0, 12, 5, 0, ids.TORCH))
        return spec

    def on_reset(self, world, rng: np.random.Generator):
        world.teleport(5.5, 5.0, 0.5)

    def step_reward(self, world):
        return self.reach_reward(world, self.TARGET)


class TntClear(Task):
    """M3.5: a 2-thick dirt wall blocks the corridor; flip the lever to
    detonate the embedded TNT and clear a path to the target."""

    name = "tnt_clear"
    preset = "flat"
    horizon = 600
    TARGET = (14.5, 5.0, 0.5)

    def scenario(self, rng: np.random.Generator):
        spec = []
        # wall: x=10..11, z=-2..2, y=5..7 dirt
        for x in (10, 11):
            for z in range(-2, 3):
                for y in (5, 6, 7):
                    spec.append(_reg(x, y, z, x, y, z, ids.DIRT))
        # lever/wire/tnt at y=6: blast clears the wall but only scalps the
        # grass top (a 2-deep crater would trap the agent)
        spec.append(_reg(7, 6, 0, 7, 6, 0, ids.LEVER))
        spec.append(_reg(8, 6, 0, 8, 6, 0, ids.WIRE))
        spec.append(_reg(9, 6, 0, 9, 6, 0, ids.TNT))
        spec.append(_reg(14, 5, 0, 14, 5, 0, ids.TORCH))
        return spec

    def on_reset(self, world, rng: np.random.Generator):
        world.teleport(4.5, 5.0, 0.5)

    def step_reward(self, world):
        return self.reach_reward(world, self.TARGET)


class LogicProbe(Task):
    """T5: gate-level combinational circuits. Two levers (A, B) feed a
    template circuit; the agent must drive the lamp to a per-episode
    target state (0 or 1).

    The world IS a synchronous digital simulator: redstone torches are NOT
    gates with 1-tick delay, wire joins are wired-OR — so each template
    computes a Boolean function of (A, B):
      or:   A + B        (0 gates — plain wired-OR into the lamp)
      nor:  ~(A + B)     (1 torch over the joined inputs)
      nand: ~A + ~B      (2 torches, outputs joined at the lamp)
      and:  ~(~A + ~B)   (3 torches — De Morgan)
    Geometry obeys the routing rules verified in the Rust truth tables:
    every horizontal neighbor of a torch joins its output net, so input
    feeds stay one level below and output routes keep 2-cell clearance.
    """

    name = "logic_probe"
    preset = "flat"
    horizon = 400

    TEMPLATES = ("or", "nor", "nand", "and")
    LEVER_A = (0, 5, 0)
    LEVER_B = (0, 5, 2)

    def scenario(self, rng: np.random.Generator):
        self.template = self.TEMPLATES[rng.integers(len(self.TEMPLATES))]
        self.init_a = int(rng.integers(2))
        self.init_b = int(rng.integers(2))
        spec = [
            _reg(0, 5, 0, 0, 5, 0, ids.LEVER | (self.init_a << 12)),
            _reg(0, 5, 2, 0, 5, 2, ids.LEVER | (self.init_b << 12)),
        ]
        t = self.template
        if t == "or":
            self.lamp = (3, 5, 1)
            spec += [
                _reg(1, 5, 0, 1, 5, 0, ids.WIRE),
                _reg(1, 5, 2, 1, 5, 2, ids.WIRE),
                _reg(2, 5, 0, 2, 5, 2, ids.WIRE),
                _reg(3, 5, 1, 3, 5, 1, ids.LAMP),
            ]
        elif t == "nor":
            self.lamp = (4, 6, 1)
            spec += [
                _reg(1, 5, 0, 1, 5, 0, ids.WIRE),
                _reg(1, 5, 2, 1, 5, 2, ids.WIRE),
                _reg(2, 5, 0, 2, 5, 2, ids.WIRE),
                _reg(2, 6, 1, 2, 6, 1, ids.REDSTONE_TORCH),
                _reg(3, 6, 1, 3, 6, 1, ids.WIRE),
                _reg(4, 6, 1, 4, 6, 1, ids.LAMP),
            ]
        elif t == "nand":
            self.lamp = (3, 6, 1)
            spec += [
                _reg(1, 5, 0, 1, 5, 0, ids.WIRE),
                _reg(1, 5, 2, 1, 5, 2, ids.WIRE),
                _reg(1, 6, 0, 1, 6, 0, ids.REDSTONE_TORCH),  # ~A
                _reg(1, 6, 2, 1, 6, 2, ids.REDSTONE_TORCH),  # ~B
                _reg(2, 6, 0, 2, 6, 2, ids.WIRE),           # ~A + ~B
                _reg(3, 6, 1, 3, 6, 1, ids.LAMP),
            ]
        else:  # and
            self.lamp = (6, 6, 1)
            spec += [
                _reg(1, 5, 0, 1, 5, 0, ids.WIRE),
                _reg(1, 5, 2, 1, 5, 2, ids.WIRE),
                _reg(1, 6, 0, 1, 6, 0, ids.REDSTONE_TORCH),  # ~A
                _reg(1, 6, 2, 1, 6, 2, ids.REDSTONE_TORCH),  # ~B
                _reg(2, 6, 0, 2, 6, 2, ids.WIRE),           # merge
                _reg(2, 5, 1, 2, 5, 1, ids.WIRE),           # drop below merge
                _reg(3, 5, 1, 3, 5, 1, ids.WIRE),           # feed D
                _reg(4, 5, 1, 4, 5, 1, ids.WIRE),           # D under T3
                _reg(4, 6, 1, 4, 6, 1, ids.REDSTONE_TORCH),  # T3: ~(~A+~B)
                _reg(5, 6, 1, 5, 6, 1, ids.WIRE),
                _reg(6, 6, 1, 6, 6, 1, ids.LAMP),
            ]
        return spec

    def _lamp(self, world) -> int:
        return (world.get_block(*self.lamp) >> 12) & 1

    def on_reset(self, world, rng: np.random.Generator):
        self.goal = int(rng.integers(2))
        # settle a clone and force a non-trivial episode: if the initial
        # lever states already satisfy the sampled goal, flip the goal
        if self._lamp(_branch_sim(world, None, ticks=8)) == self.goal:
            self.goal = 1 - self.goal
        world.teleport(-4.5, 5.0, 1.5)

    def step_reward(self, world):
        # ignore the settle transient (torches self-correct on tick 1-2)
        if world.tick() < 8:
            return 0.0, False
        if self._lamp(world) == self.goal:
            return 1.0, True
        return 0.0, False


PROBE_TASKS = ["collapse_judge", "water_routing", "bridge_over_lava", "buried_escape",
               "circuit_door", "circuit_door_two",
               "firebreak_judge", "plate_door", "tnt_clear", "logic_probe"]


def make_probe(name: str) -> Task:
    if name == "collapse_judge":
        return CollapseJudge()
    if name == "water_routing":
        return WaterRouting()
    if name == "bridge_over_lava":
        return BridgeOverLava()
    if name == "buried_escape":
        return BuriedEscape()
    if name == "circuit_door":
        return CircuitDoor(False)
    if name == "circuit_door_two":
        return CircuitDoor(True)
    if name == "firebreak_judge":
        return FirebreakJudge()
    if name == "plate_door":
        return PlateDoor()
    if name == "tnt_clear":
        return TntClear()
    if name == "logic_probe":
        return LogicProbe()
    raise KeyError(name)
