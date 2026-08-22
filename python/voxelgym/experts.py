"""Scripted oracle experts (M2): task feasibility proof + dataset generation.

Oracle license: experts read full world state (find_blocks, surface_y,
furnace_state) and manage inventory via swap_to_hotbar — the action space has
no inventory-move verb by design, so the swap simulates UI inventory
management. Every emitted action is a valid env action; recordings replay
bit-exact without any oracle calls.

CLI: python -m voxelgym.experts --task craft_stone_pickaxe --episodes 20
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

from . import ids
from .env import ACTION_KEYS, VoxelGymEnv
from .tasks import make_task, task_names


def yaw_bucket_toward(dx: float, dz: float) -> int:
    # forward = (-sin yaw, cos yaw); solve for yaw toward (dx, dz)
    yaw = math.degrees(math.atan2(-dx, dz))
    return int(round(yaw / 15.0)) % 24


def fwd_vec(yaw_bucket: int) -> tuple[float, float]:
    r = math.radians(yaw_bucket * 15.0)
    return -math.sin(r), math.cos(r)


def aim_at(eye: tuple[float, float, float], tx: float, ty: float, tz: float) -> tuple[int, int]:
    dx, dy, dz = tx - eye[0], ty - eye[1], tz - eye[2]
    yb = yaw_bucket_toward(dx, dz)
    horiz = math.hypot(dx, dz)
    if horiz < 1e-9:
        pitch = 90.0 if dy < 0 else -90.0
    else:
        pitch = math.degrees(math.atan2(-dy, horiz))  # positive = down
    pb = int(round((pitch + 60.0) / 15.0))
    return yb, max(0, min(8, pb))


def act(**kw) -> dict:
    a = {k: 0 for k in ACTION_KEYS}
    a.update(kw)
    return a


def cell_id_of(world, x: int, y: int, z: int) -> int:
    return world.get_block(x, y, z) & 0xFFF


def in_water(world) -> bool:
    x, y, z = world.agent_pos()
    return cell_id_of(world, math.floor(x), math.floor(y), math.floor(z)) == ids.WATER


class Navigator:
    """Greedy approach with stall->jump, detour, and cliff guard.

    Stall = windowed: less than 0.3 cells of displacement over the last 10
    ticks (per-tick deltas are defeated by collision oscillation).
    """

    def __init__(self):
        self._hist: list[tuple[float, float]] = []
        self._stall = 0
        self._detour = 0
        self._dir = 1

    def stalled(self, x: float, z: float) -> bool:
        self._hist.append((x, z))
        if len(self._hist) > 10:
            self._hist.pop(0)
        if len(self._hist) < 10:
            return False
        xs = [p[0] for p in self._hist]
        zs = [p[1] for p in self._hist]
        return max(xs) - min(xs) < 0.3 and max(zs) - min(zs) < 0.3

    def toward(self, world, tx: float, tz: float, allow_descent: bool = False) -> dict:
        x, y, z = world.agent_pos()
        if self.stalled(x, z):
            self._stall += 1
        else:
            self._stall = 0

        yaw = yaw_bucket_toward(tx - x, tz - z)
        jump = 0
        if self._stall > 3:
            jump = 1
        if self._stall > 12 and self._detour == 0:
            self._detour, self._dir = 25, -self._dir
            self._stall = 0
        if self._detour > 0:
            yaw = (yaw + self._dir * 6) % 24
            self._detour -= 1
            jump = 1
        elif not allow_descent:
            # cliff guard: refuse drops > 3.5 cells unless landing in water
            fx, fz = fwd_vec(yaw)
            ax, az = int(math.floor(x + fx * 1.6)), int(math.floor(z + fz * 1.6))
            sy = world.surface_y(ax, az)
            if 0 <= sy < y - 3.5:
                if cell_id_of(world, ax, sy + 1, az) != ids.WATER:
                    self._detour, self._dir = 15, -self._dir
                    yaw = (yaw + self._dir * 6) % 24
        if in_water(world):
            jump = 1
        return act(move=1, jump=jump, yaw=yaw, pitch=4)


class T1Expert:
    """Greedy nav + progress watchdog: if the best distance hasn't improved
    for 150 ticks, tunnel through the blocking cells toward the target
    (barehand mining is slow but always available)."""

    def __init__(self, task):
        self.task = task
        self.nav = Navigator()
        self._best = float("inf")
        self._best_tick = -10**9
        self._focus: tuple[int, int, int] | None = None

    def act(self, world) -> dict:
        t = self.task.target
        x, y, z = world.agent_pos()
        d = math.hypot(x - t[0], z - t[2])
        tick = world.tick()

        if self._focus is not None:
            if cell_id_of(world, *self._focus) == ids.AIR:
                self._focus = None
            else:
                yb, pb = aim_at((x, y + 1.62, z), *(c + 0.5 for c in self._focus))
                return act(mine=1, yaw=yb, pitch=pb)

        if d < self._best - 0.3:
            self._best = d
            self._best_tick = tick
        if tick - self._best_tick > 150:
            h = math.hypot(t[0] - x, t[2] - z)
            fx, fz = (t[0] - x) / h, (t[2] - z) / h
            lx, lz = -fz, fx
            for dy in (0, 1):
                for reach in (1.0, 2.0):
                    for lat in (-1, 0, 1):
                        c = (
                            int(math.floor(x + reach * fx + lat * lx)),
                            int(math.floor(y)) + dy,
                            int(math.floor(z + reach * fz + lat * lz)),
                        )
                        if cell_id_of(world, *c) not in (ids.AIR, ids.WATER, ids.LAVA, ids.TORCH):
                            self._focus = c
                            self._best_tick = tick
                            yb, pb = aim_at((x, y + 1.62, z), *(cc + 0.5 for cc in c))
                            return act(mine=1, yaw=yb, pitch=pb)
            self._best_tick = tick  # open air: let nav keep trying
        return self.nav.toward(world, t[0], t[2])


class Stage:
    """(guard, op): guard(world)->True means the stage is complete."""

    def __init__(self, guard, op):
        self.guard = guard
        self.op = op


class CraftOp:
    def __init__(self, recipe: int):
        self.recipe = recipe

    def act(self, world) -> dict:
        return act(craft=self.recipe)


class WaitOp:
    def __init__(self, ticks: int):
        self.left = ticks

    def act(self, world) -> dict:
        self.left -= 1
        return act()


class NavOp:
    def __init__(self, pos_fn, tol: float = 3.0):
        self.pos_fn = pos_fn
        self.tol = tol
        self.nav = Navigator()

    def done(self, world) -> bool:
        p = self.pos_fn()
        if p is None:
            return True
        x, _, z = world.agent_pos()
        return math.hypot(x - p[0], z - p[2]) <= self.tol

    def act(self, world) -> dict:
        p = self.pos_fn()
        return self.nav.toward(world, p[0], p[2])


class PlaceOp:
    """Place a held block on the ground ahead; records where it landed."""

    def __init__(self, item: int, sink):
        self.item = item
        self.sink = sink  # callable(x, y, z)
        self.placed = None
        self._attempt = 0
        self._reposition = 0
        self._digging = None  # (cell, yaw) when widening a pocket

    def act(self, world) -> dict:
        if self.placed is not None:
            return act()
        found = world.find_blocks(self.item, 5)
        if found:
            self.placed = found[0]
            self.sink(self.placed)
            return act()
        slot = world.swap_to_hotbar(self.item)
        if slot < 0:
            return act()  # item missing; guard shouldn't have allowed this
        x, y, z = world.agent_pos()
        fy = int(math.floor(y))
        # wedged in a 1x1 pocket (every placement target is the agent's own
        # cell): widen it — mine a side cell, then placement works
        if self._digging is not None:
            c, _yaw = self._digging
            if cell_id_of(world, *c) in (ids.AIR, ids.WATER):
                self._digging = None
            else:
                yb, pb = aim_at((x, y + 1.62, z), c[0] + 0.5, c[1] + 0.5, c[2] + 0.5)
                return act(mine=1, yaw=yb, pitch=pb, hotbar=slot)
        if self._attempt > 0 and self._attempt % 150 == 149:
            for yaw in (18, 0, 12, 6):
                fx, fz = fwd_vec(yaw)
                c = (int(math.floor(x + fx)), fy, int(math.floor(z + fz)))
                if cell_id_of(world, *c) not in (ids.AIR, ids.WATER, ids.LAVA):
                    self._digging = (c, yaw)
                    yb, pb = aim_at((x, y + 1.62, z), c[0] + 0.5, c[1] + 0.5, c[2] + 0.5)
                    return act(mine=1, yaw=yb, pitch=pb, hotbar=slot)
        # preferred pose: 2-ahead at feet level is AIR with SOLID below —
        # AND the 45-deg ray path through the 1-ahead column must be clear
        # (a solid (1-ahead, y+1) intercepts the ray first)
        for yaw in (18, 0, 12, 6):
            fx, fz = fwd_vec(yaw)
            c2 = (int(math.floor(x + 2 * fx)), fy, int(math.floor(z + 2 * fz)))
            below = (c2[0], fy - 1, c2[2])
            path1 = (int(math.floor(x + fx)), fy + 1, int(math.floor(z + fz)))
            path0 = (int(math.floor(x + fx)), fy, int(math.floor(z + fz)))
            if (
                cell_id_of(world, *c2) == ids.AIR
                and cell_id_of(world, *below) not in (ids.AIR, ids.WATER, ids.LAVA)
                and cell_id_of(world, *path1) == ids.AIR
                and cell_id_of(world, *path0) == ids.AIR
            ):
                self._attempt += 1
                return act(place=1, hotbar=slot, yaw=yaw, pitch=7)
        # every combo cycles in 4-tick slots; after a full cycle with no
        # placement, step back a few ticks — a 5-10cm AABB edge overlap can
        # veto every target from the current footing
        if self._reposition > 0:
            self._reposition -= 1
            return act(move=2, yaw=(self._attempt // 48 % 2) * 12, pitch=4, hotbar=slot)
        self._attempt += 1
        if self._attempt % 48 == 0:
            self._reposition = 6
        # cycle yaw x pitch combos across attempts to handle awkward
        # geometry (shafts, ledges); pitch 6-8 keeps the target cell clear
        # of the agent AABB at any alignment
        combos = [(yaw, pitch) for pitch in (7, 8, 6) for yaw in (18, 12, 0, 6)]
        yaw, pitch = combos[(self._attempt // 4) % len(combos)]
        return act(place=1, hotbar=slot, yaw=yaw, pitch=pitch)

    def done(self, world) -> bool:
        return self.placed is not None


class HarvestOp:
    """Navigate to and mine the nearest `block` until the guard completes.
    Handles underground targets by burrowing a staircase; `deep` targets
    (diamond) first descend to mining depth, then strip-mine if nothing is
    found nearby.
    """

    def __init__(self, block: int, tool: int | None, drop: int | None = None,
                 radius: int = 32, deep: bool = False, mine_level: int = 13, seed: int = 0,
                 at_fn=None, prefer_high: bool = False):
        self.block = block
        self.tool = tool
        self.drop = block if drop is None else drop
        self.radius = radius
        self.deep = deep
        self.mine_level = mine_level  # deep mode: descend to this y before strip-mining
        self.at_fn = at_fn  # optional zero-arg callable giving the known cell
        self.prefer_high = prefer_high
        self.focus: tuple[int, int, int] | None = None
        self.walk = 0
        self.explore_left = 0
        self.explore_yaw = 0
        self.nav = Navigator()
        self.rng = np.random.default_rng(seed)
        self.target: tuple[int, int, int] | None = None
        self.target_tick = -10**9
        self._lp: tuple[float, float] | None = None
        self._stall = 0
        self._last_yaw = 0
        self.desc: dict | None = None  # staircase descent FSM
        self.no_drops_until = -1  # unreachable-drop blacklist expiry tick
        # watchdog: last tick with tangible progress (cell broken or wanted
        # item gained). 700 ticks without progress -> recovery walk-away.
        self._last_gain = -1
        self._last_count = 0
        self._last_pos: tuple[float, float] | None = None
        self._recover_until = -1
        self.watchdog_fires = 0
        self._hold: tuple[int, int] | None = None  # held (yaw, pitch)
        self._hold_cell = None  # cell locked under the crosshair
        self._cal: list[tuple[int, int]] = []
        self._ci = 0
        self._cal_focus = None

    def _nearest(self, world):
        if self.at_fn is not None:
            p = self.at_fn()
            if p is not None and cell_id_of(world, *p) == self.block:
                return p
            return None
        ts = world.find_blocks(self.block, self.radius)
        if not ts:
            return None
        x, y, z = world.agent_pos()
        # underground (caves/tunnels): prefer cells at or below the current
        # level — ceiling ores across caverns are unreachable by tunneling
        if self._underground(world, x, y, z):
            belowish = [t for t in ts if t[1] <= y + 1]
            if belowish:
                ts = belowish
        # descent is ~2.5x the cost of level travel per cell: weigh it in
        return min(ts, key=lambda t: max(0.0, y - t[1]) * 2.5 + abs(t[0] - x) + abs(t[2] - z))

    def _reachable(self, world, t) -> bool:
        """True if the cell center is within reach AND outside the ±60 deg
        pitch dead cone (blocks nearly straight below/above can't be aimed)."""
        x, y, z = world.agent_pos()
        dy = (t[1] + 0.5) - (y + 1.62)
        horiz = math.hypot(t[0] + 0.5 - x, t[2] + 0.5 - z)
        d3 = math.hypot(horiz, dy)
        pitch = math.degrees(math.atan2(-dy, horiz)) if horiz > 1e-9 else 90.0
        return d3 <= 4.2 and -60.0 <= pitch <= 60.0

    def _mine_focus(self, world, slot: int) -> dict:
        # Hold buckets that keep ANY solid cell in the focus direction under
        # the crosshair — the focus itself or an occluder (mine through).
        # When that cell breaks, the crosshair target changes and we re-aim.
        ch = world.crosshair()
        if self._hold is not None and ch is not None:
            hit, hid = tuple(ch[0]), ch[1]
            if self._hold_cell is None and hid not in (ids.WATER, ids.LAVA):
                self._hold_cell = hit  # lock onto whatever solid we hit
            if hit == self._hold_cell:
                return act(mine=1, yaw=self._hold[0], pitch=self._hold[1], hotbar=slot)
        # (re)calibrate around the exact aim
        if self._cal_focus != self.focus:
            x, y, z = world.agent_pos()
            eye = (x, y + 1.62, z)
            fx, fy, fz = self.focus
            yb0, pb0 = aim_at(eye, fx + 0.5, fy + 0.5, fz + 0.5)
            cands = []
            for pb in (pb0, pb0 + 1, pb0 - 1, pb0 + 2, pb0 - 2):
                cands.append((yb0, max(0, min(8, pb))))
            for yb in ((yb0 + 1) % 24, (yb0 - 1) % 24):
                cands.append((yb, max(0, min(8, pb0))))
            self._cal = cands
            self._ci = 0
            self._cal_focus = self.focus
        self._hold = None
        self._hold_cell = None
        yb, pb = self._cal[self._ci % len(self._cal)]
        self._ci += 1
        self._hold = (yb, pb)
        return act(mine=1, yaw=yb, pitch=pb, hotbar=slot)

    @staticmethod
    def _underground(world, x: float, y: float, z: float) -> bool:
        return (
            cell_id_of(world, math.floor(x), math.floor(y) + 2, math.floor(z)) != ids.AIR
            or cell_id_of(world, math.floor(x), math.floor(y) + 3, math.floor(z)) != ids.AIR
        )

    def _sweep_cell(self, world, yaw: int, reach: float):
        """Nearest solid cell in a forward sweep sector (5x5 window starting
        `reach` ahead) — catches veins that drifted off the tunnel axis."""
        x, y, z = world.agent_pos()
        fx, fz = fwd_vec(yaw)
        lx, lz = -fz, fx
        fy = math.floor(y)
        for d in (reach, reach + 1.0):
            for lat in (-2, -1, 0, 1, 2):
                for dy in (2, 1, 0, -1):
                    c = (
                        math.floor(x + d * fx + lat * lx),
                        fy + dy,
                        math.floor(z + d * fz + lat * lz),
                    )
                    cid = cell_id_of(world, *c)
                    if cid != ids.AIR and cid != ids.WATER and self._reachable(world, c):
                        return c
        return None

    def _burrow_cell(self, world, yaw: int) -> tuple[int, int, int]:
        # two cells ahead, one down: far enough to escape the +-60 deg pitch
        # dead cone (pitch ~53 deg); the ray may hit the nearer cell first,
        # which is fine — mining clears the staircase either way
        x, y, z = world.agent_pos()
        fx, fz = fwd_vec(yaw)
        return (int(math.floor(x + 2 * fx)), int(math.floor(y)) - 1, int(math.floor(z + 2 * fz)))

    def act(self, world) -> dict:
        slot = 0
        if self.tool is not None and world.count_item(self.tool) > 0:
            slot = max(world.swap_to_hotbar(self.tool), 0)
        x, y, z = world.agent_pos()
        # (mining rays ignore fluids, so submersion needs no special path)

        # ---- watchdog + recovery ----
        if self._last_gain < 0:
            self._last_gain = world.tick()
            self._last_count = world.count_item(self.drop)
            self._last_pos = (x, z)
        # positional progress counts too — long legitimate approaches break
        # no cells and must not trip the watchdog
        if (world.tick() - self._last_gain) % 50 == 0 and self._last_pos is not None:
            if math.hypot(x - self._last_pos[0], z - self._last_pos[1]) > 2.0:
                self._last_gain = world.tick()
            self._last_pos = (x, z)
        cnt = world.count_item(self.drop)
        if cnt != self._last_count:
            self._last_count = cnt
            self._last_gain = world.tick()
        if self._recover_until > world.tick():
            dx, dz = fwd_vec(self.explore_yaw)
            return self.nav.toward(world, x + 15 * dx, z + 15 * dz)
        if world.tick() - self._last_gain > 200:
            self.watchdog_fires += 1
            # wedged: drop all local state and sidestep briefly; re-approach
            # from different geometry afterwards
            self.focus = None
            self.target = None
            self.desc = None
            self._stall = 0
            self.explore_yaw = int(self.rng.integers(0, 24))
            self._recover_until = world.tick() + 30
            self._last_gain = world.tick()
            dx, dz = fwd_vec(self.explore_yaw)
            return self.nav.toward(world, x + 15 * dx, z + 15 * dz)

        # continue mining the focused cell until it breaks
        if self.focus is not None:
            fcid = cell_id_of(world, *self.focus)
            if fcid == ids.AIR or fcid == ids.WATER or fcid == ids.LAVA:
                # broken — or refilled by a fluid (mining a fluid cell never
                # progresses; treating the refill as a break avoids the
                # permanent water-focus wedge)
                self._last_gain = world.tick()
                self.focus = None
                self._hold = None
                self._hold_cell = None
                self.walk = 2
            elif not self._reachable(world, self.focus):
                self.focus = None  # fell out of the aim cone; re-target
                self._hold = None
                self._hold_cell = None
            else:
                return self._mine_focus(world, slot)

        # stall rescue: (a) short stall while intending to move -> jump-walk
        # to mount 1-high ledges (no auto-step in the physics contract);
        # (b) long stall -> mine through the blocking cell, probing lateral
        # cells too (corner wedges block diagonally). Windowed detection:
        # collision oscillation defeats per-tick deltas.
        if self.nav.stalled(x, z):
            self._stall += 1
        else:
            self._stall = 0
        if 3 < self._stall <= 8:
            return act(move=1, jump=1, yaw=self._last_yaw, pitch=4, hotbar=slot)
        if self._stall > 8:
            fx, fz = fwd_vec(self._last_yaw)
            lx, lz = -fz, fx  # lateral unit (approx for 8-way yaws)
            for dy in (1, 0):
                for reach in (1.0, 2.0):
                    for lat in (-1, 0, 1):
                        c = (
                            int(math.floor(x + reach * fx + lat * lx)),
                            int(math.floor(y)) + dy,
                            int(math.floor(z + reach * fz + lat * lz)),
                        )
                        if cell_id_of(world, *c) not in (ids.AIR, ids.WATER, ids.LAVA) and self._reachable(world, c):
                            self.focus = c
                            self._stall = 0
                            return self._mine_focus(world, slot)
            self._stall = 0
            # couldn't mine our way out: ignore unreachable drops for a while
            self.no_drops_until = world.tick() + 100

        if self.walk > 0:
            self.walk -= 1
            self._last_yaw = self.explore_yaw
            return act(move=1, yaw=self.explore_yaw, pitch=4, hotbar=slot)

        # drops of the wanted item: poll every tick (cheap). On a target,
        # only grab drops within 4 cells (on the way); targetless, pursue
        # within 16.
        if world.tick() >= self.no_drops_until:
            drops = world.drops_of(self.drop)
            cap = 256.0 if self.target is None else 16.0
            drops = [d for d in drops if (d[0] - x) ** 2 + (d[1] - y) ** 2 + (d[2] - z) ** 2 <= cap]
            if drops:
                dx, dy, dz = min(drops, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2 + (p[2] - z) ** 2)
                if math.hypot(dx - x, dz - z) > 1.0:
                    self._last_yaw = yaw_bucket_toward(dx - x, dz - z)
                    a = self.nav.toward(world, dx, dz, allow_descent=True)
                    a["hotbar"] = slot
                    return a
                return act(hotbar=slot)  # close enough: pickup radius
        # block scan is expensive — cadence-gated, and the target is STICKY
        if self.target is None and world.tick() - self.target_tick >= 15:
            self.target_tick = world.tick()
            self.target = self._nearest(world)
        if self.target is not None and cell_id_of(world, *self.target) != self.block:
            self.target = None
            return act(hotbar=slot)
        t = self.target

        # direct mine when the target block itself is aimable
        if t is not None and self._reachable(world, t):
            self._br = "direct" 
            self.desc = None
            self.focus = t
            return self._mine_focus(world, slot)

        # --- vertical positioning: latched staircase descent ---
        # Once a descent starts it NEVER re-aims: re-aiming near-vertical
        # targets makes the shaft flap back into its own tailings. The
        # descent runs until the target's level is reached, then horizontal
        # tunneling takes over.
        t_xz = math.hypot(t[0] + 0.5 - x, t[2] + 0.5 - z) if t is not None else None
        if self.desc is not None and y <= self.desc["target_y"]:
            self.desc = None
        # any below-level target gets a staircase toward its level (yaw
        # frozen at creation); the old XZ gate forced a wasteful
        # descend-to-fixed-depth-then-tunnel path
        if self.desc is None and t is not None and t[1] < y - 1.2:
            yaw = yaw_bucket_toward(t[0] - x, t[2] - z) if t_xz > 0.8 else self._last_yaw
            self.desc = {"yaw": yaw, "walk": 0, "queue": [], "target_y": t[1] + 1}
        if self.desc is None and self.deep and y > self.mine_level and (t is None or t[1] < y - 1.2):
            self.desc = {"yaw": self._last_yaw, "walk": 0, "queue": [], "target_y": self.mine_level}
        if self.desc is not None:
            self._br = "desc"
            d = self.desc
            abort = False
            if d["walk"] > 0:
                # walk until actually stepping down into the notch (a fixed
                # count could cross the 1-deep notch and end level — that
                # made trenches, not staircases)
                d["walk"] -= 1
                self._last_yaw = d["yaw"]
                if y <= d.get("walk_from_y", y) - 0.9:
                    d["walk"] = 0  # descended: build the next column now
                else:
                    return act(move=1, yaw=d["yaw"], pitch=4, hotbar=slot)
            if not d["queue"]:
                # direction management per level: spiral when nearly above
                # the target (no overshoot), re-aim when drifted far —
                # the 3..6 hysteresis band keeps the frozen-yaw benefit
                if t is not None:
                    t_xz_now = math.hypot(t[0] + 0.5 - x, t[2] + 0.5 - z)
                    if t_xz_now < 3.0:
                        d["yaw"] = (d["yaw"] + 6) % 24
                    elif t_xz_now > 6.0:
                        d["yaw"] = yaw_bucket_toward(t[0] - x, t[2] - z)
                base = self._burrow_cell(world, d["yaw"])
                # cave guard: a notch opening into a 3+ drop is a cave mouth —
                # stop descending and work at the current level instead of
                # falling into the cavern
                drop = 0
                for k in range(1, 4):
                    if cell_id_of(world, base[0], base[1] - k, base[2]) in (ids.AIR, ids.WATER):
                        drop += 1
                    else:
                        break
                if drop >= 3:
                    abort = True
                    self.desc = None
                else:
                    col = [(base[0], base[1] + 1, base[2]), base]
                    d["queue"] = [
                        c for c in col
                        if cell_id_of(world, *c) not in (ids.AIR, ids.WATER) and self._reachable(world, c)
                    ]
                    if not d["queue"]:
                        d["walk"] = 10  # bounded; ends early on stepping down
                        d["walk_from_y"] = y
                        self._last_yaw = d["yaw"]
                        return act(move=1, yaw=d["yaw"], pitch=4, hotbar=slot)
            if abort:
                pass  # cave guard fired: fall through to approach/tunnel
            elif d["queue"]:
                self.focus = d["queue"].pop(0)
                self.explore_yaw = d["yaw"]
                if not d["queue"]:
                    d["walk"] = 10  # column cleared: step into the notch
                    d["walk_from_y"] = y
                return self._mine_focus(world, slot)
            else:
                self._last_yaw = d["yaw"]
                return act(move=1, yaw=d["yaw"], pitch=4, hotbar=slot)

        if t is None:
            self._br = "explore"
            # explore: surface wander; strip-mine when underground (nav is
            # for open terrain — tunnels need the front cells cleared)
            underground = self._underground(world, x, y, z)
            if (self.deep and y <= self.mine_level) or underground:
                yaw = self.explore_yaw
                fx, fz = fwd_vec(yaw)
                # flood guard: don't open cells with water directly above
                for dy in (1, 0):
                    c = (int(math.floor(x + fx)), int(math.floor(y)) + dy, int(math.floor(z + fz)))
                    cid = cell_id_of(world, *c)
                    if cid != ids.AIR and cid != ids.WATER:
                        self.focus = c
                        self._open_streak = 0
                        return self._mine_focus(world, slot)
                self._open_streak = getattr(self, "_open_streak", 0) + 1
                if self._open_streak > 10:
                    c = self._sweep_cell(world, yaw, 1.5)
                    if c is not None:
                        self.focus = c
                        self.explore_yaw = yaw
                        self._open_streak = 0
                        return self._mine_focus(world, slot)
                self._last_yaw = yaw
                return act(move=1, yaw=yaw, pitch=4, hotbar=slot)
            if self.explore_left <= 0:
                self.explore_left = 80
                self.explore_yaw = int(self.rng.integers(0, 24))
            self.explore_left -= 1
            self._last_yaw = self.explore_yaw
            dx, dz = fwd_vec(self.explore_yaw)
            a = self.nav.toward(world, x + 30 * dx, z + 30 * dz)
            a["hotbar"] = slot
            return a

        # too close horizontally to aim (dead cones below +-60 deg both
        # directions): back off facing the target until it enters the cone
        dx, dz = t[0] + 0.5 - x, t[2] + 0.5 - z
        too_close = math.hypot(dx, dz) < 1.5 and (
            t[1] < y - 0.2 or t[1] + 0.5 > y + 1.62 + 0.2
        )
        if too_close:
            yaw = yaw_bucket_toward(dx, dz)
            self._last_yaw = yaw
            return act(move=2, yaw=yaw, pitch=4, hotbar=slot)

        # approach: nav in open terrain; strip-tunnel toward the target when
        # underground (nav can't path through solid rock)
        self._br = "approach"
        if in_water(world) and t[1] < y - 0.5:
            # target under water: SINK toward it — the default swim-up
            # response would bob on the surface forever
            self._last_yaw = yaw_bucket_toward(t[0] + 0.5 - x, t[2] + 0.5 - z)
            return act(move=1, jump=0, yaw=self._last_yaw, pitch=4, hotbar=slot)
        if self._underground(world, x, y, z):
            yaw = yaw_bucket_toward(t[0] + 0.5 - x, t[2] + 0.5 - z)
            fx, fz = fwd_vec(yaw)
            for dy in (1, 0):
                c = (int(math.floor(x + fx)), int(math.floor(y)) + dy, int(math.floor(z + fz)))
                cid = cell_id_of(world, *c)
                if cid != ids.AIR and cid != ids.WATER and self._reachable(world, c):
                    self.focus = c
                    self.explore_yaw = yaw
                    self._open_streak = 0
                    return self._mine_focus(world, slot)
            # sweep side cells only after a genuinely open stretch — digging
            # every tick doubles the tunneling cost
            self._open_streak = getattr(self, "_open_streak", 0) + 1
            if self._open_streak > 10:
                c = self._sweep_cell(world, yaw, 1.5)
                if c is not None:
                    self.focus = c
                    self.explore_yaw = yaw
                    self._open_streak = 0
                    return self._mine_focus(world, slot)
            self._last_yaw = yaw
            return act(move=1, yaw=yaw, pitch=4, hotbar=slot)
        self._last_yaw = yaw_bucket_toward(t[0] + 0.5 - x, t[2] + 0.5 - z)
        a = self.nav.toward(world, t[0] + 0.5, t[2] + 0.5, allow_descent=True)
        a["hotbar"] = slot
        return a


class SmeltOp:
    """Run the furnace until `want` ingots total. Needs furnace pos."""

    def __init__(self, furnace_fn, want: int):
        self.furnace_fn = furnace_fn
        self.want = want
        self.nav = Navigator()

    def done(self, world) -> bool:
        return world.count_item(ids.ITEM_IRON_INGOT) >= self.want

    def act(self, world) -> dict:
        f = self.furnace_fn()
        x, y, z = world.agent_pos()
        if math.hypot(x - (f[0] + 0.5), z - (f[2] + 0.5)) > 2.2:
            return self.nav.toward(world, f[0] + 0.5, f[2] + 0.5)
        remaining, out_ready, fuel_left = world.furnace_state(*f)
        eye = (x, y + 1.62, z)
        yb, pb = aim_at(eye, f[0] + 0.5, f[1] + 0.5, f[2] + 0.5)
        if out_ready:
            return act(use=1, yaw=yb, pitch=pb)
        if remaining == 0 and world.count_item(ids.IRON_ORE) > 0:
            return act(use=1, yaw=yb, pitch=pb)
        return act(yaw=yb, pitch=pb)  # smelting: wait


class GatherExpert:
    """T2 ladder expert: ordered stages, each self-completing."""

    def __init__(self, goal: str, seed: int = 0):
        self.goal = goal
        self.table_pos: tuple[int, int, int] | None = None
        self.furnace_pos: tuple[int, int, int] | None = None
        self.stages: list[Stage] = self._build(seed)
        self._idx = 0

    def _build(self, seed: int) -> list[Stage]:
        def table_block_near(w) -> bool:
            # side effect: remember WHERE the table is (carry_table's
            # at_fn reads this) — the op may be skipped by the guard, so
            # detection must live here, not in PlaceOp
            found = w.find_blocks(ids.CRAFTING_TABLE, 4)
            if found:
                self.table_pos = found[0]
                return True
            return False

        S: list[Stage] = []
        S.append(Stage(lambda w: w.count_item(ids.LOG) >= 3,
                       HarvestOp(ids.LOG, None, seed=seed)))
        S.append(Stage(lambda w: w.count_item(ids.PLANKS) >= 12, CraftOp(1)))
        S.append(Stage(lambda w: w.count_item(ids.ITEM_STICK) >= 6, CraftOp(2)))
        S.append(Stage(lambda w: w.count_item(ids.CRAFTING_TABLE) >= 1, CraftOp(3)))
        S.append(Stage(lambda w: self.table_pos is not None,
                       PlaceOp(ids.CRAFTING_TABLE, lambda p: setattr(self, "table_pos", p))))
        S.append(Stage(lambda w: w.count_item(ids.ITEM_WOODEN_PICKAXE) >= 1, CraftOp(4)))
        # carry the table along (mining it drops its item form) — no return
        # trips through unclimbable terrain
        S.append(Stage(lambda w: w.count_item(ids.CRAFTING_TABLE) >= 1,
                       HarvestOp(ids.CRAFTING_TABLE, None, drop=ids.CRAFTING_TABLE, seed=seed, at_fn=lambda: self.table_pos)))
        S.append(Stage(lambda w: w.count_item(ids.COBBLESTONE) >= 12,
                       HarvestOp(ids.STONE, ids.ITEM_WOODEN_PICKAXE, drop=ids.COBBLESTONE, seed=seed)))
        S.append(Stage(table_block_near, PlaceOp(ids.CRAFTING_TABLE, lambda p: setattr(self, "table_pos", p))))
        S.append(Stage(lambda w: w.count_item(ids.ITEM_STONE_PICKAXE) >= 1, CraftOp(5)))
        if self.goal in ("collect_log", "craft_planks", "craft_table",
                         "craft_wooden_pickaxe", "mine_stone", "craft_stone_pickaxe"):
            return S
        # batch both table-needing crafts at this visit (furnace needs 8 of
        # the 12 cobble gathered), then carry the table along once
        S.append(Stage(lambda w: w.count_item(ids.FURNACE) >= 1, CraftOp(7)))
        S.append(Stage(lambda w: w.count_item(ids.CRAFTING_TABLE) >= 1,
                       HarvestOp(ids.CRAFTING_TABLE, None, drop=ids.CRAFTING_TABLE, seed=seed, at_fn=lambda: self.table_pos)))
        S.append(Stage(lambda w: w.count_item(ids.ITEM_COAL) >= 2,
                       HarvestOp(ids.COAL_ORE, ids.ITEM_STONE_PICKAXE, drop=ids.ITEM_COAL, seed=seed)))
        want = 3 if self.goal in ("craft_iron_pickaxe", "mine_diamond") else 1
        S.append(Stage(lambda w: w.count_item(ids.IRON_ORE) >= want,
                       HarvestOp(ids.IRON_ORE, ids.ITEM_STONE_PICKAXE, drop=ids.IRON_ORE, deep=True, mine_level=44, seed=seed)))
        # place the furnace wherever the agent ended up — carrying it beats
        # climbing back through the mine
        S.append(Stage(lambda w: self.furnace_pos is not None,
                       PlaceOp(ids.FURNACE, lambda p: setattr(self, "furnace_pos", p))))
        S.append(Stage(lambda w: w.count_item(ids.ITEM_IRON_INGOT) >= want,
                       SmeltOp(lambda: self.furnace_pos, want)))
        if self.goal == "smelt_iron":
            return S
        S.append(Stage(table_block_near, PlaceOp(ids.CRAFTING_TABLE, lambda p: setattr(self, "table_pos", p))))
        S.append(Stage(lambda w: w.count_item(ids.ITEM_IRON_PICKAXE) >= 1, CraftOp(6)))
        if self.goal == "craft_iron_pickaxe":
            return S
        S.append(Stage(lambda w: w.count_item(ids.ITEM_DIAMOND) >= 1,
                       HarvestOp(ids.DIAMOND_ORE, ids.ITEM_IRON_PICKAXE, drop=ids.ITEM_DIAMOND, radius=24, deep=True, mine_level=12, seed=seed)))
        return S

    @staticmethod
    def _near(pos, world, tol: float = 3.0) -> bool:
        if pos is None:
            return False
        x, _, z = world.agent_pos()
        return math.hypot(x - (pos[0] + 0.5), z - (pos[2] + 0.5)) <= tol

    def act(self, world) -> dict:
        # monotonic progression: a completed stage is never re-entered, so
        # later crafts consuming earlier items can't ping-pong the plan
        while self._idx < len(self.stages) and self.stages[self._idx].guard(world):
            self._idx += 1
        if self._idx >= len(self.stages):
            return act()
        return self.stages[self._idx].op.act(world)


class CollapseJudgeExpert:
    """Oracle: answers using the same branch simulation as the task truth."""

    def __init__(self, task):
        self.task = task
        self.nav = Navigator()
        self.goal = (-4.5, -2.5) if task.collapses else (-4.5, 3.5)

    def act(self, world) -> dict:
        return self.nav.toward(world, *self.goal)


class WaterRoutingExpert:
    """Digs the channel cells in order, breaches the rim last."""

    def __init__(self, task):
        self.task = task
        self.nav = Navigator()
        self.focus = None

    def act(self, world) -> dict:
        cells = [(x, 4, 0) for x in range(1, 7)] + [(1, 7, 0)]  # rim last
        x, y, z = world.agent_pos()
        if self.focus is not None:
            if cell_id_of(world, *self.focus) == ids.AIR:
                self.focus = None
            else:
                yb, pb = aim_at((x, y + 1.62, z), *(c + 0.5 for c in self.focus))
                return act(mine=1, yaw=yb, pitch=pb)
        for c in cells:
            if cell_id_of(world, *c) == ids.AIR:
                continue
            cx, cy, cz = c[0] + 0.5, c[1] + 0.5, c[2] + 0.5
            dy = cy - (y + 1.62)
            horiz = math.hypot(cx - x, cz - z)
            d3 = math.hypot(horiz, dy)
            pitch = math.degrees(math.atan2(-dy, horiz)) if horiz > 1e-9 else 90.0
            if d3 <= 4.0 and -60.0 <= pitch <= 60.0:
                self.focus = c
                yb, pb = aim_at((x, y + 1.62, z), cx, cy, cz)
                return act(mine=1, yaw=yb, pitch=pb)
            # stand beside the cell (not on it)
            stand = (c[0] + 0.5, c[2] + 1.5)
            return self.nav.toward(world, *stand)
        return act()  # channel done; water routes itself


class BridgeOverLavaExpert:
    """Extends a plank bridge over the trench.

    Per tick: if there is footing within 2 below the next cell, walk (jump
    when the next cell is solid — a step up); else, if firmly on the ground,
    place a plank ahead (pitch 8 lands it on the lava top at bridge level).
    Never places while airborne (a mid-jump place ramps onto the current
    plank), never steps without verified footing (no gap-runs into lava).
    """

    def __init__(self, task):
        self.task = task
        self.nav = Navigator()

    def act(self, world) -> dict:
        x, y, z = world.agent_pos()
        pad = self.task.PAD
        if x > 16.2 or world.count_item(ids.PLANKS) == 0:
            return self.nav.toward(world, pad[0], pad[2])
        slot = max(world.swap_to_hotbar(ids.PLANKS), 0)

        def solid(c) -> bool:
            return cell_id_of(world, *c) not in (ids.AIR, ids.LAVA)

        fy = int(math.floor(y))
        nz = int(math.floor(z))
        nx = int(math.floor(x + 0.9))
        if solid((nx, fy, nz)):
            return act(move=1, jump=1, yaw=18, pitch=4, hotbar=slot)
        if solid((nx, fy - 1, nz)):
            return act(move=1, yaw=18, pitch=4, hotbar=slot)
        if solid((nx, fy - 2, nz)):
            return act(move=1, yaw=18, pitch=4, hotbar=slot)
        if world.obs_pose()[5] == 1.0:
            return act(place=1, yaw=18, pitch=8, hotbar=slot)
        return act(move=1, yaw=18, pitch=4, hotbar=slot)  # airborne: carry through


class BuriedEscapeExpert:
    """Digs out of the sand burial, then walks to the pad."""

    def __init__(self, task):
        self.task = task
        self.nav = Navigator()
        self.focus = None
        self.yaw = 18  # +x toward the pad side

    def act(self, world) -> dict:
        x, y, z = world.agent_pos()
        pad = self.task.PAD
        eye = (x, y + 1.62, z)

        if self.focus is not None:
            if cell_id_of(world, *self.focus) == ids.AIR:
                self.focus = None
            else:
                yb, pb = aim_at(eye, *(c + 0.5 for c in self.focus))
                return act(mine=1, yaw=yb, pitch=pb)

        head_cell = (math.floor(x), math.floor(eye[1]), math.floor(z))
        if cell_id_of(world, *head_cell) not in (ids.AIR, ids.WATER):
            self.focus = head_cell  # origin-cell hit: any aim works
            return act(mine=1, yaw=self.yaw, pitch=4)

        fx, fz = fwd_vec(self.yaw)
        ahead_head = (int(math.floor(x + fx)), int(math.floor(y)) + 1, int(math.floor(z + fz)))
        ahead_feet = (int(math.floor(x + fx)), int(math.floor(y)), int(math.floor(z + fz)))
        for c in (ahead_head, ahead_feet):
            if cell_id_of(world, *c) not in (ids.AIR, ids.WATER):
                self.focus = c
                yb, pb = aim_at(eye, *(cc + 0.5 for cc in c))
                return act(mine=1, yaw=yb, pitch=pb)

        if math.hypot(x - pad[0], z - pad[2]) > 1.0:
            return self.nav.toward(world, pad[0], pad[2])
        return act()


class CircuitDoorExpert:
    """Flip each lever in path order, then walk through the doorway."""

    def __init__(self, task):
        self.task = task
        self.nav = Navigator()
        self.levers = [(8, 5, 0)] + ([(12, 5, 0)] if task.two else [])
        self.target = task.target

    def act(self, world) -> dict:
        x, y, z = world.agent_pos()
        for (lx, ly, lz) in self.levers:
            if cell_id_of(world, lx, ly, lz) == ids.LEVER and (world.get_block(lx, ly, lz) >> 12) & 1 == 0:
                # lever still off: approach and flip
                if math.hypot(x - (lx + 0.5), z - (lz + 0.5)) > 2.0:
                    return self.nav.toward(world, lx + 0.5, lz + 0.5)
                yb, pb = aim_at((x, y + 1.62, z), lx + 0.5, ly + 0.5, lz + 0.5)
                return act(use=1, yaw=yb, pitch=pb)
        return self.nav.toward(world, self.target[0], self.target[2])


class FirebreakJudgeExpert:
    """Oracle: answers from the branch-simulated truth."""

    def __init__(self, task):
        self.task = task
        self.nav = Navigator()
        self.goal = (-4.5, -2.5) if task.burns else (-4.5, 3.5)

    def act(self, world) -> dict:
        return self.nav.toward(world, *self.goal)


class PlateDoorExpert:
    """Walk to the target; the plate sits on the path and opens the door."""

    def __init__(self, task):
        self.task = task
        self.nav = Navigator()

    def act(self, world) -> dict:
        t = self.task.TARGET
        return self.nav.toward(world, t[0], t[2])


class TntClearExpert:
    """Flip the lever, wait out the blast, walk through the gap."""

    def __init__(self, task):
        self.task = task
        self.nav = Navigator()
        self.lever = (7, 6, 0)

    def act(self, world) -> dict:
        x, y, z = world.agent_pos()
        lx, ly, lz = self.lever
        if cell_id_of(world, lx, ly, lz) == ids.LEVER and (world.get_block(lx, ly, lz) >> 12) & 1 == 0:
            if math.hypot(x - (lx + 0.5), z - (lz + 0.5)) > 2.0:
                return self.nav.toward(world, lx + 0.5, lz + 0.5)
            yb, pb = aim_at((x, y + 1.62, z), lx + 0.5, ly + 0.5, lz + 0.5)
            return act(use=1, yaw=yb, pitch=pb)
        # lever on: wait at a safe distance until the wall is breached
        breached = cell_id_of(world, 10, 5, 0) == ids.AIR and cell_id_of(world, 11, 5, 0) == ids.AIR
        if not breached:
            if math.hypot(x - 5.5, z - 2.5) > 1.0:
                return self.nav.toward(world, 5.5, 2.5)  # stand clear of the blast
            return act()
        return self.nav.toward(world, self.task.TARGET[0], self.task.TARGET[2])


class LogicProbeExpert:
    """Oracle: evaluates the circuit's truth table by branch-simulating a
    cloned world (the sim is the ground truth), picks the satisfying input
    combo nearest to the current lever states, then walks and flips the
    levers that differ. Waits for the gate-delay settle between flips.
    """

    SETTLE = 6

    def __init__(self, task):
        self.task = task
        self.nav = Navigator()
        self.plan = None  # [(lever_cell, target_state), ...]
        self.cool = 0

    def _truth_table(self, world):
        import voxelgym_rs as rs

        out = {}
        for a in (0, 1):
            for b in (0, 1):
                scratch = rs.PyWorld(0, "void")
                scratch.restore(world.snapshot())
                scratch.set_block(*self.task.LEVER_A, ids.LEVER | (a << 12))
                scratch.set_block(*self.task.LEVER_B, ids.LEVER | (b << 12))
                for _ in range(8):
                    scratch.step((0, 0, 0, 0, 4, 0, 0, 0, 0, 0))
                out[(a, b)] = (scratch.get_block(*self.task.lamp) >> 12) & 1
        return out

    def act(self, world) -> dict:
        if self.plan is None:
            table = self._truth_table(world)
            cur = {
                self.task.LEVER_A: (world.get_block(*self.task.LEVER_A) >> 12) & 1,
                self.task.LEVER_B: (world.get_block(*self.task.LEVER_B) >> 12) & 1,
            }
            good = [c for c, v in table.items() if v == self.task.goal]
            assert good, f"template {self.task.template} cannot reach goal {self.task.goal}"
            a, b = min(good, key=lambda c: (c[0] != cur[self.task.LEVER_A]) + (c[1] != cur[self.task.LEVER_B]))
            self.plan = []
            if a != cur[self.task.LEVER_A]:
                self.plan.append((self.task.LEVER_A, a))
            if b != cur[self.task.LEVER_B]:
                self.plan.append((self.task.LEVER_B, b))
        if self.cool > 0:
            self.cool -= 1
            return act()
        if world.tick() >= 8 and (world.get_block(*self.task.lamp) >> 12) & 1 == self.task.goal:
            return act()  # satisfied; the reward fires
        for cell, target in self.plan:
            lx, ly, lz = cell
            cur = (world.get_block(lx, ly, lz) >> 12) & 1
            if cur == target:
                continue
            x, y, z = world.agent_pos()
            if math.hypot(x - (lx + 0.5), z - (lz + 0.5)) > 2.0:
                return self.nav.toward(world, lx + 0.5, lz + 0.5)
            yb, pb = aim_at((x, y + 1.62, z), lx + 0.5, ly + 0.5, lz + 0.5)
            self.cool = self.SETTLE
            return act(use=1, yaw=yb, pitch=pb)
        return act()


def make_expert(task_name: str, task, seed: int = 0):
    if task_name == "navigate_to_target":
        return T1Expert(task)
    if task_name == "collapse_judge":
        return CollapseJudgeExpert(task)
    if task_name == "water_routing":
        return WaterRoutingExpert(task)
    if task_name == "bridge_over_lava":
        return BridgeOverLavaExpert(task)
    if task_name == "buried_escape":
        return BuriedEscapeExpert(task)
    if task_name in ("circuit_door", "circuit_door_two"):
        return CircuitDoorExpert(task)
    if task_name == "firebreak_judge":
        return FirebreakJudgeExpert(task)
    if task_name == "plate_door":
        return PlateDoorExpert(task)
    if task_name == "tnt_clear":
        return TntClearExpert(task)
    if task_name == "logic_probe":
        return LogicProbeExpert(task)
    return GatherExpert(task_name, seed=seed)


def run_episode(task_name: str, seed: int, record_dir: str | None = None, render=False, epsilon: float = 0.0):
    from .recorder import Recorder

    task = make_task(task_name)
    env = VoxelGymEnv(task=task, seed=seed, render=render)
    env.reset(seed=seed)
    expert = make_expert(task_name, task, seed=seed)
    rec = Recorder(record_dir, task_name, seed, render=bool(render)) if record_dir else None
    rng = np.random.default_rng(seed + 999_983)
    success = False
    steps = 0
    while True:
        a = expert.act(env.world)
        swap = env.world.take_swap()  # inventory-management event from act()
        if epsilon > 0.0 and rng.random() < epsilon:
            # uniform random action (dataset diversity for world models)
            a = {
                "move": int(rng.integers(0, 5)),
                "jump": int(rng.integers(0, 2)),
                "sneak": int(rng.integers(0, 2)),
                "yaw": int(rng.integers(0, 24)),
                "pitch": int(rng.integers(0, 9)),
                "mine": int(rng.integers(0, 2)),
                "place": 0,
                "use": int(rng.integers(0, 2)),
                "hotbar": int(rng.integers(0, 9)),
                "craft": 0,
            }
        obs, r, term, trunc, _ = env.step(a)
        frames = None
        if render:
            frames = (obs["rgb"], obs["depth"], obs["seg"])
        if rec:
            rec.log(env.world, tuple(int(a[k]) for k in ACTION_KEYS), r, term or trunc, frames, swap=swap)
        steps += 1
        if term or trunc:
            success = bool(term) and not env.world.dead()
            break
    final_hash = env.world.hash()
    path = rec.save(final_hash) if rec else None
    env.close() if hasattr(env, "close") else None
    return success, steps, final_hash, path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=task_names())
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--record", type=str, default=None)
    args = ap.parse_args(argv)

    wins = 0
    for i in range(args.episodes):
        seed = args.seed0 + i
        ok, steps, h, _ = run_episode(args.task, seed, record_dir=args.record)
        wins += ok
        print(f"  ep {i} seed={seed}: {'OK' if ok else 'fail'} in {steps} ticks hash={h:016x}", flush=True)
    rate = wins / args.episodes
    print(f"{args.task}: success {wins}/{args.episodes} = {rate:.2f}")
    return 0 if rate >= 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())
