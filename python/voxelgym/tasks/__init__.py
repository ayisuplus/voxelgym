"""T1/T2 task families (M2). T3/T4 probe tasks arrive in M3."""

from __future__ import annotations

import math

import numpy as np

from .. import ids
from .base import RewardOutcome, Task, evidence_event_ids
from .metric import (
    agent_position_meters,
    metric_set_block,
    metric_surface_y,
)

# Achievement ladder: (milestone name -> item id counted in inventory).
LADDER: list[tuple[str, int]] = [
    ("collect_log", ids.LOG),
    ("craft_planks", ids.PLANKS),
    ("craft_table", ids.CRAFTING_TABLE),
    ("craft_wooden_pickaxe", ids.ITEM_WOODEN_PICKAXE),
    ("mine_stone", ids.COBBLESTONE),
    ("craft_stone_pickaxe", ids.ITEM_STONE_PICKAXE),
    ("place_furnace", ids.FURNACE),
    ("mine_iron_ore", ids.IRON_ORE),
    ("mine_coal", ids.ITEM_COAL),
    ("smelt_iron", ids.ITEM_IRON_INGOT),
    ("craft_iron_pickaxe", ids.ITEM_IRON_PICKAXE),
    ("mine_diamond", ids.ITEM_DIAMOND),
]
LADDER_INDEX = {name: i for i, (name, _) in enumerate(LADDER)}

T2_GOALS = [
    "collect_log", "craft_planks", "craft_table", "craft_wooden_pickaxe",
    "mine_stone", "craft_stone_pickaxe", "smelt_iron", "craft_iron_pickaxe",
    "mine_diamond",
]


class NavigateToTarget(Task):
    """T1: reach a 5-tall torch pillar. Reward = -delta_dist, +1.0 on success.

    Generation-time solvability (per plan contingency): one heightfield BFS
    from the spawn column over a +-45 box; the pillar is only placed on a
    reachable column. Passability: up-step <= 1 (jump), down-step <= 4
    (survivable), water columns treated as sea level (swimmable).
    """

    name = "navigate_to_target"
    preset = "default"
    horizon = 2400

    def _reachable_columns(self, world, x0: int, z0: int, r: int = 45) -> set[tuple[int, int]]:
        from collections import deque

        h: dict[tuple[int, int], int] = {}
        for cx in range(x0 - r, x0 + r + 1):
            for cz in range(z0 - r, z0 + r + 1):
                sy = metric_surface_y(world, cx, cz)
                if sy < 0:
                    sy = 0
                if sy < 62:  # water body above: swim at sea level
                    sy = 62
                h[(cx, cz)] = sy
        start = (x0, z0)
        seen = {start}
        q = deque([start])
        while q:
            cx, cz = q.popleft()
            ha = h[(cx, cz)]
            for nb in ((cx + 1, cz), (cx - 1, cz), (cx, cz + 1), (cx, cz - 1)):
                if nb not in h or nb in seen:
                    continue
                hb = h[nb]
                if hb - ha <= 1 and ha - hb <= 4:
                    seen.add(nb)
                    q.append(nb)
        return seen

    def on_reset(self, world, rng: np.random.Generator):
        x0, y0, z0 = agent_position_meters(world)
        spawn_column = (math.floor(x0), math.floor(z0))
        reach = self._reachable_columns(world, *spawn_column)
        tx = tz = None
        for _ in range(20):
            angle = float(rng.uniform(0, 2 * math.pi))
            dist = float(rng.uniform(20.0, 40.0))
            cx = int(round(x0 + dist * math.cos(angle)))
            cz = int(round(z0 + dist * math.sin(angle)))
            if (cx, cz) in reach:
                tx, tz = cx, cz
                break
        if tx is None:
            # fallback: any reachable column, distance as close to 25 as possible
            cands = [c for c in reach if c != spawn_column]
            cands.sort(key=lambda c: abs(math.hypot(c[0] - x0, c[1] - z0) - 25.0))
            tx, tz = cands[0]
        sy = metric_surface_y(world, tx, tz)
        if sy < 0:
            sy = math.floor(y0)
        for i in range(1, 6):
            metric_set_block(world, (tx, sy + i, tz), ids.TORCH)
        self.target = (tx + 0.5, float(sy + 1), tz + 0.5)
        self._prev = self._dist(world)

    def _dist(self, world) -> float:
        x, _, z = agent_position_meters(world)
        return math.hypot(x - self.target[0], z - self.target[2])

    def reward_outcome(self, world, events=()) -> RewardOutcome:
        d = self._dist(world)
        progress = self._prev - d
        if d <= 2.0:
            return RewardOutcome(
                total=progress + 1.0,
                components={"progress": progress, "success": 1.0},
                terminated=True,
                termination_reason="target_reached",
                evidence_event_ids=evidence_event_ids(
                    events, kinds=("agent_moved",)
                ),
                evidence_labels=("task:navigate_to_target:target_reached",),
                task_state_updates={"_prev": d},
            )
        return RewardOutcome(
            total=progress,
            components={"progress": progress},
            task_state_updates={"_prev": d},
        )


class AchievementTask(Task):
    """T2: achievement ladder goal. +1.0 on goal, +0.1 per first-time
    prerequisite milestone (shaping). Done on goal."""

    preset = "default"

    def __init__(self, goal: str):
        assert goal in T2_GOALS, goal
        self.goal = goal
        self.name = goal
        self.horizon = 24000 if goal == "mine_diamond" else 12000
        self._goal_item = dict(LADDER)[goal]
        self._goal_idx = LADDER_INDEX[goal]
        self._shaped: set[str] = set()

    def on_reset(self, world, rng: np.random.Generator):
        self._shaped = set()

    def reward_outcome(self, world, events=()) -> RewardOutcome:
        reward = 0.0
        evidence: list[str] = []
        shaped = set(self._shaped)
        for name, item in LADDER[: self._goal_idx]:
            if name not in shaped and world.count_item(item) >= 1:
                shaped.add(name)
                reward += 0.1
                evidence.append(f"task:{self.name}:milestone:{name}")
        updates = {"_shaped": sorted(shaped)} if shaped != self._shaped else {}
        if world.count_item(self._goal_item) >= 1:
            evidence.append(f"task:{self.name}:goal_achieved")
            components = {"success": 1.0}
            if reward:
                components["milestones"] = reward
            return RewardOutcome(
                total=reward + 1.0,
                components=components,
                terminated=True,
                termination_reason="goal_achieved",
                evidence_event_ids=evidence_event_ids(
                    events,
                    kinds=(
                        "inventory_changed",
                        "item_picked",
                        "crafted",
                        "smelted",
                        "block_mined",
                    ),
                ),
                evidence_labels=tuple(evidence),
                task_state_updates=updates,
            )
        return RewardOutcome(
            total=reward,
            components={"milestones": reward} if reward else {},
            evidence_event_ids=evidence_event_ids(
                events,
                kinds=(
                    "inventory_changed",
                    "item_picked",
                    "crafted",
                    "smelted",
                    "block_mined",
                ),
            ),
            evidence_labels=tuple(evidence),
            task_state_updates=updates,
        )


def make_task(name: str) -> Task:
    if name == "navigate_to_target":
        return NavigateToTarget()
    if name in T2_GOALS:
        return AchievementTask(name)
    from .probes import make_probe

    return make_probe(name)


def task_names() -> list[str]:
    from .probes import PROBE_TASKS

    return ["navigate_to_target"] + T2_GOALS + PROBE_TASKS
