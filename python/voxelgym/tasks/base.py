"""Task base protocol. Tasks are duck-typed into VoxelGymEnv:

- preset: world preset used to construct the world
- horizon: tick budget (truncation)
- scenario(rng) -> scenario spec list | None (deterministic in episode seed)
- semantic_regions(rng) -> stable region/structure tuples | None; when
  present this is the authoritative compound-scene definition
- on_reset(world, rng): place markers, teleport, grant items
- interventions_before_step(world, action): serializable interventions before physics
- reward_outcome(world, events=()): structured, world-read-only reward evaluation;
  event IDs may be selected as causal evidence without mutating the world
- state_dict()/load_state_dict(): JSON-safe mutable task state
- step_reward(world) -> (reward, terminated), retained as a read-through
  compatibility view for tasks whose source of truth is reward_outcome
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import numpy as np

from ..task_state import (
    TASK_STATE_VERSION,
    RewardOutcome,
    TaskState,
    decode_task_fields,
    encode_task_fields,
    task_type_name,
)
from .metric import agent_position_meters


def evidence_event_ids(
    events,
    *,
    kinds: set[str] | tuple[str, ...],
    locations: set[tuple[int, int, int]] | None = None,
) -> tuple[int, ...]:
    """Select only task-declared World Events as reward evidence."""

    allowed_kinds = set(kinds)
    selected: list[int] = []
    for event in events:
        if not isinstance(event, dict) or event.get("kind") not in allowed_kinds:
            continue
        location = event.get("location")
        if locations is not None and (
            location is None or tuple(location) not in locations
        ):
            continue
        event_id = event.get("id")
        if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id < 0:
            raise ValueError("reward evidence contains an invalid World Event ID")
        selected.append(event_id)
    return tuple(dict.fromkeys(selected))


class Task:
    name = "base"
    preset = "default"
    horizon: int | None = None

    def scenario(self, rng: np.random.Generator):
        return None

    def semantic_regions(self, rng: np.random.Generator):
        return None

    def on_reset(self, world, rng: np.random.Generator):
        pass

    def interventions_before_step(
        self, world, action: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Return tagged, serializable interventions for the next transition."""

        return []

    def reward_outcome(self, world, events=()) -> RewardOutcome:
        return RewardOutcome()

    def validate_reward_contract(self) -> None:
        """Reject legacy-only reward overrides before an episode can run.

        Calling an overridden ``step_reward`` to discover its scalar result is
        not safe: historical callbacks were allowed to mutate both the task
        and World State.  A task may keep a legacy method for callers outside
        Gym only when it also supplies the pure ``reward_outcome`` source of
        truth that the environment uses.
        """

        if (
            type(self).step_reward is not Task.step_reward
            and type(self).reward_outcome is Task.reward_outcome
        ):
            raise TypeError(
                "legacy step_reward overrides are not supported by VoxelGymEnv; "
                "implement pure reward_outcome(world, events=()) returning RewardOutcome"
            )

    def step_reward(self, world) -> tuple[float, bool]:
        """Compatibility adapter for callers that still consume a pair."""
        outcome = self.reward_outcome(world)
        self.commit_reward(outcome)
        return outcome.total, outcome.terminated

    def commit_reward(self, outcome: RewardOutcome) -> None:
        """Commit explicit state updates after pure reward evaluation.

        Reward payloads stay JSON-native.  Existing set/tuple fields retain
        their in-memory types when an update is committed.
        """

        prepared: dict[str, Any] = {}
        for name, raw_value in outcome.task_state_updates.items():
            if name not in self.__dict__:
                raise ValueError(f"reward update targets unknown task field {name!r}")
            current = getattr(self, name)
            value = deepcopy(raw_value)
            if isinstance(current, set):
                if not isinstance(value, list):
                    raise TypeError(f"reward update for set field {name!r} must be a list")
                value = set(value)
            elif isinstance(current, tuple):
                if not isinstance(value, list):
                    raise TypeError(f"reward update for tuple field {name!r} must be a list")
                value = tuple(value)
            prepared[name] = value
        for name, value in prepared.items():
            setattr(self, name, value)

    def state_dict(self) -> dict[str, Any]:
        """Return all instance fields as a versioned, JSON-safe value."""
        return TaskState(
            schema_version=TASK_STATE_VERSION,
            task_type=task_type_name(self),
            task_name=self.name,
            fields=encode_task_fields(dict(self.__dict__)),
        ).as_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        task_state = TaskState.from_dict(state)
        if task_state.schema_version != TASK_STATE_VERSION:
            raise ValueError(f"unsupported task state version: {state.get('schema_version')!r}")
        expected = task_type_name(self)
        if task_state.task_type != expected:
            raise ValueError(
                f"task type mismatch: expected {expected!r}, got {state.get('task_type')!r}"
            )
        if task_state.task_name != self.name:
            raise ValueError(
                f"task name mismatch: expected {self.name!r}, got {state.get('task_name')!r}"
            )
        restored = decode_task_fields(task_state.fields)
        self.__dict__.clear()
        self.__dict__.update(restored)

    def reach_reward(self, world, target, tol: float = 1.2) -> tuple[float, bool]:
        """Terminal +1.0 when the agent is within tol of target (XZ plane).

        Shared by the walk-to-a-pad probe tasks (BuriedEscape, CircuitDoor,
        PlateDoor, TntClear); BridgeOverLava adds its own height condition.
        """
        x, _, z = agent_position_meters(world)
        if math.hypot(x - target[0], z - target[2]) <= tol:
            return 1.0, True
        return 0.0, False

    def reach_outcome(
        self,
        world,
        target,
        tol: float = 1.2,
        *,
        reason: str = "target_reached",
        events=(),
    ) -> RewardOutcome:
        reward, terminated = self.reach_reward(world, target, tol)
        if not terminated:
            return RewardOutcome()
        return RewardOutcome(
            total=reward,
            components={"success": reward},
            terminated=True,
            termination_reason=reason,
            evidence_event_ids=evidence_event_ids(
                events, kinds=("agent_moved",)
            ),
            evidence_labels=(f"task:{self.name}:{reason}",),
        )


__all__ = ["RewardOutcome", "Task", "evidence_event_ids"]
