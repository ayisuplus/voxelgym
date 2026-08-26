"""Public contracts for resumable Python episodes and task outcomes."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from voxelgym import ACTION_KEYS, VoxelGymEnv
from voxelgym import ids
from voxelgym.tasks import AchievementTask, make_task, task_names
from voxelgym.task_state import EnvSnapshot
from voxelgym.tasks.base import RewardOutcome, Task
from voxelgym.tasks.probes import CollapseJudge, FirebreakJudge


def _action(**updates: int) -> dict[str, int]:
    action = {key: 0 for key in ACTION_KEYS}
    action["pitch"] = 4
    action.update(updates)
    return action


def _copied_step(env: VoxelGymEnv, action: dict[str, int]):
    obs, reward, terminated, truncated, info = env.step(action)
    return (
        {key: value.copy() for key, value in obs.items()},
        reward,
        terminated,
        truncated,
        info,
        env.world.hash(),
    )


def _assert_steps_equal(actual, expected) -> None:
    actual_obs, *actual_tail = actual
    expected_obs, *expected_tail = expected
    assert actual_obs.keys() == expected_obs.keys()
    for key in actual_obs:
        np.testing.assert_array_equal(actual_obs[key], expected_obs[key])
    assert actual_tail == expected_tail


def test_task_state_round_trips_as_json_for_every_builtin_task():
    """Every built-in task exposes a portable state, including scenario fields."""
    for name in task_names():
        task = make_task(name)
        task.scenario(np.random.default_rng(19))
        encoded = json.loads(json.dumps(task.state_dict(), sort_keys=True))

        restored = make_task(name)
        restored.load_state_dict(encoded)

        assert restored.state_dict() == encoded

    shaped = AchievementTask("craft_planks")
    shaped._shaped.add("collect_log")
    encoded = json.loads(json.dumps(shaped.state_dict()))
    restored = AchievementTask("craft_planks")
    restored.load_state_dict(encoded)
    assert restored._shaped == {"collect_log"}


def test_task_state_rejects_a_different_task_type():
    state = make_task("navigate_to_target").state_dict()
    with pytest.raises(ValueError, match="task type"):
        make_task("collect_log").load_state_dict(state)

    achievement_state = make_task("collect_log").state_dict()
    with pytest.raises(ValueError, match="task name"):
        make_task("craft_planks").load_state_dict(achievement_state)


def test_env_snapshot_restores_rng_task_world_and_sensor_continuation():
    lidar = {
        "channels": 2,
        "azimuth": 8,
        "every": 2,
        "min_elev": -10.0,
        "max_elev": 10.0,
        "max_range": 24.0,
        "noise_sigma": 0.01,
        "noise_seed": 41,
    }
    env = VoxelGymEnv(
        task=make_task("navigate_to_target"),
        seed=23,
        render=3,
        lidar=lidar,
    )
    env.reset(seed=23)
    env.step(_action(move=1, yaw=2))
    env.step(_action(move=1, yaw=2))

    snapshot = env.snapshot()
    assert snapshot.episode_seed == 23
    assert snapshot.render_sample_tick == 0
    assert snapshot.lidar_sample_tick == 2
    assert snapshot.last_frames is not None
    assert snapshot.last_scan is not None
    expected_rng = int(env.np_random.integers(1_000_000))

    future = [
        _action(move=1, yaw=2),
        _action(jump=1, yaw=2),
        _action(move=4, yaw=7),
        _action(move=1, yaw=7),
    ]
    expected = [_copied_step(env, action) for action in future]

    # Perturb every resumable layer before restoring the branch point.
    env.reset(seed=99)
    env.step(_action(move=2, yaw=13))
    env.restore(snapshot)

    assert int(env.np_random.integers(1_000_000)) == expected_rng
    actual = [_copied_step(env, action) for action in future]
    for actual_step, expected_step in zip(actual, expected, strict=True):
        _assert_steps_equal(actual_step, expected_step)


def test_env_snapshot_has_a_portable_versioned_binary_round_trip():
    env = VoxelGymEnv(task=make_task("navigate_to_target"), seed=31, render=2)
    env.reset(seed=31)
    env.step(_action(move=1, yaw=3))

    encoded = env.snapshot().to_bytes()
    decoded = EnvSnapshot.from_bytes(encoded)
    assert decoded.render_every == 2
    assert decoded.spacetime is False
    assert decoded.native_trace_state is not None
    assert decoded.native_trace_state.startswith(b"VXTR1")
    assert decoded.native_intervention_cursor == 0
    assert decoded.intervention_cursor == 0
    expected = _copied_step(env, _action(move=4, yaw=9))

    env.reset(seed=99)
    env.restore(decoded)
    _assert_steps_equal(_copied_step(env, _action(move=4, yaw=9)), expected)

    with pytest.raises(ValueError, match="EnvSnapshot"):
        EnvSnapshot.from_bytes(b"not-an-env-snapshot")


def test_env_snapshot_restores_intervention_cursor_and_guards_native_trace_state():
    class InterveningTask(Task):
        preset = "void"

        def interventions_before_step(self, world, action):
            return [{"kind": "set_cell", "at": [2, 3, 4], "cell": 1}]

    env = VoxelGymEnv(task=InterveningTask(), seed=37)
    env.reset(seed=37)
    env.step_traced(_action(), trace_level="full")
    snapshot = EnvSnapshot.from_bytes(env.snapshot().to_bytes())

    assert snapshot.intervention_cursor == 1
    assert snapshot.native_intervention_cursor == 1
    assert snapshot.native_trace_state is not None
    assert snapshot.native_trace_state.startswith(b"VXTR1")

    env.step_traced(_action(), trace_level="full")
    expected = env.oracle_view()["trace"]

    restored = VoxelGymEnv(task=InterveningTask(), seed=0)
    restored.restore(snapshot)
    assert restored.snapshot().intervention_cursor == 1
    assert restored.snapshot().native_intervention_cursor == 1
    restored.step_traced(_action(), trace_level="full")
    actual = restored.oracle_view()["trace"]
    assert actual == expected

    with pytest.raises(ValueError, match="trace-state|trace state"):
        restored.restore(replace(snapshot, native_trace_state=b"future-trace-state"))


def test_intervention_batch_is_atomic_when_a_later_spec_is_invalid():
    env = VoxelGymEnv(preset="void", seed=38)
    env.reset(seed=38)
    before = env.snapshot()

    with pytest.raises(ValueError, match="intervention kind"):
        env.step_traced(
            _action(),
            trace_level="full",
            interventions=[
                {"kind": "set_cell", "at": [2, 3, 4], "cell": ids.STONE},
                {"kind": "not_a_real_intervention"},
            ],
        )

    after = env.snapshot()
    assert env.world.get_block(2, 3, 4) == ids.AIR
    assert after.world_snapshot == before.world_snapshot
    assert after.native_trace_state == before.native_trace_state
    assert after.native_intervention_cursor == before.native_intervention_cursor
    assert after.intervention_cursor == before.intervention_cursor
    assert env.world.tick() == 0


def test_interventions_are_not_committed_for_an_invalid_action():
    env = VoxelGymEnv(preset="void", seed=38)
    env.reset(seed=38)
    before = env.snapshot()
    invalid_action = _action(move=256)

    with pytest.raises(ValueError, match="action 'move'"):
        env.step_traced(
            invalid_action,
            trace_level="full",
            interventions=[
                {"kind": "set_cell", "at": [2, 3, 4], "cell": ids.STONE},
            ],
        )

    after = env.snapshot()
    assert after.world_snapshot == before.world_snapshot
    assert after.native_trace_state == before.native_trace_state
    assert after.native_intervention_cursor == before.native_intervention_cursor
    assert after.intervention_cursor == before.intervention_cursor


def test_invalid_branch_id_is_rejected_before_task_hooks():
    class CountingInterventionTask(Task):
        preset = "void"

        def __init__(self):
            self.calls = 0

        def interventions_before_step(self, world, action):
            self.calls += 1
            return []

    task = CountingInterventionTask()
    env = VoxelGymEnv(task=task, seed=39)
    env.reset(seed=39)

    with pytest.raises(ValueError, match="branch_id"):
        env.step_traced(_action(), trace_level="events", branch_id=1 << 64)

    assert task.calls == 0
    assert env.world.tick() == 0


def test_invalid_intervention_batch_rolls_back_task_hook_state():
    class StatefulInterventionTask(Task):
        preset = "void"

        def __init__(self):
            self.calls = 0

        def interventions_before_step(self, world, action):
            self.calls += 1
            return [{"kind": "not_a_real_intervention"}]

    task = StatefulInterventionTask()
    env = VoxelGymEnv(task=task, seed=39)
    env.reset(seed=39)

    with pytest.raises(ValueError, match="intervention kind"):
        env.step_traced(_action(), trace_level="events")

    assert task.calls == 0
    assert env.world.tick() == 0
    assert env.snapshot().intervention_cursor == 0


def test_env_does_not_execute_legacy_before_step_mutation_hooks():
    class LegacyMutationTask(Task):
        preset = "void"

        def __init__(self):
            self.called = False

        def before_step(self, world, action):
            self.called = True
            world.set_block(7, 8, 9, 1)

    task = LegacyMutationTask()
    env = VoxelGymEnv(task=task, seed=39)
    env.reset(seed=39)

    env.step(_action())

    assert task.called is False
    assert env.world.get_block(7, 8, 9) == 0


def test_env_rejects_legacy_mutating_reward_without_executing_it():
    class LegacyRewardTask(Task):
        preset = "void"

        def __init__(self):
            self.called = False

        def step_reward(self, world):
            self.called = True
            world.set_block(7, 8, 9, 1)
            return 1.0, True

    task = LegacyRewardTask()

    with pytest.raises(TypeError, match="reward_outcome.*RewardOutcome"):
        VoxelGymEnv(task=task, seed=41)

    assert task.called is False


def test_task_callbacks_cannot_mutate_their_read_only_world_or_reward_state():
    class MutatingInterventionTask(Task):
        preset = "void"

        def __init__(self):
            self.calls = 0

        def interventions_before_step(self, world, action):
            self.calls += 1
            world.set_block(7, 8, 9, 1)
            return []

    intervention_task = MutatingInterventionTask()
    intervention_env = VoxelGymEnv(task=intervention_task, seed=41)
    intervention_env.reset(seed=41)
    before_hash = intervention_env.world.hash()
    with pytest.raises(TypeError, match="read-only World"):
        intervention_env.step(_action())
    assert intervention_env.world.hash() == before_hash
    assert intervention_env.world.tick() == 0
    assert intervention_task.calls == 0

    class MutatingRewardTask(Task):
        preset = "void"

        def __init__(self):
            self.calls = 0

        def reward_outcome(self, world):
            self.calls += 1
            world.set_block(7, 8, 9, 1)
            return RewardOutcome(total=1.0)

    reward_task = MutatingRewardTask()
    reward_env = VoxelGymEnv(task=reward_task, seed=42)
    reward_env.reset(seed=42)
    with pytest.raises(TypeError, match="read-only World"):
        reward_env.step(_action())
    assert reward_env.world.get_block(7, 8, 9) == 0
    assert reward_task.calls == 0


def test_reward_callback_must_return_task_state_updates_instead_of_mutating():
    class MutatingTaskState(Task):
        preset = "void"

        def __init__(self):
            self.counter = 0

        def reward_outcome(self, world):
            self.counter += 1
            return RewardOutcome(total=1.0)

    task = MutatingTaskState()
    env = VoxelGymEnv(task=task, seed=43)
    env.reset(seed=43)

    with pytest.raises(TypeError, match="mutated Task State"):
        env.step(_action())

    assert task.counter == 0


def test_env_rejects_non_structured_reward_results():
    class TupleRewardTask(Task):
        preset = "void"

        def reward_outcome(self, world):
            return 1.0, True

    env = VoxelGymEnv(task=TupleRewardTask(), seed=42)
    env.reset(seed=42)

    with pytest.raises(TypeError, match="must return RewardOutcome"):
        env.step(_action())


def test_env_fork_branches_complete_task_rng_and_sensor_state():
    env = VoxelGymEnv(
        task=make_task("navigate_to_target"), seed=43, render=2, spacetime=True
    )
    env.reset(seed=43)
    env.step(_action(move=1, yaw=3))
    branch = env.fork()

    future = [_action(move=1, yaw=3), _action(move=4, yaw=7)]
    for action in future:
        _assert_steps_equal(_copied_step(branch, action), _copied_step(env, action))

    branch.world.apply_intervention(
        {"kind": "set_cell", "at": [7, 9, 11], "cell": 1}, trace_level="off"
    )
    assert branch.world.hash() != env.world.hash()


def test_env_restore_is_atomic_when_the_world_snapshot_is_malformed():
    env = VoxelGymEnv(task=make_task("navigate_to_target"), seed=47)
    env.reset(seed=47)
    snapshot = env.snapshot()
    env._task._prev += 10.0
    before_hash = env.world.hash()
    before_task = env._task.state_dict()

    with pytest.raises(ValueError):
        env.restore(
            replace(
                snapshot,
                world_snapshot=b"malformed",
            )
        )

    assert env.world.hash() == before_hash
    assert env._task.state_dict() == before_task


@pytest.mark.parametrize(
    "lidar_config",
    ({"channels": 2}, {"channels": 2, "azimuth": 4, "every": 0}),
)
def test_env_restore_is_atomic_when_observation_configuration_is_malformed(
    lidar_config,
):
    env = VoxelGymEnv(task=make_task("navigate_to_target"), seed=49)
    env.reset(seed=49)
    snapshot = env.snapshot()
    before_hash = env.world.hash()
    before_task = env._task.state_dict()
    before_space = set(env.observation_space.spaces)

    with pytest.raises(ValueError, match="lidar_config"):
        env.restore(replace(snapshot, lidar_config=lidar_config))

    assert env.world.hash() == before_hash
    assert env._task.state_dict() == before_task
    assert set(env.observation_space.spaces) == before_space


def test_env_snapshot_restores_sensor_schedule_and_observation_profile():
    lidar = {"channels": 2, "azimuth": 4, "every": 3}
    source = VoxelGymEnv(
        task=make_task("navigate_to_target"),
        seed=53,
        render=2,
        lidar=lidar,
        spacetime=True,
        scale=2.0,
        dt_numerator=1,
        dt_denominator=40,
    )
    source.reset(seed=53)
    snapshot = EnvSnapshot.from_bytes(source.snapshot().to_bytes())

    target = VoxelGymEnv(task=make_task("navigate_to_target"), seed=0)
    target.restore(snapshot)

    assert target._render_every == 2
    assert target._lidar == lidar
    assert target._spacetime is True
    assert target.world.clock()["dt_denominator"] == 40
    assert target.world.oracle_state()["scale"] == 2.0
    assert {"rgb", "lidar_range", "clock", "egomotion"} <= set(
        target.observation_space.spaces
    )


def test_env_snapshot_captures_terminal_and_truncation_bookkeeping():
    task = AchievementTask("collect_log")
    env = VoxelGymEnv(task=task, seed=7)
    env.reset()
    env.world.give(task._goal_item, 1)
    _, reward, terminated, truncated, info = env.step(_action())

    assert reward == 1.0
    assert terminated and not truncated
    assert info["reward_outcome"] == {
        "total": 1.0,
        "components": {"success": 1.0},
        "termination_reason": "goal_achieved",
        "evidence_event_ids": [],
        "evidence_labels": ["task:collect_log:goal_achieved"],
        "task_state_updates": {},
    }
    snapshot = env.snapshot()
    assert snapshot.terminated is True
    assert snapshot.truncated is False
    assert snapshot.last_reward.termination_reason == "goal_achieved"

    env.reset(seed=8)
    env.restore(snapshot)
    restored = env.snapshot()
    assert restored.terminated is True
    assert restored.truncated is False
    assert restored.last_reward == snapshot.last_reward

    class OneTickTask(Task):
        name = "one_tick"
        horizon = 1

    short_env = VoxelGymEnv(task=OneTickTask(), preset="flat", seed=9)
    short_env.reset()
    _, _, short_terminated, short_truncated, _ = short_env.step(_action())
    short_snapshot = short_env.snapshot()
    assert not short_terminated and short_truncated
    assert short_snapshot.terminated is False
    assert short_snapshot.truncated is True
    assert short_snapshot.last_reward.termination_reason == "horizon"


def test_reward_outcome_keeps_gym_reward_scalar():
    class StructuredRewardTask(Task):
        name = "structured_reward"

        def reward_outcome(self, world) -> RewardOutcome:
            return RewardOutcome(
                total=0.75,
                components={"progress": 0.25, "success": 0.5},
                terminated=True,
                termination_reason="target_reached",
                evidence_event_ids=(17,),
                evidence_labels=("test:target_reached",),
            )

    env = VoxelGymEnv(task=StructuredRewardTask(), preset="flat")
    env.reset()
    _, reward, terminated, truncated, info = env.step(_action())

    assert isinstance(reward, float) and reward == 0.75
    assert terminated and not truncated
    assert info["reward_outcome"] == {
        "total": 0.75,
        "components": {"progress": 0.25, "success": 0.5},
        "termination_reason": "target_reached",
        "evidence_event_ids": [17],
        "evidence_labels": ["test:target_reached"],
        "task_state_updates": {},
    }


def test_collapse_reward_is_read_only_and_support_removal_is_pre_step(monkeypatch):
    class World:
        def __init__(self):
            self.position = (-4.5, 6.0, -2.5)
            self.removed: list[tuple[int, int, int, int]] = []

        def agent_pos(self):
            return self.position

        def teleport(self, *position):
            self.position = position

        def set_block(self, x, y, z, cell):
            self.removed.append((x, y, z, cell))

        def tick(self):
            return 12

    world = World()
    task = CollapseJudge()
    task.supported = 3
    monkeypatch.setattr(task, "_truth", lambda _world: True)
    task.on_reset(world, np.random.default_rng(0))
    world.position = (-4.5, 6.0, -2.5)

    # Reward evaluation is observation-only: it cannot alter world state.
    assert task.reward_outcome(world).total == 0.0
    assert world.removed == []

    specs = task.interventions_before_step(world, _action())
    for spec in specs:
        world.set_block(*spec["at"], spec["cell"])
    assert len(world.removed) == 9
    outcome = task.reward_outcome(world)
    assert outcome.total == 1.0 and outcome.terminated
    assert outcome.termination_reason == "correct_answer"


@pytest.mark.parametrize(
    ("task", "prepare"),
    [
        (
            make_task("navigate_to_target"),
            lambda task, env: env.world.teleport(
                task.target[0] + 4.0, task.target[1], task.target[2]
            ),
        ),
        (
            AchievementTask("craft_planks"),
            lambda task, env: env.world.give(ids.LOG, 1),
        ),
        (
            CollapseJudge(),
            lambda task, env: task.__dict__.update(
                collapses=True,
                _done=False,
                _committed=True,
                _answered_collapse=True,
            ),
        ),
        (
            FirebreakJudge(),
            lambda task, env: (
                task.__dict__.update(burns=True, _done=False),
                env.world.teleport(-4.5, 6.0, -2.5),
            ),
        ),
    ],
    ids=("navigate", "achievement", "collapse", "firebreak"),
)
def test_reward_evaluation_is_repeatable_and_has_no_task_or_world_side_effects(
    task, prepare
):
    env = VoxelGymEnv(task=task, preset="flat", seed=61)
    env.reset(seed=61)
    prepare(task, env)
    before_task = task.state_dict()
    before_hash = env.world.hash()

    first = task.reward_outcome(env.world)
    second = task.reward_outcome(env.world)

    assert second == first
    assert task.state_dict() == before_task
    assert env.world.hash() == before_hash


def test_env_step_commits_reward_task_updates_after_evaluation():
    evaluations: list[int] = []

    class UpdatingTask(Task):
        name = "updating"
        preset = "flat"

        def __init__(self):
            self.counter = 0

        def reward_outcome(self, world) -> RewardOutcome:
            evaluations.append(self.counter)
            return RewardOutcome(
                total=0.25,
                components={"progress": 0.25},
                task_state_updates={"counter": self.counter + 1},
            )

    task = UpdatingTask()
    env = VoxelGymEnv(task=task, seed=67)
    env.reset(seed=67)

    _, reward, _, _, info = env.step(_action())

    assert reward == 0.25
    assert evaluations == [0]
    assert task.counter == 1
    assert info["reward_outcome"]["task_state_updates"] == {"counter": 1}

    restored = EnvSnapshot.from_bytes(env.snapshot().to_bytes())
    assert restored.last_reward.task_state_updates == {"counter": 1}


def test_reward_updates_are_json_safe_and_commit_atomically():
    with pytest.raises(TypeError, match="JSON-safe"):
        RewardOutcome(task_state_updates={"bad": {1, 2}})

    class PairTask(Task):
        def __init__(self):
            self.left = 0

    task = PairTask()
    outcome = RewardOutcome(task_state_updates={"left": 1, "missing": 2})
    with pytest.raises(ValueError, match="unknown task field"):
        task.commit_reward(outcome)
    assert task.left == 0
