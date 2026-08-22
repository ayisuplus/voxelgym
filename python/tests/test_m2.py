"""M2 tests: recording/replay verification, task reward semantics."""

import numpy as np
import pytest

import voxelgym_rs as rs
from voxelgym import ids
from voxelgym.experts import run_episode
from voxelgym.replay import verify
from voxelgym.tasks import NavigateToTarget, AchievementTask


def test_record_replay_roundtrip(tmp_path):
    ok, steps, final_hash, path = run_episode("navigate_to_target", seed=5, record_dir=str(tmp_path))
    assert ok, "expert should solve T1 on seed 5"
    assert path is not None
    assert verify(path, verbose=False)


def test_t1_reward_shaping_and_success():
    w = rs.PyWorld(3, "flat")
    task = NavigateToTarget()
    rng = np.random.default_rng(3)
    task.on_reset(w, rng)
    # walk straight at the target: reward should be positive while closing
    x0, _, z0 = w.agent_pos()
    tx, _, tz = task.target
    import math

    yaw = math.degrees(math.atan2(-(tx - x0), tz - z0))
    bucket = int(round(yaw / 15.0)) % 24
    for _ in range(5):
        w.step((1, 0, 0, bucket, 4, 0, 0, 0, 0, 0))
    r, done = task.step_reward(w)
    assert r > 0  # distance decreased
    # teleport onto the pillar: success + terminal
    w.teleport(tx, task.target[1], tz)
    r, done = task.step_reward(w)
    assert done and r >= 1.0 - 1e-6


def test_achievement_goal_and_shaping():
    w = rs.PyWorld(4, "flat")
    t = AchievementTask("craft_planks")
    t.on_reset(w, np.random.default_rng(0))
    # prerequisite (log) gives shaping, goal gives +1 and done
    w.give(ids.LOG, 1)
    r, done = t.step_reward(w)
    assert r == pytest.approx(0.1) and not done
    w.give(ids.PLANKS, 1)
    r, done = t.step_reward(w)
    assert r == pytest.approx(1.0) and done


def test_achievement_no_repeat_shaping():
    w = rs.PyWorld(4, "flat")
    t = AchievementTask("craft_table")
    t.on_reset(w, np.random.default_rng(0))
    w.give(ids.LOG, 1)
    r1, _ = t.step_reward(w)
    w.give(ids.LOG, 1)
    r2, _ = t.step_reward(w)  # already shaped
    assert r1 == pytest.approx(0.1)
    assert r2 == pytest.approx(0.0)


def test_craft_through_env_action():
    """craft flows through the actual env action channel."""
    from voxelgym.env import VoxelGymEnv, ACTION_KEYS

    env = VoxelGymEnv(preset="flat", seed=1)
    env.reset()
    env.world.give(ids.LOG, 2)
    a = {k: 0 for k in ACTION_KEYS}
    a["craft"] = 1
    env.step(a)
    assert env.world.count_item(ids.PLANKS) == 4
    assert env.world.count_item(ids.LOG) == 1
