"""M1 contract tests for the gymnasium layer."""

import numpy as np
import pytest

import voxelgym_rs as rs
from voxelgym import VoxelGymEnv, ACTION_KEYS


def test_action_space_matches_contract():
    env = VoxelGymEnv(preset="flat")
    sp = env.action_space
    assert sp["move"].n == 5
    assert sp["jump"].n == 2
    assert sp["sneak"].n == 2
    assert sp["yaw"].n == 24
    assert sp["pitch"].n == 9
    assert sp["mine"].n == 2
    assert sp["place"].n == 2
    assert sp["use"].n == 2
    assert sp["hotbar"].n == 9
    assert sp["craft"].n == 8


def test_obs_space_and_step_shapes():
    env = VoxelGymEnv(preset="flat", seed=3)
    obs, info = env.reset()
    assert obs["voxels"].shape == (21, 11, 21) and obs["voxels"].dtype == np.uint16
    assert obs["inventory"].shape == (36, 2) and obs["inventory"].dtype == np.uint16
    assert obs["pose"].shape == (6,) and obs["pose"].dtype == np.float32
    assert obs["raycast"].shape == (2,) and obs["raycast"].dtype == np.uint16
    assert "rgb" not in obs  # render off
    a = {k: 0 for k in ACTION_KEYS}
    a["move"] = 1
    obs, r, term, trunc, info = env.step(a)
    assert env.observation_space.contains(obs)
    assert r == 0.0 and not term and not trunc


def test_reset_deterministic():
    e1 = VoxelGymEnv(preset="default", seed=11)
    e2 = VoxelGymEnv(preset="default", seed=11)
    o1, _ = e1.reset()
    o2, _ = e2.reset()
    np.testing.assert_array_equal(o1["pose"], o2["pose"])
    a = {k: 0 for k in ACTION_KEYS}
    a.update(move=1, yaw=3, mine=1)
    for _ in range(200):
        e1.step(a)
        e2.step(a)
    assert e1.world.hash() == e2.world.hash()


def test_yaw_action_absolute():
    env = VoxelGymEnv(preset="flat", seed=5)
    env.reset()
    a = {k: 0 for k in ACTION_KEYS}
    a["yaw"] = 6  # 90 deg
    env.step(a)
    assert env.world.obs_pose()[3] == pytest.approx(90.0)
    a["yaw"] = 0
    env.step(a)
    assert env.world.obs_pose()[3] == pytest.approx(0.0)


def test_snapshot_restore_roundtrip():
    w = rs.PyWorld(77, "default")
    for i in range(300):
        w.step((1, i % 2, 0, i % 24, i % 9, 1 if i % 7 == 0 else 0, 0, 0, 0, 0))
    snap = w.snapshot()
    h = w.hash()
    for _ in range(50):
        w.step((0, 0, 0, 0, 4, 0, 0, 0, 0, 0))
    assert w.hash() != h
    w.restore(snap)
    assert w.hash() == h


def test_smoke_walk_jump_mine():
    """M1 acceptance smoke: walk 100 ticks, jump 10 times, mine 1 dirt."""
    env = VoxelGymEnv(preset="flat", seed=1)
    env.reset()
    w = env.world
    x0, y0, z0 = w.agent_pos()

    a = {k: 0 for k in ACTION_KEYS}
    a.update(move=1, yaw=0)  # walk +z
    for _ in range(100):
        env.step(a)
    x1, y1, z1 = w.agent_pos()
    assert z1 - z0 > 10.0, f"walked only {z1 - z0} cells"

    jumps = 0
    a.update(move=0, jump=1)
    prev_ground = True
    for _ in range(200):
        env.step(a)
        on_ground = w.obs_pose()[5] == 1.0
        if prev_ground and not on_ground:
            jumps += 1
        prev_ground = on_ground
        if jumps >= 10:
            break
    assert jumps >= 10

    # mine one dirt: aim down-forward at the surface and hold mine
    ax, az = int(w.agent_pos()[0]), int(w.agent_pos()[2])
    a.update(jump=0, yaw=0, pitch=8, mine=1)  # +60 deg down toward +z
    target = None
    for _ in range(60):
        env.step(a)
        # find first air cell in the surface layer ahead
        for dz in range(1, 4):
            if w.get_block(ax, 4, az + dz) == 0:
                target = (ax, 4, az + dz)
                break
        if target:
            break
    assert target is not None, "no block mined in 60 ticks"
    assert w.get_block(*target) == 0
