"""LiDAR channel: plumbing, determinism, golden range through the env."""

import numpy as np

from voxelgym.env import VoxelGymEnv
from voxelgym.tasks import make_task

CFG = {
    "channels": 16,
    "azimuth": 64,
    "min_elev": -15.0,
    "max_elev": 15.0,
    "max_range": 64.0,
}


def test_lidar_obs_shapes_and_space():
    env = VoxelGymEnv(task=make_task("navigate_to_target"), seed=0, lidar=CFG)
    obs, _ = env.reset(seed=0)
    assert env.observation_space["lidar_range"].shape == (16, 64)
    assert obs["lidar_range"].shape == (16, 64)
    assert obs["lidar_range"].dtype == np.float32
    assert obs["lidar_intensity"].dtype == np.float32
    assert obs["lidar_seg"].dtype == np.uint16
    assert ((obs["lidar_intensity"] >= 0) & (obs["lidar_intensity"] <= 1)).all()
    assert env.observation_space.contains(
        {k: obs[k] for k in ("lidar_range", "lidar_intensity", "lidar_seg")}
        | {"voxels": obs["voxels"], "inventory": obs["inventory"],
           "pose": obs["pose"], "raycast": obs["raycast"]}
    )


def test_lidar_deterministic_across_envs():
    acts = [
        {"move": 1, "jump": 0, "sneak": 0, "yaw": 18, "pitch": 4,
         "mine": 0, "place": 0, "use": 0, "hotbar": 0, "craft": 0},
        {"move": 1, "jump": 1, "sneak": 0, "yaw": 12, "pitch": 5,
         "mine": 0, "place": 0, "use": 0, "hotbar": 0, "craft": 0},
    ]
    scans = []
    for _ in range(2):
        env = VoxelGymEnv(task=make_task("navigate_to_target"), seed=3, lidar=CFG)
        env.reset(seed=3)
        for a in acts * 5:
            obs, *_ = env.step(a)
        scans.append((obs["lidar_range"].copy(), obs["lidar_seg"].copy()))
    assert np.array_equal(scans[0][0], scans[1][0]), "range images identical"
    assert np.array_equal(scans[0][1], scans[1][1]), "seg images identical"


def test_lidar_noise_reproducible_per_tick():
    noisy = dict(CFG, noise_sigma=0.05, dropout_p=0.05)
    env = VoxelGymEnv(task=make_task("navigate_to_target"), seed=1, lidar=noisy)
    env.reset(seed=1)
    idle = {"move": 0, "jump": 0, "sneak": 0, "yaw": 0, "pitch": 4,
            "mine": 0, "place": 0, "use": 0, "hotbar": 0, "craft": 0}
    obs1, *_ = env.step(idle)
    # same tick -> cached scan is the same object content
    obs2 = env._obs()
    assert np.array_equal(obs1["lidar_range"], obs2["lidar_range"])
    # a second env at the same tick sees the identical noise draw
    env2 = VoxelGymEnv(task=make_task("navigate_to_target"), seed=1, lidar=noisy)
    env2.reset(seed=1)
    obs3, *_ = env2.step(idle)
    assert np.array_equal(obs1["lidar_range"], obs3["lidar_range"])


def test_lidar_golden_wall_through_binding():
    """Explicit-pose scan (fixed emitter): a stone wall at x=10 read from
    (5.5, 6.5, 0.5) facing +x gives the analytic DDA distance."""
    import voxelgym_rs as rs

    w = rs.PyWorld(1, "flat", None)
    for y in range(5, 9):
        for z in range(-2, 3):
            w.set_block(10, y, z, 2)  # STONE
    rng, inten, seg = w.lidar_scan(
        channels=4, azimuth_steps=8, min_elev_deg=-5.0, max_elev_deg=40.0,
        max_range=64.0, origin=(5.5, 6.5, 0.5), yaw_deg=270.0,
    )
    a_n = 8
    exact = 4.5 / np.cos(np.deg2rad(-5.0))
    assert abs(rng[0, 0] - exact) < 1e-4
    assert seg[0, 0] == 2
    assert rng[3, 0] == 0.0 and seg[3, 0] == 0xFFFF  # high beam: sky
    assert 0.0 < inten[0, 0] <= 1.0
    _ = a_n


def test_lidar_ground_returns_on_flat():
    """Down-tilted beams over flat terrain all return the grass plane."""
    env = VoxelGymEnv(preset="flat", lidar=dict(CFG, min_elev=-15.0, max_elev=-1.0))
    obs, _ = env.reset(seed=0)
    hit = obs["lidar_range"] > 0
    frac = hit.mean()
    assert frac > 0.9, f"down-looking beams hit the ground: {frac:.2f}"
    # eye 1.62 above the plane: r = 1.62/sin(|elev|) — grazing beams
    # legitimately run long; bound by max_range instead
    r = obs["lidar_range"][hit]
    assert (r > 0.5).all() and (r <= 64.0).all()
    # spot-check the geometry on the steepest channel (-15 deg)
    assert abs(r.min() - 1.62 / np.sin(np.deg2rad(15))) < 0.2
