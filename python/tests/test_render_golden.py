"""M4 render golden tests: seg/depth exactness via a python-side reference
DDA (same algorithm, same f64 op order) — seg must be exactly grass in the
ground region; depth bitwise-equal to the reference traversal."""

import math

import numpy as np
import pytest

import voxelgym_rs as rs
from voxelgym import ids
from voxelgym.env import VoxelGymEnv, ACTION_KEYS

SKY_SEG = 65535


def camera_rays(yaw_deg, pitch_deg):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    fwd = (-math.sin(yaw) * math.cos(pitch), -math.sin(pitch), math.cos(yaw) * math.cos(pitch))
    right = [fwd[2], 0.0, -fwd[0]]
    rl = math.hypot(right[0], right[2])
    right = [right[0] / rl, 0.0, right[2] / rl]
    up = (
        fwd[1] * right[2] - fwd[2] * right[1],
        fwd[2] * right[0] - fwd[0] * right[2],
        fwd[0] * right[1] - fwd[1] * right[0],
    )
    return fwd, right, up


def ref_dda(world, origin, d, max_dist):
    """Python replica of voxel_core::raycast::dda (strict blocks_ray policy)."""
    x, y, z = math.floor(origin[0]), math.floor(origin[1]), math.floor(origin[2])
    step_x = 1 if d[0] > 0 else -1
    step_y = 1 if d[1] > 0 else -1
    step_z = 1 if d[2] > 0 else -1
    inf = float("inf")
    tdx = abs(1.0 / d[0]) if d[0] != 0 else inf
    tdy = abs(1.0 / d[1]) if d[1] != 0 else inf
    tdz = abs(1.0 / d[2]) if d[2] != 0 else inf

    def bound(s, o, c):
        return (c + 1.0 - o) if s > 0 else (o - c)

    tmx = (bound(step_x, origin[0], x) / abs(d[0])) if d[0] != 0 else inf
    tmy = (bound(step_y, origin[1], y) / abs(d[1])) if d[1] != 0 else inf
    tmz = (bound(step_z, origin[2], z) / abs(d[2])) if d[2] != 0 else inf

    # note: sign conventions for the negative direction are folded in via abs
    tmx = ((x + 1.0 - origin[0]) / d[0] if step_x > 0 else (x - origin[0]) / d[0]) if d[0] != 0 else inf
    tmy = ((y + 1.0 - origin[1]) / d[1] if step_y > 0 else (y - origin[1]) / d[1]) if d[1] != 0 else inf
    tmz = ((z + 1.0 - origin[2]) / d[2] if step_z > 0 else (z - origin[2]) / d[2]) if d[2] != 0 else inf

    def blocks_ray(c):
        return (c & 0xFFF) != 0

    cell = world.get_block(x, y, z)
    if blocks_ray(cell):
        return (x, y, z, cell, 0.0)
    t = 0.0
    while True:
        if tmx <= tmy and tmx <= tmz:
            t = tmx
            tmx += tdx
            x += step_x
        elif tmy <= tmz:
            t = tmy
            tmy += tdy
            y += step_y
        else:
            t = tmz
            tmz += tdz
            z += step_z
        if t > max_dist:
            return None
        cell = world.get_block(x, y, z)
        if blocks_ray(cell):
            return (x, y, z, cell, t)


def make_world_with_pose():
    w = rs.PyWorld(7, "flat")
    # teleport onto the flat surface and set exact pose via one action:
    # yaw bucket 0 = 0 deg (+z), pitch bucket 7 = +45 deg down
    w.teleport(8.5, 5.0, 8.5)
    a = (0, 0, 0, 0, 7, 0, 0, 0, 0, 0)
    w.step(a)
    return w


def test_render_golden_flat_seg_exact():
    w = make_world_with_pose()
    rgb, depth, seg = w.render()
    # 45 deg pitch, 90 deg FOV: rows 64..127 all hit ground = grass
    ground = seg[64:, :]
    assert (ground == ids.GRASS_BLOCK).all(), "ground region must be all grass"


def test_render_golden_depth_bitwise():
    w = make_world_with_pose()
    rgb, depth, seg = w.render()
    eye = (8.5, 5.0 + 1.62, 8.5)
    fwd, right, up = camera_rays(0.0, 45.0)
    half = math.tan(math.radians(45.0))
    for (px, py) in [(64, 100), (10, 80), (120, 70), (64, 120), (0, 64), (127, 127), (63, 90)]:
        sy = 1.0 - 2.0 * (py + 0.5) / 128.0
        sx = 2.0 * (px + 0.5) / 128.0 - 1.0
        d = [
            fwd[0] + right[0] * sx * half + up[0] * sy * half,
            fwd[1] + right[1] * sx * half + up[1] * sy * half,
            fwd[2] + right[2] * sx * half + up[2] * sy * half,
        ]
        dl = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        d = [d[0] / dl, d[1] / dl, d[2] / dl]
        hit = ref_dda(w, eye, d, 96.0)
        assert hit is not None
        expected = np.float32(hit[4] * dl)
        got = np.float32(depth[py, px])
        assert got.view(np.uint32) == expected.view(np.uint32), f"bitwise depth at ({px},{py}): {got} vs {expected}"
        assert seg[py, px] == ids.GRASS_BLOCK


def test_render_deterministic_repeat():
    w = make_world_with_pose()
    _, d1, s1 = w.render()
    _, d2, s2 = w.render()
    np.testing.assert_array_equal(d1, d2)
    np.testing.assert_array_equal(s1, s2)


def test_env_render_channels():
    env = VoxelGymEnv(preset="flat", seed=2, render=True)
    obs, _ = env.reset()
    assert obs["rgb"].shape == (128, 128, 3) and obs["rgb"].dtype == np.uint8
    assert obs["depth"].dtype == np.float16
    assert obs["seg"].dtype == np.uint16
    assert env.observation_space.contains(obs)
    # render every 5 ticks reuses frames between renders
    env2 = VoxelGymEnv(preset="flat", seed=2, render=5)
    env2.reset()
    a = {k: 0 for k in ACTION_KEYS}
    a["move"] = 1
    env2.step(a)
    _, d_a, _ = env2._frames()
    env2.step(a)  # tick 2: no render (5-interval), same frames
    _, d_b, _ = env2._frames()
    np.testing.assert_array_equal(d_a, d_b)
