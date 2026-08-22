"""Property tests: the agent must NEVER interpenetrate solid geometry.

Triggered by a demo observation ("穿模"). If these pass, clipping is a
presentation artifact (camera/overlay), not a sim bug.
"""

import numpy as np
import pytest

from voxelgym import ids

NON_SOLID = {
    ids.AIR, ids.WATER, ids.LAVA, ids.TORCH, ids.WIRE, ids.LEVER,
    ids.PRESSURE_PLATE, ids.FIRE, ids.REDSTONE_TORCH, ids.REPEATER,
}


def aabb_intersects_solid(w):
    x, y, z = w.agent_pos()
    # AABB 0.6 x 1.8 x 0.6 centered on x/z, feet at y
    for cx in range(int(np.floor(x - 0.3)), int(np.floor(x + 0.3)) + 1):
        for cy in range(int(np.floor(y)), int(np.floor(y + 1.8)) + 1):
            for cz in range(int(np.floor(z - 0.3)), int(np.floor(z + 0.3)) + 1):
                cell = w.get_block(cx, cy, cz)
                bid = cell & 0xFFF
                if bid in NON_SOLID:
                    continue
                if bid == ids.DOOR and (cell >> 12) & 1:
                    continue  # open door is passable
                # precise overlap test (cell spans [c, c+1))
                if (cx + 1.0 > x - 0.3 and cx < x + 0.3
                        and cy + 1.0 > y and cy < y + 1.8
                        and cz + 1.0 > z - 0.3 and cz < z + 0.3):
                    return (cx, cy, cz, bid)
    return None


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_walk_never_clips(seed):
    """3000 ticks of random walk/jump/look on default terrain: at no tick
    may the agent AABB overlap a solid cell."""
    import voxelgym_rs as rs

    w = rs.PyWorld(seed, "default", None)
    rng = np.random.default_rng(seed)
    for t in range(3000):
        a = (
            int(rng.integers(0, 5)), int(rng.integers(0, 2)), 0,
            int(rng.integers(0, 24)), int(rng.integers(0, 9)),
            0, 0, 0, 0, 0,
        )
        w.step(a)
        if w.dead():
            return  # fall/lava deaths reset the scenario; nothing to prove
        hit = aabb_intersects_solid(w)
        assert hit is None, f"t={t} agent inside solid {hit}, pos={w.agent_pos()}"


def test_terminal_fall_onto_thin_platform_no_tunnel():
    """vy reaches -3.92 cells/tick > 1 cell: a 1-thick floor must still stop
    the fall (clip_axis sweeps, not destination-checks)."""
    import voxelgym_rs as rs

    spec = [(4, 30, 4, 6, 30, 6, ids.STONE)]  # 3x3 platform, 1 cell thick
    w = rs.PyWorld(1, "void", spec)
    w.teleport(5.5, 60.0, 5.5)
    idle = (0, 0, 0, 0, 4, 0, 0, 0, 0, 0)
    for _ in range(400):
        w.step(idle)
        if w.obs_pose()[5] >= 1.0:  # on_ground
            break
    y = w.agent_pos()[1]
    assert 30.9 <= y <= 31.1, f"must land ON TOP of the platform, y={y}"


def test_diagonal_wall_corner_no_clip():
    """Driving diagonally into an inside corner must not wedge the agent
    into either wall (per-axis resolution corner case)."""
    import voxelgym_rs as rs

    spec = []
    for y in range(5, 8):
        spec.append((6, y, 0, 6, y, 10, ids.STONE))   # wall +x
        spec.append((0, y, 6, 10, y, 6, ids.STONE))   # wall +z
    w = rs.PyWorld(1, "flat", spec)
    w.teleport(3.5, 5.0, 3.5)
    for t in range(300):
        w.step((1, 1, 0, 0, 4, 0, 0, 0, 0, 0))  # forward+jump, yaw 0 = +z... aim at the corner
        w.step((1, 1, 0, 18, 4, 0, 0, 0, 0, 0))  # +x
        hit = aabb_intersects_solid(w)
        assert hit is None, f"t={t} corner clip {hit}, pos={w.agent_pos()}"
