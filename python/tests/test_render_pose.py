import numpy as np
import pytest

import voxelgym_rs as rs
from voxelgym import ids


SKY_SEG = 0xFFFF
ORIGIN = (8.5, 6.62, 8.5)


def test_render_pose_non_square_uses_height_width_row_order():
    world = rs.PyWorld(7, "flat")

    rgb, depth, seg, normals = world.render_pose(
        ORIGIN, 0.0, 0.0, width=3, height=2
    )

    assert rgb.shape == (2, 3, 3)
    assert depth.shape == (2, 3)
    assert seg.shape == (2, 3)
    assert normals.shape == (2, 3, 3)
    assert (seg[0] == SKY_SEG).all()
    assert (seg[1] == ids.GRASS_BLOCK).all()
    assert (normals[0] == 0.0).all()
    assert (normals[1] == np.array([0.0, 1.0, 0.0], dtype=np.float32)).all()


@pytest.mark.parametrize("width,height", [(0, 2), (2, 0)])
def test_render_pose_rejects_zero_dimensions(width, height):
    world = rs.PyWorld(7, "void")

    with pytest.raises(ValueError, match="positive"):
        world.render_pose(ORIGIN, 0.0, 0.0, width=width, height=height)


def test_render_pose_rejects_dimension_overflow():
    world = rs.PyWorld(7, "void")
    usize_max = (1 << (np.dtype(np.uintp).itemsize * 8)) - 1

    with pytest.raises(ValueError, match="too large"):
        world.render_pose(ORIGIN, 0.0, 0.0, width=usize_max, height=2)
