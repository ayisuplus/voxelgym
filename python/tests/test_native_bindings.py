import numpy as np
import pytest

import voxelgym_rs as rs


IDLE = (0, 0, 0, 0, 4, 0, 0, 0, 0, 0)


def test_world_constructor_and_physics_errors_are_value_errors():
    with pytest.raises(ValueError, match="unknown preset"):
        rs.PyWorld(1, "missing")
    with pytest.raises(ValueError, match="scale"):
        rs.PyWorld(1, "void", scale=0.5)
    with pytest.raises(ValueError, match="scale"):
        rs.PyWorld(1, "void", scale=1.1)
    with pytest.raises(ValueError, match="unknown physics field"):
        rs.PyWorld(1, "void", physics={"missing": 1.0})

    world = rs.PyWorld(1, "void", physics={"gravity": 0.125})
    assert world.get_physics("gravity") == pytest.approx(0.125)
    with pytest.raises(ValueError, match="unknown physics field"):
        world.get_physics("missing")


def test_native_byte_views_match_numpy_and_palette_contract():
    world = rs.PyWorld(2, "flat")

    assert world.obs_voxels_bytes() == world.obs_voxels().tobytes(order="C")
    assert world.obs_inventory_bytes() == world.obs_inventory().tobytes(order="C")
    palette = world.palette()
    assert palette.dtype == np.uint8
    assert palette.ndim == 2 and palette.shape[1] == 3
    assert tuple(palette[rs.block_id("stone")]) == (125, 125, 125)


def test_native_queries_cover_hit_miss_and_state_consumption():
    world = rs.PyWorld(3, "void")
    stone = rs.block_id("stone")

    assert world.crosshair() is None
    assert world.cast_ray((0.5, 5.5, 0.5), (1.0, 0.0, 0.0), 4.0) == -1.0
    assert world.drops_of(stone) == []
    assert world.furnace_state(0, 0, 0) == (0, False, 0)
    assert world.take_swap() == 0

    world.set_block(2, 5, 0, stone)
    assert world.cast_ray((0.5, 5.5, 0.5), (1.0, 0.0, 0.0), 4.0) == pytest.approx(1.5)
    world.teleport(0.5, 4.0, 0.5)
    world.step((0, 0, 0, 18, 4, 0, 0, 0, 0, 0))
    hit = world.crosshair()
    assert hit is not None and hit[0] == (2, 5, 0) and hit[1] == stone


def test_restore_rejects_invalid_bytes_without_replacing_world():
    world = rs.PyWorld(4, "flat")
    before = world.hash()

    with pytest.raises(ValueError):
        world.restore(b"not a snapshot")

    assert world.hash() == before


def test_module_id_lookup_errors():
    assert rs.block_id("stone") > 0
    assert rs.item_id("coal") > 0
    with pytest.raises(ValueError, match="unknown block"):
        rs.block_id("missing")
    with pytest.raises(ValueError, match="unknown item"):
        rs.item_id("missing")
