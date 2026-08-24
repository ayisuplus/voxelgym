"""M3.5 physics truth tables: fire, pressure plate, TNT, falling impact,
and Physics runtime ablation."""

import pytest

import voxelgym_rs as rs
from conftest import IDLE, cid, run, state
from voxelgym import ids


def test_fire_burnout_and_spread():
    w = rs.PyWorld(1, "void", [
        (5, 5, 5, 7, 5, 5, ids.PLANKS),
        (6, 5, 5, 6, 5, 5, ids.FIRE | (5 << 12)),
    ])
    # eventually: the plank neighbors are consumed, then all fire burns out
    run(w, 300)
    assert cid(w.get_block(6, 5, 5)) == ids.AIR
    assert cid(w.get_block(5, 5, 5)) != ids.FIRE
    assert cid(w.get_block(7, 5, 5)) != ids.FIRE


def test_fire_damage():
    w = rs.PyWorld(1, "void", [(5, 5, 5, 5, 5, 5, ids.STONE)])
    w.set_block(5, 6, 5, ids.FIRE | (5 << 12))
    w.teleport(5.5, 6.0, 5.5)  # feet inside the fire cell
    hp0 = w.hp()
    run(w, 10)
    assert w.hp() == hp0 - 1


def test_stone_never_catches_fire():
    w = rs.PyWorld(1, "void", [
        (5, 5, 5, 5, 5, 5, ids.STONE),
        (6, 5, 5, 6, 5, 5, ids.FIRE | (5 << 12)),
        (5, 5, 6, 5, 5, 6, ids.STONE),
    ])
    run(w, 300)
    assert cid(w.get_block(5, 5, 5)) == ids.STONE
    assert cid(w.get_block(5, 5, 6)) == ids.STONE


def test_plate_opens_door():
    w = rs.PyWorld(1, "flat", [
        (9, 5, 0, 9, 5, 0, ids.PRESSURE_PLATE),
        (9, 5, 1, 9, 5, 1, ids.WIRE),
        (10, 5, 1, 10, 5, 1, ids.WIRE),
        (10, 5, 0, 10, 5, 0, ids.DOOR),
    ])
    run(w, 3)
    assert state(w.get_block(10, 5, 0)) == 0, "door closed with nobody on the plate"
    w.teleport(9.5, 5.0, 0.5)  # feet cell == plate cell
    run(w, 3)
    assert state(w.get_block(9, 5, 0)) == 1, "plate engaged"
    assert state(w.get_block(10, 5, 0)) == 1, "door open"
    w.teleport(5.5, 5.0, 5.5)  # step off
    run(w, 3)
    assert state(w.get_block(10, 5, 0)) == 0, "door closes again"


def test_tnt_blast_and_chain():
    w = rs.PyWorld(1, "flat", [
        (2, 6, 2, 2, 6, 2, ids.LEVER | (1 << 12)),
        (3, 6, 2, 3, 6, 2, ids.WIRE),
        (4, 6, 2, 4, 6, 2, ids.TNT),
        (5, 6, 2, 5, 6, 2, ids.DIRT),
        (6, 6, 2, 6, 6, 2, ids.TNT),   # chained
        (9, 6, 2, 9, 6, 2, ids.STONE), # outside r=2 of both blasts
    ])
    run(w, 20)
    assert cid(w.get_block(4, 6, 2)) == ids.AIR
    assert cid(w.get_block(5, 6, 2)) == ids.AIR
    assert cid(w.get_block(6, 6, 2)) == ids.AIR, "chained tnt detonated"
    assert cid(w.get_block(9, 6, 2)) == ids.STONE


def test_falling_block_impact_damage():
    w = rs.PyWorld(1, "void", [(5, 5, 5, 5, 5, 5, ids.STONE)])
    w.teleport(5.5, 6.0, 5.5)  # standing on the platform
    w.set_block(5, 16, 5, ids.SAND)  # 10 above the platform
    hp0 = w.hp()
    run(w, 60)
    assert w.hp() < hp0, "falling sand dealt impact damage"


def test_physics_force_mass_model():
    """L0 force model: accel = F/m. Double mass -> slower acceleration."""
    w1 = rs.PyWorld(1, "flat")
    w2 = rs.PyWorld(1, "flat", None, {"agent_mass": 2.0})
    w1.step(IDLE)
    w2.step(IDLE)
    z1 = w1.agent_pos()[2]
    z2 = w2.agent_pos()[2]
    for _ in range(20):
        w1.step((1, 0, 0, 0, 4, 0, 0, 0, 0, 0))
        w2.step((1, 0, 0, 0, 4, 0, 0, 0, 0, 0))
    d1 = w1.agent_pos()[2] - z1
    d2 = w2.agent_pos()[2] - z2
    assert d2 < d1 * 0.95, f"mass 2 accelerates slower: {d1:.3f} vs {d2:.3f}"


def test_physics_override_gravity():
    # halved gravity -> higher jump apex (fall damage is distance-based and
    # would show nothing)
    def apex(w):
        w.teleport(8.5, 5.0, 8.5)
        w.step(IDLE)  # settle on ground
        top = 5.0
        for _ in range(60):
            w.step((0, 1, 0, 0, 4, 0, 0, 0, 0, 0))
            top = max(top, w.agent_pos()[1])
        return top - 5.0

    w1 = rs.PyWorld(1, "flat")
    w2 = rs.PyWorld(1, "flat", None, {"gravity": 0.04})
    a1, a2 = apex(w1), apex(w2)
    assert 1.2 <= a1 <= 1.3
    assert a2 > 2.0, f"halved gravity lifts apex: {a2}"


def test_physics_override_water_spread():
    w = rs.PyWorld(1, "void", [(0, 5, 0, 19, 5, 19, ids.STONE)], {"water_spread": 3})
    w.set_block(10, 6, 10, ids.WATER)
    run(w, 200)
    assert cid(w.get_block(13, 6, 10)) == ids.WATER
    assert cid(w.get_block(14, 6, 10)) == ids.AIR, "spread capped at 3 by override"


def test_physics_snapshot_roundtrip():
    w = rs.PyWorld(1, "flat", None, {"gravity": 0.04, "lava_damage": 2})
    run(w, 10)
    snap = w.snapshot()
    w2 = rs.PyWorld(0, "void")
    w2.restore(snap)
    assert w2.get_physics("gravity") == pytest.approx(0.04)
    assert w2.get_physics("lava_damage") == pytest.approx(2.0)
    assert w2.hash() == w.hash()
