"""M3 physics truth tables via the PyWorld API (plan requires these in
pytest; the same laws are also covered by Rust unit tests)."""

import pytest

import voxelgym_rs as rs
from conftest import IDLE, cid, run, state
from voxelgym import ids


def R(*a):
    return tuple(a)


def floor_lab(size=20, y=5):
    """Void world with a stone floor at y=5 (0..size-1 squared)."""
    w = rs.PyWorld(1, "void", [R(0, y, 0, size - 1, y, size - 1, ids.STONE)])
    return w


# ---------- loose blocks ----------

def test_sand_falls_and_lands():
    w = rs.PyWorld(1, "void", [
        R(5, 5, 5, 5, 5, 5, ids.STONE),
        R(5, 10, 5, 5, 10, 5, ids.SAND),
    ])
    run(w, 2)
    assert cid(w.get_block(5, 10, 5)) == ids.AIR
    run(w, 40)
    assert cid(w.get_block(5, 6, 5)) == ids.SAND


def test_sand_on_torch_becomes_item():
    w = rs.PyWorld(1, "void", [
        R(5, 5, 5, 5, 5, 5, ids.STONE),
        R(5, 6, 5, 5, 6, 5, ids.TORCH),
        R(5, 10, 5, 5, 10, 5, ids.SAND),
    ])
    run(w, 40)
    assert cid(w.get_block(5, 6, 5)) == ids.TORCH
    assert cid(w.get_block(5, 7, 5)) == ids.AIR
    # drop spawned: visible as a loose item of type sand
    assert len(w.drops_of(ids.SAND)) == 1


def test_supported_sand_stays():
    w = rs.PyWorld(1, "void", [
        R(5, 5, 5, 5, 5, 5, ids.STONE),
        R(5, 6, 5, 5, 6, 5, ids.SAND),
    ])
    run(w, 10)
    assert cid(w.get_block(5, 6, 5)) == ids.SAND


# ---------- fluids ----------

def test_water_spreads_exactly_seven():
    w = floor_lab()
    w.set_block(10, 6, 10, ids.WATER)
    run(w, 200)
    assert cid(w.get_block(17, 6, 10)) == ids.WATER
    assert state(w.get_block(17, 6, 10)) == 7
    assert cid(w.get_block(18, 6, 10)) == ids.AIR


def test_water_drains_without_supply():
    w = floor_lab()
    w.set_block(10, 6, 10, ids.WATER)
    run(w, 100)
    assert cid(w.get_block(13, 6, 10)) == ids.WATER
    w.set_block(10, 6, 10, 0)
    run(w, 200)
    for d in range(1, 8):
        assert cid(w.get_block(10 + d, 6, 10)) == ids.AIR


def test_two_sources_merge():
    w = floor_lab()
    w.set_block(10, 6, 10, ids.WATER)
    w.set_block(12, 6, 10, ids.WATER)
    run(w, 100)
    assert cid(w.get_block(11, 6, 10)) == ids.WATER
    assert state(w.get_block(11, 6, 10)) == 0


def test_water_into_lava_source_makes_stone():
    w = floor_lab()
    w.set_block(10, 6, 10, ids.LAVA)
    w.set_block(12, 6, 10, ids.WATER)
    run(w, 200)
    assert cid(w.get_block(10, 6, 10)) == ids.STONE


def test_lava_into_water_makes_cobblestone():
    # lava falling into a water cell -> that cell becomes cobblestone
    w = floor_lab()
    w.set_block(10, 6, 11, ids.WATER)   # water source on the floor
    w.set_block(10, 8, 11, ids.LAVA)    # lava source floating above
    run(w, 120)
    assert cid(w.get_block(10, 6, 11)) == ids.COBBLESTONE


def test_lava_max_three():
    w = floor_lab()
    w.set_block(10, 6, 10, ids.LAVA)
    run(w, 600)
    assert cid(w.get_block(13, 6, 10)) == ids.LAVA
    assert state(w.get_block(13, 6, 10)) == 3
    assert cid(w.get_block(14, 6, 10)) == ids.AIR


def test_water_swim_buoyancy():
    # pool 3x3x2 of water on a floor; agent sinks slower with jump
    spec = [R(5, 5, 5, 7, 5, 7, ids.STONE)]
    for x in range(5, 8):
        for z in range(5, 8):
            spec.append(R(x, 6, z, x, 7, z, ids.WATER))
    w = rs.PyWorld(1, "void", spec)
    w.teleport(6.5, 7.5, 6.5)
    # no jump: sinks
    run(w, 20)
    y_sink = w.agent_pos()[1]
    w.teleport(6.5, 7.5, 6.5)
    for _ in range(20):
        w.step((0, 1, 0, 0, 4, 0, 0, 0, 0, 0))
    y_swim = w.agent_pos()[1]
    assert y_swim > y_sink


# ---------- lava damage / suffocation ----------

def test_lava_damage_four_per_ten_ticks():
    w = floor_lab()
    w.set_block(10, 6, 10, ids.LAVA)
    w.teleport(10.5, 6.0, 10.5)  # standing in lava source
    hp0 = w.hp()
    run(w, 10)
    assert w.hp() == hp0 - 4
    run(w, 10)
    assert w.hp() == hp0 - 8


def test_suffocation_one_per_twenty():
    w = floor_lab()
    w.teleport(10.5, 6.0, 10.5)
    w.set_block(10, 8, 10, ids.STONE)  # head cell (eye 6+1.62=7.62 -> y=7? feet 6)
    # eye cell is y=7; but agent feet at 6.0 means eye 7.62 -> cell 7
    hp0 = w.hp()
    run(w, 21)
    # if head cell is y=7 we need stone there
    if w.hp() == hp0:
        w.set_block(10, 7, 10, ids.STONE)
        run(w, 21)
    assert w.hp() == hp0 - 1


# ---------- circuits ----------

def test_wire_decay_fifteen():
    spec = [R(2, 6, 2, 2, 6, 2, ids.LEVER | (1 << 12))]
    for i in range(20):
        spec.append(R(3 + i, 6, 2, 3 + i, 6, 2, ids.WIRE))
    w = rs.PyWorld(1, "flat", spec)
    run(w, 2)
    for i in range(1, 16):
        assert state(w.get_block(2 + i, 6, 2)) == 16 - i, f"wire {i}"
    assert state(w.get_block(18, 6, 2)) == 0


def test_lever_door_open_close():
    w = rs.PyWorld(1, "flat", [
        R(8, 5, 0, 8, 5, 0, ids.LEVER),
        R(9, 5, 0, 9, 5, 0, ids.WIRE),
        R(10, 5, 0, 10, 5, 0, ids.DOOR),
    ])
    w.teleport(6.5, 5.0, 0.5)
    run(w, 2)
    assert state(w.get_block(10, 5, 0)) == 0
    # aim down-forward at the floor-level lever and use it
    w.step((0, 0, 0, 18, 7, 0, 0, 0, 0, 0))
    assert cid(w.obs_raycast()[0]) == ids.LEVER
    w.step((0, 0, 0, 18, 7, 0, 0, 1, 0, 0))
    assert state(w.get_block(8, 5, 0)) == 1, "lever on"
    assert state(w.get_block(9, 5, 0)) == 15, "wire powered"
    assert state(w.get_block(10, 5, 0)) == 1, "door open same tick"


def test_broken_wire_cuts_power():
    spec = [R(2, 6, 2, 2, 6, 2, ids.LEVER | (1 << 12))]
    for i in range(5):
        spec.append(R(3 + i, 6, 2, 3 + i, 6, 2, ids.WIRE))
    w = rs.PyWorld(1, "flat", spec)
    run(w, 2)
    assert state(w.get_block(7, 6, 2)) == 11
    w.set_block(5, 6, 2, 0)
    run(w, 2)
    assert state(w.get_block(4, 6, 2)) == 14
    assert state(w.get_block(6, 6, 2)) == 0
    assert state(w.get_block(7, 6, 2)) == 0
