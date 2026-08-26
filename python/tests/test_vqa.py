"""VQA family truth tests: every family's answers are proven correct on
scenario-built worlds. The core check: answers FLIP when only the relevant
world cell flips (labels come from state, not constants).

gen determinism (byte-identical manifests) is covered separately in
test_vqa_gen.py.
"""

import numpy as np
import pytest

import voxelgym_rs as rs
from conftest import IDLE, run
from voxelgym import ids
from voxelgym.vqa import FAMILIES, FAMILY_BY_NAME, Ctx
from voxelgym.vqa.families import PALETTE_BLOCKS


def R(*a):
    return tuple(a)


def nor_rig():
    """NOR(A,B) rig replicated from crates/voxel-core/src/circuit.rs
    `nor_gate_from_torch_and_wire_join` (y shifted 6->5 for the Python flat
    preset). Levers start OFF."""
    spec = [
        R(2, 5, 2, 2, 5, 2, ids.LEVER), R(2, 5, 4, 2, 5, 4, ids.LEVER),
        R(3, 5, 2, 3, 5, 2, ids.WIRE), R(3, 5, 4, 3, 5, 4, ids.WIRE),
        R(4, 5, 2, 4, 5, 2, ids.WIRE), R(4, 5, 3, 4, 5, 3, ids.WIRE), R(4, 5, 4, 4, 5, 4, ids.WIRE),
        R(4, 6, 3, 4, 6, 3, ids.REDSTONE_TORCH | (1 << 12)),
        R(5, 6, 3, 5, 6, 3, ids.WIRE),
        R(6, 6, 3, 6, 6, 3, ids.LAMP),
    ]
    w = rs.PyWorld(1, "flat", spec)
    run(w, 8)
    ctx = Ctx(task="logic_probe", lamp=(6, 6, 3), lever_a=(2, 5, 2), lever_b=(2, 5, 4))
    return w, ctx


def test_registry_has_twelve_bilingual_families():
    assert len(FAMILIES) == 12
    for f in FAMILIES:
        assert len(f.en) >= 4 and len(f.zh) >= 4, f.name
        assert f.classes and f.tasks, f.name
        assert f.needs <= {"rgb", "voxels", "prior"}, f.name
    wd = FAMILY_BY_NAME["wall_dominant"]
    wr = FAMILY_BY_NAME["wall_region"]
    assert wd.classes == PALETTE_BLOCKS
    for f in (wd, wr):
        assert f.tasks == ("pixel_gallery",), f.name
        assert f.needs == {"rgb"}, f.name


def test_emit_is_deterministic_per_seed():
    w, ctx = nor_rig()
    fam = FAMILY_BY_NAME["lamp_state"]
    a = fam.emit(w, None, ctx, np.random.default_rng(7))
    b = fam.emit(w, None, ctx, np.random.default_rng(7))
    assert a == b and a is not None
    q_en, q_zh, ans = a
    assert ans == 1  # NOR(0,0) settles lit
    assert "{" not in q_en and q_en and q_zh
    # emit wiring: draw+answer reproduce the same answer for the same seed
    params = fam.draw(np.random.default_rng(7))
    assert fam.answer(w, None, ctx, params) == ans


# ---------- door_state / lamp_state / lever_combo (circuit families) ----------


def test_door_state_flips_with_world():
    fam = FAMILY_BY_NAME["door_state"]
    w = rs.PyWorld(1, "flat", [R(10, 5, 0, 10, 5, 0, ids.DOOR)])
    ctx = Ctx(task="circuit_door", door=(10, 5, 0))
    run(w, 2)
    assert fam.answer(w, None, ctx, {}) == 0  # closed
    w.set_block(10, 5, 0, ids.DOOR | (1 << 12))
    assert fam.answer(w, None, ctx, {}) == 1  # open
    # and back
    w.set_block(10, 5, 0, ids.DOOR)
    assert fam.answer(w, None, ctx, {}) == 0


def test_lamp_state_flips_with_world():
    w, ctx = nor_rig()
    fam = FAMILY_BY_NAME["lamp_state"]
    assert fam.answer(w, None, ctx, {}) == 1  # NOR(0,0) = 1
    w.set_block(2, 5, 2, ids.LEVER | (1 << 12))  # A on
    run(w, 8)
    assert fam.answer(w, None, ctx, {}) == 0


def test_lever_combo_truth_table_matches_nor():
    w, ctx = nor_rig()
    fam = FAMILY_BY_NAME["lever_combo"]
    # plan-pin assertions: combo (1,0) -> lamp 0 and (0,0) -> lamp 1
    assert fam.answer(w, None, ctx, {"a_bit": 1, "b_bit": 0}) == 0
    assert fam.answer(w, None, ctx, {"a_bit": 0, "b_bit": 0}) == 1
    # full NOR table
    assert fam.answer(w, None, ctx, {"a_bit": 0, "b_bit": 1}) == 0
    assert fam.answer(w, None, ctx, {"a_bit": 1, "b_bit": 1}) == 0
    # the probe must not disturb the live world (snapshot isolation)
    assert (w.get_block(6, 6, 3) >> 12) & 1 == 1
    assert (w.get_block(2, 5, 2) >> 12) & 1 == 0


# ---------- see_block / count_block (seg-derived) ----------


def sand_rig():
    """7 sand cells in a row, 60 cells ahead of the agent (+z, yaw 0):
    8 rendered pixels -> count bin '1-10', see -> yes (pinned by probe)."""
    spec = [R(-10, 4, -10, 10, 4, 90, ids.STONE)]
    for i in range(7):
        x = -3 + i
        spec.append(R(x, 5, 60, x, 5, 60, ids.SAND))
    w = rs.PyWorld(1, "void", spec)
    w.teleport(0.5, 5.0, 0.5)
    w.step(IDLE)  # yaw 0 (faces +z), level pitch
    return w


def test_see_and_count_block_from_render():
    w = sand_rig()
    _, _, seg, _ = w.render()
    obs = {"seg": seg}
    see = FAMILY_BY_NAME["see_block"]
    cnt = FAMILY_BY_NAME["count_block"]
    sand = {"block_id": ids.SAND}
    lava = {"block_id": ids.LAVA}
    n_px = int(np.count_nonzero(seg == ids.SAND))
    assert 1 <= n_px <= 10, f"sand pixel count drifted: {n_px}"
    assert see.answer(w, obs, None, sand) == 1
    assert see.answer(w, obs, None, lava) == 0
    assert cnt.answer(w, obs, None, sand) == 1  # 1-10 bin
    assert cnt.answer(w, obs, None, lava) == 0  # '0' bin
    # stamp more sand right in front -> pixel count crosses the >50 bin
    for x in range(-2, 3):
        for z in range(4, 7):
            w.set_block(x, 5, z, ids.SAND)
    _, _, seg2, _ = w.render()
    obs2 = {"seg": seg2}
    assert int(np.count_nonzero(seg2 == ids.SAND)) > 50
    assert cnt.answer(w, obs2, None, sand) == 3  # >50 bin


# ---------- ray_distance ----------


def test_ray_distance_bin_edges():
    fam = FAMILY_BY_NAME["ray_distance"]

    def bin_of(centi):
        return fam.answer(None, {"raycast": np.array([ids.STONE, centi], dtype=np.uint16)}, None, {})

    assert bin_of(0) == 0 and bin_of(199) == 0
    assert bin_of(200) == 1 and bin_of(299) == 1
    assert bin_of(300) == 2 and bin_of(449) == 2
    assert bin_of(450) == 3  # 450 = reach cap / no target (binding clamps)


def test_ray_distance_from_live_raycast():
    # eye at x=10.0 facing +x (yaw 270); wall face at x=13 -> 3.00 cells
    w = rs.PyWorld(1, "flat", [R(13, 5, 9, 13, 7, 11, ids.STONE)])
    w.teleport(10.0, 5.0, 10.5)
    w.step((0, 0, 0, 18, 4, 0, 0, 0, 0, 0))  # yaw bucket 18 = 270 deg
    obs = {"raycast": w.obs_raycast()}
    assert int(obs["raycast"][1]) == 300  # exact 3-cell pin
    fam = FAMILY_BY_NAME["ray_distance"]
    assert fam.answer(w, obs, None, {}) == 2  # 3-4.5 bin


# ---------- hazard_near (prior family) ----------


def test_hazard_near_distance_threshold():
    fam = FAMILY_BY_NAME["hazard_near"]
    w = rs.PyWorld(1, "flat")
    w.teleport(0.5, 5.0, 0.5)
    assert fam.answer(w, None, None, {}) == 0  # flat preset: no lava
    w.set_block(4, 5, 0, ids.LAVA)  # 4 cells away
    assert fam.answer(w, None, None, {}) == 1
    w.set_block(4, 5, 0, 0)
    w.set_block(8, 5, 0, ids.LAVA)  # 8 cells away
    assert fam.answer(w, None, None, {}) == 0


# ---------- biome ----------


def test_biome_round_trip():
    fam = FAMILY_BY_NAME["biome"]
    w = rs.PyWorld(3, "default")
    x, _, z = w.agent_pos()
    ans = fam.answer(w, None, None, {})
    assert ans == int(w.biome_at(int(x), int(z)))
    assert 0 <= ans <= 4


# ---------- direction ----------


def test_direction_left_right_ahead():
    fam = FAMILY_BY_NAME["direction"]
    w = rs.PyWorld(1, "flat")
    w.teleport(0.5, 5.0, 0.5)
    w.step((0, 0, 0, 0, 4, 0, 0, 0, 0, 0))  # yaw bucket 0 -> faces +z
    obs = {"pose": w.obs_pose()}
    ahead = Ctx(task="t", target=(0.5, 5.0, 20.5))   # +z
    left = Ctx(task="t", target=(-19.5, 5.0, 0.5))   # -x is left at yaw 0
    right = Ctx(task="t", target=(20.5, 5.0, 0.5))   # +x
    assert fam.answer(w, obs, ahead, {}) == 2
    assert fam.answer(w, obs, left, {}) == 0
    assert fam.answer(w, obs, right, {}) == 1
    # degenerate: on top of the marker -> inapplicable
    near = Ctx(task="t", target=(0.9, 5.0, 0.9))
    assert fam.answer(w, obs, near, {}) is None
    # rotate the agent 180 deg (bucket 12): the ahead target is now behind,
    # and left/right swap
    w.step((0, 0, 0, 12, 4, 0, 0, 0, 0, 0))
    obs = {"pose": w.obs_pose()}
    assert fam.answer(w, obs, ahead, {}) in (0, 1)
    assert fam.answer(w, obs, left, {}) == 1
    assert fam.answer(w, obs, right, {}) == 0


# ---------- craftable ----------


def test_craftable_flips_with_inventory_and_table():
    fam = FAMILY_BY_NAME["craftable"]
    w = rs.PyWorld(1, "flat")
    w.teleport(0.5, 5.0, 0.5)
    assert fam.answer(w, None, None, {}) == 0  # nothing
    w.give(ids.COBBLESTONE, 3)
    w.give(ids.ITEM_STICK, 2)
    assert fam.answer(w, None, None, {}) == 0  # no table yet
    w.set_block(3, 5, 0, ids.CRAFTING_TABLE)   # 3 cells away
    assert fam.answer(w, None, None, {}) == 1
    w.set_block(3, 5, 0, 0)                     # table removed -> flips back
    assert fam.answer(w, None, None, {}) == 0
    # table back -> craftable again
    w.set_block(3, 5, 0, ids.CRAFTING_TABLE)
    assert fam.answer(w, None, None, {}) == 1
    # materials matter: too few sticks even with a table
    w2 = rs.PyWorld(1, "flat")
    w2.teleport(0.5, 5.0, 0.5)
    w2.give(ids.COBBLESTONE, 3)
    w2.give(ids.ITEM_STICK, 1)
    w2.set_block(3, 5, 0, ids.CRAFTING_TABLE)
    assert fam.answer(w2, None, None, {}) == 0
