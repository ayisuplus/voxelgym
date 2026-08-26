"""Pixel-gallery (image -> voxel wall) tests: decoder round-trips,
area-average resizing, renderer-calibrated palette mapping, wall world
staticness, gallery family truth (incl. the state flip), byte-identical
regeneration, and train.py consuming a gallery dir unmodified.
"""

import json
import struct
import zlib

import numpy as np
import pytest

import voxelgym_rs as rs
from conftest import run
from voxelgym.vqa import FAMILY_BY_NAME, Ctx
from voxelgym.vqa import imagine
from voxelgym.vqa.families import PALETTE_BLOCKS

# renderer-measured palette pins (west face, shading factor 0.6); drift in
# the renderer shading model fails this test loudly — that is the signal.
PALETTE_REF = {
    "bedrock": (51, 51, 51), "stone": (75, 75, 75), "dirt": (82, 57, 36),
    "grass_block": (66, 99, 49), "log": (64, 49, 28), "leaves": (37, 82, 43),
    "planks": (96, 76, 48), "furnace": (66, 66, 66), "iron_ore": (129, 105, 88),
    "diamond_ore": (76, 128, 116), "tnt": (131, 28, 9), "lamp": (134, 105, 48),
    "glass": (115, 139, 149),
}


def _png_chunk(ctype: bytes, body: bytes) -> bytes:
    return (struct.pack(">I", len(body)) + ctype + body
            + struct.pack(">I", zlib.crc32(ctype + body) & 0xFFFFFFFF))


def _encode_png(rgb: np.ndarray) -> bytes:
    """Minimal test-local PNG encoder: 8-bit RGB, filter-0 scanlines."""
    h, w = rgb.shape[:2]
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + np.ascontiguousarray(rgb[y]).tobytes() for y in range(h))
    return (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", zlib.compress(raw)) + _png_chunk(b"IEND", b""))


def test_ppm_and_png_round_trip(tmp_path):
    px = (np.arange(18, dtype=np.uint8).reshape(2, 3, 3) * 7)  # 3x2 (W x H), all distinct
    ppm = tmp_path / "a.ppm"
    ppm.write_bytes(b"P6 3 2 255\n" + px.tobytes())
    np.testing.assert_array_equal(imagine.load_image(str(ppm)), px)
    png = tmp_path / "a.png"
    png.write_bytes(_encode_png(px))
    np.testing.assert_array_equal(imagine.load_image(str(png)), px)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n"
                    + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 3, 2, 16, 2, 0, 0, 0))
                    + _png_chunk(b"IEND", b""))
    with pytest.raises(ValueError, match="unsupported PNG variant"):
        imagine.load_image(str(bad))


def test_downsample_area_average():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, 2:] = 255  # left half black, right half white
    small = imagine.downsample(img, 2, 2)
    assert small.dtype == np.float64 and small.shape == (2, 2, 3)
    np.testing.assert_array_equal(small[:, 0], 0.0)
    np.testing.assert_array_equal(small[:, 1], 255.0)
    big = imagine.downsample(small, 4, 4)  # enlarge: nearest -> 2x2 blocks
    np.testing.assert_array_equal(big[:2, :2], 0.0)
    np.testing.assert_array_equal(big[2:, :2], 0.0)
    np.testing.assert_array_equal(big[:2, 2:], 255.0)
    np.testing.assert_array_equal(big[2:, 2:], 255.0)


def test_palette_calibration_and_mapping():
    pal = imagine.calibrate_palette()
    assert len(pal) == 13
    assert set(pal) == set(PALETTE_BLOCKS)
    for name, ref in PALETTE_REF.items():
        assert all(abs(g - r) <= 2 for g, r in zip(pal[name], ref)), (
            f"{name}: {pal[name]} vs pinned {ref}")
    red = imagine.map_pixels(np.array([[[255.0, 0.0, 0.0]]]), pal)
    black = imagine.map_pixels(np.array([[[0.0, 0.0, 0.0]]]), pal)
    dark_gray = imagine.map_pixels(np.array([[[40.0, 40.0, 40.0]]]), pal)
    assert red[0, 0] == rs.block_id("tnt")
    # nearest by squared-euclidean over the measured palette: black's
    # closest entry is log (7281), not bedrock (7803) — pin the math
    assert black[0, 0] == rs.block_id("log")
    assert dark_gray[0, 0] == rs.block_id("bedrock")


def test_wall_world_static():
    cells = np.empty((6, 8), dtype=np.uint16)
    cells[:, :4] = rs.block_id("stone")
    cells[:, 4:] = rs.block_id("dirt")
    spec = imagine.build_wall_spec(cells)
    w = rs.PyWorld(0, "void", spec)
    run(w, 20)  # idle ticks: nothing may fall / flow / despawn
    for x, y, z, x1, y1, z1, bid in spec:
        assert (x, y, z) == (x1, y1, z1)  # no depth relief -> single cells
        assert w.get_block(x, y, z) & 0xFFF == bid


def test_gallery_families_truth_and_flip():
    H, W = 6, 6
    cells = np.full((H, W), rs.block_id("bedrock"), dtype=np.uint16)
    cells[:4] = rs.block_id("tnt")  # top 4 rows tnt, bottom 2 bedrock
    # same construction as emit_gallery: cobblestone floor catches the
    # below-horizon rays (world-bottom bedrock plane) so palette masks stay clean
    w = rs.PyWorld(0, "void", imagine.build_wall_spec(cells) + [imagine.floor_spec(W, H, 3)])
    px, py, pz, yaw = imagine.view_poses(W, H, 3)[0]  # center view
    w.teleport(px, py, pz)
    w.step((0, 0, 0, yaw, 4, 0, 0, 0, 0, 0))
    _, _, seg, _ = w.render()
    obs = {"seg": seg}
    tnt, bed = rs.block_id("tnt"), rs.block_id("bedrock")
    assert np.count_nonzero(seg == tnt) > 0 and np.count_nonzero(seg == bed) > 0
    wd = FAMILY_BY_NAME["wall_dominant"]
    wr = FAMILY_BY_NAME["wall_region"]
    assert wd.answer(w, obs, None, {}) == PALETTE_BLOCKS.index("tnt")
    assert wr.answer(w, obs, None, {"block_id": bed, "region": 2}) == 1   # lower left
    assert wr.answer(w, obs, None, {"block_id": bed, "region": 0}) == 0   # upper left
    # flip: tnt rows become bedrock -> labels must follow state
    for i in range(4):
        y = 5 + (H - 1 - i)
        for j in range(W):
            w.set_block(40, y, -W // 2 + j, bed)
    _, _, seg2, _ = w.render()
    obs2 = {"seg": seg2}
    assert wd.answer(w, obs2, None, {}) == PALETTE_BLOCKS.index("bedrock")
    assert wr.answer(w, obs2, None, {"block_id": tnt, "region": 2}) == 0
    # emit wiring: (str, str, int), no unformatted placeholders
    ctx = Ctx(task="pixel_gallery")
    for fam in (wd, wr):
        out = fam.emit(w, obs2, ctx, np.random.default_rng(0))
        assert out is not None
        q_en, q_zh, ans = out
        assert isinstance(q_en, str) and isinstance(q_zh, str) and isinstance(ans, int)
        assert "{" not in q_en and "{" not in q_zh


def _demo_images(tmp_path, n=2):
    paths = []
    colors = [((200, 30, 20), (60, 60, 60)), ((40, 120, 60), (120, 80, 40)),
              ((30, 40, 60), (150, 130, 70))]
    for k in range(n):
        img = np.zeros((3, 4, 3), dtype=np.uint8)
        img[:, :2] = colors[k % len(colors)][0]
        img[:, 2:] = colors[k % len(colors)][1]
        p = tmp_path / f"img{k}.png"
        p.write_bytes(_encode_png(img))
        paths.append(str(p))
    return paths


def test_emit_gallery_deterministic(tmp_path):
    paths = _demo_images(tmp_path, n=2)
    out1, out2 = str(tmp_path / "g1"), str(tmp_path / "g2")
    imagine.emit_gallery(paths, out1, cells=8, views=3, preview=True)
    imagine.emit_gallery(paths, out2, cells=8, views=3, preview=True)
    import os
    m1 = open(os.path.join(out1, "manifest.jsonl"), "rb").read()
    m2 = open(os.path.join(out2, "manifest.jsonl"), "rb").read()
    assert m1 == m2
    rows = [json.loads(l) for l in m1.decode("utf-8").splitlines()]
    assert rows, "manifest must not be empty"
    for r in rows:
        assert r["task"] == "pixel_gallery"
        parts = r["id"].split("/")
        assert parts[0] == "pixel_gallery"
        assert int(parts[1]) == r["seed"] and int(parts[2]) == r["tick"]
    npz = np.load(os.path.join(out1, "pixel_gallery.npz"))
    assert set(npz.files) == {"id"} | set(imagine.TENSOR_KEYS)
    ids = npz["id"].tolist()
    assert len(ids) == 2 * 3  # 2 images x 3 views
    for sid in ids:
        parts = sid.split("/")
        assert parts[0] == "pixel_gallery" and len(parts) == 3
    prev = tmp_path / "g1" / "preview"
    for name in ("img0_map.ppm", "img0_view0.ppm", "img0_view1.ppm", "img0_view2.ppm"):
        assert (prev / name).read_bytes().startswith(b"P6")


@pytest.mark.ml
def test_train_smoke_on_gallery(tmp_path):
    paths = _demo_images(tmp_path, n=3)  # seeds 0-1 -> test split; seed 2 -> train
    out = str(tmp_path / "g")
    imagine.emit_gallery(paths, out, cells=8, views=3)
    from voxelgym.vqa.train import Dataset, build_vocab, split_items, train_arm
    ds = Dataset(out)
    train_items, test_items = split_items(ds.items)
    assert train_items and test_items
    vocab = build_vocab(train_items)
    res = train_arm(ds, "rgb", train_items, test_items, vocab,
                    limit_steps=5, batch=8, eval_every=10**9)
    assert isinstance(res, dict) and "acc" in res
    assert res["acc"].get("wall_dominant") is not None
