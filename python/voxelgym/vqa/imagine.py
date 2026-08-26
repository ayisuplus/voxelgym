"""Real-image -> voxel "pixel gallery" importer: map images into the world
as block-pixel walls, render calibrated views, emit gen-compatible VQA data.

    python -m voxelgym.vqa.imagine <image...> [--cells 64 --views 3 --out data/gallery]
                                              [--depth-dir DIR] [--preview]

Each image is quantized to the 13-block palette (families.PALETTE_BLOCKS,
colors calibrated against THIS renderer at run time) and built as a
vertical wall in a void world — the sky background keeps seg-derived labels
unpolluted. `views` camera poses per image (center, then +/-W/4, +/-W/2,
... laterally, each aimed at wall center) are rendered; every applicable
family (tasks containing "pixel_gallery": see_block, count_block,
ray_distance, wall_dominant, wall_region) emits one QA per view from live
state. The per-sample rng convention matches gen.py
(default_rng(seed*1000003 + tick)) and argv image order IS the seed order,
so reruns with the same args are byte-identical in the manifest.

Output: <out>/manifest.jsonl + <out>/pixel_gallery.npz with the exact 8
tensor keys/dtypes of gen.py, so train.py consumes a gallery dir unchanged.
make_task("pixel_gallery") deliberately does NOT exist: gallery data comes
only from this CLI (passing the name to gen.py raises KeyError). train.py
splits seeds 0-1 to test, so <3 images leaves the train split empty and
with few images the test share is proportionally large — accepted, use
>=3 images for training.

Image decoding is stdlib-only: PNG (8-bit, non-interlaced, gray/RGB/RGBA)
and PPM/PNM (P6 binary). Unsupported variants raise ValueError telling the
user to convert. --depth-dir supplies optional per-image depth maps
(<stem>.png/.ppm, 255 = closest) extruded up to 4 cells toward the viewer
for real parallax; without it the wall is 1 cell thick. Extreme portraits
are aspect-distorted (H clamps to 96 cells) — accepted and documented.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import zlib
from collections import Counter

import numpy as np

import voxelgym_rs as rs

from .families import FAMILIES, PALETTE_BLOCKS, Ctx

TASK = "pixel_gallery"
WALL_X0 = 40   # wall plane (west face x); camera looks +x
WALL_Y0 = 5    # y of the wall's bottom row of cells
MAX_CELLS = 96
SKY = (0x78, 0xA6, 0xFF)

LIDAR = {"channels": 16, "azimuth": 256, "min_elev": -20, "max_elev": 10, "max_range": 48}
TENSOR_KEYS = ("rgb", "depth", "normals", "seg", "lidar_range", "voxels", "pose", "inventory")

# The renderer's ChunkGrid returns BEDROCK for y < 0 (voxel-view
# RENDER_RADIUS_CHUNKS*16 = 96-cell ray cap; off-grid reads are sky), so a
# level camera always sees a world-bottom bedrock plane below the horizon —
# and bedrock IS in the palette, which would pollute every palette mask. A
# cobblestone slab just under the wall's bottom row, spanning every pose's
# full render radius, catches those rays instead: any ray that could reach
# y<0 within the 96-cell cap crosses the slab-top plane over the slab.
# Cobblestone is in neither PALETTE_IDS nor QUERY_BLOCKS, so all family
# labels stay exact. One inclusive region tuple per image.
def floor_spec(W: int, H: int, views: int) -> tuple:
    poses = view_poses(W, H, views)
    r = 96.0  # render grid radius: voxel-view RENDER_RADIUS_CHUNKS * 16
    x0 = int(math.floor(min(p[0] for p in poses) - r))
    x1 = int(math.ceil(max(p[0] for p in poses) + r))
    z0 = int(math.floor(min(p[2] for p in poses) - r))
    z1 = int(math.ceil(max(p[2] for p in poses) + r))
    return (x0, WALL_Y0 - 1, z0, x1, WALL_Y0 - 1, z1, rs.block_id("cobblestone"))

# level-gaze orienting action: yaw bucket filled per pose, pitch bucket 4
_ORIENT = (0, 0, 0, 0, 4, 0, 0, 0, 0, 0)


# ---------------- image decoding (stdlib only) ----------------


def load_image(path: str) -> np.ndarray:
    """Decode an image file to (H, W, 3) uint8. Supported: PPM/PNM (P6)
    and PNG (8-bit, non-interlaced, color types 0/2/6)."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as f:
        data = f.read()
    if ext in (".ppm", ".pnm"):
        return _decode_ppm(data, path)
    if ext == ".png":
        return _decode_png(data, path)
    raise ValueError(
        f"unsupported image format {ext!r} in {path}; supported formats: PNG, PPM")


def _decode_ppm(data: bytes, path: str) -> np.ndarray:
    toks: list[bytes] = []
    i = 0
    while len(toks) < 4:
        while i < len(data) and data[i:i + 1].isspace():
            i += 1
        if i < len(data) and data[i:i + 1] == b"#":  # PNM comment
            while i < len(data) and data[i] != 0x0A:
                i += 1
            continue
        j = i
        while j < len(data) and not data[j:j + 1].isspace():
            j += 1
        toks.append(data[i:j])
        i = j
    if len(toks) < 4 or toks[0] != b"P6":
        raise ValueError(f"unsupported PPM variant in {path}; need P6 binary")
    w, h, maxval = int(toks[1]), int(toks[2]), int(toks[3])
    if maxval != 255:
        raise ValueError(f"unsupported PPM maxval {maxval} in {path}; need 255")
    i += 1  # single whitespace byte separating header from payload
    payload = data[i:i + w * h * 3]
    if len(payload) < w * h * 3:
        raise ValueError(f"truncated PPM payload in {path}")
    return np.frombuffer(payload, np.uint8).reshape(h, w, 3).copy()


def _decode_png(data: bytes, path: str) -> np.ndarray:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG file: {path}")
    ihdr = None
    idat = bytearray()
    pos = 8
    while pos + 12 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif ctype == b"IDAT":
            idat += body
        elif ctype == b"IEND":
            break
    if ihdr is None:
        raise ValueError(f"missing IHDR chunk in {path}")
    w, h, bit_depth, color_type, _comp, _filt, interlace = ihdr
    if bit_depth != 8 or interlace != 0 or color_type not in (0, 2, 6):
        raise ValueError(
            f"unsupported PNG variant in {path}; convert to 8-bit non-interlaced PNG or PPM")
    ch = {0: 1, 2: 3, 6: 4}[color_type]
    raw = zlib.decompress(bytes(idat))
    stride = w * ch
    out = np.zeros((h, stride), dtype=np.uint8)
    prev = bytearray(stride)
    p = 0
    for y in range(h):
        filt = raw[p]
        p += 1
        line = bytearray(raw[p:p + stride])
        p += stride
        if filt == 0:    # None
            pass
        elif filt == 1:  # Sub
            for x in range(ch, stride):
                line[x] = (line[x] + line[x - ch]) & 0xFF
        elif filt == 2:  # Up
            for x in range(stride):
                line[x] = (line[x] + prev[x]) & 0xFF
        elif filt == 3:  # Average
            for x in range(stride):
                a = line[x - ch] if x >= ch else 0
                line[x] = (line[x] + ((a + prev[x]) >> 1)) & 0xFF
        elif filt == 4:  # Paeth
            for x in range(stride):
                a = line[x - ch] if x >= ch else 0
                b = prev[x]
                c = prev[x - ch] if x >= ch else 0
                pp = a + b - c
                pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[x] = (line[x] + pr) & 0xFF
        else:
            raise ValueError(f"unsupported PNG row filter {filt} in {path}")
        out[y] = np.frombuffer(bytes(line), np.uint8)
        prev = line
    if color_type == 0:
        img = np.repeat(out.reshape(h, w, 1), 3, axis=2)
    elif color_type == 2:
        img = out.reshape(h, w, 3)
    else:  # 6: RGBA -> drop alpha
        img = out.reshape(h, w, 4)[:, :, :3]
    return np.ascontiguousarray(img)


# ---------------- resizing / palette mapping ----------------


def downsample(img: np.ndarray, W: int, H: int) -> np.ndarray:
    """Resize (h, w, 3) to (H, W, 3) float64. Shrinking an axis:
    area-average over linspace bin boundaries; enlarging: nearest index.
    Per-axis independent, deterministic."""
    out = img.astype(np.float64)
    out = _resize_axis(out, H, axis=0)
    out = _resize_axis(out, W, axis=1)
    return out


def _resize_axis(img: np.ndarray, m: int, axis: int) -> np.ndarray:
    n = img.shape[axis]
    if m == n:
        return img
    if m < n:
        bounds = np.linspace(0, n, m + 1).round().astype(int)
        sums = np.add.reduceat(img, bounds[:-1], axis=axis)
        widths = np.diff(bounds).astype(np.float64)  # strictly positive (n/m > 1)
        shape = [1] * img.ndim
        shape[axis] = m
        return sums / widths.reshape(shape)
    idx = np.linspace(0, n - 1, m).round().astype(int)
    return np.take(img, idx, axis=axis)


def calibrate_palette() -> dict[str, tuple[int, int, int]]:
    """Render-measured west-face color per palette block: each block is
    rendered alone in a void world (level gaze, facing +x) and the median
    of its seg-masked rgb pixels is the palette color. A block reading as
    sky (non-occluding) is dropped with a warning."""
    pal: dict[str, tuple[int, int, int]] = {}
    sky = np.array(SKY, dtype=np.float64)
    for name in PALETTE_BLOCKS:
        bid = rs.block_id(name)
        w = rs.PyWorld(1, "void", [(20, 5, -3, 20, 8, 3, bid)])
        w.teleport(14.5, 6.6, 0.5)
        w.step((0, 0, 0, 18, 4, 0, 0, 0, 0, 0))  # yaw bucket 18 = 270 deg: face +x, level
        rgb, _depth, seg, _normals = w.render()
        px = rgb[seg == bid].astype(np.float64)
        if len(px) == 0:
            print(f"warning: palette block {name} rendered no pixels; dropped", flush=True)
            continue
        med = np.median(px, axis=0)
        if np.all(np.abs(med - sky) <= 8):
            print(f"warning: palette block {name} reads as sky {tuple(med)}; dropped", flush=True)
            continue
        pal[name] = tuple(int(v) for v in med)
    return pal


def map_pixels(img_float: np.ndarray, palette: dict[str, tuple[int, int, int]]) -> np.ndarray:
    """Nearest palette block per pixel (squared euclidean in RGB) ->
    (H, W) uint16 block ids."""
    names = list(palette.keys())
    colors = np.array([palette[n] for n in names], dtype=np.float64)  # (P, 3)
    d2 = ((img_float[..., None, :] - colors) ** 2).sum(axis=-1)       # (H, W, P)
    nearest = d2.argmin(axis=-1)
    bids = np.array([rs.block_id(n) for n in names], dtype=np.uint16)
    return bids[nearest]


# ---------------- world construction / views ----------------


def build_wall_spec(cells: np.ndarray, depth: np.ndarray | None = None) -> list[tuple]:
    """Vertical wall at x = WALL_X0. Image col j -> z = -W//2 + j; image
    row i (top = 0) -> y = WALL_Y0 + (H-1-i). With a depth map (uint8 HxW,
    255 = closest) each pixel extrudes k = round(d/255*4) cells toward the
    viewer: x in [WALL_X0-k, WALL_X0]."""
    H, W = cells.shape
    spec = []
    for i in range(H):
        y = WALL_Y0 + (H - 1 - i)
        for j in range(W):
            z = -W // 2 + j
            bid = int(cells[i, j])
            if depth is None:
                spec.append((WALL_X0, y, z, WALL_X0, y, z, bid))
            else:
                k = int(round(int(depth[i, j]) / 255 * 4))
                spec.append((WALL_X0 - k, y, z, WALL_X0, y, z, bid))
    return spec


def view_poses(W: int, H: int, views: int) -> list[tuple[float, float, float, int]]:
    """(feet_x, feet_y, feet_z, yaw_bucket) per view. Eye at wall-center
    height cy = WALL_Y0 + H/2, distance D = max(8, round(0.65*W)) west of
    the wall, lateral offsets 0, +W/4, -W/4, +W/2, ... (cyclic), each aimed
    at wall center. Pitch is level (bucket 4), filled by the caller."""
    D = max(8, round(0.65 * W))
    cx = WALL_X0 - D
    cy = WALL_Y0 + H / 2
    cz = -W // 2 + W / 2 - 0.5
    poses = []
    for k in range(views):
        off = ((-1) ** (k + 1)) * math.ceil(k / 2) * (W / 4)
        zcam = cz + off
        # sim convention: forward = (-sin yaw, cos yaw); aim at (WALL_X0, cz)
        yaw = math.degrees(math.atan2(-(WALL_X0 - cx), cz - zcam))
        poses.append((float(cx), float(cy - 1.62), float(zcam), round(yaw / 15) % 24))
    return poses


# ---------------- emission ----------------


def _collect_obs(world, view_idx: int) -> dict:
    """The exact 8 gen tensor keys (plus raycast, consumed live by the
    ray_distance family), mirroring env._obs assembly."""
    rgb, depth, seg, normals = world.render()
    lidar_range, _inten, _lseg = world.lidar_scan(
        channels=LIDAR["channels"], azimuth_steps=LIDAR["azimuth"],
        min_elev_deg=LIDAR["min_elev"], max_elev_deg=LIDAR["max_elev"],
        max_range=LIDAR["max_range"], frame_idx=view_idx)
    return {
        "rgb": rgb,
        "depth": depth.astype(np.float16),
        "normals": normals,
        "seg": seg,
        "lidar_range": lidar_range,
        "voxels": world.obs_voxels(),
        "pose": world.obs_pose(),
        "inventory": world.obs_inventory(),
        "raycast": world.obs_raycast(),
    }


def _sample_rows(img_idx: int, view_idx: int, world, obs, crashes: Counter) -> list[dict]:
    rng = np.random.default_rng(img_idx * 1000003 + view_idx)
    rows = []
    for fam in FAMILIES:
        if TASK not in fam.tasks:
            continue
        try:
            out = fam.emit(world, obs, Ctx(task=TASK), rng)
        except Exception:
            crashes[fam.name] += 1  # absent from this sample; never fatal
            continue
        if out is None:
            continue
        q_en, q_zh, ans = out
        rows.append({
            "id": f"{TASK}/{img_idx}/{view_idx}",
            "task": TASK,
            "seed": img_idx,
            "tick": view_idx,
            "family": fam.name,
            "q_en": q_en,
            "q_zh": q_zh,
            "answer": ans,
            "needs": sorted(fam.needs),
        })
    return rows


def _load_depth(depth_dir: str | None, stem: str, H: int, W: int) -> np.ndarray | None:
    if depth_dir is None:
        return None
    for ext in (".png", ".ppm", ".pnm"):
        cand = os.path.join(depth_dir, stem + ext)
        if os.path.exists(cand):
            d = load_image(cand)[:, :, 0]
            ri = np.linspace(0, d.shape[0] - 1, H).round().astype(int)
            ci = np.linspace(0, d.shape[1] - 1, W).round().astype(int)
            return d[np.ix_(ri, ci)]
    return None


def _write_ppm(path: str, rgb: np.ndarray) -> None:
    h, w = rgb.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P6 {w} {h} 255\n".encode())
        f.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())


def _map_preview(cells_grid: np.ndarray, id_to_rgb: dict[int, tuple[int, int, int]]) -> np.ndarray:
    h, w = cells_grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for bid, color in id_to_rgb.items():
        img[cells_grid == bid] = color
    return np.repeat(np.repeat(img, 8, axis=0), 8, axis=1)  # x8 nearest upscale


def emit_gallery(image_paths, out_dir: str, cells: int = 64, views: int = 3,
                 depth_dir: str | None = None, preview: bool = False) -> Counter:
    """Build one wall world per image, render `views` poses, emit QA rows +
    stacked tensors. argv order is seed order. Returns per-family counts."""
    os.makedirs(out_dir, exist_ok=True)
    palette = calibrate_palette()
    id_to_rgb = {rs.block_id(n): palette[n] for n in palette}
    crashes: Counter = Counter()
    family_counts: Counter = Counter()
    all_samples: list = []
    with open(os.path.join(out_dir, "manifest.jsonl"), "w", encoding="utf-8") as mf:
        for img_idx, path in enumerate(image_paths):
            stem = os.path.splitext(os.path.basename(path))[0]
            img = load_image(path)
            h0, w0 = img.shape[:2]
            W = min(cells, MAX_CELLS)
            H = min(max(1, round(W * h0 / w0)), MAX_CELLS)
            cells_grid = map_pixels(downsample(img, W, H), palette)
            depth = _load_depth(depth_dir, stem, H, W)
            world = rs.PyWorld(0, "void", build_wall_spec(cells_grid, depth)
                               + [floor_spec(W, H, views)])
            prev_dir = None
            if preview:
                prev_dir = os.path.join(out_dir, "preview")
                os.makedirs(prev_dir, exist_ok=True)
                _write_ppm(os.path.join(prev_dir, f"{stem}_map.ppm"),
                           _map_preview(cells_grid, id_to_rgb))
            for view_idx, (px, py, pz, yaw) in enumerate(view_poses(W, H, views)):
                world.teleport(px, py, pz)
                a = list(_ORIENT)
                a[3] = yaw
                world.step(tuple(a))  # 1-tick gravity sag ~0.08 cells: harmless, labels from seg
                obs = _collect_obs(world, view_idx)
                all_samples.append((f"{TASK}/{img_idx}/{view_idx}", obs))
                if preview:
                    _write_ppm(os.path.join(prev_dir, f"{stem}_view{view_idx}.ppm"), obs["rgb"])
                for row in _sample_rows(img_idx, view_idx, world, obs, crashes):
                    mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    family_counts[row["family"]] += 1
            print(f"{stem}: {W}x{H} cells, {views} views", flush=True)
    ids = [sid for sid, _ in all_samples]
    arrays = {k: np.stack([obs[k] for _, obs in all_samples]) for k in TENSOR_KEYS}
    np.savez_compressed(os.path.join(out_dir, f"{TASK}.npz"), id=np.array(ids), **arrays)
    if crashes:
        print(f"family emit crashes (samples skipped, never fatal): {dict(crashes)}", flush=True)
    print("per-family QA counts:", dict(sorted(family_counts.items())), flush=True)
    return family_counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        epilog="Note: train.py sends seeds 0-1 to the test split; pass >=3 images "
               "for a non-empty train split.")
    ap.add_argument("images", nargs="+",
                    help="image files (PNG 8-bit non-interlaced, or PPM P6)")
    ap.add_argument("--cells", type=int, default=64, help="wall width in block pixels (<=96)")
    ap.add_argument("--views", type=int, default=3, help="rendered viewpoints per image")
    ap.add_argument("--out", default="data/gallery")
    ap.add_argument("--depth-dir", default=None,
                    help="optional per-image depth maps <stem>.png/.ppm (255=closest, extrusion<=4 cells)")
    ap.add_argument("--preview", action="store_true",
                    help="write <out>/preview/*.ppm (quantized map x8 + rendered views)")
    args = ap.parse_args(argv)
    emit_gallery(args.images, args.out, cells=args.cells, views=args.views,
                 depth_dir=args.depth_dir, preview=args.preview)
    return 0


if __name__ == "__main__":
    sys.exit(main())
