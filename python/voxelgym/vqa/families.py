"""VQA question families: auto-generated visual question answering with
sim-derived ground truth.

Every family's answer is computed from live sim state (render seg map,
raycast, world accessors, or a clone-sim truth probe) — never sampled, so
labels are exact by construction. `emit` draws all randomness from the
caller-provided rng (gen seeds it per sample as default_rng(seed*1000003
+ tick)), so regeneration with the same args is bit-identical.

Answer index convention: yes/no families use 0 = no, 1 = yes. Other
families order their `classes` tuple as listed.

`needs` is the set of input modalities the answer is derivable from:
"rgb" (visible in the rendered frame), "voxels" (readable from the
structured voxel/inventory state), "prior" (only learnable as a corpus
prior — e.g. lava may be buried and invisible). Families with
needs == {"prior"} are excluded from the acceptance macro.

Cell encoding reminder: raw u16 cell = (state << 12) | block_id.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

import voxelgym_rs as rs

from .. import ids

# idle action: pitch bucket 4 = level gaze; same as tests' conftest.IDLE
IDLE = (0, 0, 0, 0, 4, 0, 0, 0, 0, 0)

SKY_SEG = 0xFFFF

# blocks sampled by the see/count families (registry names)
QUERY_BLOCKS = (
    "lava", "water", "log", "sand", "stone", "leaves", "crafting_table",
    "furnace", "torch", "door", "wire", "lever", "lamp",
)
BLOCK_ZH = {
    "lava": "岩浆", "water": "水", "log": "原木", "sand": "沙子",
    "stone": "石头", "leaves": "树叶", "crafting_table": "工作台",
    "furnace": "熔炉", "torch": "火把", "door": "门", "wire": "红石线",
    "lever": "拉杆", "lamp": "红石灯",
}

# pixel-gallery palette: 13 static solid blocks whose rendered colors are
# pairwise distinct (see imagine.calibrate_palette). All non-loose (no
# falling), non-fluid, non-mechanism — a wall built from them is forever
# static in a void world. Near-duplicate grays (cobblestone/coal_ore/
# crafting_table) are deliberately excluded.
PALETTE_BLOCKS = (
    "bedrock", "stone", "dirt", "grass_block", "log", "leaves", "planks",
    "furnace", "iron_ore", "diamond_ore", "tnt", "lamp", "glass",
)
PALETTE_IDS = tuple(rs.block_id(n) for n in PALETTE_BLOCKS)
PALETTE_ZH = {
    "bedrock": "基岩", "stone": "石头", "dirt": "泥土", "grass_block": "草方块",
    "log": "原木", "leaves": "树叶", "planks": "木板", "furnace": "熔炉",
    "iron_ore": "铁矿石", "diamond_ore": "钻石矿石", "tnt": "TNT",
    "lamp": "红石灯", "glass": "玻璃",
}
# wall_region quadrants, index 0..3
REGION_EN = ("upper left", "upper right", "lower left", "lower right")
REGION_ZH = ("左上", "右上", "左下", "右下")

BIOME_NAMES = ("ocean", "plains", "desert", "hills", "volcanic")  # u8 0..4

# default gen task set (gen.py intersects family.tasks with the episode task)
ALL_TASKS = (
    "navigate_to_target", "collect_log", "smelt_iron", "circuit_door",
    "plate_door", "logic_probe", "collapse_judge",
)
DEFAULT_BIOME_TASKS = ("navigate_to_target", "collect_log", "smelt_iron")
DIRECTION_TASKS = ("navigate_to_target", "circuit_door", "plate_door", "logic_probe")
DOOR_TASKS = ("circuit_door", "plate_door")
LOGIC_TASKS = ("logic_probe",)
CRAFT_TASKS = ("collect_log", "smelt_iron")


@dataclass
class Ctx:
    """Per-episode anchors a family may need (fixed scenario cells / goals)."""

    task: str
    target: tuple[float, float, float] | None = None  # direction goal (x,y,z)
    door: tuple[int, int, int] | None = None
    lamp: tuple[int, int, int] | None = None
    lever_a: tuple[int, int, int] | None = None
    lever_b: tuple[int, int, int] | None = None


@dataclass
class Family:
    """One question family. `draw` pulls family params from the sample rng;
    `answer` computes the class index from live state (None = inapplicable,
    skip this sample); `emit` formats a drawn template pair."""

    name: str
    needs: frozenset[str]
    tasks: tuple[str, ...]
    classes: tuple[str, ...]
    en: tuple[str, ...]
    zh: tuple[str, ...]
    draw: Callable[[np.random.Generator, dict | None], dict]
    answer: Callable[[object, dict | None, Ctx, dict], int | None]

    def emit(self, world, obs, ctx: Ctx, rng: np.random.Generator):
        params = self.draw(rng, obs)
        ans = self.answer(world, obs, ctx, params)
        if ans is None:
            return None
        ti = int(rng.integers(2**31))
        q_en = self.en[ti % len(self.en)].format(**params)
        q_zh = self.zh[ti % len(self.zh)].format(**params)
        return q_en, q_zh, ans


# ---------------- family param draws ----------------


def _draw_block(rng: np.random.Generator, obs=None) -> dict:
    name = QUERY_BLOCKS[int(rng.integers(len(QUERY_BLOCKS)))]
    return {"block": name, "block_zh": BLOCK_ZH[name], "block_id": rs.block_id(name)}


def _draw_none(rng: np.random.Generator, obs=None) -> dict:
    return {}


def _draw_combo(rng: np.random.Generator, obs=None) -> dict:
    a = int(rng.integers(2))
    b = int(rng.integers(2))
    on_off = ("off", "on")
    return {
        "a": on_off[a], "b": on_off[b], "a_zh": ("关", "开")[a], "b_zh": ("关", "开")[b],
        "a_bit": a, "b_bit": b,
    }


def _draw_wall_region(rng: np.random.Generator, obs=None) -> dict:
    region = int(rng.integers(4))
    seg = None if obs is None else obs.get("seg")
    if seg is None:
        pool = list(PALETTE_BLOCKS)
    else:
        pool = [n for n, bid in zip(PALETTE_BLOCKS, PALETTE_IDS)
                if np.count_nonzero(seg == bid) > 0] or list(PALETTE_BLOCKS)
    name = pool[int(rng.integers(len(pool)))]
    return {
        "block": name, "block_zh": PALETTE_ZH[name], "block_id": rs.block_id(name),
        "region_en": REGION_EN[region], "region_zh": REGION_ZH[region],
        "region": region,
    }


# ---------------- truth mechanisms ----------------


def _ans_see_block(world, obs, ctx, p) -> int:
    return 1 if np.count_nonzero(obs["seg"] == p["block_id"]) > 0 else 0


def _ans_count_block(world, obs, ctx, p) -> int:
    n = int(np.count_nonzero(obs["seg"] == p["block_id"]))
    if n == 0:
        return 0
    if n <= 10:
        return 1
    if n <= 50:
        return 2
    return 3


def _ans_ray_distance(world, obs, ctx, p) -> int:
    d = int(obs["raycast"][1])  # centi-cells; 450 = reach cap / no target
    if d >= 450:
        return 3  # none
    if d < 200:
        return 0  # < 2 cells
    if d < 300:
        return 1  # 2-3 cells
    return 2      # 3-4.5 cells


def _ans_hazard_near(world, obs, ctx, p) -> int:
    return 1 if len(world.find_blocks(ids.LAVA, 5)) > 0 else 0


def _ans_biome(world, obs, ctx, p) -> int:
    x, _, z = world.agent_pos()
    return int(world.biome_at(int(x), int(z)))


def _ans_direction(world, obs, ctx, p) -> int | None:
    if ctx.target is None:
        return None
    x, _, z, yaw = (float(v) for v in obs["pose"][:4])
    dx = ctx.target[0] - x
    dz = ctx.target[2] - z
    if math.hypot(dx, dz) < 2.0:
        return None  # on top of the marker: bearing is degenerate
    # sim convention: forward = (-sin yaw, cos yaw), left = (-cos yaw, -sin yaw)
    yaw_t = math.degrees(math.atan2(-dx, dz))
    diff = (yaw_t - yaw + 180.0) % 360.0 - 180.0
    if abs(diff) <= 15.0:
        return 2  # ahead
    return 0 if diff > 0.0 else 1  # left / right


def _ans_door_state(world, obs, ctx, p) -> int | None:
    if ctx.door is None:
        return None
    return (world.get_block(*ctx.door) >> 12) & 1


def _ans_lamp_state(world, obs, ctx, p) -> int | None:
    if ctx.lamp is None:
        return None
    return (world.get_block(*ctx.lamp) >> 12) & 1


def _ans_lever_combo(world, obs, ctx, p) -> int | None:
    if ctx.lamp is None or ctx.lever_a is None or ctx.lever_b is None:
        return None
    # clone-sim truth probe: same pattern as LogicProbeExpert._truth_table
    # (experts.py): snapshot -> force both levers -> 8 idle settle ticks ->
    # read the lamp state bit. The sim is the ground truth.
    scratch = rs.PyWorld(0, "void")
    scratch.restore(world.snapshot())
    scratch.set_block(*ctx.lever_a, ids.LEVER | (p["a_bit"] << 12))
    scratch.set_block(*ctx.lever_b, ids.LEVER | (p["b_bit"] << 12))
    for _ in range(8):
        scratch.step(IDLE)
    return (scratch.get_block(*ctx.lamp) >> 12) & 1


def _ans_craftable(world, obs, ctx, p) -> int:
    ok = (
        world.count_item(ids.COBBLESTONE) >= 3
        and world.count_item(ids.ITEM_STICK) >= 2
        and len(world.find_blocks(ids.CRAFTING_TABLE, 4)) > 0
    )
    return 1 if ok else 0


def _ans_wall_dominant(world, obs, ctx, p) -> int | None:
    seg = obs["seg"]
    counts = np.array([np.count_nonzero(seg == bid) for bid in PALETTE_IDS])
    if counts.sum() == 0:
        return None  # wall off-frame / no palette pixels
    return int(np.argmax(counts))  # ties -> lowest palette index


def _ans_wall_region(world, obs, ctx, p) -> int | None:
    seg = obs["seg"]
    mask = np.isin(seg, PALETTE_IDS)
    if not mask.any():
        return None  # wall off-frame / no palette pixels
    rows, cols = np.nonzero(mask)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    rmid, cmid = (r0 + r1) // 2, (c0 + c1) // 2
    region = p["region"]  # 0=upper left, 1=upper right, 2=lower left, 3=lower right
    ra, rb = (r0, rmid) if region < 2 else (rmid + 1, r1)
    ca, cb = (c0, cmid) if region % 2 == 0 else (cmid + 1, c1)
    if ra > rb or ca > cb:
        return 0  # degenerate (single-row/col) bbox half: nothing can be there
    sub = seg[ra:rb + 1, ca:cb + 1]
    return 1 if np.count_nonzero(sub == p["block_id"]) > 0 else 0


# ---------------- registry ----------------

_YESNO = ("no", "yes")

FAMILIES: list[Family] = [
    Family(
        "see_block", frozenset({"rgb"}), ALL_TASKS + ("pixel_gallery",), _YESNO,
        (
            "Do you see any {block}?",
            "Is there any {block} visible?",
            "Can you see {block} in view?",
            "Is {block} visible in front of you?",
        ),
        (
            "你看到{block_zh}了吗？",
            "视野里有{block_zh}吗？",
            "你能看到{block_zh}吗？",
            "眼前有{block_zh}吗？",
        ),
        _draw_block, _ans_see_block,
    ),
    Family(
        "count_block", frozenset({"rgb"}), ALL_TASKS + ("pixel_gallery",), ("0", "1-10", "11-50", ">50"),
        (
            "How many {block} blocks are visible?",
            "How many {block} blocks do you see?",
            "Count the visible {block} blocks.",
            "How much {block} is in view?",
        ),
        (
            "能看到多少个{block_zh}？",
            "视野里有多少{block_zh}？",
            "数一数可见的{block_zh}。",
            "你看到了多少{block_zh}？",
        ),
        _draw_block, _ans_count_block,
    ),
    Family(
        "ray_distance", frozenset({"rgb"}), ALL_TASKS + ("pixel_gallery",), ("<2", "2-3", "3-4.5", "none"),
        (
            "How far is the block you are looking at?",
            "What is the distance to the block in your crosshair?",
            "How far away is the targeted block?",
            "Estimate the distance to the block you face.",
        ),
        (
            "你正对着的方块有多远？",
            "准星指的方块距离多远？",
            "你注视的方块离你多远？",
            "目标方块的距离是多少？",
        ),
        _draw_none, _ans_ray_distance,
    ),
    Family(
        "hazard_near", frozenset({"prior"}), ALL_TASKS, _YESNO,
        (
            "Is there lava within 5 cells of you?",
            "Is lava nearby, within 5 blocks?",
            "Are you within 5 cells of lava?",
            "Is there any lava close to you?",
        ),
        (
            "你附近5格内有岩浆吗？",
            "岩浆离你5格以内吗？",
            "周围5格之内有岩浆吗？",
            "你离岩浆在5格以内吗？",
        ),
        _draw_none, _ans_hazard_near,
    ),
    Family(
        "biome", frozenset({"rgb"}), DEFAULT_BIOME_TASKS, BIOME_NAMES,
        (
            "What biome are you in?",
            "Which biome is this?",
            "What kind of biome surrounds you?",
            "Name the biome you are standing in.",
        ),
        (
            "你在什么生物群系里？",
            "这里是哪种生物群系？",
            "你身处哪个生物群系？",
            "说出你所在的生物群系。",
        ),
        _draw_none, _ans_biome,
    ),
    Family(
        "direction", frozenset({"rgb"}), DIRECTION_TASKS, ("left", "right", "ahead"),
        (
            "Is the goal marker to your left, right, or ahead?",
            "Where is the goal: left, right, or ahead?",
            "Which way is the goal marker, left, right, or straight ahead?",
            "Is the target to your left, your right, or in front of you?",
        ),
        (
            "目标标记在你的左边、右边还是正前方？",
            "目标在哪个方向：左、右还是前方？",
            "目标点在你左侧、右侧还是正前方？",
            "目的地在你的左边、右边还是前面？",
        ),
        _draw_none, _ans_direction,
    ),
    Family(
        "door_state", frozenset({"voxels"}), DOOR_TASKS, _YESNO,
        (
            "Is the door open?",
            "Is the door currently open?",
            "Has the door been opened?",
            "Is the door open right now?",
        ),
        (
            "门是开着的吗？",
            "门现在开着吗？",
            "门打开了吗？",
            "这扇门当前是开着的吗？",
        ),
        _draw_none, _ans_door_state,
    ),
    Family(
        "lamp_state", frozenset({"voxels"}), LOGIC_TASKS, _YESNO,
        (
            "Is the lamp lit?",
            "Is the lamp currently on?",
            "Is the redstone lamp glowing?",
            "Is the indicator lamp lit right now?",
        ),
        (
            "灯亮着吗？",
            "红石灯现在亮吗？",
            "指示灯亮着吗？",
            "灯当前是亮的吗？",
        ),
        _draw_none, _ans_lamp_state,
    ),
    Family(
        "lever_combo", frozenset({"voxels"}), LOGIC_TASKS, _YESNO,
        (
            "If lever A is {a} and lever B is {b}, is the lamp lit?",
            "With lever A {a} and lever B {b}, would the lamp be lit?",
            "Suppose lever A is {a} and lever B is {b}: is the lamp on?",
            "When lever A is {a} and lever B is {b}, does the lamp glow?",
        ),
        (
            "如果拉杆A是{a_zh}，拉杆B是{b_zh}，灯亮吗？",
            "当拉杆A为{a_zh}、拉杆B为{b_zh}时，灯亮着吗？",
            "假设拉杆A是{a_zh}、拉杆B是{b_zh}，灯会亮吗？",
            "拉杆A为{a_zh}且拉杆B为{b_zh}时，灯是亮的吗？",
        ),
        _draw_combo, _ans_lever_combo,
    ),
    Family(
        "craftable", frozenset({"voxels"}), CRAFT_TASKS, _YESNO,
        (
            "Can you craft a stone pickaxe right now?",
            "Do you have everything needed to craft a stone pickaxe?",
            "Is a stone pickaxe craftable at this moment?",
            "Could you make a stone pickaxe immediately?",
        ),
        (
            "你现在能合成石镐吗？",
            "你现在具备合成石镐的条件吗？",
            "此刻能制作一把石镐吗？",
            "现在可以立刻合成石镐吗？",
        ),
        _draw_none, _ans_craftable,
    ),
    Family(
        "wall_dominant", frozenset({"rgb"}), ("pixel_gallery",), PALETTE_BLOCKS,
        (
            "What block covers most of the wall?",
            "Which block dominates the wall?",
            "What is the most common block on the wall?",
            "Which block takes up the largest area of the wall?",
        ),
        (
            "墙面上占比最多的是哪种方块？",
            "墙上最多的方块是什么？",
            "这面墙主要由什么方块组成？",
            "墙面覆盖面积最大的方块是哪种？",
        ),
        _draw_none, _ans_wall_dominant,
    ),
    Family(
        "wall_region", frozenset({"rgb"}), ("pixel_gallery",), _YESNO,
        (
            "Is there any {block} in the {region_en} of the wall?",
            "Does the {region_en} of the wall contain {block}?",
            "Can you see {block} on the {region_en} part of the wall?",
            "Is {block} present in the wall's {region_en}?",
        ),
        (
            "墙面上{region_zh}有{block_zh}吗？",
            "墙的{region_zh}部分能看到{block_zh}吗？",
            "{block_zh}在墙面的{region_zh}吗？",
            "墙面{region_zh}区域里有{block_zh}吗？",
        ),
        _draw_wall_region, _ans_wall_region,
    ),
]

FAMILY_BY_NAME = {f.name: f for f in FAMILIES}
