"""Block/item id constants, resolved from the Rust registry (no drift)."""

import voxelgym_rs as rs

_BLOCKS = [
    "air", "bedrock", "stone", "dirt", "grass_block", "sand", "gravel",
    "water", "lava", "log", "leaves", "planks", "crafting_table", "furnace",
    "torch", "glass", "coal_ore", "iron_ore", "diamond_ore", "cobblestone",
    "door", "wire", "lever", "fire", "pressure_plate", "tnt",
    "redstone_torch", "repeater", "lamp",
]
_ITEMS = [
    "stick", "wooden_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "coal", "iron_ingot", "diamond",
]

_g = globals()
for _n in _BLOCKS:
    _g[_n.upper()] = rs.block_id(_n)
for _n in _ITEMS:
    _g["ITEM_" + _n.upper()] = rs.item_id(_n)

__all__ = [n.upper() for n in _BLOCKS] + ["ITEM_" + n.upper() for n in _ITEMS]
