//! Static block registry — the single source of truth for world semantics.
//!
//! Cell encoding in chunks: `cell = (state << 12) | id`.
//!   - low 12 bits: block id (0..=22 MVP)
//!   - high 4 bits: per-cell state (fluid level 0..15, wire power 0..15,
//!     door open bit, lever on bit)
//!
//! Item ids: placeable blocks use their block id as item id (1..=22,
//! excluding air/water/lava which have no item form). Non-block items
//! start at 1000.

pub const STATE_SHIFT: u16 = 12;
pub const ID_MASK: u16 = 0x0FFF;

#[inline]
pub const fn cell_id(cell: u16) -> u16 {
    cell & ID_MASK
}

#[inline]
pub const fn cell_state(cell: u16) -> u16 {
    cell >> STATE_SHIFT
}

#[inline]
pub const fn make_cell(id: u16, state: u16) -> u16 {
    (state << STATE_SHIFT) | id
}

/// 6-connected neighborhood offsets (face neighbors). Shared by the
/// fluid/fire/circuit/TNT systems — one table, not four copies.
pub(crate) const DIRS6: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fluid {
    Water,
    Lava,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ToolClass {
    Pickaxe,
}

pub struct BlockDef {
    pub id: u16,
    pub name: &'static str,
    pub solid: bool,
    pub opaque: bool,
    /// Can be overwritten by placement (air, fluids).
    pub replaceable: bool,
    /// Falls when unsupported (sand, gravel).
    pub loose: bool,
    pub fluid: Option<Fluid>,
    /// Ticks to mine with bare hands (tool_mult=1). u32::MAX = unbreakable.
    pub hardness_ticks: u32,
    /// Required tool (class, min tier) for drops & 5x speed.
    pub tool: Option<(ToolClass, u8)>,
    /// Drop (item_id, count) when broken with adequate tool.
    pub drops: Option<(u16, u8)>,
    pub color: [u8; 3],
    /// Catches and is consumed by fire.
    pub flammable: bool,
}

pub const AIR: u16 = 0;
pub const BEDROCK: u16 = 1;
pub const STONE: u16 = 2;
pub const DIRT: u16 = 3;
pub const GRASS_BLOCK: u16 = 4;
pub const SAND: u16 = 5;
pub const GRAVEL: u16 = 6;
pub const WATER: u16 = 7;
pub const LAVA: u16 = 8;
pub const LOG: u16 = 9;
pub const LEAVES: u16 = 10;
pub const PLANKS: u16 = 11;
pub const CRAFTING_TABLE: u16 = 12;
pub const FURNACE: u16 = 13;
pub const TORCH: u16 = 14;
pub const GLASS: u16 = 15;
pub const COAL_ORE: u16 = 16;
pub const IRON_ORE: u16 = 17;
pub const DIAMOND_ORE: u16 = 18;
pub const COBBLESTONE: u16 = 19;
pub const DOOR: u16 = 20;
pub const WIRE: u16 = 21;
pub const LEVER: u16 = 22;
pub const FIRE: u16 = 23;
pub const PRESSURE_PLATE: u16 = 24;
pub const TNT: u16 = 25;
pub const RTORCH: u16 = 26;
pub const REPEATER: u16 = 27;
pub const LAMP: u16 = 28;

const fn b(
    id: u16,
    name: &'static str,
    solid: bool,
    opaque: bool,
    replaceable: bool,
    loose: bool,
    fluid: Option<Fluid>,
    hardness_ticks: u32,
    tool: Option<(ToolClass, u8)>,
    drops: Option<(u16, u8)>,
    color: [u8; 3],
    flammable: bool,
) -> BlockDef {
    BlockDef {
        id,
        name,
        solid,
        opaque,
        replaceable,
        loose,
        fluid,
        hardness_ticks,
        tool,
        drops,
        color,
        flammable,
    }
}

pub static BLOCKS: &[BlockDef] = &[
    b(AIR, "air", false, false, true, false, None, 0, None, None, [0, 0, 0], false),
    b(BEDROCK, "bedrock", true, true, false, false, None, u32::MAX, None, None, [0x33, 0x33, 0x33], false),
    b(STONE, "stone", true, true, false, false, None, 150, Some((ToolClass::Pickaxe, 1)), Some((COBBLESTONE, 1)), [0x7D, 0x7D, 0x7D], false),
    b(DIRT, "dirt", true, true, false, false, None, 15, None, Some((DIRT, 1)), [0x8A, 0x5F, 0x3C], false),
    b(GRASS_BLOCK, "grass_block", true, true, false, false, None, 15, None, Some((DIRT, 1)), [0x6F, 0xA6, 0x53], false),
    b(SAND, "sand", true, true, false, true, None, 15, None, Some((SAND, 1)), [0xDB, 0xD3, 0xA0], false),
    b(GRAVEL, "gravel", true, true, false, true, None, 15, None, Some((GRAVEL, 1)), [0x82, 0x7F, 0x7C], false),
    b(WATER, "water", false, false, true, false, Some(Fluid::Water), 0, None, None, [0x3F, 0x76, 0xE4], false),
    b(LAVA, "lava", false, false, true, false, Some(Fluid::Lava), 0, None, None, [0xE6, 0x5C, 0x00], false),
    b(LOG, "log", true, true, false, false, None, 45, None, Some((LOG, 1)), [0x6B, 0x52, 0x30], true),
    b(LEAVES, "leaves", true, false, false, false, None, 5, None, None, [0x3E, 0x89, 0x48], true),
    b(PLANKS, "planks", true, true, false, false, None, 30, None, Some((PLANKS, 1)), [0xA0, 0x80, 0x50], true),
    b(CRAFTING_TABLE, "crafting_table", true, true, false, false, None, 30, None, Some((CRAFTING_TABLE, 1)), [0x7A, 0x5C, 0x33], true),
    b(FURNACE, "furnace", true, true, false, false, None, 30, Some((ToolClass::Pickaxe, 1)), Some((FURNACE, 1)), [0x6E, 0x6E, 0x6E], false),
    b(TORCH, "torch", false, false, false, false, None, 5, None, Some((TORCH, 1)), [0xFF, 0xC7, 0x00], false),
    b(GLASS, "glass", true, false, false, false, None, 15, None, None, [0xC0, 0xE8, 0xF9], false),
    b(COAL_ORE, "coal_ore", true, true, false, false, None, 150, Some((ToolClass::Pickaxe, 1)), Some((ITEM_COAL, 1)), [0x5A, 0x5A, 0x5A], false),
    b(IRON_ORE, "iron_ore", true, true, false, false, None, 150, Some((ToolClass::Pickaxe, 2)), Some((IRON_ORE, 1)), [0xD8, 0xAF, 0x93], false),
    b(DIAMOND_ORE, "diamond_ore", true, true, false, false, None, 150, Some((ToolClass::Pickaxe, 3)), Some((ITEM_DIAMOND, 1)), [0x7F, 0xD6, 0xC2], false),
    b(COBBLESTONE, "cobblestone", true, true, false, false, None, 150, Some((ToolClass::Pickaxe, 1)), Some((COBBLESTONE, 1)), [0x7A, 0x7A, 0x7A], false),
    b(DOOR, "door", true, false, false, false, None, 10, None, Some((DOOR, 1)), [0x8B, 0x6B, 0x3D], true),
    b(WIRE, "wire", false, false, false, false, None, 10, None, Some((WIRE, 1)), [0xB0, 0x30, 0x30], false),
    b(LEVER, "lever", false, false, false, false, None, 10, None, Some((LEVER, 1)), [0x9A, 0x9A, 0x9A], false),
    b(FIRE, "fire", false, false, true, false, None, 0, None, None, [0xE2, 0x58, 0x22], false),
    b(PRESSURE_PLATE, "pressure_plate", false, false, false, false, None, 10, None, Some((PRESSURE_PLATE, 1)), [0xC8, 0xC8, 0xC8], false),
    b(TNT, "tnt", true, true, false, false, None, 10, None, Some((TNT, 1)), [0xDB, 0x2F, 0x0F], false),
    // redstone torch: NOT gate. Input = cell below; output = 4 horizontal
    // neighbors + the cell above. State bit0 = lit.
    b(RTORCH, "redstone_torch", false, false, false, false, None, 5, None, Some((RTORCH, 1)), [0xFF, 0x55, 0x00], false),
    // repeater: diode + unit delay + signal refresh to 15. State bits0-1 =
    // facing (0:+z, 1:-x, 2:-z, 3:+x — output side), bit2 = output on.
    b(REPEATER, "repeater", false, false, false, false, None, 10, None, Some((REPEATER, 1)), [0xB0, 0x60, 0x60], false),
    // lamp: pure sink; lit while any 6-neighbor is powered. State bit0.
    b(LAMP, "lamp", true, true, false, false, None, 30, None, Some((LAMP, 1)), [0xE0, 0xB0, 0x50], false),
];

// ---- Items ----

pub const ITEM_STICK: u16 = 1000;
pub const ITEM_WOODEN_PICKAXE: u16 = 1001;
pub const ITEM_STONE_PICKAXE: u16 = 1002;
pub const ITEM_IRON_PICKAXE: u16 = 1003;
pub const ITEM_COAL: u16 = 1004;
pub const ITEM_IRON_INGOT: u16 = 1005;
pub const ITEM_DIAMOND: u16 = 1006;

pub struct ItemDef {
    pub id: u16,
    pub name: &'static str,
    /// Tool capability granted when held: (class, tier).
    pub tool: Option<(ToolClass, u8)>,
}

pub static ITEMS: &[ItemDef] = &[
    ItemDef { id: ITEM_STICK, name: "stick", tool: None },
    ItemDef { id: ITEM_WOODEN_PICKAXE, name: "wooden_pickaxe", tool: Some((ToolClass::Pickaxe, 1)) },
    ItemDef { id: ITEM_STONE_PICKAXE, name: "stone_pickaxe", tool: Some((ToolClass::Pickaxe, 2)) },
    ItemDef { id: ITEM_IRON_PICKAXE, name: "iron_pickaxe", tool: Some((ToolClass::Pickaxe, 3)) },
    ItemDef { id: ITEM_COAL, name: "coal", tool: None },
    ItemDef { id: ITEM_IRON_INGOT, name: "iron_ingot", tool: None },
    ItemDef { id: ITEM_DIAMOND, name: "diamond", tool: None },
];

pub const MAX_STACK: u16 = 64;

#[inline]
pub fn block_def(id: u16) -> &'static BlockDef {
    &BLOCKS[id as usize]
}

pub fn block_id_by_name(name: &str) -> Option<u16> {
    BLOCKS.iter().find(|d| d.name == name).map(|d| d.id)
}

pub fn item_name(id: u16) -> Option<&'static str> {
    if id < 1000 {
        return BLOCKS.get(id as usize).map(|d| d.name);
    }
    ITEMS.iter().find(|d| d.id == id).map(|d| d.name)
}

pub fn item_id_by_name(name: &str) -> Option<u16> {
    if let Some(b) = BLOCKS.iter().find(|d| d.name == name) {
        if b.id != AIR && b.fluid.is_none() {
            return Some(b.id);
        }
        return None;
    }
    ITEMS.iter().find(|d| d.name == name).map(|d| d.id)
}

/// Tool capability of an item (None if not a tool).
pub fn item_tool(id: u16) -> Option<(ToolClass, u8)> {
    if id < 1000 {
        return None;
    }
    ITEMS.iter().find(|d| d.id == id).and_then(|d| d.tool)
}

/// Whether an item id can be placed as a block (block item forms).
pub fn is_placeable(item: u16) -> bool {
    item != AIR && item < 1000 && block_def(item).fluid.is_none()
}

/// Whether a cell stops a targeting/render ray. MVP: everything except air
/// (water/lava block rays per contract; glass blocks rays — no transparency).
#[inline]
pub fn blocks_ray(cell: u16) -> bool {
    cell_id(cell) != AIR
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn ids_unique() {
        let ids: HashSet<u16> = BLOCKS.iter().map(|d| d.id).collect();
        assert_eq!(ids.len(), BLOCKS.len());
        // contiguous from 0 so block_def() indexing is valid
        for (i, d) in BLOCKS.iter().enumerate() {
            assert_eq!(i as u16, d.id);
        }
    }

    #[test]
    fn names_unique() {
        let names: HashSet<&str> = BLOCKS.iter().map(|d| d.name).collect();
        assert_eq!(names.len(), BLOCKS.len());
        let inames: HashSet<&str> = ITEMS.iter().map(|d| d.name).collect();
        assert_eq!(inames.len(), ITEMS.len());
        assert!(names.is_disjoint(&inames));
    }

    #[test]
    fn drops_resolve() {
        for d in BLOCKS {
            if let Some((item, n)) = d.drops {
                assert!(n > 0);
                assert!(item_name(item).is_some(), "{} drops unknown item {}", d.name, item);
            }
        }
    }

    #[test]
    fn registry_matches_contract() {
        assert_eq!(BLOCKS.len(), 29);
        assert!(block_def(STONE).solid && block_def(STONE).opaque);
        assert!(block_def(AIR).replaceable);
        assert!(block_def(SAND).loose && block_def(GRAVEL).loose);
        assert!(!block_def(TORCH).solid);
        assert_eq!(block_def(GRASS_BLOCK).drops, Some((DIRT, 1)));
        assert_eq!(block_def(DIAMOND_ORE).tool, Some((ToolClass::Pickaxe, 3)));
        assert_eq!(block_def(BEDROCK).hardness_ticks, u32::MAX);
        assert_eq!(item_tool(ITEM_STONE_PICKAXE), Some((ToolClass::Pickaxe, 2)));
        assert!(is_placeable(DOOR));
        assert!(!is_placeable(WATER));
        assert!(!is_placeable(ITEM_STICK));
        assert!(block_def(LOG).flammable && block_def(PLANKS).flammable);
        assert!(!block_def(STONE).flammable && !block_def(FIRE).flammable);
        assert!(block_def(FIRE).replaceable);
    }
}
