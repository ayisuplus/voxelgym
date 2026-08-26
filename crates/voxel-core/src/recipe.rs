//! Recipe loading (recipes.toml, parsed once at startup — a parse failure
//! panics with the toml error, which carries line/column) and the `craft`
//! action. Furnace state machine lives here too (smelting is recipe-adjacent).

use std::collections::HashMap;
use std::sync::LazyLock;

use serde::Deserialize;

use crate::block::*;
use crate::world::{Event, World};

pub const SMELT_TICKS: u32 = 200;
/// Fuel heat values: items smelted per unit.
pub const FUELS: &[(u16, u8)] = &[
    (ITEM_COAL, 8),
    (PLANKS, 1),
    (LOG, 1),
];
pub const TABLE_RANGE: i32 = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recipe {
    pub id: u8,
    pub name: String,
    pub inputs: Vec<(u16, u16)>,
    pub out: u16,
    pub count: u16,
    pub needs_table: bool,
}

#[derive(Deserialize)]
struct RawRecipes {
    recipe: Vec<RawRecipe>,
}

#[derive(Deserialize)]
struct RawRecipe {
    id: u8,
    name: String,
    kind: String,
    #[serde(default)]
    inputs: HashMap<String, u16>,
    #[serde(default)]
    pattern: Vec<String>,
    #[serde(default)]
    key: HashMap<String, String>,
    out: String,
    count: u16,
    #[serde(default)]
    needs_table: bool,
}

fn parse_recipes() -> Vec<Recipe> {
    let text = include_str!("../../../recipes.toml");
    let raw: RawRecipes = toml::from_str(text).expect("recipes.toml parse failed");
    let mut out = Vec::new();
    for r in raw.recipe {
        let mut inputs: Vec<(u16, u16)> = Vec::new();
        match r.kind.as_str() {
            "shapeless" => {
                for (name, n) in &r.inputs {
                    let id = item_id_by_name(name)
                        .unwrap_or_else(|| panic!("recipes.toml recipe '{}': unknown item '{name}'", r.name));
                    inputs.push((id, *n));
                }
            }
            "shaped" => {
                let mut counts: HashMap<String, u16> = HashMap::new();
                for row in &r.pattern {
                    for ch in row.chars() {
                        if ch == ' ' {
                            continue;
                        }
                        let name = r
                            .key
                            .get(&ch.to_string())
                            .unwrap_or_else(|| panic!("recipes.toml recipe '{}': pattern char '{ch}' not in key", r.name));
                        *counts.entry(name.clone()).or_insert(0) += 1;
                    }
                }
                for (name, n) in counts {
                    let id = item_id_by_name(&name)
                        .unwrap_or_else(|| panic!("recipes.toml recipe '{}': unknown item '{name}'", r.name));
                    inputs.push((id, n));
                }
            }
            other => panic!("recipes.toml recipe '{}': unknown kind '{other}'", r.name),
        }
        inputs.sort_unstable(); // canonical order for determinism of consume
        let out_id = item_id_by_name(&r.out)
            .unwrap_or_else(|| panic!("recipes.toml recipe '{}': unknown out item '{}'", r.name, r.out));
        out.push(Recipe {
            id: r.id,
            name: r.name,
            inputs,
            out: out_id,
            count: r.count,
            needs_table: r.needs_table,
        });
    }
    out.sort_by_key(|r| r.id);
    let max = out.last().map(|r| r.id).unwrap_or(0);
    assert!(max < 8, "craft action space is Discrete(8): ids must be < 8");
    out
}

static RECIPES: LazyLock<Vec<Recipe>> = LazyLock::new(parse_recipes);

pub fn recipes() -> &'static [Recipe] {
    &RECIPES
}

pub fn recipe_by_id(id: u8) -> Option<&'static Recipe> {
    recipes().iter().find(|r| r.id == id)
}

/// True if a crafting_table block is within TABLE_RANGE (Chebyshev) of the agent.
pub fn table_nearby(world: &mut World) -> bool {
    let p = world.agent.pos;
    let (cx, cy, cz) = (p[0].floor() as i32, p[1].floor() as i32, p[2].floor() as i32);
    for x in cx - TABLE_RANGE..=cx + TABLE_RANGE {
        for y in (cy - TABLE_RANGE).max(0)..=(cy + TABLE_RANGE).min(127) {
            for z in cz - TABLE_RANGE..=cz + TABLE_RANGE {
                if cell_id(world.get_block(x, y, z)) == CRAFTING_TABLE {
                    return true;
                }
            }
        }
    }
    false
}

/// The `craft` action. Returns true on success.
pub fn craft(world: &mut World, id: u8) -> bool {
    let Some(r) = recipe_by_id(id) else {
        return false;
    };
    if r.needs_table && !table_nearby(world) {
        return false;
    }
    for &(item, n) in &r.inputs {
        if world.agent.inventory.count(item) < n {
            return false;
        }
    }
    for &(item, n) in &r.inputs {
        let ok = world.agent.inventory.consume(item, n);
        debug_assert!(ok);
    }
    let left = world.agent.inventory.add(r.out, r.count);
    if left > 0 {
        let p = world.agent.pos;
        world.spawn_item(r.out, left, [p[0], p[1] + 1.0, p[2]]);
    }
    world.events.push(Event::Crafted {
        recipe: r.id,
        out: r.out,
        count: r.count,
    });
    true
}

/// Furnace block state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FurnaceState {
    /// Ticks left on the current smelt; 0 = not smelting.
    pub remaining: u32,
    /// Smelted product waiting to be collected.
    pub out_ready: bool,
    /// Remaining smelts covered by already-loaded fuel.
    pub fuel_left: u8,
}

/// `use` on a furnace, per contract state machine:
/// product ready -> collect; else if idle and have ore + fuel -> start; else noop.
pub fn use_furnace(world: &mut World, x: i32, y: i32, z: i32) {
    let st = world.furnaces.get(&(x, y, z)).copied().unwrap_or_default();
    if st.out_ready {
        let left = world.agent.inventory.add(ITEM_IRON_INGOT, 1);
        if left > 0 {
            let p = world.agent.pos;
            world.spawn_item(ITEM_IRON_INGOT, 1, [p[0], p[1] + 1.0, p[2]]);
        }
        world.events.push(Event::Smelted { item: ITEM_IRON_INGOT });
        world.furnaces.insert((x, y, z), FurnaceState { out_ready: false, ..st });
        return;
    }
    if st.remaining > 0 {
        return; // busy
    }
    if world.agent.inventory.count(IRON_ORE) == 0 {
        return;
    }
    let mut fuel_left = st.fuel_left;
    if fuel_left == 0 {
        let mut loaded = false;
        for &(fuel, heat) in FUELS {
            if world.agent.inventory.consume(fuel, 1) {
                fuel_left += heat;
                loaded = true;
                break;
            }
        }
        if !loaded {
            return;
        }
    }
    let ok = world.agent.inventory.consume(IRON_ORE, 1);
    debug_assert!(ok);
    fuel_left -= 1;
    world.furnaces.insert(
        (x, y, z),
        FurnaceState {
            remaining: SMELT_TICKS,
            out_ready: false,
            fuel_left,
        },
    );
}

/// Phase 3 (scheduled ticks): advance furnace smelting timers.
pub fn tick_furnaces(world: &mut World) {
    for st in world.furnaces.values_mut() {
        if st.remaining > 0 {
            st.remaining -= 1;
            if st.remaining == 0 {
                st.out_ready = true;
            }
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::worldgen::Preset;

    #[test]
    fn recipes_parse_and_match_contract() {
        let rs = recipes();
        assert_eq!(rs.len(), 7);
        assert_eq!(rs[0].name, "planks");
        assert_eq!(rs[0].inputs, vec![(LOG, 1)]);
        assert_eq!(rs[0].count, 4);
        assert!(!rs[0].needs_table);
        let wp = recipe_by_id(4).unwrap();
        assert_eq!(wp.inputs, vec![(PLANKS, 3), (ITEM_STICK, 2)]);
        assert!(wp.needs_table);
        let fur = recipe_by_id(7).unwrap();
        assert_eq!(fur.inputs, vec![(COBBLESTONE, 8)]);
        assert_eq!(fur.out, FURNACE);
    }

    #[test]
    fn craft_planks_and_table_gating() {
        let mut w = World::new(1, Preset::Flat, Vec::new());
        w.agent.inventory.add(LOG, 2);
        assert!(craft(&mut w, 1));
        assert_eq!(w.agent.inventory.count(PLANKS), 4); // one craft action = one recipe
        assert_eq!(w.agent.inventory.count(LOG), 1);

        // wooden pickaxe requires a table within 4 cells
        w.agent.inventory.add(PLANKS, 3);
        w.agent.inventory.add(ITEM_STICK, 2);
        assert!(!craft(&mut w, 4)); // no table
        let p = w.agent.pos;
        w.set_block(p[0] as i32 + 1, 5, p[2] as i32, CRAFTING_TABLE);
        assert!(craft(&mut w, 4));
        assert_eq!(w.agent.inventory.count(ITEM_WOODEN_PICKAXE), 1);
        assert_eq!(w.agent.inventory.count(PLANKS), 4); // 4 + 3 - 3 used
    }

    #[test]
    fn furnace_state_machine() {
        let mut w = World::new(1, Preset::Flat, Vec::new());
        w.set_block(2, 5, 2, FURNACE);
        // no ore -> noop
        use_furnace(&mut w, 2, 5, 2);
        assert!(!w.furnaces.contains_key(&(2, 5, 2)));
        // ore but no fuel -> noop
        w.agent.inventory.add(IRON_ORE, 2);
        use_furnace(&mut w, 2, 5, 2);
        assert!(!w.furnaces.contains_key(&(2, 5, 2)));
        // fuel + ore -> starts; 1 coal covers 8 smelts
        w.agent.inventory.add(ITEM_COAL, 1);
        use_furnace(&mut w, 2, 5, 2);
        let st = w.furnaces[&(2, 5, 2)];
        assert_eq!(st.remaining, SMELT_TICKS);
        assert_eq!(st.fuel_left, 7);
        assert_eq!(w.agent.inventory.count(IRON_ORE), 1);
        // busy -> noop
        use_furnace(&mut w, 2, 5, 2);
        for _ in 0..SMELT_TICKS {
            tick_furnaces(&mut w);
        }
        assert!(w.furnaces[&(2, 5, 2)].out_ready);
        // collect
        use_furnace(&mut w, 2, 5, 2);
        assert_eq!(w.agent.inventory.count(ITEM_IRON_INGOT), 1);
        // second smelt uses stored fuel (no new coal needed)
        use_furnace(&mut w, 2, 5, 2);
        assert_eq!(w.furnaces[&(2, 5, 2)].fuel_left, 6);
        assert_eq!(w.agent.inventory.count(IRON_ORE), 0);
    }
}
