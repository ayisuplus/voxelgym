//! Milestone hook points — the M2/M3 systems plug into the tick order here
//! without reordering `tick::step`.
//!
//! Tick phases (fixed): 1 agent action -> 2 entity integration -> 3 scheduled
//! block ticks -> 4 fluids -> 5 circuits -> 6 item pickup/despawn -> 7 obs.

use crate::block::*;
use crate::world::{Event, World};

/// Phases 2b..6 after the agent body integrated.
#[inline]
pub fn after_entities(world: &mut World) {
    // 2b. item physics
    crate::item::tick_items_physics(world);
    // 2c. falling-block physics
    crate::loose::tick_falling(world);
    // this tick's block changes drive phases 3-5; changes they produce are
    // visible to next tick's phases (1-tick propagation, deterministic)
    let dirty = std::mem::take(&mut world.dirty);
    // 3. scheduled ticks: loose-block support + furnace smelting
    crate::loose::schedule_support_checks(world, &dirty);
    crate::loose::convert_due_falls(world);
    crate::recipe::tick_furnaces(world);
    // 4. fluids
    crate::fluid::tick_fluids(world, &dirty);
    // 4.5. fire (after fluids so water-extinguish resolves same-tick)
    crate::fire::tick_fire(world, &dirty);
    // 5. circuits
    crate::circuit::tick_circuits(world, &dirty);
    // 5.5. TNT (primed by powered wire / fire / lava)
    crate::tnt::tick_tnt(world);
    // 6. item merge / pickup / despawn
    crate::item::tick_items_logic(world);
}

/// Drops when a block is broken. Drops require an adequate tool when the
/// registry demands one; blocks without a tool requirement always drop.
#[inline]
pub fn block_broken(world: &mut World, x: i32, y: i32, z: i32, cell: u16, proper_tool: bool) {
    let id = cell_id(cell);
    let def = block_def(id);
    let drops_ok = def.tool.is_none() || proper_tool;
    if drops_ok {
        if let Some((item, n)) = def.drops {
            world.spawn_item(item, n as u16, [x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5]);
        }
    }
    world.events.push(Event::BlockMined { id });
}

/// `use` on blocks not handled by the M1 toggle set (door/lever): furnace.
#[inline]
pub fn use_block(world: &mut World, x: i32, y: i32, z: i32, id: u16) {
    if id == FURNACE {
        crate::recipe::use_furnace(world, x, y, z);
    }
}

/// `craft` action.
#[inline]
pub fn craft(world: &mut World, recipe: u8) {
    crate::recipe::craft(world, recipe);
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::item::DESPAWN_TICKS;
    use crate::worldgen::Preset;

    #[test]
    fn breaking_blocks_enforces_tool_drops_and_always_emits_events() {
        let mut world = World::new(1, Preset::Void, Vec::new());

        block_broken(&mut world, 2, 3, 4, STONE, false);
        assert!(world.items.is_empty());
        assert_eq!(
            world.drain_events(),
            vec![Event::BlockMined { id: STONE }]
        );

        block_broken(&mut world, -2, 3, -4, STONE, true);
        assert_eq!(world.items.len(), 1);
        assert_eq!(world.items[0].item, COBBLESTONE);
        assert_eq!(world.items[0].count, 1);
        assert_eq!(world.items[0].pos, [-1.5, 3.5, -3.5]);
        assert_eq!(
            world.drain_events(),
            vec![Event::BlockMined { id: STONE }]
        );
    }

    #[test]
    fn use_and_craft_dispatch_to_recipe_systems() {
        let mut world = World::new(2, Preset::Void, Vec::new());
        world.agent.inventory.add(LOG, 1);

        craft(&mut world, 1);
        assert_eq!(world.agent.inventory.count(LOG), 0);
        assert_eq!(world.agent.inventory.count(PLANKS), 4);

        world.agent.inventory.add(IRON_ORE, 1);
        world.agent.inventory.add(ITEM_COAL, 1);
        use_block(&mut world, 8, 9, 10, DIRT);
        assert!(world.furnaces.is_empty());
        use_block(&mut world, 8, 9, 10, FURNACE);
        assert_eq!(
            world.furnaces[&(8, 9, 10)].remaining,
            crate::recipe::SMELT_TICKS
        );
    }

    #[test]
    fn after_entities_consumes_dirty_input_and_despawns_expired_items() {
        let mut world = World::new(3, Preset::Void, Vec::new());
        world.dirty.push((30, 30, 30));
        world.spawn_item(DIRT, 1, [0.5, 10.0, 0.5]);
        world.items[0].age = DESPAWN_TICKS;

        after_entities(&mut world);

        assert!(world.dirty.is_empty());
        assert!(world.items.is_empty());
    }
}
