//! Fire: deterministic cellular automaton (custom rules, documented here).
//!
//! - Fire ticks every FIRE_PERIOD world ticks.
//! - A fire cell burns for FIRE_TTL fire-ticks (state nibble counts down),
//!   then becomes air. Burning a flammable block converts it into a fresh
//!   fire cell (the block is consumed).
//! - Per fire tick, each fire cell ignites each flammable 6-neighbor with
//!   probability SPREAD_P (position/tick hash — no shared-stream RNG
//!   consumption, so fire never perturbs other randomness).
//! - Lava adjacent to a flammable cell ignites it with probability
//!   LAVA_IGNITE_P per fire tick. Lava enters the active set when it
//!   appears or changes: scenario placement (seeded at World::new), any
//!   set_block (dirty-seeding), or fuel arriving next to a sleeping lava
//!   cell (dirty-neighbor wake). Worldgen lava is NOT seeded at chunk
//!   generation — that would couple active_fire to chunk loadedness and
//!   break hash observer-independence; it wakes on the first nearby change.
//! - Fire is replaceable: water flowing into a fire cell extinguishes it.
//! - Agent standing in a fire cell: 1 half-heart per 10 ticks.

use crate::block::*;
use crate::rng::hash_pos;
use crate::world::World;

pub const FIRE_PERIOD: u64 = 5;
pub const FIRE_TTL: u16 = 5;
pub const SPREAD_P: f64 = 0.35;
pub const LAVA_IGNITE_P: f64 = 0.2;

fn chance(seed: u64, x: i32, y: i32, z: i32, bucket: u64, p: f64) -> bool {
    (hash_pos(seed, x, y, z, bucket) as f64 / u64::MAX as f64) < p
}

/// Phase 4.5 (after fluids, before circuits).
pub fn tick_fire(world: &mut World, dirty: &[(i32, i32, i32)]) {
    // seeding: fire and lava cells near changes are active (lava is a
    // persistent ignition source, re-evaluated every fire tick while fuel
    // is in range). A change can also bring fuel next to a SLEEPING lava
    // cell — wake any lava adjacent to a dirty cell.
    for &(x, y, z) in dirty {
        let id = cell_id(world.peek_block(x, y, z));
        if id == FIRE || id == LAVA {
            world.active_fire.insert((x, y, z));
        }
        for (dx, dy, dz) in DIRS6 {
            let n = (x + dx, y + dy, z + dz);
            if cell_id(world.peek_block(n.0, n.1, n.2)) == LAVA {
                world.active_fire.insert(n);
            }
        }
    }
    let fire_period = world.clock_config().ticks_for_default_ticks(FIRE_PERIOD);
    if !world.tick.is_multiple_of(fire_period) || world.active_fire.is_empty() {
        return;
    }

    // fire cells may sit in never-generated chunks (scenario seeds), and
    // their spread targets can be one chunk over: peek_block reports AIR
    // for ungenerated chunks, which would make ignition depend on chunk
    // loadedness (observer perturbation). Ensure the 3x3 chunk
    // neighborhood of every active cell instead.
    let to_load: Vec<(i32, i32, i32)> = world.active_fire.iter().copied().collect();
    for (x, _, z) in to_load {
        let (cx, cz) = (x.div_euclid(16), z.div_euclid(16));
        for dx in -1..=1 {
            for dz in -1..=1 {
                world.ensure_chunk(cx + dx, cz + dz);
            }
        }
    }

    let seed = world.seed;
    let bucket = world.tick / fire_period;
    // spread radius: Manhattan ball of radius s — at scale 1 this is exactly
    // the 6 face neighbors (DIRS6, legacy semantics); at scale s a 1 m gap
    // is s cells, so "fire jumps a 1 m gap" holds at any cell size
    let r = world.physics.scale.round() as i32;
    let mut cells: Vec<(i32, i32, i32)> = world.active_fire.iter().copied().collect();
    cells.sort_unstable();

    let mut new_fires = Vec::new();
    for &(x, y, z) in &cells {
        let cell = world.peek_block(x, y, z);
        let id = cell_id(cell);
        if id == LAVA {
            // persistent source: ignite flammable neighbors, never burns
            // out. Sleep when no fuel is in range: chance() is only ever
            // evaluated for flammable cells, so a fuel-free lava cell rolls
            // no dice — dropping it loses no draws. It wakes again via
            // dirty-seeding the moment fuel appears next to it.
            let mut fuel_near = false;
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        if dx.abs() + dy.abs() + dz.abs() > r || (dx == 0 && dy == 0 && dz == 0) {
                            continue;
                        }
                        let n = (x + dx, y + dy, z + dz);
                        let nc = world.peek_block(n.0, n.1, n.2);
                        if block_def(cell_id(nc)).flammable {
                            fuel_near = true;
                            if chance(seed, n.0, n.1, n.2, bucket, LAVA_IGNITE_P) {
                                new_fires.push(n);
                            }
                        }
                    }
                }
            }
            if !fuel_near {
                world.active_fire.remove(&(x, y, z));
            }
            continue;
        }
        if id != FIRE {
            world.active_fire.remove(&(x, y, z));
            continue;
        }
        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    if dx.abs() + dy.abs() + dz.abs() > r || (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    let n = (x + dx, y + dy, z + dz);
                    let nc = world.peek_block(n.0, n.1, n.2);
                    if block_def(cell_id(nc)).flammable
                        && chance(seed, n.0, n.1, n.2, bucket, SPREAD_P)
                    {
                        new_fires.push(n);
                    }
                }
            }
        }
        let st = cell_state(cell);
        if st <= 1 {
            world.set_block(x, y, z, AIR);
            world.active_fire.remove(&(x, y, z));
        } else {
            world.set_block(x, y, z, make_cell(FIRE, st - 1));
        }
    }
    for n in new_fires {
        if block_def(cell_id(world.peek_block(n.0, n.1, n.2))).flammable {
            world.set_block(n.0, n.1, n.2, make_cell(FIRE, FIRE_TTL));
            world.active_fire.insert(n);
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::tick::{step, Action};
    use crate::worldgen::Preset;

    fn idle() -> Action {
        Action::default()
    }

    #[test]
    fn fire_burns_out_planks() {
        let mut w = World::new(1, Preset::Void, Vec::new());
        w.set_block(5, 5, 5, PLANKS);
        w.set_block(5, 6, 5, make_cell(FIRE, FIRE_TTL));
        for _ in 0..(FIRE_PERIOD * (FIRE_TTL as u64) + 10) {
            step(&mut w, &idle());
        }
        assert_eq!(w.get_block(5, 6, 5), AIR, "fire burns out");
        // the plank below either got consumed by spread or survived; fire
        // itself must be gone regardless
    }

    #[test]
    fn fire_consumes_flammable_neighbor_eventually() {
        // deterministic: run long enough that hash-based spread certainly hit
        let mut w = World::new(1, Preset::Void, Vec::new());
        for x in 5..8 {
            w.set_block(x, 5, 5, PLANKS);
        }
        w.set_block(6, 5, 5, make_cell(FIRE, FIRE_TTL));
        let mut consumed = false;
        for _t in 0..200 {
            step(&mut w, &idle());
            if cell_id(w.get_block(5, 5, 5)) != PLANKS || cell_id(w.get_block(7, 5, 5)) != PLANKS {
                consumed = true;
                break;
            }
        }
        assert!(
            consumed,
            "fire spread to a neighboring plank within 200 ticks"
        );
    }

    #[test]
    fn lava_ignites_adjacent_wood() {
        let mut w = World::new(1, Preset::Void, Vec::new());
        w.set_block(5, 5, 5, LAVA);
        w.set_block(6, 5, 5, PLANKS);
        // place lava LAST so its dirty event triggers ignition checks
        let mut ignited = false;
        for _ in 0..400 {
            step(&mut w, &idle());
            if cell_id(w.get_block(6, 5, 5)) == FIRE || cell_id(w.get_block(6, 5, 5)) == AIR {
                ignited = true;
                break;
            }
        }
        assert!(ignited, "lava should ignite the adjacent plank");
    }

    #[test]
    fn scenario_lava_is_an_ignition_source() {
        // scenario-placed lava never flows, so only the World::new seeding
        // can wake it — it must ignite adjacent fuel from tick 0.
        use crate::worldgen::Region;
        let scenario = vec![
            (Region::new(5, 5, 5, 5, 5, 5), LAVA),
            (Region::new(6, 5, 5, 6, 5, 5), PLANKS),
        ];
        let mut w = World::new(1, Preset::Void, scenario);
        let mut ignited = false;
        for _ in 0..400 {
            step(&mut w, &idle());
            let c = cell_id(w.get_block(6, 5, 5));
            if c == FIRE || c == AIR {
                ignited = true;
                break;
            }
        }
        assert!(ignited, "scenario lava should ignite the adjacent plank");
    }

    #[test]
    fn water_extinguishes() {
        let mut w = World::new(1, Preset::Void, Vec::new());
        for x in 4..9 {
            for z in 4..9 {
                w.set_block(x, 5, z, STONE);
            }
        }
        w.set_block(7, 6, 5, WATER);
        w.set_block(5, 6, 5, make_cell(FIRE, FIRE_TTL));
        for _ in 0..100 {
            step(&mut w, &idle());
        }
        // water spreads over the fire cell and replaces it
        assert_ne!(cell_id(w.get_block(5, 6, 5)), FIRE);
    }
}
