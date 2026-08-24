//! Fluid cellular automata (M3).
//!
//! Levels: 0 = source; 1..7 = horizontal spread (water max 7, lava max 3);
//! 8..15 = falling variants (we use exactly 8). Water ticks every 5 ticks,
//! lava every 30. Each fluid tick fully recomputes every active cell:
//!   - source: unchanged
//!   - non-source: support = min(4-neighbor min effective level + 1,
//!     falling inflow from above = 8); no support -> drain to air
//!   - water with two horizontal source neighbors and solid below -> source
//! Expansion: down into replaceable -> falling (8); horizontal from level
//! L < max -> L+1; a falling cell with solid below spreads as level 1.
//! Contact reactions (expansion into the other fluid):
//!   - water into lava SOURCE  -> that cell becomes stone
//!   - lava into water         -> that cell becomes cobblestone
//!   - water into flowing lava -> blocked (contract specifies no reaction)
//! Determinism: cells processed in sorted order; changes collected then
//! applied per pass.

use crate::block::*;
use crate::world::{World, XSet};

pub const WATER_PERIOD: u64 = 5;
pub const LAVA_PERIOD: u64 = 30;
pub const WATER_MAX: u16 = 7;
pub const LAVA_MAX: u16 = 3;
pub const FALLING: u16 = 8;

const DIRS4: [(i32, i32, i32); 4] = [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)];

fn max_spread(world: &World, f: Fluid) -> u16 {
    match f {
        Fluid::Water => world.physics.water_spread,
        Fluid::Lava => world.physics.lava_spread,
    }
}

/// Effective spread support of a neighbor cell: its level if 1..7, 0 for
/// sources and for falling cells resting on solid ground; falling cells in
/// mid-air and non-fluid cells give no support.
fn eff_support(world: &World, x: i32, y: i32, z: i32, f: Fluid) -> Option<u16> {
    let cell = world.peek_block(x, y, z);
    let id = cell_id(cell);
    let def = block_def(id);
    if def.fluid != Some(f) {
        return None;
    }
    let st = cell_state(cell);
    if st == 0 {
        return Some(0);
    }
    if st < FALLING {
        return Some(st);
    }
    // falling variant: supports horizontally only when resting on solid
    let below = world.peek_block(x, y - 1, z);
    if block_def(cell_id(below)).solid {
        Some(0)
    } else {
        None
    }
}

/// Compute the new state for an existing fluid cell. None = drain.
fn recompute(world: &World, x: i32, y: i32, z: i32, f: Fluid, cur_state: u16) -> Option<u16> {
    if cur_state == 0 {
        return Some(0); // sources are stable
    }
    // falling inflow from above dominates
    let above = world.peek_block(x, y + 1, z);
    if cell_id(above) != AIR && block_def(cell_id(above)).fluid == Some(f) {
        return Some(FALLING);
    }
    // horizontal support
    let mut best: Option<u16> = None;
    for (dx, _, dz) in DIRS4 {
        if let Some(e) = eff_support(world, x + dx, y, z + dz, f) {
            let cand = e + 1;
            best = Some(best.map_or(cand, |b: u16| b.min(cand)));
        }
    }
    match best {
        Some(level) if level <= max_spread(world, f) => {
            // water source formation: two horizontal source neighbors + solid below
            if f == Fluid::Water && level >= 1 {
                let below = world.peek_block(x, y - 1, z);
                if block_def(cell_id(below)).solid {
                    let mut sources = 0;
                    for (dx, _, dz) in DIRS4 {
                        let n = world.peek_block(x + dx, y, z + dz);
                        if block_def(cell_id(n)).fluid == Some(f) && cell_state(n) == 0 {
                            sources += 1;
                        }
                    }
                    if sources >= 2 {
                        return Some(0);
                    }
                }
            }
            Some(level)
        }
        _ => None,
    }
}

pub fn tick_fluids(world: &mut World, dirty: &[(i32, i32, i32)]) {
    // seed active set from this tick's changes
    for &(x, y, z) in dirty {
        for (dx, dy, dz) in [(0, 0, 0)].into_iter().chain(DIRS6) {
            let c = (x + dx, y + dy, z + dz);
            if block_def(cell_id(world.peek_block(c.0, c.1, c.2))).fluid.is_some() {
                world.active_fluids.insert(c);
            }
        }
    }
    let t = world.tick;
    let do_water = t % world.physics.water_period == 0;
    let do_lava = t % world.physics.lava_period == 0;
    if !do_water && !do_lava {
        return;
    }
    if world.active_fluids.is_empty() {
        return;
    }
    // fluid cells may sit in never-generated chunks
    let to_load: Vec<(i32, i32, i32)> = world.active_fluids.iter().copied().collect();
    for (x, _, z) in to_load {
        world.ensure_chunk(x.div_euclid(16), z.div_euclid(16));
    }

    let mut cells: Vec<(i32, i32, i32)> = world.active_fluids.iter().copied().collect();
    cells.sort_unstable();

    let mut changes: Vec<(i32, i32, i32, u16)> = Vec::new();
    let mut keep_active: XSet<(i32, i32, i32)> = XSet::default();

    // ---- pass 1: recompute existing fluid cells ----
    //
    // Wake/sleep discipline: a cell whose neighborhood did not change
    // recomputes to its current state by definition (recompute reads only
    // 6-neighbors, the cell below, and 4 horizontal sources), so stable
    // cells are dropped from the active set — equilibrium fluid bodies stop
    // consuming ticks. Reactivation is guaranteed by (a) dirty-seeding
    // above (any set_block in the neighborhood reinserts the cell next
    // tick) and (b) 6-neighbor insertion whenever a cell changes or drains.
    for &(x, y, z) in &cells {
        let cell = world.peek_block(x, y, z);
        let id = cell_id(cell);
        let Some(fluid) = block_def(id).fluid else {
            continue; // no longer fluid: drop from active set
        };
        if (fluid == Fluid::Water && !do_water) || (fluid == Fluid::Lava && !do_lava) {
            keep_active.insert((x, y, z));
            continue;
        }
        let st = cell_state(cell);
        match recompute(world, x, y, z, fluid, st) {
            Some(new_st) => {
                if new_st != st {
                    changes.push((x, y, z, make_cell(id, new_st)));
                    keep_active.insert((x, y, z));
                    for (dx, dy, dz) in DIRS6 {
                        keep_active.insert((x + dx, y + dy, z + dz));
                    }
                }
                // else: stable — sleep until a neighbor change wakes us.
            }
            None => {
                changes.push((x, y, z, AIR));
                keep_active.insert((x, y, z));
                for (dx, dy, dz) in DIRS6 {
                    keep_active.insert((x + dx, y + dy, z + dz));
                }
            }
        }
    }
    for &(x, y, z, cell) in &changes {
        world.set_block(x, y, z, cell);
    }

    // ---- pass 2: expansion into replaceable / reaction cells ----
    let mut creations: Vec<(i32, i32, i32, u16)> = Vec::new();
    for &(x, y, z) in &cells {
        let cell = world.peek_block(x, y, z);
        let id = cell_id(cell);
        let Some(fluid) = block_def(id).fluid else {
            continue;
        };
        if (fluid == Fluid::Water && !do_water) || (fluid == Fluid::Lava && !do_lava) {
            continue;
        }
        let st = cell_state(cell);
        let fid = id;

        // downward flow
        let below = world.peek_block(x, y - 1, z);
        let bid = cell_id(below);
        if y > 0 && can_flow_into(bid) && block_def(bid).fluid != Some(fluid) {
            if let Some(nc) = reaction_or_cell(below, fluid, fid, FALLING) {
                creations.push((x, y - 1, z, nc));
                keep_active.insert((x, y - 1, z));
            }
        }

        // horizontal spread
        let resting_falling = st >= FALLING && {
            let b = world.peek_block(x, y - 1, z);
            block_def(cell_id(b)).solid
        };
        let spread_level = if st < max_spread(world, fluid) {
            Some(st + 1)
        } else if resting_falling {
            Some(1)
        } else {
            None
        };
        if let Some(level) = spread_level {
            for (dx, _, dz) in DIRS4 {
                let (nx, ny, nz) = (x + dx, y, z + dz);
                let ncell = world.peek_block(nx, ny, nz);
                let nid = cell_id(ncell);
                if !can_flow_into(nid) {
                    continue;
                }
                // same fluid: levels are owned by the recompute pass
                if block_def(nid).fluid == Some(fluid) {
                    continue;
                }
                if let Some(nc) = reaction_or_cell(ncell, fluid, fid, level) {
                    creations.push((nx, ny, nz, nc));
                    keep_active.insert((nx, ny, nz));
                }
            }
        }
    }
    // deterministic: sorted, later duplicates overwrite earlier (same rule
    // application order every run)
    creations.sort_unstable();
    creations.dedup_by_key(|c| (c.0, c.1, c.2));
    for &(x, y, z, cell) in &creations {
        world.set_block(x, y, z, cell);
        keep_active.insert((x, y, z));
    }

    // prune: cells that are neither fluid nor adjacent to fluid fall out
    let mut final_active: XSet<(i32, i32, i32)> = XSet::default();
    for c in keep_active {
        let cell = world.peek_block(c.0, c.1, c.2);
        if block_def(cell_id(cell)).fluid.is_some() {
            final_active.insert(c);
        }
    }
    world.active_fluids = final_active;
}

/// Can a fluid expand INTO this cell? Air is replaceable; the other fluid is
/// reaction-eligible. Torch/wire/etc. are NOT replaceable (simplification:
/// fluids don't break attachments).
fn can_flow_into(id: u16) -> bool {
    let def = block_def(id);
    def.replaceable || def.fluid.is_some()
}

/// Expansion target resolution. `cur` = current cell content, `f`/`fid` the
/// incoming fluid, `level` the level it would arrive at. Returns the new
/// cell value, or None if blocked.
fn reaction_or_cell(cur: u16, f: Fluid, fid: u16, level: u16) -> Option<u16> {
    let id = cell_id(cur);
    let def = block_def(id);
    match def.fluid {
        None => {
            if def.replaceable {
                Some(make_cell(fid, level))
            } else {
                None
            }
        }
        Some(other) if other == f => Some(make_cell(fid, level)), // same fluid: level update
        Some(Fluid::Lava) => {
            // water expanding into lava
            if cell_state(cur) == 0 {
                Some(STONE) // water into lava source -> stone
            } else {
                None // into flowing lava: blocked
            }
        }
        Some(Fluid::Water) => {
            // lava expanding into water -> cobblestone at that cell
            Some(COBBLESTONE)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tick::{step, Action};
    use crate::worldgen::Preset;

    fn idle() -> Action {
        Action::default()
    }

    /// Void world with a stone floor and a water source; returns the world.
    fn water_lab() -> World {
        let mut w = World::new(1, Preset::Void, Vec::new());
        for x in 0..20 {
            for z in 0..20 {
                w.set_block(x, 5, z, STONE);
            }
        }
        w
    }

    #[test]
    fn water_spreads_exactly_seven() {
        let mut w = water_lab();
        w.set_block(10, 6, 10, WATER); // source
        for _ in 0..200 {
            step(&mut w, &idle());
        }
        // max horizontal reach: cells at distance d have level d, up to 7
        assert_eq!(cell_state(w.get_block(17, 6, 10)), 7);
        assert_eq!(cell_id(w.get_block(17, 6, 10)), WATER);
        assert_eq!(cell_id(w.get_block(18, 6, 10)), AIR, "8th cell must be air");
        // diagonal reach: |dx|+|dz| = 7
        assert_eq!(cell_id(w.get_block(14, 6, 13)), WATER);
        assert_eq!(cell_state(w.get_block(14, 6, 13)), 7);
    }

    #[test]
    fn falling_variant_and_landing_spread() {
        let mut w = World::new(1, Preset::Void, Vec::new());
        // pillar gap: platform at y=5 with a hole at (10,10); ledge below
        for x in 8..=12 {
            for z in 8..=12 {
                if (x, z) != (10, 10) {
                    w.set_block(x, 10, z, STONE);
                }
            }
        }
        w.set_block(10, 5, 10, STONE); // landing pad
        w.set_block(10, 11, 10, WATER);
        for _ in 0..200 {
            step(&mut w, &idle());
        }
        // falling column through the hole
        assert_eq!(cell_id(w.get_block(10, 9, 10)), WATER);
        assert_eq!(cell_state(w.get_block(10, 9, 10)), FALLING);
        assert_eq!(cell_state(w.get_block(10, 6, 10)), FALLING);
        // resting on the pad: spreads horizontally at level 1
        assert_eq!(cell_id(w.get_block(11, 6, 10)), WATER);
        assert_eq!(cell_state(w.get_block(11, 6, 10)), 1);
    }

    #[test]
    fn two_sources_form_new_source() {
        let mut w = water_lab();
        w.set_block(10, 6, 10, WATER);
        w.set_block(12, 6, 10, WATER);
        for _ in 0..100 {
            step(&mut w, &idle());
        }
        assert_eq!(cell_id(w.get_block(11, 6, 10)), WATER);
        assert_eq!(cell_state(w.get_block(11, 6, 10)), 0, "merged into a source");
    }

    #[test]
    fn water_drains_without_supply() {
        let mut w = water_lab();
        w.set_block(10, 6, 10, WATER);
        for _ in 0..100 {
            step(&mut w, &idle());
        }
        assert_eq!(cell_id(w.get_block(13, 6, 10)), WATER);
        // remove the source: everything drains
        w.set_block(10, 6, 10, AIR);
        for _ in 0..200 {
            step(&mut w, &idle());
        }
        for d in 1..=7 {
            assert_eq!(cell_id(w.get_block(10 + d, 6, 10)), AIR, "level {d} should drain");
        }
    }

    #[test]
    fn water_into_lava_source_makes_stone() {
        let mut w = water_lab();
        w.set_block(10, 6, 10, LAVA); // lava source
        w.set_block(12, 6, 10, WATER);
        for _ in 0..200 {
            step(&mut w, &idle());
        }
        assert_eq!(w.get_block(10, 6, 10), STONE, "water reached lava source -> stone");
    }

    #[test]
    fn lava_into_water_makes_cobblestone() {
        let mut w = water_lab();
        // contain the water so it can't run away: basin walls
        w.set_block(10, 6, 10, WATER);
        w.set_block(12, 6, 10, LAVA);
        for _ in 0..400 {
            step(&mut w, &idle());
        }
        // lava spreads 3 max: reaches x=13..? lava from 12: covers 13,14,15(level<=3)
        // water spreads 7: from 10 covers 3..17 — they meet at 11:
        // whichever flows into the other: lava into water cell -> cobblestone
        // at the meeting cell; or water into lava source(12) -> stone at 12.
        let c11 = cell_id(w.get_block(11, 6, 10));
        let c12 = cell_id(w.get_block(12, 6, 10));
        assert!(
            c11 == COBBLESTONE || c12 == STONE || c11 == STONE || c12 == COBBLESTONE,
            "reaction happened: c11={c11} c12={c12}"
        );
    }

    #[test]
    fn lava_max_three() {
        let mut w = water_lab();
        w.set_block(10, 6, 10, LAVA);
        for _ in 0..600 {
            step(&mut w, &idle());
        }
        assert_eq!(cell_id(w.get_block(13, 6, 10)), LAVA);
        assert_eq!(cell_state(w.get_block(13, 6, 10)), 3);
        assert_eq!(cell_id(w.get_block(14, 6, 10)), AIR, "lava stops at 3");
    }
}
