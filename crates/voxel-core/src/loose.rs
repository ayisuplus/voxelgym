//! Loose blocks (sand/gravel): support checks, scheduled fall conversion,
//! falling-block entities.
//!
//! Rules (contract): on neighborhood change, a loose block with a
//! `replaceable` cell below converts to a falling entity 1 tick later.
//! Falling: vy = (vy - 0.04) * 0.98. Landing on a solid face restores the
//! block; landing on a non-solid, non-replaceable cell (torch/wire/...)
//! turns it into an item drop instead.

use crate::block::*;
use crate::entity::clip_axis;
use crate::world::World;

pub const FALL_HALF: f64 = 0.49;
pub const FALL_HEIGHT: f64 = 0.98;
pub const FALL_GRAVITY: f64 = 0.04;
pub const FALL_GRAVITY_MULT: f64 = 0.98;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FallingBlock {
    pub id: u64,
    pub block: u16,
    /// Center position.
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    /// Accumulated downward travel (for impact damage).
    pub fall_dist: f64,
}

/// Phase 3a: queue support checks for dirty cells and the cell above each
/// (a block removal undermines the block on top).
pub fn schedule_support_checks(world: &mut World, dirty: &[(i32, i32, i32)]) {
    for &(x, y, z) in dirty {
        for (cx, cy, cz) in [(x, y, z), (x, y + 1, z)] {
            let cell = world.get_block(cx, cy, cz);
            if !block_def(cell_id(cell)).loose {
                continue;
            }
            let below = world.get_block(cx, cy - 1, cz);
            if !block_def(cell_id(below)).replaceable {
                continue;
            }
            let due = world.tick + world.clock_config().ticks_for_default_ticks(1);
            // O(1) dedup via the side set (a collapse avalanche schedules
            // thousands of cells per tick; a linear scan here is O(n^2))
            if world.scheduled_set.insert((cx, cy, cz)) {
                world.scheduled_falls.push((cx, cy, cz, due));
            }
        }
    }
}

/// Phase 3b: convert due loose blocks into falling entities.
pub fn convert_due_falls(world: &mut World) {
    let tick = world.tick;
    let mut keep = Vec::with_capacity(world.scheduled_falls.len());
    let scheduled = std::mem::take(&mut world.scheduled_falls);
    for (x, y, z, due) in scheduled {
        if due > tick {
            keep.push((x, y, z, due));
            continue;
        }
        let cell = world.get_block(x, y, z);
        let id = cell_id(cell);
        if !block_def(id).loose {
            continue; // changed meanwhile
        }
        let below = world.get_block(x, y - 1, z);
        if !block_def(cell_id(below)).replaceable {
            continue; // supported meanwhile
        }
        world.set_block(x, y, z, AIR);
        let fid = world.next_falling_id;
        world.next_falling_id += 1;
        world.falling.push(FallingBlock {
            id: fid,
            block: id,
            pos: [x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5],
            vel: [0.0; 3],
            fall_dist: 0.0,
        });
    }
    world.scheduled_set = keep.iter().map(|&(x, y, z, _)| (x, y, z)).collect();
    world.scheduled_falls = keep;
}

/// Phase 2c: falling-block physics + landing conversion.
pub fn tick_falling(world: &mut World) {
    let step_ratio = world.clock_config().default_step_ratio();
    let mut falling = std::mem::take(&mut world.falling);
    let mut landed: Vec<(u64, u16, i32, i32, i32)> = Vec::new();
    for fb in falling.iter_mut() {
        let sc = world.physics.scale;
        let (fh, fht) = (FALL_HALF * sc, FALL_HEIGHT * sc);
        let mut min = [fb.pos[0] - fh, fb.pos[1] - fht / 2.0, fb.pos[2] - fh];
        let mut max = [fb.pos[0] + fh, fb.pos[1] + fht / 2.0, fb.pos[2] + fh];
        let vel = fb.vel;
        let requested_dy =
            crate::physics::gravity_step(vel[1], FALL_GRAVITY * sc, FALL_GRAVITY_MULT, step_ratio)
                .0;
        let dy = clip_axis(world, &mut min, &mut max, 1, requested_dy);
        let clipped_down = requested_dy < 0.0 && dy != requested_dy;
        if requested_dy < 0.0 {
            fb.fall_dist += -dy;
        }
        let dx = clip_axis(world, &mut min, &mut max, 0, vel[0]);
        if dx != vel[0] {
            fb.vel[0] = 0.0;
        }
        let dz = clip_axis(world, &mut min, &mut max, 2, vel[2]);
        if dz != vel[2] {
            fb.vel[2] = 0.0;
        }
        fb.pos = [
            (min[0] + max[0]) / 2.0,
            (min[1] + max[1]) / 2.0,
            (min[2] + max[2]) / 2.0,
        ];
        if clipped_down {
            // impact damage: block AABB vs agent AABB at the landing point
            let fall_dist_m = fb.fall_dist / sc;
            if fall_dist_m >= 2.0 {
                let agent_min = world.agent.aabb_min();
                let agent_max = world.agent.aabb_max();
                let aabb_overlap = min[0] < agent_max[0]
                    && max[0] > agent_min[0]
                    && min[1] < agent_max[1]
                    && max[1] > agent_min[1]
                    && min[2] < agent_max[2]
                    && max[2] > agent_min[2];
                if aabb_overlap {
                    // custom: floor(fall distance in meters) half-hearts
                    let dmg = fall_dist_m.floor() as i32;
                    world.agent.hp -= dmg;
                    if world.agent.hp <= 0 {
                        world.agent.hp = 0;
                        world.agent.dead = true;
                    }
                }
            }
            // bottom rests on a solid top at min[1]; landing cell is floor(min[1])
            let lx = fb.pos[0].floor() as i32;
            let ly = min[1].floor() as i32;
            let lz = fb.pos[2].floor() as i32;
            landed.push((fb.id, fb.block, lx, ly, lz));
            fb.vel = [0.0; 3];
            fb.pos[1] = -1.0e9; // mark dead; removed below
        } else {
            // gravity for next tick (loose-block constant, not entity 0.08;
            // spatial -> scales with the world)
            fb.vel[1] = crate::physics::gravity_step(
                fb.vel[1],
                FALL_GRAVITY * sc,
                FALL_GRAVITY_MULT,
                step_ratio,
            )
            .1;
            let term = -3.92 * sc;
            if fb.vel[1] < term {
                fb.vel[1] = term;
            }
        }
    }
    falling.retain(|fb| fb.pos[1] > -1.0e8);
    world.falling = falling;
    for (_, block, x, y, z) in landed {
        land(world, block, x, y, z);
    }
}

fn land(world: &mut World, block: u16, x: i32, y: i32, z: i32) {
    let cur = world.get_block(x, y, z);
    let def = block_def(cell_id(cur));
    if def.replaceable {
        world.set_block(x, y, z, make_cell(block, 0));
    } else if !def.solid {
        // torch/wire/lever/door etc: becomes an item drop
        if let Some((item, n)) = block_def(block).drops {
            world.spawn_item(
                item,
                n as u16,
                [x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5],
            );
        }
    } else {
        // should not happen (clip stopped at the first non-solid), but be
        // conservative: drop as item rather than overwrite a solid block
        if let Some((item, n)) = block_def(block).drops {
            world.spawn_item(
                item,
                n as u16,
                [
                    x as f64 + 0.5,
                    y as f64 + 0.5 + world.scale(),
                    z as f64 + 0.5,
                ],
            );
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::tick::{step, Action};
    use crate::worldgen::Preset;

    fn void_world() -> World {
        World::new(1, Preset::Void, Vec::new())
    }

    #[test]
    fn sand_falls_after_one_tick_and_lands() {
        let mut w = void_world();
        w.set_block(5, 5, 5, STONE); // platform
        w.set_block(5, 10, 5, SAND); // floating sand — unsupported
                                     // manually trigger neighbor change: set_block already marks dirty
        let idle = Action::default();
        step(&mut w, &idle); // schedule (due tick 1... world.tick increments)
        assert_eq!(
            w.get_block(5, 10, 5),
            SAND,
            "still a block on the scheduling tick"
        );
        step(&mut w, &idle); // conversion happens
        assert_eq!(w.get_block(5, 10, 5), AIR);
        assert_eq!(w.falling.len(), 1);
        for _ in 0..40 {
            step(&mut w, &idle);
        }
        assert_eq!(w.get_block(5, 6, 5), SAND, "landed on the platform");
        assert!(w.falling.is_empty());
    }

    #[test]
    fn sand_on_torch_becomes_item() {
        let mut w = void_world();
        w.set_block(5, 5, 5, STONE);
        w.set_block(5, 6, 5, TORCH);
        w.set_block(5, 10, 5, SAND);
        let idle = Action::default();
        for _ in 0..40 {
            step(&mut w, &idle);
        }
        assert_eq!(cell_id(w.get_block(5, 6, 5)), TORCH, "torch intact");
        assert_eq!(cell_id(w.get_block(5, 7, 5)), AIR);
        assert!(w.items.iter().any(|i| i.item == SAND), "sand drop spawned");
    }

    #[test]
    fn supported_sand_stays() {
        let mut w = void_world();
        w.set_block(5, 5, 5, STONE);
        w.set_block(5, 6, 5, SAND);
        let idle = Action::default();
        for _ in 0..5 {
            step(&mut w, &idle);
        }
        assert_eq!(w.get_block(5, 6, 5), SAND);
        assert!(w.falling.is_empty());
    }

    #[test]
    fn neighbor_change_undermines() {
        let mut w = void_world();
        w.set_block(5, 5, 5, STONE);
        w.set_block(5, 6, 5, SAND);
        w.set_block(5, 7, 5, SAND); // supported by the lower sand
        let idle = Action::default();
        step(&mut w, &idle);
        assert!(w.falling.is_empty());
        // break the support
        w.set_block(5, 6, 5, AIR);
        for _ in 0..30 {
            step(&mut w, &idle);
        }
        assert_eq!(
            w.get_block(5, 6, 5),
            SAND,
            "upper sand landed where support was"
        );
    }
}
