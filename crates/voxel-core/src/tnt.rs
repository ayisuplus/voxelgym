//! TNT: primed by an adjacent powered wire / fire / lava, detonates 2 ticks
//! later. Blast: Chebyshev radius 2, destroys everything except bedrock,
//! chains other TNT with a 2-4 tick fuse (position hash). Agent damage
//! falls off with distance; knockback is outward + upward. Custom
//! constants documented here (no MC exactness claim).

use crate::block::*;
use crate::rng::hash_pos;
use crate::world::World;

pub const BLAST_R: i32 = 2;
pub const FUSE_TICKS: u64 = 2;
/// max(0, floor(14 - 2.3 * dist)) half-hearts, dist in meters
pub const DMG_A: f64 = 14.0;
pub const DMG_B: f64 = 2.3;
pub const KNOCKBACK: f64 = 0.8;

/// Phase 5.5 (after circuits).
pub fn tick_tnt(world: &mut World) {
    // prime live TNT adjacent to a trigger
    if !world.tnt_cells.is_empty() {
        let mut cells: Vec<(i32, i32, i32)> = world.tnt_cells.iter().copied().collect();
        cells.sort_unstable();
        for c in cells {
            let mut primed = false;
            for (dx, dy, dz) in DIRS6 {
                let n = world.peek_block(c.0 + dx, c.1 + dy, c.2 + dz);
                let id = cell_id(n);
                if id == FIRE || id == LAVA || (id == WIRE && cell_state(n) > 0) {
                    primed = true;
                    break;
                }
            }
            if primed {
                world.tnt_cells.remove(&c);
                let fuse = world.clock_config().ticks_for_default_ticks(FUSE_TICKS);
                world.pending_booms.push((c.0, c.1, c.2, world.tick + fuse));
            }
        }
    }

    let tick = world.tick;
    let mut fire_now: Vec<(i32, i32, i32)> = Vec::new();
    let mut still: Vec<(i32, i32, i32, u64)> = Vec::new();
    for &(x, y, z, d) in &world.pending_booms {
        if d <= tick {
            fire_now.push((x, y, z));
        } else {
            still.push((x, y, z, d));
        }
    }
    world.pending_booms = still;
    fire_now.sort_unstable();
    for (x, y, z) in fire_now {
        world.set_block(x, y, z, AIR);
        explode(world, x, y, z);
    }
}

fn explode(world: &mut World, bx: i32, by: i32, bz: i32) {
    let tick = world.tick;
    let s = world.physics.scale;
    let blast_r = (BLAST_R as f64 * s).round() as i32;
    let ymax = world.height() - 1;
    // Load every chunk the blast touches first: destruction must not depend
    // on whether an observer happened to generate a chunk (peek_block
    // reports AIR for ungenerated chunks, which would shield their blocks).
    for cx in (bx - blast_r).div_euclid(16)..=(bx + blast_r).div_euclid(16) {
        for cz in (bz - blast_r).div_euclid(16)..=(bz + blast_r).div_euclid(16) {
            world.ensure_chunk(cx, cz);
        }
    }
    for dx in -blast_r..=blast_r {
        for dy in -blast_r..=blast_r {
            for dz in -blast_r..=blast_r {
                let (x, y, z) = (bx + dx, by + dy, bz + dz);
                if y < 0 || y > ymax {
                    continue;
                }
                let cell = world.peek_block(x, y, z);
                let id = cell_id(cell);
                if id == BEDROCK || id == AIR {
                    continue;
                }
                if id == TNT && world.tnt_cells.contains(&(x, y, z)) {
                    // chain: 2..4 tick fuse, position-hash derived
                    let extra = hash_pos(world.seed, x, y, z, tick) % 3;
                    world.tnt_cells.remove(&(x, y, z));
                    let fuse = world
                        .clock_config()
                        .ticks_for_default_ticks(FUSE_TICKS + extra);
                    world.pending_booms.push((x, y, z, tick + fuse));
                }
                world.set_block(x, y, z, AIR);
            }
        }
    }
    // agent damage + knockback
    let p = world.agent.pos;
    let (ax, ay, az) = (p[0], p[1] + world.agent.height / 2.0, p[2]);
    let (dx, dy, dz) = (
        ax - (bx as f64 + 0.5),
        ay - (by as f64 + 0.5),
        az - (bz as f64 + 0.5),
    );
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    // damage formula is written in meters: convert cell distance back
    let dist_m = dist / s;
    if dist_m < 5.0 && dist > 1e-6 {
        let dmg = (DMG_A - DMG_B * dist_m).floor().max(0.0) as i32;
        if dmg > 0 {
            world.agent.hp -= dmg;
            if world.agent.hp <= 0 {
                world.agent.hp = 0;
                world.agent.dead = true;
            }
        }
        let step_ratio = world.clock_config().default_step_ratio();
        let intensity = 1.0 - dist_m / 5.0;
        let horizontal_k = KNOCKBACK * s * step_ratio * intensity;
        // Horizontal state stores displacement for the configured step;
        // vertical state remains the canonical 20 Hz affine-recurrence value
        // and is fractionally integrated by gravity_step.
        let vertical_k = KNOCKBACK * s * intensity;
        world.agent.vel[0] += dx / dist * horizontal_k;
        world.agent.vel[1] += dy / dist * vertical_k + 0.25 * s * intensity;
        world.agent.vel[2] += dz / dist * horizontal_k;
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::clock::ClockConfig;
    use crate::tick::{step, Action};
    use crate::worldgen::Preset;

    fn idle() -> Action {
        Action::default()
    }

    #[test]
    fn wired_tnt_blasts() {
        let mut w = World::new(1, Preset::Flat, Vec::new());
        // lever(on) -> wire -> tnt; dirt pillar nearby; marker block far away
        w.set_block(2, 6, 2, make_cell(LEVER, 1));
        w.set_block(3, 6, 2, WIRE);
        w.set_block(4, 6, 2, TNT);
        w.set_block(5, 6, 2, DIRT);
        w.set_block(5, 7, 2, DIRT);
        w.set_block(0, 6, 2, STONE); // 4 cells from the blast: outside radius
        for _ in 0..10 {
            step(&mut w, &idle());
        }
        assert_eq!(w.get_block(4, 6, 2), AIR, "tnt consumed");
        assert_eq!(w.get_block(5, 6, 2), AIR, "dirt destroyed");
        assert_eq!(w.get_block(5, 7, 2), AIR);
        assert_eq!(w.get_block(0, 6, 2), STONE, "out-of-radius block survives");
        assert_eq!(
            cell_id(w.get_block(2, 6, 2)),
            AIR,
            "lever is inside r=2 and is destroyed"
        );
    }

    #[test]
    fn chain_reaction() {
        let mut w = World::new(1, Preset::Flat, Vec::new());
        w.set_block(2, 6, 2, make_cell(LEVER, 1));
        w.set_block(3, 6, 2, WIRE);
        w.set_block(4, 6, 2, TNT);
        w.set_block(6, 6, 2, TNT); // inside blast radius of the first
        for _ in 0..4 {
            step(&mut w, &idle());
        }
        // first detonated; second primed (fuse 2..4) — gone within 10 ticks
        for _ in 0..10 {
            step(&mut w, &idle());
        }
        assert_eq!(w.get_block(6, 6, 2), AIR, "chained tnt detonated");
    }

    #[test]
    fn agent_takes_blast_damage() {
        let mut w = World::new(1, Preset::Flat, Vec::new());
        w.set_block(5, 6, 5, LAVA);
        w.set_block(5, 6, 6, TNT);
        w.agent.pos = [5.5, 5.0, 7.5]; // ~1.7 cells from the blast
        w.agent.on_ground = true;
        let hp0 = w.agent.hp;
        for _ in 0..10 {
            step(&mut w, &idle());
        }
        assert!(w.agent.hp < hp0, "blast damaged the agent");
    }

    #[test]
    fn knockback_has_the_same_physical_impulse_at_twenty_and_forty_hz() {
        let mut w20 = World::new_with_clock(3, Preset::Void, Vec::new(), ClockConfig::default());
        let mut w40 = World::new_with_clock(
            3,
            Preset::Void,
            Vec::new(),
            ClockConfig::new(1, 40).unwrap(),
        );
        w20.agent.pos = [2.5, 5.0, 0.5];
        w40.agent.pos = w20.agent.pos;

        explode(&mut w20, 0, 5, 0);
        explode(&mut w40, 0, 5, 0);

        assert_eq!(w20.agent.hp, w40.agent.hp);
        assert!((w20.agent.vel[0] / 0.05 - w40.agent.vel[0] / 0.025).abs() < 1e-12);
        assert!((w20.agent.vel[2] / 0.05 - w40.agent.vel[2] / 0.025).abs() < 1e-12);
        assert!((w20.agent.vel[1] - w40.agent.vel[1]).abs() < 1e-12);
    }
}
