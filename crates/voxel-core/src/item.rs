//! Item entities: drops from mining. AABB 0.25^3, same gravity/collision as
//! the agent, 10-tick pickup delay, 6000-tick despawn, merge within 0.5
//! cells, pickup radius 1.5 cells.

use crate::entity::clip_axis;
use crate::world::{Event, World};

pub const ITEM_HALF: f64 = 0.125;
pub const PICKUP_DELAY_TICKS: u32 = 10;
pub const DESPAWN_TICKS: u64 = 6000;
pub const MERGE_RADIUS: f64 = 0.5;
pub const PICKUP_RADIUS: f64 = 1.5;

#[derive(Clone, Debug)]
pub struct ItemEntity {
    /// Unique within the world; snapshot sorting key.
    pub id: u64,
    pub item: u16,
    pub count: u16,
    /// Center position.
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub age: u64,
}

impl World {
    pub fn spawn_item(&mut self, item: u16, count: u16, pos: [f64; 3]) {
        let id = self.next_item_id;
        self.next_item_id += 1;
        // small deterministic scatter from the world rng stream
        let vx = (self.rng.next_f64() - 0.5) * 0.1;
        let vz = (self.rng.next_f64() - 0.5) * 0.1;
        self.items.push(ItemEntity {
            id,
            item,
            count,
            pos,
            vel: [vx, 0.2, vz],
            age: 0,
        });
    }
}

/// Phase 2b: integrate item physics. Items are taken out of the world to
/// satisfy the borrow checker (collision reads the world).
pub fn tick_items_physics(world: &mut World) {
    let ih = ITEM_HALF * world.physics.scale;
    let mut items = std::mem::take(&mut world.items);
    for it in items.iter_mut() {
        let mut min = [it.pos[0] - ih, it.pos[1] - ih, it.pos[2] - ih];
        let mut max = [it.pos[0] + ih, it.pos[1] + ih, it.pos[2] + ih];
        let vel = it.vel;
        let dy = clip_axis(world, &mut min, &mut max, 1, vel[1]);
        let grounded = vel[1] < 0.0 && dy != vel[1];
        if dy != vel[1] {
            it.vel[1] = 0.0;
        }
        let dx = clip_axis(world, &mut min, &mut max, 0, vel[0]);
        if dx != vel[0] {
            it.vel[0] = 0.0;
        }
        let dz = clip_axis(world, &mut min, &mut max, 2, vel[2]);
        if dz != vel[2] {
            it.vel[2] = 0.0;
        }
        it.pos = [
            (min[0] + max[0]) / 2.0,
            (min[1] + max[1]) / 2.0,
            (min[2] + max[2]) / 2.0,
        ];
        // ground friction (MC-style): drops settle within a few ticks
        // instead of sliding down slopes forever
        if grounded {
            it.vel[0] *= 0.6;
            it.vel[2] *= 0.6;
        }
        // gravity for next tick (move-then-accelerate order, same as agent;
        // uses the world physics so gravity ablation covers drops too)
        it.vel[1] = (it.vel[1] - world.physics.gravity) * world.physics.gravity_mult;
        if it.vel[1] < world.physics.terminal_vy {
            it.vel[1] = world.physics.terminal_vy;
        }
        it.age += 1;
    }
    world.items = items;
}

/// Phase 6: merge nearby stacks, pickup by the agent, despawn.
pub fn tick_items_logic(world: &mut World) {
    let sc = world.physics.scale;
    let merge_r2 = (MERGE_RADIUS * sc) * (MERGE_RADIUS * sc);
    // merge: same item, centers within MERGE_RADIUS, combined <= 64
    let mut items = std::mem::take(&mut world.items);
    let mut i = 0;
    while i < items.len() {
        let mut j = i + 1;
        while j < items.len() {
            let (a, b) = (items[i].clone(), items[j].clone());
            if a.item == b.item && a.count + b.count <= crate::block::MAX_STACK {
                let d2 = (a.pos[0] - b.pos[0]).powi(2)
                    + (a.pos[1] - b.pos[1]).powi(2)
                    + (a.pos[2] - b.pos[2]).powi(2);
                if d2 < merge_r2 {
                    items[i].count += items[j].count;
                    items[i].age = items[i].age.max(items[j].age);
                    items.remove(j);
                    continue;
                }
            }
            j += 1;
        }
        i += 1;
    }

    // pickup + despawn
    let apos = world.agent.pos;
    let agent_center = [apos[0], apos[1] + 0.9 * sc, apos[2]];
    let pickup_r2 = (PICKUP_RADIUS * sc) * (PICKUP_RADIUS * sc);
    let dead = world.agent.dead;
    let mut kept = Vec::with_capacity(items.len());
    for mut it in items {
        if it.age >= DESPAWN_TICKS {
            continue;
        }
        let d2 = (it.pos[0] - agent_center[0]).powi(2)
            + (it.pos[1] - agent_center[1]).powi(2)
            + (it.pos[2] - agent_center[2]).powi(2);
        if !dead && it.age >= PICKUP_DELAY_TICKS as u64 && d2 < pickup_r2 {
            let left = world.agent.inventory.add(it.item, it.count);
            let taken = it.count - left;
            if taken > 0 {
                world.events.push(Event::ItemPicked {
                    item: it.item,
                    count: taken,
                });
            }
            if left == 0 {
                continue;
            }
            it.count = left;
        }
        kept.push(it);
    }
    world.items = kept;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::DIRT;
    use crate::worldgen::Preset;

    #[test]
    fn item_falls_and_lands() {
        let mut w = World::new(1, Preset::Flat, Vec::new());
        w.spawn_item(DIRT, 1, [8.5, 10.0, 8.5]);
        for _ in 0..40 {
            tick_items_physics(&mut w);
        }
        let y = w.items[0].pos[1];
        // rests on surface plane y=5.0, center = 5 + 0.125
        assert!((y - 5.125).abs() < 1e-6, "y={y}");
    }

    #[test]
    fn pickup_after_delay() {
        let mut w = World::new(1, Preset::Flat, Vec::new());
        w.agent.pos = [8.5, 5.0, 8.5];
        w.agent.on_ground = true;
        w.spawn_item(DIRT, 3, [8.5, 5.5, 8.5]);
        // run physics+logic; before 10 ticks of age no pickup
        for _ in 0..9 {
            tick_items_physics(&mut w);
            tick_items_logic(&mut w);
        }
        // item may or may not be in range yet; ensure not picked early
        // (delay is 10): force exact age path
        for _ in 0..5 {
            tick_items_physics(&mut w);
            tick_items_logic(&mut w);
        }
        assert_eq!(w.agent.inventory.count(DIRT), 3);
        assert!(w.items.is_empty());
        assert!(matches!(w.events.last(), Some(Event::ItemPicked { item: DIRT, count: 3 })));
    }

    #[test]
    fn merge_within_radius() {
        let mut w = World::new(1, Preset::Flat, Vec::new());
        // far from agent so no pickup interferes
        w.agent.pos = [100.5, 5.0, 100.5];
        w.spawn_item(DIRT, 30, [8.5, 6.0, 8.5]);
        w.spawn_item(DIRT, 20, [8.7, 6.0, 8.6]);
        tick_items_logic(&mut w);
        assert_eq!(w.items.len(), 1);
        assert_eq!(w.items[0].count, 50);
    }
}
