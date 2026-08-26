//! Entity physics — Minecraft public constants, Y -> X -> Z axis-separated
//! collision resolution against the voxel grid.
//!
//! Agent AABB: 0.6 x 1.8 x 0.6 (pos = center of feet), eye height 1.62.
//! Movement model: velocity is accelerated toward wish_dir * target_speed
//! with per-tick accel clamp (ground 0.1, air 0.02). This is a deliberate
//! simplification of MC's friction-based model (no slipperiness table);
//! all constants live here for ablation.

use crate::block::*;
use crate::inventory::Inventory;
use crate::world::World;

pub const WALK_SPEED: f64 = 0.2159; // 4.317 m/s at 20 TPS
pub const SNEAK_MULT: f64 = 0.3;
pub const ACCEL_GROUND: f64 = 0.1;
pub const ACCEL_AIR: f64 = 0.02;
pub const JUMP_VY: f64 = 0.42;
pub const GRAVITY: f64 = 0.08;
pub const GRAVITY_MULT: f64 = 0.98;
pub const TERMINAL_VY: f64 = -3.92;
pub const HALF_WIDTH: f64 = 0.3;
pub const HEIGHT: f64 = 1.8;
pub const EYE_HEIGHT: f64 = 1.62;
pub const MAX_HP: i32 = 20; // half-hearts
/// Water movement (custom simplification, see contract): buoyant drift.
pub const WATER_VY_MULT: f64 = 0.8;
pub const WATER_SINK: f64 = 0.02;
pub const WATER_SWIM_UP: f64 = 0.04;
pub const WATER_H_MULT: f64 = 0.5;

#[derive(Clone, Debug)]
pub struct Agent {
    /// Center of feet.
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    /// Degrees. yaw: 0 = +z (south), 90 = -x (west), MC convention.
    pub yaw: f32,
    /// Degrees, -90..90 (action buckets limit to -60..60).
    pub pitch: f32,
    pub on_ground: bool,
    pub hp: i32,
    pub fall_distance: f64,
    pub dead: bool,
    pub suffocation_timer: u32,
    /// Ticks since last lava damage (M3).
    pub lava_timer: u32,
    /// Ticks standing in fire.
    pub fire_timer: u32,
    pub inventory: Inventory,
    /// Selected hotbar slot 0..8.
    pub selected: usize,
    /// Body dims in cells: HALF_WIDTH/HEIGHT/EYE_HEIGHT * world scale.
    /// (Fields, not consts, so the same physics code runs at any scale.)
    pub half_width: f64,
    pub height: f64,
    pub eye_height: f64,
}

impl Agent {
    pub fn new(spawn: [f64; 3], scale: f64) -> Self {
        Agent {
            pos: spawn,
            vel: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            on_ground: false,
            hp: MAX_HP,
            fall_distance: 0.0,
            dead: false,
            suffocation_timer: 0,
            lava_timer: 0,
            fire_timer: 0,
            inventory: Inventory::new(),
            selected: 0,
            half_width: HALF_WIDTH * scale,
            height: HEIGHT * scale,
            eye_height: EYE_HEIGHT * scale,
        }
    }

    pub fn eye(&self) -> [f64; 3] {
        [self.pos[0], self.pos[1] + self.eye_height, self.pos[2]]
    }

    /// Unit look vector from yaw/pitch (MC convention).
    pub fn look(&self) -> [f64; 3] {
        let yaw = (self.yaw as f64).to_radians();
        let pitch = (self.pitch as f64).to_radians();
        [
            -yaw.sin() * pitch.cos(),
            -pitch.sin(),
            yaw.cos() * pitch.cos(),
        ]
    }

    /// Horizontal forward (movement basis).
    pub fn forward(&self) -> [f64; 3] {
        let yaw = (self.yaw as f64).to_radians();
        [-yaw.sin(), 0.0, yaw.cos()]
    }

    /// AABB min corner.
    pub fn aabb_min(&self) -> [f64; 3] {
        [
            self.pos[0] - self.half_width,
            self.pos[1],
            self.pos[2] - self.half_width,
        ]
    }

    pub fn aabb_max(&self) -> [f64; 3] {
        [
            self.pos[0] + self.half_width,
            self.pos[1] + self.height,
            self.pos[2] + self.half_width,
        ]
    }
}

/// AABB of an arbitrary box entity (item 0.25^3, falling block ~1^3).
pub fn aabb_collides(world: &mut World, min: [f64; 3], max: [f64; 3]) -> bool {
    let x0 = min[0].floor() as i32;
    let y0 = min[1].floor() as i32;
    let z0 = min[2].floor() as i32;
    let x1 = (max[0] - 1e-9).floor() as i32;
    let y1 = (max[1] - 1e-9).floor() as i32;
    let z1 = (max[2] - 1e-9).floor() as i32;
    for y in y0..=y1 {
        for z in z0..=z1 {
            for x in x0..=x1 {
                if world.is_solid(x, y, z) {
                    return true;
                }
            }
        }
    }
    false
}

/// Move an AABB along one axis with clipping against solid cells.
/// Returns the actual displacement applied (<= requested).
pub(crate) fn clip_axis(world: &mut World, min: &mut [f64; 3], max: &mut [f64; 3], axis: usize, d: f64) -> f64 {
    if d == 0.0 {
        return 0.0;
    }
    let mut allowed = d;
    // Swept: find nearest solid boundary along the axis within the
    // slab swept by the box cross-section.
    let (a1, a2) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let lo0 = min[a1].floor() as i32;
    let hi0 = (max[a1] - 1e-9).floor() as i32;
    let lo1 = min[a2].floor() as i32;
    let hi1 = (max[a2] - 1e-9).floor() as i32;

    if d > 0.0 {
        let start = (max[axis] - 1e-9).floor() as i32;
        let end = (max[axis] + d).floor() as i32;
        'cells: for c in start..=end {
            for b in lo1..=hi1 {
                for a in lo0..=hi0 {
                    let (x, y, z) = match axis {
                        0 => (c, a, b),
                        1 => (a, c, b),
                        _ => (a, b, c),
                    };
                    if world.is_solid(x, y, z) {
                        let limit = c as f64 - max[axis] - 1e-7;
                        if limit < allowed {
                            allowed = limit.max(0.0);
                        }
                        continue 'cells;
                    }
                }
            }
        }
    } else {
        let start = (min[axis] + d).floor() as i32;
        let end = min[axis].floor() as i32;
        'cells: for c in (start..=end).rev() {
            for b in lo1..=hi1 {
                for a in lo0..=hi0 {
                    let (x, y, z) = match axis {
                        0 => (c, a, b),
                        1 => (a, c, b),
                        _ => (a, b, c),
                    };
                    if world.is_solid(x, y, z) {
                        let limit = (c + 1) as f64 - min[axis] + 1e-7;
                        if limit > allowed {
                            allowed = limit.min(0.0);
                        }
                        continue 'cells;
                    }
                }
            }
        }
    }
    min[axis] += allowed;
    max[axis] += allowed;
    allowed
}

pub struct MoveInput {
    /// Wish direction in world XZ (unit-ish, from move+yaw).
    pub wish_x: f64,
    pub wish_z: f64,
    pub jump: bool,
    pub sneak: bool,
}

/// One physics tick for the agent. Order: input accel -> integrate Y,X,Z.
pub fn tick_agent(world: &mut World, input: &MoveInput) {
    if world.agent.dead {
        return;
    }

    let feet_fluid = world.fluid_at_feet();
    let in_water = feet_fluid == Some(Fluid::Water);

    // --- horizontal propulsion as a FORCE model ---
    // The controller requests a propulsion force toward the wish velocity;
    // the force is limited by contact type: F_ground (friction-limited grip)
    // or F_air (weak air control). Acceleration = F / m (Newton). Defaults
    // (m=1, F_g=0.1, F_a=0.02) reproduce the MC constants exactly.
    let ph = world.physics;
    let speed = ph.walk_speed * if input.sneak { ph.sneak_mult } else { 1.0 } * if in_water { WATER_H_MULT } else { 1.0 };
    let force_limit = if world.agent.on_ground { ph.ground_force } else { ph.air_force };
    let accel = force_limit / ph.agent_mass;
    let tx = input.wish_x * speed;
    let tz = input.wish_z * speed;
    world.agent.vel[0] += (tx - world.agent.vel[0]).clamp(-accel, accel);
    world.agent.vel[2] += (tz - world.agent.vel[2]).clamp(-accel, accel);

    // --- vertical velocity setup: jump imparts vy; gravity applies AFTER
    // movement (MC order: travel with current velocity, then accelerate). ---
    if in_water {
        // fall distance is reset by water; no landing damage in water
        world.agent.fall_distance = 0.0;
    } else if input.jump && world.agent.on_ground {
        world.agent.vel[1] = ph.jump_vy;
        world.agent.on_ground = false;
    }

    // --- integrate with collision, Y -> X -> Z ---
    let mut min = world.agent.aabb_min();
    let mut max = world.agent.aabb_max();
    let vel = world.agent.vel;

    let dy = clip_axis(world, &mut min, &mut max, 1, vel[1]);
    if vel[1] < 0.0 {
        if !in_water {
            world.agent.fall_distance += -dy;
        }
        if dy != vel[1] {
            // landed
            world.agent.on_ground = true;
            if !in_water {
                let dist = world.agent.fall_distance;
                // epsilon: fall_distance accumulates per-tick float error
                // (a 10-cell fall sums to 9.9999...); 1e-6 restores intent
                let dmg = (dist - ph.fall_safe + 1e-6).floor().max(0.0) as i32;
                if dmg > 0 {
                    world.agent.hp -= dmg;
                    if world.agent.hp <= 0 {
                        world.agent.hp = 0;
                        world.agent.dead = true;
                    }
                }
            }
            world.agent.fall_distance = 0.0;
            world.agent.vel[1] = 0.0;
        } else {
            world.agent.on_ground = false;
        }
    } else if vel[1] > 0.0 {
        if dy != vel[1] {
            world.agent.vel[1] = 0.0; // bumped head
        }
        world.agent.on_ground = false;
    }

    let dx = clip_axis(world, &mut min, &mut max, 0, vel[0]);
    if dx != vel[0] {
        world.agent.vel[0] = 0.0;
    }
    let dz = clip_axis(world, &mut min, &mut max, 2, vel[2]);
    if dz != vel[2] {
        world.agent.vel[2] = 0.0;
    }

    world.agent.pos = [
        (min[0] + max[0]) / 2.0,
        min[1],
        (min[2] + max[2]) / 2.0,
    ];

    // --- gravity / buoyancy for the NEXT tick's displacement ---
    // Force law: F_g = -m*g (mass-independent accel g; Newtonian) plus a
    // linear drag applied to the post-gravity velocity (MC's (v-g)*0.98
    // form, kept verbatim for literature comparability).
    if in_water {
        let sc = ph.scale;
        let mut vy = world.agent.vel[1] * WATER_VY_MULT - WATER_SINK * sc;
        if input.jump {
            vy += WATER_SWIM_UP * sc;
        }
        world.agent.vel[1] = vy.clamp(ph.terminal_vy, 1.0 * sc);
    } else {
        world.agent.vel[1] = (world.agent.vel[1] - ph.gravity) * ph.gravity_mult;
        if world.agent.vel[1] < ph.terminal_vy {
            world.agent.vel[1] = ph.terminal_vy;
        }
    }

    // --- suffocation: head inside solid block, 1 half-heart per 20 ticks ---
    let eye = world.agent.eye();
    let head_solid = world.is_solid(
        eye[0].floor() as i32,
        eye[1].floor() as i32,
        eye[2].floor() as i32,
    );
    if head_solid {
        world.agent.suffocation_timer += 1;
        if world.agent.suffocation_timer % 20 == 0 {
            world.agent.hp -= world.physics.suffocate_damage;
            if world.agent.hp <= 0 {
                world.agent.hp = 0;
                world.agent.dead = true;
            }
        }
    } else {
        world.agent.suffocation_timer = 0;
    }

    // --- lava contact: lava_damage half-hearts per 10 ticks ---
    let eye_fluid = world.fluid_at(
        eye[0].floor() as i32,
        eye[1].floor() as i32,
        eye[2].floor() as i32,
    );
    let in_lava = feet_fluid == Some(Fluid::Lava) || eye_fluid == Some(Fluid::Lava);
    if in_lava {
        world.agent.lava_timer += 1;
        if world.agent.lava_timer % 10 == 0 {
            world.agent.hp -= world.physics.lava_damage;
            if world.agent.hp <= 0 {
                world.agent.hp = 0;
                world.agent.dead = true;
            }
        }
    } else {
        world.agent.lava_timer = 0;
    }

    // --- fire contact: 1 half-heart per 10 ticks standing in fire ---
    let feet_id = cell_id(world.get_block(
        world.agent.pos[0].floor() as i32,
        world.agent.pos[1].floor() as i32,
        world.agent.pos[2].floor() as i32,
    ));
    if feet_id == FIRE {
        world.agent.fire_timer += 1;
        if world.agent.fire_timer % 10 == 0 {
            world.agent.hp -= 1;
            if world.agent.hp <= 0 {
                world.agent.hp = 0;
                world.agent.dead = true;
            }
        }
    } else {
        world.agent.fire_timer = 0;
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::worldgen::Preset;

    fn flat_world() -> World {
        World::new(1, Preset::Flat, Vec::new())
    }

    fn idle() -> MoveInput {
        MoveInput { wish_x: 0.0, wish_z: 0.0, jump: false, sneak: false }
    }

    #[test]
    fn falls_and_lands_on_ground() {
        let mut w = flat_world();
        // flat surface grass at y=4, spawn feet y=5
        for _ in 0..10 {
            tick_agent(&mut w, &idle());
        }
        assert!(w.agent.on_ground);
        assert!((w.agent.pos[1] - 5.0).abs() < 1e-6);
        assert_eq!(w.agent.hp, MAX_HP);
    }

    #[test]
    fn ten_block_fall_damage() {
        let mut w = flat_world();
        w.agent.pos = [8.5, 15.0, 8.5]; // 10 blocks above feet-level 5
        while !w.agent.on_ground {
            tick_agent(&mut w, &idle());
        }
        assert_eq!(w.agent.hp, MAX_HP - 7, "fall of 10 -> floor(10-3)=7 half-hearts");
    }

    #[test]
    fn three_block_fall_no_damage() {
        let mut w = flat_world();
        w.agent.pos = [8.5, 8.0, 8.5]; // 3 blocks
        while !w.agent.on_ground {
            tick_agent(&mut w, &idle());
        }
        assert_eq!(w.agent.hp, MAX_HP);
    }

    #[test]
    fn wall_stops_motion() {
        let mut w = flat_world();
        // wall at x=10 column
        for y in 5..8 {
            for z in 0..16 {
                w.set_block(10, y, z, STONE);
            }
        }
        w.agent.pos = [8.5, 5.0, 8.5];
        let input = MoveInput { wish_x: 1.0, wish_z: 0.0, jump: false, sneak: false };
        for _ in 0..40 {
            tick_agent(&mut w, &input);
        }
        // must never penetrate the wall face at x=10 (AABB max x <= 10)
        assert!(w.agent.pos[0] + HALF_WIDTH <= 10.0 + 1e-6, "x={}", w.agent.pos[0]);
        assert!(w.agent.vel[0].abs() < 1e-9);
    }

    #[test]
    fn jump_apex_in_range() {
        let mut w = flat_world();
        w.agent.pos = [8.5, 5.0, 8.5];
        w.agent.on_ground = true;
        let input = MoveInput { wish_x: 0.0, wish_z: 0.0, jump: true, sneak: false };
        let mut apex = 5.0f64;
        for _ in 0..40 {
            tick_agent(&mut w, &input);
            apex = apex.max(w.agent.pos[1]);
        }
        let rise = apex - 5.0;
        assert!((1.2..=1.3).contains(&rise), "apex rise {rise}");
    }

    #[test]
    fn walk_speed_capped() {
        let mut w = flat_world();
        w.agent.pos = [8.5, 5.0, 8.5];
        let input = MoveInput { wish_x: 0.0, wish_z: 1.0, jump: false, sneak: false };
        for _ in 0..30 {
            tick_agent(&mut w, &input);
        }
        assert!((w.agent.vel[2] - WALK_SPEED).abs() < 1e-9);
        let start = w.agent.pos[2];
        for _ in 0..20 {
            tick_agent(&mut w, &input);
        }
        let dist = w.agent.pos[2] - start;
        assert!((dist - 20.0 * WALK_SPEED).abs() < 0.05);
    }

    #[test]
    fn suffocation_kills_slowly() {
        let mut w = flat_world();
        w.agent.pos = [8.5, 5.0, 8.5];
        // encase head: eye at 5+1.62=6.62 -> cell y=6
        w.set_block(8, 6, 8, STONE);
        for _ in 0..20 {
            tick_agent(&mut w, &idle());
        }
        assert_eq!(w.agent.hp, MAX_HP - 1);
    }

    #[test]
    fn mass_ablation_changes_acceleration_not_gravity() {
        use crate::physics::Physics;
        // Newtonian check: doubling mass halves propulsion acceleration but
        // leaves gravity (jump apex) untouched
        fn run(mass: f64) -> (f64, f64) {
            let mut w = World::new(1, Preset::Flat, Vec::new());
            let mut ph = Physics::default();
            ph.agent_mass = mass;
            w.physics = ph;
            w.agent.pos = [8.5, 5.0, 8.5];
            w.agent.on_ground = true;
            // ticks to reach 90% of walk speed from rest
            let input = MoveInput { wish_x: 0.0, wish_z: 1.0, jump: false, sneak: false };
            let mut ticks = 0;
            while w.agent.vel[2] < 0.9 * w.physics.walk_speed && ticks < 100 {
                tick_agent(&mut w, &input);
                ticks += 1;
            }
            // jump apex
            let mut w2 = World::new(1, Preset::Flat, Vec::new());
            let mut ph2 = Physics::default();
            ph2.agent_mass = mass;
            w2.physics = ph2;
            w2.agent.pos = [8.5, 5.0, 8.5];
            w2.agent.on_ground = true;
            let jump = MoveInput { wish_x: 0.0, wish_z: 0.0, jump: true, sneak: false };
            let mut apex = 5.0f64;
            for _ in 0..40 {
                tick_agent(&mut w2, &jump);
                apex = apex.max(w2.agent.pos[1]);
            }
            (ticks as f64, apex - 5.0)
        }
        let (t1, a1) = run(1.0);
        let (t2, a2) = run(2.0);
        assert!(t2 > t1, "heavier accelerates slower: {t1} vs {t2}");
        assert!((a1 - a2).abs() < 1e-9, "gravity is mass-independent: {a1} vs {a2}");
    }

    #[test]
    fn thrown_item_is_ballistic() {
        // item tossed horizontally from a height follows a parabola: no air
        // drag on items, so range = vx * t_fall
        let mut w = World::new(1, Preset::Flat, Vec::new());
        w.spawn_item(crate::block::DIRT, 1, [8.5, 20.5, 8.5]);
        w.items[0].vel = [0.2, 0.0, 0.0]; // override the spawn scatter
        w.items[0].pos = [8.5, 20.5, 8.5];
        for _ in 0..120 {
            crate::item::tick_items_physics(&mut w);
        }
        let it = &w.items[0];
        // rested on the flat surface (top y=5): center at 5.125
        assert!((it.pos[1] - 5.125).abs() < 0.02, "rested: y={}", it.pos[1]);
        let x = it.pos[0];
        // free fall from 20.5: t ~ sqrt(2*15/0.08) ~ 19 ticks, range ~3.9
        assert!(x > 10.5, "ballistic range: {x}");
        assert!(x < 14.0, "but not unbounded: {x}");
    }

    #[test]
    fn y_below_zero_is_solid() {
        assert!(flat_world().is_solid(0, -1, 0));
        assert!(!flat_world().is_solid(0, 128, 0));
    }
}
