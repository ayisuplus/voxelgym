//! Tick orchestration — the fixed 7-phase in-tick order lives here.
//! Wrong ordering produces ghost interactions, so all phases funnel through
//! `step()` and nothing else mutates the world per-tick.
//!
//!   1. agent action (look / hotbar / mine / place / use / craft)
//!   2. entity integration (agent physics; M2: items; M3: falling blocks)
//!   3. scheduled block ticks (M3: loose-block fall conversion)
//!   4. fluid ticks (M3)
//!   5. circuit BFS (M3)
//!   6. item pickup / despawn (M2)
//!   7. observation construction — on demand via obs getters (no per-tick cost)

use crate::block::*;
use crate::entity::{tick_agent, MoveInput};
use crate::raycast::{RayHit, REACH};
use crate::world::{MiningState, World};

/// One action per tick. Field order matches the gymnasium action dict:
/// (move, jump, sneak, yaw, pitch, mine, place, use, hotbar, craft).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Action {
    /// 0 idle, 1 forward, 2 back, 3 left, 4 right
    pub mv: u8,
    pub jump: bool,
    pub sneak: bool,
    /// Absolute yaw bucket 0..23 (15 deg each).
    pub yaw: u8,
    /// Absolute pitch bucket 0..8 -> -60..+60 deg.
    pub pitch: u8,
    pub mine: bool,
    pub place: bool,
    pub use_: bool,
    /// 0..8
    pub hotbar: u8,
    /// Recipe id, 0 = noop (M2).
    pub craft: u8,
}

impl Action {
    pub fn from_parts(parts: &[u8]) -> Self {
        let g = |i: usize| parts.get(i).copied().unwrap_or(0);
        Action {
            mv: g(0).min(4),
            jump: g(1) != 0,
            sneak: g(2) != 0,
            yaw: g(3) % 24,
            pitch: g(4).min(8),
            mine: g(5) != 0,
            place: g(6) != 0,
            use_: g(7) != 0,
            hotbar: g(8) % 9,
            craft: g(9),
        }
    }

    pub fn pitch_deg(&self) -> f32 {
        -60.0 + 15.0 * self.pitch as f32
    }

    pub fn yaw_deg(&self) -> f32 {
        15.0 * self.yaw as f32
    }
}

/// Current crosshair target (DDA from eye, physical reach 4.5 meters).
/// Targeting ignores fluids (`blocks_target`); the renderer uses the strict
/// `dda` policy. The returned hit distance is normalized to meters.
pub fn raycast_target(world: &mut World) -> Option<RayHit> {
    let eye = world.agent.eye();
    let look = world.agent.look();
    let scale = world.scale();
    let reach = REACH * scale;
    crate::raycast::dda_with(
        eye,
        look,
        reach,
        |x, y, z| world.get_block(x, y, z),
        crate::raycast::blocks_target,
    )
    .map(|mut hit| {
        hit.dist /= scale;
        hit
    })
}

pub fn step(world: &mut World, action: &Action) {
    // ---- 1. agent action ----
    apply_action(world, action);

    // ---- 2. entity integration ----
    let fwd = world.agent.forward();
    let (wx, wz) = match action.mv {
        1 => (fwd[0], fwd[2]),
        2 => (-fwd[0], -fwd[2]),
        3 => (-fwd[2], fwd[0]), // left = forward rotated +90 deg
        4 => (fwd[2], -fwd[0]),
        _ => (0.0, 0.0),
    };
    let input = MoveInput {
        wish_x: wx,
        wish_z: wz,
        jump: action.jump,
        sneak: action.sneak,
    };
    tick_agent(world, &input);

    // ---- 3..6: M2/M3 hooks ----
    crate::hooks::after_entities(world);

    // ---- 7. obs: on demand ----
    world.tick += 1;
    if world.place_cooldown > 0 {
        world.place_cooldown -= 1;
    }
}

fn apply_action(world: &mut World, action: &Action) {
    world.agent.yaw = action.yaw_deg();
    world.agent.pitch = action.pitch_deg();
    world.agent.selected = action.hotbar as usize;

    if world.agent.dead {
        world.mining = None;
        return;
    }

    // --- mining ---
    if action.mine {
        let hit = raycast_target(world);
        match hit {
            Some(h) => {
                let id = cell_id(h.cell);
                let def = block_def(id);
                if def.hardness_ticks == 0 || def.fluid.is_some() || id == AIR {
                    world.mining = None; // not mineable
                } else {
                    let held = world.agent.inventory.held(world.agent.selected);
                    let tool = item_tool(held.item);
                    let proper = match def.tool {
                        None => false,
                        Some((cls, tier)) => tool.is_some_and(|(c, t)| c == cls && t >= tier),
                    };
                    let mult = if proper { 5.0 } else { 1.0 };
                    // Legacy hardness <=5 is an intentional one-default-tick
                    // interaction, independent of tool.  Express that as a
                    // physical duration before applying the configured clock
                    // so 40 Hz takes two ticks (the same 0.05 seconds) rather
                    // than silently turning it into a ten-tick mine.
                    let duration_default_ticks = if def.hardness_ticks <= 5 {
                        1.0
                    } else {
                        def.hardness_ticks as f64 / mult
                    };
                    let add = world.clock_config().default_step_ratio() / duration_default_ticks;
                    let same = world.mining.is_some_and(|m| m.target == (h.x, h.y, h.z));
                    let mut progress = if same {
                        world.mining.unwrap().progress
                    } else {
                        0.0
                    };
                    progress += add;
                    if progress >= 1.0 {
                        break_block(world, h.x, h.y, h.z, proper);
                        world.mining = None;
                    } else {
                        world.mining = Some(MiningState {
                            target: (h.x, h.y, h.z),
                            progress,
                        });
                    }
                }
            }
            None => world.mining = None,
        }
    } else {
        world.mining = None;
    }

    // --- placement ---
    if action.place && world.place_cooldown == 0 {
        let held = world.agent.inventory.held(world.agent.selected);
        if held.count > 0 && is_placeable(held.item) {
            if let Some(h) = raycast_target(world) {
                let (tx, ty, tz) = (h.x + h.face[0], h.y + h.face[1], h.z + h.face[2]);
                let cur = world.get_block(tx, ty, tz);
                if block_def(cell_id(cur)).replaceable {
                    // must not intersect the agent AABB (items checked in M2)
                    let overlaps_agent = {
                        let mn = world.agent.aabb_min();
                        let mx = world.agent.aabb_max();
                        tx as f64 + 1.0 > mn[0]
                            && (tx as f64) < mx[0]
                            && ty as f64 + 1.0 > mn[1]
                            && (ty as f64) < mx[1]
                            && tz as f64 + 1.0 > mn[2]
                            && (tz as f64) < mx[2]
                    };
                    let solid = block_def(held.item).solid;
                    if !(solid && overlaps_agent) {
                        let item = world.agent.inventory.consume_held(world.agent.selected);
                        debug_assert_eq!(item, Some(held.item));
                        // default states: torches place LIT (the circuit tick
                        // would correct a dark torch anyway, one tick later);
                        // repeaters take their facing from the agent's yaw
                        // (output side away from the player, MC convention)
                        let cell = match held.item {
                            RTORCH => make_cell(RTORCH, 1),
                            REPEATER => {
                                let d4 =
                                    ((world.agent.yaw / 90.0).round() as i32).rem_euclid(4) as u16;
                                make_cell(REPEATER, d4)
                            }
                            other => other,
                        };
                        world.set_block(tx, ty, tz, cell);
                        world.place_cooldown = world
                            .clock_config()
                            .ticks_for_default_ticks(4)
                            .min(u8::MAX as u64)
                            as u8;
                    }
                }
            }
        }
    }

    // --- use ---
    if action.use_ {
        if let Some(h) = raycast_target(world) {
            let id = cell_id(h.cell);
            match id {
                DOOR | LEVER => {
                    let st = cell_state(h.cell) ^ 1;
                    world.set_block(h.x, h.y, h.z, make_cell(id, st));
                }
                _ => crate::hooks::use_block(world, h.x, h.y, h.z, id),
            }
        }
    }

    // --- craft (M2) ---
    if action.craft != 0 {
        crate::hooks::craft(world, action.craft);
    }
}

/// Break a block: set air; M2 adds drops as item entities.
fn break_block(world: &mut World, x: i32, y: i32, z: i32, proper_tool: bool) {
    let cell = world.get_block(x, y, z);
    world.set_block(x, y, z, AIR);
    crate::hooks::block_broken(world, x, y, z, cell, proper_tool);
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::worldgen::Preset;

    fn act(pitch: u8, mine: bool) -> Action {
        Action {
            pitch,
            mine,
            ..Default::default()
        }
    }

    fn flat_at_spawn() -> World {
        let mut w = World::new(1, Preset::Flat, Vec::new());
        // settle on ground
        for _ in 0..5 {
            step(&mut w, &Action::default());
        }
        w
    }

    #[test]
    fn mine_dirt_by_looking_down() {
        let mut w = flat_at_spawn();
        let (ax, az) = (w.agent.pos[0].floor() as i32, w.agent.pos[2].floor() as i32);
        assert_eq!(w.get_block(ax + 1, 4, az), GRASS_BLOCK);
        // pitch bucket 8 = +60 deg (down), yaw 18 = 270 deg -> +x:
        // ray enters cell (ax+1, 4, az) ~1.87 cells out.
        let mut a = act(8, true);
        a.yaw = 18;
        for _ in 0..40 {
            step(&mut w, &a);
        }
        assert_eq!(w.get_block(ax + 1, 4, az), AIR);
        // M1: no drops yet — nothing in inventory
        assert_eq!(w.agent.inventory.count(DIRT), 0);
    }

    #[test]
    fn bedrock_unmineable() {
        let mut w = flat_at_spawn();
        let ax = w.agent.pos[0].floor() as i32;
        let az = w.agent.pos[2].floor() as i32;
        w.set_block(ax + 1, 5, az, BEDROCK);
        let mut a = act(8, true);
        a.yaw = 18;
        for _ in 0..100 {
            step(&mut w, &a);
        }
        assert_eq!(w.get_block(ax + 1, 5, az), BEDROCK);
    }

    #[test]
    fn place_consumes_and_cools_down() {
        let mut w = flat_at_spawn();
        let ax = w.agent.pos[0].floor() as i32;
        let az = w.agent.pos[2].floor() as i32;
        w.agent.inventory.add(COBBLESTONE, 5);
        w.agent.selected = 0;
        // look +x, 60 deg down: hits top face of (ax+1, 4) -> place at (ax+1, 5)
        let a = Action {
            yaw: 18,
            pitch: 8,
            place: true,
            ..Default::default()
        };
        step(&mut w, &a);
        assert_eq!(w.get_block(ax + 1, 5, az), COBBLESTONE);
        assert_eq!(w.agent.inventory.count(COBBLESTONE), 4);
        // cooldown: immediate second place is a no-op
        step(&mut w, &a);
        assert_eq!(w.agent.inventory.count(COBBLESTONE), 4);
        assert_eq!(w.get_block(ax + 1, 5, az), COBBLESTONE);
    }

    #[test]
    fn place_blocked_inside_agent() {
        let mut w = flat_at_spawn();
        w.agent.inventory.add(STONE, 3);
        let ax = w.agent.pos[0].floor() as i32;
        let az = w.agent.pos[2].floor() as i32;
        // wall at eye row in +x: level ray hits its -x face, target cell is
        // the agent's own head cell -> solid placement must be refused
        for y in 5..=8 {
            w.set_block(ax + 1, y, az, STONE);
        }
        let a = Action {
            yaw: 18,
            pitch: 4, // level
            place: true,
            ..Default::default()
        };
        step(&mut w, &a);
        assert_eq!(w.get_block(ax, 6, az), AIR);
        assert_eq!(w.agent.inventory.count(STONE), 3);
    }

    #[test]
    fn use_toggles_door_and_lever() {
        let mut w = flat_at_spawn();
        let ax = w.agent.pos[0].floor() as i32;
        let az = w.agent.pos[2].floor() as i32;
        // eye row is y=6 (feet 5, eye 6.62)
        w.set_block(ax + 1, 6, az, DOOR);
        w.set_block(ax - 1, 6, az, LEVER);
        let a = Action {
            yaw: 18,  // +x
            pitch: 4, // level
            use_: true,
            ..Default::default()
        };
        step(&mut w, &a);
        let door = w.get_block(ax + 1, 6, az);
        assert_eq!(cell_id(door), DOOR);
        assert_eq!(cell_state(door), 1);
        // door open -> not solid
        assert!(!w.is_solid(ax + 1, 6, az));
        let b = Action { yaw: 6, ..a }; // 90 deg -> -x
        step(&mut w, &b);
        assert_eq!(cell_state(w.get_block(ax - 1, 6, az)), 1);
    }
}
