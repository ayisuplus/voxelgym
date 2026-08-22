//! Runtime physics overrides — the ablation surface.
//!
//! Every field defaults to the contract constant (Minecraft public values).
//! A non-default Physics set at world construction changes world semantics
//! and is serialized in snapshots (it is part of the state for replay).

use crate::entity;
use crate::fluid;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Physics {
    pub gravity: f64,          // entity::GRAVITY
    pub gravity_mult: f64,     // entity::GRAVITY_MULT
    pub terminal_vy: f64,      // entity::TERMINAL_VY
    pub jump_vy: f64,          // entity::JUMP_VY
    pub walk_speed: f64,       // entity::WALK_SPEED
    pub sneak_mult: f64,       // entity::SNEAK_MULT
    pub water_spread: u16,     // fluid::WATER_MAX
    pub lava_spread: u16,      // fluid::LAVA_MAX
    pub water_period: u64,     // fluid::WATER_PERIOD
    pub lava_period: u64,      // fluid::LAVA_PERIOD
    pub fall_safe: f64,        // free-fall distance before damage
    pub lava_damage: i32,      // half-hearts per 10 ticks in lava
    pub suffocate_damage: i32, // half-hearts per 20 ticks with head in solid
    /// Agent mass (dimensionless, MC-scale). Newtonian gravity is
    /// mass-independent; mass enters via acceleration = force / mass.
    pub agent_mass: f64,       // 1.0
    /// Max ground propulsion force (accel 0.1 at mass 1 — MC ground accel).
    pub ground_force: f64,
    /// Max air-control force (accel 0.02 at mass 1 — MC air accel).
    pub air_force: f64,
    /// World scale: cells per meter (1.0 = Minecraft 1 m cells; 2.0 = 0.5 m
    /// cells). Structural knob: set at world construction, serialized in
    /// snapshots. All spatial constants (here and the consts in entity/
    /// raycast/item/loose/fire/tnt) are multiplied by it; temporal
    /// constants (tick periods, damage intervals) are not. Circuit power
    /// range is a 4-bit discrete semantic and is intentionally NOT scaled.
    pub scale: f64,
}

impl Default for Physics {
    fn default() -> Self {
        Physics {
            gravity: entity::GRAVITY,
            gravity_mult: entity::GRAVITY_MULT,
            terminal_vy: entity::TERMINAL_VY,
            jump_vy: entity::JUMP_VY,
            walk_speed: entity::WALK_SPEED,
            sneak_mult: entity::SNEAK_MULT,
            water_spread: fluid::WATER_MAX,
            lava_spread: fluid::LAVA_MAX,
            water_period: fluid::WATER_PERIOD,
            lava_period: fluid::LAVA_PERIOD,
            fall_safe: 3.0,
            lava_damage: 4,
            suffocate_damage: 1,
            agent_mass: 1.0,
            ground_force: entity::ACCEL_GROUND,
            air_force: entity::ACCEL_AIR,
            scale: 1.0,
        }
    }
}

impl Physics {
    pub const FIELDS: &[&str] = &[
        "gravity", "gravity_mult", "terminal_vy", "jump_vy", "walk_speed", "sneak_mult",
        "water_spread", "lava_spread", "water_period", "lava_period", "fall_safe",
        "lava_damage", "suffocate_damage", "agent_mass", "ground_force", "air_force",
        "scale",
    ];

    pub fn set(&mut self, key: &str, value: f64) -> Result<(), String> {
        match key {
            "gravity" => self.gravity = value,
            "gravity_mult" => self.gravity_mult = value,
            "terminal_vy" => self.terminal_vy = value,
            "jump_vy" => self.jump_vy = value,
            "walk_speed" => self.walk_speed = value,
            "sneak_mult" => self.sneak_mult = value,
            "water_spread" => self.water_spread = value as u16,
            "lava_spread" => self.lava_spread = value as u16,
            "water_period" => self.water_period = value.max(1.0) as u64,
            "lava_period" => self.lava_period = value.max(1.0) as u64,
            "fall_safe" => self.fall_safe = value,
            "lava_damage" => self.lava_damage = value as i32,
            "suffocate_damage" => self.suffocate_damage = value as i32,
            "agent_mass" => self.agent_mass = value.max(1e-6),
            "ground_force" => self.ground_force = value,
            "air_force" => self.air_force = value,
            "scale" => self.scale = value.max(1e-6),
            _ => return Err(format!("unknown physics field '{key}'")),
        }
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<f64> {
        Some(match key {
            "gravity" => self.gravity,
            "gravity_mult" => self.gravity_mult,
            "terminal_vy" => self.terminal_vy,
            "jump_vy" => self.jump_vy,
            "walk_speed" => self.walk_speed,
            "sneak_mult" => self.sneak_mult,
            "water_spread" => self.water_spread as f64,
            "lava_spread" => self.lava_spread as f64,
            "water_period" => self.water_period as f64,
            "lava_period" => self.lava_period as f64,
            "fall_safe" => self.fall_safe,
            "lava_damage" => self.lava_damage as f64,
            "suffocate_damage" => self.suffocate_damage as f64,
            "agent_mass" => self.agent_mass,
            "ground_force" => self.ground_force,
            "air_force" => self.air_force,
            "scale" => self.scale,
            _ => return None,
        })
    }

    /// Returns a copy with every SPATIAL field multiplied by `s` (called
    /// once at world construction; `scale` itself is set to `s`). Ratios
    /// (sneak_mult, gravity_mult) and time-based fields (periods, damage
    /// per N ticks) stay untouched.
    pub fn spatially_scaled(mut self, s: f64) -> Self {
        if s != 1.0 {
            self.gravity *= s;
            self.terminal_vy *= s;
            self.jump_vy *= s;
            self.walk_speed *= s;
            self.ground_force *= s;
            self.air_force *= s;
            self.fall_safe *= s;
            self.water_spread = (self.water_spread as f64 * s).round() as u16;
            self.lava_spread = (self.lava_spread as f64 * s).round() as u16;
        }
        self.scale = s;
        self
    }

    pub(crate) fn write_to(&self, buf: &mut Vec<u8>) {
        for k in Self::FIELDS {
            buf.extend_from_slice(&self.get(k).unwrap().to_bits().to_le_bytes());
        }
    }

    pub(crate) fn read_from(r: &mut crate::world::Reader) -> Result<Self, String> {
        let mut p = Physics::default();
        for k in Self::FIELDS {
            let v = f64::from_bits(r.u64()?);
            p.set(k, v)?;
        }
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_contract() {
        let p = Physics::default();
        assert_eq!(p.gravity, 0.08);
        assert_eq!(p.jump_vy, 0.42);
        assert_eq!(p.walk_speed, 0.2159);
        assert_eq!(p.water_spread, 7);
        assert_eq!(p.lava_spread, 3);
        assert_eq!(p.water_period, 5);
        assert_eq!(p.lava_period, 30);
    }

    #[test]
    fn field_roundtrip() {
        let mut p = Physics::default();
        p.set("gravity", 0.12).unwrap();
        p.set("water_spread", 3.0).unwrap();
        assert_eq!(p.get("gravity"), Some(0.12));
        assert_eq!(p.get("water_spread"), Some(3.0));
        assert!(p.set("nonsense", 1.0).is_err());
    }
}
