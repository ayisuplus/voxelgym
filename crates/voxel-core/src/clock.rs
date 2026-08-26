//! Exact simulation-time configuration and step-boundary clocks.
//!
//! `World::tick` counts completed transitions. Therefore the clock observed
//! immediately before the first step is tick 0, and the clock immediately
//! after that step is tick 1. A tick duration is stored as a reduced positive
//! rational so snapshots never depend on floating-point normalization.

/// Immutable duration of one simulation tick, expressed in seconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClockConfig {
    numerator: u64,
    denominator: u64,
}

impl ClockConfig {
    /// The default Minecraft-compatible 20 Hz clock (1/20 second per tick).
    pub const DEFAULT: Self = Self {
        numerator: 1,
        denominator: 20,
    };

    /// Construct a positive rational tick duration and reduce it canonically.
    pub fn new(numerator: u64, denominator: u64) -> Result<Self, String> {
        if denominator == 0 {
            return Err("tick duration denominator must be non-zero".into());
        }
        if numerator == 0 {
            return Err("tick duration must be positive".into());
        }
        let divisor = gcd(numerator, denominator);
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    pub const fn numerator(self) -> u64 {
        self.numerator
    }

    pub const fn denominator(self) -> u64 {
        self.denominator
    }

    pub fn seconds_per_tick(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }

    /// Ratio between this step duration and the legacy 20 Hz step.
    ///
    /// Per-step velocities scale by this factor and per-step accelerations
    /// by its square.  A value of `1.0` therefore preserves every historical
    /// default bit-for-bit.
    pub fn default_step_ratio(self) -> f64 {
        self.seconds_per_tick() / Self::DEFAULT.seconds_per_tick()
    }

    /// Convert a duration expressed in historical 20 Hz ticks to this
    /// clock.  Durations are rounded up so a scheduled effect never fires
    /// earlier than the physical duration it represents.
    pub fn ticks_for_default_ticks(self, default_ticks: u64) -> u64 {
        if default_ticks == 0 {
            return 0;
        }
        // ceil((default_ticks / 20) / (numerator / denominator))
        let numerator = default_ticks as u128 * self.denominator as u128;
        let denominator = 20u128 * self.numerator as u128;
        numerator.div_ceil(denominator).min(u64::MAX as u128) as u64
    }

    /// Exact elapsed time at a step boundary.
    pub fn time_at_tick(self, tick: u64) -> (u128, u64) {
        reduce_u128(tick as u128 * self.numerator as u128, self.denominator)
    }
}

impl Default for ClockConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A value view of simulation time at a transition boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimClock {
    config: ClockConfig,
    tick: u64,
}

impl SimClock {
    /// Clock at the initial boundary, before any transitions have completed.
    pub const fn new(config: ClockConfig) -> Self {
        Self { config, tick: 0 }
    }

    /// Clock at a boundary after `tick` completed transitions.
    pub const fn at_tick(config: ClockConfig, tick: u64) -> Self {
        Self { config, tick }
    }

    pub const fn config(self) -> ClockConfig {
        self.config
    }

    pub const fn tick(self) -> u64 {
        self.tick
    }

    pub fn elapsed_fraction(self) -> (u128, u64) {
        self.config.time_at_tick(self.tick)
    }

    pub fn elapsed_seconds(self) -> f64 {
        self.tick as f64 * self.config.seconds_per_tick()
    }

    pub fn remaining_ticks(self, horizon_tick: u64) -> u64 {
        horizon_tick.saturating_sub(self.tick)
    }

    pub fn remaining_seconds(self, horizon_tick: u64) -> f64 {
        self.remaining_ticks(horizon_tick) as f64 * self.config.seconds_per_tick()
    }

    pub fn sample_time_fraction(self, sample_tick: u64) -> (u128, u64) {
        self.config.time_at_tick(sample_tick)
    }

    /// Age of a sample in ticks, or `None` when it claims to come from the future.
    pub fn data_age_ticks(self, sample_tick: u64) -> Option<u64> {
        self.tick.checked_sub(sample_tick)
    }

    pub fn data_age_seconds(self, sample_tick: u64) -> Option<f64> {
        self.data_age_ticks(sample_tick)
            .map(|ticks| ticks as f64 * self.config.seconds_per_tick())
    }
}

const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

fn reduce_u128(numerator: u128, denominator: u64) -> (u128, u64) {
    if numerator == 0 {
        return (0, 1);
    }
    let divisor = gcd_u128(numerator, denominator as u128);
    (numerator / divisor, denominator / divisor as u64)
}

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}
