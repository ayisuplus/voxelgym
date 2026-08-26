//! World: chunk columns + agent + global state, with canonical
//! snapshot/restore/hash (the determinism contract).

use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use xxhash_rust::xxh3::Xxh3;

use crate::block::*;
use crate::chunk::*;
use crate::clock::{ClockConfig, SimClock};
use crate::entity::Agent;
use crate::inventory::Inventory;
use crate::item::ItemEntity;
use crate::loose::FallingBlock;
use crate::physics::Physics;
use crate::recipe::FurnaceState;
use crate::rng::Rng;
use crate::worldgen::{
    apply_scenario, derive_semantic_regions, generate_chunk, generate_chunk_v7,
    scale_scenario_to_cells, scale_semantic_regions_to_cells, validate_semantic_regions,
    validate_semantic_scenario, Preset, ScenarioSpec, SemanticRegionSpec,
};

/// Snapshot format v8 persists the exact clock, stable semantic regions, and
/// ordered dirty-cell queue. v7 added falling-block distance and v6 added
/// per-chunk `touched` flags; v5-v7 remain readable with the historical
/// 20 Hz clock, derived semantic IDs, and an empty dirty queue.
pub const SNAPSHOT_VERSION: u32 = 8;

const FALLING_DISTANCE_SNAPSHOT_VERSION: u32 = 7;
const CLOCK_SNAPSHOT_VERSION: u32 = 8;
const SEMANTIC_REGIONS_SNAPSHOT_VERSION: u32 = 8;
const DIRTY_QUEUE_SNAPSHOT_VERSION: u32 = 8;

fn checked_spatial_scale(scale: f64) -> Result<(usize, usize), String> {
    let height = CHUNK_Y as f64 * scale;
    if !scale.is_finite()
        || scale < 1.0
        || !height.is_finite()
        || height.fract() != 0.0
        || height > i32::MAX as f64
    {
        return Err(format!(
            "invalid spatial scale {scale}: expected finite scale >= 1 with integral, i32-representable world height"
        ));
    }
    let chunk_h = height as usize;
    let chunk_cells = CHUNK_X
        .checked_mul(chunk_h)
        .and_then(|cells| cells.checked_mul(CHUNK_Z))
        .ok_or_else(|| format!("invalid spatial scale {scale}: chunk volume overflows usize"))?;
    Ok((chunk_h, chunk_cells))
}

/// Fixed-seed xxh3 hasher for world-state maps/sets. Unlike `RandomState`,
/// iteration order is stable across processes (determinism hygiene), and it
/// is several times faster than SipHash on the small keys used here. Every
/// consumer that iterates sorts first, so no semantics depend on order.
pub(crate) type XBuild = BuildHasherDefault<Xxh3>;
pub(crate) type XMap<K, V> = HashMap<K, V, XBuild>;
pub(crate) type XSet<K> = std::collections::HashSet<K, XBuild>;

/// Per-tick events for achievement hooks (transient; not part of snapshots —
/// they are derived from state transitions, never their source).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Event {
    ItemPicked { item: u16, count: u16 },
    Crafted { recipe: u8, out: u16, count: u16 },
    BlockMined { id: u16 },
    Smelted { item: u16 },
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MiningState {
    pub target: (i32, i32, i32),
    pub progress: f64,
}

pub struct World {
    pub seed: u64,
    pub preset: Preset,
    pub(crate) physics: Physics,
    /// Scenario overlay resolved into this world's cell frame at creation.
    /// Constructor inputs remain canonical scale-1 meter volumes.
    pub scenario: ScenarioSpec,
    semantic_regions: Vec<SemanticRegionSpec>,
    pub chunks: XMap<(i32, i32), Chunk>,
    clock_config: ClockConfig,
    /// Number of completed simulation transitions. Tick 0 is the boundary
    /// before the first step; after one step this value is 1.
    pub tick: u64,
    pub rng: Rng,
    pub agent: Agent,
    pub mining: Option<MiningState>,
    pub place_cooldown: u8,
    /// Cells changed since last tick — neighbor-change queue for M3
    /// (loose-block support checks). Kept in M1 so semantics are stable.
    pub dirty: Vec<(i32, i32, i32)>,
    pub items: Vec<ItemEntity>,
    pub furnaces: XMap<(i32, i32, i32), FurnaceState>,
    pub events: Vec<Event>,
    pub next_item_id: u64,
    /// Falling loose-block entities (M3).
    pub falling: Vec<FallingBlock>,
    /// Scheduled support checks: (x, y, z, due_tick).
    pub scheduled_falls: Vec<(i32, i32, i32, u64)>,
    /// Dedup index over scheduled_falls positions (O(1) membership instead
    /// of a linear scan per dirty cell). Rebuilt from the vec on restore.
    pub scheduled_set: XSet<(i32, i32, i32)>,
    /// Fluid cells needing automata attention (M3).
    pub active_fluids: XSet<(i32, i32, i32)>,
    /// All circuit cells (wire/lever/door) for BFS recompute (M3).
    pub circuit_cells: XSet<(i32, i32, i32)>,
    /// Pressure-plate subset size of circuit_cells — maintained by
    /// circuit::on_cell_changed, derived (not hashed/snapshotted; rebuilt
    /// on restore). Plates read the agent position, so they force a
    /// recompute every tick regardless of block changes.
    pub plate_count: u32,
    /// True when the last circuit recompute produced zero state changes:
    /// the network is at a fixpoint and phase 5 is skipped until a relevant
    /// dirty cell or a plate. Synchronous semantics make this exact —
    /// next = f(current), so a zero-update recompute proves all future
    /// recomputes are no-ops until external input changes. Derived.
    pub circuit_settled: bool,
    /// Burning cells (fire CA).
    pub active_fire: XSet<(i32, i32, i32)>,
    /// Live TNT cells (blast triggers scan their neighborhoods).
    pub tnt_cells: XSet<(i32, i32, i32)>,
    /// Primed explosions: (x, y, z, due_tick).
    pub pending_booms: Vec<(i32, i32, i32, u64)>,
    pub next_falling_id: u64,
    /// Last oracle inventory-swap item (behavior-trace meta; NOT part of
    /// snapshot/hash — it's an input event, not sim state).
    pub last_swap: Option<u16>,
}

impl World {
    pub fn new(seed: u64, preset: Preset, scenario: ScenarioSpec) -> Self {
        Self::new_scaled_with_clock(seed, preset, scenario, 1.0, ClockConfig::default())
    }

    pub fn new_with_clock(
        seed: u64,
        preset: Preset,
        scenario: ScenarioSpec,
        clock_config: ClockConfig,
    ) -> Self {
        Self::new_scaled_with_clock(seed, preset, scenario, 1.0, clock_config)
    }

    /// `scale` = cells per meter (1.0 = MC 1 m cells; 2.0 = 0.5 m cells).
    /// Physical world size is invariant: chunk height becomes 128*scale,
    /// noise is sampled per meter, all spatial constants multiply by scale.
    pub fn new_scaled(seed: u64, preset: Preset, scenario: ScenarioSpec, scale: f64) -> Self {
        Self::new_scaled_with_clock(seed, preset, scenario, scale, ClockConfig::default())
    }

    pub fn new_scaled_with_clock(
        seed: u64,
        preset: Preset,
        scenario: ScenarioSpec,
        scale: f64,
        clock_config: ClockConfig,
    ) -> Self {
        let semantic_regions = derive_semantic_regions(&scenario);
        Self::build_from_canonical_scenario(
            seed,
            preset,
            scenario,
            semantic_regions,
            scale,
            clock_config,
        )
    }

    /// Construct a scale-1, 20 Hz world with explicit stable semantic IDs.
    pub fn new_with_semantic_regions(
        seed: u64,
        preset: Preset,
        semantic_regions: Vec<SemanticRegionSpec>,
    ) -> Result<Self, String> {
        Self::new_scaled_with_clock_and_semantic_regions(
            seed,
            preset,
            semantic_regions,
            1.0,
            ClockConfig::default(),
        )
    }

    /// Construct a scaled 20 Hz world with explicit stable semantic IDs.
    pub fn new_scaled_with_semantic_regions(
        seed: u64,
        preset: Preset,
        semantic_regions: Vec<SemanticRegionSpec>,
        scale: f64,
    ) -> Result<Self, String> {
        Self::new_scaled_with_clock_and_semantic_regions(
            seed,
            preset,
            semantic_regions,
            scale,
            ClockConfig::default(),
        )
    }

    /// Construct a world whose semantic regions and compound-structure IDs
    /// are supplied explicitly in canonical scale-1 meter coordinates.
    pub fn new_scaled_with_clock_and_semantic_regions(
        seed: u64,
        preset: Preset,
        semantic_regions: Vec<SemanticRegionSpec>,
        scale: f64,
        clock_config: ClockConfig,
    ) -> Result<Self, String> {
        validate_semantic_regions(&semantic_regions)?;
        let scenario = semantic_regions
            .iter()
            .map(|spec| (spec.region, spec.cell))
            .collect();
        Ok(Self::build_from_canonical_scenario(
            seed,
            preset,
            scenario,
            semantic_regions,
            scale,
            clock_config,
        ))
    }

    fn build_from_canonical_scenario(
        seed: u64,
        preset: Preset,
        scenario: ScenarioSpec,
        semantic_regions: Vec<SemanticRegionSpec>,
        scale: f64,
        clock_config: ClockConfig,
    ) -> Self {
        checked_spatial_scale(scale).unwrap_or_else(|error| panic!("{error}"));
        let scenario = scale_scenario_to_cells(scenario, scale);
        let semantic_regions = scale_semantic_regions_to_cells(semantic_regions, scale);
        let mut w = World {
            seed,
            preset,
            physics: Physics::default()
                .spatially_scaled(scale)
                .temporally_scaled(clock_config),
            scenario,
            semantic_regions,
            chunks: XMap::default(),
            clock_config,
            tick: 0,
            rng: Rng::new(seed, 1),
            agent: Agent::new([0.5, 8.0 * scale, 0.5], scale),
            mining: None,
            place_cooldown: 0,
            dirty: Vec::new(),
            items: Vec::new(),
            furnaces: XMap::default(),
            events: Vec::new(),
            next_item_id: 1,
            falling: Vec::new(),
            scheduled_falls: Vec::new(),
            scheduled_set: XSet::default(),
            active_fluids: XSet::default(),
            circuit_cells: XSet::default(),
            plate_count: 0,
            circuit_settled: false,
            active_fire: XSet::default(),
            tnt_cells: XSet::default(),
            pending_booms: Vec::new(),
            next_falling_id: 1,
            last_swap: None,
        };
        // Seed M3 systems from the scenario spec: fluid cells enter the
        // active set, loose blocks get an early support check, circuit
        // cells register for BFS recompute.
        let scenario = w.scenario.clone();
        for (region, cell) in &scenario {
            let id = cell_id(*cell);
            let is_fluid = block_def(id).fluid.is_some();
            let is_loose = block_def(id).loose;
            let is_circuit = crate::circuit::is_circuit(id);
            if !(is_fluid || is_loose || is_circuit) {
                if id == FIRE {
                    for x in region.x0..=region.x1 {
                        for y in region.y0..=region.y1 {
                            for z in region.z0..=region.z1 {
                                w.active_fire.insert((x, y, z));
                            }
                        }
                    }
                }
                if id == TNT {
                    for x in region.x0..=region.x1 {
                        for y in region.y0..=region.y1 {
                            for z in region.z0..=region.z1 {
                                w.tnt_cells.insert((x, y, z));
                            }
                        }
                    }
                }
                continue;
            }
            for x in region.x0..=region.x1 {
                for y in region.y0..=region.y1 {
                    for z in region.z0..=region.z1 {
                        if is_fluid {
                            w.active_fluids.insert((x, y, z));
                            // scenario lava is an ignition source from tick
                            // 0 ("evaluated when lava appears" — scenario
                            // placement IS appearance). Worldgen lava is
                            // deliberately NOT seeded: that would make
                            // active_fire depend on chunk loadedness and
                            // break the hash's observer-independence.
                            if id == LAVA {
                                w.active_fire.insert((x, y, z));
                            }
                        }
                        if is_loose {
                            let due = w.clock_config.ticks_for_default_ticks(1);
                            w.scheduled_falls.push((x, y, z, due));
                            w.scheduled_set.insert((x, y, z));
                        }
                        if is_circuit && w.circuit_cells.insert((x, y, z)) && id == PRESSURE_PLATE {
                            w.plate_count += 1;
                        }
                    }
                }
            }
        }
        // A scenario-seeded circuit has no dirty cells yet; force one
        // initial recompute by marking a circuit cell dirty.
        if !w.circuit_cells.is_empty() {
            let c = *w.circuit_cells.iter().next().unwrap();
            w.dirty.push(c);
        }
        let spawn = w.find_spawn(0, 0);
        w.agent = Agent::new(spawn, w.physics.scale);
        w
    }

    /// Immutable episode clock configuration.
    pub const fn clock_config(&self) -> ClockConfig {
        self.clock_config
    }

    /// Immutable structural cell density in cells per meter.
    pub const fn scale(&self) -> f64 {
        self.physics.scale()
    }

    /// Read-only access to the effective, spatially and temporally scaled
    /// physics configuration for this episode.
    pub const fn physics(&self) -> &Physics {
        &self.physics
    }

    /// Stable semantic regions resolved into this world's cell frame.
    pub fn semantic_regions(&self) -> &[SemanticRegionSpec] {
        &self.semantic_regions
    }

    /// Consume a newly constructed world and replace its default physics
    /// with canonical scale-1, 20 Hz overrides.
    ///
    /// The world's immutable scale and clock transforms are applied exactly
    /// once. Call this immediately on a `World::new*` result; runtime physics
    /// mutation is deliberately not exposed across crate boundaries.
    pub fn with_physics(mut self, canonical: Physics) -> Self {
        assert_eq!(
            canonical.scale(),
            1.0,
            "with_physics expects canonical scale-1 overrides"
        );
        let scale = self.scale();
        self.physics = canonical
            .spatially_scaled(scale)
            .temporally_scaled(self.clock_config);
        self
    }

    /// Simulation time at the current transition boundary.
    pub const fn sim_clock(&self) -> SimClock {
        SimClock::at_tick(self.clock_config, self.tick)
    }

    /// World height in cells: 128 * scale.
    pub fn height(&self) -> i32 {
        (CHUNK_Y as f64 * self.physics.scale) as i32
    }

    /// Highest solid top + 1 at column (x, z); fallback y=8*scale for void.
    pub fn find_spawn(&mut self, x: i32, z: i32) -> [f64; 3] {
        let center_offset = 0.5 * self.scale();
        for y in (0..self.height()).rev() {
            if self.is_solid(x, y, z) {
                return [
                    x as f64 + center_offset,
                    (y + 1) as f64,
                    z as f64 + center_offset,
                ];
            }
        }
        let s = self.physics.scale;
        [x as f64 + center_offset, 8.0 * s, z as f64 + center_offset]
    }

    pub fn ensure_chunk(&mut self, cx: i32, cz: i32) -> &mut Chunk {
        let scale = self.physics.scale;
        self.chunks.entry((cx, cz)).or_insert_with(|| {
            let mut c = generate_chunk(self.seed, self.preset, cx, cz, scale);
            apply_scenario(&mut c, cx, cz, &self.scenario);
            c
        })
    }

    /// Raw cell at world coords. y<0 -> bedrock, y>=height -> air.
    pub fn get_block(&mut self, x: i32, y: i32, z: i32) -> u16 {
        if y < WORLD_MIN_Y {
            return BEDROCK;
        }
        if y >= self.height() {
            return AIR;
        }
        let (cx, cz) = (x.div_euclid(16), z.div_euclid(16));
        let (lx, lz) = (x.rem_euclid(16) as usize, z.rem_euclid(16) as usize);
        self.ensure_chunk(cx, cz).get(lx, y as usize, lz)
    }

    /// Read-only variant that never generates (un-generated -> AIR).
    /// Only use where generation is guaranteed or air-default is correct.
    pub fn peek_block(&self, x: i32, y: i32, z: i32) -> u16 {
        if y < WORLD_MIN_Y {
            return BEDROCK;
        }
        if y >= self.height() {
            return AIR;
        }
        let (cx, cz) = (x.div_euclid(16), z.div_euclid(16));
        match self.chunks.get(&(cx, cz)) {
            Some(c) => c.get(
                x.rem_euclid(16) as usize,
                y as usize,
                z.rem_euclid(16) as usize,
            ),
            None => AIR,
        }
    }

    /// Compatibility mutation path for trusted simulation-generated cells.
    ///
    /// Unknown block IDs are ignored before chunk generation or hook
    /// execution. Public adapters that need to report invalid input should use
    /// [`World::try_set_block`].
    pub fn set_block(&mut self, x: i32, y: i32, z: i32, cell: u16) {
        let _ = self.try_set_block(x, y, z, cell);
    }

    /// Set one cell after validating its block ID, atomically with respect to
    /// both world state and mutation hooks.
    pub fn try_set_block(&mut self, x: i32, y: i32, z: i32, cell: u16) -> Result<(), String> {
        validate_cell(cell)?;
        if !(WORLD_MIN_Y..self.height()).contains(&y) {
            return Ok(());
        }
        let (cx, cz) = (x.div_euclid(16), z.div_euclid(16));
        let (lx, lz) = (x.rem_euclid(16) as usize, z.rem_euclid(16) as usize);
        let old = {
            let c = self.ensure_chunk(cx, cz);
            let old = c.get(lx, y as usize, lz);
            if old != cell {
                c.set(lx, y as usize, lz, cell);
                c.touched = true;
            }
            old
        };
        if old != cell {
            self.dirty.push((x, y, z));
            crate::circuit::on_cell_changed(self, x, y, z, old, cell);
        }
        Ok(())
    }

    pub fn is_solid(&mut self, x: i32, y: i32, z: i32) -> bool {
        let cell = self.get_block(x, y, z);
        let id = cell_id(cell);
        if id == DOOR && cell_state(cell) & 1 == 1 {
            return false; // open door does not collide
        }
        block_def(id).solid
    }

    /// Fluid at a position (if any).
    pub fn fluid_at(&mut self, x: i32, y: i32, z: i32) -> Option<Fluid> {
        block_def(cell_id(self.get_block(x, y, z))).fluid
    }

    /// Fluid at the agent's feet cell.
    pub fn fluid_at_feet(&mut self) -> Option<Fluid> {
        let p = self.agent.pos;
        self.fluid_at(
            p[0].floor() as i32,
            p[1].floor() as i32,
            p[2].floor() as i32,
        )
    }

    /// 21(x) x 11(y) x 21(z) one-metre window centered on the eye's metric
    /// voxel, y in [eye_y-4m, eye_y+6m]. Flat vec, index [dx][dy][dz]
    /// (dz fastest). At scale > 1 each output element samples the centre of
    /// its canonical one-metre volume, preserving the policy's physical
    /// field of view and tensor shape.
    ///
    /// Iterates in chunk-aligned runs: each (dx, z-run) resolves its chunk
    /// once instead of hashing (cx, cz) per cell — 4851 map lookups become
    /// at most 27 chunk resolutions plus flat array indexing.
    pub fn voxel_window(&mut self) -> Vec<u16> {
        let eye = self.agent.eye();
        let scale = self.scale();
        if scale != 1.0 {
            let metric_eye = [
                (eye[0] / scale).floor() as i32,
                (eye[1] / scale).floor() as i32,
                (eye[2] / scale).floor() as i32,
            ];
            let sample_cell =
                |metric_cell: i32| ((metric_cell as f64 + 0.5) * scale).floor() as i32;
            let mut out = Vec::with_capacity(21 * 11 * 21);
            for dx in -10..=10 {
                let x = sample_cell(metric_eye[0] + dx);
                for dy in -4..=6 {
                    let y = sample_cell(metric_eye[1] + dy);
                    for dz in -10..=10 {
                        let z = sample_cell(metric_eye[2] + dz);
                        out.push(self.get_block(x, y, z));
                    }
                }
            }
            return out;
        }
        let ex = eye[0].floor() as i32;
        let ey = eye[1].floor() as i32;
        let ez = eye[2].floor() as i32;
        let height = self.height();
        let mut out = Vec::with_capacity(21 * 11 * 21);
        for dx in -10..=10 {
            let x = ex + dx;
            let (cx, lx) = (x.div_euclid(16), x.rem_euclid(16) as usize);
            // z runs aligned to chunk borders (21 cells span up to 3 chunks)
            let mut runs = [(0i32, 0i32); 3]; // (chunk_z, dz_start)
            let mut n_runs = 0;
            let mut dz = -10;
            while dz <= 10 {
                let cz = (ez + dz).div_euclid(16);
                self.ensure_chunk(cx, cz);
                runs[n_runs] = (cz, dz);
                n_runs += 1;
                dz = ((cz + 1) * 16 - ez).min(11); // first dz of the next chunk
            }
            for dy in -4..=6 {
                let y = ey + dy;
                if y < WORLD_MIN_Y {
                    out.extend_from_slice(&[BEDROCK; 21]);
                    continue;
                }
                if y >= height {
                    out.extend_from_slice(&[AIR; 21]);
                    continue;
                }
                let yu = y as usize;
                for i in 0..n_runs {
                    let (cz, dz0) = runs[i];
                    let dz1 = if i + 1 < n_runs {
                        runs[i + 1].1 - 1
                    } else {
                        10
                    };
                    let c = &self.chunks[&(cx, cz)];
                    let lz0 = (ez + dz0).rem_euclid(16) as usize;
                    for lz in lz0..=lz0 + (dz1 - dz0) as usize {
                        out.push(c.get(lx, yu, lz));
                    }
                }
            }
        }
        out
    }

    // ---- canonical snapshot / restore / hash ----

    pub fn snapshot(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"VXG1");
        put_u32(&mut buf, SNAPSHOT_VERSION);
        self.write_head(&mut buf);
        self.write_chunks_raw(&mut buf);
        self.write_tail(&mut buf);
        self.write_v8_extension(&mut buf);
        buf
    }

    fn write_head(&self, buf: &mut Vec<u8>) {
        put_u64(buf, self.tick);
        put_u64(buf, self.clock_config.numerator());
        put_u64(buf, self.clock_config.denominator());
        self.write_head_after_clock(buf);
    }

    fn write_head_legacy_v7(&self, buf: &mut Vec<u8>) {
        put_u64(buf, self.tick);
        self.write_head_after_clock(buf);
    }

    fn write_head_after_clock(&self, buf: &mut Vec<u8>) {
        put_u64(buf, self.seed);
        buf.push(self.preset.as_u8());
        put_u32(buf, self.scenario.len() as u32);
        for (r, cell) in &self.scenario {
            for v in [r.x0, r.y0, r.z0, r.x1, r.y1, r.z1] {
                put_i32(buf, v);
            }
            put_u16(buf, *cell);
        }
        let (rs, ri) = self.rng.state();
        put_u128(buf, rs);
        put_u128(buf, ri);
        self.physics.write_to(buf);
    }

    fn write_chunks_raw(&self, buf: &mut Vec<u8>) {
        let mut keys: Vec<(i32, i32)> = self.chunks.keys().copied().collect();
        keys.sort_unstable();
        put_u32(buf, keys.len() as u32);
        for k in keys {
            put_i32(buf, k.0);
            put_i32(buf, k.1);
            let c = &self.chunks[&k];
            buf.push(c.touched as u8);
            push_u16_le_blocks(buf, &c.blocks);
        }
    }

    fn write_tail(&self, buf: &mut Vec<u8>) {
        let a = &self.agent;
        for v in a.pos {
            put_f64(buf, v);
        }
        for v in a.vel {
            put_f64(buf, v);
        }
        put_f32(buf, a.yaw);
        put_f32(buf, a.pitch);
        buf.push(a.on_ground as u8);
        put_i32(buf, a.hp);
        put_f64(buf, a.fall_distance);
        buf.push(a.dead as u8);
        for s in &a.inventory.slots {
            put_u16(buf, s.item);
            put_u16(buf, s.count);
        }
        buf.push(a.selected as u8);
        put_u32(buf, a.suffocation_timer);
        put_u32(buf, a.lava_timer);
        put_u32(buf, a.fire_timer);

        match &self.mining {
            Some(m) => {
                buf.push(1);
                put_i32(buf, m.target.0);
                put_i32(buf, m.target.1);
                put_i32(buf, m.target.2);
                put_f64(buf, m.progress);
            }
            None => buf.push(0),
        }
        buf.push(self.place_cooldown);

        // items sorted by id (canonical order)
        let mut items: Vec<&ItemEntity> = self.items.iter().collect();
        items.sort_by_key(|i| i.id);
        put_u32(buf, items.len() as u32);
        for it in items {
            put_u64(buf, it.id);
            put_u16(buf, it.item);
            put_u16(buf, it.count);
            for v in it.pos {
                put_f64(buf, v);
            }
            for v in it.vel {
                put_f64(buf, v);
            }
            put_u64(buf, it.age);
        }
        // furnaces sorted by position
        let mut furns: Vec<(&(i32, i32, i32), &FurnaceState)> = self.furnaces.iter().collect();
        furns.sort_by_key(|(k, _)| *k);
        put_u32(buf, furns.len() as u32);
        for ((x, y, z), st) in furns {
            put_i32(buf, *x);
            put_i32(buf, *y);
            put_i32(buf, *z);
            put_u32(buf, st.remaining);
            buf.push(st.out_ready as u8);
            buf.push(st.fuel_left);
        }
        put_u64(buf, self.next_item_id);

        // M3: falling blocks, scheduled falls, active fluids, circuit cells
        let mut falling: Vec<&FallingBlock> = self.falling.iter().collect();
        falling.sort_by_key(|f| f.id);
        put_u32(buf, falling.len() as u32);
        for f in falling {
            put_u64(buf, f.id);
            put_u16(buf, f.block);
            for v in f.pos {
                put_f64(buf, v);
            }
            for v in f.vel {
                put_f64(buf, v);
            }
            put_f64(buf, f.fall_dist);
        }
        let mut sched: Vec<(i32, i32, i32, u64)> = self.scheduled_falls.clone();
        sched.sort_unstable();
        put_u32(buf, sched.len() as u32);
        for (x, y, z, due) in sched {
            put_i32(buf, x);
            put_i32(buf, y);
            put_i32(buf, z);
            put_u64(buf, due);
        }
        let mut fluids: Vec<(i32, i32, i32)> = self.active_fluids.iter().copied().collect();
        fluids.sort_unstable();
        put_u32(buf, fluids.len() as u32);
        for (x, y, z) in fluids {
            put_i32(buf, x);
            put_i32(buf, y);
            put_i32(buf, z);
        }
        let mut circuits: Vec<(i32, i32, i32)> = self.circuit_cells.iter().copied().collect();
        circuits.sort_unstable();
        put_u32(buf, circuits.len() as u32);
        for (x, y, z) in circuits {
            put_i32(buf, x);
            put_i32(buf, y);
            put_i32(buf, z);
        }
        put_u64(buf, self.next_falling_id);
        // fire / tnt / pending explosions
        let mut fires: Vec<(i32, i32, i32)> = self.active_fire.iter().copied().collect();
        fires.sort_unstable();
        put_u32(buf, fires.len() as u32);
        for (x, y, z) in fires {
            put_i32(buf, x);
            put_i32(buf, y);
            put_i32(buf, z);
        }
        let mut tnts: Vec<(i32, i32, i32)> = self.tnt_cells.iter().copied().collect();
        tnts.sort_unstable();
        put_u32(buf, tnts.len() as u32);
        for (x, y, z) in tnts {
            put_i32(buf, x);
            put_i32(buf, y);
            put_i32(buf, z);
        }
        let mut booms = self.pending_booms.clone();
        booms.sort_unstable();
        put_u32(buf, booms.len() as u32);
        for (x, y, z, due) in booms {
            put_i32(buf, x);
            put_i32(buf, y);
            put_i32(buf, z);
            put_u64(buf, due);
        }
    }

    /// Snapshot v8 trailer. Keeping this after the historical body lets the
    /// v5-v7 parser retain its exact wire layout while v8 persists derived
    /// semantic labels and pending dirty-cell work.
    fn write_v8_extension(&self, buf: &mut Vec<u8>) {
        put_u32(buf, self.semantic_regions.len() as u32);
        for spec in &self.semantic_regions {
            put_u64(buf, spec.region_id.get());
            put_u64(buf, spec.structure_id.get());
            for value in [
                spec.region.x0,
                spec.region.y0,
                spec.region.z0,
                spec.region.x1,
                spec.region.y1,
                spec.region.z1,
            ] {
                put_i32(buf, value);
            }
            put_u16(buf, spec.cell);
        }
        // Ordering is preserved because dirty processing order can affect
        // deterministic scheduler insertion order on the next transition.
        put_u32(buf, self.dirty.len() as u32);
        for &(x, y, z) in &self.dirty {
            put_i32(buf, x);
            put_i32(buf, y);
            put_i32(buf, z);
        }
    }

    /// Determinism contract hash: content-based — per chunk, only cells
    /// differing from a fresh generation of that chunk contribute. The set
    /// of LOADED chunks then can't affect the hash (lazy generation is a
    /// pure cache), so observers like `find_blocks` don't perturb replay
    /// verification.
    pub fn hash(&self) -> u64 {
        self.hash_with_clock_identity(true)
    }

    /// Historical v5-v7 replay hash, which predates clock identity, stable
    /// semantic metadata, and persistence of the ordered dirty-cell queue.
    /// This exists only for read-only verification of previously recorded
    /// episodes; new snapshots and bundles must always use [`Self::hash`].
    pub fn legacy_hash_v7(&self) -> u64 {
        self.hash_with_clock_identity(false)
    }

    fn hash_with_clock_identity(&self, include_clock: bool) -> u64 {
        use xxhash_rust::xxh3::Xxh3;
        let mut h = Xxh3::new();
        let mut head = Vec::new();
        if include_clock {
            self.write_head(&mut head);
        } else {
            self.write_head_legacy_v7(&mut head);
        }
        h.update(&head);

        let mut keys: Vec<(i32, i32)> = self.chunks.keys().copied().collect();
        keys.sort_unstable();
        for (cx, cz) in keys {
            let cur = &self.chunks[&(cx, cz)];
            if !cur.touched {
                // pristine: zero diffs by definition — skip the worldgen
                // re-run this chunk would otherwise cost (FBM noise over
                // 32k cells per loaded chunk per hash call).
                continue;
            }
            let mut pristine = if include_clock {
                generate_chunk(self.seed, self.preset, cx, cz, self.physics.scale)
            } else {
                generate_chunk_v7(self.seed, self.preset, cx, cz, self.physics.scale)
            };
            apply_scenario(&mut pristine, cx, cz, &self.scenario);
            let mut diffs: Vec<(u32, u16)> = Vec::new();
            for i in 0..cur.blocks.len() {
                if cur.blocks[i] != pristine.blocks[i] {
                    diffs.push((i as u32, cur.blocks[i]));
                }
            }
            if !diffs.is_empty() {
                h.update(&cx.to_le_bytes());
                h.update(&cz.to_le_bytes());
                h.update(&(diffs.len() as u32).to_le_bytes());
                for (i, c) in diffs {
                    h.update(&i.to_le_bytes());
                    h.update(&c.to_le_bytes());
                }
            }
        }

        let mut tail = Vec::new();
        self.write_tail(&mut tail);
        h.update(&tail);
        if include_clock {
            let mut extension = Vec::new();
            self.write_v8_extension(&mut extension);
            h.update(&extension);
        }
        h.digest()
    }

    pub fn restore(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader::new(bytes);
        if r.take(4)? != b"VXG1" {
            return Err("bad magic".into());
        }
        let version = r.u32()?;
        if !(5..=SNAPSHOT_VERSION).contains(&version) {
            return Err(format!("unsupported snapshot version {version}"));
        }
        let tick = r.u64()?;
        let clock_config = if version >= CLOCK_SNAPSHOT_VERSION {
            ClockConfig::new(r.u64()?, r.u64()?)?
        } else {
            ClockConfig::default()
        };
        let seed = r.u64()?;
        let preset = Preset::from_u8(r.u8()?).ok_or("bad preset")?;
        let nscen = r.bounded_count("scenario", 26)?;
        let mut scenario = Vec::with_capacity(nscen);
        for _ in 0..nscen {
            let x0 = r.i32()?;
            let y0 = r.i32()?;
            let z0 = r.i32()?;
            let x1 = r.i32()?;
            let y1 = r.i32()?;
            let z1 = r.i32()?;
            let cell = r.u16()?;
            validate_cell(cell)
                .map_err(|error| format!("invalid snapshot scenario cell: {error}"))?;
            scenario.push((crate::worldgen::Region::new(x0, y0, z0, x1, y1, z1), cell));
        }
        let rs = r.u128()?;
        let ri = r.u128()?;
        let physics = Physics::read_from(&mut r)?;

        let (chunk_h, chunk_cells) = checked_spatial_scale(physics.scale)?;
        let chunk_record_bytes = 8usize
            .checked_add(usize::from(version >= 6))
            .and_then(|header| {
                chunk_cells
                    .checked_mul(std::mem::size_of::<u16>())
                    .and_then(|blocks| header.checked_add(blocks))
            })
            .ok_or_else(|| "snapshot chunk record size overflows usize".to_string())?;
        let nchunks = r.bounded_count("chunk", chunk_record_bytes)?;
        let mut chunks = XMap::default();
        chunks.reserve(nchunks);
        for _ in 0..nchunks {
            let cx = r.i32()?;
            let cz = r.i32()?;
            // v6 stores the touched flag; v5 chunks are conservatively
            // treated as touched (hash correctness never depends on it).
            let touched = if version >= 6 { r.u8()? != 0 } else { true };
            let mut blocks = vec![0u16; chunk_cells];
            for (index, block) in blocks.iter_mut().enumerate() {
                let cell = r.u16()?;
                validate_cell(cell).map_err(|error| {
                    format!("invalid snapshot chunk ({cx}, {cz}) cell at index {index}: {error}")
                })?;
                *block = cell;
            }
            chunks.insert(
                (cx, cz),
                Chunk {
                    blocks,
                    generated: true,
                    h: chunk_h,
                    touched,
                },
            );
        }

        let mut pos = [0.0; 3];
        for v in pos.iter_mut() {
            *v = r.f64()?;
        }
        let mut vel = [0.0; 3];
        for v in vel.iter_mut() {
            *v = r.f64()?;
        }
        let yaw = r.f32()?;
        let pitch = r.f32()?;
        let on_ground = r.u8()? != 0;
        let hp = r.i32()?;
        let fall_distance = r.f64()?;
        let dead = r.u8()? != 0;
        let mut inv = Inventory::new();
        for (index, s) in inv.slots.iter_mut().enumerate() {
            s.item = r.u16()?;
            s.count = r.u16()?;
            if !is_known_item(s.item) {
                return Err(format!(
                    "invalid snapshot inventory slot {index}: unknown item id {}",
                    s.item
                ));
            }
            if s.count > MAX_STACK {
                return Err(format!(
                    "invalid snapshot inventory slot {index}: count {} exceeds maximum {MAX_STACK}",
                    s.count
                ));
            }
        }
        let selected = r.u8()?;
        let suffocation_timer = r.u32()?;
        let lava_timer = r.u32()?;
        let fire_timer = r.u32()?;

        let mining = if r.u8()? != 0 {
            let x = r.i32()?;
            let y = r.i32()?;
            let z = r.i32()?;
            let progress = r.f64()?;
            Some(MiningState {
                target: (x, y, z),
                progress,
            })
        } else {
            None
        };
        let place_cooldown = r.u8()?;

        let nitems = r.bounded_count("item", 68)?;
        let mut items = Vec::with_capacity(nitems);
        for _ in 0..nitems {
            let id = r.u64()?;
            let item = r.u16()?;
            let count = r.u16()?;
            if !is_known_item(item) {
                return Err(format!(
                    "invalid snapshot item entity {id}: unknown item id {item}"
                ));
            }
            if count == 0 || count > MAX_STACK {
                return Err(format!(
                    "invalid snapshot item entity {id}: count {count} must be in 1..={MAX_STACK}"
                ));
            }
            let mut pos = [0.0; 3];
            for v in pos.iter_mut() {
                *v = r.f64()?;
            }
            let mut vel = [0.0; 3];
            for v in vel.iter_mut() {
                *v = r.f64()?;
            }
            let age = r.u64()?;
            items.push(ItemEntity {
                id,
                item,
                count,
                pos,
                vel,
                age,
            });
        }
        let nfurn = r.bounded_count("furnace", 18)?;
        let mut furnaces = XMap::default();
        furnaces.reserve(nfurn);
        for _ in 0..nfurn {
            let x = r.i32()?;
            let y = r.i32()?;
            let z = r.i32()?;
            let remaining = r.u32()?;
            let out_ready = r.u8()? != 0;
            let fuel_left = r.u8()?;
            furnaces.insert(
                (x, y, z),
                FurnaceState {
                    remaining,
                    out_ready,
                    fuel_left,
                },
            );
        }
        let next_item_id = r.u64()?;

        let falling_record_bytes = if version >= FALLING_DISTANCE_SNAPSHOT_VERSION {
            66
        } else {
            58
        };
        let nfalling = r.bounded_count("falling block", falling_record_bytes)?;
        let mut falling = Vec::with_capacity(nfalling);
        for _ in 0..nfalling {
            let id = r.u64()?;
            let block = r.u16()?;
            if !is_known_block(block) {
                return Err(format!(
                    "invalid snapshot falling block {id}: unknown block id {block}"
                ));
            }
            let mut pos = [0.0; 3];
            for v in pos.iter_mut() {
                *v = r.f64()?;
            }
            let mut vel = [0.0; 3];
            for v in vel.iter_mut() {
                *v = r.f64()?;
            }
            let fall_dist = if version >= FALLING_DISTANCE_SNAPSHOT_VERSION {
                r.f64()?
            } else {
                0.0
            };
            falling.push(FallingBlock {
                id,
                block,
                pos,
                vel,
                fall_dist,
            });
        }
        let nsched = r.bounded_count("scheduled fall", 20)?;
        let mut scheduled_falls = Vec::with_capacity(nsched);
        for _ in 0..nsched {
            let x = r.i32()?;
            let y = r.i32()?;
            let z = r.i32()?;
            let due = r.u64()?;
            scheduled_falls.push((x, y, z, due));
        }
        let read_set = |r: &mut Reader, label: &str| -> Result<XSet<(i32, i32, i32)>, String> {
            let n = r.bounded_count(label, 12)?;
            let mut s = XSet::default();
            s.reserve(n);
            for _ in 0..n {
                s.insert((r.i32()?, r.i32()?, r.i32()?));
            }
            Ok(s)
        };
        let active_fluids = read_set(&mut r, "active fluid")?;
        let circuit_cells = read_set(&mut r, "circuit cell")?;
        let next_falling_id = r.u64()?;
        let active_fire = read_set(&mut r, "active fire")?;
        let tnt_cells = read_set(&mut r, "TNT cell")?;
        let nbooms = r.bounded_count("pending explosion", 20)?;
        let mut pending_booms = Vec::with_capacity(nbooms);
        for _ in 0..nbooms {
            let x = r.i32()?;
            let y = r.i32()?;
            let z = r.i32()?;
            let due = r.u64()?;
            pending_booms.push((x, y, z, due));
        }
        let (semantic_regions, dirty) = if version >= SEMANTIC_REGIONS_SNAPSHOT_VERSION {
            let count = r.bounded_count("semantic region", 42)?;
            let mut semantic_regions = Vec::with_capacity(count);
            for _ in 0..count {
                let region_id = crate::spatial::RegionId::new(r.u64()?);
                let structure_id = crate::spatial::StructureId::new(r.u64()?);
                let region = crate::worldgen::Region::new(
                    r.i32()?,
                    r.i32()?,
                    r.i32()?,
                    r.i32()?,
                    r.i32()?,
                    r.i32()?,
                );
                let cell = r.u16()?;
                validate_cell(cell)
                    .map_err(|error| format!("invalid snapshot semantic region cell: {error}"))?;
                semantic_regions.push(SemanticRegionSpec::new(
                    region_id,
                    structure_id,
                    region,
                    cell,
                ));
            }
            validate_semantic_scenario(&scenario, &semantic_regions)?;
            let dirty = if version >= DIRTY_QUEUE_SNAPSHOT_VERSION {
                let dirty_count = r.bounded_count("dirty queue", 12)?;
                let mut dirty = Vec::with_capacity(dirty_count);
                for _ in 0..dirty_count {
                    dirty.push((r.i32()?, r.i32()?, r.i32()?));
                }
                dirty
            } else {
                Vec::new()
            };
            (semantic_regions, dirty)
        } else {
            (derive_semantic_regions(&scenario), Vec::new())
        };
        r.finish()?;

        let mut agent = Agent::new(pos, physics.scale);
        agent.vel = vel;
        agent.yaw = yaw;
        agent.pitch = pitch;
        agent.on_ground = on_ground;
        agent.hp = hp;
        agent.fall_distance = fall_distance;
        agent.dead = dead;
        agent.inventory = inv;
        agent.selected = (selected as usize).min(8);
        agent.suffocation_timer = suffocation_timer;
        agent.lava_timer = lava_timer;
        agent.fire_timer = fire_timer;

        // Derived caches: rebuilt from the loaded state, never serialized.
        let scheduled_set: XSet<(i32, i32, i32)> = scheduled_falls
            .iter()
            .map(|&(x, y, z, _)| (x, y, z))
            .collect();
        let height = (CHUNK_Y as f64 * physics.scale) as i32;
        let mut plate_count = 0u32;
        for &(x, y, z) in &circuit_cells {
            if !(WORLD_MIN_Y..height).contains(&y) {
                continue;
            }
            let (cx, cz) = (x.div_euclid(16), z.div_euclid(16));
            if let Some(c) = chunks.get(&(cx, cz)) {
                let cell = c.get(
                    x.rem_euclid(16) as usize,
                    y as usize,
                    z.rem_euclid(16) as usize,
                );
                if cell_id(cell) == PRESSURE_PLATE {
                    plate_count += 1;
                }
            }
        }

        Ok(World {
            seed,
            preset,
            physics,
            scenario,
            chunks,
            clock_config,
            tick,
            rng: Rng::from_state(rs, ri),
            agent,
            mining,
            place_cooldown,
            dirty,
            items,
            furnaces,
            events: Vec::new(),
            next_item_id,
            falling,
            scheduled_falls,
            scheduled_set,
            active_fluids,
            circuit_cells,
            plate_count,
            circuit_settled: false,
            active_fire,
            tnt_cells,
            pending_booms,
            next_falling_id,
            last_swap: None,
            semantic_regions,
        })
    }

    /// Take accumulated per-tick events (achievement hooks read these).
    pub fn drain_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }

    /// Oracle-only inventory management for scripted experts: the action
    /// space has no inventory-move verb (by design), so experts use this to
    /// pull an item into the selected hotbar slot (swap). Returns the slot
    /// now holding the item, or -1 if absent.
    pub fn swap_to_hotbar(&mut self, item: u16) -> i32 {
        let sel = self.agent.selected;
        let slots = &mut self.agent.inventory.slots;
        if slots[sel].item == item && slots[sel].count > 0 {
            return sel as i32;
        }
        if let Some(i) = slots
            .iter()
            .take(9)
            .position(|slot| slot.item == item && slot.count > 0)
        {
            self.agent.selected = i;
            self.last_swap = Some(item); // meta only: recorded, never hashed
            return i as i32;
        }
        if let Some(i) = slots
            .iter()
            .skip(9)
            .position(|slot| slot.item == item && slot.count > 0)
            .map(|i| i + 9)
        {
            slots.swap(sel, i);
            self.last_swap = Some(item);
            return sel as i32;
        }
        -1
    }

    /// Highest solid block y at column (x, z); -1 if none.
    /// Resolves the chunk once instead of hashing per cell.
    pub fn surface_y(&mut self, x: i32, z: i32) -> i32 {
        let height = self.height();
        let (cx, cz) = (x.div_euclid(16), z.div_euclid(16));
        let (lx, lz) = (x.rem_euclid(16) as usize, z.rem_euclid(16) as usize);
        let c = self.ensure_chunk(cx, cz);
        for y in (0..height).rev() {
            let cell = c.get(lx, y as usize, lz);
            let id = cell_id(cell);
            if id == DOOR && cell_state(cell) & 1 == 1 {
                continue; // open door does not collide (is_solid rule)
            }
            if block_def(id).solid {
                return y;
            }
        }
        -1
    }

    /// All cells with this block id within Chebyshev radius of (cx, cy, cz).
    /// Oracle query for scripted experts. Returns cells in exact
    /// (x, z, y)-ascending order — experts index into the result.
    /// Chunk-column scan: one chunk resolution per column instead of a
    /// hash lookup per cell (~100x fewer lookups at radius 48).
    pub fn find_blocks(
        &mut self,
        id: u16,
        cx: i32,
        cy: i32,
        cz: i32,
        radius: i32,
    ) -> Vec<(i32, i32, i32)> {
        let r = radius.clamp(0, 48);
        let (y0, y1) = ((cy - r).max(WORLD_MIN_Y), (cy + r).min(self.height() - 1));
        let mut out = Vec::new();
        let (chx0, chx1) = ((cx - r).div_euclid(16), (cx + r).div_euclid(16));
        let (chz0, chz1) = ((cz - r).div_euclid(16), (cz + r).div_euclid(16));
        for chx in chx0..=chx1 {
            for chz in chz0..=chz1 {
                let chunk = self.ensure_chunk(chx, chz);
                let (x0, x1) = ((cx - r).max(chx * 16), (cx + r).min(chx * 16 + 15));
                let (z0, z1) = ((cz - r).max(chz * 16), (cz + r).min(chz * 16 + 15));
                for x in x0..=x1 {
                    let lx = x.rem_euclid(16) as usize;
                    for z in z0..=z1 {
                        let lz = z.rem_euclid(16) as usize;
                        for y in y0..=y1 {
                            if cell_id(chunk.get(lx, y as usize, lz)) == id {
                                out.push((x, y, z));
                            }
                        }
                    }
                }
            }
        }
        // chunk-column iteration is not (x, z, y)-ordered; restore it
        out.sort_unstable_by_key(|&(x, y, z)| (x, z, y));
        out
    }
}

fn put_u16(b: &mut Vec<u8>, v: u16) {
    b.extend_from_slice(&v.to_le_bytes());
}

/// Append a u16 cell array as little-endian bytes. On LE targets this is
/// one memcpy per chunk instead of 32768 two-byte extends.
pub fn push_u16_le_blocks(buf: &mut Vec<u8>, blocks: &[u16]) {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: u16 has no padding and every bit pattern is valid, so
        // reinterpreting &[u16] as &[u8] reads the same object; on LE the
        // memory layout already equals the to_le_bytes stream.
        let bytes = unsafe {
            std::slice::from_raw_parts(blocks.as_ptr() as *const u8, std::mem::size_of_val(blocks))
        };
        buf.extend_from_slice(bytes);
    }
    #[cfg(target_endian = "big")]
    {
        for b in blocks {
            put_u16(buf, *b);
        }
    }
}
fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u128(b: &mut Vec<u8>, v: u128) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_i32(b: &mut Vec<u8>, v: i32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_f64(b: &mut Vec<u8>, v: f64) {
    b.extend_from_slice(&v.to_bits().to_le_bytes());
}
fn put_f32(b: &mut Vec<u8>, v: f32) {
    b.extend_from_slice(&v.to_bits().to_le_bytes());
}

pub(crate) struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if n > self.remaining() {
            return Err("snapshot truncated".into());
        }
        let end = self.pos + n;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    pub(crate) fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }
    fn bounded_count(&mut self, label: &str, min_record_bytes: usize) -> Result<usize, String> {
        debug_assert!(min_record_bytes > 0);
        let count = self.u32()? as usize;
        let required = count
            .checked_mul(min_record_bytes)
            .ok_or_else(|| format!("snapshot {label} count exceeds addressable byte range"))?;
        if required > self.remaining() {
            return Err(format!("snapshot {label} count exceeds remaining bytes"));
        }
        Ok(count)
    }
    pub(crate) fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    pub(crate) fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub(crate) fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(crate) fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub(crate) fn u128(&mut self) -> Result<u128, String> {
        Ok(u128::from_le_bytes(self.take(16)?.try_into().unwrap()))
    }
    pub(crate) fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(crate) fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_bits(u64::from_le_bytes(
            self.take(8)?.try_into().unwrap(),
        )))
    }
    pub(crate) fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_bits(u32::from_le_bytes(
            self.take(4)?.try_into().unwrap(),
        )))
    }

    fn finish(&self) -> Result<(), String> {
        if self.pos == self.buf.len() {
            Ok(())
        } else {
            Err("snapshot has trailing bytes".into())
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    const V8_SCENARIO_COUNT_OFFSET: usize = 4 + 4 + 8 + 8 + 8 + 8 + 1;

    fn empty_v8_physics_field_offset(field: &str) -> usize {
        let physics_offset = V8_SCENARIO_COUNT_OFFSET + 4 + 16 + 16;
        let field_index = Physics::FIELDS
            .iter()
            .position(|candidate| *candidate == field)
            .unwrap();
        physics_offset + field_index * 8
    }

    fn world_with_falling_distance(fall_dist: f64) -> World {
        let mut world = World::new(7, Preset::Void, Vec::new());
        world.falling.push(FallingBlock {
            id: 1,
            block: SAND,
            pos: [5.5, 12.5, 5.5],
            vel: [0.0, -0.25, 0.0],
            fall_dist,
        });
        world
    }

    fn legacy_falling_snapshot(version: u32) -> Vec<u8> {
        let fall_dist = 1_234.567_89_f64;
        let mut world = world_with_falling_distance(fall_dist);
        // Keep the fixture independent of the v5/v6 per-chunk touched-byte
        // difference; this test targets their shared missing falling field.
        world.chunks.clear();
        let mut bytes = world.snapshot();
        // v5-v7 end after the historical tail; v8 appends empty semantic
        // and dirty-queue counts for this fixture.
        bytes.truncate(bytes.len() - 8);
        bytes[4..8].copy_from_slice(&version.to_le_bytes());
        // Versions before v8 predate the two clock-rational fields.
        bytes.drain(16..32);

        let encoded = fall_dist.to_bits().to_le_bytes();
        let offsets: Vec<usize> = bytes
            .windows(encoded.len())
            .enumerate()
            .filter_map(|(offset, window)| (window == encoded).then_some(offset))
            .collect();
        assert_eq!(offsets.len(), 1, "fixture fall distance must be unique");
        bytes.drain(offsets[0]..offsets[0] + encoded.len());
        bytes
    }

    #[test]
    fn bounds_semantics() {
        let mut w = World::new(1, Preset::Void, Vec::new());
        assert_eq!(w.get_block(0, -5, 0), BEDROCK);
        assert_eq!(w.get_block(0, 128, 0), AIR);
        assert!(w.is_solid(3, -1, 3));
        assert!(!w.is_solid(3, 128, 3));
    }

    #[test]
    fn lazy_generation_and_set() {
        let mut w = World::new(1, Preset::Flat, Vec::new());
        let n = w.chunks.len(); // spawn column already generated
        assert_eq!(w.get_block(100, 0, 100), BEDROCK);
        assert_eq!(w.chunks.len(), n + 1);
        w.set_block(5, 10, 5, STONE);
        assert_eq!(w.get_block(5, 10, 5), STONE);
        // set outside y range is ignored
        w.set_block(5, 200, 5, STONE);
        assert_eq!(w.get_block(5, 200, 5), AIR);
    }

    #[test]
    fn invalid_public_cell_mutations_are_rejected_before_any_world_write() {
        let mut world = World::new(1, Preset::Void, Vec::new());
        let before_snapshot = world.snapshot();
        let before_hash = world.hash();

        assert_eq!(
            world.try_set_block(100, 10, 100, u16::MAX),
            Err("unknown block id 4095 in cell 65535".to_string())
        );
        assert_eq!(world.snapshot(), before_snapshot);
        assert_eq!(world.hash(), before_hash);

        // The legacy trusted-call compatibility wrapper is safe as well: it
        // performs the same pre-write guard and does not generate a chunk.
        world.set_block(100, 10, 100, u16::MAX);
        assert_eq!(world.snapshot(), before_snapshot);
        assert_eq!(world.hash(), before_hash);
    }

    #[test]
    fn negative_coords() {
        let mut w = World::new(1, Preset::Flat, Vec::new());
        w.set_block(-1, 10, -1, SAND);
        assert_eq!(w.get_block(-1, 10, -1), SAND);
        w.set_block(-16, 11, -17, GRAVEL);
        assert_eq!(w.get_block(-16, 11, -17), GRAVEL);
    }

    #[test]
    fn snapshot_roundtrip_hash() {
        let mut w = World::new(42, Preset::Default, Vec::new());
        w.set_block(3, 70, -8, PLANKS);
        for _ in 0..50 {
            w.rng.next_u64();
        }
        let h1 = w.hash();
        let bytes = w.snapshot();
        let w2 = World::restore(&bytes).unwrap();
        assert_eq!(w2.hash(), h1);
        // byte-identical re-serialization
        assert_eq!(w2.snapshot(), bytes);
    }

    #[test]
    fn hash_distinguishes_falling_distance() {
        let near = world_with_falling_distance(0.5);
        let far = world_with_falling_distance(4.5);

        assert_ne!(near.hash(), far.hash());
    }

    #[test]
    fn new_snapshots_use_version_eight() {
        let bytes = World::new(7, Preset::Void, Vec::new()).snapshot();
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());

        assert_eq!(version, 8);
    }

    #[test]
    fn legacy_falling_snapshots_restore_with_conservative_distance() {
        for version in [5, 6] {
            let restored = World::restore(&legacy_falling_snapshot(version)).unwrap();

            assert_eq!(restored.falling.len(), 1);
            assert_eq!(restored.falling[0].fall_dist, 0.0);
        }
    }

    #[test]
    fn restore_rejects_trailing_snapshot_bytes() {
        let mut bytes = World::new(7, Preset::Void, Vec::new()).snapshot();
        bytes.push(0xA5);

        assert_eq!(
            World::restore(&bytes).err().as_deref(),
            Some("snapshot has trailing bytes")
        );
    }

    #[test]
    fn restore_rejects_an_oversized_scenario_count_before_allocation() {
        let mut bytes = World::new(7, Preset::Void, Vec::new()).snapshot();
        bytes[V8_SCENARIO_COUNT_OFFSET..V8_SCENARIO_COUNT_OFFSET + 4]
            .copy_from_slice(&u32::MAX.to_le_bytes());

        assert_eq!(
            World::restore(&bytes).err().as_deref(),
            Some("snapshot scenario count exceeds remaining bytes")
        );
    }

    #[test]
    fn restore_rejects_unknown_block_ids_in_persisted_cells() {
        let mut scenario_world = World::new(7, Preset::Void, Vec::new());
        scenario_world
            .scenario
            .push((crate::worldgen::Region::new(0, 0, 0, 0, 0, 0), u16::MAX));
        assert_eq!(
            World::restore(&scenario_world.snapshot()).err().as_deref(),
            Some("invalid snapshot scenario cell: unknown block id 4095 in cell 65535")
        );

        let mut chunk_world = World::new(8, Preset::Void, Vec::new());
        chunk_world
            .chunks
            .get_mut(&(0, 0))
            .expect("spawn chunk")
            .blocks[0] = u16::MAX;
        let error = World::restore(&chunk_world.snapshot())
            .err()
            .expect("unknown persisted chunk cell must be rejected");
        assert!(
            error.contains("invalid snapshot chunk (0, 0) cell at index 0")
                && error.contains("unknown block id 4095 in cell 65535"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn restore_rejects_unknown_persisted_inventory_and_entity_ids() {
        let mut inventory_world = World::new(9, Preset::Void, Vec::new());
        inventory_world.agent.inventory.slots[0] = crate::inventory::Stack {
            item: u16::MAX,
            count: 1,
        };
        assert_eq!(
            World::restore(&inventory_world.snapshot()).err().as_deref(),
            Some("invalid snapshot inventory slot 0: unknown item id 65535")
        );

        let mut item_world = World::new(10, Preset::Void, Vec::new());
        item_world.items.push(ItemEntity {
            id: 7,
            item: u16::MAX,
            count: 1,
            pos: [0.5, 5.0, 0.5],
            vel: [0.0; 3],
            age: 0,
        });
        assert_eq!(
            World::restore(&item_world.snapshot()).err().as_deref(),
            Some("invalid snapshot item entity 7: unknown item id 65535")
        );

        let mut falling_world = World::new(11, Preset::Void, Vec::new());
        falling_world.falling.push(FallingBlock {
            id: 4,
            block: u16::MAX,
            pos: [0.5, 5.0, 0.5],
            vel: [0.0; 3],
            fall_dist: 0.0,
        });
        assert_eq!(
            World::restore(&falling_world.snapshot()).err().as_deref(),
            Some("invalid snapshot falling block 4: unknown block id 65535")
        );
    }

    #[test]
    fn restore_rejects_a_non_finite_spatial_scale() {
        let mut bytes = World::new(7, Preset::Void, Vec::new()).snapshot();
        let scale_offset = empty_v8_physics_field_offset("scale");
        bytes[scale_offset..scale_offset + 8]
            .copy_from_slice(&f64::INFINITY.to_bits().to_le_bytes());

        assert_eq!(
            World::restore(&bytes).err().as_deref(),
            Some("invalid physics field 'scale': value must be finite")
        );
    }

    #[test]
    fn restore_rejects_scales_outside_the_world_constructor_contract() {
        let valid = World::new(7, Preset::Void, Vec::new()).snapshot();
        let scale_offset = empty_v8_physics_field_offset("scale");
        let too_tall = (i32::MAX as f64 + 1.0) / CHUNK_Y as f64;

        for scale in [0.0, 0.5, 1.1, too_tall] {
            let mut bytes = valid.clone();
            bytes[scale_offset..scale_offset + 8].copy_from_slice(&scale.to_bits().to_le_bytes());

            let error = World::restore(&bytes)
                .err()
                .expect("scale must be rejected");
            assert!(
                error.starts_with("invalid spatial scale"),
                "scale {scale} returned {error}"
            );
        }
    }

    #[test]
    fn restore_rejects_noncanonical_physics_values() {
        let valid = World::new(7, Preset::Void, Vec::new()).snapshot();
        let cases = [
            ("gravity", f64::NAN, "value must be finite"),
            ("water_spread", -1.0, "expected an integer"),
            ("water_period", 0.0, "expected an integer"),
            ("agent_mass", 0.0, "value must be positive"),
        ];

        for (field, value, expected) in cases {
            let mut bytes = valid.clone();
            let offset = empty_v8_physics_field_offset(field);
            bytes[offset..offset + 8].copy_from_slice(&value.to_bits().to_le_bytes());

            let error = World::restore(&bytes)
                .err()
                .expect("noncanonical physics value must be rejected");
            assert!(
                error.contains(expected),
                "field {field}={value} returned {error}"
            );
        }
    }

    #[test]
    fn active_falling_snapshot_preserves_future_simulation() {
        let mut original = World::new(7, Preset::Void, Vec::new());
        original.set_block(5, 5, 5, STONE);
        original.set_block(5, 12, 5, SAND);
        original.agent.pos = [5.5, 6.0, 5.5];
        let idle = crate::tick::Action::default();
        for _ in 0..6 {
            crate::tick::step(&mut original, &idle);
        }
        assert_eq!(original.falling.len(), 1);
        assert!(original.falling[0].fall_dist > 0.0);

        let bytes = original.snapshot();
        let mut restored = World::restore(&bytes).unwrap();
        assert_eq!(restored.snapshot(), bytes);
        assert_eq!(restored.hash(), original.hash());

        for _ in 0..60 {
            crate::tick::step(&mut original, &idle);
            crate::tick::step(&mut restored, &idle);
            assert_eq!(restored.hash(), original.hash());
        }
        assert_eq!(restored.agent.hp, original.agent.hp);
    }

    #[test]
    fn snapshot_roundtrip_preserves_extended_simulation_state() {
        let scenario = vec![(
            crate::worldgen::Region::new(-1, 3, -1, 1, 3, 1),
            make_cell(WATER, 2),
        )];
        let mut world = World::new_scaled(99, Preset::Void, scenario, 2.0);
        world.tick = 41;
        world.agent.pos = [2.5, 12.0, -3.5];
        world.agent.vel = [0.1, -0.2, 0.3];
        world.agent.yaw = 135.0;
        world.agent.pitch = -20.0;
        world.agent.on_ground = true;
        world.agent.hp = 13;
        world.agent.fall_distance = 2.25;
        world.agent.inventory.add(LOG, 3);
        world.agent.selected = 4;
        world.agent.suffocation_timer = 7;
        world.agent.lava_timer = 8;
        world.agent.fire_timer = 9;
        world.mining = Some(MiningState {
            target: (-2, 5, 7),
            progress: 0.75,
        });
        world.place_cooldown = 2;
        world.spawn_item(DIRT, 6, [4.5, 9.0, -1.5]);
        world.items[0].age = 123;
        world.furnaces.insert(
            (3, 4, 5),
            FurnaceState {
                remaining: 17,
                out_ready: false,
                fuel_left: 2,
            },
        );
        world.falling.push(FallingBlock {
            id: 8,
            block: GRAVEL,
            pos: [8.5, 30.5, 8.5],
            vel: [0.0, -0.4, 0.0],
            fall_dist: 4.0,
        });
        world.scheduled_falls.push((-5, 8, 2, 44));
        world.scheduled_set.insert((-5, 8, 2));
        world.set_block(7, 6, 7, PRESSURE_PLATE);
        world.active_fire.insert((8, 6, 7));
        world.tnt_cells.insert((9, 6, 7));
        world.pending_booms.push((9, 6, 7, 50));
        world.next_falling_id = 9;
        world.last_swap = Some(LOG);

        let bytes = world.snapshot();
        let restored = World::restore(&bytes).unwrap();

        assert_eq!(restored.snapshot(), bytes);
        assert_eq!(restored.hash(), world.hash());
        assert_eq!(restored.mining, world.mining);
        assert!(restored.scheduled_set.contains(&(-5, 8, 2)));
        assert_eq!(restored.plate_count, 1);
        assert_eq!(restored.dirty, world.dirty);
        assert!(restored.events.is_empty());
        assert_eq!(restored.last_swap, None);
    }

    #[test]
    fn v8_hashes_dirty_order_and_restore_rejects_a_malformed_dirty_count() {
        let clean = World::new(17, Preset::Void, Vec::new());
        let mut dirty = World::restore(&clean.snapshot()).unwrap();
        dirty.dirty.push((123_456, 34, -654_321));

        assert_ne!(dirty.hash(), clean.hash());
        assert_eq!(dirty.legacy_hash_v7(), clean.legacy_hash_v7());

        let mut malformed = dirty.snapshot();
        // Empty semantic set, followed by dirty count and one 12-byte coord.
        let dirty_count_offset = malformed.len() - 16;
        malformed[dirty_count_offset..dirty_count_offset + 4]
            .copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            World::restore(&malformed).err().as_deref(),
            Some("snapshot dirty queue count exceeds remaining bytes")
        );
    }

    #[test]
    fn restore_reports_malformed_snapshot_classes() {
        const PRESET_OFFSET: usize = 4 + 4 + 8 + 8 + 8 + 8;

        let valid = World::new(5, Preset::Void, Vec::new()).snapshot();
        let mut bad_magic = valid.clone();
        bad_magic[..4].copy_from_slice(b"NOPE");
        let mut old_version = valid.clone();
        old_version[4..8].copy_from_slice(&4u32.to_le_bytes());
        let mut future_version = valid.clone();
        future_version[4..8].copy_from_slice(&9u32.to_le_bytes());
        let mut bad_preset = valid.clone();
        bad_preset[PRESET_OFFSET] = u8::MAX;
        let truncated = valid[..valid.len() - 1].to_vec();

        for (bytes, expected) in [
            (Vec::new(), "snapshot truncated"),
            (bad_magic, "bad magic"),
            (old_version, "unsupported snapshot version 4"),
            (future_version, "unsupported snapshot version 9"),
            (bad_preset, "bad preset"),
            (truncated, "snapshot truncated"),
        ] {
            assert_eq!(World::restore(&bytes).err().as_deref(), Some(expected));
        }
    }

    #[test]
    fn read_only_peek_does_not_generate_a_chunk() {
        let mut world = World::new(3, Preset::Flat, Vec::new());
        let loaded = world.chunks.len();

        assert_eq!(world.peek_block(160, 0, -160), AIR);
        assert_eq!(world.chunks.len(), loaded);
        assert_eq!(world.get_block(160, 0, -160), BEDROCK);
        assert_eq!(world.chunks.len(), loaded + 1);
    }

    #[test]
    fn voxel_window_is_axis_ordered_and_fills_vertical_bounds() {
        let mut world = World::new(4, Preset::Void, Vec::new());
        world.agent.pos = [0.5, 10.0, 0.5];
        world.set_block(-10, 7, -10, STONE);
        world.set_block(0, 11, 0, DIRT);
        world.set_block(10, 17, 10, LOG);

        let window = world.voxel_window();

        assert_eq!(window.len(), 21 * 11 * 21);
        assert_eq!(window[0], STONE);
        assert_eq!(window[10 * 11 * 21 + 4 * 21 + 10], DIRT);
        assert_eq!(window[window.len() - 1], LOG);

        world.agent.pos[1] = -1.0;
        let below_ground = world.voxel_window();
        for dx in 0..21 {
            let column = dx * 11 * 21;
            assert!(below_ground[column..column + 4 * 21]
                .iter()
                .all(|&cell| cell == BEDROCK));
        }

        let top = world.height() - 1;
        world.set_block(0, top, 0, STONE);
        world.agent.pos[1] = world.height() as f64;
        let above_world = world.voxel_window();
        let center_column = 10 * 11 * 21;
        assert_eq!(above_world[center_column + 2 * 21 + 10], STONE);
        assert_eq!(above_world[center_column + 3 * 21 + 10], AIR);
        for dx in 0..21 {
            let column = dx * 11 * 21;
            assert!(above_world[column + 3 * 21..column + 11 * 21]
                .iter()
                .all(|&cell| cell == AIR));
        }
    }

    #[test]
    fn block_queries_honor_collision_and_ordering_contracts() {
        let mut world = World::new(6, Preset::Void, Vec::new());
        world.set_block(2, 3, 2, STONE);
        world.set_block(2, 8, 2, make_cell(DOOR, 1));
        world.set_block(3, 8, 2, DOOR);
        assert_eq!(world.surface_y(2, 2), 3);
        assert_eq!(world.surface_y(3, 2), 8);
        assert_eq!(world.surface_y(4, 2), -1);

        for &(x, y, z) in &[(0, 6, 0), (-1, 4, 0), (-1, 5, -1), (-2, 7, 0)] {
            world.set_block(x, y, z, DIAMOND_ORE);
        }
        assert_eq!(
            world.find_blocks(DIAMOND_ORE, -1, 5, 0, 2),
            vec![(-2, 7, 0), (-1, 5, -1), (-1, 4, 0), (0, 6, 0)]
        );
        assert_eq!(
            world.find_blocks(DIAMOND_ORE, -1, 5, -1, -10),
            vec![(-1, 5, -1)]
        );
        assert!(world.find_blocks(DIAMOND_ORE, 0, -20, 0, 2).is_empty());
    }

    #[test]
    fn hotbar_swap_covers_selected_hotbar_inventory_and_missing_items() {
        let mut world = World::new(8, Preset::Void, Vec::new());
        world.agent.selected = 2;
        world.agent.inventory.slots[2] = crate::inventory::Stack {
            item: DIRT,
            count: 1,
        };
        world.agent.inventory.slots[5] = crate::inventory::Stack {
            item: STONE,
            count: 2,
        };
        world.agent.inventory.slots[10] = crate::inventory::Stack {
            item: LOG,
            count: 3,
        };

        assert_eq!(world.swap_to_hotbar(DIRT), 2);
        assert_eq!(world.last_swap, None);
        assert_eq!(world.swap_to_hotbar(STONE), 5);
        assert_eq!(world.agent.selected, 5);
        assert_eq!(world.last_swap, Some(STONE));
        assert_eq!(world.swap_to_hotbar(LOG), 5);
        assert_eq!(world.agent.inventory.slots[5].item, LOG);
        assert_eq!(world.agent.inventory.slots[10].item, STONE);
        assert_eq!(world.swap_to_hotbar(ITEM_DIAMOND), -1);
    }
}
