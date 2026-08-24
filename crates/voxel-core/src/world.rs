//! World: chunk columns + agent + global state, with canonical
//! snapshot/restore/hash (the determinism contract).

use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use xxhash_rust::xxh3::Xxh3;

use crate::block::*;
use crate::chunk::*;
use crate::entity::Agent;
use crate::inventory::Inventory;
use crate::item::ItemEntity;
use crate::loose::FallingBlock;
use crate::physics::Physics;
use crate::recipe::FurnaceState;
use crate::rng::Rng;
use crate::worldgen::{apply_scenario, generate_chunk, Preset, ScenarioSpec};

/// Snapshot format v6: per-chunk `touched` flag byte. v5 snapshots still
/// load (every chunk conservatively marked touched).
pub const SNAPSHOT_VERSION: u32 = 6;

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
    pub physics: Physics,
    pub scenario: ScenarioSpec,
    pub chunks: XMap<(i32, i32), Chunk>,
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
        Self::new_scaled(seed, preset, scenario, 1.0)
    }

    /// `scale` = cells per meter (1.0 = MC 1 m cells; 2.0 = 0.5 m cells).
    /// Physical world size is invariant: chunk height becomes 128*scale,
    /// noise is sampled per meter, all spatial constants multiply by scale.
    pub fn new_scaled(seed: u64, preset: Preset, scenario: ScenarioSpec, scale: f64) -> Self {
        assert!(
            scale >= 1.0 && (128.0 * scale).fract() == 0.0,
            "scale must be >= 1 with 128*scale integral (got {scale})"
        );
        let mut w = World {
            seed,
            preset,
            physics: Physics::default().spatially_scaled(scale),
            scenario,
            chunks: XMap::default(),
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
                            w.scheduled_falls.push((x, y, z, 1));
                            w.scheduled_set.insert((x, y, z));
                        }
                        if is_circuit {
                            if w.circuit_cells.insert((x, y, z)) && id == PRESSURE_PLATE {
                                w.plate_count += 1;
                            }
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

    /// World height in cells: 128 * scale.
    pub fn height(&self) -> i32 {
        (CHUNK_Y as f64 * self.physics.scale) as i32
    }

    /// Highest solid top + 1 at column (x, z); fallback y=8*scale for void.
    pub fn find_spawn(&mut self, x: i32, z: i32) -> [f64; 3] {
        for y in (0..self.height()).rev() {
            if self.is_solid(x, y, z) {
                return [x as f64 + 0.5, (y + 1) as f64, z as f64 + 0.5];
            }
        }
        let s = self.physics.scale;
        [x as f64 + 0.5, 8.0 * s, z as f64 + 0.5]
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

    pub fn set_block(&mut self, x: i32, y: i32, z: i32, cell: u16) {
        if !(WORLD_MIN_Y..self.height()).contains(&y) {
            return;
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
        self.fluid_at(p[0].floor() as i32, p[1].floor() as i32, p[2].floor() as i32)
    }

    /// 21(x) x 11(y) x 21(z) window centered on the eye column,
    /// y in [eye_y-4, eye_y+6]. Flat vec, index [dx][dy][dz] (dz fastest).
    ///
    /// Iterates in chunk-aligned runs: each (dx, z-run) resolves its chunk
    /// once instead of hashing (cx, cz) per cell — 4851 map lookups become
    /// at most 27 chunk resolutions plus flat array indexing.
    pub fn voxel_window(&mut self) -> Vec<u16> {
        let eye = self.agent.eye();
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
                    let dz1 = if i + 1 < n_runs { runs[i + 1].1 - 1 } else { 10 };
                    let c = &self.chunks[&(cx, cz)];
                    let mut lz = (ez + dz0).rem_euclid(16) as usize;
                    for _ in dz0..=dz1 {
                        out.push(c.get(lx, yu, lz));
                        lz += 1;
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
        buf
    }

    fn write_head(&self, buf: &mut Vec<u8>) {
        put_u64(buf, self.tick);
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

    /// Determinism contract hash: content-based — per chunk, only cells
    /// differing from a fresh generation of that chunk contribute. The set
    /// of LOADED chunks then can't affect the hash (lazy generation is a
    /// pure cache), so observers like `find_blocks` don't perturb replay
    /// verification.
    pub fn hash(&self) -> u64 {
        use xxhash_rust::xxh3::Xxh3;
        let mut h = Xxh3::new();
        let mut head = Vec::new();
        self.write_head(&mut head);
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
            let mut pristine = generate_chunk(self.seed, self.preset, cx, cz, self.physics.scale);
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
        let seed = r.u64()?;
        let preset = Preset::from_u8(r.u8()?).ok_or("bad preset")?;
        let nscen = r.u32()? as usize;
        let mut scenario = Vec::with_capacity(nscen);
        for _ in 0..nscen {
            let x0 = r.i32()?;
            let y0 = r.i32()?;
            let z0 = r.i32()?;
            let x1 = r.i32()?;
            let y1 = r.i32()?;
            let z1 = r.i32()?;
            let cell = r.u16()?;
            scenario.push((crate::worldgen::Region::new(x0, y0, z0, x1, y1, z1), cell));
        }
        let rs = r.u128()?;
        let ri = r.u128()?;
        let physics = Physics::read_from(&mut r)?;

        let nchunks = r.u32()? as usize;
        let chunk_h = (CHUNK_Y as f64 * physics.scale) as usize;
        let mut chunks = XMap::default();
        chunks.reserve(nchunks);
        for _ in 0..nchunks {
            let cx = r.i32()?;
            let cz = r.i32()?;
            // v6 stores the touched flag; v5 chunks are conservatively
            // treated as touched (hash correctness never depends on it).
            let touched = if version >= 6 { r.u8()? != 0 } else { true };
            let mut blocks = vec![0u16; CHUNK_X * chunk_h * CHUNK_Z];
            for b in blocks.iter_mut() {
                *b = r.u16()?;
            }
            chunks.insert((cx, cz), Chunk { blocks, generated: true, h: chunk_h, touched });
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
        for s in inv.slots.iter_mut() {
            s.item = r.u16()?;
            s.count = r.u16()?;
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
            Some(MiningState { target: (x, y, z), progress })
        } else {
            None
        };
        let place_cooldown = r.u8()?;

        let nitems = r.u32()? as usize;
        let mut items = Vec::with_capacity(nitems);
        for _ in 0..nitems {
            let id = r.u64()?;
            let item = r.u16()?;
            let count = r.u16()?;
            let mut pos = [0.0; 3];
            for v in pos.iter_mut() {
                *v = r.f64()?;
            }
            let mut vel = [0.0; 3];
            for v in vel.iter_mut() {
                *v = r.f64()?;
            }
            let age = r.u64()?;
            items.push(ItemEntity { id, item, count, pos, vel, age });
        }
        let nfurn = r.u32()? as usize;
        let mut furnaces = XMap::default();
        furnaces.reserve(nfurn);
        for _ in 0..nfurn {
            let x = r.i32()?;
            let y = r.i32()?;
            let z = r.i32()?;
            let remaining = r.u32()?;
            let out_ready = r.u8()? != 0;
            let fuel_left = r.u8()?;
            furnaces.insert((x, y, z), FurnaceState { remaining, out_ready, fuel_left });
        }
        let next_item_id = r.u64()?;

        let nfalling = r.u32()? as usize;
        let mut falling = Vec::with_capacity(nfalling);
        for _ in 0..nfalling {
            let id = r.u64()?;
            let block = r.u16()?;
            let mut pos = [0.0; 3];
            for v in pos.iter_mut() {
                *v = r.f64()?;
            }
            let mut vel = [0.0; 3];
            for v in vel.iter_mut() {
                *v = r.f64()?;
            }
            let fall_dist = r.f64()?;
            falling.push(FallingBlock { id, block, pos, vel, fall_dist });
        }
        let nsched = r.u32()? as usize;
        let mut scheduled_falls = Vec::with_capacity(nsched);
        for _ in 0..nsched {
            let x = r.i32()?;
            let y = r.i32()?;
            let z = r.i32()?;
            let due = r.u64()?;
            scheduled_falls.push((x, y, z, due));
        }
        let read_set = |r: &mut Reader| -> Result<XSet<(i32, i32, i32)>, String> {
            let n = r.u32()? as usize;
            let mut s = XSet::default();
            s.reserve(n);
            for _ in 0..n {
                s.insert((r.i32()?, r.i32()?, r.i32()?));
            }
            Ok(s)
        };
        let active_fluids = read_set(&mut r)?;
        let circuit_cells = read_set(&mut r)?;
        let next_falling_id = r.u64()?;
        let active_fire = read_set(&mut r)?;
        let tnt_cells = read_set(&mut r)?;
        let nbooms = r.u32()? as usize;
        let mut pending_booms = Vec::with_capacity(nbooms);
        for _ in 0..nbooms {
            let x = r.i32()?;
            let y = r.i32()?;
            let z = r.i32()?;
            let due = r.u64()?;
            pending_booms.push((x, y, z, due));
        }

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
        let scheduled_set: XSet<(i32, i32, i32)> =
            scheduled_falls.iter().map(|&(x, y, z, _)| (x, y, z)).collect();
        let height = (CHUNK_Y as f64 * physics.scale) as i32;
        let mut plate_count = 0u32;
        for &(x, y, z) in &circuit_cells {
            if !(WORLD_MIN_Y..height).contains(&y) {
                continue;
            }
            let (cx, cz) = (x.div_euclid(16), z.div_euclid(16));
            if let Some(c) = chunks.get(&(cx, cz)) {
                let cell = c.get(x.rem_euclid(16) as usize, y as usize, z.rem_euclid(16) as usize);
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
            tick,
            rng: Rng::from_state(rs, ri),
            agent,
            mining,
            place_cooldown,
            dirty: Vec::new(),
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
        for i in 0..9 {
            if slots[i].item == item && slots[i].count > 0 {
                self.agent.selected = i;
                self.last_swap = Some(item); // meta only: recorded, never hashed
                return i as i32;
            }
        }
        for i in 9..36 {
            if slots[i].item == item && slots[i].count > 0 {
                slots.swap(sel, i);
                self.last_swap = Some(item);
                return sel as i32;
            }
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
    pub fn find_blocks(&mut self, id: u16, cx: i32, cy: i32, cz: i32, radius: i32) -> Vec<(i32, i32, i32)> {
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
        if self.pos + n > self.buf.len() {
            return Err("snapshot truncated".into());
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
