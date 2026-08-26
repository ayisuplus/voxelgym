//! PyO3 bindings: `voxelgym_rs.PyWorld` (single sim) and
//! `voxelgym_rs.PyWorldBatch` (rayon-parallel vector stepping).

use numpy::{PyArray1, PyArray2, PyArray3, PyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use voxel_core::block::cell_id;
use voxel_core::tick::{raycast_target, Action};
use voxel_core::worldgen::{Preset, Region, ScenarioSpec};
use voxel_core::World;

type ActionTuple = (u8, u8, u8, u8, u8, u8, u8, u8, u8, u8);
type RenderOutput<'py> = (
    Bound<'py, PyArray3<u8>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<u16>>,
    Bound<'py, PyArray3<f32>>,
);

fn validate_render_dimensions(width: usize, height: usize) -> PyResult<()> {
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err(format!(
            "render dimensions must be positive, got {width}x{height}"
        )));
    }
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "render dimensions are too large, got {width}x{height}"
            ))
        })?;
    Ok(())
}

fn frame_to_numpy<'py>(py: Python<'py>, frame: voxel_view::Frame) -> PyResult<RenderOutput<'py>> {
    let voxel_view::Frame {
        width,
        height,
        rgb,
        depth,
        seg,
        normals,
    } = frame;
    validate_render_dimensions(width, height)?;

    let rgb = ndarray::Array3::from_shape_vec((height, width, 3), rgb)
        .map_err(|err| PyValueError::new_err(format!("invalid RGB frame: {err}")))?;
    let depth = ndarray::Array2::from_shape_vec((height, width), depth)
        .map_err(|err| PyValueError::new_err(format!("invalid depth frame: {err}")))?;
    let seg = ndarray::Array2::from_shape_vec((height, width), seg)
        .map_err(|err| PyValueError::new_err(format!("invalid segmentation frame: {err}")))?;
    let normals = ndarray::Array3::from_shape_vec((height, width, 3), normals)
        .map_err(|err| PyValueError::new_err(format!("invalid normals frame: {err}")))?;

    Ok((
        PyArray3::from_owned_array(py, rgb),
        PyArray2::from_owned_array(py, depth),
        PyArray2::from_owned_array(py, seg),
        PyArray3::from_owned_array(py, normals),
    ))
}

fn to_action(t: &ActionTuple) -> Action {
    Action::from_parts(&[t.0, t.1, t.2, t.3, t.4, t.5, t.6, t.7, t.8, t.9])
}

fn parse_scenario(raw: Option<Vec<(i32, i32, i32, i32, i32, i32, u16)>>) -> ScenarioSpec {
    raw.unwrap_or_default()
        .into_iter()
        .map(|(x0, y0, z0, x1, y1, z1, cell)| (Region::new(x0, y0, z0, x1, y1, z1), cell))
        .collect()
}

fn make_world(seed: u64, preset: &str, scenario: ScenarioSpec) -> PyResult<World> {
    make_world_scaled(seed, preset, scenario, 1.0)
}

fn make_world_scaled(seed: u64, preset: &str, scenario: ScenarioSpec, scale: f64) -> PyResult<World> {
    let p = Preset::from_str(preset)
        .ok_or_else(|| PyValueError::new_err(format!("unknown preset '{preset}'")))?;
    if !(scale >= 1.0 && (128.0 * scale).fract() == 0.0) {
        return Err(PyValueError::new_err(
            format!("scale must be >= 1 with 128*scale integral (got {scale})"),
        ));
    }
    Ok(World::new_scaled(seed, p, scenario, scale))
}

fn apply_physics(w: &mut World, overrides: Option<std::collections::HashMap<String, f64>>) -> PyResult<()> {
    if let Some(map) = overrides {
        for (k, v) in map {
            w.physics.set(&k, v).map_err(PyValueError::new_err)?;
        }
    }
    Ok(())
}

#[pyclass]
pub struct PyWorld {
    world: World,
}

#[pymethods]
impl PyWorld {
    #[new]
    #[pyo3(signature = (seed, preset = "default", scenario = None, physics = None, scale = 1.0))]
    fn new(seed: u64, preset: &str, scenario: Option<Vec<(i32, i32, i32, i32, i32, i32, u16)>>,
           physics: Option<std::collections::HashMap<String, f64>>, scale: f64) -> PyResult<Self> {
        let mut w = make_world_scaled(seed, preset, parse_scenario(scenario), scale)?;
        apply_physics(&mut w, physics)?;
        Ok(PyWorld { world: w })
    }

    /// Read a physics field (see voxel_core::physics::Physics::FIELDS).
    fn get_physics(&self, key: &str) -> PyResult<f64> {
        self.world
            .physics
            .get(key)
            .ok_or_else(|| PyValueError::new_err(format!("unknown physics field '{key}'")))
    }

    /// Advance one tick. Action = 10-tuple per the gym contract.
    fn step(&mut self, action: ActionTuple) {
        let a = to_action(&action);
        voxel_core::step(&mut self.world, &a);
    }

    /// (21, 11, 21) uint16 raw cells, axes (x, y, z), eye column centered,
    /// y in [-4, +6] relative to the eye cell.
    fn obs_voxels<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray3<u16>> {
        let flat = self.world.voxel_window();
        let arr = ndarray::Array3::from_shape_vec((21, 11, 21), flat).unwrap();
        numpy::PyArray3::from_owned_array(py, arr)
    }

    /// (36, 2) uint16 (item_id, count).
    fn obs_inventory<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u16>> {
        let mut rows = Vec::with_capacity(36);
        for s in &self.world.agent.inventory.slots {
            rows.push(vec![s.item, s.count]);
        }
        PyArray2::from_vec2(py, &rows).unwrap()
    }

    /// Raw (21, 11, 21) u16 window as bytes (C order) — the recorder's hot
    /// path: skips the numpy roundtrip of obs_voxels().tobytes().
    fn obs_voxels_bytes<'py>(&mut self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        let flat = self.world.voxel_window();
        let mut buf = Vec::with_capacity(flat.len() * 2);
        voxel_core::world::push_u16_le_blocks(&mut buf, &flat);
        pyo3::types::PyBytes::new(py, &buf)
    }

    /// Raw (36, 2) u16 inventory as bytes — same recorder fast path.
    fn obs_inventory_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        let mut buf = Vec::with_capacity(36 * 4);
        for s in &self.world.agent.inventory.slots {
            buf.extend_from_slice(&s.item.to_le_bytes());
            buf.extend_from_slice(&s.count.to_le_bytes());
        }
        pyo3::types::PyBytes::new(py, &buf)
    }

    /// (6,) float32: x, y, z, yaw deg, pitch deg, on_ground in {0, 1}.
    fn obs_pose<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let a = &self.world.agent;
        PyArray1::from_vec(
            py,
            vec![
                a.pos[0] as f32,
                a.pos[1] as f32,
                a.pos[2] as f32,
                a.yaw,
                a.pitch,
                a.on_ground as u8 as f32,
            ],
        )
    }

    /// (2,) uint16: crosshair block id, distance in centi-cells (450 = reach).
    /// No target -> (0, 450).
    fn obs_raycast<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<u16>> {
        match raycast_target(&mut self.world) {
            Some(h) => PyArray1::from_vec(
                py,
                vec![cell_id(h.cell), (h.dist * 100.0).round().min(450.0) as u16],
            ),
            None => PyArray1::from_vec(py, vec![0u16, 450u16]),
        }
    }

    // ---- helpers for tests / experts / tasks (oracle-grade) ----

    fn give(&mut self, item: u16, count: u16) {
        self.world.agent.inventory.add(item, count);
    }

    fn count_item(&self, item: u16) -> u16 {
        self.world.agent.inventory.count(item)
    }

    fn get_block(&mut self, x: i32, y: i32, z: i32) -> u16 {
        self.world.get_block(x, y, z)
    }

    fn set_block(&mut self, x: i32, y: i32, z: i32, cell: u16) {
        self.world.set_block(x, y, z, cell);
    }

    fn teleport(&mut self, x: f64, y: f64, z: f64) {
        self.world.agent.pos = [x, y, z];
        self.world.agent.vel = [0.0; 3];
        self.world.agent.fall_distance = 0.0;
    }

    fn agent_pos(&self) -> (f64, f64, f64) {
        (self.world.agent.pos[0], self.world.agent.pos[1], self.world.agent.pos[2])
    }

    fn hp(&self) -> i32 {
        self.world.agent.hp
    }

    fn dead(&self) -> bool {
        self.world.agent.dead
    }

    fn tick(&self) -> u64 {
        self.world.tick
    }

    /// All cells with this block id within Chebyshev radius of the agent.
    /// Oracle query for scripted experts.
    fn find_blocks(&mut self, id: u16, radius: i32) -> Vec<(i32, i32, i32)> {
        let p = self.world.agent.pos;
        self.world.find_blocks(
            id,
            p[0].floor() as i32,
            p[1].floor() as i32,
            p[2].floor() as i32,
            radius,
        )
    }

    // ---- determinism contract ----

    fn snapshot<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, &self.world.snapshot())
    }

    fn restore(&mut self, bytes: &[u8]) -> PyResult<()> {
        self.world = World::restore(bytes).map_err(PyValueError::new_err)?;
        Ok(())
    }

    fn hash(&self) -> u64 {
        self.world.hash()
    }

    /// Drain per-tick events as (kind, a, b) tuples:
    /// ("pickup", item, count) ("craft", recipe, out) ("mine", block, 0)
    /// ("smelt", item, 0).
    fn drain_events(&mut self) -> Vec<(&'static str, u16, u16)> {
        self.world
            .drain_events()
            .into_iter()
            .map(|e| match e {
                voxel_core::Event::ItemPicked { item, count } => ("pickup", item, count),
                voxel_core::Event::Crafted { recipe, out, .. } => ("craft", recipe as u16, out),
                voxel_core::Event::BlockMined { id } => ("mine", id, 0),
                voxel_core::Event::Smelted { item } => ("smelt", item, 0),
            })
            .collect()
    }

    /// Highest solid block y at (x, z); -1 if none.
    fn surface_y(&mut self, x: i32, z: i32) -> i32 {
        self.world.surface_y(x, z)
    }

    /// Oracle-only: pull `item` into the selected hotbar slot (swap).
    /// Returns the slot now holding it, or -1 if absent.
    fn swap_to_hotbar(&mut self, item: u16) -> i32 {
        self.world.swap_to_hotbar(item)
    }

    /// Crosshair block: ((x,y,z), block_id, dist_centicells), None if no hit.
    fn crosshair(&mut self) -> Option<((i32, i32, i32), u16, u16)> {
        raycast_target(&mut self.world).map(|h| {
            ((h.x, h.y, h.z), cell_id(h.cell), (h.dist * 100.0).round().min(450.0) as u16)
        })
    }

    /// Positions of loose item entities of this item id (oracle pickup aid).
    fn drops_of(&self, item: u16) -> Vec<(f64, f64, f64)> {
        self.world
            .items
            .iter()
            .filter(|i| i.item == item)
            .map(|i| (i.pos[0], i.pos[1], i.pos[2]))
            .collect()
    }

    /// Furnace state at a cell: (remaining_ticks, out_ready, fuel_left).
    fn furnace_state(&self, x: i32, y: i32, z: i32) -> (u32, bool, u8) {
        let st = self
            .world
            .furnaces
            .get(&(x, y, z))
            .copied()
            .unwrap_or_default();
        (st.remaining, st.out_ready, st.fuel_left)
    }

    /// Render the agent view: (rgb (128,128,3) u8, depth (128,128) f32 cells,
    /// seg (128,128) u16 block ids, normals (128,128,3) f32 unit axis,
    /// [0,0,0] on sky miss). SKY_SEG=0xFFFF on miss.
    fn render<'py>(&mut self, py: Python<'py>) -> PyResult<RenderOutput<'py>> {
        let f = py.allow_threads(|| voxel_view::render(&mut self.world, 128, 128, 90.0));
        frame_to_numpy(py, f)
    }

    /// Free camera render: (origin, yaw_deg, pitch_deg). Positive pitch
    /// looks down. Same channels as render().
    #[pyo3(signature = (origin, yaw_deg, pitch_deg, width=128, height=128, fov_deg=90.0))]
    fn render_pose<'py>(
        &mut self,
        py: Python<'py>,
        origin: (f64, f64, f64),
        yaw_deg: f64,
        pitch_deg: f64,
        width: usize,
        height: usize,
        fov_deg: f64,
    ) -> PyResult<RenderOutput<'py>> {
        validate_render_dimensions(width, height)?;
        let f = py.allow_threads(|| {
            voxel_view::render_from(
                &mut self.world,
                [origin.0, origin.1, origin.2],
                yaw_deg,
                pitch_deg,
                width,
                height,
                fov_deg,
            )
        });
        frame_to_numpy(py, f)
    }

    /// Block-id -> [r,g,b] palette (row i = block id i; SKY_SEG 0xFFFF is
    /// handled client-side).
    fn palette<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<u8>> {
        let rows: Vec<[u8; 3]> = voxel_core::block::BLOCKS.iter().map(|d| d.color).collect();
        let flat: Vec<u8> = rows.iter().flat_map(|c| c.iter().copied()).collect();
        let arr = ndarray::Array2::from_shape_vec((rows.len(), 3), flat).unwrap();
        numpy::PyArray2::from_owned_array(py, arr)
    }

    /// Cast a single DDA ray; returns hit distance in cells, or -1.0.
    /// Solids only (wires/torchs don't block camera rays).
    fn cast_ray(
        &mut self,
        origin: (f64, f64, f64),
        dir: (f64, f64, f64),
        max_dist: f64,
    ) -> f64 {
        let w = &mut self.world;
        let hit = voxel_core::raycast::dda_with(
            [origin.0, origin.1, origin.2],
            [dir.0, dir.1, dir.2],
            max_dist,
            |x, y, z| w.get_block(x, y, z),
            |c| voxel_core::block::block_def(voxel_core::block::cell_id(c)).solid,
        );
        hit.map(|h| h.dist).unwrap_or(-1.0)
    }

    /// Spinning multi-beam LiDAR scan from the agent eye (or an explicit
    /// pose for a fixed emitter block). Returns (range, intensity, seg),
    /// each (channels, azimuth_steps); range 0 = no return, seg SKY=0xFFFF.
    /// The scan is a pure function of (world state, cfg, frame_idx) — noise
    /// is position-hashed, so replays are byte-identical.
    #[pyo3(signature = (channels, azimuth_steps, min_elev_deg, max_elev_deg, max_range, noise_sigma=0.0, dropout_p=0.0, noise_seed=0, frame_idx=0, origin=None, yaw_deg=None))]
    fn lidar_scan<'py>(
        &mut self,
        py: Python<'py>,
        channels: usize,
        azimuth_steps: usize,
        min_elev_deg: f64,
        max_elev_deg: f64,
        max_range: f64,
        noise_sigma: f64,
        dropout_p: f64,
        noise_seed: u64,
        frame_idx: u64,
        origin: Option<(f64, f64, f64)>,
        yaw_deg: Option<f64>,
    ) -> (
        Bound<'py, numpy::PyArray2<f32>>,
        Bound<'py, numpy::PyArray2<f32>>,
        Bound<'py, numpy::PyArray2<u16>>,
    ) {
        let cfg = voxel_view::lidar::LidarConfig {
            channels,
            azimuth_steps,
            min_elev_deg,
            max_elev_deg,
            max_range,
            noise_sigma,
            dropout_p,
            noise_seed,
        };
        let org = origin
            .map(|(x, y, z)| [x, y, z])
            .unwrap_or_else(|| self.world.agent.eye());
        let yaw = yaw_deg.unwrap_or(self.world.agent.yaw as f64);
        let s = py.allow_threads(|| {
            voxel_view::lidar::scan(&mut self.world, &cfg, org, yaw, frame_idx)
        });
        let shape = (s.channels, s.azimuth_steps);
        let range = ndarray::Array2::from_shape_vec(shape, s.range).unwrap();
        let inten = ndarray::Array2::from_shape_vec(shape, s.intensity).unwrap();
        let seg = ndarray::Array2::from_shape_vec(shape, s.seg).unwrap();
        (
            numpy::PyArray2::from_owned_array(py, range),
            numpy::PyArray2::from_owned_array(py, inten),
            numpy::PyArray2::from_owned_array(py, seg),
        )
    }

    /// Biome at column (x, z): 0 ocean, 1 plains, 2 desert, 3 hills, 4 volcanic.
    fn biome_at(&self, x: i32, z: i32) -> u8 {
        voxel_core::worldgen::biome_at(self.world.seed, x, z).as_u8()
    }

    /// Take the pending inventory-swap event (0 if none since last take).
    /// Part of the behavior trace: recorded by the Recorder, applied by replay.
    fn take_swap(&mut self) -> u16 {
        self.world.last_swap.take().unwrap_or(0)
    }
}

/// A shard of worlds stepped in parallel with rayon (GIL released).
#[pyclass]
pub struct PyWorldBatch {
    worlds: Vec<World>,
}

impl PyWorldBatch {
    /// Parallel step over all worlds (GIL released); per-world dead flags.
    fn step_actions(&mut self, py: Python<'_>, acts: Vec<Action>) -> Vec<bool> {
        py.allow_threads(|| {
            self.worlds
                .par_iter_mut()
                .zip(acts.par_iter())
                .map(|(w, a)| {
                    voxel_core::step(w, a);
                    w.agent.dead
                })
                .collect()
        })
    }
}

#[pymethods]
impl PyWorldBatch {
    #[new]
    #[pyo3(signature = (specs))]
    fn new(specs: Vec<(u64, String)>) -> PyResult<Self> {
        let mut worlds = Vec::with_capacity(specs.len());
        for (seed, preset) in &specs {
            worlds.push(make_world(*seed, preset, Vec::new())?);
        }
        Ok(PyWorldBatch { worlds })
    }

    fn len(&self) -> usize {
        self.worlds.len()
    }

    /// Step every world once. Returns per-world dead flags.
    fn step_batch(&mut self, py: Python<'_>, actions: Vec<ActionTuple>) -> Vec<bool> {
        let acts: Vec<Action> = actions.iter().map(to_action).collect();
        self.step_actions(py, acts)
    }

    /// Same as step_batch but takes a contiguous (N, 10) uint8 numpy array —
    /// avoids building N Python tuples (the dominant Python-side cost when
    /// stepping large batches).
    fn step_batch_np<'py>(
        &mut self,
        py: Python<'py>,
        actions: numpy::PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Vec<bool>> {
        let arr = actions.as_array();
        if arr.ndim() != 2 || arr.ncols() != 10 {
            return Err(PyValueError::new_err("actions must have shape (N, 10)"));
        }
        let mut acts = Vec::with_capacity(arr.nrows());
        for row in arr.rows() {
            let mut parts = [0u8; 10];
            if let Some(s) = row.as_slice() {
                parts.copy_from_slice(s);
            } else {
                for (p, v) in parts.iter_mut().zip(row.iter()) {
                    *p = *v;
                }
            }
            acts.push(Action::from_parts(&parts));
        }
        Ok(self.step_actions(py, acts))
    }

    /// Stacked (N, 21, 11, 21) uint16 voxel windows.
    fn obs_voxels_batch<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray4<u16>> {
        let flats: Vec<Vec<u16>> = py.allow_threads(|| {
            self.worlds
                .par_iter_mut()
                .map(|w| w.voxel_window())
                .collect()
        });
        let n = flats.len();
        let mut all = Vec::with_capacity(n * 21 * 11 * 21);
        for f in flats {
            all.extend_from_slice(&f);
        }
        let arr = ndarray::Array4::from_shape_vec((n, 21, 11, 21), all).unwrap();
        PyArray4::from_owned_array(py, arr)
    }

    /// Stacked (N, 36, 2) uint16 inventories.
    fn obs_inventory_batch<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<u16>> {
        let all: Vec<Vec<Vec<u16>>> = self
            .worlds
            .iter()
            .map(|w| w.agent.inventory.slots.iter().map(|s| vec![s.item, s.count]).collect())
            .collect();
        PyArray3::from_vec3(py, &all).unwrap()
    }

    /// Stacked (N, 6) float32 poses.
    fn obs_pose_batch<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let all: Vec<Vec<f32>> = self
            .worlds
            .iter()
            .map(|w| {
                let a = &w.agent;
                vec![
                    a.pos[0] as f32,
                    a.pos[1] as f32,
                    a.pos[2] as f32,
                    a.yaw,
                    a.pitch,
                    a.on_ground as u8 as f32,
                ]
            })
            .collect();
        PyArray2::from_vec2(py, &all).unwrap()
    }

    /// Stacked (N, 2) uint16 raycasts.
    fn obs_raycast_batch<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<u16>> {
        let all: Vec<Vec<u16>> = self
            .worlds
            .iter_mut()
            .map(|w| match raycast_target(w) {
                Some(h) => vec![cell_id(h.cell), (h.dist * 100.0).round().min(450.0) as u16],
                None => vec![0u16, 450u16],
            })
            .collect();
        PyArray2::from_vec2(py, &all).unwrap()
    }

    fn hashes(&self) -> Vec<u64> {
        self.worlds.iter().map(|w| w.hash()).collect()
    }
}

#[pyfunction]
fn block_id(name: &str) -> PyResult<u16> {
    voxel_core::block::block_id_by_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown block '{name}'")))
}

#[pyfunction]
fn item_id(name: &str) -> PyResult<u16> {
    voxel_core::block::item_id_by_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown item '{name}'")))
}

#[pymodule]
fn voxelgym_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWorld>()?;
    m.add_class::<PyWorldBatch>()?;
    m.add_function(wrap_pyfunction!(block_id, m)?)?;
    m.add_function(wrap_pyfunction!(item_id, m)?)?;
    Ok(())
}
