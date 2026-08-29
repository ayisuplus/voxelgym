//! PyO3 bindings: `voxelgym_rs.PyWorld` (single sim) and
//! `voxelgym_rs.PyWorldBatch` (rayon-parallel vector stepping).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use numpy::{PyArray1, PyArray2, PyArray3, PyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use rayon::prelude::*;
use voxel_core::block::{cell_id, cell_state, Fluid, DOOR};
use voxel_core::clock::{ClockConfig, SimClock};
use voxel_core::entity::aabb_collides;
use voxel_core::spatial::{CellCoord, RegionId, StructureId, WorldPos};
use voxel_core::tick::{raycast_target, Action};
use voxel_core::trace::{
    EventKind, InterventionOutcome, InterventionSpec, Phase, RootCause, StateDelta, StepOutcome,
    SubjectRef, TraceLevel, TraceState, TraceValue, WorldEvent,
};
use voxel_core::worldgen::{Preset, Region, ScenarioSpec, SemanticRegionSpec};
use voxel_core::World;

type ActionTuple = (u8, u8, u8, u8, u8, u8, u8, u8, u8, u8);
type ScenarioRegionTuple = (i32, i32, i32, i32, i32, i32, u16);
type SemanticRegionTuple = (u64, u64, i32, i32, i32, i32, i32, i32, u16);
type RenderOutput<'py> = (
    Bound<'py, PyArray3<u8>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<u16>>,
    Bound<'py, PyArray3<f32>>,
);
type LidarOutput<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<u16>>,
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

fn parse_scenario(raw: Option<Vec<ScenarioRegionTuple>>) -> ScenarioSpec {
    raw.unwrap_or_default()
        .into_iter()
        .map(|(x0, y0, z0, x1, y1, z1, cell)| (Region::new(x0, y0, z0, x1, y1, z1), cell))
        .collect()
}

fn parse_semantic_regions(
    raw: Option<Vec<SemanticRegionTuple>>,
) -> Option<Vec<SemanticRegionSpec>> {
    raw.map(|regions| {
        regions
            .into_iter()
            .map(|(region_id, structure_id, x0, y0, z0, x1, y1, z1, cell)| {
                SemanticRegionSpec::new(
                    RegionId::new(region_id),
                    StructureId::new(structure_id),
                    Region::new(x0, y0, z0, x1, y1, z1),
                    cell,
                )
            })
            .collect()
    })
}

fn make_world(seed: u64, preset: &str, scenario: ScenarioSpec) -> PyResult<World> {
    make_world_scaled(seed, preset, scenario, 1.0)
}

fn make_world_scaled(
    seed: u64,
    preset: &str,
    scenario: ScenarioSpec,
    scale: f64,
) -> PyResult<World> {
    make_world_scaled_with_clock(seed, preset, scenario, scale, ClockConfig::default(), None)
}

fn make_world_scaled_with_clock(
    seed: u64,
    preset: &str,
    scenario: ScenarioSpec,
    scale: f64,
    clock: ClockConfig,
    semantic_regions: Option<Vec<SemanticRegionSpec>>,
) -> PyResult<World> {
    let p = Preset::from_str(preset)
        .ok_or_else(|| PyValueError::new_err(format!("unknown preset '{preset}'")))?;
    if !(scale >= 1.0 && (128.0 * scale).fract() == 0.0) {
        return Err(PyValueError::new_err(format!(
            "scale must be >= 1 with 128*scale integral (got {scale})"
        )));
    }
    for (_, cell) in &scenario {
        voxel_core::block::validate_cell(*cell).map_err(PyValueError::new_err)?;
    }
    match semantic_regions {
        Some(regions) => {
            World::new_scaled_with_clock_and_semantic_regions(seed, p, regions, scale, clock)
                .map_err(PyValueError::new_err)
        }
        None => Ok(World::new_scaled_with_clock(
            seed, p, scenario, scale, clock,
        )),
    }
}

fn clock_to_dict<'py>(py: Python<'py>, clock: SimClock) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    let config = clock.config();
    let (elapsed_numerator, elapsed_denominator) = clock.elapsed_fraction();
    result.set_item("tick", clock.tick())?;
    result.set_item("dt_numerator", config.numerator())?;
    result.set_item("dt_denominator", config.denominator())?;
    result.set_item("seconds_per_tick", config.seconds_per_tick())?;
    result.set_item("elapsed_numerator", elapsed_numerator)?;
    result.set_item("elapsed_denominator", elapsed_denominator)?;
    result.set_item("elapsed_seconds", clock.elapsed_seconds())?;
    Ok(result)
}

fn clock_with_context_to_dict<'py>(
    py: Python<'py>,
    clock: SimClock,
    horizon_tick: Option<u64>,
    sample_tick: Option<u64>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = clock_to_dict(py, clock)?;
    if let Some(horizon_tick) = horizon_tick {
        result.set_item("remaining_ticks", clock.remaining_ticks(horizon_tick))?;
        result.set_item("remaining_seconds", clock.remaining_seconds(horizon_tick))?;
    }
    if let Some(sample_tick) = sample_tick {
        let (sample_numerator, sample_denominator) = clock.sample_time_fraction(sample_tick);
        result.set_item("sample_tick", sample_tick)?;
        result.set_item("sample_time_numerator", sample_numerator)?;
        result.set_item("sample_time_denominator", sample_denominator)?;
        result.set_item(
            "sample_time_seconds",
            sample_tick as f64 * clock.config().seconds_per_tick(),
        )?;
        result.set_item("data_age_ticks", clock.data_age_ticks(sample_tick))?;
        result.set_item("data_age_seconds", clock.data_age_seconds(sample_tick))?;
    }
    Ok(result)
}

fn parse_trace_level(value: &str) -> PyResult<TraceLevel> {
    match value {
        "off" => Ok(TraceLevel::Off),
        "events" => Ok(TraceLevel::Events),
        "full" => Ok(TraceLevel::Full),
        _ => Err(PyValueError::new_err(format!(
            "unknown trace level '{value}'; expected 'off', 'events', or 'full'"
        ))),
    }
}

fn phase_name(value: Phase) -> &'static str {
    match value {
        Phase::Intervention => "intervention",
        Phase::AgentAction => "agent_action",
        Phase::EntityIntegration => "entity_integration",
        Phase::Scheduled => "scheduled",
        Phase::Fluid => "fluid",
        Phase::Fire => "fire",
        Phase::Circuit => "circuit",
        Phase::Tnt => "tnt",
        Phase::ItemLogic => "item_logic",
        Phase::Observation => "observation",
    }
}

fn event_kind_name(value: EventKind) -> &'static str {
    match value {
        EventKind::ActionApplied => "action_applied",
        EventKind::InterventionApplied => "intervention_applied",
        EventKind::AgentMoved => "agent_moved",
        EventKind::VelocityChanged => "velocity_changed",
        EventKind::Collision => "collision",
        EventKind::Damage => "damage",
        EventKind::Death => "death",
        EventKind::InventoryChanged => "inventory_changed",
        EventKind::ItemPicked => "item_picked",
        EventKind::Crafted => "crafted",
        EventKind::BlockMined => "block_mined",
        EventKind::BlockPlaced => "block_placed",
        EventKind::BlockChanged => "block_changed",
        EventKind::BlockFallScheduled => "block_fall_scheduled",
        EventKind::BlockFell => "block_fell",
        EventKind::Smelted => "smelted",
        EventKind::FluidChanged => "fluid_changed",
        EventKind::Ignited => "ignited",
        EventKind::Extinguished => "extinguished",
        EventKind::CircuitChanged => "circuit_changed",
        EventKind::TntPrimed => "tnt_primed",
        EventKind::Explosion => "explosion",
        EventKind::EntitySpawned => "entity_spawned",
        EventKind::EntityDespawned => "entity_despawned",
        EventKind::StateChanged => "state_changed",
    }
}

fn subject_to_dict<'py>(py: Python<'py>, value: &SubjectRef) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    match value {
        SubjectRef::World => result.set_item("kind", "world")?,
        SubjectRef::Agent(id) => {
            result.set_item("kind", "agent")?;
            result.set_item("id", id.get())?;
        }
        SubjectRef::Cell(at) => {
            result.set_item("kind", "cell")?;
            result.set_item("at", (at.x, at.y, at.z))?;
        }
        SubjectRef::Entity(id) => {
            result.set_item("kind", "entity")?;
            result.set_item("id", id.get())?;
        }
        SubjectRef::InventorySlot(slot) => {
            result.set_item("kind", "inventory_slot")?;
            result.set_item("slot", *slot)?;
        }
        SubjectRef::Scheduler(name) => {
            result.set_item("kind", "scheduler")?;
            result.set_item("name", *name)?;
        }
    }
    Ok(result)
}

fn root_cause_to_dict<'py>(py: Python<'py>, value: &RootCause) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    match value {
        RootCause::Action { branch_id, tick } => {
            result.set_item("kind", "action")?;
            result.set_item("branch_id", *branch_id)?;
            result.set_item("tick", *tick)?;
        }
        RootCause::Intervention {
            branch_id,
            intervention_id,
        } => {
            result.set_item("kind", "intervention")?;
            result.set_item("branch_id", *branch_id)?;
            result.set_item("intervention_id", *intervention_id)?;
        }
        RootCause::Periodic { tick, mechanism } => {
            result.set_item("kind", "periodic")?;
            result.set_item("tick", *tick)?;
            result.set_item("mechanism", *mechanism)?;
        }
        RootCause::Exogenous {
            branch_id,
            tick,
            ordinal,
            mechanism,
        } => {
            result.set_item("kind", "exogenous")?;
            result.set_item("branch_id", *branch_id)?;
            result.set_item("tick", *tick)?;
            result.set_item("ordinal", *ordinal)?;
            result.set_item("mechanism", *mechanism)?;
        }
    }
    Ok(result)
}

fn event_to_dict<'py>(py: Python<'py>, value: &WorldEvent) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("id", value.id)?;
    result.set_item("tick", value.tick)?;
    result.set_item("phase", phase_name(value.phase))?;
    result.set_item("kind", event_kind_name(value.kind))?;
    if let Some(actor) = &value.actor {
        result.set_item("actor", subject_to_dict(py, actor)?)?;
    } else {
        result.set_item("actor", py.None())?;
    }
    if let Some(target) = &value.target {
        result.set_item("target", subject_to_dict(py, target)?)?;
    } else {
        result.set_item("target", py.None())?;
    }
    if let Some(location) = value.location {
        result.set_item("location", (location.x, location.y, location.z))?;
    } else {
        result.set_item("location", py.None())?;
    }
    result.set_item("mechanism", value.mechanism)?;
    result.set_item("parent_ids", &value.parent_ids)?;
    result.set_item("root_cause", root_cause_to_dict(py, &value.root_cause)?)?;
    Ok(result)
}

fn set_trace_value(
    py: Python<'_>,
    result: &Bound<'_, PyDict>,
    key: &str,
    value: &TraceValue,
) -> PyResult<()> {
    match value {
        TraceValue::None => result.set_item(key, py.None()),
        TraceValue::Bool(value) => result.set_item(key, *value),
        TraceValue::I64(value) => result.set_item(key, *value),
        TraceValue::U64(value) => result.set_item(key, *value),
        TraceValue::Cell(value) => result.set_item(key, *value),
        TraceValue::CellSequence(values) => result.set_item(
            key,
            values
                .iter()
                .map(|value| (value.x, value.y, value.z))
                .collect::<Vec<_>>(),
        ),
        TraceValue::ItemStack { item, count } => {
            let stack = PyDict::new(py);
            stack.set_item("kind", "item_stack")?;
            stack.set_item("item", *item)?;
            stack.set_item("count", *count)?;
            result.set_item(key, stack)
        }
        TraceValue::F64Bits(bits) => {
            let exact = PyDict::new(py);
            exact.set_item("kind", "f64_bits")?;
            exact.set_item("bits", *bits)?;
            exact.set_item("value", f64::from_bits(*bits))?;
            result.set_item(key, exact)
        }
        TraceValue::Vec3Bits(bits) => {
            let exact = PyDict::new(py);
            exact.set_item("kind", "vec3_bits")?;
            exact.set_item("bits", *bits)?;
            exact.set_item("value", bits.map(f64::from_bits))?;
            result.set_item(key, exact)
        }
    }
}

fn delta_to_dict<'py>(py: Python<'py>, value: &StateDelta) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("event_id", value.event_id)?;
    result.set_item("subject", subject_to_dict(py, &value.subject)?)?;
    result.set_item("field_or_cell", value.field_or_cell)?;
    set_trace_value(py, &result, "before", &value.before)?;
    set_trace_value(py, &result, "after", &value.after)?;
    Ok(result)
}

fn step_outcome_to_dict<'py>(
    py: Python<'py>,
    outcome: &StepOutcome,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("clock_before", clock_to_dict(py, outcome.clock_before)?)?;
    result.set_item("clock_after", clock_to_dict(py, outcome.clock_after)?)?;
    result.set_item("before_hash", outcome.before_hash)?;
    result.set_item("after_hash", outcome.after_hash)?;
    let events = outcome
        .events
        .iter()
        .map(|event| event_to_dict(py, event))
        .collect::<PyResult<Vec<_>>>()?;
    let deltas = outcome
        .deltas
        .iter()
        .map(|delta| delta_to_dict(py, delta))
        .collect::<PyResult<Vec<_>>>()?;
    result.set_item("events", events)?;
    result.set_item("deltas", deltas)?;
    Ok(result)
}

fn parse_intervention(spec: &Bound<'_, PyDict>) -> PyResult<InterventionSpec> {
    let tag = if let Some(value) = spec.get_item("kind")? {
        value.extract::<String>()?
    } else if let Some(value) = spec.get_item("type")? {
        value.extract::<String>()?
    } else {
        return Err(PyValueError::new_err(
            "intervention requires a string 'kind' tag",
        ));
    };
    match tag.as_str() {
        "set_cell" => {
            let at = parse_i32_triplet(
                &spec
                    .get_item("at")?
                    .ok_or_else(|| PyValueError::new_err("set_cell intervention requires 'at'"))?,
                "set_cell 'at'",
            )?;
            let cell = spec
                .get_item("cell")?
                .ok_or_else(|| PyValueError::new_err("set_cell intervention requires 'cell'"))?
                .extract::<u16>()?;
            Ok(InterventionSpec::SetCell {
                at: CellCoord::from(at),
                cell,
            })
        }
        "teleport_agent" => {
            let position = parse_f64_triplet(
                &spec.get_item("position")?.ok_or_else(|| {
                    PyValueError::new_err("teleport_agent intervention requires 'position'")
                })?,
                "teleport_agent 'position'",
            )?;
            let position = WorldPos::new(position.0, position.1, position.2);
            if CellCoord::try_from_world_pos(position).is_none() {
                return Err(PyValueError::new_err(
                    "teleport_agent position must be finite and fit world coordinates",
                ));
            }
            Ok(InterventionSpec::TeleportAgent { position })
        }
        "set_agent_velocity" => {
            let velocity = parse_f64_triplet(
                &spec.get_item("velocity")?.ok_or_else(|| {
                    PyValueError::new_err("set_agent_velocity intervention requires 'velocity'")
                })?,
                "set_agent_velocity 'velocity'",
            )?;
            let velocity = [velocity.0, velocity.1, velocity.2];
            if !velocity.iter().all(|component| component.is_finite()) {
                return Err(PyValueError::new_err(
                    "set_agent_velocity values must be finite",
                ));
            }
            Ok(InterventionSpec::SetAgentVelocity { velocity })
        }
        "give_item" => {
            let item = spec
                .get_item("item")?
                .ok_or_else(|| PyValueError::new_err("give_item intervention requires 'item'"))?
                .extract::<u16>()?;
            let count = spec
                .get_item("count")?
                .ok_or_else(|| PyValueError::new_err("give_item intervention requires 'count'"))?
                .extract::<u16>()?;
            Ok(InterventionSpec::GiveItem { item, count })
        }
        "swap_to_hotbar" => {
            let item = spec
                .get_item("item")?
                .ok_or_else(|| {
                    PyValueError::new_err("swap_to_hotbar intervention requires 'item'")
                })?
                .extract::<u16>()?;
            Ok(InterventionSpec::SwapToHotbar { item })
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown intervention kind '{tag}'"
        ))),
    }
}

fn parse_i32_triplet(value: &Bound<'_, PyAny>, name: &str) -> PyResult<(i32, i32, i32)> {
    let values = value.extract::<Vec<i32>>()?;
    if values.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must contain exactly three coordinates"
        )));
    }
    Ok((values[0], values[1], values[2]))
}

fn parse_f64_triplet(value: &Bound<'_, PyAny>, name: &str) -> PyResult<(f64, f64, f64)> {
    let values = value.extract::<Vec<f64>>()?;
    if values.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must contain exactly three coordinates"
        )));
    }
    Ok((values[0], values[1], values[2]))
}

fn intervention_outcome_to_dict<'py>(
    py: Python<'py>,
    outcome: &InterventionOutcome,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("clock", clock_to_dict(py, outcome.clock)?)?;
    result.set_item("before_hash", outcome.before_hash)?;
    result.set_item("after_hash", outcome.after_hash)?;
    if let Some(event) = &outcome.event {
        result.set_item("event", event_to_dict(py, event)?)?;
    } else {
        result.set_item("event", py.None())?;
    }
    let deltas = outcome
        .deltas
        .iter()
        .map(|delta| delta_to_dict(py, delta))
        .collect::<PyResult<Vec<_>>>()?;
    result.set_item("deltas", deltas)?;
    Ok(result)
}

fn agent_can_stand(
    world: &mut World,
    at: CellCoord,
    half_width: f64,
    height: f64,
    lattice_offset: [f64; 2],
) -> bool {
    if at.y <= 0 {
        return false;
    }
    let center_x = at.x as f64 + lattice_offset[0];
    let center_z = at.z as f64 + lattice_offset[1];
    let min = [center_x - half_width, at.y as f64, center_z - half_width];
    let max = [
        center_x + half_width,
        at.y as f64 + height,
        center_z + half_width,
    ];
    if aabb_collides(world, min, max) {
        return false;
    }
    // Probe the exact body footprint immediately below the candidate feet;
    // this uses the same collision policy as agent integration (including
    // open doors and other non-solid stateful blocks).
    aabb_collides(
        world,
        [min[0], min[1] - 1e-6, min[2]],
        [max[0], min[1], max[2]],
    )
}

/// The live controller treats water at the centre of the feet cell as a
/// supported swimming state even when there is no solid voxel immediately
/// below the body. Keep that test identical to `World::fluid_at_feet` so the
/// graph does not invent buoyancy from water merely touching the AABB edge.
fn agent_is_swimming(world: &mut World, at: CellCoord, lattice_offset: [f64; 2]) -> bool {
    let center_x = at.x as f64 + lattice_offset[0];
    let center_z = at.z as f64 + lattice_offset[1];
    world.fluid_at(center_x.floor() as i32, at.y, center_z.floor() as i32) == Some(Fluid::Water)
}

/// Whether a cell names a physically occupiable navigation pose. Dry poses
/// need real voxel support; unsupported poses are admitted only inside the
/// finite set of water cells that activates the live swimming controller.
/// This keeps deep pools navigable without turning the implicit y=0 safety
/// plane (or an unbounded void) into graph support.
fn agent_can_occupy(
    world: &mut World,
    at: CellCoord,
    half_width: f64,
    height: f64,
    lattice_offset: [f64; 2],
) -> bool {
    if agent_can_stand(world, at, half_width, height, lattice_offset) {
        return true;
    }
    if at.y <= 0 || !agent_is_swimming(world, at, lattice_offset) {
        return false;
    }
    let center_x = at.x as f64 + lattice_offset[0];
    let center_z = at.z as f64 + lattice_offset[1];
    !aabb_collides(
        world,
        [center_x - half_width, at.y as f64, center_z - half_width],
        [
            center_x + half_width,
            at.y as f64 + height,
            center_z + half_width,
        ],
    )
}

/// Whether a graph pose is supported by an actual solid voxel rather than
/// only the engine's implicit lower world boundary. The latter is a physics
/// safety plane, not a scene-graph surface; treating it as an unbounded
/// navigable floor makes bounded path queries expand forever in void worlds.
fn agent_has_voxel_support(
    world: &mut World,
    at: CellCoord,
    half_width: f64,
    lattice_offset: [f64; 2],
) -> bool {
    let center_x = at.x as f64 + lattice_offset[0];
    let center_z = at.z as f64 + lattice_offset[1];
    let min_x = (center_x - half_width).floor() as i32;
    let max_x = (center_x + half_width - 1e-9).floor() as i32;
    let min_z = (center_z - half_width).floor() as i32;
    let max_z = (center_z + half_width - 1e-9).floor() as i32;
    (min_x..=max_x).any(|x| (min_z..=max_z).any(|z| world.is_solid(x, at.y.saturating_sub(1), z)))
}

/// Conservative edge predicate for the same body geometry and collision
/// oracle used by live agent integration. Horizontal edges may include a
/// physical one-metre jump; sampled sweep checks prevent paths through low
/// ceilings or solid corners. Drops are discovered separately by executing
/// the real action/physics transition on a snapshot branch.
fn agent_can_traverse(
    world: &mut World,
    from: CellCoord,
    to: CellCoord,
    half_width: f64,
    height: f64,
    lattice_offset: [f64; 2],
) -> bool {
    if !agent_can_occupy(world, to, half_width, height, lattice_offset) {
        return false;
    }
    let dx = (to.x - from.x).abs();
    let dz = (to.z - from.z).abs();
    if dx + dz != 1 {
        return false;
    }
    let scale = world.scale();
    let rise = to.y - from.y;
    let max_rise = scale.ceil() as i32;
    if rise > max_rise || rise < 0 {
        return false;
    }
    // Swimming is currently proven only for level cardinal movement: held
    // move+jump reaches such a neighbour under the real water controller.
    // Do not reuse the dry jump parabola for vertical water edges; those need
    // their own controller rollout before they can be exposed as reachable.
    if rise != 0
        && (agent_is_swimming(world, from, lattice_offset)
            || agent_is_swimming(world, to, lattice_offset))
    {
        return false;
    }

    let from_center = [
        from.x as f64 + lattice_offset[0],
        from.y as f64,
        from.z as f64 + lattice_offset[1],
    ];
    let to_center = [
        to.x as f64 + lattice_offset[0],
        to.y as f64,
        to.z as f64 + lattice_offset[1],
    ];
    for sample in 1..=8 {
        let t = sample as f64 / 8.0;
        let feet_y = if rise > 0 {
            // The live jump reaches a little over one metre.  The arc here
            // is deliberately conservative about headroom while still
            // clearing a one-metre ledge.
            from_center[1] + (to_center[1] - from_center[1]) * t + 1.1 * scale * 4.0 * t * (1.0 - t)
        } else {
            from_center[1]
        };
        let center_x = from_center[0] + (to_center[0] - from_center[0]) * t;
        let center_z = from_center[2] + (to_center[2] - from_center[2]) * t;
        if aabb_collides(
            world,
            [center_x - half_width, feet_y, center_z - half_width],
            [
                center_x + half_width,
                feet_y + height,
                center_z + half_width,
            ],
        ) {
            return false;
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn rollout_drop_with_duration(
    snapshot: &[u8],
    from: CellCoord,
    yaw: u8,
    lattice_offset: [f64; 2],
    drive_default_ticks: u64,
    half_width: f64,
    height: f64,
) -> Option<CellCoord> {
    let mut branch = World::restore(snapshot).ok()?;
    if branch.agent.dead {
        return None;
    }
    branch.agent.pos = [
        from.x as f64 + lattice_offset[0],
        from.y as f64,
        from.z as f64 + lattice_offset[1],
    ];
    branch.agent.vel = [0.0; 3];
    branch.agent.on_ground = true;
    branch.agent.fall_distance = 0.0;

    let drive_steps = branch
        .clock_config()
        .ticks_for_default_ticks(drive_default_ticks)
        .clamp(1, 2_000);
    let max_steps = branch
        .clock_config()
        .ticks_for_default_ticks(80)
        .clamp(drive_steps, 2_000);
    let selected = branch.agent.selected.min(8) as u8;
    for index in 0..max_steps {
        voxel_core::step(
            &mut branch,
            &Action {
                mv: u8::from(index < drive_steps),
                yaw,
                pitch: 4,
                hotbar: selected,
                ..Action::default()
            },
        );
        if branch.agent.dead {
            return None;
        }
        // The collision solver can keep `on_ground` true while stepping
        // down a one-cell ledge, so a lower supported pose is a completed
        // descent even when no sampled tick observed an airborne state.
        if branch.agent.on_ground {
            let landing = CellCoord::from_world_pos(WorldPos::from(branch.agent.pos));
            if landing.y < from.y
                && agent_can_stand(&mut branch, landing, half_width, height, lattice_offset)
                && agent_has_voxel_support(&mut branch, landing, half_width, lattice_offset)
            {
                return Some(landing);
            }
            // Stop a failed short probe once ground friction has settled it
            // back on the source level; continuing to 80 ticks cannot turn
            // it into a descent without another movement action.
            if index >= drive_steps
                && landing.y >= from.y
                && branch.agent.vel[0].abs() < 1e-9
                && branch.agent.vel[2].abs() < 1e-9
            {
                return None;
            }
        }
    }
    None
}

/// Discover lower landings by running the actual agent controller, collision
/// solver, fall damage, and seven-phase transition on canonical snapshot
/// branches. The shortest successful cardinal input realizes a cautious
/// step-down; a committed 1.25-second input additionally captures reachable
/// landings across wider gaps.
fn rollout_drop_neighbors(
    world: &World,
    from: CellCoord,
    dx: i32,
    dz: i32,
    lattice_offset: [f64; 2],
) -> Vec<CellCoord> {
    let yaw = match (dx, dz) {
        (0, 1) => 0,
        (-1, 0) => 6,
        (0, -1) => 12,
        (1, 0) => 18,
        _ => return Vec::new(),
    };

    let snapshot = world.snapshot();
    let mut landings = Vec::with_capacity(2);
    let mut cautious_duration = None;
    for drive_default_ticks in 1..=25 {
        if let Some(landing) = rollout_drop_with_duration(
            &snapshot,
            from,
            yaw,
            lattice_offset,
            drive_default_ticks,
            world.agent.half_width,
            world.agent.height,
        ) {
            landings.push(landing);
            cautious_duration = Some(drive_default_ticks);
            break;
        }
    }
    if cautious_duration != Some(25) {
        if let Some(landing) = rollout_drop_with_duration(
            &snapshot,
            from,
            yaw,
            lattice_offset,
            25,
            world.agent.half_width,
            world.agent.height,
        ) {
            landings.push(landing);
        }
    }
    landings.sort_unstable();
    landings.dedup();
    landings
}

/// Prove one vertical swimming edge by executing the real water controller
/// on an isolated canonical snapshot. Upward motion holds jump; downward
/// motion releases it and relies on the controller's buoyant sink. A graph
/// edge is returned only after the live body actually enters the adjacent
/// water cell while alive and without horizontal cell drift.
fn rollout_vertical_swim_neighbor(
    world: &World,
    from: CellCoord,
    direction: i32,
    lattice_offset: [f64; 2],
    half_width: f64,
    height: f64,
) -> Option<CellCoord> {
    if !matches!(direction, -1 | 1) {
        return None;
    }
    let mut branch = World::restore(&world.snapshot()).ok()?;
    if branch.agent.dead {
        return None;
    }
    branch.agent.pos = [
        from.x as f64 + lattice_offset[0],
        from.y as f64,
        from.z as f64 + lattice_offset[1],
    ];
    branch.agent.vel = [0.0; 3];
    branch.agent.on_ground = false;
    branch.agent.fall_distance = 0.0;

    let target = CellCoord::new(from.x, from.y + direction, from.z);
    let max_steps = branch
        .clock_config()
        .ticks_for_default_ticks(40)
        .clamp(1, 2_000);
    let selected = branch.agent.selected.min(8) as u8;
    for _ in 0..max_steps {
        voxel_core::step(
            &mut branch,
            &Action {
                jump: direction > 0,
                pitch: 4,
                hotbar: selected,
                ..Action::default()
            },
        );
        if branch.agent.dead {
            return None;
        }
        let reached = CellCoord::from_world_pos(WorldPos::from(branch.agent.pos));
        if reached.x != from.x || reached.z != from.z {
            return None;
        }
        if reached == target {
            return (agent_is_swimming(&mut branch, reached, lattice_offset)
                && agent_can_occupy(&mut branch, reached, half_width, height, lattice_offset))
            .then_some(reached);
        }
        if (direction > 0 && reached.y > target.y) || (direction < 0 && reached.y < target.y) {
            return None;
        }
    }
    None
}

fn movement_neighbors(
    world: &mut World,
    from: CellCoord,
    half_width: f64,
    height: f64,
    lattice_offset: [f64; 2],
) -> Vec<CellCoord> {
    let scale = world.scale();
    let from_is_swimming = agent_is_swimming(world, from, lattice_offset);
    let max_rise = scale.ceil() as i32;
    let mut offsets = Vec::with_capacity((max_rise + 1) as usize);
    offsets.push(0);
    offsets.extend(1..=max_rise);

    // Preserve edge cost semantics in the exploration order: ordinary
    // walk/jump edges are always considered before long drop rollouts.  A
    // coordinate-only sort can put a distant negative-coordinate landing
    // ahead of an adjacent flat cell and exhaust a small, otherwise
    // sufficient visit budget before the local path is examined.
    let mut local_neighbors = Vec::new();
    let mut drop_neighbors = Vec::new();
    if from_is_swimming {
        for direction in [-1, 1] {
            if let Some(candidate) = rollout_vertical_swim_neighbor(
                world,
                from,
                direction,
                lattice_offset,
                half_width,
                height,
            ) {
                // Recheck the canonical query state: the proof branch may
                // advance fluid scheduling, but graph nodes describe the
                // caller's current immutable spatial boundary.
                if agent_can_occupy(world, candidate, half_width, height, lattice_offset) {
                    local_neighbors.push(candidate);
                }
            }
        }
    }
    for (dx, dz) in [(-1, 0), (0, -1), (0, 1), (1, 0)] {
        let mut found_local = false;
        for &dy in &offsets {
            let candidate = CellCoord::new(from.x + dx, from.y + dy, from.z + dz);
            if agent_can_traverse(world, from, candidate, half_width, height, lattice_offset) {
                local_neighbors.push(candidate);
                found_local = true;
                // For one horizontal direction, the nearest usable landing
                // is the edge live motion reaches first.
                break;
            }
        }
        if !found_local && !from_is_swimming {
            for landing in rollout_drop_neighbors(world, from, dx, dz, lattice_offset) {
                if agent_can_stand(world, landing, half_width, height, lattice_offset) {
                    drop_neighbors.push(landing);
                }
            }
        }
    }
    local_neighbors.sort_unstable();
    drop_neighbors.sort_unstable();
    local_neighbors.extend(drop_neighbors);
    local_neighbors
}

fn movement_shortest_path(
    world: &mut World,
    start: CellCoord,
    goal: CellCoord,
    max_visited: usize,
    half_width: f64,
    height: f64,
) -> Result<Option<Vec<CellCoord>>, String> {
    // CellCoord names the cell containing the agent's feet, but at finer
    // voxel densities the physical body is not necessarily centered at
    // `cell + 0.5` (the canonical scale-2 spawn lies exactly on a fine-cell
    // boundary).  Anchor the search lattice to the live continuous pose so
    // every candidate uses the same AABB alignment as an actual rollout.
    let live_cell = CellCoord::from_world_pos(WorldPos::from(world.agent.pos));
    let lattice_offset = if start == live_cell {
        [
            world.agent.pos[0] - start.x as f64,
            world.agent.pos[2] - start.z as f64,
        ]
    } else {
        // An arbitrary cell query has no continuous pose to anchor it, so its
        // explicit and deterministic contract is the geometric cell center.
        [0.5, 0.5]
    };
    if !agent_can_occupy(world, start, half_width, height, lattice_offset)
        || !agent_can_occupy(world, goal, half_width, height, lattice_offset)
    {
        return Ok(None);
    }
    if max_visited == 0 {
        return Err("spatial query exceeded its 0-cell visit limit".to_string());
    }
    if start == goal {
        return Ok(Some(vec![start]));
    }

    let mut examined = BTreeSet::from([start]);
    let mut frontier = VecDeque::from([start]);
    let mut parents = BTreeMap::new();
    while let Some(current) = frontier.pop_front() {
        for neighbor in movement_neighbors(world, current, half_width, height, lattice_offset) {
            if !examined.insert(neighbor) {
                continue;
            }
            if examined.len() > max_visited {
                return Err(format!(
                    "spatial query exceeded its {max_visited}-cell visit limit"
                ));
            }
            parents.insert(neighbor, current);
            if neighbor == goal {
                let mut path = vec![goal];
                let mut cursor = goal;
                while cursor != start {
                    cursor = parents[&cursor];
                    path.push(cursor);
                }
                path.reverse();
                return Ok(Some(path));
            }
            frontier.push_back(neighbor);
        }
    }
    Ok(None)
}

fn cell_is_solid(cell: u16) -> bool {
    let id = cell_id(cell);
    !(id == DOOR && cell_state(cell) & 1 == 1) && voxel_core::block::block_def(id).solid
}

fn apply_physics(
    w: World,
    overrides: Option<std::collections::HashMap<String, f64>>,
) -> PyResult<World> {
    if let Some(map) = overrides {
        let mut physics = voxel_core::physics::Physics::default();
        for (k, v) in map {
            physics.set(&k, v).map_err(PyValueError::new_err)?;
        }
        return Ok(w.with_physics(physics));
    }
    Ok(w)
}

#[pyclass]
pub struct PyWorld {
    world: World,
    trace_state: TraceState,
    next_intervention_id: u64,
}

#[pymethods]
impl PyWorld {
    #[new]
    #[pyo3(signature = (seed, preset = "default", scenario = None, physics = None, scale = 1.0, dt_numerator = 1, dt_denominator = 20, semantic_regions = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        seed: u64,
        preset: &str,
        scenario: Option<Vec<ScenarioRegionTuple>>,
        physics: Option<std::collections::HashMap<String, f64>>,
        scale: f64,
        dt_numerator: u64,
        dt_denominator: u64,
        semantic_regions: Option<Vec<SemanticRegionTuple>>,
    ) -> PyResult<Self> {
        if semantic_regions.is_some() && scenario.as_ref().is_some_and(|value| !value.is_empty()) {
            return Err(PyValueError::new_err(
                "pass either scenario or semantic_regions, not both",
            ));
        }
        let clock =
            ClockConfig::new(dt_numerator, dt_denominator).map_err(PyValueError::new_err)?;
        let w = make_world_scaled_with_clock(
            seed,
            preset,
            parse_scenario(scenario),
            scale,
            clock,
            parse_semantic_regions(semantic_regions),
        )?;
        let w = apply_physics(w, physics)?;
        Ok(PyWorld {
            world: w,
            trace_state: TraceState::default(),
            next_intervention_id: 0,
        })
    }

    /// Exact step-boundary clock metadata. Tick counts completed transitions.
    #[pyo3(signature = (*, horizon_tick = None, sample_tick = None))]
    fn clock<'py>(
        &self,
        py: Python<'py>,
        horizon_tick: Option<u64>,
        sample_tick: Option<u64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        clock_with_context_to_dict(py, self.world.sim_clock(), horizon_tick, sample_tick)
    }

    /// Oracle-only, frame-explicit kinematics in both engine and metric units.
    fn oracle_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        let scale = self.world.scale();
        let dt = self.world.clock_config().seconds_per_tick();
        let canonical_vertical_dt = ClockConfig::default().seconds_per_tick();
        let velocity_meters_per_second = |velocity: [f64; 3]| {
            (
                velocity[0] / scale / dt,
                velocity[1] / scale / canonical_vertical_dt,
                velocity[2] / scale / dt,
            )
        };
        let pos = self.world.agent.pos;
        let vel = self.world.agent.vel;
        result.set_item("frame_id", 0u64)?;
        result.set_item("scale", scale)?;
        result.set_item("meters_per_cell", 1.0 / scale)?;
        result.set_item("position_cells", (pos[0], pos[1], pos[2]))?;
        result.set_item(
            "position_meters",
            (pos[0] / scale, pos[1] / scale, pos[2] / scale),
        )?;
        result.set_item("velocity_state_cells", (vel[0], vel[1], vel[2]))?;
        result.set_item(
            "velocity_meters_per_second",
            velocity_meters_per_second(vel),
        )?;
        result.set_item("yaw_degrees", self.world.agent.yaw)?;
        result.set_item("pitch_degrees", self.world.agent.pitch)?;
        result.set_item("health", self.world.agent.hp)?;
        result.set_item("dead", self.world.agent.dead)?;
        result.set_item("selected_hotbar", self.world.agent.selected)?;
        let inventory = self
            .world
            .agent
            .inventory
            .slots
            .iter()
            .enumerate()
            .map(|(slot, stack)| (slot, stack.item, stack.count))
            .collect::<Vec<_>>();
        result.set_item("inventory", inventory)?;

        let mut entities = Vec::new();
        let agent = PyDict::new(py);
        agent.set_item("kind", "agent")?;
        agent.set_item("id", 0u64)?;
        agent.set_item("frame_id", 0u64)?;
        agent.set_item("position_cells", (pos[0], pos[1], pos[2]))?;
        agent.set_item(
            "position_meters",
            (pos[0] / scale, pos[1] / scale, pos[2] / scale),
        )?;
        entities.push(agent);
        for item in &self.world.items {
            let entity = PyDict::new(py);
            entity.set_item("kind", "item")?;
            entity.set_item("id", voxel_core::EntityId::item(item.id).get())?;
            entity.set_item("frame_id", 0u64)?;
            entity.set_item("item", item.item)?;
            entity.set_item("count", item.count)?;
            entity.set_item("position_cells", item.pos)?;
            entity.set_item(
                "position_meters",
                item.pos.map(|component| component / scale),
            )?;
            entity.set_item("velocity_state_cells", item.vel)?;
            entity.set_item(
                "velocity_meters_per_second",
                velocity_meters_per_second(item.vel),
            )?;
            entity.set_item("age_ticks", item.age)?;
            entities.push(entity);
        }
        for falling in &self.world.falling {
            let entity = PyDict::new(py);
            entity.set_item("kind", "falling_block")?;
            entity.set_item("id", voxel_core::EntityId::falling_block(falling.id).get())?;
            entity.set_item("frame_id", 0u64)?;
            entity.set_item("block", falling.block)?;
            entity.set_item("position_cells", falling.pos)?;
            entity.set_item(
                "position_meters",
                falling.pos.map(|component| component / scale),
            )?;
            entity.set_item("velocity_state_cells", falling.vel)?;
            entity.set_item(
                "velocity_meters_per_second",
                velocity_meters_per_second(falling.vel),
            )?;
            entities.push(entity);
        }
        result.set_item("entities", entities)?;

        let mut regions = Vec::new();
        for spec in self.world.semantic_regions() {
            let region = spec.region;
            let semantic = PyDict::new(py);
            semantic.set_item("region_id", spec.region_id.get())?;
            semantic.set_item("structure_id", spec.structure_id.get())?;
            semantic.set_item("frame_id", 0u64)?;
            semantic.set_item(
                "bounds_cells",
                (
                    region.x0, region.y0, region.z0, region.x1, region.y1, region.z1,
                ),
            )?;
            semantic.set_item(
                "bounds_meters_half_open",
                (
                    region.x0 as f64 / scale,
                    region.y0 as f64 / scale,
                    region.z0 as f64 / scale,
                    (region.x1 as f64 + 1.0) / scale,
                    (region.y1 as f64 + 1.0) / scale,
                    (region.z1 as f64 + 1.0) / scale,
                ),
            )?;
            semantic.set_item("cell", spec.cell)?;
            regions.push(semantic);
        }
        result.set_item("semantic_regions", regions)?;
        result.set_item("clock", clock_to_dict(py, self.world.sim_clock())?)?;
        Ok(result)
    }

    /// Read a physics field (see voxel_core::physics::Physics::FIELDS).
    fn get_physics(&self, key: &str) -> PyResult<f64> {
        self.world
            .physics()
            .get(key)
            .ok_or_else(|| PyValueError::new_err(format!("unknown physics field '{key}'")))
    }

    /// Effective episode physics after immutable clock and spatial transforms.
    fn physics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        for key in voxel_core::physics::Physics::FIELDS {
            result.set_item(
                *key,
                self.world
                    .physics()
                    .get(key)
                    .expect("Physics::FIELDS must contain readable fields"),
            )?;
        }
        Ok(result)
    }

    /// Advance one tick. Action = 10-tuple per the gym contract.
    fn step(&mut self, action: ActionTuple) {
        let a = to_action(&action);
        voxel_core::step(&mut self.world, &a);
        // An unrecorded transition is a deliberate causal-observation gap;
        // future traced effects must become exogenous rather than point to a
        // stale event that predates the gap.
        self.trace_state.invalidate();
    }

    /// Advance one transition and return a JSON-native derived causal trace.
    #[pyo3(signature = (action, trace_level = "events", branch_id = 0))]
    fn step_traced<'py>(
        &mut self,
        py: Python<'py>,
        action: ActionTuple,
        trace_level: &str,
        branch_id: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let level = parse_trace_level(trace_level)?;
        let outcome = voxel_core::trace::step_traced_with_state(
            &mut self.world,
            &to_action(&action),
            level,
            branch_id,
            &mut self.trace_state,
        );
        step_outcome_to_dict(py, &outcome)
    }

    /// Clone all simulation state through the canonical World Snapshot contract.
    fn fork(&self) -> PyResult<Self> {
        let world = voxel_core::trace::fork_world(&self.world).map_err(PyValueError::new_err)?;
        Ok(Self {
            world,
            trace_state: self.trace_state.clone(),
            next_intervention_id: self.next_intervention_id,
        })
    }

    /// Apply one tagged, serializable intervention to this branch.
    #[pyo3(signature = (spec, trace_level = "events", branch_id = 0, intervention_id = None))]
    fn apply_intervention<'py>(
        &mut self,
        py: Python<'py>,
        spec: &Bound<'_, PyDict>,
        trace_level: &str,
        branch_id: u64,
        intervention_id: Option<u64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let intervention = parse_intervention(spec)?;
        let level = parse_trace_level(trace_level)?;
        voxel_core::trace::validate_intervention(&intervention).map_err(PyValueError::new_err)?;
        let intervention_id = match intervention_id {
            Some(value) => {
                self.next_intervention_id =
                    self.next_intervention_id
                        .max(value.checked_add(1).ok_or_else(|| {
                            PyValueError::new_err("intervention_id must be less than u64::MAX")
                        })?);
                value
            }
            None => {
                let value = self.next_intervention_id;
                self.next_intervention_id = value.checked_add(1).ok_or_else(|| {
                    PyValueError::new_err("automatic intervention IDs are exhausted")
                })?;
                value
            }
        };
        let outcome = voxel_core::trace::apply_intervention_with_state(
            &mut self.world,
            &intervention,
            level,
            branch_id,
            intervention_id,
            &mut self.trace_state,
        )
        .map_err(PyValueError::new_err)?;
        intervention_outcome_to_dict(py, &outcome)
    }

    /// Compare equal-length factual and counterfactual rollouts from this state.
    fn compare_branches<'py>(
        &self,
        py: Python<'py>,
        intervention: &Bound<'_, PyDict>,
        control_actions: Vec<ActionTuple>,
        treatment_actions: Vec<ActionTuple>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if control_actions.len() != treatment_actions.len() {
            return Err(PyValueError::new_err(format!(
                "branch rollouts require equal-length action sequences (got {} and {})",
                control_actions.len(),
                treatment_actions.len()
            )));
        }
        if control_actions != treatment_actions {
            return Err(PyValueError::new_err(
                "branch rollouts require identical action sequences; the intervention must be the only differing input",
            ));
        }
        let intervention = parse_intervention(intervention)?;
        let control_actions = control_actions.iter().map(to_action).collect::<Vec<_>>();
        let treatment_actions = treatment_actions.iter().map(to_action).collect::<Vec<_>>();
        let comparison = voxel_core::trace::compare_branches(
            &self.world,
            &intervention,
            &control_actions,
            &treatment_actions,
        )
        .map_err(PyValueError::new_err)?;
        let result = PyDict::new(py);
        result.set_item("common_before_hash", comparison.common_before_hash)?;
        result.set_item("control_after_hash", comparison.control_after_hash)?;
        result.set_item("treatment_after_hash", comparison.treatment_after_hash)?;
        result.set_item("diverged", comparison.diverged)?;
        Ok(result)
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

    /// (2,) uint16: crosshair block id, distance in centimetres (450 = 4.5 m reach).
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

    fn give(&mut self, item: u16, count: u16) -> PyResult<()> {
        if !voxel_core::block::is_known_item(item) {
            return Err(PyValueError::new_err(format!("unknown item id {item}")));
        }
        self.world.agent.inventory.add(item, count);
        self.trace_state.invalidate();
        Ok(())
    }

    fn count_item(&self, item: u16) -> u16 {
        self.world.agent.inventory.count(item)
    }

    fn get_block(&mut self, x: i32, y: i32, z: i32) -> u16 {
        self.world.get_block(x, y, z)
    }

    fn set_block(&mut self, x: i32, y: i32, z: i32, cell: u16) -> PyResult<()> {
        self.world
            .try_set_block(x, y, z, cell)
            .map_err(PyValueError::new_err)?;
        self.trace_state.invalidate();
        Ok(())
    }

    fn teleport(&mut self, x: f64, y: f64, z: f64) {
        self.world.agent.pos = [x, y, z];
        self.world.agent.vel = [0.0; 3];
        self.world.agent.fall_distance = 0.0;
        self.trace_state.invalidate();
    }

    fn agent_pos(&self) -> (f64, f64, f64) {
        (
            self.world.agent.pos[0],
            self.world.agent.pos[1],
            self.world.agent.pos[2],
        )
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

    /// Immutable cells-per-metre density for task/oracle coordinate adapters.
    fn scale(&self) -> f64 {
        self.world.scale()
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

    /// Nearest candidate cell to a world position (agent position by default).
    /// Equal distances are resolved lexicographically for deterministic data.
    #[pyo3(signature = (candidates, origin = None))]
    fn nearest(
        &self,
        candidates: Vec<(i32, i32, i32)>,
        origin: Option<(f64, f64, f64)>,
    ) -> PyResult<Option<(i32, i32, i32)>> {
        let origin = origin
            .map(|position| [position.0, position.1, position.2])
            .unwrap_or(self.world.agent.pos);
        if !origin.iter().all(|component| component.is_finite()) {
            return Err(PyValueError::new_err("nearest origin must be finite"));
        }
        Ok(voxel_core::spatial::nearest(
            WorldPos::from(origin),
            candidates.into_iter().map(CellCoord::from),
        )
        .map(Into::into))
    }

    /// Euclidean containment in world-cell coordinates.
    fn within(&self, origin: (f64, f64, f64), candidate: (f64, f64, f64), radius: f64) -> bool {
        voxel_core::spatial::within(
            WorldPos::new(origin.0, origin.1, origin.2),
            WorldPos::new(candidate.0, candidate.1, candidate.2),
            radius,
        )
    }

    fn adjacent(&self, left: (i32, i32, i32), right: (i32, i32, i32)) -> bool {
        voxel_core::spatial::adjacent(CellCoord::from(left), CellCoord::from(right))
    }

    fn above(&self, candidate: (i32, i32, i32), reference: (i32, i32, i32)) -> bool {
        voxel_core::spatial::above(CellCoord::from(candidate), CellCoord::from(reference))
    }

    fn below(&self, candidate: (i32, i32, i32), reference: (i32, i32, i32)) -> bool {
        voxel_core::spatial::below(CellCoord::from(candidate), CellCoord::from(reference))
    }

    /// Whether the target cell is the first solid cell reached by a DDA ray.
    fn visible(&mut self, origin: (f64, f64, f64), target: (i32, i32, i32)) -> PyResult<bool> {
        let origin = [origin.0, origin.1, origin.2];
        if !origin.iter().all(|component| component.is_finite()) {
            return Err(PyValueError::new_err("visible origin must be finite"));
        }
        let center = [
            target.0 as f64 + 0.5,
            target.1 as f64 + 0.5,
            target.2 as f64 + 0.5,
        ];
        let delta = [
            center[0] - origin[0],
            center[1] - origin[1],
            center[2] - origin[2],
        ];
        let distance = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        if distance == 0.0 {
            return Ok(true);
        }
        let direction = [
            delta[0] / distance,
            delta[1] / distance,
            delta[2] / distance,
        ];
        let world = &mut self.world;
        let hit = voxel_core::raycast::dda_with(
            origin,
            direction,
            distance,
            |x, y, z| world.get_block(x, y, z),
            cell_is_solid,
        );
        Ok(match hit {
            Some(hit) => (hit.x, hit.y, hit.z) == target,
            None => true,
        })
    }

    /// Solid six-connected component containing `start`.
    #[pyo3(signature = (start, max_visited = 4096))]
    fn connected_component(
        &mut self,
        start: (i32, i32, i32),
        max_visited: usize,
    ) -> PyResult<Vec<(i32, i32, i32)>> {
        voxel_core::spatial::connected_component(CellCoord::from(start), max_visited, |at| {
            self.world.is_solid(at.x, at.y, at.z)
        })
        .map(|cells| cells.into_iter().map(Into::into).collect())
        .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Shortest path under the live agent body, jump, swim, drop, and collision constraints.
    #[pyo3(signature = (start, goal, max_visited = 4096))]
    fn shortest_path(
        &mut self,
        start: (i32, i32, i32),
        goal: (i32, i32, i32),
        max_visited: usize,
    ) -> PyResult<Option<Vec<(i32, i32, i32)>>> {
        let half_width = self.world.agent.half_width;
        let height = self.world.agent.height;
        movement_shortest_path(
            &mut self.world,
            CellCoord::from(start),
            CellCoord::from(goal),
            max_visited,
            half_width,
            height,
        )
        .map(|path| path.map(|cells| cells.into_iter().map(Into::into).collect()))
        .map_err(PyValueError::new_err)
    }

    /// Reachability under the same collision and visit-limit contract as shortest_path.
    #[pyo3(signature = (start, goal, max_visited = 4096))]
    fn reachable(
        &mut self,
        start: (i32, i32, i32),
        goal: (i32, i32, i32),
        max_visited: usize,
    ) -> PyResult<bool> {
        let half_width = self.world.agent.half_width;
        let height = self.world.agent.height;
        movement_shortest_path(
            &mut self.world,
            CellCoord::from(start),
            CellCoord::from(goal),
            max_visited,
            half_width,
            height,
        )
        .map(|path| path.is_some())
        .map_err(PyValueError::new_err)
    }

    // ---- determinism contract ----

    fn snapshot<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, &self.world.snapshot())
    }

    /// Snapshot the derived causal-lineage continuation state for EnvSnapshot.
    fn trace_state_snapshot<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, &self.trace_state.snapshot())
    }

    /// Sorted causal parents that predate a recorder starting at this boundary.
    fn trace_external_parent_ids(&self) -> Vec<u64> {
        self.trace_state.external_parent_ids()
    }

    /// Snapshot the native allocator used when callers omit an intervention
    /// ID.  It is derived interaction state (not physical World State), but
    /// must travel with an EnvSnapshot for exact causal continuation.
    fn intervention_cursor(&self) -> u64 {
        self.next_intervention_id
    }

    /// Restore derived causal lineage without changing canonical World State.
    fn restore_trace_state(&mut self, bytes: &[u8]) -> PyResult<()> {
        self.trace_state = TraceState::restore(bytes).map_err(PyValueError::new_err)?;
        Ok(())
    }

    fn restore_intervention_cursor(&mut self, cursor: u64) {
        self.next_intervention_id = cursor;
    }

    fn restore(&mut self, bytes: &[u8]) -> PyResult<()> {
        self.world = World::restore(bytes).map_err(PyValueError::new_err)?;
        self.trace_state = TraceState::default();
        self.next_intervention_id = 0;
        Ok(())
    }

    fn hash(&self) -> u64 {
        self.world.hash()
    }

    /// Read-only compatibility hash for Episode Bundle v1 sidecars.
    fn legacy_hash_v7(&self) -> u64 {
        self.world.legacy_hash_v7()
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
        let slot = self.world.swap_to_hotbar(item);
        if slot >= 0 {
            self.trace_state.invalidate();
        }
        slot
    }

    /// Crosshair block: ((x,y,z), block_id, distance centimetres), None if no hit.
    fn crosshair(&mut self) -> Option<((i32, i32, i32), u16, u16)> {
        raycast_target(&mut self.world).map(|h| {
            (
                (h.x, h.y, h.z),
                cell_id(h.cell),
                (h.dist * 100.0).round().min(450.0) as u16,
            )
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

    /// Render the agent view: (rgb (128,128,3) u8, depth (128,128) f32 metres,
    /// seg (128,128) u16 block ids, normals (128,128,3) f32 unit axis,
    /// [0,0,0] on sky miss). SKY_SEG=0xFFFF on miss.
    fn render<'py>(&mut self, py: Python<'py>) -> PyResult<RenderOutput<'py>> {
        let f = py.allow_threads(|| voxel_view::render(&mut self.world, 128, 128, 90.0));
        frame_to_numpy(py, f)
    }

    /// Free camera render: (origin, yaw_deg, pitch_deg). Positive pitch
    /// looks down. Same channels as render().
    #[pyo3(signature = (origin, yaw_deg, pitch_deg, width=128, height=128, fov_deg=90.0))]
    #[allow(clippy::too_many_arguments)]
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
    fn cast_ray(&mut self, origin: (f64, f64, f64), dir: (f64, f64, f64), max_dist: f64) -> f64 {
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
    /// pose for a fixed emitter block). Returns (range metres, intensity, seg),
    /// each (channels, azimuth_steps); range 0 = no return, seg SKY=0xFFFF.
    /// The scan is a pure function of (world state, cfg, frame_idx) — noise
    /// is position-hashed, so replays are byte-identical.
    #[pyo3(signature = (channels, azimuth_steps, min_elev_deg, max_elev_deg, max_range, noise_sigma=0.0, dropout_p=0.0, noise_seed=0, frame_idx=0, origin=None, yaw_deg=None))]
    #[allow(clippy::too_many_arguments)]
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
    ) -> LidarOutput<'py> {
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
        let s = py
            .allow_threads(|| voxel_view::lidar::scan(&mut self.world, &cfg, org, yaw, frame_idx));
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
    fn validate_action_count(&self, count: usize) -> PyResult<()> {
        if count == self.worlds.len() {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "expected {} actions, got {count}",
                self.worlds.len()
            )))
        }
    }

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
    fn step_batch(&mut self, py: Python<'_>, actions: Vec<ActionTuple>) -> PyResult<Vec<bool>> {
        self.validate_action_count(actions.len())?;
        let acts: Vec<Action> = actions.iter().map(to_action).collect();
        Ok(self.step_actions(py, acts))
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
        if arr.ndim() != 2 || arr.nrows() != self.worlds.len() || arr.ncols() != 10 {
            return Err(PyValueError::new_err(format!(
                "actions must have shape ({}, 10)",
                self.worlds.len()
            )));
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
            .map(|w| {
                w.agent
                    .inventory
                    .slots
                    .iter()
                    .map(|s| vec![s.item, s.count])
                    .collect()
            })
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
