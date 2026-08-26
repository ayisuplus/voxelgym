//! Optional, derived transition traces for world-model supervision.
//!
//! The [`World`] remains the sole source of simulation truth.  Traces are
//! observations of a transition: enabling them must never alter scheduling,
//! random draws, snapshots, or hashes.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use xxhash_rust::xxh3::xxh3_64;

use crate::block::{cell_id, AIR, DOOR, FIRE, LAVA, LEVER, WATER};
use crate::clock::SimClock;
use crate::spatial::{CellCoord, EntityId, WorldPos};
use crate::world::{Event, World};
use crate::Action;

/// Amount of transition provenance requested by a caller.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TraceLevel {
    /// Preserve the original hot path: no hashes, events, or deltas.
    #[default]
    Off,
    /// Record semantic events but omit exact state values.
    Events,
    /// Record semantic events, before/after hashes, and exact state deltas.
    Full,
}

/// Stable phase names for the fixed in-tick order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Phase {
    Intervention = 0,
    AgentAction = 1,
    EntityIntegration = 2,
    Scheduled = 3,
    Fluid = 4,
    Fire = 5,
    Circuit = 6,
    Tnt = 7,
    ItemLogic = 8,
    Observation = 9,
}

/// Mechanism-level event kinds.  The enum is deliberately semantic rather
/// than a mirror of implementation functions, so recorded datasets survive
/// refactors of the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventKind {
    ActionApplied,
    InterventionApplied,
    AgentMoved,
    VelocityChanged,
    Collision,
    Damage,
    Death,
    InventoryChanged,
    ItemPicked,
    Crafted,
    BlockMined,
    BlockPlaced,
    BlockChanged,
    /// A loose block support check was queued for a later tick.
    BlockFallScheduled,
    BlockFell,
    Smelted,
    FluidChanged,
    Ignited,
    Extinguished,
    CircuitChanged,
    /// A TNT fuse was created; the later explosion points back here.
    TntPrimed,
    Explosion,
    EntitySpawned,
    EntityDespawned,
    StateChanged,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SubjectRef {
    World,
    Agent(EntityId),
    Cell(CellCoord),
    Entity(EntityId),
    InventorySlot(u8),
    Scheduler(&'static str),
}

/// Typed values used by exact state deltas. Floating-point values are kept as
/// IEEE bit patterns so comparisons and serialization remain deterministic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceValue {
    None,
    Bool(bool),
    I64(i64),
    U64(u64),
    F64Bits(u64),
    Vec3Bits([u64; 3]),
    Cell(u16),
    /// Exact ordered queue contents. Unlike a set, this preserves repeated
    /// coordinates and therefore fully describes scheduler input state.
    CellSequence(Vec<CellCoord>),
    ItemStack {
        item: u16,
        count: u16,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RootCause {
    Action {
        branch_id: u64,
        tick: u64,
    },
    Intervention {
        branch_id: u64,
        intervention_id: u64,
    },
    Periodic {
        tick: u64,
        mechanism: &'static str,
    },
    /// State entered the tracing boundary without a prior trace event. This
    /// is distinct from a periodic simulation mechanism: it normally means
    /// setup code or another explicit caller mutated a public World field.
    Exogenous {
        branch_id: u64,
        tick: u64,
        ordinal: u64,
        mechanism: &'static str,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldEvent {
    pub id: u64,
    pub tick: u64,
    pub phase: Phase,
    pub kind: EventKind,
    pub actor: Option<SubjectRef>,
    pub target: Option<SubjectRef>,
    pub location: Option<CellCoord>,
    pub mechanism: &'static str,
    pub parent_ids: Vec<u64>,
    pub root_cause: RootCause,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateDelta {
    pub event_id: u64,
    pub subject: SubjectRef,
    pub field_or_cell: &'static str,
    pub before: TraceValue,
    pub after: TraceValue,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StepOutcome {
    pub clock_before: SimClock,
    pub clock_after: SimClock,
    pub before_hash: Option<u64>,
    pub after_hash: Option<u64>,
    pub events: Vec<WorldEvent>,
    pub deltas: Vec<StateDelta>,
}

/// Stateful, derived causal lineage carried between traced transitions.
///
/// The physical World never reads this value.  It only remembers which
/// earlier trace event scheduled a loose-block conversion or primed a TNT
/// fuse, plus provenance for dirty cells created by tracked interventions or
/// mechanisms.  Cloning/forking a world should clone this state as well.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TraceState {
    scheduled_falls: BTreeMap<(CellCoord, u64), Lineage>,
    pending_booms: BTreeMap<(CellCoord, u64), Lineage>,
    dirty_cells: BTreeMap<CellCoord, Lineage>,
    next_root_ordinal: u64,
    next_intervention_ordinal: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Lineage {
    event_id: u64,
    root_cause: RootCause,
}

impl TraceState {
    const SNAPSHOT_MAGIC: &'static [u8; 5] = b"VXTR1";

    /// Stable allocator state used when an untraced external mutation must
    /// be represented by a new exogenous root.
    pub const fn next_root_ordinal(&self) -> u64 {
        self.next_root_ordinal
    }

    /// Stable allocator state used to disambiguate repeated intervention IDs
    /// at the same boundary.
    pub const fn next_intervention_ordinal(&self) -> u64 {
        self.next_intervention_ordinal
    }

    /// Event IDs referenced by pending lineage at the current boundary.
    ///
    /// A recorder starting from an arbitrary EnvSnapshot declares this exact
    /// set as external ancestry instead of guessing from missing event rows.
    pub fn external_parent_ids(&self) -> Vec<u64> {
        self.scheduled_falls
            .values()
            .chain(self.pending_booms.values())
            .chain(self.dirty_cells.values())
            .map(|lineage| lineage.event_id)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Invalidate causal mappings after any transition or intervention that
    /// was intentionally not traced. Allocator positions are retained so a
    /// later tracked intervention cannot reuse an earlier event ID.
    #[inline(always)]
    pub fn invalidate(&mut self) {
        self.scheduled_falls.clear();
        self.pending_booms.clear();
        self.dirty_cells.clear();
    }

    /// Stable binary representation suitable for embedding in EnvSnapshot.
    pub fn snapshot(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(Self::SNAPSHOT_MAGIC);
        put_u64(&mut out, self.next_root_ordinal);
        put_u64(&mut out, self.next_intervention_ordinal);
        put_u32(&mut out, self.scheduled_falls.len() as u32);
        for ((at, due), lineage) in &self.scheduled_falls {
            put_coord(&mut out, *at);
            put_u64(&mut out, *due);
            put_lineage(&mut out, lineage);
        }
        put_u32(&mut out, self.pending_booms.len() as u32);
        for ((at, due), lineage) in &self.pending_booms {
            put_coord(&mut out, *at);
            put_u64(&mut out, *due);
            put_lineage(&mut out, lineage);
        }
        put_u32(&mut out, self.dirty_cells.len() as u32);
        for (at, lineage) in &self.dirty_cells {
            put_coord(&mut out, *at);
            put_lineage(&mut out, lineage);
        }
        out
    }

    /// Restore a trace lineage snapshot without consulting or mutating World.
    pub fn restore(bytes: &[u8]) -> Result<Self, String> {
        let mut reader = TraceReader::new(bytes);
        if reader.take(Self::SNAPSHOT_MAGIC.len())? != Self::SNAPSHOT_MAGIC {
            return Err("bad trace-state magic".into());
        }
        let next_root_ordinal = reader.u64()?;
        let next_intervention_ordinal = reader.u64()?;
        let scheduled_falls = read_lineage_map_with_due(&mut reader)?;
        let pending_booms = read_lineage_map_with_due(&mut reader)?;
        let count = reader.u32()? as usize;
        let mut dirty_cells = BTreeMap::new();
        for _ in 0..count {
            let at = reader.coord()?;
            let lineage = reader.lineage()?;
            if dirty_cells.insert(at, lineage).is_some() {
                return Err("duplicate dirty-cell trace lineage".into());
            }
        }
        if !reader.remaining().is_empty() {
            return Err("trailing bytes in trace-state snapshot".into());
        }
        Ok(Self {
            scheduled_falls,
            pending_booms,
            dirty_cells,
            next_root_ordinal,
            next_intervention_ordinal,
        })
    }

    fn exogenous_root(&mut self, branch_id: u64, tick: u64, mechanism: &'static str) -> RootCause {
        let ordinal = self.next_root_ordinal;
        self.next_root_ordinal = self.next_root_ordinal.wrapping_add(1);
        RootCause::Exogenous {
            branch_id,
            tick,
            ordinal,
            mechanism,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AgentTrace {
    pos: [u64; 3],
    vel: [u64; 3],
    yaw: u32,
    pitch: u32,
    on_ground: bool,
    hp: i32,
    fall_distance: u64,
    dead: bool,
    suffocation_timer: u32,
    lava_timer: u32,
    fire_timer: u32,
    selected: usize,
    inventory: [(u16, u16); 36],
}

impl AgentTrace {
    fn capture(world: &World) -> Self {
        let mut inventory = [(0, 0); 36];
        for (dst, slot) in inventory.iter_mut().zip(&world.agent.inventory.slots) {
            *dst = (slot.item, slot.count);
        }
        Self {
            pos: world.agent.pos.map(f64::to_bits),
            vel: world.agent.vel.map(f64::to_bits),
            yaw: world.agent.yaw.to_bits(),
            pitch: world.agent.pitch.to_bits(),
            on_ground: world.agent.on_ground,
            hp: world.agent.hp,
            fall_distance: world.agent.fall_distance.to_bits(),
            dead: world.agent.dead,
            suffocation_timer: world.agent.suffocation_timer,
            lava_timer: world.agent.lava_timer,
            fire_timer: world.agent.fire_timer,
            selected: world.agent.selected,
            inventory,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MiningTrace {
    target: CellCoord,
    progress: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ItemTrace {
    item: u16,
    count: u16,
    pos: [u64; 3],
    vel: [u64; 3],
    age: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FallingTrace {
    block: u16,
    pos: [u64; 3],
    vel: [u64; 3],
    fall_distance: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FurnaceTrace {
    remaining: u32,
    out_ready: bool,
    fuel_left: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WorldTrace {
    tick: u64,
    rng: (u128, u128),
    agent: AgentTrace,
    mining: Option<MiningTrace>,
    place_cooldown: u8,
    dirty: Vec<CellCoord>,
    items: BTreeMap<u64, ItemTrace>,
    furnaces: BTreeMap<CellCoord, FurnaceTrace>,
    next_item_id: u64,
    falling: BTreeMap<u64, FallingTrace>,
    scheduled_falls: BTreeSet<(CellCoord, u64)>,
    scheduled_set: BTreeSet<CellCoord>,
    active_fluids: BTreeSet<CellCoord>,
    circuit_cells: BTreeSet<CellCoord>,
    active_fire: BTreeSet<CellCoord>,
    tnt_cells: BTreeSet<CellCoord>,
    pending_booms: BTreeSet<(CellCoord, u64)>,
    next_falling_id: u64,
}

impl WorldTrace {
    fn capture(world: &World) -> Self {
        Self {
            tick: world.tick,
            rng: world.rng.state(),
            agent: AgentTrace::capture(world),
            mining: world.mining.map(|mining| MiningTrace {
                target: CellCoord::new(mining.target.0, mining.target.1, mining.target.2),
                progress: mining.progress.to_bits(),
            }),
            place_cooldown: world.place_cooldown,
            dirty: world
                .dirty
                .iter()
                .map(|&(x, y, z)| CellCoord::new(x, y, z))
                .collect(),
            items: world
                .items
                .iter()
                .map(|item| {
                    (
                        item.id,
                        ItemTrace {
                            item: item.item,
                            count: item.count,
                            pos: item.pos.map(f64::to_bits),
                            vel: item.vel.map(f64::to_bits),
                            age: item.age,
                        },
                    )
                })
                .collect(),
            furnaces: world
                .furnaces
                .iter()
                .map(|(&(x, y, z), furnace)| {
                    (
                        CellCoord::new(x, y, z),
                        FurnaceTrace {
                            remaining: furnace.remaining,
                            out_ready: furnace.out_ready,
                            fuel_left: furnace.fuel_left,
                        },
                    )
                })
                .collect(),
            next_item_id: world.next_item_id,
            falling: world
                .falling
                .iter()
                .map(|falling| {
                    (
                        falling.id,
                        FallingTrace {
                            block: falling.block,
                            pos: falling.pos.map(f64::to_bits),
                            vel: falling.vel.map(f64::to_bits),
                            fall_distance: falling.fall_dist.to_bits(),
                        },
                    )
                })
                .collect(),
            scheduled_falls: world
                .scheduled_falls
                .iter()
                .map(|&(x, y, z, due)| (CellCoord::new(x, y, z), due))
                .collect(),
            scheduled_set: coord_set(&world.scheduled_set),
            active_fluids: coord_set(&world.active_fluids),
            circuit_cells: coord_set(&world.circuit_cells),
            active_fire: coord_set(&world.active_fire),
            tnt_cells: coord_set(&world.tnt_cells),
            pending_booms: world
                .pending_booms
                .iter()
                .map(|&(x, y, z, due)| (CellCoord::new(x, y, z), due))
                .collect(),
            next_falling_id: world.next_falling_id,
        }
    }
}

fn coord_set(source: &crate::world::XSet<(i32, i32, i32)>) -> BTreeSet<CellCoord> {
    source
        .iter()
        .map(|&(x, y, z)| CellCoord::new(x, y, z))
        .collect()
}

#[derive(Default)]
struct Attribution {
    cells: BTreeMap<CellCoord, u64>,
    scheduled_added: BTreeMap<(CellCoord, u64), u64>,
    scheduled_removed: BTreeMap<(CellCoord, u64), u64>,
    booms_added: BTreeMap<(CellCoord, u64), u64>,
    booms_removed: BTreeMap<(CellCoord, u64), u64>,
    falling: BTreeMap<u64, u64>,
}

struct Builder {
    level: TraceLevel,
    tick: u64,
    branch_id: u64,
    phase_ordinals: [u32; 10],
    root_id: u64,
    root_cause: RootCause,
    events: Vec<EventRecord>,
    deltas: Vec<StateDelta>,
    periodic_roots: HashMap<(Phase, &'static str), Lineage>,
}

struct EventRecord {
    ordinal: u32,
    event: WorldEvent,
}

impl Builder {
    fn action(level: TraceLevel, branch_id: u64, tick: u64) -> Self {
        let root_cause = RootCause::Action { branch_id, tick };
        let root_id = event_id(branch_id, tick, Phase::AgentAction, 0);
        let mut this = Self {
            level,
            tick,
            branch_id,
            phase_ordinals: [0; 10],
            root_id,
            root_cause: root_cause.clone(),
            events: Vec::new(),
            deltas: Vec::new(),
            periodic_roots: HashMap::new(),
        };
        this.phase_ordinals[Phase::AgentAction as usize] = 1;
        this.events.push(EventRecord {
            ordinal: 0,
            event: WorldEvent {
                id: root_id,
                tick,
                phase: Phase::AgentAction,
                kind: EventKind::ActionApplied,
                actor: Some(SubjectRef::Agent(EntityId::AGENT)),
                target: Some(SubjectRef::World),
                location: None,
                mechanism: "agent_action",
                parent_ids: Vec::new(),
                root_cause,
            },
        });
        this
    }

    fn next_id(&mut self, phase: Phase) -> (u64, u32) {
        let ordinal = self.phase_ordinals[phase as usize];
        self.phase_ordinals[phase as usize] = ordinal.wrapping_add(1);
        (event_id(self.branch_id, self.tick, phase, ordinal), ordinal)
    }

    fn action_lineage(&self) -> Lineage {
        Lineage {
            event_id: self.root_id,
            root_cause: self.root_cause.clone(),
        }
    }

    fn event_lineage(&self, event_id: u64) -> Option<Lineage> {
        self.events
            .iter()
            .find(|record| record.event.id == event_id)
            .map(|record| Lineage {
                event_id,
                root_cause: record.event.root_cause.clone(),
            })
    }

    fn mechanism_event_lineage(
        &self,
        event_id: u64,
        mechanism: &'static str,
    ) -> Option<(Phase, Lineage)> {
        let record = self
            .events
            .iter()
            .find(|record| record.event.id == event_id && record.event.mechanism == mechanism)?;
        Some((
            record.event.phase,
            Lineage {
                event_id,
                root_cause: record.event.root_cause.clone(),
            },
        ))
    }

    fn event(
        &mut self,
        phase: Phase,
        kind: EventKind,
        target: Option<SubjectRef>,
        location: Option<CellCoord>,
        mechanism: &'static str,
    ) -> u64 {
        let lineage = Lineage {
            event_id: self.root_id,
            root_cause: self.root_cause.clone(),
        };
        self.caused_event(phase, kind, target, location, mechanism, &lineage)
    }

    fn caused_event(
        &mut self,
        phase: Phase,
        kind: EventKind,
        target: Option<SubjectRef>,
        location: Option<CellCoord>,
        mechanism: &'static str,
        lineage: &Lineage,
    ) -> u64 {
        let (id, ordinal) = self.next_id(phase);
        let actor = matches!(lineage.root_cause, RootCause::Action { .. })
            .then_some(SubjectRef::Agent(EntityId::AGENT));
        self.events.push(EventRecord {
            ordinal,
            event: WorldEvent {
                id,
                tick: self.tick,
                phase,
                kind,
                actor,
                target,
                location,
                mechanism,
                parent_ids: vec![lineage.event_id],
                root_cause: lineage.root_cause.clone(),
            },
        });
        id
    }

    fn root_event(
        &mut self,
        phase: Phase,
        mechanism: &'static str,
        root_cause: RootCause,
    ) -> Lineage {
        let (id, ordinal) = self.next_id(phase);
        self.events.push(EventRecord {
            ordinal,
            event: WorldEvent {
                id,
                tick: self.tick,
                phase,
                kind: EventKind::StateChanged,
                actor: None,
                target: Some(SubjectRef::World),
                location: None,
                mechanism,
                parent_ids: Vec::new(),
                root_cause: root_cause.clone(),
            },
        });
        Lineage {
            event_id: id,
            root_cause,
        }
    }

    fn periodic_lineage(&mut self, phase: Phase, mechanism: &'static str) -> Lineage {
        if let Some(lineage) = self.periodic_roots.get(&(phase, mechanism)) {
            return lineage.clone();
        }
        let lineage = self.root_event(
            phase,
            mechanism,
            RootCause::Periodic {
                tick: self.tick,
                mechanism,
            },
        );
        self.periodic_roots
            .insert((phase, mechanism), lineage.clone());
        lineage
    }

    fn autonomous_event(
        &mut self,
        phase: Phase,
        kind: EventKind,
        target: Option<SubjectRef>,
        location: Option<CellCoord>,
        mechanism: &'static str,
    ) -> u64 {
        let lineage = self.periodic_lineage(phase, mechanism);
        self.caused_event(phase, kind, target, location, mechanism, &lineage)
    }

    fn delta(
        &mut self,
        event_id: u64,
        subject: SubjectRef,
        field_or_cell: &'static str,
        before: TraceValue,
        after: TraceValue,
    ) {
        if self.level == TraceLevel::Full && before != after {
            self.deltas.push(StateDelta {
                event_id,
                subject,
                field_or_cell,
                before,
                after,
            });
        }
    }

    fn finish(mut self) -> (Vec<WorldEvent>, Vec<StateDelta>) {
        self.events
            .sort_by_key(|record| (record.event.phase as u8, record.ordinal));
        let rank: HashMap<u64, usize> = self
            .events
            .iter()
            .enumerate()
            .map(|(index, record)| (record.event.id, index))
            .collect();
        self.deltas
            .sort_by_key(|delta| rank.get(&delta.event_id).copied().unwrap_or(usize::MAX));
        (
            self.events.into_iter().map(|record| record.event).collect(),
            self.deltas,
        )
    }
}

fn event_id(branch_id: u64, tick: u64, phase: Phase, ordinal: u32) -> u64 {
    let mut bytes = [0u8; 21];
    bytes[..8].copy_from_slice(&branch_id.to_le_bytes());
    bytes[8..16].copy_from_slice(&tick.to_le_bytes());
    bytes[16] = phase as u8;
    bytes[17..].copy_from_slice(&ordinal.to_le_bytes());
    xxh3_64(&bytes)
}

fn intervention_event_id(branch_id: u64, tick: u64, intervention_id: u64) -> u64 {
    let mut bytes = [0u8; 25];
    bytes[..8].copy_from_slice(&branch_id.to_le_bytes());
    bytes[8..16].copy_from_slice(&tick.to_le_bytes());
    bytes[16] = Phase::Intervention as u8;
    bytes[17..].copy_from_slice(&intervention_id.to_le_bytes());
    xxh3_64(&bytes)
}

fn intervention_event_id_with_ordinal(
    branch_id: u64,
    tick: u64,
    intervention_id: u64,
    ordinal: u64,
) -> u64 {
    if ordinal == 0 {
        return intervention_event_id(branch_id, tick, intervention_id);
    }
    let mut bytes = [0u8; 33];
    bytes[..8].copy_from_slice(&branch_id.to_le_bytes());
    bytes[8..16].copy_from_slice(&tick.to_le_bytes());
    bytes[16] = Phase::Intervention as u8;
    bytes[17..25].copy_from_slice(&intervention_id.to_le_bytes());
    bytes[25..].copy_from_slice(&ordinal.to_le_bytes());
    xxh3_64(&bytes)
}

/// Advance one transition while deriving an optional causal trace.
///
/// `TraceLevel::Off` delegates directly to the original hot path. Events and
/// full traces intentionally use the same step implementation and compare the
/// resulting public state; traces can therefore never drive the simulation.
pub fn step_traced(
    world: &mut World,
    action: &Action,
    level: TraceLevel,
    branch_id: u64,
) -> StepOutcome {
    if level == TraceLevel::Off {
        let clock_before = world.sim_clock();
        crate::step(world, action);
        return StepOutcome {
            clock_before,
            clock_after: world.sim_clock(),
            before_hash: None,
            after_hash: None,
            events: Vec::new(),
            deltas: Vec::new(),
        };
    }
    step_traced_with_state(world, action, level, branch_id, &mut TraceState::default())
}

/// Stateful form of [`step_traced`]. Carry the same `TraceState` across
/// transitions (and clone/snapshot it with a branch) to preserve scheduler
/// and TNT causal parents across tick boundaries.
#[inline(always)]
pub fn step_traced_with_state(
    world: &mut World,
    action: &Action,
    level: TraceLevel,
    branch_id: u64,
    trace_state: &mut TraceState,
) -> StepOutcome {
    let clock_before = world.sim_clock();
    if level == TraceLevel::Off {
        crate::step(world, action);
        // An omitted transition deliberately breaks observable lineage. Do
        // not leave a future event pointing at an event that was not emitted.
        trace_state.invalidate();
        return StepOutcome {
            clock_before,
            clock_after: world.sim_clock(),
            before_hash: None,
            after_hash: None,
            events: Vec::new(),
            deltas: Vec::new(),
        };
    }

    step_traced_recorded(world, action, level, branch_id, trace_state, clock_before)
}

#[cold]
#[inline(never)]
fn step_traced_recorded(
    world: &mut World,
    action: &Action,
    level: TraceLevel,
    branch_id: u64,
    trace_state: &mut TraceState,
    clock_before: SimClock,
) -> StepOutcome {
    debug_assert_ne!(level, TraceLevel::Off);
    let before_hash = (level == TraceLevel::Full).then(|| world.hash());
    // Traced transitions intentionally pay for an isolated pre-state.  This
    // keeps the normal step path untouched and lets us derive cell-level
    // changes even when a subsystem consumes its dirty queue internally.
    let mut before_world = World::restore(&world.snapshot())
        .expect("a live world must restore from its canonical snapshot");
    let before = WorldTrace::capture(world);
    let old_event_len = world.events.len();
    let mut builder = Builder::action(level, branch_id, world.tick);

    crate::step(world, action);

    let after = WorldTrace::capture(world);
    let root_id = builder.root_id;
    builder.delta(
        root_id,
        SubjectRef::World,
        "tick",
        TraceValue::U64(clock_before.tick()),
        TraceValue::U64(world.tick),
    );

    let mut attribution = Attribution::default();
    let mut explosions = Vec::new();
    emit_scheduler_events(
        &mut builder,
        trace_state,
        &before,
        &after,
        action,
        &mut attribution,
        &mut explosions,
    );

    let semantic_events: Vec<_> = world.events[old_event_len..].to_vec();
    for event in semantic_events {
        let (kind, target, mechanism) = match event {
            Event::ItemPicked { item, .. } => (
                EventKind::ItemPicked,
                Some(SubjectRef::InventorySlot(world.agent.selected as u8)),
                if item == 0 {
                    "item_pickup_empty"
                } else {
                    "item_pickup"
                },
            ),
            Event::Crafted { .. } => (EventKind::Crafted, None, "crafting"),
            // Cell comparison below has the exact target and is therefore the
            // canonical semantic event for mining.
            Event::BlockMined { .. } => continue,
            Event::Smelted { .. } => (EventKind::Smelted, None, "smelting"),
        };
        builder.event(Phase::ItemLogic, kind, target, None, mechanism);
    }

    emit_block_changes(
        &mut builder,
        trace_state,
        &mut before_world,
        world,
        action,
        &explosions,
        &mut attribution,
    );
    emit_agent_changes(&mut builder, &before, &after, &explosions, &attribution);
    emit_item_changes(&mut builder, &before, &after);
    emit_falling_changes(&mut builder, &before, &after, &mut attribution);
    emit_furnace_changes(&mut builder, &before, &after, action);
    emit_scheduler_deltas(&mut builder, &before, &after, &attribution);
    emit_active_set_deltas(&mut builder, &before, &after, &attribution);
    emit_misc_state_deltas(&mut builder, &before, &after);
    update_trace_state(trace_state, &after, &attribution, &mut builder);

    let (events, deltas) = builder.finish();

    StepOutcome {
        clock_before,
        clock_after: world.sim_clock(),
        before_hash,
        after_hash: (level == TraceLevel::Full).then(|| world.hash()),
        events,
        deltas,
    }
}

fn emit_scheduler_events(
    builder: &mut Builder,
    state: &mut TraceState,
    before: &WorldTrace,
    after: &WorldTrace,
    action: &Action,
    attribution: &mut Attribution,
    explosions: &mut Vec<(CellCoord, u64, Lineage)>,
) {
    let mut next_scheduled = BTreeMap::new();

    for key in before.scheduled_falls.difference(&after.scheduled_falls) {
        let lineage = state.scheduled_falls.get(key).cloned().unwrap_or_else(|| {
            exogenous_lineage(builder, state, Phase::Scheduled, "scheduled_fall")
        });
        let spawned = after
            .falling
            .iter()
            .find(|(id, falling)| {
                !before.falling.contains_key(id) && cell_from_bits(falling.pos) == key.0
            })
            .map(|(id, _)| *id);
        let (kind, mechanism) = if spawned.is_some() {
            (EventKind::BlockFell, "falling_block")
        } else {
            (EventKind::StateChanged, "scheduled_fall_cancelled")
        };
        let event_id = builder.caused_event(
            Phase::Scheduled,
            kind,
            Some(SubjectRef::Cell(key.0)),
            Some(key.0),
            mechanism,
            &lineage,
        );
        attribution.scheduled_removed.insert(*key, event_id);
        attribution.cells.entry(key.0).or_insert(event_id);
        if let Some(id) = spawned {
            attribution.falling.insert(id, event_id);
        }
    }

    for key in &after.scheduled_falls {
        if before.scheduled_falls.contains(key) {
            if let Some(lineage) = state.scheduled_falls.get(key) {
                next_scheduled.insert(*key, lineage.clone());
            }
            continue;
        }
        let source = scheduler_source(builder, state, before, key.0, action);
        let event_id = builder.caused_event(
            Phase::Scheduled,
            EventKind::BlockFallScheduled,
            Some(SubjectRef::Cell(key.0)),
            Some(key.0),
            "scheduled_fall",
            &source,
        );
        attribution.scheduled_added.insert(*key, event_id);
        next_scheduled.insert(
            *key,
            Lineage {
                event_id,
                root_cause: source.root_cause,
            },
        );
    }
    state.scheduled_falls = next_scheduled;

    // Explosions are consumed before newly chained TNT is registered, so a
    // chained prime can directly parent the explosion that created its fuse.
    let mut next_booms = BTreeMap::new();
    for key in before.pending_booms.difference(&after.pending_booms) {
        let source = state
            .pending_booms
            .get(key)
            .cloned()
            .unwrap_or_else(|| exogenous_lineage(builder, state, Phase::Tnt, "tnt"));
        let due = key.1 <= before.tick;
        let (kind, mechanism) = if due {
            (EventKind::Explosion, "tnt_explosion")
        } else {
            (EventKind::StateChanged, "tnt_fuse_cancelled")
        };
        let event_id = builder.caused_event(
            Phase::Tnt,
            kind,
            Some(SubjectRef::Cell(key.0)),
            Some(key.0),
            mechanism,
            &source,
        );
        attribution.booms_removed.insert(*key, event_id);
        if due {
            let lineage = Lineage {
                event_id,
                root_cause: source.root_cause,
            };
            explosions.push((key.0, event_id, lineage));
            attribution.cells.insert(key.0, event_id);
        }
    }

    for key in &after.pending_booms {
        if before.pending_booms.contains(key) {
            if let Some(lineage) = state.pending_booms.get(key) {
                next_booms.insert(*key, lineage.clone());
            }
            continue;
        }
        let blast_radius = (crate::tnt::BLAST_R as f64).mul_add(1.0, 0.0) as i32;
        let source = explosions
            .iter()
            .find(|(origin, _, _)| chebyshev(*origin, key.0) <= blast_radius)
            .map(|(_, _, lineage)| lineage.clone())
            .unwrap_or_else(|| tnt_source(builder, state, before, key.0, action));
        let event_id = builder.caused_event(
            Phase::Tnt,
            EventKind::TntPrimed,
            Some(SubjectRef::Cell(key.0)),
            Some(key.0),
            "tnt_prime",
            &source,
        );
        attribution.booms_added.insert(*key, event_id);
        next_booms.insert(
            *key,
            Lineage {
                event_id,
                root_cause: source.root_cause,
            },
        );
    }
    state.pending_booms = next_booms;
}

fn scheduler_source(
    builder: &mut Builder,
    state: &mut TraceState,
    before: &WorldTrace,
    at: CellCoord,
    action: &Action,
) -> Lineage {
    for source in [at, at.offset(0, -1, 0)] {
        if let Some(lineage) = state.dirty_cells.get(&source) {
            return lineage.clone();
        }
    }
    if before
        .dirty
        .iter()
        .any(|source| *source == at || *source == at.offset(0, -1, 0))
    {
        return exogenous_lineage(builder, state, Phase::Scheduled, "scheduled_fall");
    }
    if action_changes_world(action) {
        builder.action_lineage()
    } else {
        builder.periodic_lineage(Phase::Scheduled, "falling_block")
    }
}

fn tnt_source(
    builder: &mut Builder,
    state: &mut TraceState,
    before: &WorldTrace,
    at: CellCoord,
    action: &Action,
) -> Lineage {
    let candidates = std::iter::once(at).chain(at.neighbors6());
    for source in candidates {
        if let Some(lineage) = state.dirty_cells.get(&source) {
            return lineage.clone();
        }
    }
    let candidates = std::iter::once(at).chain(at.neighbors6());
    if before
        .dirty
        .iter()
        .any(|dirty| candidates.clone().any(|at| at == *dirty))
    {
        return exogenous_lineage(builder, state, Phase::Tnt, "tnt");
    }
    if action_changes_world(action) {
        builder.action_lineage()
    } else {
        builder.periodic_lineage(Phase::Tnt, "tnt")
    }
}

fn exogenous_lineage(
    builder: &mut Builder,
    state: &mut TraceState,
    phase: Phase,
    mechanism: &'static str,
) -> Lineage {
    let root = state.exogenous_root(builder.branch_id, builder.tick, mechanism);
    builder.root_event(phase, mechanism, root)
}

fn action_changes_world(action: &Action) -> bool {
    action.mine || action.place || action.use_ || action.craft != 0
}

fn chebyshev(a: CellCoord, b: CellCoord) -> i32 {
    (a.x - b.x)
        .abs()
        .max((a.y - b.y).abs())
        .max((a.z - b.z).abs())
}

fn cell_from_bits(position: [u64; 3]) -> CellCoord {
    CellCoord::from_world_pos(WorldPos::from_cells(position.map(f64::from_bits)))
}

fn emit_block_changes(
    builder: &mut Builder,
    trace_state: &TraceState,
    before: &mut World,
    after: &mut World,
    action: &Action,
    explosions: &[(CellCoord, u64, Lineage)],
    attribution: &mut Attribution,
) {
    let mut touched_chunks = BTreeSet::new();
    for (key, chunk) in &before.chunks {
        if chunk.touched {
            touched_chunks.insert(*key);
        }
    }
    for (key, chunk) in &after.chunks {
        if chunk.touched {
            touched_chunks.insert(*key);
        }
    }

    let height = before.height().max(after.height());
    for (cx, cz) in touched_chunks {
        for y in 0..height {
            for lz in 0..16i32 {
                for lx in 0..16i32 {
                    let at = CellCoord::new(cx * 16 + lx, y, cz * 16 + lz);
                    let old = before.get_block(at.x, at.y, at.z);
                    let new = after.get_block(at.x, at.y, at.z);
                    if old == new {
                        continue;
                    }
                    let old_id = cell_id(old);
                    let new_id = cell_id(new);
                    if let Some(event_id) = attribution.cells.get(&at).copied() {
                        builder.delta(
                            event_id,
                            SubjectRef::Cell(at),
                            "cell",
                            TraceValue::Cell(old),
                            TraceValue::Cell(new),
                        );
                        continue;
                    }
                    let blast_radius =
                        (crate::tnt::BLAST_R as f64 * after.physics.scale).round() as i32;
                    if let Some((_, event_id, _)) = explosions
                        .iter()
                        .find(|(origin, _, _)| chebyshev(*origin, at) <= blast_radius)
                    {
                        attribution.cells.insert(at, *event_id);
                        builder.delta(
                            *event_id,
                            SubjectRef::Cell(at),
                            "cell",
                            TraceValue::Cell(old),
                            TraceValue::Cell(new),
                        );
                        continue;
                    }
                    let (phase, kind, mechanism) = if action.mine && old_id != AIR && new_id == AIR
                    {
                        (Phase::AgentAction, EventKind::BlockMined, "mining")
                    } else if old_id == FIRE && new_id == AIR {
                        (Phase::Fire, EventKind::Extinguished, "fire")
                    } else if new_id == FIRE {
                        (Phase::Fire, EventKind::Ignited, "fire")
                    } else if matches!(old_id, WATER | LAVA) || matches!(new_id, WATER | LAVA) {
                        (Phase::Fluid, EventKind::FluidChanged, "fluid")
                    } else if action.place && old_id == AIR && new_id != AIR {
                        (Phase::AgentAction, EventKind::BlockPlaced, "placement")
                    } else if action.use_ && old_id == new_id && matches!(old_id, DOOR | LEVER) {
                        (Phase::AgentAction, EventKind::CircuitChanged, "block_state")
                    } else if old_id == new_id {
                        (Phase::Circuit, EventKind::CircuitChanged, "block_state")
                    } else {
                        (Phase::Scheduled, EventKind::BlockChanged, "block_update")
                    };
                    let subject = SubjectRef::Cell(at);
                    let id = if phase == Phase::AgentAction {
                        builder.event(phase, kind, Some(subject.clone()), Some(at), mechanism)
                    } else if let Some(lineage) = block_change_lineage(
                        builder,
                        trace_state,
                        attribution,
                        at,
                        phase,
                        mechanism,
                        after.scale(),
                    ) {
                        builder.caused_event(
                            phase,
                            kind,
                            Some(subject.clone()),
                            Some(at),
                            mechanism,
                            &lineage,
                        )
                    } else {
                        builder.autonomous_event(
                            phase,
                            kind,
                            Some(subject.clone()),
                            Some(at),
                            mechanism,
                        )
                    };
                    attribution.cells.insert(at, id);
                    builder.delta(
                        id,
                        subject,
                        "cell",
                        TraceValue::Cell(old),
                        TraceValue::Cell(new),
                    );
                }
            }
        }
    }
}

/// Prefer an explicit input or an earlier event in the same mechanism over
/// inventing a periodic root.  `dirty_cells` is the cross-boundary hand-off:
/// mechanisms consume those cells during this transition, so nearby changes
/// remain descendants of the action/intervention that dirtied the world.
fn block_change_lineage(
    builder: &Builder,
    trace_state: &TraceState,
    attribution: &Attribution,
    at: CellCoord,
    phase: Phase,
    mechanism: &'static str,
    scale: f64,
) -> Option<Lineage> {
    let radius = match phase {
        Phase::Fire | Phase::Tnt => scale.ceil() as u64,
        Phase::Fluid | Phase::Circuit | Phase::Scheduled => 1,
        _ => 0,
    };
    let nearest_dirty = trace_state
        .dirty_cells
        .iter()
        .filter_map(|(source, lineage)| {
            let distance = source.manhattan_distance(at);
            (distance <= radius).then_some((distance, *source, lineage))
        })
        .min_by_key(|(distance, source, _)| (*distance, *source))
        .map(|(_, _, lineage)| lineage.clone());
    if nearest_dirty.is_some() {
        return nearest_dirty;
    }

    attribution
        .cells
        .iter()
        .filter_map(|(source, event_id)| {
            let distance = source.manhattan_distance(at);
            if distance > radius {
                return None;
            }
            let lineage = builder.event_lineage(*event_id)?;
            let same_mechanism = builder
                .events
                .iter()
                .find(|record| record.event.id == *event_id)
                .is_some_and(|record| record.event.mechanism == mechanism);
            same_mechanism.then_some((distance, *source, lineage))
        })
        .min_by_key(|(distance, source, _)| (*distance, *source))
        .map(|(_, _, lineage)| lineage)
}

fn emit_agent_changes(
    builder: &mut Builder,
    before: &WorldTrace,
    after: &WorldTrace,
    explosions: &[(CellCoord, u64, Lineage)],
    attribution: &Attribution,
) {
    let subject = SubjectRef::Agent(EntityId::AGENT);
    let old = &before.agent;
    let new = &after.agent;
    let action_id = builder.root_id;

    builder.delta(
        action_id,
        subject.clone(),
        "yaw",
        TraceValue::U64(old.yaw as u64),
        TraceValue::U64(new.yaw as u64),
    );
    builder.delta(
        action_id,
        subject.clone(),
        "pitch",
        TraceValue::U64(old.pitch as u64),
        TraceValue::U64(new.pitch as u64),
    );
    builder.delta(
        action_id,
        subject.clone(),
        "selected",
        TraceValue::U64(old.selected as u64),
        TraceValue::U64(new.selected as u64),
    );
    emit_mining_deltas(builder, action_id, &before.mining, &after.mining);
    builder.delta(
        action_id,
        SubjectRef::World,
        "place_cooldown",
        TraceValue::U64(before.place_cooldown as u64),
        TraceValue::U64(after.place_cooldown as u64),
    );

    if old.pos != new.pos {
        let position = new.pos.map(f64::from_bits);
        let id = builder.event(
            Phase::EntityIntegration,
            EventKind::AgentMoved,
            Some(subject.clone()),
            Some(cell_from_bits(new.pos)),
            "agent_motion",
        );
        builder.delta(
            id,
            subject.clone(),
            "position",
            TraceValue::Vec3Bits(old.pos),
            TraceValue::Vec3Bits(position.map(f64::to_bits)),
        );
    }

    let dynamics_changed = old.vel != new.vel
        || old.on_ground != new.on_ground
        || old.fall_distance != new.fall_distance
        || old.suffocation_timer != new.suffocation_timer
        || old.lava_timer != new.lava_timer
        || old.fire_timer != new.fire_timer;
    let mut collision_lineage = None;
    if dynamics_changed {
        let dynamics_kind = if old.on_ground != new.on_ground {
            EventKind::Collision
        } else {
            EventKind::VelocityChanged
        };
        let id = builder.event(
            Phase::EntityIntegration,
            dynamics_kind,
            Some(subject.clone()),
            None,
            "agent_dynamics",
        );
        if dynamics_kind == EventKind::Collision {
            collision_lineage = builder.event_lineage(id);
        }
        builder.delta(
            id,
            subject.clone(),
            "velocity",
            TraceValue::Vec3Bits(old.vel),
            TraceValue::Vec3Bits(new.vel),
        );
        builder.delta(
            id,
            subject.clone(),
            "on_ground",
            TraceValue::Bool(old.on_ground),
            TraceValue::Bool(new.on_ground),
        );
        builder.delta(
            id,
            subject.clone(),
            "fall_distance",
            TraceValue::F64Bits(old.fall_distance),
            TraceValue::F64Bits(new.fall_distance),
        );
        builder.delta(
            id,
            subject.clone(),
            "suffocation_timer",
            TraceValue::U64(old.suffocation_timer as u64),
            TraceValue::U64(new.suffocation_timer as u64),
        );
        builder.delta(
            id,
            subject.clone(),
            "lava_timer",
            TraceValue::U64(old.lava_timer as u64),
            TraceValue::U64(new.lava_timer as u64),
        );
        builder.delta(
            id,
            subject.clone(),
            "fire_timer",
            TraceValue::U64(old.fire_timer as u64),
            TraceValue::U64(new.fire_timer as u64),
        );
    }

    if old.hp != new.hp || old.dead != new.dead {
        let kind = if !old.dead && new.dead {
            EventKind::Death
        } else if new.hp < old.hp {
            EventKind::Damage
        } else {
            EventKind::StateChanged
        };
        let contact_cell = cell_from_bits(new.pos);
        let contact_lineage = |mechanism| {
            attribution
                .cells
                .get(&contact_cell)
                .and_then(|event_id| builder.mechanism_event_lineage(*event_id, mechanism))
        };
        let inherited = explosions
            .last()
            .map(|(_, _, lineage)| (Phase::Tnt, lineage.clone()))
            .or_else(|| {
                (new.hp < old.hp && new.fire_timer > old.fire_timer)
                    .then(|| contact_lineage("fire"))
                    .flatten()
            })
            .or_else(|| {
                (new.hp < old.hp && new.lava_timer > old.lava_timer)
                    .then(|| contact_lineage("fluid"))
                    .flatten()
            })
            .or_else(|| {
                (new.hp < old.hp)
                    .then(|| collision_lineage.clone())
                    .flatten()
                    .map(|lineage| (Phase::EntityIntegration, lineage))
            });
        let id = if let Some((phase, lineage)) = inherited {
            builder.caused_event(
                phase,
                kind,
                Some(subject.clone()),
                None,
                "health_update",
                &lineage,
            )
        } else {
            builder.event(
                Phase::EntityIntegration,
                kind,
                Some(subject.clone()),
                None,
                "health_update",
            )
        };
        builder.delta(
            id,
            subject.clone(),
            "hp",
            TraceValue::I64(old.hp as i64),
            TraceValue::I64(new.hp as i64),
        );
        builder.delta(
            id,
            subject,
            "dead",
            TraceValue::Bool(old.dead),
            TraceValue::Bool(new.dead),
        );
    }

    for (index, (old_stack, new_stack)) in
        old.inventory.iter().zip(new.inventory.iter()).enumerate()
    {
        if old_stack == new_stack {
            continue;
        }
        let slot = SubjectRef::InventorySlot(index as u8);
        let id = builder.event(
            Phase::ItemLogic,
            EventKind::InventoryChanged,
            Some(slot.clone()),
            None,
            "inventory",
        );
        builder.delta(
            id,
            slot,
            "stack",
            TraceValue::ItemStack {
                item: old_stack.0,
                count: old_stack.1,
            },
            TraceValue::ItemStack {
                item: new_stack.0,
                count: new_stack.1,
            },
        );
    }
}

fn emit_mining_deltas(
    builder: &mut Builder,
    event_id: u64,
    before: &Option<MiningTrace>,
    after: &Option<MiningTrace>,
) {
    if before.as_ref().map(|mining| mining.target) != after.as_ref().map(|mining| mining.target) {
        if let Some(mining) = before {
            builder.delta(
                event_id,
                SubjectRef::Cell(mining.target),
                "mining_target",
                TraceValue::Bool(true),
                TraceValue::Bool(false),
            );
        }
        if let Some(mining) = after {
            builder.delta(
                event_id,
                SubjectRef::Cell(mining.target),
                "mining_target",
                TraceValue::Bool(false),
                TraceValue::Bool(true),
            );
        }
    }
    match (before, after) {
        (Some(old), Some(new)) if old.target == new.target => builder.delta(
            event_id,
            SubjectRef::Cell(old.target),
            "mining_progress",
            TraceValue::F64Bits(old.progress),
            TraceValue::F64Bits(new.progress),
        ),
        (Some(old), None) => builder.delta(
            event_id,
            SubjectRef::Cell(old.target),
            "mining_progress",
            TraceValue::F64Bits(old.progress),
            TraceValue::None,
        ),
        (None, Some(new)) => builder.delta(
            event_id,
            SubjectRef::Cell(new.target),
            "mining_progress",
            TraceValue::None,
            TraceValue::F64Bits(new.progress),
        ),
        _ => {}
    }
}

fn emit_item_changes(builder: &mut Builder, before: &WorldTrace, after: &WorldTrace) {
    let ids: BTreeSet<u64> = before
        .items
        .keys()
        .chain(after.items.keys())
        .copied()
        .collect();
    for id in ids {
        let old = before.items.get(&id);
        let new = after.items.get(&id);
        if old == new {
            continue;
        }
        let (kind, phase, mechanism) = match (old, new) {
            (None, Some(_)) => (EventKind::EntitySpawned, Phase::ItemLogic, "item_spawn"),
            (Some(_), None) => (EventKind::EntityDespawned, Phase::ItemLogic, "item_despawn"),
            _ => (
                EventKind::StateChanged,
                Phase::EntityIntegration,
                "item_physics",
            ),
        };
        let subject = SubjectRef::Entity(EntityId::item(id));
        let event_id = builder.event(phase, kind, Some(subject.clone()), None, mechanism);
        emit_item_deltas(builder, event_id, subject, old, new);
    }
}

fn emit_item_deltas(
    builder: &mut Builder,
    event_id: u64,
    subject: SubjectRef,
    before: Option<&ItemTrace>,
    after: Option<&ItemTrace>,
) {
    builder.delta(
        event_id,
        subject.clone(),
        "present",
        TraceValue::Bool(before.is_some()),
        TraceValue::Bool(after.is_some()),
    );
    builder.delta(
        event_id,
        subject.clone(),
        "item",
        before.map_or(TraceValue::None, |item| TraceValue::U64(item.item as u64)),
        after.map_or(TraceValue::None, |item| TraceValue::U64(item.item as u64)),
    );
    builder.delta(
        event_id,
        subject.clone(),
        "count",
        before.map_or(TraceValue::None, |item| TraceValue::U64(item.count as u64)),
        after.map_or(TraceValue::None, |item| TraceValue::U64(item.count as u64)),
    );
    builder.delta(
        event_id,
        subject.clone(),
        "position",
        before.map_or(TraceValue::None, |item| TraceValue::Vec3Bits(item.pos)),
        after.map_or(TraceValue::None, |item| TraceValue::Vec3Bits(item.pos)),
    );
    builder.delta(
        event_id,
        subject.clone(),
        "velocity",
        before.map_or(TraceValue::None, |item| TraceValue::Vec3Bits(item.vel)),
        after.map_or(TraceValue::None, |item| TraceValue::Vec3Bits(item.vel)),
    );
    builder.delta(
        event_id,
        subject,
        "age",
        before.map_or(TraceValue::None, |item| TraceValue::U64(item.age)),
        after.map_or(TraceValue::None, |item| TraceValue::U64(item.age)),
    );
}

fn emit_falling_changes(
    builder: &mut Builder,
    before: &WorldTrace,
    after: &WorldTrace,
    attribution: &mut Attribution,
) {
    let ids: BTreeSet<u64> = before
        .falling
        .keys()
        .chain(after.falling.keys())
        .copied()
        .collect();
    for id in ids {
        let old = before.falling.get(&id);
        let new = after.falling.get(&id);
        if old == new {
            continue;
        }
        let event_id = if let Some(event_id) = attribution.falling.get(&id) {
            *event_id
        } else {
            let (kind, phase, mechanism) = match (old, new) {
                (None, Some(_)) => (EventKind::BlockFell, Phase::Scheduled, "falling_block"),
                (Some(_), None) => (
                    EventKind::BlockFell,
                    Phase::EntityIntegration,
                    "falling_block_landed",
                ),
                _ => (
                    EventKind::StateChanged,
                    Phase::EntityIntegration,
                    "falling_block_physics",
                ),
            };
            let subject = SubjectRef::Entity(EntityId::falling_block(id));
            let event_id = builder.event(phase, kind, Some(subject), None, mechanism);
            attribution.falling.insert(id, event_id);
            event_id
        };
        emit_falling_deltas(
            builder,
            event_id,
            SubjectRef::Entity(EntityId::falling_block(id)),
            old,
            new,
        );
    }
}

fn emit_falling_deltas(
    builder: &mut Builder,
    event_id: u64,
    subject: SubjectRef,
    before: Option<&FallingTrace>,
    after: Option<&FallingTrace>,
) {
    builder.delta(
        event_id,
        subject.clone(),
        "present",
        TraceValue::Bool(before.is_some()),
        TraceValue::Bool(after.is_some()),
    );
    builder.delta(
        event_id,
        subject.clone(),
        "block",
        before.map_or(TraceValue::None, |block| TraceValue::Cell(block.block)),
        after.map_or(TraceValue::None, |block| TraceValue::Cell(block.block)),
    );
    builder.delta(
        event_id,
        subject.clone(),
        "position",
        before.map_or(TraceValue::None, |block| TraceValue::Vec3Bits(block.pos)),
        after.map_or(TraceValue::None, |block| TraceValue::Vec3Bits(block.pos)),
    );
    builder.delta(
        event_id,
        subject.clone(),
        "velocity",
        before.map_or(TraceValue::None, |block| TraceValue::Vec3Bits(block.vel)),
        after.map_or(TraceValue::None, |block| TraceValue::Vec3Bits(block.vel)),
    );
    builder.delta(
        event_id,
        subject,
        "fall_distance",
        before.map_or(TraceValue::None, |block| {
            TraceValue::F64Bits(block.fall_distance)
        }),
        after.map_or(TraceValue::None, |block| {
            TraceValue::F64Bits(block.fall_distance)
        }),
    );
}

fn emit_furnace_changes(
    builder: &mut Builder,
    before: &WorldTrace,
    after: &WorldTrace,
    action: &Action,
) {
    let cells: BTreeSet<CellCoord> = before
        .furnaces
        .keys()
        .chain(after.furnaces.keys())
        .copied()
        .collect();
    for at in cells {
        let old = before.furnaces.get(&at);
        let new = after.furnaces.get(&at);
        if old == new {
            continue;
        }
        let phase = if action.use_ {
            Phase::AgentAction
        } else {
            Phase::Scheduled
        };
        let event_id = builder.event(
            phase,
            EventKind::StateChanged,
            Some(SubjectRef::Cell(at)),
            Some(at),
            "furnace",
        );
        builder.delta(
            event_id,
            SubjectRef::Cell(at),
            "furnace_present",
            TraceValue::Bool(old.is_some()),
            TraceValue::Bool(new.is_some()),
        );
        builder.delta(
            event_id,
            SubjectRef::Cell(at),
            "furnace_remaining",
            old.map_or(TraceValue::None, |state| {
                TraceValue::U64(state.remaining as u64)
            }),
            new.map_or(TraceValue::None, |state| {
                TraceValue::U64(state.remaining as u64)
            }),
        );
        builder.delta(
            event_id,
            SubjectRef::Cell(at),
            "furnace_out_ready",
            old.map_or(TraceValue::None, |state| TraceValue::Bool(state.out_ready)),
            new.map_or(TraceValue::None, |state| TraceValue::Bool(state.out_ready)),
        );
        builder.delta(
            event_id,
            SubjectRef::Cell(at),
            "furnace_fuel_left",
            old.map_or(TraceValue::None, |state| {
                TraceValue::U64(state.fuel_left as u64)
            }),
            new.map_or(TraceValue::None, |state| {
                TraceValue::U64(state.fuel_left as u64)
            }),
        );
    }
}

fn emit_scheduler_deltas(
    builder: &mut Builder,
    before: &WorldTrace,
    after: &WorldTrace,
    attribution: &Attribution,
) {
    for key in before.scheduled_falls.difference(&after.scheduled_falls) {
        let event_id = attribution.scheduled_removed[key];
        builder.delta(
            event_id,
            SubjectRef::Cell(key.0),
            "scheduled_fall_due_tick",
            TraceValue::U64(key.1),
            TraceValue::None,
        );
    }
    for key in after.scheduled_falls.difference(&before.scheduled_falls) {
        let event_id = attribution.scheduled_added[key];
        builder.delta(
            event_id,
            SubjectRef::Cell(key.0),
            "scheduled_fall_due_tick",
            TraceValue::None,
            TraceValue::U64(key.1),
        );
    }
    emit_membership_deltas(
        builder,
        &before.scheduled_set,
        &after.scheduled_set,
        Phase::Scheduled,
        "scheduled_fall",
        "scheduled_set_member",
        Some(&attribution.cells),
    );

    for key in before.pending_booms.difference(&after.pending_booms) {
        let event_id = attribution.booms_removed[key];
        builder.delta(
            event_id,
            SubjectRef::Cell(key.0),
            "pending_boom_due_tick",
            TraceValue::U64(key.1),
            TraceValue::None,
        );
    }
    for key in after.pending_booms.difference(&before.pending_booms) {
        let event_id = attribution.booms_added[key];
        builder.delta(
            event_id,
            SubjectRef::Cell(key.0),
            "pending_boom_due_tick",
            TraceValue::None,
            TraceValue::U64(key.1),
        );
    }
}

fn emit_active_set_deltas(
    builder: &mut Builder,
    before: &WorldTrace,
    after: &WorldTrace,
    attribution: &Attribution,
) {
    let mut state_events = attribution.cells.clone();
    for ((at, _), event_id) in attribution
        .booms_added
        .iter()
        .chain(attribution.booms_removed.iter())
    {
        state_events.entry(*at).or_insert(*event_id);
    }
    emit_membership_deltas(
        builder,
        &before.active_fluids,
        &after.active_fluids,
        Phase::Fluid,
        "fluid",
        "active_fluid",
        Some(&state_events),
    );
    emit_membership_deltas(
        builder,
        &before.circuit_cells,
        &after.circuit_cells,
        Phase::Circuit,
        "circuit",
        "circuit_cell",
        Some(&state_events),
    );
    emit_membership_deltas(
        builder,
        &before.active_fire,
        &after.active_fire,
        Phase::Fire,
        "fire",
        "active_fire",
        Some(&state_events),
    );
    emit_membership_deltas(
        builder,
        &before.tnt_cells,
        &after.tnt_cells,
        Phase::Tnt,
        "tnt",
        "live_tnt",
        Some(&state_events),
    );
}

fn emit_membership_deltas(
    builder: &mut Builder,
    before: &BTreeSet<CellCoord>,
    after: &BTreeSet<CellCoord>,
    phase: Phase,
    mechanism: &'static str,
    field: &'static str,
    attribution: Option<&BTreeMap<CellCoord, u64>>,
) {
    for (at, old, new) in before
        .difference(after)
        .map(|at| (*at, true, false))
        .chain(after.difference(before).map(|at| (*at, false, true)))
    {
        let event_id = attribution
            .and_then(|events| events.get(&at).copied())
            .unwrap_or_else(|| {
                builder.autonomous_event(
                    phase,
                    EventKind::StateChanged,
                    Some(SubjectRef::Cell(at)),
                    Some(at),
                    mechanism,
                )
            });
        builder.delta(
            event_id,
            SubjectRef::Cell(at),
            field,
            TraceValue::Bool(old),
            TraceValue::Bool(new),
        );
    }
}

fn emit_misc_state_deltas(builder: &mut Builder, before: &WorldTrace, after: &WorldTrace) {
    let root_id = builder.root_id;
    builder.delta(
        root_id,
        SubjectRef::Scheduler("dirty"),
        "dirty_queue",
        TraceValue::CellSequence(before.dirty.clone()),
        TraceValue::CellSequence(after.dirty.clone()),
    );
    let (old_state, old_inc) = before.rng;
    let (new_state, new_inc) = after.rng;
    for (field, old, new) in [
        ("rng_state_low", old_state as u64, new_state as u64),
        (
            "rng_state_high",
            (old_state >> 64) as u64,
            (new_state >> 64) as u64,
        ),
        ("rng_increment_low", old_inc as u64, new_inc as u64),
        (
            "rng_increment_high",
            (old_inc >> 64) as u64,
            (new_inc >> 64) as u64,
        ),
        ("next_item_id", before.next_item_id, after.next_item_id),
        (
            "next_falling_id",
            before.next_falling_id,
            after.next_falling_id,
        ),
    ] {
        builder.delta(
            root_id,
            SubjectRef::World,
            field,
            TraceValue::U64(old),
            TraceValue::U64(new),
        );
    }
}

fn update_trace_state(
    state: &mut TraceState,
    after: &WorldTrace,
    attribution: &Attribution,
    builder: &mut Builder,
) {
    state.dirty_cells.clear();
    for at in &after.dirty {
        let lineage = attribution
            .cells
            .get(at)
            .and_then(|event_id| builder.event_lineage(*event_id))
            .unwrap_or_else(|| builder.periodic_lineage(Phase::Scheduled, "block_update"));
        state.dirty_cells.insert(*at, lineage);
    }
}

/// A serializable intervention vocabulary. Python bindings expose the same
/// tagged representation instead of accepting arbitrary mutation closures.
#[derive(Clone, Debug, PartialEq)]
pub enum InterventionSpec {
    SetCell {
        at: CellCoord,
        cell: u16,
    },
    TeleportAgent {
        position: WorldPos,
    },
    SetAgentVelocity {
        velocity: [f64; 3],
    },
    GiveItem {
        item: u16,
        count: u16,
    },
    /// Pull an existing item into the selected hotbar slot using the same
    /// inventory semantics as scripted experts.  Recording this as an input
    /// keeps inventory management inside the causal transition boundary.
    SwapToHotbar {
        item: u16,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct InterventionOutcome {
    pub clock: SimClock,
    pub before_hash: Option<u64>,
    pub after_hash: Option<u64>,
    pub event: Option<WorldEvent>,
    pub deltas: Vec<StateDelta>,
}

pub fn apply_intervention(
    world: &mut World,
    spec: &InterventionSpec,
    level: TraceLevel,
    branch_id: u64,
    intervention_id: u64,
) -> Result<InterventionOutcome, String> {
    apply_intervention_with_state(
        world,
        spec,
        level,
        branch_id,
        intervention_id,
        &mut TraceState::default(),
    )
}

/// Stateful form of [`apply_intervention`]. The resulting event is registered
/// as provenance for dirty-cell work observed by later traced transitions.
pub fn apply_intervention_with_state(
    world: &mut World,
    spec: &InterventionSpec,
    level: TraceLevel,
    branch_id: u64,
    intervention_id: u64,
    trace_state: &mut TraceState,
) -> Result<InterventionOutcome, String> {
    validate_intervention(spec)?;
    let clock = world.sim_clock();
    let before_hash = (level == TraceLevel::Full).then(|| world.hash());
    if level == TraceLevel::Off {
        mutate(world, spec)?;
        trace_state.invalidate();
        return Ok(InterventionOutcome {
            clock,
            before_hash: None,
            after_hash: None,
            event: None,
            deltas: Vec::new(),
        });
    }

    let ordinal = trace_state.next_intervention_ordinal;
    trace_state.next_intervention_ordinal = trace_state.next_intervention_ordinal.wrapping_add(1);
    let id = intervention_event_id_with_ordinal(branch_id, world.tick, intervention_id, ordinal);
    let root = RootCause::Intervention {
        branch_id,
        intervention_id,
    };
    let before = WorldTrace::capture(world);
    let (target, location, before_cell) = match spec {
        InterventionSpec::SetCell { at, .. } => (
            SubjectRef::Cell(*at),
            Some(*at),
            Some(world.get_block(at.x, at.y, at.z)),
        ),
        InterventionSpec::TeleportAgent { position } => (
            SubjectRef::Agent(EntityId::AGENT),
            Some(CellCoord::from_world_pos(*position)),
            None,
        ),
        InterventionSpec::SetAgentVelocity { .. }
        | InterventionSpec::GiveItem { .. }
        | InterventionSpec::SwapToHotbar { .. } => (SubjectRef::Agent(EntityId::AGENT), None, None),
    };
    mutate(world, spec)?;
    let after = WorldTrace::capture(world);
    let event = WorldEvent {
        id,
        tick: world.tick,
        phase: Phase::Intervention,
        kind: EventKind::InterventionApplied,
        actor: None,
        target: Some(target.clone()),
        location,
        mechanism: "intervention",
        parent_ids: Vec::new(),
        root_cause: root.clone(),
    };
    let mut deltas = Vec::new();
    emit_intervention_deltas(
        level,
        &mut deltas,
        id,
        spec,
        &before,
        &after,
        before_cell,
        world,
    );
    if let InterventionSpec::SetCell { at, .. } = spec {
        if before_cell != Some(world.get_block(at.x, at.y, at.z)) {
            trace_state.dirty_cells.insert(
                *at,
                Lineage {
                    event_id: id,
                    root_cause: root,
                },
            );
        }
    }
    Ok(InterventionOutcome {
        clock,
        before_hash,
        after_hash: (level == TraceLevel::Full).then(|| world.hash()),
        event: Some(event),
        deltas,
    })
}

#[allow(clippy::too_many_arguments)]
fn emit_intervention_deltas(
    level: TraceLevel,
    deltas: &mut Vec<StateDelta>,
    event_id: u64,
    spec: &InterventionSpec,
    before: &WorldTrace,
    after: &WorldTrace,
    before_cell: Option<u16>,
    world: &mut World,
) {
    let agent = SubjectRef::Agent(EntityId::AGENT);
    push_delta(
        level,
        deltas,
        event_id,
        SubjectRef::Scheduler("dirty"),
        "dirty_queue",
        TraceValue::CellSequence(before.dirty.clone()),
        TraceValue::CellSequence(after.dirty.clone()),
    );
    match spec {
        InterventionSpec::SetCell { at, .. } => {
            push_delta(
                level,
                deltas,
                event_id,
                SubjectRef::Cell(*at),
                "cell",
                TraceValue::Cell(before_cell.expect("set-cell captures its target")),
                TraceValue::Cell(world.get_block(at.x, at.y, at.z)),
            );
            for (before_set, after_set, field) in [
                (
                    &before.scheduled_set,
                    &after.scheduled_set,
                    "scheduled_set_member",
                ),
                (&before.active_fluids, &after.active_fluids, "active_fluid"),
                (&before.circuit_cells, &after.circuit_cells, "circuit_cell"),
                (&before.active_fire, &after.active_fire, "active_fire"),
                (&before.tnt_cells, &after.tnt_cells, "live_tnt"),
            ] {
                push_intervention_membership_deltas(
                    level, deltas, event_id, before_set, after_set, field,
                );
            }
        }
        InterventionSpec::TeleportAgent { .. } => {
            push_delta(
                level,
                deltas,
                event_id,
                agent.clone(),
                "position",
                TraceValue::Vec3Bits(before.agent.pos),
                TraceValue::Vec3Bits(after.agent.pos),
            );
            push_delta(
                level,
                deltas,
                event_id,
                agent.clone(),
                "velocity",
                TraceValue::Vec3Bits(before.agent.vel),
                TraceValue::Vec3Bits(after.agent.vel),
            );
            push_delta(
                level,
                deltas,
                event_id,
                agent,
                "fall_distance",
                TraceValue::F64Bits(before.agent.fall_distance),
                TraceValue::F64Bits(after.agent.fall_distance),
            );
        }
        InterventionSpec::SetAgentVelocity { .. } => push_delta(
            level,
            deltas,
            event_id,
            agent,
            "velocity",
            TraceValue::Vec3Bits(before.agent.vel),
            TraceValue::Vec3Bits(after.agent.vel),
        ),
        InterventionSpec::GiveItem { item, .. } => {
            push_delta(
                level,
                deltas,
                event_id,
                agent,
                "inventory_count",
                TraceValue::U64(inventory_count(&before.agent, *item)),
                TraceValue::U64(inventory_count(&after.agent, *item)),
            );
            for (index, (old, new)) in before
                .agent
                .inventory
                .iter()
                .zip(after.agent.inventory.iter())
                .enumerate()
            {
                push_delta(
                    level,
                    deltas,
                    event_id,
                    SubjectRef::InventorySlot(index as u8),
                    "stack",
                    TraceValue::ItemStack {
                        item: old.0,
                        count: old.1,
                    },
                    TraceValue::ItemStack {
                        item: new.0,
                        count: new.1,
                    },
                );
            }
        }
        InterventionSpec::SwapToHotbar { .. } => {
            push_delta(
                level,
                deltas,
                event_id,
                agent,
                "selected",
                TraceValue::U64(before.agent.selected as u64),
                TraceValue::U64(after.agent.selected as u64),
            );
            for (index, (old, new)) in before
                .agent
                .inventory
                .iter()
                .zip(after.agent.inventory.iter())
                .enumerate()
            {
                push_delta(
                    level,
                    deltas,
                    event_id,
                    SubjectRef::InventorySlot(index as u8),
                    "stack",
                    TraceValue::ItemStack {
                        item: old.0,
                        count: old.1,
                    },
                    TraceValue::ItemStack {
                        item: new.0,
                        count: new.1,
                    },
                );
            }
        }
    }
}

fn push_intervention_membership_deltas(
    level: TraceLevel,
    deltas: &mut Vec<StateDelta>,
    event_id: u64,
    before: &BTreeSet<CellCoord>,
    after: &BTreeSet<CellCoord>,
    field: &'static str,
) {
    for (at, old, new) in before
        .difference(after)
        .map(|at| (*at, true, false))
        .chain(after.difference(before).map(|at| (*at, false, true)))
    {
        push_delta(
            level,
            deltas,
            event_id,
            SubjectRef::Cell(at),
            field,
            TraceValue::Bool(old),
            TraceValue::Bool(new),
        );
    }
}

fn inventory_count(agent: &AgentTrace, item: u16) -> u64 {
    agent
        .inventory
        .iter()
        .filter(|stack| stack.0 == item)
        .map(|stack| stack.1 as u64)
        .sum()
}

fn push_delta(
    level: TraceLevel,
    deltas: &mut Vec<StateDelta>,
    event_id: u64,
    subject: SubjectRef,
    field_or_cell: &'static str,
    before: TraceValue,
    after: TraceValue,
) {
    if level == TraceLevel::Full && before != after {
        deltas.push(StateDelta {
            event_id,
            subject,
            field_or_cell,
            before,
            after,
        });
    }
}

/// Validate every fallible intervention field before tracing cursors or world
/// state can advance.
pub fn validate_intervention(spec: &InterventionSpec) -> Result<(), String> {
    match spec {
        InterventionSpec::SetCell { cell, .. } => crate::block::validate_cell(*cell),
        InterventionSpec::TeleportAgent { position } => {
            if position.cells().iter().all(|value| value.is_finite()) {
                Ok(())
            } else {
                Err("teleport position must contain only finite values".into())
            }
        }
        InterventionSpec::SetAgentVelocity { velocity } => {
            if velocity.iter().all(|value| value.is_finite()) {
                Ok(())
            } else {
                Err("agent velocity must contain only finite values".into())
            }
        }
        InterventionSpec::GiveItem { item, .. } | InterventionSpec::SwapToHotbar { item } => {
            if crate::block::is_known_item(*item) {
                Ok(())
            } else {
                Err(format!("unknown item id {item}"))
            }
        }
    }
}

fn mutate(world: &mut World, spec: &InterventionSpec) -> Result<(), String> {
    match spec {
        InterventionSpec::SetCell { at, cell } => {
            world.try_set_block(at.x, at.y, at.z, *cell)?;
        }
        InterventionSpec::TeleportAgent { position } => {
            world.agent.pos = position.cells();
            world.agent.vel = [0.0; 3];
            world.agent.fall_distance = 0.0;
        }
        InterventionSpec::SetAgentVelocity { velocity } => world.agent.vel = *velocity,
        InterventionSpec::GiveItem { item, count } => {
            world.agent.inventory.add(*item, *count);
        }
        InterventionSpec::SwapToHotbar { item } => {
            world.swap_to_hotbar(*item);
            // `last_swap` is legacy recorder metadata, not World State.  The
            // intervention itself is now the authoritative serialized input.
            world.last_swap = None;
        }
    }
    Ok(())
}

/// Create an independent branch using the canonical snapshot contract.
pub fn fork_world(world: &World) -> Result<World, String> {
    World::restore(&world.snapshot())
}

#[derive(Debug)]
pub struct BranchComparison {
    pub common_before_hash: u64,
    pub control_after_hash: u64,
    pub treatment_after_hash: u64,
    pub diverged: bool,
}

/// Compare equal-length factual and counterfactual action sequences from one
/// immutable pre-state.
pub fn compare_branches(
    source: &World,
    treatment: &InterventionSpec,
    control_actions: &[Action],
    treatment_actions: &[Action],
) -> Result<BranchComparison, String> {
    if control_actions.len() != treatment_actions.len() {
        return Err(format!(
            "counterfactual rollouts require equal-length action sequences (got {} and {})",
            control_actions.len(),
            treatment_actions.len()
        ));
    }
    if control_actions != treatment_actions {
        return Err(
            "counterfactual rollouts require identical action sequences; the intervention must be the only differing input"
                .to_owned(),
        );
    }
    let common_before_hash = source.hash();
    let mut control = fork_world(source)?;
    let mut treated = fork_world(source)?;
    apply_intervention(&mut treated, treatment, TraceLevel::Off, 1, 0)?;
    for action in control_actions {
        crate::step(&mut control, action);
    }
    for action in treatment_actions {
        crate::step(&mut treated, action);
    }
    let control_after_hash = control.hash();
    let treatment_after_hash = treated.hash();
    Ok(BranchComparison {
        common_before_hash,
        control_after_hash,
        treatment_after_hash,
        diverged: control_after_hash != treatment_after_hash,
    })
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_coord(out: &mut Vec<u8>, at: CellCoord) {
    put_i32(out, at.x);
    put_i32(out, at.y);
    put_i32(out, at.z);
}

fn put_lineage(out: &mut Vec<u8>, lineage: &Lineage) {
    put_u64(out, lineage.event_id);
    match &lineage.root_cause {
        RootCause::Action { branch_id, tick } => {
            out.push(0);
            put_u64(out, *branch_id);
            put_u64(out, *tick);
        }
        RootCause::Intervention {
            branch_id,
            intervention_id,
        } => {
            out.push(1);
            put_u64(out, *branch_id);
            put_u64(out, *intervention_id);
        }
        RootCause::Periodic { tick, mechanism } => {
            out.push(2);
            put_u64(out, *tick);
            put_mechanism(out, mechanism);
        }
        RootCause::Exogenous {
            branch_id,
            tick,
            ordinal,
            mechanism,
        } => {
            out.push(3);
            put_u64(out, *branch_id);
            put_u64(out, *tick);
            put_u64(out, *ordinal);
            put_mechanism(out, mechanism);
        }
    }
}

fn put_mechanism(out: &mut Vec<u8>, mechanism: &str) {
    put_u32(out, mechanism.len() as u32);
    out.extend_from_slice(mechanism.as_bytes());
}

fn read_lineage_map_with_due(
    reader: &mut TraceReader<'_>,
) -> Result<BTreeMap<(CellCoord, u64), Lineage>, String> {
    let count = reader.u32()? as usize;
    let mut values = BTreeMap::new();
    for _ in 0..count {
        let at = reader.coord()?;
        let due = reader.u64()?;
        let lineage = reader.lineage()?;
        if values.insert((at, due), lineage).is_some() {
            return Err("duplicate scheduled trace lineage".into());
        }
    }
    Ok(values)
}

struct TraceReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> TraceReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| "trace-state length overflow".to_string())?;
        if end > self.bytes.len() {
            return Err("truncated trace-state snapshot".into());
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn remaining(&self) -> &'a [u8] {
        &self.bytes[self.offset..]
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }

    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }

    fn coord(&mut self) -> Result<CellCoord, String> {
        Ok(CellCoord::new(self.i32()?, self.i32()?, self.i32()?))
    }

    fn mechanism(&mut self) -> Result<&'static str, String> {
        let len = self.u32()? as usize;
        let raw = std::str::from_utf8(self.take(len)?)
            .map_err(|_| "invalid trace mechanism encoding".to_string())?;
        intern_mechanism(raw).ok_or_else(|| format!("unknown trace mechanism {raw:?}"))
    }

    fn lineage(&mut self) -> Result<Lineage, String> {
        let event_id = self.u64()?;
        let root_cause = match self.take(1)?[0] {
            0 => RootCause::Action {
                branch_id: self.u64()?,
                tick: self.u64()?,
            },
            1 => RootCause::Intervention {
                branch_id: self.u64()?,
                intervention_id: self.u64()?,
            },
            2 => RootCause::Periodic {
                tick: self.u64()?,
                mechanism: self.mechanism()?,
            },
            3 => RootCause::Exogenous {
                branch_id: self.u64()?,
                tick: self.u64()?,
                ordinal: self.u64()?,
                mechanism: self.mechanism()?,
            },
            tag => return Err(format!("unknown trace root-cause tag {tag}")),
        };
        Ok(Lineage {
            event_id,
            root_cause,
        })
    }
}

fn intern_mechanism(value: &str) -> Option<&'static str> {
    Some(match value {
        "agent_action" => "agent_action",
        "agent_motion" => "agent_motion",
        "agent_dynamics" => "agent_dynamics",
        "collision_resolution" => "collision_resolution",
        "health_update" => "health_update",
        "inventory" => "inventory",
        "item_spawn" => "item_spawn",
        "item_despawn" => "item_despawn",
        "item_physics" => "item_physics",
        "item_pickup" => "item_pickup",
        "item_pickup_empty" => "item_pickup_empty",
        "crafting" => "crafting",
        "mining" => "mining",
        "placement" => "placement",
        "smelting" => "smelting",
        "furnace" => "furnace",
        "fluid" => "fluid",
        "fire" => "fire",
        "circuit" => "circuit",
        "block_state" => "block_state",
        "block_update" => "block_update",
        "scheduled_fall" => "scheduled_fall",
        "scheduled_fall_cancelled" => "scheduled_fall_cancelled",
        "falling_block" => "falling_block",
        "falling_block_landed" => "falling_block_landed",
        "falling_block_physics" => "falling_block_physics",
        "tnt" => "tnt",
        "tnt_prime" => "tnt_prime",
        "tnt_explosion" => "tnt_explosion",
        "tnt_fuse_cancelled" => "tnt_fuse_cancelled",
        _ => return None,
    })
}
