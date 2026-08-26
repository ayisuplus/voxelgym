use std::collections::{HashMap, HashSet};

use voxel_core::block::{AIR, DIRT, DOOR, FIRE, SAND, STONE, TNT, WATER};
use voxel_core::spatial::{CellCoord, EntityId, WorldPos};
use voxel_core::trace::{
    apply_intervention_with_state as try_apply_intervention_with_state, step_traced_with_state,
    EventKind, InterventionOutcome, InterventionSpec, Phase, RootCause, TraceLevel, TraceState,
    TraceValue,
};
use voxel_core::worldgen::Preset;
use voxel_core::{Action, World};

fn idle() -> Action {
    Action {
        pitch: 4,
        ..Action::default()
    }
}

fn apply_intervention_with_state(
    world: &mut World,
    spec: &InterventionSpec,
    level: TraceLevel,
    branch_id: u64,
    intervention_id: u64,
    trace_state: &mut TraceState,
) -> InterventionOutcome {
    try_apply_intervention_with_state(world, spec, level, branch_id, intervention_id, trace_state)
        .expect("valid intervention")
}

#[test]
fn scheduled_sand_fall_keeps_the_intervention_parent_across_ticks() {
    let mut world = World::new(7, Preset::Void, Vec::new());
    world.set_block(5, 5, 5, STONE);
    world.set_block(5, 6, 5, SAND);
    voxel_core::step(&mut world, &idle());

    let mut trace = TraceState::default();
    let intervention = apply_intervention_with_state(
        &mut world,
        &InterventionSpec::SetCell {
            at: CellCoord::new(5, 5, 5),
            cell: AIR,
        },
        TraceLevel::Full,
        31,
        8,
        &mut trace,
    );
    let intervention_event = intervention.event.expect("tracked intervention");

    let scheduled = step_traced_with_state(&mut world, &idle(), TraceLevel::Full, 31, &mut trace);
    let schedule_event = scheduled
        .events
        .iter()
        .find(|event| event.kind == EventKind::BlockFallScheduled)
        .expect("support removal schedules the sand");
    assert_eq!(schedule_event.parent_ids, vec![intervention_event.id]);
    assert_eq!(schedule_event.root_cause, intervention_event.root_cause);

    let fallen = step_traced_with_state(&mut world, &idle(), TraceLevel::Full, 31, &mut trace);
    let fall_event = fallen
        .events
        .iter()
        .find(|event| event.kind == EventKind::BlockFell)
        .expect("due support check converts the block");
    assert_eq!(fall_event.parent_ids, vec![schedule_event.id]);
    assert_eq!(fall_event.root_cause, intervention_event.root_cause);
}

#[test]
fn primed_tnt_keeps_the_intervention_parent_until_explosion() {
    let mut world = World::new(11, Preset::Void, Vec::new());
    world.set_block(4, 6, 4, TNT);
    voxel_core::step(&mut world, &idle());

    let mut trace = TraceState::default();
    let intervention = apply_intervention_with_state(
        &mut world,
        &InterventionSpec::SetCell {
            at: CellCoord::new(4, 6, 5),
            cell: FIRE,
        },
        TraceLevel::Events,
        91,
        2,
        &mut trace,
    );
    let intervention_event = intervention.event.expect("tracked intervention");

    let primed = step_traced_with_state(&mut world, &idle(), TraceLevel::Events, 91, &mut trace);
    let prime_event = primed
        .events
        .iter()
        .find(|event| event.kind == EventKind::TntPrimed)
        .expect("adjacent fire primes TNT");
    assert_eq!(prime_event.parent_ids, vec![intervention_event.id]);
    assert_eq!(prime_event.root_cause, intervention_event.root_cause);

    let mut explosion = None;
    for _ in 0..4 {
        let outcome =
            step_traced_with_state(&mut world, &idle(), TraceLevel::Events, 91, &mut trace);
        if let Some(event) = outcome
            .events
            .into_iter()
            .find(|event| event.kind == EventKind::Explosion)
        {
            explosion = Some(event);
            break;
        }
    }
    let explosion = explosion.expect("fuse eventually explodes");
    assert_eq!(explosion.parent_ids, vec![prime_event.id]);
    assert_eq!(explosion.root_cause, intervention_event.root_cause);
}

#[test]
fn fluid_changes_inherit_the_intervention_that_introduced_the_source() {
    let mut world = World::new(12, Preset::Void, Vec::new());
    world.set_block(0, 4, 0, STONE);
    let mut trace = TraceState::default();
    let intervention = apply_intervention_with_state(
        &mut world,
        &InterventionSpec::SetCell {
            at: CellCoord::new(0, 5, 0),
            cell: WATER,
        },
        TraceLevel::Full,
        92,
        3,
        &mut trace,
    );
    let intervention_event = intervention.event.expect("tracked intervention");

    let spread = step_traced_with_state(&mut world, &idle(), TraceLevel::Full, 92, &mut trace);
    let fluid_events: Vec<_> = spread
        .events
        .iter()
        .filter(|event| event.kind == EventKind::FluidChanged)
        .collect();
    assert!(!fluid_events.is_empty(), "the source spreads on tick zero");
    assert!(fluid_events.iter().all(|event| {
        event.parent_ids == vec![intervention_event.id]
            && event.root_cause == intervention_event.root_cause
    }));
}

#[test]
fn trace_state_snapshot_restores_ids_parents_and_allocators() {
    let mut world = World::new(13, Preset::Void, Vec::new());
    world.set_block(2, 5, 2, STONE);
    world.set_block(2, 6, 2, SAND);
    voxel_core::step(&mut world, &idle());
    let mut trace = TraceState::default();
    apply_intervention_with_state(
        &mut world,
        &InterventionSpec::SetCell {
            at: CellCoord::new(2, 5, 2),
            cell: AIR,
        },
        TraceLevel::Full,
        4,
        1,
        &mut trace,
    );
    step_traced_with_state(&mut world, &idle(), TraceLevel::Full, 4, &mut trace);

    let world_snapshot = world.snapshot();
    let trace_snapshot = trace.snapshot();
    let mut restored_world = World::restore(&world_snapshot).unwrap();
    let mut restored_trace = TraceState::restore(&trace_snapshot).unwrap();
    assert_eq!(restored_trace.snapshot(), trace_snapshot);
    assert_eq!(
        restored_trace.next_root_ordinal(),
        trace.next_root_ordinal()
    );
    assert_eq!(
        restored_trace.next_intervention_ordinal(),
        trace.next_intervention_ordinal()
    );

    let expected = step_traced_with_state(&mut world, &idle(), TraceLevel::Full, 4, &mut trace);
    let actual = step_traced_with_state(
        &mut restored_world,
        &idle(),
        TraceLevel::Full,
        4,
        &mut restored_trace,
    );
    assert_eq!(actual, expected);
    assert_eq!(restored_trace.snapshot(), trace.snapshot());
}

#[test]
fn events_and_deltas_are_canonical_by_phase_and_event_group() {
    let mut world = World::new(19, Preset::Void, Vec::new());
    world.set_block(6, 8, 6, FIRE);
    world.set_block(6, 8, 7, TNT);
    world.spawn_item(DIRT, 1, [3.5, 9.0, 3.5]);
    let mut trace = TraceState::default();
    let outcome = step_traced_with_state(
        &mut world,
        &Action {
            yaw: 7,
            pitch: 2,
            hotbar: 3,
            ..Action::default()
        },
        TraceLevel::Full,
        1,
        &mut trace,
    );

    assert!(outcome
        .events
        .windows(2)
        .all(|pair| (pair[0].phase as u8) <= (pair[1].phase as u8)));
    let rank: HashMap<_, _> = outcome
        .events
        .iter()
        .enumerate()
        .map(|(index, event)| (event.id, index))
        .collect();
    assert!(outcome
        .deltas
        .windows(2)
        .all(|pair| rank[&pair[0].event_id] <= rank[&pair[1].event_id]));
    assert!(outcome
        .deltas
        .iter()
        .all(|delta| rank.contains_key(&delta.event_id)));
}

#[test]
fn full_trace_describes_orientation_selected_and_item_motion_exactly() {
    let mut world = World::new(23, Preset::Void, Vec::new());
    world.spawn_item(DIRT, 3, [20.5, 20.5, 20.5]);
    let item_id = EntityId::item(world.items[0].id);
    let before_pos = world.items[0].pos.map(f64::to_bits);
    let before_vel = world.items[0].vel.map(f64::to_bits);
    let mut trace = TraceState::default();

    let outcome = step_traced_with_state(
        &mut world,
        &Action {
            yaw: 5,
            pitch: 3,
            hotbar: 2,
            ..Action::default()
        },
        TraceLevel::Full,
        2,
        &mut trace,
    );
    let by_field = |subject: &voxel_core::trace::SubjectRef, field: &str| {
        outcome
            .deltas
            .iter()
            .find(|delta| &delta.subject == subject && delta.field_or_cell == field)
            .unwrap_or_else(|| panic!("missing {subject:?}.{field}"))
    };
    let agent = voxel_core::trace::SubjectRef::Agent(EntityId::AGENT);
    assert_eq!(
        by_field(&agent, "yaw").after,
        TraceValue::U64((75.0f32).to_bits() as u64)
    );
    assert_eq!(
        by_field(&agent, "pitch").after,
        TraceValue::U64((-15.0f32).to_bits() as u64)
    );
    assert_eq!(by_field(&agent, "selected").after, TraceValue::U64(2));

    let item = voxel_core::trace::SubjectRef::Entity(item_id);
    assert_eq!(
        by_field(&item, "position").before,
        TraceValue::Vec3Bits(before_pos)
    );
    assert_eq!(
        by_field(&item, "velocity").before,
        TraceValue::Vec3Bits(before_vel)
    );
    assert_eq!(by_field(&item, "age").after, TraceValue::U64(1));
}

#[test]
fn intervention_deltas_report_every_actual_agent_and_inventory_change() {
    let mut world = World::new(29, Preset::Void, Vec::new());
    world.agent.vel = [1.0, -2.0, 3.0];
    world.agent.fall_distance = 4.5;
    let mut trace = TraceState::default();
    let teleported = apply_intervention_with_state(
        &mut world,
        &InterventionSpec::TeleportAgent {
            position: WorldPos::from_cells([9.0, 10.0, 11.0]),
        },
        TraceLevel::Full,
        5,
        1,
        &mut trace,
    );
    let fields: Vec<_> = teleported
        .deltas
        .iter()
        .map(|delta| delta.field_or_cell)
        .collect();
    assert!(fields.contains(&"position"));
    assert!(fields.contains(&"velocity"));
    assert!(fields.contains(&"fall_distance"));

    let given = apply_intervention_with_state(
        &mut world,
        &InterventionSpec::GiveItem {
            item: STONE,
            count: 70,
        },
        TraceLevel::Full,
        5,
        2,
        &mut trace,
    );
    assert!(given.deltas.iter().any(|delta| {
        delta.subject == voxel_core::trace::SubjectRef::InventorySlot(0)
            && delta.before == TraceValue::ItemStack { item: 0, count: 0 }
            && delta.after
                == TraceValue::ItemStack {
                    item: STONE,
                    count: 64,
                }
    }));
    assert!(given.deltas.iter().any(|delta| {
        delta.subject == voxel_core::trace::SubjectRef::InventorySlot(1)
            && delta.after
                == TraceValue::ItemStack {
                    item: STONE,
                    count: 6,
                }
    }));
}

#[test]
fn an_untracked_scheduler_input_is_exogenous_not_periodic() {
    let mut world = World::new(37, Preset::Void, Vec::new());
    world.set_block(3, 8, 3, SAND);
    let mut trace = TraceState::default();
    let outcome = step_traced_with_state(&mut world, &idle(), TraceLevel::Events, 6, &mut trace);
    let scheduled = outcome
        .events
        .iter()
        .find(|event| event.kind == EventKind::BlockFallScheduled)
        .expect("direct setup schedules a fall");
    assert!(matches!(scheduled.root_cause, RootCause::Exogenous { .. }));
}

#[test]
fn mining_tnt_is_not_reported_as_an_explosion() {
    let mut world = World::new(41, Preset::Flat, Vec::new());
    for _ in 0..6 {
        voxel_core::step(&mut world, &idle());
    }
    let ax = world.agent.pos[0].floor() as i32;
    let az = world.agent.pos[2].floor() as i32;
    let target = CellCoord::new(ax + 1, 4, az);
    world.set_block(target.x, target.y, target.z, TNT);
    let action = Action {
        yaw: 18,
        pitch: 8,
        mine: true,
        ..Action::default()
    };
    let mut trace = TraceState::default();
    let mut mined = None;
    for _ in 0..80 {
        let outcome =
            step_traced_with_state(&mut world, &action, TraceLevel::Events, 7, &mut trace);
        if world.get_block(target.x, target.y, target.z) == AIR {
            mined = Some(outcome);
            break;
        }
    }
    let outcome = mined.expect("TNT was mined");
    assert!(outcome
        .events
        .iter()
        .any(|event| { event.kind == EventKind::BlockMined && event.location == Some(target) }));
    assert!(!outcome
        .events
        .iter()
        .any(|event| { event.kind == EventKind::Explosion && event.location == Some(target) }));
}

#[test]
fn direct_use_toggle_is_an_action_caused_block_change() {
    let mut world = World::new(42, Preset::Flat, Vec::new());
    for _ in 0..5 {
        voxel_core::step(&mut world, &idle());
    }
    let ax = world.agent.pos[0].floor() as i32;
    let az = world.agent.pos[2].floor() as i32;
    let target = CellCoord::new(ax + 1, 6, az);
    world.set_block(target.x, target.y, target.z, DOOR);
    let mut trace = TraceState::default();

    let outcome = step_traced_with_state(
        &mut world,
        &Action {
            yaw: 18,
            pitch: 4,
            use_: true,
            ..Action::default()
        },
        TraceLevel::Events,
        7,
        &mut trace,
    );
    let toggled = outcome
        .events
        .iter()
        .find(|event| event.location == Some(target) && event.kind == EventKind::CircuitChanged)
        .expect("the door toggle is traced");
    assert_eq!(toggled.phase, Phase::AgentAction);
    assert!(matches!(toggled.root_cause, RootCause::Action { .. }));
    assert_eq!(
        toggled.parent_ids,
        vec![
            outcome
                .events
                .iter()
                .find(|event| event.kind == EventKind::ActionApplied)
                .expect("action root")
                .id
        ]
    );
}

#[test]
fn explosion_damage_inherits_the_explosion_event() {
    let mut world = World::new(44, Preset::Void, Vec::new());
    world.agent.pos = [0.5, 5.0, 3.5];
    world.agent.vel = [0.0; 3];
    world.pending_booms.push((0, 5, 0, 0));
    let mut trace = TraceState::default();

    let outcome = step_traced_with_state(&mut world, &idle(), TraceLevel::Full, 8, &mut trace);
    let explosion = outcome
        .events
        .iter()
        .find(|event| event.kind == EventKind::Explosion)
        .expect("the due fuse explodes");
    let damage = outcome
        .events
        .iter()
        .find(|event| event.kind == EventKind::Damage)
        .expect("the nearby agent is damaged");
    assert_eq!(damage.phase, Phase::Tnt);
    assert_eq!(damage.parent_ids, vec![explosion.id]);
    assert_eq!(damage.root_cause, explosion.root_cause);
}

#[test]
fn fall_damage_inherits_the_landing_collision_event() {
    let mut world = World::new(45, Preset::Flat, Vec::new());
    world.agent.pos = [8.5, 15.0, 8.5];
    world.agent.vel = [0.0; 3];
    world.agent.on_ground = false;
    let mut trace = TraceState::default();

    let landing = (0..80)
        .find_map(|_| {
            let outcome =
                step_traced_with_state(&mut world, &idle(), TraceLevel::Full, 9, &mut trace);
            outcome
                .events
                .iter()
                .any(|event| event.kind == EventKind::Damage)
                .then_some(outcome)
        })
        .expect("the ten-block fall lands and causes damage");
    let collision = landing
        .events
        .iter()
        .find(|event| event.kind == EventKind::Collision)
        .expect("landing collision");
    let damage = landing
        .events
        .iter()
        .find(|event| event.kind == EventKind::Damage)
        .expect("fall damage");
    assert_eq!(damage.phase, Phase::EntityIntegration);
    assert_eq!(damage.parent_ids, vec![collision.id]);
    assert_eq!(damage.root_cause, collision.root_cause);
}

#[test]
fn fire_contact_damage_inherits_the_fire_mechanism_event() {
    let mut world = World::new(46, Preset::Flat, Vec::new());
    for _ in 0..5 {
        voxel_core::step(&mut world, &idle());
    }
    let feet = CellCoord::from_world_pos(WorldPos::from_cells(world.agent.pos));
    world.set_block(feet.x, feet.y, feet.z, FIRE);
    world.agent.fire_timer = 9;
    let mut trace = TraceState::default();

    let outcome = step_traced_with_state(&mut world, &idle(), TraceLevel::Full, 10, &mut trace);
    let fire = outcome
        .events
        .iter()
        .find(|event| event.location == Some(feet) && event.mechanism == "fire")
        .expect("fire mechanism event at the contact cell");
    let damage = outcome
        .events
        .iter()
        .find(|event| event.kind == EventKind::Damage)
        .expect("fire contact damage");
    assert_eq!(damage.phase, Phase::Fire);
    assert_eq!(damage.parent_ids, vec![fire.id]);
    assert_eq!(damage.root_cause, fire.root_cause);
}

#[test]
fn full_trace_does_not_use_a_circular_world_hash_delta() {
    let mut world = World::new(43, Preset::Void, Vec::new());
    let mut trace = TraceState::default();
    let outcome = step_traced_with_state(
        &mut world,
        &Action {
            yaw: 1,
            ..Action::default()
        },
        TraceLevel::Full,
        8,
        &mut trace,
    );
    assert!(outcome
        .deltas
        .iter()
        .all(|delta| delta.field_or_cell != "world_hash"));
}

#[test]
fn full_trace_preserves_dirty_queue_order_and_duplicates() {
    let mut world = World::new(47, Preset::Void, Vec::new());
    let first = CellCoord::new(8, 9, 10);
    let second = CellCoord::new(-3, 4, -5);
    world.dirty = vec![
        (first.x, first.y, first.z),
        (second.x, second.y, second.z),
        (first.x, first.y, first.z),
    ];
    let mut trace = TraceState::default();

    let outcome = step_traced_with_state(&mut world, &idle(), TraceLevel::Full, 9, &mut trace);
    let dirty = outcome
        .deltas
        .iter()
        .find(|delta| delta.field_or_cell == "dirty_queue")
        .expect("dirty queue change is explicit");
    assert_eq!(
        dirty.subject,
        voxel_core::trace::SubjectRef::Scheduler("dirty")
    );
    assert_eq!(
        dirty.before,
        TraceValue::CellSequence(vec![first, second, first])
    );
    assert_eq!(dirty.after, TraceValue::CellSequence(Vec::new()));
}

#[test]
fn set_cell_intervention_reports_the_exact_dirty_queue_change() {
    let mut world = World::new(53, Preset::Void, Vec::new());
    world.dirty.clear();
    let at = CellCoord::new(7, 8, 9);
    let mut trace = TraceState::default();

    let outcome = apply_intervention_with_state(
        &mut world,
        &InterventionSpec::SetCell { at, cell: STONE },
        TraceLevel::Full,
        10,
        1,
        &mut trace,
    );
    let dirty = outcome
        .deltas
        .iter()
        .find(|delta| delta.field_or_cell == "dirty_queue")
        .expect("intervention dirty enqueue is explicit");
    assert_eq!(dirty.before, TraceValue::CellSequence(Vec::new()));
    assert_eq!(dirty.after, TraceValue::CellSequence(vec![at]));
}

#[test]
fn set_cell_intervention_reports_every_active_set_membership_change() {
    let mut world = World::new(54, Preset::Void, Vec::new());
    world.dirty.clear();
    let at = CellCoord::new(7, 8, 9);
    let mut trace = TraceState::default();

    let outcome = apply_intervention_with_state(
        &mut world,
        &InterventionSpec::SetCell { at, cell: WATER },
        TraceLevel::Full,
        10,
        2,
        &mut trace,
    );
    let changed: HashSet<_> = outcome
        .deltas
        .iter()
        .filter_map(|delta| {
            if delta.field_or_cell != "active_fluid"
                || delta.before != TraceValue::Bool(false)
                || delta.after != TraceValue::Bool(true)
            {
                return None;
            }
            match delta.subject {
                voxel_core::trace::SubjectRef::Cell(cell) => Some(cell),
                _ => None,
            }
        })
        .collect();
    let expected = HashSet::from([
        at,
        at.offset(-1, 0, 0),
        at.offset(1, 0, 0),
        at.offset(0, -1, 0),
        at.offset(0, 1, 0),
        at.offset(0, 0, -1),
        at.offset(0, 0, 1),
    ]);
    assert_eq!(changed, expected);
}

#[test]
fn untraced_intervention_invalidates_lineage_instead_of_leaving_a_hidden_parent() {
    let mut world = World::new(59, Preset::Void, Vec::new());
    world.set_block(5, 5, 5, STONE);
    world.set_block(5, 6, 5, SAND);
    voxel_core::step(&mut world, &idle());
    let mut trace = TraceState::default();

    let tracked = apply_intervention_with_state(
        &mut world,
        &InterventionSpec::SetCell {
            at: CellCoord::new(20, 20, 20),
            cell: STONE,
        },
        TraceLevel::Events,
        12,
        1,
        &mut trace,
    )
    .event
    .expect("tracked event");
    let untraced = apply_intervention_with_state(
        &mut world,
        &InterventionSpec::SetCell {
            at: CellCoord::new(5, 5, 5),
            cell: AIR,
        },
        TraceLevel::Off,
        12,
        2,
        &mut trace,
    );
    assert!(untraced.event.is_none());

    let outcome = step_traced_with_state(&mut world, &idle(), TraceLevel::Events, 12, &mut trace);
    let scheduled = outcome
        .events
        .iter()
        .find(|event| event.kind == EventKind::BlockFallScheduled)
        .expect("support removal schedules sand");
    assert_ne!(scheduled.parent_ids, vec![tracked.id]);
    assert!(matches!(scheduled.root_cause, RootCause::Exogenous { .. }));
    assert!(scheduled
        .parent_ids
        .iter()
        .all(|parent| outcome.events.iter().any(|event| event.id == *parent)));
}
