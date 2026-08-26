use voxel_core::block::STONE;
use voxel_core::spatial::CellCoord;
use voxel_core::trace::{
    apply_intervention, apply_intervention_with_state, step_traced, InterventionSpec, TraceLevel,
    TraceState, TraceValue,
};
use voxel_core::worldgen::Preset;
use voxel_core::{step, Action, Stack, World};

#[test]
fn tracing_never_changes_world_semantics() {
    let mut plain = World::new(41, Preset::Flat, Vec::new());
    let mut traced = World::new(41, Preset::Flat, Vec::new());
    let action = Action {
        mv: 1,
        yaw: 0,
        pitch: 4,
        ..Action::default()
    };

    step(&mut plain, &action);
    let outcome = step_traced(&mut traced, &action, TraceLevel::Full, 7);

    assert_eq!(plain.hash(), traced.hash());
    assert_eq!(outcome.clock_before.tick(), 0);
    assert_eq!(outcome.clock_after.tick(), 1);
    assert_eq!(
        outcome.before_hash,
        Some(World::new(41, Preset::Flat, Vec::new()).hash())
    );
    assert_eq!(outcome.after_hash, Some(traced.hash()));
}

#[test]
fn full_trace_has_a_rooted_acyclic_event_graph_and_state_deltas() {
    let mut world = World::new(9, Preset::Flat, Vec::new());
    let outcome = step_traced(&mut world, &Action::default(), TraceLevel::Full, 3);

    assert!(!outcome.events.is_empty());
    assert!(!outcome.deltas.is_empty());
    assert_eq!(outcome.events[0].parent_ids, Vec::<u64>::new());
    let ids: std::collections::HashSet<_> = outcome.events.iter().map(|event| event.id).collect();
    assert_eq!(ids.len(), outcome.events.len());
    for event in &outcome.events {
        for parent in &event.parent_ids {
            assert!(ids.contains(parent));
            assert!(*parent != event.id);
        }
    }
    assert!(outcome
        .deltas
        .iter()
        .all(|delta| ids.contains(&delta.event_id)));
}

#[test]
fn event_only_trace_omits_exact_deltas_and_hashes() {
    let mut world = World::new(5, Preset::Flat, Vec::new());
    let outcome = step_traced(&mut world, &Action::default(), TraceLevel::Events, 0);

    assert!(!outcome.events.is_empty());
    assert!(outcome.deltas.is_empty());
    assert_eq!(outcome.before_hash, None);
    assert_eq!(outcome.after_hash, None);
}

#[test]
fn intervention_delta_reports_the_actual_post_mutation_inventory_count() {
    let mut world = World::new(5, Preset::Void, Vec::new());
    let outcome = apply_intervention(
        &mut world,
        &InterventionSpec::GiveItem {
            item: STONE,
            count: u16::MAX,
        },
        TraceLevel::Full,
        0,
        0,
    )
    .expect("valid intervention");

    assert_eq!(world.agent.inventory.count(STONE), 36 * 64);
    let total = outcome
        .deltas
        .iter()
        .find(|delta| delta.field_or_cell == "inventory_count")
        .expect("aggregate count remains available to existing consumers");
    assert_eq!(total.after, TraceValue::U64(36 * 64));
    assert_eq!(
        outcome
            .deltas
            .iter()
            .filter(|delta| delta.field_or_cell == "stack")
            .count(),
        36,
        "full tracing also records every affected slot"
    );
}

#[test]
fn hotbar_swap_intervention_reports_exact_inventory_and_selection_changes() {
    let mut world = World::new(6, Preset::Void, Vec::new());
    world.agent.selected = 2;
    world.agent.inventory.slots[11] = Stack {
        item: STONE,
        count: 3,
    };
    let before_hash = world.hash();

    let outcome = apply_intervention(
        &mut world,
        &InterventionSpec::SwapToHotbar { item: STONE },
        TraceLevel::Full,
        5,
        7,
    )
    .expect("valid intervention");

    assert_ne!(world.hash(), before_hash);
    assert_eq!(world.agent.selected, 2);
    assert_eq!(world.agent.inventory.slots[2].item, STONE);
    assert_eq!(world.agent.inventory.slots[2].count, 3);
    assert_eq!(world.last_swap, None);
    assert_eq!(
        outcome
            .deltas
            .iter()
            .filter(|delta| delta.field_or_cell == "stack")
            .count(),
        2
    );
}

#[test]
fn rejected_set_cell_intervention_emits_no_false_delta() {
    let mut world = World::new(5, Preset::Void, Vec::new());
    let out_of_bounds = CellCoord::new(0, world.height(), 0);
    let before_hash = world.hash();

    let outcome = apply_intervention(
        &mut world,
        &InterventionSpec::SetCell {
            at: out_of_bounds,
            cell: STONE,
        },
        TraceLevel::Full,
        0,
        0,
    )
    .expect("valid intervention");

    assert_eq!(world.hash(), before_hash);
    assert!(outcome.deltas.is_empty());
}

#[test]
fn invalid_set_cell_intervention_is_atomic_for_world_and_trace_state() {
    let mut world = World::new(7, Preset::Void, Vec::new());
    let mut trace_state = TraceState::default();
    let before_world = world.snapshot();
    let before_hash = world.hash();
    let before_trace = trace_state.clone();

    let error = apply_intervention_with_state(
        &mut world,
        &InterventionSpec::SetCell {
            at: CellCoord::new(100, 10, 100),
            cell: u16::MAX,
        },
        TraceLevel::Full,
        9,
        3,
        &mut trace_state,
    )
    .unwrap_err();

    assert_eq!(error, "unknown block id 4095 in cell 65535");
    assert_eq!(world.snapshot(), before_world);
    assert_eq!(world.hash(), before_hash);
    assert_eq!(trace_state, before_trace);
}
