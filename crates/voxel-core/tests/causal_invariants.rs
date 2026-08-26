use std::collections::{HashMap, HashSet};
#[cfg(all(not(debug_assertions), not(coverage_nightly)))]
use std::hint::black_box;
#[cfg(all(not(debug_assertions), not(coverage_nightly)))]
use std::time::{Duration, Instant};

use voxel_core::trace::{
    apply_intervention, compare_branches, step_traced, EventKind, InterventionSpec, RootCause,
    TraceLevel, WorldEvent,
};
#[cfg(all(not(debug_assertions), not(coverage_nightly)))]
use voxel_core::trace::{step_traced_with_state, TraceState};
use voxel_core::worldgen::Preset;
use voxel_core::{step, Action, CellCoord, World, STONE, WATER};

fn idle() -> Action {
    Action {
        pitch: 4,
        ..Action::default()
    }
}

#[test]
fn all_trace_levels_preserve_each_transition_and_final_hash() {
    let mut source = World::new(83, Preset::Flat, Vec::new());
    source.set_block(8, 8, 8, WATER);
    let snapshot = source.snapshot();
    let mut off = World::restore(&snapshot).unwrap();
    let mut events = World::restore(&snapshot).unwrap();
    let mut full = World::restore(&snapshot).unwrap();

    for tick in 0..48u8 {
        let action = Action {
            mv: tick % 5,
            jump: tick % 7 == 0,
            yaw: tick.wrapping_mul(5) % 24,
            pitch: 4,
            ..Action::default()
        };
        let off_outcome = step_traced(&mut off, &action, TraceLevel::Off, 12);
        let event_outcome = step_traced(&mut events, &action, TraceLevel::Events, 12);
        let full_outcome = step_traced(&mut full, &action, TraceLevel::Full, 12);

        assert!(off_outcome.events.is_empty());
        assert!(off_outcome.deltas.is_empty());
        assert!(event_outcome.deltas.is_empty());
        assert_eq!(event_outcome.before_hash, None);
        assert_eq!(event_outcome.after_hash, None);
        assert_eq!(full_outcome.after_hash, Some(full.hash()));
        assert_eq!(
            off.hash(),
            events.hash(),
            "hash diverged at transition {tick}"
        );
        assert_eq!(
            off.hash(),
            full.hash(),
            "hash diverged at transition {tick}"
        );
        assert_eq!(off.snapshot(), events.snapshot());
        assert_eq!(off.snapshot(), full.snapshot());
    }
}

#[test]
fn causal_events_form_a_rooted_dag() {
    let mut world = World::new(91, Preset::Flat, Vec::new());
    world.set_block(8, 8, 8, WATER);
    let intervention = apply_intervention(
        &mut world,
        &InterventionSpec::SetCell {
            at: CellCoord::new(2, 8, 2),
            cell: STONE,
        },
        TraceLevel::Full,
        44,
        1,
    )
    .expect("valid intervention");
    let mut recorded = intervention.event.into_iter().collect::<Vec<_>>();
    for _ in 0..3 {
        recorded.extend(step_traced(&mut world, &idle(), TraceLevel::Full, 44).events);
    }

    let graph: HashMap<_, _> = recorded.iter().map(|event| (event.id, event)).collect();
    assert_eq!(graph.len(), recorded.len(), "event IDs must be unique");
    let mut marks = HashMap::new();
    for event in &recorded {
        visit_acyclic(event.id, &graph, &mut marks);
        if !event.parent_ids.is_empty() {
            assert!(reaches_declared_root(event.id, &graph, &mut HashSet::new()));
        }
    }

    assert!(recorded.iter().any(|event| {
        event.parent_ids.is_empty() && matches!(event.root_cause, RootCause::Action { .. })
    }));
    assert!(recorded.iter().any(|event| {
        event.parent_ids.is_empty() && matches!(event.root_cause, RootCause::Intervention { .. })
    }));
    assert!(recorded.iter().any(|event| {
        event.parent_ids.is_empty() && matches!(event.root_cause, RootCause::Periodic { .. })
    }));
}

fn visit_acyclic(id: u64, graph: &HashMap<u64, &WorldEvent>, marks: &mut HashMap<u64, u8>) {
    match marks.get(&id) {
        Some(2) => return,
        Some(1) => panic!("causal event graph contains a cycle at {id}"),
        _ => {}
    }
    marks.insert(id, 1);
    let event = graph[&id];
    for parent_id in &event.parent_ids {
        let parent = graph
            .get(parent_id)
            .unwrap_or_else(|| panic!("event {id} has missing parent {parent_id}"));
        assert!(
            parent.tick <= event.tick,
            "event {id} depends on a future event"
        );
        visit_acyclic(*parent_id, graph, marks);
    }
    marks.insert(id, 2);
}

fn reaches_declared_root(
    id: u64,
    graph: &HashMap<u64, &WorldEvent>,
    seen: &mut HashSet<u64>,
) -> bool {
    if !seen.insert(id) {
        return false;
    }
    let event = graph[&id];
    if event.parent_ids.is_empty() {
        return matches!(
            (event.kind, &event.root_cause),
            (EventKind::ActionApplied, RootCause::Action { .. })
                | (
                    EventKind::InterventionApplied,
                    RootCause::Intervention { .. }
                )
                | (EventKind::StateChanged, RootCause::Periodic { .. })
                | (EventKind::StateChanged, RootCause::Exogenous { .. })
        );
    }
    event
        .parent_ids
        .iter()
        .any(|parent| reaches_declared_root(*parent, graph, seen))
}

#[test]
fn counterfactual_branches_diverge_only_after_the_intervention() {
    let mut source = World::new(17, Preset::Flat, Vec::new());
    for _ in 0..5 {
        step(&mut source, &idle());
    }
    let common_snapshot = source.snapshot();
    let common_hash = source.hash();
    let mut factual = World::restore(&common_snapshot).unwrap();
    let mut counterfactual = World::restore(&common_snapshot).unwrap();
    assert_eq!(factual.snapshot(), counterfactual.snapshot());
    assert_eq!(factual.hash(), common_hash);
    assert_eq!(counterfactual.hash(), common_hash);

    let treatment = InterventionSpec::SetCell {
        at: CellCoord::new(20, 20, 20),
        cell: STONE,
    };
    let applied = apply_intervention(&mut counterfactual, &treatment, TraceLevel::Full, 3, 9)
        .expect("valid intervention");
    assert_eq!(applied.before_hash, Some(common_hash));
    assert_eq!(
        factual.hash(),
        common_hash,
        "control branch changed before rollout"
    );
    assert_ne!(counterfactual.hash(), common_hash);

    let actions = vec![idle(); 16];
    for action in &actions {
        step(&mut factual, action);
        step(&mut counterfactual, action);
    }
    assert_ne!(factual.hash(), counterfactual.hash());

    let comparison = compare_branches(&source, &treatment, &actions, &actions).unwrap();
    assert_eq!(comparison.common_before_hash, common_hash);
    assert_eq!(comparison.control_after_hash, factual.hash());
    assert_eq!(comparison.treatment_after_hash, counterfactual.hash());
    assert!(comparison.diverged);

    let mut confounded_actions = actions.clone();
    confounded_actions[0].mv = 1;
    let error = compare_branches(&source, &treatment, &actions, &confounded_actions).unwrap_err();
    assert!(error.contains("identical action sequences"));
}

/// This is deliberately a release-only performance contract. Instrumented
/// coverage builds and debug builds do not measure the production hot path.
#[cfg(all(not(debug_assertions), not(coverage_nightly)))]
#[test]
fn trace_off_stateful_path_stays_within_five_percent_of_plain_step() {
    const WARMUP_STEPS: usize = 100_000;
    const TRIALS: usize = 15;
    const STEPS: usize = 500_000;
    let mut base = World::new(101, Preset::Flat, Vec::new());
    for _ in 0..64 {
        step(&mut base, &idle());
    }
    let snapshot = base.snapshot();

    // Warm both paths before collecting samples so one-time instruction-cache,
    // allocator, and CPU-frequency effects cannot dominate the comparison.
    black_box(time_plain(&snapshot, WARMUP_STEPS));
    black_box(time_trace_off(&snapshot, WARMUP_STEPS));

    let mut plain_times = Vec::with_capacity(TRIALS);
    let mut off_times = Vec::with_capacity(TRIALS);

    for trial in 0..TRIALS {
        if trial % 2 == 0 {
            plain_times.push(time_plain(&snapshot, STEPS));
            off_times.push(time_trace_off(&snapshot, STEPS));
        } else {
            off_times.push(time_trace_off(&snapshot, STEPS));
            plain_times.push(time_plain(&snapshot, STEPS));
        }
    }

    let plain = median(&mut plain_times);
    let trace_off = median(&mut off_times);
    let plain_nanos = plain.as_nanos();
    let trace_nanos = trace_off.as_nanos();
    let ratio = trace_nanos as f64 / plain_nanos as f64;
    eprintln!(
        "trace-off throughput: plain median={plain:?}, trace median={trace_off:?}, ratio={ratio:.4}"
    );
    assert!(
        trace_nanos.saturating_mul(100) <= plain_nanos.saturating_mul(105),
        "TraceLevel::Off regressed the hot path by more than 5%: plain={plain:?}, trace_off={trace_off:?}, ratio={ratio:.4}"
    );
}

#[cfg(all(not(debug_assertions), not(coverage_nightly)))]
fn time_plain(snapshot: &[u8], steps: usize) -> Duration {
    let mut world = World::restore(snapshot).unwrap();
    let action = idle();
    let started = Instant::now();
    for _ in 0..steps {
        step(black_box(&mut world), black_box(&action));
    }
    let elapsed = started.elapsed();
    black_box((world.tick, world.agent.pos));
    elapsed
}

#[cfg(all(not(debug_assertions), not(coverage_nightly)))]
fn time_trace_off(snapshot: &[u8], steps: usize) -> Duration {
    let mut world = World::restore(snapshot).unwrap();
    let action = idle();
    let mut trace_state = TraceState::default();
    let started = Instant::now();
    for _ in 0..steps {
        black_box(step_traced_with_state(
            black_box(&mut world),
            black_box(&action),
            TraceLevel::Off,
            0,
            &mut trace_state,
        ));
    }
    let elapsed = started.elapsed();
    black_box((world.tick, world.agent.pos, trace_state));
    elapsed
}

#[cfg(all(not(debug_assertions), not(coverage_nightly)))]
fn median(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}
