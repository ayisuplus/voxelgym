use voxel_core::physics::Physics;
use voxel_core::tick::{step, Action};
use voxel_core::worldgen::Preset;
use voxel_core::{ClockConfig, SimClock, World, AIR, LEAVES};

#[test]
fn clock_config_is_a_positive_reduced_rational() {
    let config = ClockConfig::new(2, 40).unwrap();

    assert_eq!(config.numerator(), 1);
    assert_eq!(config.denominator(), 20);
    assert_eq!(ClockConfig::default(), config);
    assert_eq!(
        ClockConfig::new(0, 20).unwrap_err(),
        "tick duration must be positive"
    );
    assert_eq!(
        ClockConfig::new(1, 0).unwrap_err(),
        "tick duration denominator must be non-zero"
    );
}

#[test]
fn sim_clock_reports_step_boundaries_and_sensor_age() {
    let clock = SimClock::at_tick(ClockConfig::new(1, 20).unwrap(), 7);

    assert_eq!(clock.tick(), 7);
    assert_eq!(clock.elapsed_fraction(), (7, 20));
    assert_eq!(clock.remaining_ticks(10), 3);
    assert_eq!(clock.sample_time_fraction(5), (1, 4));
    assert_eq!(clock.data_age_ticks(5), Some(2));
    assert_eq!(clock.data_age_ticks(8), None);
}

#[test]
fn world_snapshot_v8_roundtrips_clock_and_hashes_it() {
    let fast = ClockConfig::new(1, 40).unwrap();
    let world = World::new_with_clock(11, Preset::Void, Vec::new(), fast);
    let snapshot = world.snapshot();

    assert_eq!(u32::from_le_bytes(snapshot[4..8].try_into().unwrap()), 8);
    assert_eq!(world.clock_config(), fast);
    assert_eq!(world.sim_clock().tick(), 0);

    let restored = World::restore(&snapshot).unwrap();
    assert_eq!(restored.clock_config(), fast);
    assert_eq!(restored.snapshot(), snapshot);
    assert_eq!(restored.hash(), world.hash());

    let default_clock = World::new(11, Preset::Void, Vec::new());
    assert_ne!(default_clock.hash(), world.hash());
}

#[test]
fn legacy_hash_omits_v8_only_state_extensions() {
    let world = World::new(29, Preset::Flat, Vec::new());
    let restored = World::restore(&world.snapshot()).unwrap();

    assert_eq!(world.legacy_hash_v7(), restored.legacy_hash_v7());
    assert_ne!(world.hash(), world.legacy_hash_v7());
}

#[test]
fn version_seven_snapshot_defaults_to_twenty_hertz() {
    let world = World::new(17, Preset::Void, Vec::new());
    let mut legacy = world.snapshot();
    legacy.truncate(legacy.len() - 8); // empty v8 semantic + dirty trailer
    legacy[4..8].copy_from_slice(&7u32.to_le_bytes());
    legacy.drain(16..32);

    let restored = World::restore(&legacy).unwrap();

    assert_eq!(restored.clock_config(), ClockConfig::default());
    assert_eq!(restored.tick, world.tick);
    assert!(restored.dirty.is_empty());
}

#[test]
fn physics_scale_can_be_read_but_not_mutated() {
    let mut physics = Physics::default().spatially_scaled(2.0);

    assert_eq!(physics.scale(), 2.0);
    assert_eq!(physics.get("scale"), Some(2.0));
    assert_eq!(
        physics.set("scale", 3.0).unwrap_err(),
        "physics field 'scale' is immutable after world construction"
    );
    assert_eq!(physics.get("scale"), Some(2.0));
}

#[test]
fn clock_quantizes_legacy_tick_durations_without_running_early() {
    let slow = ClockConfig::new(1, 10).unwrap();
    let fast = ClockConfig::new(1, 40).unwrap();

    assert_eq!(slow.ticks_for_default_ticks(20), 10);
    assert_eq!(fast.ticks_for_default_ticks(20), 40);
    assert_eq!(fast.ticks_for_default_ticks(1), 2);
    assert_eq!(ClockConfig::default().ticks_for_default_ticks(7), 7);
    assert_eq!(
        ClockConfig::new(1, u64::MAX)
            .unwrap()
            .ticks_for_default_ticks(u64::MAX),
        u64::MAX
    );
}

#[test]
fn construction_scales_per_tick_physics_to_the_clock() {
    let slow = World::new_with_clock(
        1,
        Preset::Void,
        Vec::new(),
        ClockConfig::new(1, 10).unwrap(),
    );
    let fast = World::new_with_clock(
        1,
        Preset::Void,
        Vec::new(),
        ClockConfig::new(1, 40).unwrap(),
    );

    assert_eq!(
        slow.physics().walk_speed,
        2.0 * Physics::default().walk_speed
    );
    assert_eq!(
        fast.physics().walk_speed,
        0.5 * Physics::default().walk_speed
    );
    assert_eq!(slow.physics().gravity, Physics::default().gravity);
    assert_eq!(fast.physics().gravity, Physics::default().gravity);
    assert_eq!(slow.physics().jump_vy, Physics::default().jump_vy);
    assert_eq!(fast.physics().jump_vy, Physics::default().jump_vy);
    assert_eq!(slow.physics().water_period, 3);
    assert_eq!(fast.physics().water_period, 10);
}

#[test]
fn free_fall_and_jump_compose_exactly_across_tick_rates() {
    fn advance(world: &mut World, ticks: usize, action: &Action) {
        for _ in 0..ticks {
            step(world, action);
        }
    }

    let fast_clock = ClockConfig::new(1, 40).unwrap();
    let mut fall20 = World::new(71, Preset::Void, Vec::new());
    let mut fall40 = World::new_with_clock(71, Preset::Void, Vec::new(), fast_clock);
    for world in [&mut fall20, &mut fall40] {
        world.agent.pos = [0.5, 100.0, 0.5];
        world.agent.vel = [0.0; 3];
        world.agent.on_ground = false;
    }
    advance(&mut fall20, 20, &Action::default());
    advance(&mut fall40, 40, &Action::default());
    assert!((fall20.agent.pos[1] - fall40.agent.pos[1]).abs() < 1e-10);
    assert!((fall20.agent.vel[1] - fall40.agent.vel[1]).abs() < 1e-10);

    let mut jump20 = World::new(72, Preset::Flat, Vec::new());
    let mut jump40 = World::new_with_clock(72, Preset::Flat, Vec::new(), fast_clock);
    jump20.agent.on_ground = true;
    jump40.agent.on_ground = true;
    let jump = Action {
        jump: true,
        ..Action::default()
    };
    step(&mut jump20, &jump);
    step(&mut jump40, &jump);
    // Compare at 0.25 seconds, before the discrete landing collision. Landing
    // itself is quantized to the first boundary whose sweep reaches ground.
    advance(&mut jump20, 4, &Action::default());
    advance(&mut jump40, 9, &Action::default());
    assert!(
        (jump20.agent.pos[1] - jump40.agent.pos[1]).abs() < 1e-6,
        "jump position: 20Hz={} 40Hz={}",
        jump20.agent.pos[1],
        jump40.agent.pos[1]
    );
    assert!(
        (jump20.agent.vel[1] - jump40.agent.vel[1]).abs() < 1e-6,
        "jump velocity: 20Hz={} 40Hz={}",
        jump20.agent.vel[1],
        jump40.agent.vel[1]
    );
}

#[test]
fn locomotion_tracks_physical_time_across_tick_rates() {
    fn advance(world: &mut World, ticks: usize, action: &Action) {
        for _ in 0..ticks {
            step(world, action);
        }
    }

    let mut hz20 = World::new(7, Preset::Flat, Vec::new());
    let mut hz40 = World::new_with_clock(
        7,
        Preset::Flat,
        Vec::new(),
        ClockConfig::new(1, 40).unwrap(),
    );
    let idle = Action::default();
    advance(&mut hz20, 20, &idle);
    advance(&mut hz40, 40, &idle);

    let forward = Action {
        mv: 1,
        ..Action::default()
    };
    advance(&mut hz20, 40, &forward);
    advance(&mut hz40, 80, &forward);

    assert!((hz20.agent.pos[2] - hz40.agent.pos[2]).abs() < 0.12);
    assert!((hz20.sim_clock().elapsed_seconds() - 3.0).abs() < f64::EPSILON);
    assert!((hz40.sim_clock().elapsed_seconds() - 3.0).abs() < f64::EPSILON);
}

#[test]
fn instant_mining_keeps_its_one_default_tick_duration_across_tick_rates() {
    let mut hz20 = World::new(73, Preset::Flat, Vec::new());
    let mut hz40 = World::new_with_clock(
        73,
        Preset::Flat,
        Vec::new(),
        ClockConfig::new(1, 40).unwrap(),
    );
    let x = hz20.agent.pos[0].floor() as i32 + 1;
    let y = (hz20.agent.pos[1] + hz20.agent.eye_height).floor() as i32;
    let z = hz20.agent.pos[2].floor() as i32;
    hz20.set_block(x, y, z, LEAVES);
    hz40.set_block(x, y, z, LEAVES);
    let mine = Action {
        yaw: 18,
        pitch: 4,
        mine: true,
        ..Action::default()
    };

    step(&mut hz20, &mine);
    step(&mut hz40, &mine);
    assert_eq!(hz20.get_block(x, y, z), AIR);
    assert_eq!(hz40.get_block(x, y, z), LEAVES);
    step(&mut hz40, &mine);
    assert_eq!(hz40.get_block(x, y, z), AIR);
    assert_eq!(hz20.sim_clock().elapsed_fraction(), (1, 20));
    assert_eq!(hz40.sim_clock().elapsed_fraction(), (1, 20));
}
