use voxel_core::block::{AIR, CRAFTING_TABLE, DIRT, SAND, STONE};
use voxel_core::loose::{tick_falling, FallingBlock};
use voxel_core::physics::Physics;
use voxel_core::recipe::table_nearby;
use voxel_core::tnt::tick_tnt;
use voxel_core::{ClockConfig, Preset, Region, World};

#[test]
fn scale_two_crafting_range_and_world_height_are_measured_in_meters() {
    let mut within_range = World::new_scaled(1, Preset::Void, Vec::new(), 2.0);
    within_range.agent.pos = [20.5, 250.0, 20.5];
    within_range.set_block(28, 250, 20, CRAFTING_TABLE);
    assert!(table_nearby(&mut within_range));

    let mut outside_range = World::new_scaled(1, Preset::Void, Vec::new(), 2.0);
    outside_range.agent.pos = [20.5, 250.0, 20.5];
    outside_range.set_block(29, 250, 20, CRAFTING_TABLE);
    assert!(!table_nearby(&mut outside_range));
}

#[test]
fn scale_two_tnt_knockback_uses_the_scaled_agent_center() {
    let mut world = World::new_scaled(2, Preset::Void, Vec::new(), 2.0);
    world.agent.pos = [20.5, 20.0, 24.5];
    world.agent.vel = [0.0; 3];
    world.pending_booms.push((20, 20, 20, 0));

    tick_tnt(&mut world);

    let delta: [f64; 3] = [0.0, 1.3, 4.0];
    let distance = (delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    let distance_m = distance / 2.0;
    let attenuation = 1.0 - distance_m / 5.0;
    let impulse = 0.8 * 2.0 * attenuation;
    let expected_y = delta[1] / distance * impulse + 0.25 * 2.0 * attenuation;
    let expected_z = delta[2] / distance * impulse;
    assert!((world.agent.vel[1] - expected_y).abs() < 1e-12);
    assert!((world.agent.vel[2] - expected_z).abs() < 1e-12);
}

fn falling_impact_world(agent_y: f64, fall_distance: f64) -> World {
    let mut world = World::new_scaled(3, Preset::Void, Vec::new(), 2.0);
    world.set_block(10, 10, 10, STONE);
    world.agent.pos = [10.5, agent_y, 10.5];
    world.falling.push(FallingBlock {
        id: 1,
        block: SAND,
        pos: [10.5, 12.5, 10.5],
        vel: [0.0, -1.0, 0.0],
        fall_dist: fall_distance,
    });
    world
}

#[test]
fn scale_two_falling_blocks_use_agent_aabb_and_meter_damage() {
    let mut world = falling_impact_world(8.0, 4.0);

    tick_falling(&mut world);

    assert_eq!(world.agent.hp, 18, "a 2.26 m impact deals two half-hearts");
}

#[test]
fn scale_two_falling_blocks_do_not_damage_below_two_meters() {
    let mut world = falling_impact_world(10.0, 3.0);

    tick_falling(&mut world);

    assert_eq!(world.agent.hp, 20);
}

#[test]
fn scenario_regions_are_canonical_meter_volumes_at_world_construction() {
    let scenario = vec![(Region::new(-1, 2, -2, 0, 3, -1), STONE)];
    let mut scale_one = World::new_scaled(4, Preset::Void, scenario.clone(), 1.0);
    let mut scale_two = World::new_scaled(4, Preset::Void, scenario, 2.0);

    assert_eq!(scale_one.get_block(-1, 2, -2), STONE);
    assert_eq!(scale_one.get_block(0, 3, -1), STONE);
    assert_eq!(scale_one.get_block(1, 3, -1), AIR);

    for x in -2..=1 {
        for y in 4..=7 {
            for z in -4..=-1 {
                assert_eq!(scale_two.get_block(x, y, z), STONE, "{x},{y},{z}");
            }
        }
    }
    assert_eq!(scale_two.get_block(-3, 4, -4), AIR);
    assert_eq!(scale_two.get_block(2, 4, -4), AIR);
    assert_eq!(scale_two.get_block(-2, 8, -4), AIR);
    assert_eq!(scale_two.get_block(-2, 4, 0), AIR);
}

#[test]
fn physics_overrides_are_applied_once_at_construction() {
    let mut canonical = Physics::default();
    canonical.set("walk_speed", 0.4).unwrap();
    canonical.set("gravity", 0.1).unwrap();
    let world = World::new_scaled_with_clock(
        5,
        Preset::Void,
        Vec::new(),
        2.0,
        ClockConfig::new(1, 40).unwrap(),
    )
    .with_physics(canonical);

    assert_eq!(world.scale(), 2.0);
    assert_eq!(world.physics().scale(), 2.0);
    assert_eq!(world.physics().walk_speed, 0.4);
    // Vertical recurrence parameters remain canonical per 20 Hz step after
    // spatial scaling; the clock-aware integrator applies the fractional
    // half-step exactly at 40 Hz.
    assert_eq!(world.physics().gravity, 0.2);
}

#[test]
fn local_voxel_window_keeps_a_fixed_metric_extent_and_shape() {
    let scenario = vec![
        (Region::new(3, 20, -1, 3, 20, -1), STONE),
        (Region::new(-4, 18, 5, -4, 18, 5), DIRT),
    ];
    let mut scale_one = World::new_scaled(6, Preset::Void, scenario.clone(), 1.0);
    let mut scale_two = World::new_scaled(6, Preset::Void, scenario, 2.0);
    let eye_one = [0.5, 20.5, 0.5];
    let eye_two = [1.0, 41.0, 1.0];
    scale_one.agent.pos = [
        eye_one[0],
        eye_one[1] - scale_one.agent.eye_height,
        eye_one[2],
    ];
    scale_two.agent.pos = [
        eye_two[0],
        eye_two[1] - scale_two.agent.eye_height,
        eye_two[2],
    ];

    let one = scale_one.voxel_window();
    let two = scale_two.voxel_window();
    assert_eq!(one.len(), 21 * 11 * 21);
    assert_eq!(two, one);
}

#[test]
fn preset_spawn_pose_is_metric_equivalent_across_scales() {
    for preset in [Preset::Flat, Preset::Default] {
        let one = World::new_scaled(7, preset, Vec::new(), 1.0);
        let two = World::new_scaled(7, preset, Vec::new(), 2.0);
        assert_eq!(
            two.agent.pos.map(|component| component / 2.0),
            one.agent.pos,
            "{preset:?} spawn must describe the same metric pose",
        );
    }
}
