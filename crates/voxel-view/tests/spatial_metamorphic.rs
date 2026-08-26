use std::collections::BTreeSet;

use voxel_core::raycast::dda;
use voxel_core::spatial::{above, adjacent, connected_component, shortest_path, within};
use voxel_core::tick::raycast_target;
use voxel_core::{CellCoord, Preset, World, WorldPos, DIRT, PLANKS, STONE};
use voxel_view::lidar::{scan, LidarConfig};
use voxel_view::render_from;

fn stamp_wall(world: &mut World, x: i32, center_y: i32, center_z: i32, scale: i32) {
    for y in center_y - 2 * scale..=center_y + 2 * scale {
        for z in center_z - 3 * scale..=center_z + 3 * scale {
            world.set_block(x, y, z, STONE);
        }
    }
}

fn exact_lidar(max_range: f64) -> LidarConfig {
    LidarConfig {
        channels: 3,
        azimuth_steps: 8,
        min_elev_deg: -5.0,
        max_elev_deg: 5.0,
        max_range,
        noise_sigma: 0.0,
        dropout_p: 0.0,
        noise_seed: 0,
    }
}

fn assert_f32_slice_close(left: &[f32], right: &[f32], tolerance: f32) {
    assert_eq!(left.len(), right.len());
    for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
        assert!(
            (left - right).abs() <= tolerance,
            "index {index}: {left} != {right} within {tolerance}"
        );
    }
}

fn shift(cell: CellCoord, dx: i32, dz: i32) -> CellCoord {
    CellCoord::new(cell.x + dx, cell.y, cell.z + dz)
}

fn rotate_quarter_turn(cell: CellCoord) -> CellCoord {
    CellCoord::new(-cell.z, cell.y, cell.x)
}

fn window_cell(window: &[u16], dx: i32, dy: i32, dz: i32) -> u16 {
    let x = (dx + 10) as usize;
    let y = (dy + 4) as usize;
    let z = (dz + 10) as usize;
    window[(x * 11 + y) * 21 + z]
}

fn stamp_scaled_cell(world: &mut World, cell: CellCoord, scale: i32, block: u16) {
    for dx in 0..scale {
        for dy in 0..scale {
            for dz in 0..scale {
                world.set_block(
                    cell.x * scale + dx,
                    cell.y * scale + dy,
                    cell.z * scale + dz,
                    block,
                );
            }
        }
    }
}

#[test]
fn integer_translation_across_negative_chunks_preserves_queries_and_sensors() {
    const DX: i32 = -32;
    const DZ: i32 = -32;
    let mut positive = World::new(41, Preset::Void, Vec::new());
    let mut negative = World::new(41, Preset::Void, Vec::new());
    positive.agent.pos = [15.5, 20.0, 15.5];
    negative.agent.pos = [-16.5, 20.0, -16.5];

    stamp_wall(&mut positive, 20, 21, 15, 1);
    stamp_wall(&mut negative, -12, 21, -17, 1);
    for (cell, block) in [
        (CellCoord::new(5, 18, 7), DIRT),
        (CellCoord::new(14, 24, 20), PLANKS),
        (CellCoord::new(25, 17, 25), STONE),
    ] {
        positive.set_block(cell.x, cell.y, cell.z, block);
        let translated = shift(cell, DX, DZ);
        negative.set_block(translated.x, translated.y, translated.z, block);
    }

    assert_eq!(positive.voxel_window(), negative.voxel_window());

    let positive_origin = [15.5, 21.5, 15.5];
    let negative_origin = [-16.5, 21.5, -16.5];
    let positive_hit = dda(positive_origin, [1.0, 0.0, 0.0], 12.0, |x, y, z| {
        positive.peek_block(x, y, z)
    })
    .unwrap();
    let negative_hit = dda(negative_origin, [1.0, 0.0, 0.0], 12.0, |x, y, z| {
        negative.peek_block(x, y, z)
    })
    .unwrap();
    assert_eq!(negative_hit.x - positive_hit.x, DX);
    assert_eq!(negative_hit.z - positive_hit.z, DZ);
    assert_eq!(negative_hit.cell, positive_hit.cell);
    assert_eq!(negative_hit.face, positive_hit.face);
    assert_eq!(negative_hit.dist, positive_hit.dist);

    let positive_frame = render_from(&mut positive, positive_origin, 270.0, 0.0, 5, 3, 35.0);
    let negative_frame = render_from(&mut negative, negative_origin, 270.0, 0.0, 5, 3, 35.0);
    assert_eq!(positive_frame.rgb, negative_frame.rgb);
    assert_eq!(positive_frame.seg, negative_frame.seg);
    assert_eq!(positive_frame.normals, negative_frame.normals);
    assert_eq!(positive_frame.depth, negative_frame.depth);

    let config = exact_lidar(12.0);
    let positive_scan = scan(&mut positive, &config, positive_origin, 270.0, 0);
    let negative_scan = scan(&mut negative, &config, negative_origin, 270.0, 0);
    assert_eq!(positive_scan.range, negative_scan.range);
    assert_eq!(positive_scan.seg, negative_scan.seg);
    assert_eq!(positive_scan.intensity, negative_scan.intensity);

    let corridor = BTreeSet::from([
        CellCoord::new(14, 21, 15),
        CellCoord::new(15, 21, 15),
        CellCoord::new(16, 21, 15),
        CellCoord::new(16, 22, 15),
    ]);
    let translated: BTreeSet<_> = corridor
        .iter()
        .copied()
        .map(|cell| shift(cell, DX, DZ))
        .collect();
    let start = CellCoord::new(14, 21, 15);
    let goal = CellCoord::new(16, 22, 15);
    let path = shortest_path(start, goal, 32, |cell| corridor.contains(&cell))
        .unwrap()
        .unwrap();
    let translated_path = shortest_path(shift(start, DX, DZ), shift(goal, DX, DZ), 32, |cell| {
        translated.contains(&cell)
    })
    .unwrap()
    .unwrap();
    assert_eq!(
        path,
        translated_path
            .into_iter()
            .map(|cell| shift(cell, -DX, -DZ))
            .collect::<Vec<_>>()
    );

    let component = connected_component(start, 32, |cell| corridor.contains(&cell)).unwrap();
    let translated_component =
        connected_component(shift(start, DX, DZ), 32, |cell| translated.contains(&cell)).unwrap();
    assert_eq!(
        component,
        translated_component
            .into_iter()
            .map(|cell| shift(cell, -DX, -DZ))
            .collect::<Vec<_>>()
    );
    assert!(adjacent(start, CellCoord::new(15, 21, 15)));
    assert!(above(goal, CellCoord::new(16, 21, 15)));
    assert_eq!(
        within(
            WorldPos::new(15.5, 21.0, 15.5),
            WorldPos::new(16.5, 22.0, 15.5),
            2.0,
        ),
        within(
            WorldPos::new(-16.5, 21.0, -16.5),
            WorldPos::new(-15.5, 22.0, -16.5),
            2.0,
        )
    );
}

#[test]
fn quarter_turn_rotation_preserves_voxel_sensor_and_path_geometry() {
    let mut original = World::new(52, Preset::Void, Vec::new());
    let mut rotated = World::new(52, Preset::Void, Vec::new());
    original.agent.pos = [0.5, 18.5, 0.5];
    rotated.agent.pos = [0.5, 18.5, 0.5];
    for y in 18..=22 {
        for z in -3..=3 {
            let cell = CellCoord::new(5, y, z);
            original.set_block(cell.x, cell.y, cell.z, STONE);
            let turned = rotate_quarter_turn(cell);
            rotated.set_block(turned.x, turned.y, turned.z, STONE);
        }
    }
    for (cell, block) in [
        (CellCoord::new(-4, 17, 7), DIRT),
        (CellCoord::new(8, 24, -6), PLANKS),
    ] {
        original.set_block(cell.x, cell.y, cell.z, block);
        let turned = rotate_quarter_turn(cell);
        rotated.set_block(turned.x, turned.y, turned.z, block);
    }

    let original_window = original.voxel_window();
    let rotated_window = rotated.voxel_window();
    for dx in -10..=10 {
        for dy in -4..=6 {
            for dz in -10..=10 {
                assert_eq!(
                    window_cell(&original_window, dx, dy, dz),
                    window_cell(&rotated_window, -dz, dy, dx),
                    "window offset ({dx},{dy},{dz})"
                );
            }
        }
    }

    let origin = [0.5, 20.5, 0.5];
    let original_hit = dda(origin, [1.0, 0.0, 0.0], 12.0, |x, y, z| {
        original.peek_block(x, y, z)
    })
    .unwrap();
    let rotated_hit = dda(origin, [0.0, 0.0, 1.0], 12.0, |x, y, z| {
        rotated.peek_block(x, y, z)
    })
    .unwrap();
    assert_eq!(
        CellCoord::new(rotated_hit.x, rotated_hit.y, rotated_hit.z),
        rotate_quarter_turn(CellCoord::new(
            original_hit.x,
            original_hit.y,
            original_hit.z,
        ))
    );
    assert_eq!(rotated_hit.face, [0, 0, -1]);
    assert_eq!(rotated_hit.dist, original_hit.dist);

    let original_frame = render_from(&mut original, origin, 270.0, 0.0, 5, 3, 35.0);
    let rotated_frame = render_from(&mut rotated, origin, 0.0, 0.0, 5, 3, 35.0);
    assert_eq!(original_frame.seg, rotated_frame.seg);
    assert_f32_slice_close(&original_frame.depth, &rotated_frame.depth, 1e-5);
    for (original_normal, rotated_normal) in original_frame
        .normals
        .as_chunks::<3>()
        .0
        .iter()
        .zip(rotated_frame.normals.as_chunks::<3>().0.iter())
    {
        assert_eq!(
            rotated_normal,
            &[-original_normal[2], original_normal[1], original_normal[0]]
        );
    }

    let config = exact_lidar(12.0);
    let original_scan = scan(&mut original, &config, origin, 270.0, 0);
    let rotated_scan = scan(&mut rotated, &config, origin, 0.0, 0);
    assert_eq!(original_scan.seg, rotated_scan.seg);
    assert_f32_slice_close(&original_scan.range, &rotated_scan.range, 1e-5);
    assert_f32_slice_close(&original_scan.intensity, &rotated_scan.intensity, 1e-5);

    let corridor = BTreeSet::from([
        CellCoord::new(0, 20, 0),
        CellCoord::new(1, 20, 0),
        CellCoord::new(2, 20, 0),
        CellCoord::new(2, 20, 1),
    ]);
    let turned_corridor: BTreeSet<_> = corridor.iter().copied().map(rotate_quarter_turn).collect();
    let path = shortest_path(
        CellCoord::new(0, 20, 0),
        CellCoord::new(2, 20, 1),
        32,
        |cell| corridor.contains(&cell),
    )
    .unwrap()
    .unwrap();
    let turned_path = shortest_path(
        CellCoord::new(0, 20, 0),
        CellCoord::new(-1, 20, 2),
        32,
        |cell| turned_corridor.contains(&cell),
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        path.into_iter()
            .map(rotate_quarter_turn)
            .collect::<Vec<_>>(),
        turned_path
    );
}

#[test]
fn doubled_cell_density_preserves_metric_depth_range_and_reachability() {
    let mut scale_one = World::new_scaled(63, Preset::Void, Vec::new(), 1.0);
    let mut scale_two = World::new_scaled(63, Preset::Void, Vec::new(), 2.0);
    for y in 18..=22 {
        for z in -3..=3 {
            let cell = CellCoord::new(5, y, z);
            stamp_scaled_cell(&mut scale_one, cell, 1, STONE);
            stamp_scaled_cell(&mut scale_two, cell, 2, STONE);
        }
    }

    let origin_one = [0.75, 20.75, 0.75];
    let origin_two = [1.5, 41.5, 1.5];
    scale_one.agent.pos = [
        origin_one[0],
        origin_one[1] - scale_one.agent.eye_height,
        origin_one[2],
    ];
    scale_two.agent.pos = [
        origin_two[0],
        origin_two[1] - scale_two.agent.eye_height,
        origin_two[2],
    ];
    scale_one.agent.yaw = 270.0;
    scale_two.agent.yaw = 270.0;
    scale_one.agent.pitch = 0.0;
    scale_two.agent.pitch = 0.0;
    let hit_one = raycast_target(&mut scale_one).unwrap();
    let hit_two = raycast_target(&mut scale_two).unwrap();
    assert_eq!(hit_one.cell, hit_two.cell);
    assert_eq!(hit_one.face, hit_two.face);
    assert!((hit_one.dist - hit_two.dist).abs() < 1e-12);

    let frame_one = render_from(&mut scale_one, origin_one, 270.0, 0.0, 5, 3, 35.0);
    let frame_two = render_from(&mut scale_two, origin_two, 270.0, 0.0, 5, 3, 35.0);
    assert_eq!(frame_one.rgb, frame_two.rgb);
    assert_eq!(frame_one.seg, frame_two.seg);
    assert_eq!(frame_one.normals, frame_two.normals);
    assert_f32_slice_close(&frame_one.depth, &frame_two.depth, 1e-5);

    let scan_one = scan(&mut scale_one, &exact_lidar(12.0), origin_one, 270.0, 0);
    let scan_two = scan(&mut scale_two, &exact_lidar(12.0), origin_two, 270.0, 0);
    assert_eq!(scan_one.seg, scan_two.seg);
    assert_f32_slice_close(&scan_one.range, &scan_two.range, 1e-5);
    assert_f32_slice_close(&scan_one.intensity, &scan_two.intensity, 1e-5);

    let corridor_one: BTreeSet<_> = (0..=4).map(|x| CellCoord::new(x, 20, 0)).collect();
    let corridor_two: BTreeSet<_> = corridor_one
        .iter()
        .flat_map(|cell| {
            (0..2).flat_map(move |dx| {
                (0..2).flat_map(move |dy| {
                    (0..2).map(move |dz| {
                        CellCoord::new(cell.x * 2 + dx, cell.y * 2 + dy, cell.z * 2 + dz)
                    })
                })
            })
        })
        .collect();
    let path_one = shortest_path(
        CellCoord::new(0, 20, 0),
        CellCoord::new(4, 20, 0),
        64,
        |cell| corridor_one.contains(&cell),
    )
    .unwrap()
    .unwrap();
    let path_two = shortest_path(
        CellCoord::new(0, 40, 0),
        CellCoord::new(8, 40, 0),
        256,
        |cell| corridor_two.contains(&cell),
    )
    .unwrap()
    .unwrap();
    assert_eq!(path_one.len() - 1, (path_two.len() - 1) / 2);

    let scale = voxel_core::SpatialScale::new(2, 1).unwrap();
    assert_eq!(
        scale.world_to_metric(WorldPos::from_cells(origin_two)),
        voxel_core::MetricPos::new(origin_one[0], origin_one[1], origin_one[2])
    );
}

#[test]
fn renderer_and_lidar_keep_their_physical_horizon_at_scale_two() {
    let mut scale_one = World::new_scaled(71, Preset::Void, Vec::new(), 1.0);
    let mut scale_two = World::new_scaled(71, Preset::Void, Vec::new(), 2.0);
    for y in 18..=22 {
        for z in -2..=2 {
            stamp_scaled_cell(&mut scale_one, CellCoord::new(70, y, z), 1, STONE);
            stamp_scaled_cell(&mut scale_two, CellCoord::new(70, y, z), 2, STONE);
        }
    }

    let origin_one = [0.5, 20.5, 0.5];
    let origin_two = [1.0, 41.0, 1.0];
    let frame_one = render_from(&mut scale_one, origin_one, 270.0, 0.0, 1, 1, 35.0);
    let frame_two = render_from(&mut scale_two, origin_two, 270.0, 0.0, 1, 1, 35.0);
    assert_eq!(frame_one.seg, vec![STONE]);
    assert_eq!(frame_two.seg, frame_one.seg);
    assert_f32_slice_close(&frame_one.depth, &frame_two.depth, 1e-5);

    let lidar = exact_lidar(80.0);
    let scan_one = scan(&mut scale_one, &lidar, origin_one, 270.0, 0);
    let scan_two = scan(&mut scale_two, &lidar, origin_two, 270.0, 0);
    assert_eq!(scan_one.seg[lidar.azimuth_steps], STONE);
    assert_eq!(scan_two.seg, scan_one.seg);
    assert_f32_slice_close(&scan_one.range, &scan_two.range, 1e-5);
    assert_f32_slice_close(&scan_one.intensity, &scan_two.intensity, 1e-5);
}
