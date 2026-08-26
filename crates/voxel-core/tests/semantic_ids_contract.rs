use std::collections::BTreeSet;

use voxel_core::trace::{apply_intervention, InterventionSpec, TraceLevel};
use voxel_core::{
    step, Action, CellCoord, ClockConfig, Preset, Region, RegionId, SemanticRegionSpec,
    StructureId, World, DIRT, SAND, STONE,
};

fn id_pairs(world: &World) -> Vec<(u64, u64)> {
    let mut ids: Vec<_> = world
        .semantic_regions()
        .iter()
        .map(|spec| (spec.region_id.get(), spec.structure_id.get()))
        .collect();
    ids.sort_unstable();
    ids
}

#[test]
fn legacy_scenario_content_ids_ignore_order_and_cell_density() {
    let first = (Region::new(-2, 4, 7, -1, 5, 8), STONE);
    let second = (Region::new(3, 9, -4, 5, 9, -2), DIRT);
    let scale_one = World::new(1, Preset::Void, vec![first, second]);
    let reordered = World::new(1, Preset::Void, vec![second, first]);
    let scale_two = World::new_scaled(1, Preset::Void, vec![first, second], 2.0);

    assert_eq!(id_pairs(&scale_one), id_pairs(&reordered));
    assert_eq!(id_pairs(&scale_one), id_pairs(&scale_two));
    assert!(id_pairs(&scale_one)
        .iter()
        .all(|&(region, structure)| region != 0 && structure != 0));
    assert_eq!(
        scale_two.semantic_regions()[0].region,
        scale_one.semantic_regions()[0].region.scaled_to_cells(2.0)
    );
}

#[test]
fn explicit_regions_can_share_a_structure_and_roundtrip_v8() {
    let structure = StructureId::new(700);
    let specs = vec![
        SemanticRegionSpec::new(
            RegionId::new(101),
            structure,
            Region::new(0, 5, 0, 1, 6, 1),
            STONE,
        ),
        SemanticRegionSpec::new(
            RegionId::new(102),
            structure,
            Region::new(2, 5, 0, 3, 7, 1),
            DIRT,
        ),
    ];
    let world = World::new_scaled_with_clock_and_semantic_regions(
        2,
        Preset::Void,
        specs,
        2.0,
        ClockConfig::default(),
    )
    .unwrap();

    assert_eq!(world.semantic_regions().len(), 2);
    assert!(world
        .semantic_regions()
        .iter()
        .all(|spec| spec.structure_id == structure));
    assert_eq!(
        world.semantic_regions()[0].region,
        Region::new(0, 10, 0, 3, 13, 3)
    );

    let snapshot = world.snapshot();
    let restored = World::restore(&snapshot).unwrap();
    assert_eq!(restored.semantic_regions(), world.semantic_regions());
    assert_eq!(restored.snapshot(), snapshot);
    assert_eq!(restored.hash(), world.hash());
}

#[test]
fn explicit_semantic_ids_are_nonzero_and_region_ids_are_unique() {
    let region = Region::new(0, 5, 0, 0, 5, 0);
    let spec = |region_id, structure_id| {
        SemanticRegionSpec::new(
            RegionId::new(region_id),
            StructureId::new(structure_id),
            region,
            STONE,
        )
    };

    assert_eq!(
        World::new_with_semantic_regions(1, Preset::Void, vec![spec(0, 1)])
            .err()
            .as_deref(),
        Some("semantic region id must be non-zero")
    );
    assert_eq!(
        World::new_with_semantic_regions(1, Preset::Void, vec![spec(1, 0)])
            .err()
            .as_deref(),
        Some("semantic structure id must be non-zero")
    );
    assert_eq!(
        World::new_with_semantic_regions(1, Preset::Void, vec![spec(1, 1), spec(1, 2)])
            .err()
            .as_deref(),
        Some("duplicate semantic region id 1")
    );
}

#[test]
fn v8_hash_includes_semantic_ids_while_legacy_hash_excludes_them() {
    let region = Region::new(1, 5, 1, 2, 6, 2);
    let build = |region_id, structure_id| {
        World::new_with_semantic_regions(
            3,
            Preset::Void,
            vec![SemanticRegionSpec::new(
                RegionId::new(region_id),
                StructureId::new(structure_id),
                region,
                STONE,
            )],
        )
        .unwrap()
    };
    let first = build(11, 21);
    let second = build(12, 22);

    assert_ne!(first.hash(), second.hash());
    assert_eq!(first.legacy_hash_v7(), second.legacy_hash_v7());
}

#[test]
fn snapshot_rejects_zero_and_duplicate_persisted_region_ids() {
    let specs = vec![
        SemanticRegionSpec::new(
            RegionId::new(11),
            StructureId::new(21),
            Region::new(0, 5, 0, 0, 5, 0),
            STONE,
        ),
        SemanticRegionSpec::new(
            RegionId::new(12),
            StructureId::new(21),
            Region::new(1, 5, 0, 1, 5, 0),
            DIRT,
        ),
    ];
    let world = World::new_with_semantic_regions(4, Preset::Void, specs).unwrap();
    let snapshot = world.snapshot();
    // v8 trailer = semantic_count + 2 * (ids + region + cell) + dirty_count.
    const SEMANTIC_RECORD_BYTES: usize = 8 + 8 + 6 * 4 + 2;
    let trailer_start = snapshot.len() - (4 + 2 * SEMANTIC_RECORD_BYTES + 4);
    let first_id = trailer_start + 4;
    let second_id = first_id + SEMANTIC_RECORD_BYTES;

    let mut zero = snapshot.clone();
    zero[first_id..first_id + 8].copy_from_slice(&0u64.to_le_bytes());
    assert_eq!(
        World::restore(&zero).err().as_deref(),
        Some("semantic region id must be non-zero")
    );

    let mut zero_structure = snapshot.clone();
    zero_structure[first_id + 8..first_id + 16].copy_from_slice(&0u64.to_le_bytes());
    assert_eq!(
        World::restore(&zero_structure).err().as_deref(),
        Some("semantic structure id must be non-zero")
    );

    let mut duplicate = snapshot;
    let id = duplicate[first_id..first_id + 8].to_vec();
    duplicate[second_id..second_id + 8].copy_from_slice(&id);
    assert_eq!(
        World::restore(&duplicate).err().as_deref(),
        Some("duplicate semantic region id 11")
    );
}

fn assert_pending_cell_change_roundtrips(mut world: World) {
    let before = world.snapshot();
    let mut restored = World::restore(&before).unwrap();
    assert_eq!(restored.snapshot(), before);

    step(&mut world, &Action::default());
    step(&mut restored, &Action::default());
    assert_eq!(restored.snapshot(), world.snapshot());
    assert_eq!(restored.hash(), world.hash());
}

#[test]
fn v8_snapshot_preserves_dirty_work_from_set_cell_and_intervention() {
    let mut direct = World::new(5, Preset::Void, Vec::new());
    direct.set_block(4, 10, 4, SAND);
    assert_pending_cell_change_roundtrips(direct);

    let mut intervened = World::new(6, Preset::Void, Vec::new());
    apply_intervention(
        &mut intervened,
        &InterventionSpec::SetCell {
            at: CellCoord::new(4, 10, 4),
            cell: SAND,
        },
        TraceLevel::Off,
        0,
        1,
    )
    .expect("valid intervention");
    assert_pending_cell_change_roundtrips(intervened);
}

#[test]
fn derived_legacy_region_ids_are_unique_for_distinct_content() {
    let scenario = vec![
        (Region::new(0, 5, 0, 0, 5, 0), STONE),
        (Region::new(1, 5, 0, 1, 5, 0), STONE),
        (Region::new(0, 5, 0, 0, 5, 0), DIRT),
    ];
    let world = World::new(7, Preset::Void, scenario);
    let ids: BTreeSet<_> = world
        .semantic_regions()
        .iter()
        .map(|spec| spec.region_id)
        .collect();
    assert_eq!(ids.len(), 3);
}

#[test]
fn version_seven_snapshot_derives_the_same_content_ids_read_only() {
    let scenario = vec![
        (Region::new(-3, 5, 2, -2, 6, 3), STONE),
        (Region::new(4, 7, -1, 4, 8, 0), DIRT),
    ];
    let world = World::new(8, Preset::Void, scenario);
    let expected = id_pairs(&world);
    let semantic_trailer_bytes = 4 + world.semantic_regions().len() * (8 + 8 + 6 * 4 + 2) + 4;
    let mut legacy = world.snapshot();
    legacy.truncate(legacy.len() - semantic_trailer_bytes);
    legacy[4..8].copy_from_slice(&7u32.to_le_bytes());
    legacy.drain(16..32);

    let restored = World::restore(&legacy).unwrap();
    assert_eq!(id_pairs(&restored), expected);
    assert_eq!(restored.legacy_hash_v7(), world.legacy_hash_v7());
}
