#![cfg_attr(all(test, coverage_nightly), feature(coverage_attribute))]

pub mod block;
pub mod chunk;
pub mod circuit;
pub mod clock;
pub mod entity;
pub mod fire;
pub mod fluid;
pub mod hooks;
pub mod inventory;
pub mod item;
pub mod loose;
pub mod physics;
pub mod raycast;
pub mod recipe;
pub mod rng;
pub mod spatial;
pub mod tick;
pub mod tnt;
pub mod trace;
pub mod world;
pub mod worldgen;

pub use block::*;
pub use chunk::{Chunk, CHUNK_VOL, SEA_LEVEL, WORLD_MAX_Y, WORLD_MIN_Y};
pub use clock::{ClockConfig, SimClock};
pub use entity::Agent;
pub use inventory::{Inventory, Stack};
pub use item::ItemEntity;
pub use rng::Rng;
pub use spatial::{
    Aabb, CellCoord, EntityId, FrameError, FrameId, FrameTransform, MetricPos, Pose, RegionId,
    SpatialScale, StructureId, WorldPos,
};
pub use tick::{step, Action};
pub use world::{Event, World};
pub use worldgen::{
    derive_semantic_regions, scale_scenario_to_cells, Preset, Region, ScenarioSpec,
    SemanticRegionSpec,
};
