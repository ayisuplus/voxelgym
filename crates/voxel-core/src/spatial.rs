//! Strong spatial types and deterministic, world-independent queries.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

/// A continuous position measured in world cells.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WorldPos {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl WorldPos {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub const fn from_cells(cells: [f64; 3]) -> Self {
        Self::new(cells[0], cells[1], cells[2])
    }

    pub const fn cells(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

impl From<[f64; 3]> for WorldPos {
    fn from(value: [f64; 3]) -> Self {
        Self::from_cells(value)
    }
}

impl From<WorldPos> for [f64; 3] {
    fn from(value: WorldPos) -> Self {
        value.cells()
    }
}

/// A discrete voxel address in the world frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CellCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl CellCoord {
    pub const ORIGIN: Self = Self::new(0, 0, 0);

    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Returns the cell containing a valid engine position.
    ///
    /// Engine coordinates are finite and constrained well inside the `i32`
    /// range. Use [`Self::try_from_world_pos`] at untrusted boundaries.
    pub fn from_world_pos(position: WorldPos) -> Self {
        Self::try_from_world_pos(position)
            .expect("world position must be finite and fit in a cell coordinate")
    }

    pub fn try_from_world_pos(position: WorldPos) -> Option<Self> {
        Some(Self::new(
            floor_to_i32(position.x)?,
            floor_to_i32(position.y)?,
            floor_to_i32(position.z)?,
        ))
    }

    pub const fn cells(self) -> [i32; 3] {
        [self.x, self.y, self.z]
    }

    pub const fn offset(self, dx: i32, dy: i32, dz: i32) -> Self {
        Self::new(self.x + dx, self.y + dy, self.z + dz)
    }

    /// Six face-neighbors in lexicographic `(x, y, z)` order.
    pub const fn neighbors6(self) -> [Self; 6] {
        [
            self.offset(-1, 0, 0),
            self.offset(0, -1, 0),
            self.offset(0, 0, -1),
            self.offset(0, 0, 1),
            self.offset(0, 1, 0),
            self.offset(1, 0, 0),
        ]
    }

    pub fn manhattan_distance(self, other: Self) -> u64 {
        u64::from(self.x.abs_diff(other.x))
            + u64::from(self.y.abs_diff(other.y))
            + u64::from(self.z.abs_diff(other.z))
    }

    pub fn is_adjacent_to(self, other: Self) -> bool {
        adjacent(self, other)
    }
}

impl From<[i32; 3]> for CellCoord {
    fn from(value: [i32; 3]) -> Self {
        Self::new(value[0], value[1], value[2])
    }
}

impl From<(i32, i32, i32)> for CellCoord {
    fn from(value: (i32, i32, i32)) -> Self {
        Self::new(value.0, value.1, value.2)
    }
}

impl From<CellCoord> for [i32; 3] {
    fn from(value: CellCoord) -> Self {
        value.cells()
    }
}

impl From<CellCoord> for (i32, i32, i32) {
    fn from(value: CellCoord) -> Self {
        (value.x, value.y, value.z)
    }
}

fn floor_to_i32(value: f64) -> Option<i32> {
    let value = value.floor();
    if value.is_finite() && value >= i32::MIN as f64 && value <= i32::MAX as f64 {
        Some(value as i32)
    } else {
        None
    }
}

/// Whether two cells share a face.
pub fn adjacent(left: CellCoord, right: CellCoord) -> bool {
    left.manhattan_distance(right) == 1
}

/// Whether `candidate` is anywhere above `reference` in the same column.
pub fn above(candidate: CellCoord, reference: CellCoord) -> bool {
    candidate.x == reference.x && candidate.z == reference.z && candidate.y > reference.y
}

/// Whether `candidate` is anywhere below `reference` in the same column.
pub fn below(candidate: CellCoord, reference: CellCoord) -> bool {
    above(reference, candidate)
}

/// Whether two world-cell positions are no farther apart than `radius`.
pub fn within(origin: WorldPos, candidate: WorldPos, radius: f64) -> bool {
    if !all_finite(origin) || !all_finite(candidate) || !radius.is_finite() || radius < 0.0 {
        return false;
    }
    let dx = candidate.x - origin.x;
    let dy = candidate.y - origin.y;
    let dz = candidate.z - origin.z;
    dx * dx + dy * dy + dz * dz <= radius * radius
}

/// Nearest candidate cell center to a continuous world position.
/// Equal distances are resolved lexicographically for deterministic labels.
pub fn nearest<I>(origin: WorldPos, candidates: I) -> Option<CellCoord>
where
    I: IntoIterator<Item = CellCoord>,
{
    if !all_finite(origin) {
        return None;
    }
    candidates.into_iter().min_by(|left, right| {
        let distance = |cell: &CellCoord| {
            let dx = cell.x as f64 + 0.5 - origin.x;
            let dy = cell.y as f64 + 0.5 - origin.y;
            let dz = cell.z as f64 + 0.5 - origin.z;
            dx * dx + dy * dy + dz * dz
        };
        distance(left)
            .total_cmp(&distance(right))
            .then_with(|| left.cmp(right))
    })
}

/// A finite guard for queries over an otherwise unbounded voxel lattice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchError {
    VisitLimitExceeded { limit: usize },
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VisitLimitExceeded { limit } => {
                write!(f, "spatial query exceeded its {limit}-cell visit limit")
            }
        }
    }
}

impl Error for SearchError {}

/// Finds the face-connected occupied component containing `start`.
///
/// The returned coordinates are sorted lexicographically. `occupied` should
/// be a pure predicate; it is evaluated at most once for each examined cell.
pub fn connected_component<F>(
    start: CellCoord,
    max_visited: usize,
    mut occupied: F,
) -> Result<Vec<CellCoord>, SearchError>
where
    F: FnMut(CellCoord) -> bool,
{
    if !occupied(start) {
        return Ok(Vec::new());
    }
    if max_visited == 0 {
        return Err(SearchError::VisitLimitExceeded { limit: 0 });
    }

    let mut examined = BTreeSet::from([start]);
    let mut component = vec![start];
    let mut frontier = VecDeque::from([start]);

    while let Some(current) = frontier.pop_front() {
        for neighbor in current.neighbors6() {
            if !examined.insert(neighbor) || !occupied(neighbor) {
                continue;
            }
            if component.len() == max_visited {
                return Err(SearchError::VisitLimitExceeded { limit: max_visited });
            }
            component.push(neighbor);
            frontier.push_back(neighbor);
        }
    }

    component.sort_unstable();
    Ok(component)
}

/// Finds a shortest face-neighbor path, including `start` and `goal`.
///
/// Equal-length paths are resolved by lexicographic neighbor order. Both
/// endpoints must satisfy `passable`. The visit limit makes unreachable
/// searches safe even when the predicate describes an unbounded space.
pub fn shortest_path<F>(
    start: CellCoord,
    goal: CellCoord,
    max_visited: usize,
    mut passable: F,
) -> Result<Option<Vec<CellCoord>>, SearchError>
where
    F: FnMut(CellCoord) -> bool,
{
    if !passable(start) {
        return Ok(None);
    }
    if max_visited == 0 {
        return Err(SearchError::VisitLimitExceeded { limit: 0 });
    }
    if start == goal {
        return Ok(Some(vec![start]));
    }

    let mut examined = BTreeSet::from([start]);
    let mut frontier = VecDeque::from([start]);
    let mut parents = BTreeMap::new();
    let mut visited_count = 1usize;

    while let Some(current) = frontier.pop_front() {
        for neighbor in current.neighbors6() {
            if !examined.insert(neighbor) || !passable(neighbor) {
                continue;
            }
            if visited_count == max_visited {
                return Err(SearchError::VisitLimitExceeded { limit: max_visited });
            }
            visited_count += 1;
            parents.insert(neighbor, current);
            if neighbor == goal {
                return Ok(Some(reconstruct_path(start, goal, &parents)));
            }
            frontier.push_back(neighbor);
        }
    }

    Ok(None)
}

fn reconstruct_path(
    start: CellCoord,
    goal: CellCoord,
    parents: &BTreeMap<CellCoord, CellCoord>,
) -> Vec<CellCoord> {
    let mut path = vec![goal];
    let mut current = goal;
    while current != start {
        current = parents[&current];
        path.push(current);
    }
    path.reverse();
    path
}

/// Reports whether a face-neighbor route exists under the same bounded
/// semantics as [`shortest_path`].
pub fn reachable<F>(
    start: CellCoord,
    goal: CellCoord,
    max_visited: usize,
    passable: F,
) -> Result<bool, SearchError>
where
    F: FnMut(CellCoord) -> bool,
{
    Ok(shortest_path(start, goal, max_visited, passable)?.is_some())
}

macro_rules! stable_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u64);

        impl $name {
            pub const fn new(value: u64) -> Self {
                Self(value)
            }

            pub const fn get(self) -> u64 {
                self.0
            }
        }

        impl From<u64> for $name {
            fn from(value: u64) -> Self {
                Self::new(value)
            }
        }

        impl From<$name> for u64 {
            fn from(value: $name) -> Self {
                value.get()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

stable_id!(/// Stable identifier for a coordinate frame.
    FrameId);
stable_id!(/// Stable identifier for a dynamic entity.
    EntityId);
stable_id!(/// Stable identifier for a semantic multi-cell structure.
    StructureId);
stable_id!(/// Stable identifier for a semantic region.
    RegionId);

impl FrameId {
    pub const WORLD: Self = Self(0);
}

impl EntityId {
    /// Reserved identifier for the current single-agent actor.
    pub const AGENT: Self = Self(0);

    /// Namespace a world-local item id without colliding with actors or
    /// other dynamic entity classes.
    pub const fn item(world_local_id: u64) -> Self {
        Self((1u64 << 62) | (world_local_id & ((1u64 << 62) - 1)))
    }

    /// Namespace a world-local falling-block id.
    pub const fn falling_block(world_local_id: u64) -> Self {
        Self((2u64 << 62) | (world_local_id & ((1u64 << 62) - 1)))
    }
}

/// Position and Minecraft-convention orientation in an explicit frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub frame: FrameId,
    pub position: WorldPos,
    /// Degrees; `0` faces +Z and `90` faces -X.
    pub yaw_degrees: f32,
    /// Degrees; negative values look upward.
    pub pitch_degrees: f32,
}

impl Pose {
    pub const fn new(
        frame: FrameId,
        position: WorldPos,
        yaw_degrees: f32,
        pitch_degrees: f32,
    ) -> Self {
        Self {
            frame,
            position,
            yaw_degrees,
            pitch_degrees,
        }
    }

    pub const fn world(position: WorldPos, yaw_degrees: f32, pitch_degrees: f32) -> Self {
        Self::new(FrameId::WORLD, position, yaw_degrees, pitch_degrees)
    }
}

/// Error returned when composing or applying incompatible coordinate frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameError {
    NonFinite,
    FrameMismatch { expected: FrameId, actual: FrameId },
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => f.write_str("frame transform values must be finite"),
            Self::FrameMismatch { expected, actual } => write!(
                f,
                "coordinate frame mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl Error for FrameError {}

/// Rigid upright transform from one local voxel frame into its parent frame.
///
/// Voxel worlds share a gravity axis, so frame orientation is an exact yaw
/// around +Y plus a continuous translation in cell units.  Positive yaw uses
/// the simulator convention: local +Z maps toward parent -X at 90 degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameTransform {
    pub local_frame: FrameId,
    pub parent_frame: FrameId,
    pub origin_in_parent: WorldPos,
    pub yaw_degrees: f64,
}

impl FrameTransform {
    pub fn new(
        local_frame: FrameId,
        parent_frame: FrameId,
        origin_in_parent: WorldPos,
        yaw_degrees: f64,
    ) -> Result<Self, FrameError> {
        if !all_finite(origin_in_parent) || !yaw_degrees.is_finite() {
            return Err(FrameError::NonFinite);
        }
        Ok(Self {
            local_frame,
            parent_frame,
            origin_in_parent,
            yaw_degrees: yaw_degrees.rem_euclid(360.0),
        })
    }

    pub fn identity(frame: FrameId) -> Self {
        Self::new(frame, frame, WorldPos::default(), 0.0)
            .expect("identity frame transform is finite")
    }

    /// Convert a point expressed in `local_frame` into `parent_frame`.
    pub fn local_to_parent(self, local: WorldPos) -> Result<WorldPos, FrameError> {
        if !all_finite(local) {
            return Err(FrameError::NonFinite);
        }
        let (sin, cos) = self.yaw_degrees.to_radians().sin_cos();
        Ok(WorldPos::new(
            cos.mul_add(local.x, -sin * local.z) + self.origin_in_parent.x,
            local.y + self.origin_in_parent.y,
            sin.mul_add(local.x, cos * local.z) + self.origin_in_parent.z,
        ))
    }

    /// Convert a parent-frame point back into `local_frame`.
    pub fn parent_to_local(self, parent: WorldPos) -> Result<WorldPos, FrameError> {
        if !all_finite(parent) {
            return Err(FrameError::NonFinite);
        }
        let dx = parent.x - self.origin_in_parent.x;
        let dy = parent.y - self.origin_in_parent.y;
        let dz = parent.z - self.origin_in_parent.z;
        let (sin, cos) = self.yaw_degrees.to_radians().sin_cos();
        Ok(WorldPos::new(
            cos.mul_add(dx, sin * dz),
            dy,
            (-sin).mul_add(dx, cos * dz),
        ))
    }

    pub fn transform_pose(self, local: Pose) -> Result<Pose, FrameError> {
        if local.frame != self.local_frame {
            return Err(FrameError::FrameMismatch {
                expected: self.local_frame,
                actual: local.frame,
            });
        }
        Ok(Pose::new(
            self.parent_frame,
            self.local_to_parent(local.position)?,
            (f64::from(local.yaw_degrees) + self.yaw_degrees).rem_euclid(360.0) as f32,
            local.pitch_degrees,
        ))
    }

    /// Compose `self: A -> B` with `next: B -> C`, yielding `A -> C`.
    pub fn then(self, next: Self) -> Result<Self, FrameError> {
        if self.parent_frame != next.local_frame {
            return Err(FrameError::FrameMismatch {
                expected: self.parent_frame,
                actual: next.local_frame,
            });
        }
        Self::new(
            self.local_frame,
            next.parent_frame,
            next.local_to_parent(self.origin_in_parent)?,
            self.yaw_degrees + next.yaw_degrees,
        )
    }

    pub fn inverse(self) -> Self {
        let inverse_yaw = (-self.yaw_degrees).rem_euclid(360.0);
        let (sin, cos) = inverse_yaw.to_radians().sin_cos();
        let translation = WorldPos::new(
            cos.mul_add(-self.origin_in_parent.x, -sin * -self.origin_in_parent.z),
            -self.origin_in_parent.y,
            sin.mul_add(-self.origin_in_parent.x, cos * -self.origin_in_parent.z),
        );
        Self {
            local_frame: self.parent_frame,
            parent_frame: self.local_frame,
            origin_in_parent: translation,
            yaw_degrees: inverse_yaw,
        }
    }
}

/// A continuous position measured in meters.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MetricPos {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl MetricPos {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// Error returned when constructing invalid continuous bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AabbError {
    NonFinite,
    Inverted,
}

impl fmt::Display for AabbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => f.write_str("AABB bounds must be finite"),
            Self::Inverted => f.write_str("AABB minimum must not exceed its maximum"),
        }
    }
}

impl Error for AabbError {}

/// Axis-aligned bounds in continuous world-cell coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: WorldPos,
    pub max: WorldPos,
}

impl Aabb {
    pub fn new(min: WorldPos, max: WorldPos) -> Result<Self, AabbError> {
        if !all_finite(min) || !all_finite(max) {
            return Err(AabbError::NonFinite);
        }
        if min.x > max.x || min.y > max.y || min.z > max.z {
            return Err(AabbError::Inverted);
        }
        Ok(Self { min, max })
    }

    /// Inclusive containment. Touching a boundary does not leave the box.
    pub fn contains(self, position: WorldPos) -> bool {
        all_finite(position)
            && position.x >= self.min.x
            && position.x <= self.max.x
            && position.y >= self.min.y
            && position.y <= self.max.y
            && position.z >= self.min.z
            && position.z <= self.max.z
    }

    /// Inclusive intersection: touching faces, edges, or corners intersect.
    pub fn intersects(self, other: Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }
}

fn all_finite(position: WorldPos) -> bool {
    position.x.is_finite() && position.y.is_finite() && position.z.is_finite()
}

/// Error returned when a spatial scale cannot describe physical space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleError {
    ZeroNumerator,
    ZeroDenominator,
}

impl fmt::Display for ScaleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroNumerator => f.write_str("cells-per-meter numerator must be non-zero"),
            Self::ZeroDenominator => f.write_str("cells-per-meter denominator must be non-zero"),
        }
    }
}

impl Error for ScaleError {}

/// Exact structural scale expressed as reduced cells per meter.
///
/// A scale of `2/1` means each cell is 0.5 meters wide. Keeping the ratio
/// exact prevents configuration identity from depending on floating-point
/// formatting while conversions remain compatible with the engine's `f64`
/// positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpatialScale {
    cells_per_meter_numerator: u32,
    cells_per_meter_denominator: u32,
}

impl Default for SpatialScale {
    fn default() -> Self {
        Self::ONE
    }
}

impl SpatialScale {
    pub const ONE: Self = Self {
        cells_per_meter_numerator: 1,
        cells_per_meter_denominator: 1,
    };

    pub fn new(numerator: u32, denominator: u32) -> Result<Self, ScaleError> {
        if numerator == 0 {
            return Err(ScaleError::ZeroNumerator);
        }
        if denominator == 0 {
            return Err(ScaleError::ZeroDenominator);
        }
        let divisor = gcd(numerator, denominator);
        Ok(Self {
            cells_per_meter_numerator: numerator / divisor,
            cells_per_meter_denominator: denominator / divisor,
        })
    }

    pub const fn numerator(self) -> u32 {
        self.cells_per_meter_numerator
    }

    pub const fn denominator(self) -> u32 {
        self.cells_per_meter_denominator
    }

    pub fn cells_per_meter(self) -> f64 {
        self.numerator() as f64 / self.denominator() as f64
    }

    pub fn meters_per_cell(self) -> f64 {
        self.denominator() as f64 / self.numerator() as f64
    }

    pub fn world_to_metric(self, position: WorldPos) -> MetricPos {
        let factor = self.meters_per_cell();
        MetricPos::new(
            position.x * factor,
            position.y * factor,
            position.z * factor,
        )
    }

    pub fn metric_to_world(self, position: MetricPos) -> WorldPos {
        let factor = self.cells_per_meter();
        WorldPos::new(
            position.x * factor,
            position.y * factor,
            position.z * factor,
        )
    }

    pub fn cell_min_to_metric(self, cell: CellCoord) -> MetricPos {
        self.world_to_metric(WorldPos::new(cell.x as f64, cell.y as f64, cell.z as f64))
    }

    pub fn cell_containing_metric(self, position: MetricPos) -> Option<CellCoord> {
        CellCoord::try_from_world_pos(self.metric_to_world(position))
    }

    /// Convert a metric point in a local frame directly to parent world-cell
    /// coordinates, making unit and frame conversion one explicit operation.
    pub fn local_metric_to_parent_world(
        self,
        transform: FrameTransform,
        position: MetricPos,
    ) -> Result<WorldPos, FrameError> {
        transform.local_to_parent(self.metric_to_world(position))
    }

    /// Convert parent world-cell coordinates into metric coordinates in the
    /// transform's local frame.
    pub fn parent_world_to_local_metric(
        self,
        transform: FrameTransform,
        position: WorldPos,
    ) -> Result<MetricPos, FrameError> {
        Ok(self.world_to_metric(transform.parent_to_local(position)?))
    }
}

const fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn spatial_scale_round_trips_between_cell_and_metric_space() {
        let scale = SpatialScale::new(2, 1).unwrap();
        let world = WorldPos::new(3.0, -4.0, 5.0);

        let metric = scale.world_to_metric(world);

        assert_eq!(metric, MetricPos::new(1.5, -2.0, 2.5));
        assert_eq!(scale.metric_to_world(metric), world);
    }

    #[test]
    fn default_spatial_scale_is_reduced_one_cell_per_meter() {
        assert_eq!(SpatialScale::default(), SpatialScale::new(4, 4).unwrap());
    }

    #[test]
    fn frame_transform_round_trips_a_quarter_turn_and_pose() {
        let local = FrameId::new(7);
        let transform =
            FrameTransform::new(local, FrameId::WORLD, WorldPos::new(10.0, 2.0, -3.0), 90.0)
                .unwrap();
        let point = WorldPos::new(0.0, 1.0, 2.0);
        let parent = transform.local_to_parent(point).unwrap();
        assert!((parent.x - 8.0).abs() < 1e-12);
        assert!((parent.y - 3.0).abs() < 1e-12);
        assert!((parent.z + 3.0).abs() < 1e-12);
        let restored = transform.parent_to_local(parent).unwrap();
        assert!((restored.x - point.x).abs() < 1e-12);
        assert!((restored.y - point.y).abs() < 1e-12);
        assert!((restored.z - point.z).abs() < 1e-12);

        let pose = transform
            .transform_pose(Pose::new(local, point, 300.0, -15.0))
            .unwrap();
        assert_eq!(pose.frame, FrameId::WORLD);
        assert_eq!(pose.yaw_degrees, 30.0);
        assert_eq!(pose.pitch_degrees, -15.0);
    }

    #[test]
    fn frame_composition_inverse_and_metric_conversion_are_consistent() {
        let a = FrameId::new(1);
        let b = FrameId::new(2);
        let first = FrameTransform::new(a, b, WorldPos::new(2.0, 0.0, 1.0), 90.0).unwrap();
        let second =
            FrameTransform::new(b, FrameId::WORLD, WorldPos::new(-4.0, 3.0, 5.0), 270.0).unwrap();
        let composed = first.then(second).unwrap();
        let local = WorldPos::new(1.25, -2.0, 0.5);
        let sequential = second
            .local_to_parent(first.local_to_parent(local).unwrap())
            .unwrap();
        let direct = composed.local_to_parent(local).unwrap();
        assert!((direct.x - sequential.x).abs() < 1e-12);
        assert!((direct.y - sequential.y).abs() < 1e-12);
        assert!((direct.z - sequential.z).abs() < 1e-12);
        let restored = composed.inverse().local_to_parent(direct).unwrap();
        assert!((restored.x - local.x).abs() < 1e-12);
        assert!((restored.y - local.y).abs() < 1e-12);
        assert!((restored.z - local.z).abs() < 1e-12);

        let scale = SpatialScale::new(2, 1).unwrap();
        let metric = MetricPos::new(0.5, 1.0, -0.25);
        let parent = scale
            .local_metric_to_parent_world(composed, metric)
            .unwrap();
        let metric_restored = scale
            .parent_world_to_local_metric(composed, parent)
            .unwrap();
        assert!((metric_restored.x - metric.x).abs() < 1e-12);
        assert!((metric_restored.y - metric.y).abs() < 1e-12);
        assert!((metric_restored.z - metric.z).abs() < 1e-12);

        let wrong_pose = Pose::world(WorldPos::default(), 0.0, 0.0);
        assert_eq!(
            first.transform_pose(wrong_pose),
            Err(FrameError::FrameMismatch {
                expected: a,
                actual: FrameId::WORLD,
            })
        );
    }

    #[test]
    fn containing_cell_uses_floor_for_negative_world_positions() {
        let position = WorldPos::from_cells([-0.01, 7.99, -16.0]);

        let cell = CellCoord::from_world_pos(position);

        assert_eq!(cell, CellCoord::new(-1, 7, -16));
        assert_eq!(position.cells(), [-0.01, 7.99, -16.0]);
    }

    #[test]
    fn stable_ids_remain_distinct_domain_types() {
        let entity = EntityId::AGENT;
        let structure = StructureId::new(1);
        let region = RegionId::new(1);

        assert_eq!(entity.get(), 0);
        assert_eq!(structure.get(), region.get());
    }

    #[test]
    fn aabb_rejects_inverted_bounds_and_contains_its_boundary() {
        let min = WorldPos::new(-1.0, 2.0, 3.0);
        let max = WorldPos::new(1.0, 4.0, 5.0);

        let bounds = Aabb::new(min, max).unwrap();

        assert!(bounds.contains(min));
        assert!(Aabb::new(max, min).is_err());
    }

    #[test]
    fn pose_carries_an_explicit_coordinate_frame() {
        let pose = Pose::new(FrameId::WORLD, WorldPos::new(1.0, 2.0, 3.0), 90.0, -30.0);

        assert_eq!(pose.frame, FrameId::WORLD);
        assert_eq!(pose.position, WorldPos::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn cell_metric_transforms_preserve_negative_cell_ownership() {
        let scale = SpatialScale::new(2, 1).unwrap();
        let cell = CellCoord::new(-1, 3, 4);

        assert_eq!(
            scale.cell_min_to_metric(cell),
            MetricPos::new(-0.5, 1.5, 2.0)
        );
        assert_eq!(
            scale.cell_containing_metric(MetricPos::new(-0.01, 1.99, 2.49)),
            Some(cell)
        );
    }

    #[test]
    fn six_neighbor_relations_have_stable_lexicographic_order() {
        let origin = CellCoord::ORIGIN;

        assert_eq!(
            origin.neighbors6(),
            [
                CellCoord::new(-1, 0, 0),
                CellCoord::new(0, -1, 0),
                CellCoord::new(0, 0, -1),
                CellCoord::new(0, 0, 1),
                CellCoord::new(0, 1, 0),
                CellCoord::new(1, 0, 0),
            ]
        );
        assert!(adjacent(origin, CellCoord::new(0, 1, 0)));
        assert!(above(CellCoord::new(0, 4, 0), origin));
        assert!(below(origin, CellCoord::new(0, 4, 0)));
    }

    #[test]
    fn within_uses_euclidean_distance_in_world_cell_units() {
        let origin = WorldPos::new(0.0, 0.0, 0.0);

        assert!(within(origin, WorldPos::new(3.0, 4.0, 0.0), 5.0));
        assert!(!within(origin, WorldPos::new(3.0, 4.0, 0.1), 5.0));
    }

    #[test]
    fn nearest_uses_cell_centers_and_lexicographic_ties() {
        let origin = WorldPos::new(0.5, 0.5, 0.5);
        let candidates = [CellCoord::new(1, 0, 0), CellCoord::new(-1, 0, 0)];
        assert_eq!(nearest(origin, candidates), Some(CellCoord::new(-1, 0, 0)));
        assert_eq!(nearest(WorldPos::new(f64::NAN, 0.0, 0.0), candidates), None);
    }

    #[test]
    fn connected_component_is_sorted_and_excludes_disconnected_cells() {
        let occupied = [
            CellCoord::new(0, 0, 0),
            CellCoord::new(1, 0, 0),
            CellCoord::new(1, 1, 0),
            CellCoord::new(9, 9, 9),
        ];

        let component =
            connected_component(CellCoord::ORIGIN, 16, |cell| occupied.contains(&cell)).unwrap();

        assert_eq!(
            component,
            vec![
                CellCoord::new(0, 0, 0),
                CellCoord::new(1, 0, 0),
                CellCoord::new(1, 1, 0),
            ]
        );
    }

    #[test]
    fn shortest_path_uses_stable_tie_breaking_around_obstacles() {
        let start = CellCoord::new(0, 0, 0);
        let goal = CellCoord::new(2, 0, 0);
        let blocked = CellCoord::new(1, 0, 0);
        let passable = |cell: CellCoord| {
            cell.y == 0
                && (0..=2).contains(&cell.x)
                && (-1..=1).contains(&cell.z)
                && cell != blocked
        };

        let path = shortest_path(start, goal, 32, passable).unwrap().unwrap();

        assert_eq!(
            path,
            vec![
                CellCoord::new(0, 0, 0),
                CellCoord::new(0, 0, -1),
                CellCoord::new(1, 0, -1),
                CellCoord::new(2, 0, -1),
                CellCoord::new(2, 0, 0),
            ]
        );
    }
}
