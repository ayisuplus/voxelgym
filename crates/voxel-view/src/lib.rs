#![cfg_attr(all(test, coverage_nightly), feature(coverage_attribute))]

//! voxel-view: CPU DDA raycast renderer.
//!
//! One DDA ray per pixel through the voxel grid (strict policy: fluids
//! included via `dda`/`blocks_ray`). A hit yields color, depth, and segment
//! id in a single traversal — depth and segmentation are exact BY
//! CONSTRUCTION (same traversal as the sim's raycast), bit-identical across
//! machines on the same platform, zero GPU/driver dependency.
//!
//! Shading: registry base color x face factor (top 1.0, north/south 0.8,
//! east/west 0.6, bottom 0.5). Sky (miss): constant color, depth = max_dist,
//! seg = SKY_SEG (0xFFFF).

use rayon::prelude::*;
use voxel_core::block::*;
use voxel_core::raycast::dda;
use voxel_core::World;

pub mod lidar;

pub const SKY_SEG: u16 = 0xFFFF;
pub const SKY_COLOR: [u8; 3] = [0x78, 0xA6, 0xFF];
pub const RENDER_RADIUS_CHUNKS: i32 = 6;

/// Flat chunk-pointer grid over the render radius around an origin: DDA
/// steps cost an array index instead of a HashMap lookup per cell (the
/// dominant cost). Shared by the renderer and the LiDAR scanner — one
/// setup, one boundary policy (y<0 bedrock, y>=top air, off-grid air).
pub(crate) struct ChunkGrid<'w> {
    grid: Vec<Option<&'w voxel_core::Chunk>>,
    /// chunk coords of the grid's (0, 0) corner
    ox: i32,
    oz: i32,
    side: usize,
    top: i32,
    scale: f64,
    max_dist_cells: f64,
}

impl<'w> ChunkGrid<'w> {
    pub(crate) fn new(world: &'w mut World, origin: [f64; 3]) -> Self {
        Self::with_max_distance_meters(world, origin, (RENDER_RADIUS_CHUNKS * 16) as f64)
    }

    pub(crate) fn with_max_distance_meters(
        world: &'w mut World,
        origin: [f64; 3],
        requested_meters: f64,
    ) -> Self {
        let ecx = (origin[0].floor() as i32).div_euclid(16);
        let ecz = (origin[2].floor() as i32).div_euclid(16);
        let scale = world.scale();
        let physical_horizon = requested_meters
            .max(0.0)
            .min((RENDER_RADIUS_CHUNKS * 16) as f64);
        let max_dist_cells = physical_horizon * scale;
        let radius_chunks = (max_dist_cells / 16.0).ceil() as i32;
        // pre-generate the radius so ray reads never trigger generation
        // mid-pass (and to bound ray cost)
        for cx in ecx - radius_chunks..=ecx + radius_chunks {
            for cz in ecz - radius_chunks..=ecz + radius_chunks {
                world.ensure_chunk(cx, cz);
            }
        }
        let side = (2 * radius_chunks + 1) as usize;
        let (ox, oz) = (ecx - radius_chunks, ecz - radius_chunks);
        let mut grid: Vec<Option<&voxel_core::Chunk>> = vec![None; side * side];
        for cz in oz..=ecz + radius_chunks {
            for cx in ox..=ecx + radius_chunks {
                grid[(cz - oz) as usize * side + (cx - ox) as usize] = world.chunks.get(&(cx, cz));
            }
        }
        ChunkGrid {
            grid,
            ox,
            oz,
            side,
            top: world.height(),
            scale,
            max_dist_cells,
        }
    }

    #[inline]
    pub(crate) fn get(&self, x: i32, y: i32, z: i32) -> u16 {
        if y < 0 {
            return BEDROCK;
        }
        if y >= self.top {
            return AIR;
        }
        let gx = x.div_euclid(16) - self.ox;
        let gz = z.div_euclid(16) - self.oz;
        if gx < 0 || gz < 0 || gx >= self.side as i32 || gz >= self.side as i32 {
            return AIR;
        }
        match self.grid[(gz as usize) * self.side + gx as usize] {
            Some(c) => c.get(
                x.rem_euclid(16) as usize,
                y as usize,
                z.rem_euclid(16) as usize,
            ),
            None => AIR,
        }
    }

    /// Max ray distance the grid covers in cell units for DDA traversal.
    pub(crate) const fn max_dist_cells(&self) -> f64 {
        self.max_dist_cells
    }

    pub(crate) const fn scale(&self) -> f64 {
        self.scale
    }
}

/// Face shading by normal: +y top 1.0, -y bottom 0.5, +-z 0.8, +-x 0.6.
pub fn face_shade(face: [i32; 3]) -> f64 {
    match face {
        [0, 1, 0] => 1.0,
        [0, -1, 0] => 0.5,
        [0, 0, 1] | [0, 0, -1] => 0.8,
        _ => 0.6,
    }
}

pub struct Frame {
    pub width: usize,
    pub height: usize,
    /// RGB u8, row-major, 3 channels.
    pub rgb: Vec<u8>,
    /// Depth in meters (ray parameter t), f32.
    pub depth: Vec<f32>,
    /// Block id per pixel; SKY_SEG on miss.
    pub seg: Vec<u16>,
    /// Surface normal per pixel, f32 unit axis, row-major 3 channels;
    /// [0,0,0] on sky miss (a real normal is a unit vector, never zero).
    /// With rgb+depth+seg this completes the per-pixel vector:
    /// [r, g, b, depth, block_id, nx, ny, nz].
    pub normals: Vec<f32>,
}

/// Camera basis from agent yaw/pitch (degrees, MC convention:
/// yaw 0 = +z, positive pitch = down).
pub fn camera_rays(yaw_deg: f64, pitch_deg: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let yaw = yaw_deg.to_radians();
    let pitch = pitch_deg.to_radians();
    let fwd = [
        -yaw.sin() * pitch.cos(),
        -pitch.sin(),
        yaw.cos() * pitch.cos(),
    ];
    // right = normalize(cross(fwd, world_up))
    let mut right = [fwd[2], 0.0, -fwd[0]]; // cross(fwd, [0,1,0]) = (fz, 0, -fx)... sign checked below
    let rl = (right[0] * right[0] + right[2] * right[2]).sqrt();
    if rl < 1e-9 {
        right = [1.0, 0.0, 0.0];
    } else {
        right = [right[0] / rl, 0.0, right[2] / rl];
    }
    // up = cross(fwd, right)
    let up = [
        fwd[1] * right[2] - fwd[2] * right[1],
        fwd[2] * right[0] - fwd[0] * right[2],
        fwd[0] * right[1] - fwd[1] * right[0],
    ];
    (fwd, right, up)
}

/// Render from the agent's eye. rayon over rows; every row computes
/// independently, so parallelism never affects the bytes.
pub fn render(world: &mut World, width: usize, height: usize, fov_deg: f64) -> Frame {
    let eye = world.agent.eye();
    let yaw = world.agent.yaw as f64;
    let pitch = world.agent.pitch as f64;
    render_from(world, eye, yaw, pitch, width, height, fov_deg)
}

/// Render from an arbitrary pose (free camera: third-person, top-down,
/// surveillance). Same determinism guarantees as `render`.
pub fn render_from(
    world: &mut World,
    eye: [f64; 3],
    yaw_deg: f64,
    pitch_deg: f64,
    width: usize,
    height: usize,
    fov_deg: f64,
) -> Frame {
    let grid = ChunkGrid::new(world, eye);
    let max_dist_cells = grid.max_dist_cells();
    let scale = grid.scale();
    let max_dist_meters = max_dist_cells / scale;
    let (fwd, right, up) = camera_rays(yaw_deg, pitch_deg);
    let half = (fov_deg / 2.0).to_radians().tan();

    let mut rgb = vec![0u8; width * height * 3];
    let mut depth = vec![0f32; width * height];
    let mut seg = vec![0u16; width * height];
    let mut normals = vec![0f32; width * height * 3];

    rgb.par_chunks_mut(width * 3)
        .zip(depth.par_chunks_mut(width))
        .zip(seg.par_chunks_mut(width))
        .zip(normals.par_chunks_mut(width * 3))
        .enumerate()
        .for_each(|(py, (((rgb_row, dep_row), seg_row), nrm_row))| {
            // camera pinhole: pixel coords to [-1, 1], py=0 is TOP of image
            let sy = 1.0 - 2.0 * (py as f64 + 0.5) / height as f64;
            for px in 0..width {
                let sx = 2.0 * (px as f64 + 0.5) / width as f64 - 1.0;
                let mut dir = [
                    fwd[0] + right[0] * sx * half + up[0] * sy * half,
                    fwd[1] + right[1] * sx * half + up[1] * sy * half,
                    fwd[2] + right[2] * sx * half + up[2] * sy * half,
                ];
                let dl = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
                dir = [dir[0] / dl, dir[1] / dl, dir[2] / dl];
                let hit = dda(eye, dir, max_dist_cells, |x, y, z| grid.get(x, y, z));
                match hit {
                    Some(h) => {
                        let base = block_def(cell_id(h.cell)).color;
                        let shade = face_shade(h.face);
                        rgb_row[px * 3] = (base[0] as f64 * shade) as u8;
                        rgb_row[px * 3 + 1] = (base[1] as f64 * shade) as u8;
                        rgb_row[px * 3 + 2] = (base[2] as f64 * shade) as u8;
                        dep_row[px] = (h.dist * dl / scale) as f32;
                        seg_row[px] = cell_id(h.cell);
                        nrm_row[px * 3] = h.face[0] as f32;
                        nrm_row[px * 3 + 1] = h.face[1] as f32;
                        nrm_row[px * 3 + 2] = h.face[2] as f32;
                    }
                    None => {
                        rgb_row[px * 3] = SKY_COLOR[0];
                        rgb_row[px * 3 + 1] = SKY_COLOR[1];
                        rgb_row[px * 3 + 2] = SKY_COLOR[2];
                        dep_row[px] = max_dist_meters as f32;
                        seg_row[px] = SKY_SEG;
                        // normals stay [0,0,0] on miss
                    }
                }
            }
        });

    Frame {
        width,
        height,
        rgb,
        depth,
        seg,
        normals,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use voxel_core::worldgen::Preset;

    /// Downward-tilted camera (45 deg) on flat terrain: every pixel well
    /// below the top edge must be grass with depth matching a reference DDA
    /// re-run bitwise (same algorithm, regression pin).
    #[test]
    fn flat_world_golden() {
        let mut w = World::new(7, Preset::Flat, Vec::new());
        w.agent.pos = [8.5, 5.0, 8.5];
        w.agent.yaw = 0.0; // +z
        w.agent.pitch = 45.0; // look down 45 deg (positive pitch = down)
        w.agent.on_ground = true;
        let f = render(&mut w, 128, 128, 90.0);
        assert_eq!(f.seg.len(), 128 * 128);

        // with 45 deg pitch and 90 deg FOV, the top edge is exactly
        // horizontal; rows from the middle down all hit the ground plane
        for py in 64..128 {
            for px in 0..128 {
                assert_eq!(
                    f.seg[py * 128 + px],
                    GRASS_BLOCK,
                    "ground region must be grass (px={px}, py={py})"
                );
            }
        }
        // reference DDA re-run for sample pixels: bitwise depth equality
        let eye = w.agent.eye();
        let (fwd, right, up) = camera_rays(0.0, 45.0);
        let half = (45.0f64).to_radians().tan();
        for (px, py) in [
            (64, 100),
            (10, 80),
            (120, 70),
            (64, 120),
            (0, 64),
            (127, 127),
        ] {
            let sy = 1.0 - 2.0 * (py as f64 + 0.5) / 128.0;
            let sx = 2.0 * (px as f64 + 0.5) / 128.0 - 1.0;
            let mut dir = [
                fwd[0] + right[0] * sx * half + up[0] * sy * half,
                fwd[1] + right[1] * sx * half + up[1] * sy * half,
                fwd[2] + right[2] * sx * half + up[2] * sy * half,
            ];
            let dl = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            dir = [dir[0] / dl, dir[1] / dl, dir[2] / dl];
            let h = dda(eye, dir, 96.0, |x, y, z| w.peek_block(x, y, z)).expect("ground hit");
            let expected = (h.dist * dl) as f32;
            assert_eq!(
                f.depth[py * 128 + px],
                expected,
                "depth bitwise (px={px},py={py})"
            );
            assert_eq!(f.seg[py * 128 + px], GRASS_BLOCK);
        }
    }

    #[test]
    fn depth_seg_consistent_with_voxels() {
        // looking straight-ish down at a pillar of stone on flat ground:
        // center pixels must report stone, off-center grass
        let mut w = World::new(3, Preset::Flat, Vec::new());
        for y in 5..9 {
            w.set_block(10, y, 8, STONE);
        }
        w.agent.pos = [6.5, 5.0, 8.5];
        w.agent.yaw = 270.0; // degrees: +x
        w.agent.pitch = 0.0;
        w.agent.on_ground = true;
        let f = render(&mut w, 128, 128, 90.0);
        // the pillar should appear somewhere right of center with STONE seg
        let stone_px = f.seg.iter().filter(|&&s| s == STONE).count();
        assert!(stone_px > 50, "pillar visible: {stone_px} px");
        // depths on stone pixels should be a few cells (face at 3.5)
        let i = f.seg.iter().position(|&s| s == STONE).unwrap();
        assert!(
            (3.0..=6.0).contains(&f.depth[i]),
            "depth {} in range",
            f.depth[i]
        );
    }

    #[test]
    fn camera_basis_is_canonical_and_orthonormal_at_the_poles() {
        let (forward, right, up) = camera_rays(0.0, 0.0);
        assert_eq!(forward, [-0.0, -0.0, 1.0]);
        assert_eq!(right, [1.0, 0.0, 0.0]);
        assert_eq!(up, [-0.0, 1.0, 0.0]);

        let (forward, right, up) = camera_rays(37.0, 90.0);
        for axis in [forward, right, up] {
            let length = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            assert!((length - 1.0).abs() < 1e-12);
        }
        for (a, b) in [(forward, right), (forward, up), (right, up)] {
            let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            assert!(dot.abs() < 1e-12);
        }
    }

    #[test]
    fn face_shading_covers_every_axis_class() {
        assert_eq!(face_shade([0, 1, 0]), 1.0);
        assert_eq!(face_shade([0, -1, 0]), 0.5);
        assert_eq!(face_shade([0, 0, 1]), 0.8);
        assert_eq!(face_shade([0, 0, -1]), 0.8);
        assert_eq!(face_shade([1, 0, 0]), 0.6);
        assert_eq!(face_shade([-1, 0, 0]), 0.6);
    }

    #[test]
    fn non_square_void_render_is_a_complete_sky_frame() {
        let mut world = World::new(11, Preset::Void, Vec::new());
        world.agent.pos = [2.5, 40.0, -1.5];
        world.agent.yaw = 15.0;
        world.agent.pitch = 0.0;
        let eye = world.agent.eye();

        let from_pose = render_from(&mut world, eye, 15.0, 0.0, 3, 2, 70.0);
        assert_eq!((from_pose.width, from_pose.height), (3, 2));
        assert_eq!(from_pose.rgb.len(), 3 * 2 * 3);
        assert_eq!(from_pose.depth.len(), 3 * 2);
        assert_eq!(from_pose.seg.len(), 3 * 2);
        assert_eq!(from_pose.normals.len(), 3 * 2 * 3);
        assert!(from_pose
            .rgb
            .as_chunks::<3>()
            .0
            .iter()
            .all(|pixel| *pixel == SKY_COLOR));
        assert!(from_pose.depth.iter().all(|&depth| depth == 96.0));
        assert!(from_pose.seg.iter().all(|&segment| segment == SKY_SEG));
        assert!(from_pose.normals.iter().all(|&normal| normal == 0.0));

        let from_agent = render(&mut world, 3, 2, 70.0);
        assert_eq!(from_agent.rgb, from_pose.rgb);
        assert_eq!(from_agent.depth, from_pose.depth);
        assert_eq!(from_agent.seg, from_pose.seg);
        assert_eq!(from_agent.normals, from_pose.normals);
    }
}
