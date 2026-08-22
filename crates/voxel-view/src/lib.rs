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
    /// Depth in cells (ray parameter t), f32.
    pub depth: Vec<f32>,
    /// Block id per pixel; SKY_SEG on miss.
    pub seg: Vec<u16>,
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
    // pre-generate the render radius so ray reads never trigger generation
    // mid-render (and to bound ray cost)
    let eye = world.agent.eye();
    let ecx = (eye[0].floor() as i32).div_euclid(16);
    let ecz = (eye[2].floor() as i32).div_euclid(16);
    for cx in ecx - RENDER_RADIUS_CHUNKS..=ecx + RENDER_RADIUS_CHUNKS {
        for cz in ecz - RENDER_RADIUS_CHUNKS..=ecz + RENDER_RADIUS_CHUNKS {
            world.ensure_chunk(cx, cz);
        }
    }

    let max_dist = (RENDER_RADIUS_CHUNKS * 16) as f64;
    let (fwd, right, up) = camera_rays(world.agent.yaw as f64, world.agent.pitch as f64);
    let half = (fov_deg / 2.0).to_radians().tan();

    // flat chunk-pointer grid over the render radius: DDA steps then cost an
    // array index instead of a HashMap lookup per cell (the dominant cost)
    let side = (2 * RENDER_RADIUS_CHUNKS + 1) as usize;
    let mut grid: Vec<Option<&voxel_core::Chunk>> = vec![None; side * side];
    for cz in ecz - RENDER_RADIUS_CHUNKS..=ecz + RENDER_RADIUS_CHUNKS {
        for cx in ecx - RENDER_RADIUS_CHUNKS..=ecx + RENDER_RADIUS_CHUNKS {
            let gx = (cx - (ecx - RENDER_RADIUS_CHUNKS)) as usize;
            let gz = (cz - (ecz - RENDER_RADIUS_CHUNKS)) as usize;
            grid[gz * side + gx] = world.chunks.get(&(cx, cz));
        }
    }
    let get = |x: i32, y: i32, z: i32| -> u16 {
        if y < 0 {
            return BEDROCK;
        }
        if y > 127 {
            return AIR;
        }
        let gx = x.div_euclid(16) - (ecx - RENDER_RADIUS_CHUNKS);
        let gz = z.div_euclid(16) - (ecz - RENDER_RADIUS_CHUNKS);
        if gx < 0 || gz < 0 || gx >= side as i32 || gz >= side as i32 {
            return AIR;
        }
        match grid[(gz as usize) * side + gx as usize] {
            Some(c) => c.get(x.rem_euclid(16) as usize, y as usize, z.rem_euclid(16) as usize),
            None => AIR,
        }
    };

    let mut rgb = vec![0u8; width * height * 3];
    let mut depth = vec![0f32; width * height];
    let mut seg = vec![0u16; width * height];

    rgb.par_chunks_mut(width * 3)
        .zip(depth.par_chunks_mut(width))
        .zip(seg.par_chunks_mut(width))
        .enumerate()
        .for_each(|(py, ((rgb_row, dep_row), seg_row))| {
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
                let hit = dda(eye, dir, max_dist, &get);
                match hit {
                    Some(h) => {
                        let base = block_def(cell_id(h.cell)).color;
                        let shade = face_shade(h.face);
                        rgb_row[px * 3] = (base[0] as f64 * shade) as u8;
                        rgb_row[px * 3 + 1] = (base[1] as f64 * shade) as u8;
                        rgb_row[px * 3 + 2] = (base[2] as f64 * shade) as u8;
                        dep_row[px] = (h.dist * dl) as f32; // cells along the unit ray
                        seg_row[px] = cell_id(h.cell);
                    }
                    None => {
                        rgb_row[px * 3] = SKY_COLOR[0];
                        rgb_row[px * 3 + 1] = SKY_COLOR[1];
                        rgb_row[px * 3 + 2] = SKY_COLOR[2];
                        dep_row[px] = max_dist as f32;
                        seg_row[px] = SKY_SEG;
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
    }
}

#[cfg(test)]
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
        for (px, py) in [(64, 100), (10, 80), (120, 70), (64, 120), (0, 64), (127, 127)] {
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
            assert_eq!(f.depth[py * 128 + px], expected, "depth bitwise (px={px},py={py})");
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
        assert!((3.0..=6.0).contains(&f.depth[i]), "depth {} in range", f.depth[i]);
    }
}
