//! Spinning multi-beam LiDAR model over the same DDA core as the renderer.
//!
//! One DDA ray per (elevation channel, azimuth step) — a scan IS a camera
//! whose film is a cylinder: the range image is the standard H_channels x
//! W_azimuth grid consumed by RangeNet++/PointPillars-style perception.
//! Range, surface material (seg id) and incidence normal come out of the
//! same traversal, so ground truth is exact BY CONSTRUCTION and
//! bit-identical per platform.
//!
//! Physical flavor (documented simplifications):
//! - intensity = block albedo (registry color luminance) x |cos incidence|
//!   / (1 + k*r^2) — inverse-square-ish falloff, no multi-return, no beam
//!   divergence cone.
//! - optional per-beam noise: Gaussian range noise + random dropout,
//!   seeded by (noise_seed, frame_idx, beam) position hashing — a scan is
//!   a pure function of (world state, config, frame index), so replays
//!   reproduce byte-identical range images.
//! - no rolling-shutter motion distortion: the sim is tick-synchronous.

use rayon::prelude::*;
use voxel_core::block::*;
use voxel_core::raycast::dda;
use voxel_core::rng::hash_pos;
use voxel_core::World;

use crate::{ChunkGrid, SKY_SEG};

#[derive(Clone, Copy, Debug)]
pub struct LidarConfig {
    /// Elevation beams (16 = VLP-16-like, 64 = HDL-64-like).
    pub channels: usize,
    /// Azimuth steps per full rotation (column count of the range image).
    pub azimuth_steps: usize,
    /// Elevation span, degrees; positive = up. Row 0 = min_elev.
    pub min_elev_deg: f64,
    pub max_elev_deg: f64,
    /// Max range in cells; misses return range 0 / SKY_SEG / intensity 0.
    pub max_range: f64,
    /// Gaussian range noise stddev in cells; 0.0 = exact.
    pub noise_sigma: f64,
    /// Per-beam dropout probability in [0, 1): the beam returns nothing.
    pub dropout_p: f64,
    /// Noise stream seed (combined with frame_idx + beam index).
    pub noise_seed: u64,
}

impl Default for LidarConfig {
    /// VLP-16-ish: 16 channels, ±15 deg, 512 azimuths, 64-cell range.
    fn default() -> Self {
        LidarConfig {
            channels: 16,
            azimuth_steps: 512,
            min_elev_deg: -15.0,
            max_elev_deg: 15.0,
            max_range: 64.0,
            noise_sigma: 0.0,
            dropout_p: 0.0,
            noise_seed: 0,
        }
    }
}

pub struct Scan {
    pub channels: usize,
    pub azimuth_steps: usize,
    /// [C, A] row-major; 0.0 = no return.
    pub range: Vec<f32>,
    /// [C, A] 0..=1.
    pub intensity: Vec<f32>,
    /// [C, A] block id; SKY_SEG on miss.
    pub seg: Vec<u16>,
}

/// Beam direction: azimuth 0 faces the sensor yaw (MC convention:
/// 0 deg = +z, 90 = -x); elevation positive up.
pub fn beam_dir(yaw_deg: f64, azimuth_deg: f64, elev_deg: f64) -> [f64; 3] {
    let az = (yaw_deg + azimuth_deg).to_radians();
    let el = elev_deg.to_radians();
    [-az.sin() * el.cos(), el.sin(), az.cos() * el.cos()]
}

/// Deterministic per-beam Gaussian (Box-Muller over position hashes).
fn beam_noise(cfg: &LidarConfig, frame_idx: u64, beam: usize) -> (f64, f64) {
    let h1 = hash_pos(cfg.noise_seed, beam as i32, frame_idx as i32, 0, 71);
    let h2 = hash_pos(cfg.noise_seed, beam as i32, frame_idx as i32, 0, 72);
    let u1 = ((h1 >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64); // (0,1]
    let u2 = (h2 >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos(), u2)
}

/// Full scan from `origin` (typically the agent eye, but any pose works —
/// a fixed emitter block is just a constant origin/yaw).
pub fn scan(
    world: &mut World,
    cfg: &LidarConfig,
    origin: [f64; 3],
    yaw_deg: f64,
    frame_idx: u64,
) -> Scan {
    let grid = ChunkGrid::new(world, origin);
    let max_range = cfg.max_range.min(grid.max_dist());

    let (c_n, a_n) = (cfg.channels, cfg.azimuth_steps);
    let mut range = vec![0f32; c_n * a_n];
    let mut intensity = vec![0f32; c_n * a_n];
    let mut seg = vec![SKY_SEG; c_n * a_n];

    range
        .par_chunks_mut(a_n)
        .zip(intensity.par_chunks_mut(a_n))
        .zip(seg.par_chunks_mut(a_n))
        .enumerate()
        .for_each(|(c, ((r_row, i_row), s_row))| {
            let elev = if c_n > 1 {
                cfg.min_elev_deg + (cfg.max_elev_deg - cfg.min_elev_deg) * (c as f64 / (c_n - 1) as f64)
            } else {
                cfg.min_elev_deg
            };
            for a in 0..a_n {
                let az_deg = 360.0 * (a as f64 / a_n as f64);
                let dir = beam_dir(yaw_deg, az_deg, elev);
                let beam = c * a_n + a;
                let (noise, u2) = if cfg.noise_sigma > 0.0 || cfg.dropout_p > 0.0 {
                    beam_noise(cfg, frame_idx, beam)
                } else {
                    (0.0, 1.0)
                };
                if cfg.dropout_p > 0.0 && u2 < cfg.dropout_p {
                    continue; // no return: zeros/SKY already in place
                }
                if let Some(h) = dda(origin, dir, max_range, |x, y, z| grid.get(x, y, z)) {
                    let mut r = h.dist;
                    if cfg.noise_sigma > 0.0 {
                        r = (r + noise * cfg.noise_sigma).max(0.0);
                    }
                    r_row[a] = r as f32;
                    s_row[a] = cell_id(h.cell);
                    // albedo x |cos incidence| / (1 + k r^2)
                    let col = block_def(cell_id(h.cell)).color;
                    let albedo = (0.299 * col[0] as f64 + 0.587 * col[1] as f64 + 0.114 * col[2] as f64)
                        / 255.0;
                    let cos_i = (dir[0] * h.face[0] as f64
                        + dir[1] * h.face[1] as f64
                        + dir[2] * h.face[2] as f64)
                    .abs();
                    i_row[a] = (albedo * cos_i / (1.0 + 0.02 * r * r)).clamp(0.0, 1.0) as f32;
                }
            }
        });

    Scan {
        channels: c_n,
        azimuth_steps: a_n,
        range,
        intensity,
        seg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::worldgen::Preset;
    use voxel_core::World;

    fn cfg4x8() -> LidarConfig {
        LidarConfig {
            channels: 4,
            azimuth_steps: 8,
            min_elev_deg: -5.0,
            max_elev_deg: 40.0,
            max_range: 64.0,
            noise_sigma: 0.0,
            dropout_p: 0.0,
            noise_seed: 0,
        }
    }

    /// Flat world with a stone wall plane at x=10 (y 5..8, z -2..2);
    /// sensor at (5.5, 6.5, 0.5) facing +x (MC yaw 270).
    fn wall_world() -> World {
        let mut w = World::new(1, Preset::Flat, Vec::new());
        for y in 5..=8 {
            for z in -2..=2 {
                w.set_block(10, y, z, STONE);
            }
        }
        w
    }

    #[test]
    fn golden_wall_range_and_seg() {
        let mut w = wall_world();
        let origin = [5.5, 6.5, 0.5];
        // channels -5..40 deg: row 0 (-5) and row 1 (~10) hit the wall
        // (top y=8 vs ray height 6.5 + 4.5*tan(10deg) = 7.29 < 8.5 face);
        // rows 2-3 clear the top -> sky.
        let s = scan(&mut w, &cfg4x8(), origin, 270.0, 0);
        let a_n = 8;
        // azimuth 0 = +x straight at the wall: exact DDA range 4.5
        let r0 = s.range[0 * a_n] as f64;
        assert!((r0 - 4.5 / 5f64.to_radians().cos()).abs() < 1e-5, "row0 range {}", r0);
        assert_eq!(s.seg[0], STONE as u16);
        let r1 = s.range[a_n] as f64;
        let el1: f64 = -5.0 + 45.0 / 3.0;
        let want = 4.5 / (el1.to_radians().cos());
        assert!((r1 - want).abs() < 1e-4, "row1 range {} want {}", r1, want);
        assert_eq!(s.seg[a_n], STONE as u16);
        // row 2 (25 deg) grazes the wall's top cell (6.5 + 4.5*tan25 =
        // 8.6 < 9.0); row 3 (40 deg) clears it -> sky
        let r2 = s.range[2 * a_n] as f64;
        let el2 = 25f64.to_radians();
        assert!((r2 - 4.5 / el2.cos()).abs() < 1e-4, "row2 range {}", r2);
        assert_eq!(s.seg[2 * a_n], STONE as u16);
        assert_eq!(s.range[3 * a_n], 0.0, "row3 sky");
        assert_eq!(s.seg[3 * a_n], SKY_SEG);
        // intensity sane on a hit
        assert!(s.intensity[0] > 0.0 && s.intensity[0] <= 1.0);
    }

    #[test]
    fn scans_are_deterministic_with_noise() {
        let cfg = LidarConfig {
            noise_sigma: 0.05,
            dropout_p: 0.1,
            noise_seed: 7,
            ..cfg4x8()
        };
        let mut w = wall_world();
        let origin = [5.5, 6.5, 0.5];
        let a = scan(&mut w, &cfg, origin, 270.0, 3);
        let b = scan(&mut w, &cfg, origin, 270.0, 3);
        assert_eq!(a.range, b.range, "same frame_idx -> same noise");
        assert_eq!(a.seg, b.seg);
        let c = scan(&mut w, &cfg, origin, 270.0, 4);
        assert_ne!(a.range, c.range, "next frame -> fresh noise draw");
    }

    #[test]
    fn dropout_one_kills_every_beam() {
        let cfg = LidarConfig {
            dropout_p: 1.0,
            noise_seed: 1,
            ..cfg4x8()
        };
        let mut w = wall_world();
        let s = scan(&mut w, &cfg, [5.5, 6.5, 0.5], 270.0, 0);
        assert!(s.range.iter().all(|&r| r == 0.0));
        assert!(s.seg.iter().all(|&v| v == SKY_SEG));
    }

    #[test]
    fn exact_mode_is_exact() {
        let mut w = wall_world();
        let s = scan(&mut w, &cfg4x8(), [5.5, 6.5, 0.5], 270.0, 0);
        // DDA ray parameter = euclidean distance for unit dirs
        let exact = 4.5 / 5f64.to_radians().cos();
        assert!((s.range[0] as f64 - exact).abs() < 1e-6);
    }
}
