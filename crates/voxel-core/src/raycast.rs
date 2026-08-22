//! Amanatides & Woo DDA voxel traversal for targeting (max reach 4.5 cells).
//! Stops at any cell where `blocks_ray(cell)` is true.

use crate::block::*;

pub const REACH: f64 = 4.5;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RayHit {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub cell: u16,
    /// Ray parameter t at entry (units of cells along the direction vector).
    pub dist: f64,
    /// Normal of the face entered (unit axis).
    pub face: [i32; 3],
}

/// DDA from `origin` along unit `dir`, up to `max_dist` cells.
/// `get` fetches raw cells; `hit` decides whether a cell stops the ray.
/// Splitting the hit predicate out lets targeting ignore fluids while the
/// renderer keeps them opaque.
pub fn dda_with<G: FnMut(i32, i32, i32) -> u16, H: Fn(u16) -> bool>(
    origin: [f64; 3],
    dir: [f64; 3],
    max_dist: f64,
    mut get: G,
    hit: H,
) -> Option<RayHit> {
    let mut x = origin[0].floor() as i32;
    let mut y = origin[1].floor() as i32;
    let mut z = origin[2].floor() as i32;

    let step_x = if dir[0] > 0.0 { 1 } else { -1 };
    let step_y = if dir[1] > 0.0 { 1 } else { -1 };
    let step_z = if dir[2] > 0.0 { 1 } else { -1 };

    let inf = f64::INFINITY;
    let t_delta_x = if dir[0] != 0.0 { (1.0 / dir[0]).abs() } else { inf };
    let t_delta_y = if dir[1] != 0.0 { (1.0 / dir[1]).abs() } else { inf };
    let t_delta_z = if dir[2] != 0.0 { (1.0 / dir[2]).abs() } else { inf };

    let mut t_max_x = if dir[0] != 0.0 {
        let bound = if step_x > 0 { x as f64 + 1.0 } else { x as f64 };
        (bound - origin[0]) / dir[0]
    } else {
        inf
    };
    let mut t_max_y = if dir[1] != 0.0 {
        let bound = if step_y > 0 { y as f64 + 1.0 } else { y as f64 };
        (bound - origin[1]) / dir[1]
    } else {
        inf
    };
    let mut t_max_z = if dir[2] != 0.0 {
        let bound = if step_z > 0 { z as f64 + 1.0 } else { z as f64 };
        (bound - origin[2]) / dir[2]
    } else {
        inf
    };

    // Origin cell first (face = zero: entered from inside).
    let cell = get(x, y, z);
    if hit(cell) {
        return Some(RayHit { x, y, z, cell, dist: 0.0, face: [0, 0, 0] });
    }

    #[allow(unused_assignments)]
    let mut t = 0.0f64;
    #[allow(unused_assignments)]
    let mut face = [0i32; 3];
    loop {
        if t_max_x <= t_max_y && t_max_x <= t_max_z {
            t = t_max_x;
            t_max_x += t_delta_x;
            x += step_x;
            face = [-step_x, 0, 0];
        } else if t_max_y <= t_max_z {
            t = t_max_y;
            t_max_y += t_delta_y;
            y += step_y;
            face = [0, -step_y, 0];
        } else {
            t = t_max_z;
            t_max_z += t_delta_z;
            z += step_z;
            face = [0, 0, -step_z];
        }
        if t > max_dist {
            return None;
        }
        let cell = get(x, y, z);
        if hit(cell) {
            return Some(RayHit { x, y, z, cell, dist: t, face });
        }
    }
}

/// Default traversal: every non-air cell stops the ray (renderer policy).
pub fn dda<F: FnMut(i32, i32, i32) -> u16>(
    origin: [f64; 3],
    dir: [f64; 3],
    max_dist: f64,
    get: F,
) -> Option<RayHit> {
    dda_with(origin, dir, max_dist, get, blocks_ray)
}

/// Targeting policy: fluids never stop the mining/use ray (MC behavior;
/// also prevents unrecoverable submerged states). The renderer keeps
/// fluids opaque.
pub fn blocks_target(cell: u16) -> bool {
    let id = cell_id(cell);
    id != AIR && block_def(id).fluid.is_none()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::World;
    use crate::worldgen::Preset;

    /// Brute-force reference: sample the ray densely, report the first
    /// non-air cell entered. Only meaningful away from exact face ties.
    fn brute(world: &mut World, origin: [f64; 3], dir: [f64; 3], max_dist: f64) -> Option<(i32, i32, i32, u16)> {
        let steps = (max_dist * 1000.0) as usize;
        let mut last = (
            origin[0].floor() as i32,
            origin[1].floor() as i32,
            origin[2].floor() as i32,
        );
        for i in 0..=steps {
            let t = i as f64 / 1000.0;
            let p = [
                origin[0] + dir[0] * t,
                origin[1] + dir[1] * t,
                origin[2] + dir[2] * t,
            ];
            let c = (p[0].floor() as i32, p[1].floor() as i32, p[2].floor() as i32);
            if c != last {
                let cell = world.get_block(c.0, c.1, c.2);
                if blocks_ray(cell) {
                    return Some((c.0, c.1, c.2, cell));
                }
                last = c;
            } else {
                let cell = world.get_block(c.0, c.1, c.2);
                if blocks_ray(cell) {
                    return Some((c.0, c.1, c.2, cell));
                }
            }
        }
        None
    }

    #[test]
    fn dda_matches_bruteforce() {
        let mut world = World::new(777, Preset::Default, Vec::new());
        let mut rng = crate::rng::Rng::new(1234, 9);
        let mut checked = 0;
        let mut attempts = 0;
        while checked < 1000 && attempts < 20000 {
            attempts += 1;
            let ox = (rng.next_f64() - 0.5) * 60.0;
            let oy = 55.0 + rng.next_f64() * 30.0;
            let oz = (rng.next_f64() - 0.5) * 60.0;
            // random direction, avoid near-axis ties that make the
            // brute-force reference ambiguous at face boundaries
            let theta = rng.next_f64() * std::f64::consts::TAU;
            let phi = (rng.next_f64() * 2.0 - 1.0).acos();
            let dir = [
                phi.sin() * theta.cos(),
                phi.cos(),
                phi.sin() * theta.sin(),
            ];
            if dir.iter().any(|d| d.abs() < 0.05) {
                continue; // skip near-axis directions (tie-prone)
            }
            let origin = [ox, oy, oz];
            let got = dda(origin, dir, REACH, |x, y, z| world.get_block(x, y, z));
            let want = brute(&mut world, origin, dir, REACH);
            match (got, want) {
                (Some(g), Some(w)) => {
                    assert_eq!((g.x, g.y, g.z), (w.0, w.1, w.2), "origin={origin:?} dir={dir:?}");
                    assert_eq!(cell_id(g.cell), cell_id(w.3));
                }
                (None, None) => {}
                (g, w) => panic!("mismatch dda={g:?} brute={w:?} origin={origin:?} dir={dir:?}"),
            }
            checked += 1;
        }
        assert_eq!(checked, 1000);
    }

    #[test]
    fn reach_limit() {
        let mut world = World::new(1, Preset::Flat, Vec::new());
        // eye 6.62 above surface plane y=5.0; floor cells y=4
        let origin = [8.5, 6.62, 8.5];
        let hit = dda(origin, [0.0, -1.0, 0.0], REACH, |x, y, z| world.get_block(x, y, z));
        assert!(hit.is_some());
        let h = hit.unwrap();
        assert_eq!(h.y, 4);
        assert_eq!(h.face, [0, 1, 0]);
        assert!((h.dist - 1.62).abs() < 1e-9, "dist={}", h.dist);
        // horizontal: nothing within reach on flat world
        let miss = dda(origin, [1.0, 0.0, 0.0], REACH, |x, y, z| world.get_block(x, y, z));
        assert!(miss.is_none());
    }
}
