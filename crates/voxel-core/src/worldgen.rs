//! World generation: `default` / `flat` / `void` presets + ScenarioSpec overlay.
//!
//! Determinism: all noise from `noise` crate Fbm<OpenSimplex> seeded from the
//! world seed; all per-column randomness from position hashing (`rng::hash2`),
//! never from sequential RNG — chunk generation is order-independent.

use noise::{Fbm, MultiFractal, NoiseFn, OpenSimplex};

use crate::block::*;
use crate::chunk::*;
use crate::rng::hash2;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Preset {
    Default,
    Flat,
    Void,
}

impl Preset {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "default" => Some(Preset::Default),
            "flat" => Some(Preset::Flat),
            "void" => Some(Preset::Void),
            _ => None,
        }
    }

    pub fn as_u8(self) -> u8 {
        match self {
            Preset::Default => 0,
            Preset::Flat => 1,
            Preset::Void => 2,
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Preset::Default),
            1 => Some(Preset::Flat),
            2 => Some(Preset::Void),
            _ => None,
        }
    }
}

/// Inclusive box region.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Region {
    pub x0: i32,
    pub y0: i32,
    pub z0: i32,
    pub x1: i32,
    pub y1: i32,
    pub z1: i32,
}

impl Region {
    pub fn new(x0: i32, y0: i32, z0: i32, x1: i32, y1: i32, z1: i32) -> Self {
        Region {
            x0: x0.min(x1),
            y0: y0.min(y1),
            z0: z0.min(z1),
            x1: x0.max(x1),
            y1: y0.max(y1),
            z1: z0.max(z1),
        }
    }

    pub fn contains(&self, x: i32, y: i32, z: i32) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1 && z >= self.z0 && z <= self.z1
    }
}

/// ScenarioSpec = list of (region, raw cell) to stamp over the preset terrain.
pub type ScenarioSpec = Vec<(Region, u16)>;

/// Surface height (grass y) of the default terrain at world column (x, z),
/// before biome adjustment.
pub fn default_height(seed: u64, x: i32, z: i32) -> i32 {
    let fbm = height_noise(seed);
    let v = fbm.get([x as f64 / 64.0, z as f64 / 64.0]);
    (64.0 + v * 16.0).floor() as i32
}

// ---------------------------------------------------------------- biomes --

/// Terrain biomes. Vertical layering: sky (air) above the surface,
/// surface zone, underground (stone + caves below the top 3 cells).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Biome {
    Ocean,
    Plains,
    Desert,
    Hills,
    Volcanic,
}

impl Biome {
    pub fn as_u8(self) -> u8 {
        match self {
            Biome::Ocean => 0,
            Biome::Plains => 1,
            Biome::Desert => 2,
            Biome::Hills => 3,
            Biome::Volcanic => 4,
        }
    }
}

fn biome_noise(seed: u64) -> Fbm<OpenSimplex> {
    Fbm::<OpenSimplex>::new(((seed ^ 0x51ED_275B_A35A_9E11) as u32)).set_octaves(2)
}

/// Biome at column (x, z). Thresholds tuned on the measured 2-octave FBM
/// distribution (p17/p50/p73/p92): ~17% ocean, 33% plains, 23% desert,
/// 19% hills, ~8% volcanic.
pub fn biome_at(seed: u64, x: i32, z: i32) -> Biome {
    let v = biome_noise(seed).get([x as f64 / 96.0, z as f64 / 96.0]);
    if v < -0.16 {
        Biome::Ocean
    } else if v < 0.007 {
        Biome::Plains
    } else if v < 0.20 {
        Biome::Desert
    } else if v < 0.31 {
        Biome::Hills
    } else {
        Biome::Volcanic
    }
}

fn cave_noise(seed: u64) -> Fbm<OpenSimplex> {
    Fbm::<OpenSimplex>::new(((seed ^ 0x73BD_A5D1_9E42_6C77) as u32)).set_octaves(2)
}

/// 3D cave field: carve where the scaled noise exceeds CAVE_T (measured
/// p96-ish → a few percent of underground cells, forming connected worms).
const CAVE_T: f64 = 0.17;
/// Deep-cave lava: carved cells at/below this y with solid below may become
/// lava sources (hash chance).
const CAVE_LAVA_Y: i32 = 10;

fn height_noise(seed: u64) -> Fbm<OpenSimplex> {
    Fbm::<OpenSimplex>::new((seed & 0xFFFF_FFFF) as u32)
        .set_octaves(4)
        .set_frequency(1.0)
}

fn ore_noise(seed: u64, salt: u32) -> Fbm<OpenSimplex> {
    Fbm::<OpenSimplex>::new(((seed ^ 0x9E37_79B9_7F4A_7C15) as u32).wrapping_add(salt))
        .set_octaves(3)
        .set_frequency(1.0)
}

/// The contract thresholds (0.72/0.78/0.85) assume noise spanning ~[-1, 1].
/// Measured Fbm<OpenSimplex> 3-octave range is only +-0.46 (p98 = 0.246,
/// p99.5 = 0.292), so samples are scaled up to make the literal thresholds
/// meaningful: coal ~p98, iron ~p99.3, diamond ~p99.5 of stone cells.
pub const ORE_NOISE_SCALE: f64 = 2.927;

/// Generate one chunk column for the given preset. Scenario overlay is applied
/// separately by the caller (`World::ensure_chunk`).
pub fn generate_chunk(seed: u64, preset: Preset, cx: i32, cz: i32) -> Chunk {
    let mut chunk = Chunk::empty();
    match preset {
        Preset::Void => {}
        Preset::Flat => {
            // bedrock y0, dirt y1..=3, grass y4 (literal contract stack)
            for lz in 0..16 {
                for lx in 0..16 {
                    chunk.set(lx, 0, lz, BEDROCK);
                    for y in 1..=3 {
                        chunk.set(lx, y, lz, DIRT);
                    }
                    chunk.set(lx, 4, lz, GRASS_BLOCK);
                }
            }
        }
        Preset::Default => gen_default(seed, cx, cz, &mut chunk),
    }
    chunk.generated = true;
    chunk
}

fn gen_default(seed: u64, cx: i32, cz: i32, chunk: &mut Chunk) {
    let coal = ore_noise(seed, 11);
    let iron = ore_noise(seed, 23);
    let diamond = ore_noise(seed, 37);
    let caves = cave_noise(seed);

    for lz in 0..16 {
        for lx in 0..16 {
            let x = cx * 16 + lx as i32;
            let z = cz * 16 + lz as i32;
            let biome = biome_at(seed, x, z);
            let h_raw = default_height(seed, x, z);
            let h = match biome {
                Biome::Ocean => h_raw - 9, // deep basin
                Biome::Hills => 64 + ((h_raw - 64) as f64 * 1.8) as i32, // amplified relief
                _ => h_raw,
            }
            .clamp(6, WORLD_MAX_Y - 1);

            // slope for exposed-rock decision (hills)
            let slope = if biome == Biome::Hills {
                let hx = default_height(seed, x + 1, z);
                let hz = default_height(seed, x, z + 1);
                ((hx - h_raw).abs() + (hz - h_raw).abs()) as f64 * 1.8
            } else {
                0.0
            };

            // --- base layers ---
            chunk.set(lx, 0, lz, BEDROCK);
            for y in 1..=h {
                let id = if y <= h - 4 {
                    STONE
                } else if y <= h - 1 {
                    match biome {
                        Biome::Desert => SAND,
                        _ => DIRT,
                    }
                } else {
                    match biome {
                        Biome::Ocean | Biome::Desert => SAND,
                        Biome::Hills => {
                            if slope > 2.5 {
                                STONE // exposed rock on steep slopes
                            } else {
                                GRASS_BLOCK
                            }
                        }
                        Biome::Volcanic => {
                            if hash2(seed ^ 0xCAFE, x, z) % 1000 < 300 {
                                COBBLESTONE
                            } else {
                                GRASS_BLOCK
                            }
                        }
                        Biome::Plains => GRASS_BLOCK,
                    }
                };
                chunk.set(lx, y as usize, lz, id);
            }

            // --- ores: 3D noise threshold, replace stone only ---
            for y in 1..=h {
                let p = [x as f64 * 0.15, y as f64 * 0.15, z as f64 * 0.15];
                let cur = chunk.get(lx, y as usize, lz);
                if cur != STONE {
                    continue;
                }
                if (5..=90).contains(&y) && coal.get(p) * ORE_NOISE_SCALE > 0.72 {
                    chunk.set(lx, y as usize, lz, COAL_ORE);
                } else if (5..=48).contains(&y) && iron.get(p) * ORE_NOISE_SCALE > 0.78 {
                    chunk.set(lx, y as usize, lz, IRON_ORE);
                } else if (1..=16).contains(&y) && diamond.get(p) * ORE_NOISE_SCALE > 0.85 {
                    chunk.set(lx, y as usize, lz, DIAMOND_ORE);
                }
            }

            // --- caves: carve underground, never below y=3 (solid floor) ---
            for y in 3..=(h - 4).max(3) {
                if y > h - 4 {
                    break;
                }
                let v = caves.get([x as f64 / 18.0, y as f64 / 18.0, z as f64 / 18.0]);
                if v > CAVE_T {
                    chunk.set(lx, y as usize, lz, AIR);
                }
            }
            // deep-cave lava pockets: carved floor cells at y<=10
            if h > 8 {
                for y in 3..=CAVE_LAVA_Y.min(h - 4) {
                    let cur = chunk.get(lx, y as usize, lz);
                    if cur == AIR && chunk.get(lx, y as usize - 1, lz) != AIR {
                        if hash2(seed ^ 0x1A1A, x, z) % 1000 < 220 {
                            chunk.set(lx, y as usize, lz, LAVA);
                        }
                    }
                }
            }

            // --- volcanic surface lava pools ---
            if biome == Biome::Volcanic && hash2(seed ^ 0xF1AE, x, z) % 1000 < 6 {
                // 1-deep basin of lava at the surface
                chunk.set(lx, h as usize, lz, LAVA);
                if h + 1 <= WORLD_MAX_Y {
                    chunk.set(lx, (h + 1) as usize, lz, COBBLESTONE);
                }
            }

            // --- water fill & beaches ---
            let near_water = h < SEA_LEVEL
                || default_height(seed, x + 1, z) < SEA_LEVEL
                || default_height(seed, x - 1, z) < SEA_LEVEL
                || default_height(seed, x, z + 1) < SEA_LEVEL
                || default_height(seed, x, z - 1) < SEA_LEVEL;
            if biome != Biome::Desert && biome != Biome::Volcanic && h <= SEA_LEVEL + 2 && near_water {
                let surf = chunk.get(lx, h as usize, lz);
                if surf == GRASS_BLOCK || surf == DIRT {
                    chunk.set(lx, h as usize, lz, SAND);
                }
            }
            if h < SEA_LEVEL {
                for y in (h + 1)..=SEA_LEVEL {
                    let cur = chunk.get(lx, y as usize, lz);
                    if cur == AIR {
                        chunk.set(lx, y as usize, lz, WATER); // state 0 = source
                    }
                }
            }

            // --- trees: plains only ---
            let th = hash2(seed, x, z);
            if biome == Biome::Plains && th % 1000 < 20 && h > SEA_LEVEL {
                let surf = chunk.get(lx, h as usize, lz);
                if surf == GRASS_BLOCK {
                    place_tree(chunk, lx, h + 1, lz, 4 + (th >> 20) % 3);
                }
            }
        }
    }
}

/// Trunk of `height` logs at (lx, y0, lz); leaf ball radius 2 around top.
/// Writes are clamped to this chunk — cross-border parts are dropped, which
/// is deterministic (same for any generation order).
fn place_tree(chunk: &mut Chunk, lx: usize, y0: i32, lz: usize, height: u64) {
    let top = y0 + height as i32 - 1;
    for dy in -2i32..=2 {
        for dx in -2i32..=2 {
            for dz in -2i32..=2 {
                if dx * dx + dy * dy + dz * dz > 6 {
                    continue;
                }
                let (x, y, z) = (lx as i32 + dx, top + dy, lz as i32 + dz);
                if !(0..16).contains(&x) || !(0..16).contains(&z) || !(0..128).contains(&y) {
                    continue;
                }
                let cur = chunk.get(x as usize, y as usize, z as usize);
                if cell_id(cur) == AIR {
                    chunk.set(x as usize, y as usize, z as usize, LEAVES);
                }
            }
        }
    }
    for y in y0..=top {
        if (0..128).contains(&y) {
            chunk.set(lx, y as usize, lz, LOG);
        }
    }
}

/// Apply the scenario overlay to a freshly generated chunk.
pub fn apply_scenario(chunk: &mut Chunk, cx: i32, cz: i32, scenario: &ScenarioSpec) {
    for (region, cell) in scenario {
        let x0 = region.x0.max(cx * 16);
        let x1 = region.x1.min(cx * 16 + 15);
        let z0 = region.z0.max(cz * 16);
        let z1 = region.z1.min(cz * 16 + 15);
        if x0 > x1 || z0 > z1 {
            continue;
        }
        for z in z0..=z1 {
            for x in x0..=x1 {
                for y in region.y0..=region.y1 {
                    if (0..128).contains(&y) {
                        chunk.set(
                            (x - cx * 16) as usize,
                            y as usize,
                            (z - cz * 16) as usize,
                            *cell,
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk_bytes(c: &Chunk) -> &[u16; CHUNK_VOL] {
        &c.blocks
    }

    #[test]
    fn same_seed_same_bytes() {
        let a = generate_chunk(42, Preset::Default, 3, -7);
        let b = generate_chunk(42, Preset::Default, 3, -7);
        assert_eq!(chunk_bytes(&a), chunk_bytes(&b));
    }

    #[test]
    fn different_seed_differs() {
        let a = generate_chunk(1, Preset::Default, 0, 0);
        let b = generate_chunk(2, Preset::Default, 0, 0);
        assert_ne!(chunk_bytes(&a), chunk_bytes(&b));
    }

    #[test]
    fn flat_layer_order() {
        let c = generate_chunk(99, Preset::Flat, -2, 5);
        for lz in 0..16 {
            for lx in 0..16 {
                assert_eq!(c.get(lx, 0, lz), BEDROCK);
                for y in 1..=3 {
                    assert_eq!(c.get(lx, y, lz), DIRT);
                }
                assert_eq!(c.get(lx, 4, lz), GRASS_BLOCK);
                for y in 5..128 {
                    assert_eq!(c.get(lx, y, lz), AIR);
                }
            }
        }
    }

    #[test]
    fn void_is_empty() {
        let c = generate_chunk(7, Preset::Void, 0, 0);
        assert!(c.blocks.iter().all(|&b| b == 0));
    }

    #[test]
    fn scenario_stamps() {
        let mut c = generate_chunk(1, Preset::Flat, 0, 0);
        let spec = vec![(Region::new(-1, 5, -1, 20, 8, 30), STONE)];
        apply_scenario(&mut c, 0, 0, &spec);
        assert_eq!(c.get(0, 5, 0), STONE);
        assert_eq!(c.get(15, 8, 15), STONE);
        assert_eq!(c.get(3, 9, 3), AIR); // above region
        assert_eq!(c.get(3, 4, 3), GRASS_BLOCK); // below region
    }

    #[test]
    fn default_has_sea_and_bedrock() {
        // Find a water cell somewhere in a patch of chunks — sea level fill.
        let mut found_water = false;
        'outer: for cx in -3..=3 {
            for cz in -3..=3 {
                let c = generate_chunk(1234, Preset::Default, cx, cz);
                for y in 1..=SEA_LEVEL as usize {
                    for i in 0..256 {
                        let lx = i % 16;
                        let lz = i / 16;
                        if cell_id(c.get(lx, y, lz)) == WATER {
                            found_water = true;
                            break 'outer;
                        }
                    }
                }
            }
        }
        assert!(found_water, "no ocean found in 7x7 chunks");
    }

    #[test]
    fn biomes_all_present_and_stable() {
        // over a 3km sample all five biomes appear; per-column biome is
        // position-deterministic (same call twice -> same biome)
        let mut counts = [0u32; 5];
        for x in (-1500..1500).step_by(16) {
            for z in (-1500..1500).step_by(16) {
                let b = biome_at(42, x, z);
                assert_eq!(b, biome_at(42, x, z));
                counts[b.as_u8() as usize] += 1;
            }
        }
        assert!(counts.iter().all(|&c| c > 100), "biome counts: {counts:?}");
    }

    #[test]
    fn caves_exist_but_never_break_deep_floor() {
        let mut carved = 0u32;
        let mut total_stone = 0u32;
        for cx in 0..4 {
            for cz in 0..4 {
                let c = generate_chunk(9, Preset::Default, cx, cz);
                for i in 0..256 {
                    let lx = i % 16;
                    let lz = i / 16;
                    for y in 3..40usize {
                        let id = cell_id(c.get(lx, y, lz));
                        if id == STONE || id == AIR {
                            total_stone += 1;
                        }
                        if id == AIR {
                            carved += 1;
                        }
                        // bedrock layer and just above it are never carved
                        assert_ne!(cell_id(c.get(lx, 0, lz)), AIR);
                        assert_ne!(cell_id(c.get(lx, 1, lz)), AIR);
                        assert_ne!(cell_id(c.get(lx, 2, lz)), AIR);
                    }
                }
            }
        }
        let frac = carved as f64 / total_stone as f64;
        assert!(frac > 0.005 && frac < 0.20, "cave fraction {frac}");
    }

    #[test]
    fn volcanic_has_surface_lava_and_ocean_is_deep() {
        // find a volcanic column in a wide scan, then check its chunk for lava
        let mut found_lava = false;
        'outer: for x in (-800..800).step_by(16) {
            for z in (-800..800).step_by(16) {
                if biome_at(5, x, z) != Biome::Volcanic {
                    continue;
                }
                let c = generate_chunk(5, Preset::Default, x.div_euclid(16), z.div_euclid(16));
                for i in 0..CHUNK_VOL {
                    if cell_id(c.blocks[i]) == LAVA {
                        found_lava = true;
                        break 'outer;
                    }
                }
            }
        }
        assert!(found_lava, "no volcanic surface lava found");
        // ocean basin: the biome-adjusted floor (h_raw - 9) dips well below
        // sea level somewhere
        let deep = (-800..800).step_by(8).any(|x| {
            (-800..800).step_by(8).any(|z| {
                biome_at(5, x, z) == Biome::Ocean && default_height(5, x, z) - 9 < SEA_LEVEL - 5
            })
        });
        assert!(deep, "no deep ocean basin found");
    }
}


