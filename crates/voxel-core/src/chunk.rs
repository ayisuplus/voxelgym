//! Chunk storage: 16(x) x H(y) x 16(z) direct u16 array, no palette.
//!
//! H = 128 at scale 1 (1 m cells). The world scale knob (0.5 m cells at
//! scale 2) doubles H so the PHYSICAL height stays constant — the index
//! math below is H-free: idx = y*256 + z*16 + x.

pub const CHUNK_X: usize = 16;
/// Scale-1 chunk height in cells (128 m at 1 m cells).
pub const CHUNK_Y: usize = 128;
pub const CHUNK_Z: usize = 16;
pub const CHUNK_VOL: usize = CHUNK_X * CHUNK_Y * CHUNK_Z; // 32768

pub const WORLD_MIN_Y: i32 = 0;
/// Scale-1 max y (127). At scale S the world height is 128*S cells.
pub const WORLD_MAX_Y: i32 = 127;
/// Scale-1 sea level (62). Multiply by the world scale.
pub const SEA_LEVEL: i32 = 62;

#[derive(Clone)]
pub struct Chunk {
    /// idx = y*256 + z*16 + x (H-free); len = 16*H*16.
    pub blocks: Vec<u16>,
    /// True once worldgen (preset + scenario) has been applied.
    pub generated: bool,
    /// Height in cells (128 * world scale).
    pub h: usize,
    /// True once any cell diverged from pristine generation via
    /// `World::set_block`. `World::hash` re-diffs only touched chunks
    /// against worldgen; untouched chunks contribute nothing by definition.
    pub touched: bool,
}

impl Chunk {
    pub fn empty() -> Self {
        Self::with_height(CHUNK_Y)
    }

    pub fn with_height(h: usize) -> Self {
        Chunk {
            blocks: vec![0u16; CHUNK_X * h * CHUNK_Z],
            generated: false,
            h,
            touched: false,
        }
    }

    #[inline]
    pub const fn idx(lx: usize, y: usize, lz: usize) -> usize {
        y * 256 + lz * 16 + lx
    }

    #[inline]
    pub fn get(&self, lx: usize, y: usize, lz: usize) -> u16 {
        self.blocks[Self::idx(lx, y, lz)]
    }

    #[inline]
    pub fn set(&mut self, lx: usize, y: usize, lz: usize, cell: u16) {
        self.blocks[Self::idx(lx, y, lz)] = cell;
    }
}
