//! Chunk storage: 16(x) x 128(y) x 16(z) direct u16 array, no palette.

pub const CHUNK_X: usize = 16;
pub const CHUNK_Y: usize = 128;
pub const CHUNK_Z: usize = 16;
pub const CHUNK_VOL: usize = CHUNK_X * CHUNK_Y * CHUNK_Z; // 32768

pub const WORLD_MIN_Y: i32 = 0;
pub const WORLD_MAX_Y: i32 = 127;
pub const SEA_LEVEL: i32 = 62;

#[derive(Clone)]
pub struct Chunk {
    /// idx = y*256 + z*16 + x
    pub blocks: Box<[u16; CHUNK_VOL]>,
    /// True once worldgen (preset + scenario) has been applied.
    pub generated: bool,
}

impl Chunk {
    pub fn empty() -> Self {
        Chunk {
            blocks: Box::new([0u16; CHUNK_VOL]),
            generated: false,
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
