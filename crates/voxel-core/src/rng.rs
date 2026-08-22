//! Deterministic RNG: PCG64 (LCG mod 2^128, XSL-RR output permutation).
//!
//! Hand-rolled (instead of `rand_pcg`) so the full state is a single `u128`
//! that round-trips through snapshots byte-exactly — the determinism contract
//! requires snapshot/restore to preserve every random stream.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng {
    state: u128,
    inc: u128,
}

const MULT: u128 = 47026247687942121848144207491837523525;

impl Rng {
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut rng = Rng {
            state: 0,
            inc: ((stream as u128) << 1) | 1,
        };
        rng.next_u64();
        rng.state = rng.state.wrapping_add(seed as u128);
        rng.next_u64();
        rng
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(MULT).wrapping_add(self.inc);
        let xorshifted = (((self.state >> 64) ^ self.state) >> 64) as u64;
        let rot = (self.state >> 122) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [0, n).
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// Full state for snapshots.
    pub fn state(&self) -> (u128, u128) {
        (self.state, self.inc)
    }

    pub fn from_state(state: u128, inc: u128) -> Self {
        Rng { state, inc }
    }
}

/// Position-hash (splitmix64 mix) for deterministic per-column decisions
/// (tree placement). Independent of the sequential RNG so worldgen of one
/// chunk never depends on how many other chunks were generated before it.
pub fn hash2(seed: u64, x: i32, z: i32) -> u64 {
    let mut h = seed ^ 0x9E37_79B9_7F4A_7C15;
    h = h.wrapping_add(x as i64 as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 29;
    h = h.wrapping_add(z as i64 as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 32;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^ (h >> 31)
}

/// 4-input position hash (x, y, z, salt/tick bucket) for spread/ignite
/// decisions that must not consume the sequential RNG stream.
pub fn hash_pos(seed: u64, x: i32, y: i32, z: i32, salt: u64) -> u64 {
    let mut h = seed ^ 0x243F_6A88_85A3_08D3;
    h = h.wrapping_add(x as i64 as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 29;
    h = h.wrapping_add(y as i64 as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 32;
    h = h.wrapping_add(z as i64 as u64).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 29;
    h = h.wrapping_add(salt).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^ (h >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_sequence() {
        let mut a = Rng::new(42, 1);
        let mut b = Rng::new(42, 1);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn state_roundtrip() {
        let mut a = Rng::new(7, 3);
        for _ in 0..17 {
            a.next_u64();
        }
        let (s, i) = a.state();
        let mut b = Rng::from_state(s, i);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
