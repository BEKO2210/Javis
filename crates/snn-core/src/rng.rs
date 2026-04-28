//! Tiny deterministic PRNG (PCG-style LCG variant) used for tests and
//! Poisson input. Pulls in zero dependencies and is fully reproducible
//! given the same seed.

#[derive(Debug, Clone, Copy)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(0x9E3779B97F4A7C15) }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform `f32` in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Uniform `f32` in [lo, hi).
    pub fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// Bernoulli trial with probability `p`.
    pub fn bernoulli(&mut self, p: f32) -> bool {
        self.next_f32() < p
    }
}
