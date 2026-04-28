//! Word-level text encoder.
//!
//! Each token hashes deterministically to exactly `k` distinct bits in
//! `[0, n)`. The text-level SDR is the union of its tokens — so two
//! sentences sharing a word share that word's exact bits, by construction.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::sdr::Sdr;

#[derive(Debug, Clone, Copy)]
pub struct TextEncoder {
    pub n: u32,
    pub k: u32,
}

impl TextEncoder {
    pub fn new(n: u32, k: u32) -> Self {
        assert!(k > 0, "k must be > 0");
        assert!(n > k * 4, "n must be much larger than k for sparsity");
        Self { n, k }
    }

    /// Lowercases, splits on whitespace, strips leading/trailing
    /// non-alphanumerics. Empty tokens are dropped.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|w| {
                w.to_lowercase()
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .filter(|w| !w.is_empty())
            .collect()
    }

    /// Hash one token to exactly `k` distinct indices in `[0, n)`.
    /// Stable across calls within a single build.
    pub fn encode_word(&self, word: &str) -> Sdr {
        let mut indices: Vec<u32> = Vec::with_capacity(self.k as usize);
        let mut salt: u64 = 0;
        // Bitset to detect duplicates without HashSet allocation.
        let mut seen = vec![false; self.n as usize];

        while indices.len() < self.k as usize {
            let mut h = DefaultHasher::new();
            word.hash(&mut h);
            salt.hash(&mut h);
            let v = (h.finish() % self.n as u64) as u32;
            let bucket = v as usize;
            if !seen[bucket] {
                seen[bucket] = true;
                indices.push(v);
            }
            salt += 1;
        }
        indices.sort_unstable();
        Sdr { n: self.n, indices }
    }

    /// Encode a full string as the union of its token SDRs.
    pub fn encode(&self, text: &str) -> Sdr {
        let tokens = self.tokenize(text);
        let mut sdr = Sdr::new(self.n);
        for tok in &tokens {
            let part = self.encode_word(tok);
            sdr = sdr.union(&part);
        }
        sdr
    }
}
