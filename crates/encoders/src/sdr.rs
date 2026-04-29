//! Sparse Distributed Representation: a sorted, unique list of active
//! bit indices in an address space of size `n`.

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Sdr {
    pub n: u32,
    pub indices: Vec<u32>,
}

impl Sdr {
    pub fn new(n: u32) -> Self {
        Self {
            n,
            indices: Vec::new(),
        }
    }

    /// Build an SDR from raw indices; sorts and dedups in place.
    pub fn from_indices(n: u32, mut indices: Vec<u32>) -> Self {
        indices.sort_unstable();
        indices.dedup();
        debug_assert!(indices.last().is_none_or(|&v| v < n));
        Self { n, indices }
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn density(&self) -> f32 {
        if self.n == 0 {
            0.0
        } else {
            self.indices.len() as f32 / self.n as f32
        }
    }

    /// Set-union, returning a new SDR. Both inputs must share `n`.
    pub fn union(&self, other: &Sdr) -> Sdr {
        assert_eq!(self.n, other.n, "SDRs must share the same address space");
        let mut out = Vec::with_capacity(self.indices.len() + other.indices.len());
        let (mut i, mut j) = (0, 0);
        while i < self.indices.len() && j < other.indices.len() {
            let a = self.indices[i];
            let b = other.indices[j];
            if a == b {
                out.push(a);
                i += 1;
                j += 1;
            } else if a < b {
                out.push(a);
                i += 1;
            } else {
                out.push(b);
                j += 1;
            }
        }
        out.extend_from_slice(&self.indices[i..]);
        out.extend_from_slice(&other.indices[j..]);
        Sdr {
            n: self.n,
            indices: out,
        }
    }

    /// Cardinality of the intersection — the standard SDR similarity score.
    pub fn overlap(&self, other: &Sdr) -> usize {
        assert_eq!(self.n, other.n, "SDRs must share the same address space");
        let (mut i, mut j, mut count) = (0usize, 0usize, 0usize);
        while i < self.indices.len() && j < other.indices.len() {
            let a = self.indices[i];
            let b = other.indices[j];
            if a == b {
                count += 1;
                i += 1;
                j += 1;
            } else if a < b {
                i += 1;
            } else {
                j += 1;
            }
        }
        count
    }
}
