//! Heterosynaptic L2 / L1 weight normalisation.
//!
//! Synaptic homeostasis (`crate::homeostasis`) preserves the *relative*
//! pattern STDP shaped while moving the absolute scale up or down. That
//! still allows the *total* incoming weight on a popular post-neuron to
//! grow until every E→E synapse gets potentiated to the bound. The
//! cortical fix observed by Royer & Paré (Nature 2003), Chistiakova &
//! Volgushev (Frontiers Comp Neurosci 2014) and recently formalised by
//! Field et al. 2020 is *heterosynaptic* normalisation: when one
//! synapse onto neuron `i` strengthens, the others on the same neuron
//! weaken to keep the total bounded.
//!
//! This module implements that as a periodic per-post-neuron rescaling:
//!
//! ```text
//!   sum   = Σ_j w_ij^p     (p = 1 or 2)
//!   target = target_norm
//!   factor = (target / sum)^(1/p)            if sum > target_safe
//!   w_ij  *= factor                          for every excitatory pre j
//! ```
//!
//! Defaults: L2 norm capped at 1.0 of incoming excitatory weight every
//! `apply_every` steps. Off by default; pass [`HeterosynapticParams::l2()`]
//! into [`crate::Network::enable_heterosynaptic`] to switch on.

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum NormKind {
    /// `Σ |w|` — biologically interpretable as "total synaptic budget".
    L1,
    /// `√Σ w²` — the more aggressive "vector length" cap.
    L2,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct HeterosynapticParams {
    /// Norm to enforce.
    pub kind: NormKind,
    /// Soft target for the chosen norm. Synapses are rescaled when
    /// the post-neuron's incoming-weight norm exceeds this value.
    pub target: f32,
    /// Run the normalisation pass every N steps. Smaller is more
    /// aggressive but also more expensive.
    pub apply_every: u32,
    /// Hard floor below the target — when `sum < min_active_sum` the
    /// pass is skipped for that post-neuron. Avoids amplifying noise
    /// in completely silent rows.
    pub min_active_sum: f32,
    /// Master switch.
    pub enabled: bool,
}

impl HeterosynapticParams {
    /// L2 normalisation with a sensible default target — useful for
    /// the R2 saturation problem documented in `notes/43`.
    pub fn l2() -> Self {
        Self {
            kind: NormKind::L2,
            target: 1.5,
            apply_every: 200,
            min_active_sum: 0.1,
            enabled: true,
        }
    }

    pub fn l1() -> Self {
        Self {
            kind: NormKind::L1,
            target: 4.0,
            apply_every: 200,
            min_active_sum: 0.1,
            enabled: true,
        }
    }
}

impl Default for HeterosynapticParams {
    fn default() -> Self {
        Self {
            kind: NormKind::L2,
            target: 1.5,
            apply_every: 200,
            min_active_sum: 0.1,
            enabled: false,
        }
    }
}
