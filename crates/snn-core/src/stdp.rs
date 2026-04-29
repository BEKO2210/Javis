//! Pair-based Spike-Timing Dependent Plasticity.
//!
//! Each neuron carries a pre- and post-synaptic trace that decays
//! exponentially. On a post-synaptic spike, weights of incoming synapses
//! grow proportionally to the pre-trace of their source. On a pre-synaptic
//! spike, weights shrink proportionally to the post-trace of their target.
//!
//! Two bounding modes:
//!
//! - **Hard bounds** (default): `w_new = clamp(w + Δ, w_min, w_max)`.
//!   The classical pair-STDP behaviour. Weights pile up at the bounds
//!   under sustained co-activity, which is a well-known criticism of
//!   the rule (Morrison et al., 2008).
//! - **Soft bounds** (`soft_bounds = true`): multiplicative,
//!   self-bounding update inspired by Behavioral Timescale Synaptic
//!   Plasticity (BTSP, Bittner et al. 2017; Milstein et al. 2024
//!   PLOS Comp Bio):
//!
//!   `Δw_LTP =   a_plus  * pre_trace  * (w_max - w)`
//!   `Δw_LTD =   a_minus * post_trace * (w - w_min)`
//!
//!   The factors `(w_max - w)` and `(w - w_min)` make the update
//!   shrink as `w` approaches the bound — weights settle into a
//!   smooth distribution between `w_min` and `w_max` instead of
//!   piling up at the clamps. No `clamp()` call needed at all.

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct StdpParams {
    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    pub w_min: f32,
    pub w_max: f32,
    /// Use BTSP-style multiplicative soft bounds instead of a hard
    /// clamp. Default `false` (classical pair-STDP) for backward
    /// compatibility — every existing tuned test stays bit-identical.
    #[serde(default)]
    pub soft_bounds: bool,
}

impl Default for StdpParams {
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.012,
            tau_plus: 20.0,
            tau_minus: 20.0,
            w_min: 0.0,
            w_max: 5.0,
            soft_bounds: false,
        }
    }
}
