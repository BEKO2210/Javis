//! Pair- and triplet-based Spike-Timing Dependent Plasticity.
//!
//! ## Pair-STDP (default)
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
//!
//! ## Triplet-STDP (Pfister & Gerstner 2006)
//!
//! Set `a3_plus > 0` (and optionally `a3_minus > 0`) to switch on the
//! triplet rule. Two extra slow traces — `pre_trace2` (τ_x ≈ 100 ms)
//! and `post_trace2` (τ_y ≈ 125 ms) — are read *just before* their
//! own update, so the LTP triplet term sees the post-trace history
//! prior to the current spike, matching the original Pfister formulation.
//!
//! `Δw_LTP = pre_trace[pre]  * (a_plus  + a3_plus  * post_trace2_pre[post])`
//! `Δw_LTD = post_trace[post] * (a_minus + a3_minus * pre_trace2_pre[pre])`
//!
//! The triplet term gives the rule the BCM-style frequency dependence
//! that pair-STDP lacks: low-rate pre+post pairs see plain pair-STDP,
//! high-rate burst-coincidences potentiate disproportionately, which
//! captures the experimentally observed visual-cortex data Pair-STDP
//! cannot fit. Defaults are `a3_plus = a3_minus = 0`, so triplet
//! contributions vanish and behaviour is bit-identical to the
//! pre-iter-44 pair rule.

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
    /// Triplet-STDP LTP coefficient (Pfister-Gerstner 2006). Reads the
    /// slow `post_trace2` *before* its own update at every post-spike.
    /// `0.0` keeps pair-STDP behaviour. Typical visual-cortex fit:
    /// `a3_plus = 6.2e-3`, `a_plus = 5e-3`.
    #[serde(default)]
    pub a3_plus: f32,
    /// Triplet-STDP LTD coefficient. Reads the slow `pre_trace2` at
    /// every pre-spike. `0.0` (default) → no triplet contribution to
    /// LTD; the slow term is rarely needed when iSTDP is in charge of
    /// LTD bookkeeping.
    #[serde(default)]
    pub a3_minus: f32,
    /// Slow pre-trace decay (ms). Used for the LTD triplet term.
    /// Default 100 ms — Pfister-Gerstner visual-cortex fit.
    #[serde(default = "default_tau_x")]
    pub tau_x: f32,
    /// Slow post-trace decay (ms). Used for the LTP triplet term.
    /// Default 125 ms — Pfister-Gerstner visual-cortex fit.
    #[serde(default = "default_tau_y")]
    pub tau_y: f32,
}

fn default_tau_x() -> f32 {
    100.0
}
fn default_tau_y() -> f32 {
    125.0
}

impl StdpParams {
    /// True iff the triplet contribution is active. Used by `Network`
    /// to decide whether to allocate the slow trace buffers.
    pub fn triplet_enabled(&self) -> bool {
        self.a3_plus != 0.0 || self.a3_minus != 0.0
    }
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
            a3_plus: 0.0,
            a3_minus: 0.0,
            tau_x: default_tau_x(),
            tau_y: default_tau_y(),
        }
    }
}
