//! Reward-modulated STDP (R-STDP) with eligibility traces.
//!
//! Closes the temporal-credit-assignment loop that pure pair/triplet STDP
//! cannot solve: the brain frequently has to associate pre/post coincidences
//! with reward signals that arrive *seconds later*. The mechanism follows
//! the three-factor learning framework laid out by Frémaux & Gerstner
//! (Front. Neural Circuits 2016) and the cortical dopamine modulation
//! work of Izhikevich (Cerebral Cortex 2007):
//!
//! ```text
//!   eligibility[s] += pair_kernel(pre_trace[pre], post_trace[post])
//!   eligibility[s] *= exp(-dt / tau_e)        // every timestep
//!   w[s]           += eta * modulator(t) * eligibility[s] * dt
//! ```
//!
//! - `pair_kernel` is the same Hebbian product the existing pair-STDP
//!   already computes; we reuse `pre_trace` / `post_trace` so no extra
//!   scratch is needed.
//! - `eligibility[s]` is a per-synapse "synaptic tag" that decays with
//!   `tau_e` (typically 0.5–2 s, much longer than the STDP traces).
//! - `modulator(t)` is a *global* scalar — the dopamine surrogate. It
//!   can be set externally each step via [`Network::set_neuromodulator`]
//!   or [`crate::Brain::set_neuromodulator`]; it can be positive
//!   (reward), negative (punishment) or zero (baseline).
//!
//! When `modulator == 0` the eligibility tag continues decaying but no
//! weight update happens, faithfully matching the dopamine-gated
//! plasticity observed in striatum and frontal cortex.
//!
//! Default `eta = 0.0` keeps reward learning off; existing networks
//! see no behaviour change unless [`Network::enable_reward_learning`]
//! is called.

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct RewardParams {
    /// Learning rate of the reward-gated weight update (per ms).
    pub eta: f32,
    /// Eligibility-trace decay time constant (ms). Long: 500–2000 ms
    /// is biologically defensible.
    pub tau_eligibility_ms: f32,
    /// Coefficient on the Hebbian product written into the trace at a
    /// post-spike (LTP-like contribution to eligibility).
    pub a_plus: f32,
    /// Coefficient on the Hebbian product written into the trace at a
    /// pre-spike (LTD-like contribution to eligibility). Stored as a
    /// non-negative magnitude — the rule subtracts internally.
    pub a_minus: f32,
    /// Hard bounds on the synapse weight after a reward update.
    pub w_min: f32,
    pub w_max: f32,
    /// If true, only excitatory synapses participate (the default
    /// striatal-cortical reading). Inhibitory I→E weights are normally
    /// shaped by iSTDP and homeostasis, not by dopamine.
    pub excitatory_only: bool,
}

impl Default for RewardParams {
    fn default() -> Self {
        Self {
            eta: 0.0,
            tau_eligibility_ms: 1000.0,
            a_plus: 0.01,
            a_minus: 0.012,
            w_min: 0.0,
            w_max: 5.0,
            excitatory_only: true,
        }
    }
}

impl RewardParams {
    /// Sensible defaults for "I want reward learning on" without
    /// hand-tuning every coefficient. `eta` is set to a small but
    /// non-zero value; the rest matches the pair-STDP defaults.
    pub fn enabled() -> Self {
        Self {
            eta: 1e-3,
            ..Self::default()
        }
    }
}
