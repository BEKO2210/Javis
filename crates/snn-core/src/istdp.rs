//! Inhibitory Spike-Timing Dependent Plasticity (iSTDP).
//!
//! Anti-Hebbian learning rule applied to synapses whose pre-neuron is
//! Inhibitory and whose post-neuron is Excitatory. Two regimes per
//! pre-spike:
//!
//! - **LTD on co-activation:** when an I neuron fires while its
//!   target E neuron has just fired (high post-trace), the I→E weight
//!   shrinks. The assembly the I belongs to gradually frees its own
//!   E members from inhibition.
//!
//! - **LTP on pre-only:** when an I neuron fires and its target E
//!   neuron is silent (post-trace ≈ 0), the I→E weight grows. The
//!   I cell builds a wall of inhibition around E neurons that are
//!   not part of its assembly — i.e. that belong to competing engrams.
//!
//! Update on every I pre-spike, for each I→E outgoing synapse:
//! `dw = a_plus - a_minus * post_trace_e[post]`, then
//! `w = clamp(w + dw, w_min, w_max)`.
//!
//! Stored weights are non-negative magnitudes, exactly like the
//! E-side. The sign that turns them into a hyperpolarising current
//! is applied by `Network::step` from the pre-neuron's `NeuronKind`,
//! so the learning rule needs no sign book-keeping.

#[derive(Debug, Clone, Copy)]
pub struct IStdpParams {
    /// Baseline LTP per I-pre-spike — what gets added when the post
    /// neuron is silent in the recent window.
    pub a_plus: f32,
    /// Coefficient on the post-trace that drives LTD when the post
    /// neuron is co-active. Effective LTP/LTD threshold is
    /// `a_plus / a_minus` in trace units.
    pub a_minus: f32,
    /// Time constant of the post-trace (ms) used to detect recent
    /// E-firing. Mirrors the standard STDP `tau_minus`.
    pub tau_minus: f32,
    pub w_min: f32,
    pub w_max: f32,
}

impl Default for IStdpParams {
    fn default() -> Self {
        Self {
            a_plus: 0.0,
            a_minus: 0.0,
            tau_minus: 20.0,
            w_min: 0.0,
            w_max: 5.0,
        }
    }
}
