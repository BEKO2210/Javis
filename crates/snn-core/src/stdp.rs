//! Pair-based Spike-Timing Dependent Plasticity.
//!
//! Each neuron carries a pre- and post-synaptic trace that decays
//! exponentially. On a post-synaptic spike, weights of incoming synapses
//! grow proportionally to the pre-trace of their source. On a pre-synaptic
//! spike, weights shrink proportionally to the post-trace of their target.

#[derive(Debug, Clone, Copy)]
pub struct StdpParams {
    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    pub w_min: f32,
    pub w_max: f32,
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
        }
    }
}
