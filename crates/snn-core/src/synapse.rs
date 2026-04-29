//! Static directed synapse with weight.
//!
//! When the pre-synaptic neuron fires, `weight` is added to the
//! post-synaptic neuron's `i_syn` channel. The decay time constant
//! `τ_syn` is held by `Network` (one global per network), not on
//! the synapse itself — every synapse decays at the same rate.

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Synapse {
    pub pre: usize,
    pub post: usize,
    pub weight: f32,
}

impl Synapse {
    pub fn new(pre: usize, post: usize, weight: f32) -> Self {
        Self { pre, post, weight }
    }
}
