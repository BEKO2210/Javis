//! Static directed synapse with weight and exponential post-synaptic current.
//!
//! When the pre-synaptic neuron fires, `weight` is added to the
//! post-synaptic neuron's `i_syn`, which then decays with `tau_syn`.

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Synapse {
    pub pre: usize,
    pub post: usize,
    pub weight: f32,
    pub tau_syn: f32,
}

impl Synapse {
    pub fn new(pre: usize, post: usize, weight: f32) -> Self {
        Self { pre, post, weight, tau_syn: 5.0 }
    }
}
