//! Leaky Integrate-and-Fire neuron with Dale's-principle E/I labelling.
//!
//! The neuron struct holds *only* configuration — `params` and `kind`.
//! Transient state (membrane potential, refractory clock, last-spike
//! timestamp, homeostatic activity trace) lives in parallel `Vec<f32>`
//! buffers on the parent [`crate::Network`], so the integration loop
//! iterates contiguous SoA slices that the autovectoriser can pack
//! into AVX/AVX-512 lanes. The pre-iteration-22 layout had every
//! transient field embedded in this struct (AoS); see notes/41 for
//! the migration rationale and bench numbers.
//!
//! Membrane dynamics:  τ_m · dV/dt = -(V - V_rest) + R_m · I(t)
//! Discretised with forward Euler at timestep `dt` (ms). The neuron's
//! `kind` (Excitatory or Inhibitory) determines the *sign* with which
//! its outgoing synapses act on post-synaptic membranes — synaptic
//! weights themselves stay non-negative.

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NeuronKind {
    Excitatory,
    Inhibitory,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct LifParams {
    pub v_rest: f32,
    pub v_reset: f32,
    pub v_threshold: f32,
    pub tau_m: f32,
    pub r_m: f32,
    pub refractory: f32,
}

impl Default for LifParams {
    fn default() -> Self {
        Self {
            v_rest: -70.0,
            v_reset: -75.0,
            v_threshold: -55.0,
            tau_m: 20.0,
            r_m: 10.0,
            refractory: 2.0,
        }
    }
}

/// Configuration of a single LIF neuron. Carries the integration
/// parameters and Dale-principle kind only — *no* transient state.
/// 32 bytes after padding (kind is `u8` + 3B alignment + 24B params),
/// so two neurons share a 64 B cache line during the LIF inner loop.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct LifNeuron {
    pub params: LifParams,
    pub kind: NeuronKind,
}

impl LifNeuron {
    pub fn new(params: LifParams) -> Self {
        Self::with_kind(params, NeuronKind::Excitatory)
    }

    pub fn excitatory(params: LifParams) -> Self {
        Self::with_kind(params, NeuronKind::Excitatory)
    }

    pub fn inhibitory(params: LifParams) -> Self {
        Self::with_kind(params, NeuronKind::Inhibitory)
    }

    pub fn with_kind(params: LifParams, kind: NeuronKind) -> Self {
        Self { params, kind }
    }
}
