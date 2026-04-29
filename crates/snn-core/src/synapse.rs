//! Static directed synapse with weight and synaptic-channel kind.
//!
//! When the pre-synaptic neuron fires, `weight` is added to the
//! post-synaptic neuron's per-channel `i_syn` slot. Each channel has
//! its own decay time constant on `Network`:
//!
//! - **AMPA** (default) — fast excitatory, τ ≈ 5 ms
//! - **NMDA** — slow excitatory, τ ≈ 100 ms
//! - **GABA** — inhibitory, τ ≈ 10 ms
//!
//! The default `Synapse::new` uses AMPA so existing networks keep
//! their previous behaviour. Pre-iteration-10 snapshots without a
//! `kind` field deserialise into AMPA via `#[serde(default)]`.

/// Synaptic-receptor kind. Determines which `Network` τ governs the
/// decay of the post-synaptic current contribution.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum SynapseKind {
    /// Fast excitatory. Default — uses `Network::tau_syn_ms`.
    #[default]
    Ampa,
    /// Slow excitatory. Uses `Network::tau_nmda_ms`.
    Nmda,
    /// Inhibitory. Uses `Network::tau_gaba_ms`. Note this is
    /// orthogonal to the pre-neuron's `NeuronKind`: an inhibitory
    /// pre-neuron with an AMPA synapse still delivers a *negative*
    /// PSC (sign comes from the pre's kind), but the decay follows
    /// AMPA-τ. The GABA channel exists for setups where one wants
    /// inhibitory synapses to also have GABAergic decay kinetics.
    Gaba,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Synapse {
    pub pre: usize,
    pub post: usize,
    pub weight: f32,
    /// Receptor kind. Defaults to AMPA so old snapshots round-trip.
    #[serde(default)]
    pub kind: SynapseKind,
}

impl Synapse {
    /// Create an AMPA synapse (default fast excitatory).
    pub fn new(pre: usize, post: usize, weight: f32) -> Self {
        Self {
            pre,
            post,
            weight,
            kind: SynapseKind::Ampa,
        }
    }

    /// Create a synapse with an explicit receptor kind.
    pub fn with_kind(pre: usize, post: usize, weight: f32, kind: SynapseKind) -> Self {
        Self {
            pre,
            post,
            weight,
            kind,
        }
    }
}
