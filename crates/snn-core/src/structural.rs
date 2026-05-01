//! Structural plasticity — sprouting and pruning of synapses.
//!
//! All other plasticity rules in Javis adjust the *strength* of a fixed
//! topology. Cortex is not a fixed topology: idle synapses retract
//! within days, and bursts of correlated activity grow new ones (Yang
//! et al. 2009 *Nature*; Holtmaat & Svoboda 2009 *Nat Rev Neurosci*).
//!
//! The breakthrough this unlocks for Javis: **engram capacity is no
//! longer a hard topology constant**. The R2 layer can grow a new
//! recurrent edge for each newly-encountered concept and prune edges
//! that have decayed below a threshold for long enough — the network
//! reorganises while it learns.
//!
//! Two complementary passes, applied periodically:
//!
//! - **Pruning**: every E→E synapse with `weight < prune_threshold`
//!   for at least `prune_age_steps` consecutive evaluations is
//!   removed (its slot stays in the `synapses` vector but is marked
//!   dead via weight 0 and removed from the `outgoing`/`incoming`
//!   buckets). The actual vector compaction happens lazily in
//!   [`crate::Network::compact_synapses`] so the pass itself stays
//!   `O(active synapses)` per call.
//!
//! - **Sprouting**: when two excitatory neurons in the same network
//!   have both fired recently (their `pre_trace` and `post_trace` are
//!   both above a threshold) but no synapse currently links them, a
//!   new synapse with `weight = sprout_initial` is added — capped at
//!   `max_new_per_step` per call so a hot regime cannot blow up the
//!   topology.
//!
//! Default `enabled = false`. Wire on via
//! [`crate::Network::enable_structural`].

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct StructuralParams {
    /// Synapses below this weight are candidates for pruning.
    pub prune_threshold: f32,
    /// Number of consecutive structural-pass evaluations a synapse
    /// must stay below `prune_threshold` before it is removed.
    pub prune_age_steps: u32,
    /// Pre-trace threshold a candidate source neuron must exceed at
    /// the moment of the structural pass for sprouting to be considered.
    pub sprout_pre_trace: f32,
    /// Post-trace threshold for the candidate target.
    pub sprout_post_trace: f32,
    /// Initial weight assigned to a freshly sprouted synapse.
    pub sprout_initial: f32,
    /// Maximum number of synapses sprouted in one structural pass.
    /// Caps the worst-case cost.
    pub max_new_per_step: u32,
    /// Run the structural pass every N steps. Defaults to 1000 (≈ 100 ms
    /// at dt = 0.1 ms) — much rarer than per-step plasticity.
    pub apply_every: u32,
    /// Master switch.
    pub enabled: bool,
}

impl Default for StructuralParams {
    fn default() -> Self {
        Self {
            prune_threshold: 0.01,
            prune_age_steps: 5,
            sprout_pre_trace: 0.5,
            sprout_post_trace: 0.5,
            sprout_initial: 0.05,
            max_new_per_step: 8,
            apply_every: 1000,
            enabled: false,
        }
    }
}

impl StructuralParams {
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }
}

/// Per-synapse "below threshold" age counter. Lives outside the hot
/// `Synapse` struct so the integration loop never touches it.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct PruneCounter {
    pub age: u32,
    /// `true` once this slot has been pruned. Pruned slots stay in the
    /// `synapses` vector at weight 0 and out of the adjacency buckets,
    /// keeping every existing `usize`/`u32` index stable.
    pub dead: bool,
}
