//! Metaplasticity — the BCM sliding LTP/LTD threshold.
//!
//! Plain pair/triplet STDP has no mechanism to keep the *direction* of
//! plasticity stable across sustained activity. Cortical experiments
//! (Bienenstock-Cooper-Munro 1982, refined by Cooper & Bear 2012) show
//! that a post-synaptic neuron's LTP/LTD crossover depends on its own
//! recent firing rate `θ`:
//!
//! ```text
//!   θ(t) = ⟨ rate(t) ⟩ ²            (sliding, slow average of rate²)
//!   factor = (rate − θ) / (rate + θ + ε)    bounded in (-1, 1)
//!   modulator(post) = clamp(factor, -k_max, +k_max) + 1
//! ```
//!
//! The `+ 1` shifts the modulator so the default state (rate ≈ θ)
//! reproduces standard STDP, recently-quiet neurons (rate < θ) bias
//! their incoming synapses towards LTD, and recently-saturated neurons
//! (rate > θ) bias towards LTP — the BCM-style stability mechanism.
//!
//! Implementation: a fast rate trace decays with `tau_rate_ms` (≈ 100 ms),
//! a slow squared-rate trace decays with `tau_theta_ms` (≈ 10 s). Both
//! are per-post-neuron, allocated lazily, and updated every step. The
//! resulting modulator multiplies the pair- and triplet-STDP `Δw` on
//! incoming excitatory synapses.
//!
//! Default `enabled = false`. With it on, networks learn just as fast
//! initially but no longer drift into runaway LTP under prolonged drive.

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct MetaplasticityParams {
    /// Fast rate-trace decay (ms). Roughly the LTP/LTD time window.
    pub tau_rate_ms: f32,
    /// Slow θ trace decay (ms). Long: 5–60 s captures the BCM regime
    /// in which θ tracks the *long-term* mean rate squared.
    pub tau_theta_ms: f32,
    /// Strength of the BCM modulation. `1.0` reproduces the textbook
    /// formula; smaller values dampen the effect (useful if you want
    /// only a gentle stabilising nudge).
    pub strength: f32,
    /// Maximum |modulator − 1| applied to a single STDP update.
    /// Bounds runaway in either direction.
    pub k_max: f32,
    /// Master switch. Default `false` so existing networks behave
    /// identically; flip via [`crate::Network::enable_metaplasticity`].
    pub enabled: bool,
}

impl Default for MetaplasticityParams {
    fn default() -> Self {
        Self {
            tau_rate_ms: 100.0,
            tau_theta_ms: 10_000.0,
            strength: 1.0,
            k_max: 0.5,
            enabled: false,
        }
    }
}

impl MetaplasticityParams {
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }

    /// Compute the per-synapse modulator from a post-neuron's fast rate
    /// and its slow θ. Returns `1.0` when both are zero (cold start).
    /// The output sits in `[1 - k_max, 1 + k_max]`.
    pub fn modulator(&self, rate: f32, theta: f32) -> f32 {
        if !self.enabled {
            return 1.0;
        }
        let denom = rate + theta + 1e-6;
        if denom <= 0.0 {
            return 1.0;
        }
        let raw = self.strength * (rate - theta) / denom;
        let clamped = raw.clamp(-self.k_max, self.k_max);
        1.0 + clamped
    }
}
