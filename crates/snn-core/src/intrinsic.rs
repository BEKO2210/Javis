//! Intrinsic plasticity — a per-neuron adaptive threshold that drives
//! every cell towards its target firing rate.
//!
//! Where homeostatic *synaptic* scaling adjusts incoming weights, this
//! rule adjusts the post-synaptic neuron's own *threshold*. The bio
//! analogue is spike-frequency adaptation (SFA) backed by the slower
//! Na⁺/K⁺-ATPase pump current: a neuron that has been firing too much
//! drifts its threshold up, a neuron that has been silent drifts it
//! down, until both settle at the configured target rate (Desai et al.
//! 1999, Chrol-Cannon & Jin 2014 for SNN context).
//!
//! Implementation:
//!
//! ```text
//!   adapt(t+dt) = adapt(t) * exp(-dt / tau_adapt)         (decay)
//!   adapt      += alpha_spike at every post-spike          (rise)
//!   v_thresh_eff = v_threshold_base + beta * (adapt - target)
//! ```
//!
//! The effective threshold is recomputed once per step from the
//! current `adapt` trace; the LIF integration keeps using its existing
//! threshold field, but the `Network` substitutes
//! `v_thresh_eff[idx]` when comparing.
//!
//! Off by default — `Network::enable_intrinsic_plasticity` is opt-in.

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct IntrinsicParams {
    /// Adaptation increment per spike. Bigger → faster adaptation.
    pub alpha_spike: f32,
    /// Adaptation trace decay time constant (ms). Long: 1–10 s is
    /// where SFA literature operates.
    pub tau_adapt_ms: f32,
    /// Trace value the rule is steering towards. With `tau_adapt_ms`
    /// = 2000 ms, an `a_target = 5` translates to a target rate of
    /// roughly 2.5 Hz under steady drive.
    pub a_target: f32,
    /// Coupling between the trace deviation and the threshold offset.
    /// Bigger β → stiffer regulation, but at large values the network
    /// can oscillate between super-threshold and sub-threshold regimes
    /// at every step.
    pub beta: f32,
    /// Hard bounds on the threshold *offset* relative to its base
    /// value. Keeps the network well-behaved even when the trace
    /// becomes very large or very small.
    pub offset_min: f32,
    pub offset_max: f32,
    /// Master switch. Default `false`.
    pub enabled: bool,
}

impl Default for IntrinsicParams {
    fn default() -> Self {
        Self {
            alpha_spike: 1.0,
            tau_adapt_ms: 2000.0,
            a_target: 5.0,
            beta: 0.5,
            offset_min: -10.0,
            offset_max: 10.0,
            enabled: false,
        }
    }
}

impl IntrinsicParams {
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }
}
