//! Homeostatic synaptic scaling.
//!
//! Where STDP learns *which* synapses encode a pattern, homeostasis
//! enforces *how much total drive* a neuron may receive over the long
//! run. The rule is multiplicative — every incoming excitatory weight
//! is scaled by the same factor — so the relative weight pattern that
//! STDP shaped is preserved while the absolute scale tracks the
//! neuron's own firing rate.
//!
//! Update per scaling event for each post-synaptic neuron `i`:
//! `factor = 1 + eta * (A_target - A_trace_i)`, and then
//! `w_ij = clamp(w_ij * factor)` for every excitatory pre `j`.
//!
//! `A_trace` is an exponentially-decaying spike count maintained on
//! `LifNeuron` itself; in equilibrium under a steady rate `r` Hz,
//! `A_trace` is approximately `r * tau_homeo / 1000`.
//!
//! Default `eta_scale = 0.0` — homeostasis is off unless explicitly
//! enabled, so older baseline tests behave exactly as before.

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct HomeostasisParams {
    /// Learning rate of the multiplicative scaling. `0.0` disables.
    pub eta_scale: f32,
    /// Target activity-trace value (not Hz directly — see equilibrium
    /// formula above; with `tau_homeo_ms = 2000`, an `a_target` of `5`
    /// means a target rate of ~2.5 Hz).
    pub a_target: f32,
    /// Time constant of the activity trace (ms). Long: 1–10 s typical.
    pub tau_homeo_ms: f32,
    /// Run the scaling pass every N steps. Larger N is cheaper; the
    /// trace itself is updated every step regardless.
    pub apply_every: u32,
    /// If true, weights can only ever shrink (factor capped at 1.0).
    /// Useful when STDP already does the potentiation and homeostasis
    /// only needs to stop runaway. Avoids the unstable weight-pumping
    /// regime where low-activity neurons get their weights amplified
    /// by `factor > 1`, which can spread activity through recurrent
    /// loops and trigger network-wide hyperactivity.
    pub scale_only_down: bool,
}

impl Default for HomeostasisParams {
    fn default() -> Self {
        Self {
            eta_scale: 0.0,
            a_target: 5.0,
            tau_homeo_ms: 2000.0,
            apply_every: 100,
            scale_only_down: false,
        }
    }
}
