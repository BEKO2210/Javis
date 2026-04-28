//! Poisson spike generator for external input.
//!
//! In each timestep `dt` (ms), each generator independently produces a
//! "spike" with probability `1 - exp(-rate*dt/1000)`. A spike injects
//! `current_per_spike` (nA) into its target neuron's external input
//! for that step.

use crate::rng::Rng;

#[derive(Debug, Clone, Copy)]
pub struct PoissonInput {
    pub target: usize,
    pub rate_hz: f32,
    pub current_per_spike: f32,
}

/// Fill `external` with the contributions of all generators for one step.
/// `external` is *not* zeroed; callers may mix in other inputs. `dt` is ms.
pub fn drive(generators: &[PoissonInput], external: &mut [f32], dt: f32, rng: &mut Rng) {
    let dt_s = dt * 1e-3;
    for g in generators {
        let p = 1.0 - (-g.rate_hz * dt_s).exp();
        if rng.bernoulli(p) {
            if let Some(slot) = external.get_mut(g.target) {
                *slot += g.current_per_spike;
            }
        }
    }
}
