//! Offline replay / consolidation.
//!
//! In hippocampus and cortex, recently-encoded engrams are *re-played*
//! during quiet wakefulness and slow-wave sleep, often in compressed
//! time and reverse order. The replay drives the same plasticity
//! machinery the original experience did — but at a different rate
//! and crucially at higher dopaminergic baseline — and consolidates
//! synaptic tags into long-term changes (Buzsáki 2015 *Hippocampus*,
//! Wilson & McNaughton 1994).
//!
//! Javis exposes this as an explicit `consolidate()` step that:
//!
//! 1. Picks the `top_k` excitatory neurons in the chosen region whose
//!    incoming weight sum is the largest (the "most engrammed" cells).
//! 2. Drives each of them with a brief above-threshold pulse, separated
//!    by `gap_ms`, while STDP / metaplasticity / reward all stay on.
//! 3. Optionally shuffles the order each call so the replay doesn't
//!    re-tag the same edges every time.
//!
//! The mechanism is intentionally light-weight: it uses the already-
//! built `Network::step` path, so every plasticity rule that fires
//! during waking learning fires during replay too.

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ReplayParams {
    /// Number of top-engram cells to drive per replay call.
    pub top_k: u32,
    /// Above-threshold drive each chosen neuron receives (nA).
    pub drive_current: f32,
    /// How long each pulse is held (ms).
    pub pulse_ms: f32,
    /// Gap between pulses (ms).
    pub gap_ms: f32,
    /// If `> 0`, the global neuromodulator is set to this value
    /// during replay and restored to the previous value after — the
    /// "reward replay" regime that biases consolidation towards
    /// recently-rewarded engrams.
    pub neuromod_during: f32,
    /// Reverse the order of replayed cells on alternate calls — a
    /// crude proxy for hippocampal forward / reverse replay.
    pub alternate_reverse: bool,
}

impl Default for ReplayParams {
    fn default() -> Self {
        Self {
            top_k: 16,
            drive_current: 3.0,
            pulse_ms: 5.0,
            gap_ms: 5.0,
            neuromod_during: 0.0,
            alternate_reverse: true,
        }
    }
}

impl ReplayParams {
    /// Sensible defaults for "drive a small replay cycle now".
    pub fn quick() -> Self {
        Self::default()
    }

    /// Replay a wider set of engrams, useful at the end of a training
    /// epoch.
    pub fn epoch_end() -> Self {
        Self {
            top_k: 64,
            pulse_ms: 8.0,
            gap_ms: 8.0,
            ..Self::default()
        }
    }
}
