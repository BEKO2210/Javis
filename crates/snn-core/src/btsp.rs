//! Behavioral-Timescale Synaptic Plasticity (BTSP) — plateau-gated
//! retroactive potentiation kernel.
//!
//! BTSP is the biologically-canonical rule for binding a sensory cue
//! to a delayed reward / target signal across timescales (~100 ms to
//! ~seconds) that pair-STDP cannot reach (Bittner et al. 2017, *Nat
//! Neurosci*; Magee & Grienberger 2020, *Annu Rev Neurosci*; Milstein
//! et al. 2024, *PLOS Comp Bio*).
//!
//! ## Rule
//!
//! Two state machines co-exist on the network:
//!
//! 1. **Per-synapse eligibility tag** — additive on every pre-spike,
//!    exponentially decays with `eligibility_window_ms`. The tag is a
//!    "memory" of which presynaptic cells were recently active on a
//!    given synapse. No post-spike is required to grow the tag —
//!    that's the credit-assignment fix vs pair-STDP, which only sees
//!    pre-post coincidences within ~10 ms.
//!
//! 2. **Per-post-cell plateau armer** — the post-cell's recent
//!    activity is tracked by a fast exponential `burst_trace`
//!    (decay constant `plateau_window_ms`). When the trace crosses
//!    `plateau_threshold_spikes` from below, the post-cell enters
//!    `plateau_armed` state until `current_time +
//!    post_plateau_decay_ms`.
//!
//! On the *transition* from disarmed → armed, all incoming synapses
//! to that post-cell receive a *one-shot* additive potentiation:
//!
//!   `Δw[s] = potentiation_strength × tag_strength[s]`
//!
//! The weight is clamped to `[w_min, w_max]` and the tag is consumed
//! (`tag_strength[s] = 0`). Subsequent post-spikes within the
//! `post_plateau_decay_ms` window do *not* re-trigger; the plateau
//! must disarm (post silence ≥ decay) before another potentiation
//! event can fire.
//!
//! ## Locality
//!
//! `target_gated = true` (default) restricts the potentiation event
//! to the synapses incoming to the specific post-cell that hit
//! plateau. This is the per-post-cell credit assignment that pair-
//! STDP and global-modulator R-STDP both lack — only the target
//! cells (the ones the teacher signal is supervising) receive the
//! retroactive potentiation, regardless of how many other cells in
//! the network are firing.
//!
//! `target_gated = false` is the ablation control: ANY post-cell
//! hitting plateau triggers potentiation on *every* synapse with
//! non-zero tag (network-wide). Useful for verifying that the
//! per-post-cell locality is the ingredient that matters.
//!
//! ## Participation filter
//!
//! `Network::enable_btsp` accepts an optional `post_filter` which,
//! when present, restricts the rule to the listed post-cells only.
//! Other post-cells in the same network are completely outside the
//! BTSP machinery — their incoming-tag accumulation, plateau-arming
//! and potentiation are all skipped. This is what lets iter-67 apply
//! BTSP to the R2-E → C1 sub-pathway while leaving R2-R2 R-STDP
//! intact in the same `Network`.
//!
//! ## Off path
//!
//! Off by default (`Network::btsp = None`). When off, every BTSP
//! check in `Network::step` short-circuits at the `Option::is_none`
//! check; the per-synapse and per-post traces are never allocated;
//! existing snn-core numerics stay bit-identical.

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct BtspParams {
    /// Time constant of the per-synapse eligibility tag decay (ms).
    /// The tag accumulates +1 on every pre-spike and decays as
    /// `tag(t+dt) = tag(t) * exp(-dt / eligibility_window_ms)`.
    /// Default 200 ms — long enough to span cue → delay →
    /// prediction → teacher lead-in in the iter-46 6-phase
    /// schedule (~80 ms), with safety margin.
    pub eligibility_window_ms: f32,

    /// Time constant of the per-post-cell `burst_trace` decay (ms).
    /// The trace accumulates +1 on every post-spike and is compared
    /// against `plateau_threshold_spikes` for plateau-arming.
    /// Default 30 ms — matches the iter-46 teacher clamp window
    /// width so a clamped target cell's saturation firing reliably
    /// crosses the threshold within the clamp.
    pub plateau_window_ms: f32,

    /// Plateau-arm threshold on the post-cell `burst_trace`.
    /// Crossing this value from below transitions the post-cell to
    /// `plateau_armed`. Default 5 — well above the noise floor of
    /// background firing, well below the saturation count of a
    /// clamped target cell (~15 spikes within 30 ms under 500 nA).
    pub plateau_threshold_spikes: f32,

    /// Per-tagged-pre-spike weight increment applied at the
    /// plateau-arm transition (`Δw = potentiation_strength × tag`).
    /// Default 0.4 — two pre-spikes during the eligibility window
    /// saturate the synapse to `w_max = 0.8`. Biologically a
    /// "two-trial" learning rate, in line with Bittner 2017's
    /// observed in-vivo plateau-induced field formation.
    pub potentiation_strength: f32,

    /// How long the plateau stays armed after the most recent
    /// post-spike, before auto-disarming. Subsequent post-spikes
    /// during this window extend the armed period (re-arm the
    /// disarm timer) but do NOT re-trigger one-shot potentiation —
    /// that requires a full disarm-then-rearm cycle (post silence
    /// ≥ this duration, then another threshold crossing).
    /// Default 50 ms.
    pub post_plateau_decay_ms: f32,

    /// Lower weight bound (inclusive). Clamped on every
    /// potentiation. Default 0.
    pub w_min: f32,

    /// Upper weight bound (inclusive). Clamped on every
    /// potentiation. Default 0.8 (matches iter-46 R-STDP `w_max`
    /// for cross-rule consistency on R2-R2; the iter-67 ENTRY uses
    /// the same value on R2-E → C1).
    pub w_max: f32,

    /// `true` (default): the plateau-arm event applies potentiation
    /// only to synapses incoming to the specific post-cell that
    /// crossed threshold. This is the per-post-cell credit
    /// assignment.
    /// `false` (ablation): ANY post-cell's plateau-arm event
    /// applies potentiation to *every* synapse in the network with
    /// a non-zero tag, regardless of which post-cell hit plateau.
    /// Use the ablation only to verify that per-post-cell locality
    /// is what makes the rule work.
    pub target_gated: bool,
}

impl Default for BtspParams {
    fn default() -> Self {
        Self {
            eligibility_window_ms: 200.0,
            plateau_window_ms: 30.0,
            plateau_threshold_spikes: 5.0,
            potentiation_strength: 0.4,
            post_plateau_decay_ms: 50.0,
            w_min: 0.0,
            w_max: 0.8,
            target_gated: true,
        }
    }
}

impl BtspParams {
    /// Convenience constructor for the iter-67 locked smoke
    /// configuration. Identical to `Default::default()` plus an
    /// asserts-ish self-check that the params are mutually
    /// consistent (positive windows, non-negative thresholds,
    /// `w_min ≤ w_max`).
    pub fn iter67_smoke() -> Self {
        let p = Self::default();
        assert!(p.eligibility_window_ms > 0.0);
        assert!(p.plateau_window_ms > 0.0);
        assert!(p.plateau_threshold_spikes >= 0.0);
        assert!(p.potentiation_strength.is_finite());
        assert!(p.post_plateau_decay_ms >= 0.0);
        assert!(p.w_min <= p.w_max);
        p
    }
}
