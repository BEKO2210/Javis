//! Network of LIF neurons connected by synapses.
//!
//! Storage layout (sparse adjacency, GPU-portable):
//! - `synapses` — flat `Vec<Synapse>` of all edges.
//! - `outgoing[pre]` — edge indices whose pre-neuron is `pre`.
//! - `incoming[post]` — edge indices whose post-neuron is `post`.
//!
//! When a neuron fires we iterate only its `outgoing` (delivery + LTD) and
//! `incoming` (LTP) buckets — O(degree) instead of O(E) per spike.
//!
//! Excitatory pre-neurons add `+weight` to the post-synaptic current;
//! inhibitory pre-neurons subtract. Weights themselves are non-negative
//! magnitudes, clamped by STDP into `[w_min, w_max]`.

use crate::btsp::BtspParams;
use crate::heterosynaptic::{HeterosynapticParams, NormKind};
use crate::homeostasis::HomeostasisParams;
use crate::intrinsic::IntrinsicParams;
use crate::istdp::IStdpParams;
use crate::metaplasticity::MetaplasticityParams;
use crate::neuron::{LifNeuron, NeuronKind};
use crate::reward::RewardParams;
use crate::stdp::StdpParams;
use crate::structural::{PruneCounter, StructuralParams};
use crate::synapse::{Synapse, SynapseKind};

fn default_tau_syn_ms() -> f32 {
    5.0
}
fn default_tau_nmda_ms() -> f32 {
    100.0
}
fn default_tau_gaba_ms() -> f32 {
    10.0
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Network {
    /// Per-neuron *configuration* — parameters and Dale-principle
    /// kind. Indexed parallel to all of the per-neuron transient
    /// state vectors below. The transient state lives outside
    /// `LifNeuron` so the LIF integration loop iterates SoA slices
    /// (autovectoriser-friendly, ~75 % of recall pipeline time).
    pub neurons: Vec<LifNeuron>,
    pub synapses: Vec<Synapse>,
    pub outgoing: Vec<Vec<u32>>,
    pub incoming: Vec<Vec<u32>>,
    /// Membrane potential per neuron — transient, parallel to
    /// `neurons`. Initialised to `v_rest` on `add_neuron` and on
    /// `reset_state`.
    #[serde(skip, default)]
    pub v: Vec<f32>,
    /// Per-neuron refractory clock; the neuron is refractory while
    /// `time < refractory_until[i]`. Defaults to `f32::NEG_INFINITY`
    /// (always past).
    #[serde(skip, default)]
    pub refractory_until: Vec<f32>,
    /// Last spike timestamp per neuron — transient.
    #[serde(skip, default)]
    pub last_spike: Vec<f32>,
    /// Exponentially-decaying spike counter used by homeostatic
    /// synaptic scaling — transient.
    #[serde(skip, default)]
    pub activity_trace: Vec<f32>,
    /// AMPA synaptic current channel — transient, rebuilt on first
    /// step after a snapshot load. The default channel; `Brain` and
    /// most existing code only ever touch this one.
    #[serde(skip, default)]
    pub i_syn: Vec<f32>,
    /// NMDA channel. Only allocated lazily when the first NMDA
    /// synapse fires — most networks never use it.
    #[serde(skip, default)]
    pub i_syn_nmda: Vec<f32>,
    /// GABA channel. Lazily allocated, see `i_syn_nmda`.
    #[serde(skip, default)]
    pub i_syn_gaba: Vec<f32>,
    /// STDP pre-trace per neuron — transient.
    #[serde(skip, default)]
    pub pre_trace: Vec<f32>,
    /// STDP post-trace per neuron — transient.
    #[serde(skip, default)]
    pub post_trace: Vec<f32>,
    /// Simulation clock — transient.
    #[serde(skip, default)]
    pub time: f32,
    pub dt: f32,
    /// AMPA decay time constant (ms). Default 5 ms — biologically the
    /// AMPA range. Settable via [`Network::set_tau_syn_ms`]. Older
    /// snapshots without this field deserialise with the default.
    #[serde(default = "default_tau_syn_ms")]
    pub tau_syn_ms: f32,
    /// NMDA decay time constant (ms). Default 100 ms.
    #[serde(default = "default_tau_nmda_ms")]
    pub tau_nmda_ms: f32,
    /// GABA decay time constant (ms). Default 10 ms.
    #[serde(default = "default_tau_gaba_ms")]
    pub tau_gaba_ms: f32,
    pub stdp: Option<StdpParams>,
    pub istdp: Option<IStdpParams>,
    pub homeostasis: Option<HomeostasisParams>,
    /// Step counter — transient.
    #[serde(skip, default)]
    pub step_counter: u64,
    /// Cumulative count of synaptic deliveries since construction —
    /// transient.
    #[serde(skip, default)]
    pub synapse_events: u64,

    // ==== iter-44 "breakthrough" plasticity stack ====
    /// BCM-style metaplasticity (sliding LTP/LTD threshold per post).
    #[serde(default)]
    pub metaplasticity: Option<MetaplasticityParams>,
    /// Intrinsic plasticity / spike-frequency adaptation.
    #[serde(default)]
    pub intrinsic: Option<IntrinsicParams>,
    /// Heterosynaptic normalisation (per-post weight-norm cap).
    #[serde(default)]
    pub heterosynaptic: Option<HeterosynapticParams>,
    /// Structural plasticity (sprouting + pruning).
    #[serde(default)]
    pub structural: Option<StructuralParams>,
    /// Three-factor reward-modulated STDP. The eligibility tag and
    /// the global modulator are kept on `Network` so all learning
    /// machinery sees the same scalar.
    #[serde(default)]
    pub reward: Option<RewardParams>,
    /// Iter-67 BTSP plateau-eligibility rule (Bittner 2017 / Magee
    /// & Grienberger 2020). Per-post-cell plateau detection +
    /// per-synapse eligibility tag. See `crate::btsp` for the
    /// rule semantics. Off by default (`None`) ⇒ every snn-core
    /// callsite stays bit-identical to the pre-iter-67 path.
    #[serde(default)]
    pub btsp: Option<BtspParams>,

    /// Slow pre-trace `r2` (Pfister-Gerstner triplet LTD term). Empty
    /// unless `stdp.triplet_enabled() == true`.
    #[serde(skip, default)]
    pub pre_trace2: Vec<f32>,
    /// Slow post-trace `o2` (triplet LTP term). Empty unless triplet
    /// is on.
    #[serde(skip, default)]
    pub post_trace2: Vec<f32>,
    /// Per-synapse eligibility trace for R-STDP. Empty unless
    /// reward learning is on.
    #[serde(skip, default)]
    pub eligibility: Vec<f32>,
    /// Global neuromodulator (dopamine surrogate). Multiplied with
    /// the eligibility trace to drive reward-gated weight updates.
    /// Set via [`Network::set_neuromodulator`]. Default 0 — no
    /// reward gating.
    #[serde(skip, default)]
    pub neuromodulator: f32,
    /// Fast per-post rate trace (BCM `y`). Empty unless
    /// metaplasticity is on.
    #[serde(skip, default)]
    pub rate_trace: Vec<f32>,
    /// Slow per-post threshold trace (BCM `θ`). Empty unless
    /// metaplasticity is on.
    #[serde(skip, default)]
    pub theta_trace: Vec<f32>,
    /// Per-neuron threshold offset added to `LifParams::v_threshold`
    /// when intrinsic plasticity is on.
    #[serde(skip, default)]
    pub v_thresh_offset: Vec<f32>,
    /// Per-neuron adaptation trace driving the threshold offset
    /// above. Decays with `IntrinsicParams::tau_adapt_ms`.
    #[serde(skip, default)]
    pub adapt_trace: Vec<f32>,
    /// Per-synapse "consecutive structural-pass evaluations below
    /// threshold" counter. Empty unless structural plasticity is on.
    #[serde(skip, default)]
    pub prune_counters: Vec<PruneCounter>,
    /// Number of synapses currently marked dead (weight 0, not in
    /// the `outgoing` / `incoming` buckets). The slot stays in
    /// `synapses` to keep every `usize` / `u32` index stable.
    #[serde(skip, default)]
    pub dead_synapses: u32,
    /// Replay alternation flag — flipped each `Network::consolidate`
    /// call when `ReplayParams::alternate_reverse` is on.
    #[serde(skip, default)]
    pub replay_flip: bool,

    // ==== iter-67 BTSP plateau-eligibility rule (transient) ====
    /// Per-synapse BTSP eligibility tag. Empty unless
    /// `Network::enable_btsp` was called. Tag accumulates +1 on
    /// every pre-spike whose post-cell is in the BTSP filter, and
    /// decays exponentially with `BtspParams::eligibility_window_ms`.
    /// Consumed (set to 0) at the plateau-arm transition on its
    /// post-cell.
    #[serde(skip, default)]
    pub btsp_synapse_tag: Vec<f32>,
    /// Per-post-cell fast burst trace. Empty unless BTSP is on.
    /// `+1` on every post-spike (for cells in the filter); decays
    /// with `BtspParams::plateau_window_ms`. Compared against
    /// `BtspParams::plateau_threshold_spikes` for plateau-arm
    /// detection.
    #[serde(skip, default)]
    pub btsp_post_burst_trace: Vec<f32>,
    /// Per-post-cell time-of-disarm. `f32::NEG_INFINITY` when not
    /// armed. Set to `current_time + post_plateau_decay_ms` when
    /// the burst trace crosses the threshold from below; refreshed
    /// (extended) on every subsequent post-spike during the armed
    /// window. The disarm transition is checked at the start of
    /// every step.
    #[serde(skip, default)]
    pub btsp_post_armed_until: Vec<f32>,
    /// Per-post-cell mask: `true` ⇒ this post-cell participates in
    /// the BTSP rule. Empty when BTSP is off; full-`true` when
    /// `enable_btsp` was called without a filter; selectively
    /// `true` when called with a filter. Stored as `Vec<bool>` for
    /// cache-friendly random access in the hot loop.
    #[serde(skip, default)]
    pub btsp_post_mask: Vec<bool>,
    /// Diagnostic counter — total plateau-arm transitions since
    /// last reset. Cleared by `Network::reset_state`.
    #[serde(skip, default)]
    pub btsp_plateau_events: u64,
    /// Diagnostic counter — total per-synapse one-shot
    /// potentiation events (`Δw > 0` applied) since last reset.
    /// Cleared by `Network::reset_state`.
    #[serde(skip, default)]
    pub btsp_potentiation_events: u64,

    // ==== iter-67-β R2-recurrent partial-echo-state scale ====
    /// Iter-67-γ.1 split: per-kind multipliers applied to
    /// outgoing-synapse delivery (NOT to the stored weight, NOT
    /// to the membrane potential) when both `pre` and `post`
    /// indices are strictly less than `recurrent_scale_pre_max`.
    /// `recurrent_e_scale` is used when the pre-cell is
    /// excitatory, `recurrent_i_scale` when it is inhibitory.
    /// Both default 1.0 (no scaling, off path: a single boolean
    /// check).  Plasticity rules (STDP / iSTDP / R-STDP / BTSP)
    /// read the un-scaled stored weight, so the scale only
    /// attenuates the post-synaptic current — exactly what the
    /// iter-67-β / γ.1 prompts ask for ("Skalierung muss auf den
    /// Synapsen-Input angewendet werden, nicht auf das
    /// Membranpotenzial selbst").
    /// iter-67-β used a uniform `recurrent_scale` (E and I
    /// scaled together) which iter-67-β's sweep proved cannot
    /// decouple gain from selectivity — recurrent inhibition
    /// dominates at any uniform scale ≤ 0.80.  iter-67-γ.1 splits
    /// E and I so the operator can hold E at full strength while
    /// reducing I, exposing the cue-engram via the imbalance.
    #[serde(skip, default = "default_one_f32")]
    pub recurrent_e_scale: f32,
    #[serde(skip, default = "default_one_f32")]
    pub recurrent_i_scale: f32,
    /// Index threshold for the recurrent-scale filter (post
    /// indices `< this` AND pre indices `< this`).  Default
    /// `u32::MAX` → all-pairs (which combined with the default
    /// E and I scales of 1.0 yields a no-op).  iter-67-γ.1 sets
    /// this to `r2_n_used` so only R2-R2 synapses get scaled
    /// (R2-E → C1 synapses with post >= r2_n_used are
    /// unaffected).
    #[serde(skip, default = "default_u32_max")]
    pub recurrent_scale_pre_max: u32,
}

fn default_one_f32() -> f32 {
    1.0
}
fn default_u32_max() -> u32 {
    u32::MAX
}

impl Network {
    pub fn new(dt: f32) -> Self {
        Self {
            neurons: Vec::new(),
            synapses: Vec::new(),
            outgoing: Vec::new(),
            incoming: Vec::new(),
            v: Vec::new(),
            refractory_until: Vec::new(),
            last_spike: Vec::new(),
            activity_trace: Vec::new(),
            i_syn: Vec::new(),
            i_syn_nmda: Vec::new(),
            i_syn_gaba: Vec::new(),
            pre_trace: Vec::new(),
            post_trace: Vec::new(),
            time: 0.0,
            dt,
            tau_syn_ms: default_tau_syn_ms(),
            tau_nmda_ms: default_tau_nmda_ms(),
            tau_gaba_ms: default_tau_gaba_ms(),
            stdp: None,
            istdp: None,
            homeostasis: None,
            step_counter: 0,
            synapse_events: 0,
            metaplasticity: None,
            intrinsic: None,
            heterosynaptic: None,
            structural: None,
            reward: None,
            btsp: None,
            pre_trace2: Vec::new(),
            post_trace2: Vec::new(),
            eligibility: Vec::new(),
            neuromodulator: 0.0,
            rate_trace: Vec::new(),
            theta_trace: Vec::new(),
            v_thresh_offset: Vec::new(),
            adapt_trace: Vec::new(),
            prune_counters: Vec::new(),
            dead_synapses: 0,
            replay_flip: false,
            btsp_synapse_tag: Vec::new(),
            btsp_post_burst_trace: Vec::new(),
            btsp_post_armed_until: Vec::new(),
            btsp_post_mask: Vec::new(),
            btsp_plateau_events: 0,
            btsp_potentiation_events: 0,
            recurrent_e_scale: 1.0,
            recurrent_i_scale: 1.0,
            recurrent_scale_pre_max: u32::MAX,
        }
    }

    pub fn add_neuron(&mut self, n: LifNeuron) -> usize {
        let id = self.neurons.len();
        // Per-neuron transient state defaults match the pre-iter-22
        // `LifNeuron` field defaults: v at rest, refr/last at -inf,
        // activity trace at zero. Pushed in lock-step with `neurons`.
        self.v.push(n.params.v_rest);
        self.refractory_until.push(f32::NEG_INFINITY);
        self.last_spike.push(f32::NEG_INFINITY);
        self.activity_trace.push(0.0);
        self.neurons.push(n);
        self.outgoing.push(Vec::new());
        self.incoming.push(Vec::new());
        self.i_syn.push(0.0);
        // NMDA / GABA channels are kept lazy — they only get sized when
        // a synapse of that kind first delivers, see `kind_channel_mut`.
        self.pre_trace.push(0.0);
        self.post_trace.push(0.0);
        // iter-44 lazy per-neuron buffers: keyed off the *feature*
        // configuration, not the current buffer length, so `enable_*`
        // can be called before any neurons exist (the common pattern
        // in tests).
        if self.stdp.is_some_and(|s| s.triplet_enabled()) {
            self.pre_trace2.push(0.0);
            self.post_trace2.push(0.0);
        }
        if self.metaplasticity.is_some() {
            self.rate_trace.push(0.0);
            self.theta_trace.push(0.0);
        }
        if self.intrinsic.is_some() {
            self.adapt_trace.push(0.0);
            self.v_thresh_offset.push(0.0);
        }
        id
    }

    /// Borrow the per-neuron `i_syn` slot for the requested receptor
    /// kind, allocating the channel buffer on first use.
    fn ensure_channel(&mut self, kind: SynapseKind) -> &mut Vec<f32> {
        let n = self.neurons.len();
        match kind {
            SynapseKind::Ampa => &mut self.i_syn,
            SynapseKind::Nmda => {
                if self.i_syn_nmda.len() != n {
                    self.i_syn_nmda = vec![0.0; n];
                }
                &mut self.i_syn_nmda
            }
            SynapseKind::Gaba => {
                if self.i_syn_gaba.len() != n {
                    self.i_syn_gaba = vec![0.0; n];
                }
                &mut self.i_syn_gaba
            }
        }
    }

    /// Wire a synapse from `pre` to `post` with the given weight. Both
    /// indices must reference existing neurons; weight must be finite.
    /// Self-loops are allowed but rare in cortical wiring.
    pub fn connect(&mut self, pre: usize, post: usize, weight: f32) -> usize {
        let n = self.neurons.len();
        assert!(
            pre < n,
            "Network::connect: pre {pre} out of bounds (only {n} neurons)",
        );
        assert!(
            post < n,
            "Network::connect: post {post} out of bounds (only {n} neurons)",
        );
        assert!(
            weight.is_finite(),
            "Network::connect: weight must be finite, got {weight}",
        );
        let id = self.synapses.len();
        assert!(
            id < u32::MAX as usize,
            "Network::connect: synapse count exceeds u32 capacity",
        );
        self.synapses.push(Synapse::new(pre, post, weight));
        self.outgoing[pre].push(id as u32);
        self.incoming[post].push(id as u32);
        // Keep per-synapse iter-44 buffers in lock-step with
        // `synapses`. Keyed off the feature flag — same rationale as
        // `add_neuron`'s buffers: lets `enable_*` be called before any
        // synapse exists.
        if self.reward.is_some() {
            self.eligibility.push(0.0);
        }
        if self.structural.is_some() {
            self.prune_counters.push(PruneCounter::default());
        }
        id
    }

    /// Ensure the transient buffers (`i_syn` channels, STDP traces)
    /// have the right length for the current neuron count. Called
    /// after loading a snapshot, where `#[serde(skip)]` left them
    /// empty. NMDA / GABA channels are only sized if any synapse
    /// uses that kind — keeps the snapshot cheap for default
    /// AMPA-only networks.
    pub fn ensure_transient_state(&mut self) {
        let n = self.neurons.len();
        // Per-neuron LIF state. Defaults match `add_neuron`: rest /
        // -inf / -inf / 0.
        if self.v.len() != n {
            self.v = self.neurons.iter().map(|nu| nu.params.v_rest).collect();
        }
        if self.refractory_until.len() != n {
            self.refractory_until = vec![f32::NEG_INFINITY; n];
        }
        if self.last_spike.len() != n {
            self.last_spike = vec![f32::NEG_INFINITY; n];
        }
        if self.activity_trace.len() != n {
            self.activity_trace = vec![0.0; n];
        }
        if self.i_syn.len() != n {
            self.i_syn = vec![0.0; n];
        }
        if self.pre_trace.len() != n {
            self.pre_trace = vec![0.0; n];
        }
        if self.post_trace.len() != n {
            self.post_trace = vec![0.0; n];
        }
        let need_nmda = self.synapses.iter().any(|s| s.kind == SynapseKind::Nmda);
        let need_gaba = self.synapses.iter().any(|s| s.kind == SynapseKind::Gaba);
        if need_nmda && self.i_syn_nmda.len() != n {
            self.i_syn_nmda = vec![0.0; n];
        }
        if need_gaba && self.i_syn_gaba.len() != n {
            self.i_syn_gaba = vec![0.0; n];
        }
        // iter-44: re-seat per-neuron buffers for whichever extended
        // plasticity rule is enabled. Each re-allocation happens only
        // if the feature is configured; off-by-default networks pay
        // no cost.
        self.ensure_triplet_traces();
        if self.metaplasticity.is_some() {
            if self.rate_trace.len() != n {
                self.rate_trace = vec![0.0; n];
            }
            if self.theta_trace.len() != n {
                self.theta_trace = vec![0.0; n];
            }
        }
        if self.intrinsic.is_some() {
            if self.adapt_trace.len() != n {
                self.adapt_trace = vec![0.0; n];
            }
            if self.v_thresh_offset.len() != n {
                self.v_thresh_offset = vec![0.0; n];
            }
        }
        let s = self.synapses.len();
        if self.reward.is_some() && self.eligibility.len() != s {
            self.eligibility = vec![0.0; s];
        }
        if self.structural.is_some() && self.prune_counters.len() != s {
            self.prune_counters = vec![PruneCounter::default(); s];
        }
    }

    /// Set the AMPA decay time constant (ms). Must be positive.
    pub fn set_tau_syn_ms(&mut self, tau_syn_ms: f32) {
        assert!(
            tau_syn_ms > 0.0 && tau_syn_ms.is_finite(),
            "tau_syn_ms must be positive and finite, got {tau_syn_ms}",
        );
        self.tau_syn_ms = tau_syn_ms;
    }

    /// Set every synaptic channel's τ at once. All values must be
    /// positive and finite.
    pub fn set_synaptic_taus(&mut self, ampa_ms: f32, nmda_ms: f32, gaba_ms: f32) {
        for (label, value) in [("ampa", ampa_ms), ("nmda", nmda_ms), ("gaba", gaba_ms)] {
            assert!(
                value > 0.0 && value.is_finite(),
                "tau_{label} must be positive and finite, got {value}",
            );
        }
        self.tau_syn_ms = ampa_ms;
        self.tau_nmda_ms = nmda_ms;
        self.tau_gaba_ms = gaba_ms;
    }

    pub fn enable_stdp(&mut self, params: StdpParams) {
        self.stdp = Some(params);
        // Allocate the slow Pfister-Gerstner traces if the triplet
        // contribution is on. Idempotent — calling enable_stdp twice
        // with the same params won't double-allocate.
        self.ensure_triplet_traces();
    }

    pub fn disable_stdp(&mut self) {
        self.stdp = None;
    }

    pub fn enable_homeostasis(&mut self, params: HomeostasisParams) {
        self.homeostasis = Some(params);
    }

    pub fn disable_homeostasis(&mut self) {
        self.homeostasis = None;
    }

    pub fn enable_istdp(&mut self, params: IStdpParams) {
        self.istdp = Some(params);
    }

    pub fn disable_istdp(&mut self) {
        self.istdp = None;
    }

    // ---- iter-44 plasticity setters --------------------------------

    /// Switch on BCM-style metaplasticity. Allocates per-post `rate`
    /// and `θ` traces on first call so existing networks stay slim
    /// when the feature is unused.
    pub fn enable_metaplasticity(&mut self, params: MetaplasticityParams) {
        let n = self.neurons.len();
        if self.rate_trace.len() != n {
            self.rate_trace = vec![0.0; n];
        }
        if self.theta_trace.len() != n {
            self.theta_trace = vec![0.0; n];
        }
        self.metaplasticity = Some(params);
    }

    pub fn disable_metaplasticity(&mut self) {
        self.metaplasticity = None;
    }

    /// Switch on adaptive-threshold intrinsic plasticity. Allocates
    /// the per-neuron `adapt_trace` and `v_thresh_offset` on first call.
    pub fn enable_intrinsic_plasticity(&mut self, params: IntrinsicParams) {
        let n = self.neurons.len();
        if self.adapt_trace.len() != n {
            self.adapt_trace = vec![0.0; n];
        }
        if self.v_thresh_offset.len() != n {
            self.v_thresh_offset = vec![0.0; n];
        }
        self.intrinsic = Some(params);
    }

    pub fn disable_intrinsic_plasticity(&mut self) {
        self.intrinsic = None;
    }

    pub fn enable_heterosynaptic(&mut self, params: HeterosynapticParams) {
        self.heterosynaptic = Some(params);
    }

    pub fn disable_heterosynaptic(&mut self) {
        self.heterosynaptic = None;
    }

    /// Switch on structural plasticity. Allocates the per-synapse
    /// `prune_counters` to match the current synapse count; resized
    /// lazily as new edges are wired.
    pub fn enable_structural(&mut self, params: StructuralParams) {
        let s = self.synapses.len();
        if self.prune_counters.len() != s {
            self.prune_counters = vec![PruneCounter::default(); s];
        }
        self.structural = Some(params);
    }

    pub fn disable_structural(&mut self) {
        self.structural = None;
    }

    /// Switch on three-factor reward-modulated learning. Allocates a
    /// per-synapse `eligibility` trace to match `synapses.len()`.
    pub fn enable_reward_learning(&mut self, params: RewardParams) {
        let s = self.synapses.len();
        if self.eligibility.len() != s {
            self.eligibility = vec![0.0; s];
        }
        self.reward = Some(params);
    }

    pub fn disable_reward_learning(&mut self) {
        self.reward = None;
    }

    /// Iter-67: switch on the BTSP plateau-eligibility rule. See
    /// `crate::btsp` for the rule semantics.
    ///
    /// `post_filter`:
    /// - `None` ⇒ every post-cell in the network participates in
    ///   BTSP.
    /// - `Some(indices)` ⇒ only the listed post-cells participate.
    ///   Pre-spike events whose post is *not* in the filter do not
    ///   accumulate tags; post-spike events on cells not in the
    ///   filter do not contribute to the plateau-arm logic. This
    ///   is what lets the iter-67 caller restrict the rule to the
    ///   C1 cell index range while leaving R2-R2 plasticity intact
    ///   in the same `Network`.
    ///
    /// Allocates per-synapse `btsp_synapse_tag` (size =
    /// `synapses.len()`) + per-post-cell `btsp_post_burst_trace`,
    /// `btsp_post_armed_until` (both size = `neurons.len()`) +
    /// `btsp_post_mask` (also `neurons.len()`).
    pub fn enable_btsp(&mut self, params: BtspParams, post_filter: Option<&[usize]>) {
        let n = self.neurons.len();
        let s = self.synapses.len();
        if self.btsp_synapse_tag.len() != s {
            self.btsp_synapse_tag = vec![0.0; s];
        }
        if self.btsp_post_burst_trace.len() != n {
            self.btsp_post_burst_trace = vec![0.0; n];
        }
        if self.btsp_post_armed_until.len() != n {
            self.btsp_post_armed_until = vec![f32::NEG_INFINITY; n];
        }
        let mut mask = match post_filter {
            None => vec![true; n],
            Some(_) => vec![false; n],
        };
        if let Some(indices) = post_filter {
            for &idx in indices {
                if idx < n {
                    mask[idx] = true;
                }
            }
        }
        self.btsp_post_mask = mask;
        self.btsp_plateau_events = 0;
        self.btsp_potentiation_events = 0;
        self.btsp = Some(params);
    }

    /// Iter-67: switch off BTSP. Leaves the per-synapse / per-post
    /// transient buffers in place (cheap on memory; idle without the
    /// `Some(params)` flag) so a subsequent `enable_btsp` can
    /// re-arm without re-allocating.
    pub fn disable_btsp(&mut self) {
        self.btsp = None;
    }

    /// Iter-67-β: uniform recurrent-synapse delivery scale.
    /// Sets both `recurrent_e_scale` and `recurrent_i_scale` to
    /// the same value.  When the resulting scale != 1.0, every
    /// spike-driven synaptic delivery on a synapse with both
    /// `pre` and `post` indices `< pre_max` has its `weight`
    /// multiplied by the per-pre-kind scale BEFORE being added
    /// to the post-cell's synaptic-current channel.  The stored
    /// synapse `weight` is unchanged; STDP / R-STDP / BTSP read
    /// the un-scaled value.  iter-67-γ.1 supersedes this with
    /// the E/I-specific setters below; this uniform API is kept
    /// for backward-compat with iter-67-β code paths and for
    /// callers that don't need E/I separation.
    pub fn set_recurrent_scale(&mut self, scale: f32, pre_max: usize) {
        let s = if scale.is_finite() { scale } else { 1.0 };
        self.recurrent_e_scale = s;
        self.recurrent_i_scale = s;
        self.recurrent_scale_pre_max = pre_max.min(u32::MAX as usize) as u32;
    }

    /// Iter-67-γ.1: set the per-network recurrent-synapse delivery
    /// scales SEPARATELY for excitatory and inhibitory pre-cells.
    /// Used by the iter-67-γ.1 partial-echo-state teacher phase:
    /// hold E recurrent at full strength (`e_scale = 1.0`) while
    /// reducing I-suppression (`i_scale = 0.3`), exposing the
    /// cue-engram via the resulting E/I imbalance.  Both scales
    /// apply only to synapses where both `pre` and `post`
    /// indices are `< pre_max` (i.e. the R2-R2 recurrent block,
    /// not the R2-E → C1 feedforward suffix).
    pub fn set_recurrent_e_i_scales(&mut self, e_scale: f32, i_scale: f32, pre_max: usize) {
        self.recurrent_e_scale = if e_scale.is_finite() { e_scale } else { 1.0 };
        self.recurrent_i_scale = if i_scale.is_finite() { i_scale } else { 1.0 };
        self.recurrent_scale_pre_max = pre_max.min(u32::MAX as usize) as u32;
    }

    /// Iter-67-β / γ.1: reset to no scaling.  Default-state
    /// restorer.  Both E and I scales return to 1.0.
    pub fn clear_recurrent_scale(&mut self) {
        self.recurrent_e_scale = 1.0;
        self.recurrent_i_scale = 1.0;
        self.recurrent_scale_pre_max = u32::MAX;
    }

    /// Set the global neuromodulator (dopamine surrogate). The next
    /// [`Network::step`] reads this value when `reward` is `Some`.
    /// Persisting state across steps is the caller's responsibility:
    /// the modulator is *not* automatically reset, so a tonic baseline
    /// can be modelled by leaving it set.
    pub fn set_neuromodulator(&mut self, value: f32) {
        self.neuromodulator = if value.is_finite() { value } else { 0.0 };
    }

    /// Triplet-STDP allocations are tied to whether the configured
    /// `StdpParams` need them. Idempotent.
    fn ensure_triplet_traces(&mut self) {
        let want = self.stdp.map(|s| s.triplet_enabled()).unwrap_or(false);
        let n = self.neurons.len();
        if want {
            if self.pre_trace2.len() != n {
                self.pre_trace2 = vec![0.0; n];
            }
            if self.post_trace2.len() != n {
                self.post_trace2 = vec![0.0; n];
            }
        }
    }

    /// Reset transient state (membrane potentials, synaptic currents,
    /// STDP traces, homeostatic activity traces, refractory clocks,
    /// time, step counter, event counter). Synapse weights and network
    /// topology are preserved.
    pub fn reset_state(&mut self) {
        for (idx, n) in self.neurons.iter().enumerate() {
            self.v[idx] = n.params.v_rest;
            self.refractory_until[idx] = f32::NEG_INFINITY;
            self.last_spike[idx] = f32::NEG_INFINITY;
            self.activity_trace[idx] = 0.0;
            self.i_syn[idx] = 0.0;
            if idx < self.i_syn_nmda.len() {
                self.i_syn_nmda[idx] = 0.0;
            }
            if idx < self.i_syn_gaba.len() {
                self.i_syn_gaba[idx] = 0.0;
            }
            self.pre_trace[idx] = 0.0;
            self.post_trace[idx] = 0.0;
            if idx < self.pre_trace2.len() {
                self.pre_trace2[idx] = 0.0;
            }
            if idx < self.post_trace2.len() {
                self.post_trace2[idx] = 0.0;
            }
            if idx < self.rate_trace.len() {
                self.rate_trace[idx] = 0.0;
            }
            if idx < self.theta_trace.len() {
                self.theta_trace[idx] = 0.0;
            }
            if idx < self.adapt_trace.len() {
                self.adapt_trace[idx] = 0.0;
            }
            if idx < self.v_thresh_offset.len() {
                self.v_thresh_offset[idx] = 0.0;
            }
        }
        for x in self.eligibility.iter_mut() {
            *x = 0.0;
        }
        self.neuromodulator = 0.0;
        for c in self.prune_counters.iter_mut() {
            c.age = 0;
        }
        // iter-67 BTSP transient state: tags + burst traces + plateau
        // armed-until clear on every reset_state. Synapse weights
        // themselves survive (they're topology + learned state, not
        // transient). The diagnostic counters
        // (`btsp_plateau_events`, `btsp_potentiation_events`) are
        // intentionally NOT reset here — they accumulate across
        // multiple trials within a training epoch (each trial calls
        // reset_state) so the caller can read a per-epoch total.
        // Use `enable_btsp` to clear the counters (it does that as
        // part of its setup).
        for x in self.btsp_synapse_tag.iter_mut() {
            *x = 0.0;
        }
        for x in self.btsp_post_burst_trace.iter_mut() {
            *x = 0.0;
        }
        for x in self.btsp_post_armed_until.iter_mut() {
            *x = f32::NEG_INFINITY;
        }
        self.time = 0.0;
        self.step_counter = 0;
        self.synapse_events = 0;
    }

    /// Advance the network one timestep with optional external currents.
    /// `external` length must equal the number of neurons (or be empty,
    /// meaning zero external input). Returns indices that fired.
    pub fn step(&mut self, external: &[f32]) -> Vec<usize> {
        let dt = self.dt;
        let t = self.time;

        // 1) Decay every active synaptic channel with its own τ.
        //    NMDA / GABA buffers are only walked if they were ever
        //    sized (i.e. at least one synapse of that kind delivered).
        let decay_ampa = (-dt / self.tau_syn_ms.max(1e-3)).exp();
        for x in self.i_syn.iter_mut() {
            *x *= decay_ampa;
        }
        if !self.i_syn_nmda.is_empty() {
            let decay_nmda = (-dt / self.tau_nmda_ms.max(1e-3)).exp();
            for x in self.i_syn_nmda.iter_mut() {
                *x *= decay_nmda;
            }
        }
        if !self.i_syn_gaba.is_empty() {
            let decay_gaba = (-dt / self.tau_gaba_ms.max(1e-3)).exp();
            for x in self.i_syn_gaba.iter_mut() {
                *x *= decay_gaba;
            }
        }

        // 2) Decay plasticity traces. `pre_trace` is only used by E-side
        //    STDP. `post_trace` is shared between STDP (E-side) and
        //    iSTDP (I→E side) — both interpret it as "this post-neuron
        //    has fired recently". When both are active we decay with
        //    the STDP `tau_minus` for compatibility.
        if let Some(p) = self.stdp {
            let dp = (-dt / p.tau_plus).exp();
            let dm = (-dt / p.tau_minus).exp();
            for x in self.pre_trace.iter_mut() {
                *x *= dp;
            }
            for x in self.post_trace.iter_mut() {
                *x *= dm;
            }
            // Triplet slow-trace decays. Skipped (and buffers stay
            // empty) when `triplet_enabled() == false`.
            if p.triplet_enabled() {
                let dpx = (-dt / p.tau_x.max(1e-3)).exp();
                let dpy = (-dt / p.tau_y.max(1e-3)).exp();
                for x in self.pre_trace2.iter_mut() {
                    *x *= dpx;
                }
                for x in self.post_trace2.iter_mut() {
                    *x *= dpy;
                }
            }
        } else if let Some(ip) = self.istdp {
            // iSTDP only — decay just the post-trace using its own tau.
            let dm = (-dt / ip.tau_minus).exp();
            for x in self.post_trace.iter_mut() {
                *x *= dm;
            }
        }

        // 2b) Decay homeostatic activity traces (long time constant).
        if let Some(h) = self.homeostasis {
            let decay = (-dt / h.tau_homeo_ms).exp();
            for x in self.activity_trace.iter_mut() {
                *x *= decay;
            }
        }

        // 2c) iter-44 trace decays — metaplasticity rate/θ, intrinsic
        //     adaptation, eligibility tag. Each runs only if its
        //     buffer was actually sized by the corresponding `enable_*`.
        if let Some(m) = self.metaplasticity {
            if m.enabled && !self.rate_trace.is_empty() {
                let drate = (-dt / m.tau_rate_ms.max(1e-3)).exp();
                let dtheta = (-dt / m.tau_theta_ms.max(1e-3)).exp();
                for x in self.rate_trace.iter_mut() {
                    *x *= drate;
                }
                // θ tracks the squared *long-term* rate. The `(1-dtheta)`
                // weighting keeps the trace a low-pass-filtered estimate
                // of `rate^2` — slow enough for stability, fast enough
                // to track concept drift.
                for (theta, rate) in self.theta_trace.iter_mut().zip(self.rate_trace.iter()) {
                    *theta = *theta * dtheta + (1.0 - dtheta) * (*rate) * (*rate);
                }
            }
        }
        if let Some(ip) = self.intrinsic {
            if ip.enabled && !self.adapt_trace.is_empty() {
                let da = (-dt / ip.tau_adapt_ms.max(1e-3)).exp();
                for x in self.adapt_trace.iter_mut() {
                    *x *= da;
                }
                // Recompute the per-neuron threshold offset from the
                // current adaptation trace. This loop is straight-line
                // and autovectoriser-friendly.
                for (offset, &adapt) in self.v_thresh_offset.iter_mut().zip(self.adapt_trace.iter())
                {
                    *offset = (ip.beta * (adapt - ip.a_target)).clamp(ip.offset_min, ip.offset_max);
                }
            }
        }
        if let Some(rp) = self.reward {
            if !self.eligibility.is_empty() {
                let de = (-dt / rp.tau_eligibility_ms.max(1e-3)).exp();
                for x in self.eligibility.iter_mut() {
                    *x *= de;
                }
            }
        }
        // 2d) iter-67 BTSP trace decays — per-synapse eligibility tag
        //     and per-post-cell burst trace. Plateau-armed-until is a
        //     timestamp, not a decay; auto-disarm is checked at use
        //     sites below. Off path skips on `Option::is_none`.
        if let Some(bp) = self.btsp {
            if !self.btsp_synapse_tag.is_empty() {
                let dtag = (-dt / bp.eligibility_window_ms.max(1e-3)).exp();
                for x in self.btsp_synapse_tag.iter_mut() {
                    *x *= dtag;
                }
            }
            if !self.btsp_post_burst_trace.is_empty() {
                let dburst = (-dt / bp.plateau_window_ms.max(1e-3)).exp();
                for x in self.btsp_post_burst_trace.iter_mut() {
                    *x *= dburst;
                }
            }
        }

        // 3) Step every LIF, recording spikes. SoA layout: per-neuron
        //    transient state lives in parallel `Vec<f32>` slices on
        //    `self`; the inner loop is a single straight-line walk
        //    over indices, autovectoriser-friendly.
        let intrinsic_active =
            self.intrinsic.is_some_and(|p| p.enabled) && !self.v_thresh_offset.is_empty();
        let mut fired: Vec<usize> = Vec::new();
        let n = self.neurons.len();
        for idx in 0..n {
            let ext = external.get(idx).copied().unwrap_or(0.0);
            // Sum across every active receptor channel.
            let mut total = ext + self.i_syn[idx];
            if let Some(v) = self.i_syn_nmda.get(idx) {
                total += *v;
            }
            if let Some(v) = self.i_syn_gaba.get(idx) {
                total += *v;
            }
            // Inline LIF math, parallel-Vec form. Identical to the
            // pre-iter-22 `LifNeuron::step` body — same forward-Euler
            // discretisation, same threshold-and-reset semantics.
            let p = &self.neurons[idx].params;
            let v = &mut self.v[idx];
            let refr = &mut self.refractory_until[idx];
            if t < *refr {
                *v = p.v_reset;
                continue;
            }
            let dv = dt / p.tau_m * (-(*v - p.v_rest) + p.r_m * total);
            *v += dv;
            // Effective threshold: classical fixed `v_threshold` *plus*
            // the intrinsic-plasticity offset when that feature is on.
            // Off path: `intrinsic_active = false` → no read, no branch
            // mispredict, byte-identical to the pre-iter-44 hot loop.
            let v_th = if intrinsic_active {
                p.v_threshold + self.v_thresh_offset[idx]
            } else {
                p.v_threshold
            };
            if *v >= v_th {
                *v = p.v_reset;
                *refr = t + p.refractory;
                self.last_spike[idx] = t;
                fired.push(idx);
            }
        }

        // 4) Deliver synaptic effects + plasticity updates for spikes this step.
        let stdp = self.stdp;
        let istdp = self.istdp;
        let any_post_trace = stdp.is_some() || istdp.is_some();
        let homeo_active = self.homeostasis.is_some();
        let triplet_active = stdp.is_some_and(|s| s.triplet_enabled());
        let meta_active =
            self.metaplasticity.is_some_and(|m| m.enabled) && !self.rate_trace.is_empty();
        let intrinsic_spike_active =
            self.intrinsic.is_some_and(|p| p.enabled) && !self.adapt_trace.is_empty();
        let reward_active = self.reward.is_some() && !self.eligibility.is_empty();
        let meta_params = self.metaplasticity;
        let intrinsic_params = self.intrinsic;
        let reward_params = self.reward;
        // iter-67 BTSP precompute. Off path: every BTSP block in the
        // spike loop short-circuits at this single boolean.
        let btsp_active = self.btsp.is_some()
            && !self.btsp_post_mask.is_empty()
            && !self.btsp_synapse_tag.is_empty()
            && !self.btsp_post_burst_trace.is_empty();
        let btsp_params = self.btsp;

        for &src in &fired {
            if stdp.is_some() {
                self.pre_trace[src] += 1.0;
            }
            if any_post_trace {
                self.post_trace[src] += 1.0;
            }
            if homeo_active {
                self.activity_trace[src] += 1.0;
            }
            // iter-44: rate/adapt traces & triplet slow traces tick on
            // every spike too. Ordering matters for triplet — the slow
            // post-trace is *read* before its own update in the LTP
            // path, matching the Pfister-Gerstner formulation.
            if meta_active {
                self.rate_trace[src] += 1.0;
            }
            if intrinsic_spike_active {
                self.adapt_trace[src] += intrinsic_params.map(|p| p.alpha_spike).unwrap_or(0.0);
            }
            let src_kind = self.neurons[src].kind;
            let sign: f32 = match src_kind {
                NeuronKind::Excitatory => 1.0,
                NeuronKind::Inhibitory => -1.0,
            };

            // Outgoing edges from `src`: deliver PSC, optional plasticity.
            //
            // - E-pre + STDP: classical LTD using post_trace[post].
            // - I-pre + iSTDP: anti-Hebbian update on I→E edges using
            //   `dw = a_plus - a_minus * post_trace[post]`. Pre-only
            //   firing (silent E) drives LTP; co-activity (E recently
            //   fired) drives LTD. Magnitudes stay non-negative.
            let n_out = self.outgoing[src].len();
            // iter-67-β / γ.1: precompute the recurrent-scale state
            // once per spike (`src` is constant inside this loop).
            // The per-pre-kind scale is selected by `src_kind` —
            // iter-67-γ.1 splits E and I scaling to decouple R2's
            // E/I balance during the BTSP teacher phase. Off path
            // (both scales = 1.0): the inner-loop branch is `false`,
            // no read of the post-cell index against pre_max, no
            // multiply. Bit-identical to pre-iter-67-β numerics
            // when both scales are at default.
            let active_scale = match src_kind {
                NeuronKind::Excitatory => self.recurrent_e_scale,
                NeuronKind::Inhibitory => self.recurrent_i_scale,
            };
            let scale_recurrent =
                active_scale != 1.0 && (src as u32) < self.recurrent_scale_pre_max;
            let recurrent_pre_max = self.recurrent_scale_pre_max;
            for i in 0..n_out {
                let eid = self.outgoing[src][i] as usize;
                let post = self.synapses[eid].post;
                let w = self.synapses[eid].weight;
                let kind = self.synapses[eid].kind;
                let channel = self.ensure_channel(kind);
                // iter-67-β / γ.1 recurrent scale: only apply when
                // the synapse is "recurrent" by the index criterion
                // (both pre and post < pre_max).  Stored weight
                // unchanged; STDP / R-STDP / BTSP downstream read
                // the un-scaled `self.synapses[eid].weight`.  γ.1's
                // per-kind scaling produces an E/I imbalance during
                // the teacher window: `e_scale = 1.0, i_scale = 0.3`
                // keeps E firing while reducing I-suppression, so
                // the strongest cue-engram E-cells dominate.
                let delivered_w = if scale_recurrent && (post as u32) < recurrent_pre_max {
                    w * active_scale
                } else {
                    w
                };
                channel[post] += sign * delivered_w;
                self.synapse_events += 1;
                // iter-67 BTSP: per-pre-spike eligibility tag tick on
                // synapses targeting filtered post-cells. Off path:
                // `btsp_active = false` skips the read entirely. Only
                // accumulates on excitatory pre — inhibitory R2-I →
                // post is not subject to BTSP potentiation.
                if btsp_active
                    && matches!(src_kind, NeuronKind::Excitatory)
                    && self.btsp_post_mask[post]
                {
                    self.btsp_synapse_tag[eid] += 1.0;
                }
                match src_kind {
                    NeuronKind::Excitatory => {
                        if let Some(p) = stdp {
                            // pre-spike → LTD on this outgoing synapse
                            // proportional to the post-trace. Triplet:
                            // multiply by `(a_minus + a3_minus * pre_trace2[pre])`
                            // *read before* the slow pre-trace tick.
                            let amin = if triplet_active {
                                p.a_minus + p.a3_minus * self.pre_trace2[src]
                            } else {
                                p.a_minus
                            };
                            // BCM modulator on the post-neuron — modulates
                            // both LTP and LTD symmetrically. Off path:
                            // `meta_active = false` → constant 1.0.
                            let modulator = if meta_active {
                                meta_params
                                    .map(|m| {
                                        m.modulator(self.rate_trace[post], self.theta_trace[post])
                                    })
                                    .unwrap_or(1.0)
                            } else {
                                1.0
                            };
                            let dw_pre = amin * self.post_trace[post] * modulator;
                            let new_w = if p.soft_bounds {
                                w - dw_pre * (w - p.w_min)
                            } else {
                                (w - dw_pre).clamp(p.w_min, p.w_max)
                            };
                            self.synapses[eid].weight = new_w;
                            // Reward eligibility: pre-spike contribution.
                            if reward_active {
                                if let Some(rp) = reward_params {
                                    if !rp.excitatory_only
                                        || self.neurons[post].kind == NeuronKind::Excitatory
                                    {
                                        self.eligibility[eid] -= rp.a_minus * self.post_trace[post];
                                    }
                                }
                            }
                        }
                    }
                    NeuronKind::Inhibitory => {
                        if let Some(ip) = istdp {
                            if self.neurons[post].kind == NeuronKind::Excitatory {
                                let dw = ip.a_plus - ip.a_minus * self.post_trace[post];
                                let new_w = (w + dw).clamp(ip.w_min, ip.w_max);
                                self.synapses[eid].weight = new_w;
                            }
                        }
                    }
                }
            }

            // Incoming edges to `src`: classical STDP LTP (E-side).
            // iSTDP does not need an incoming-side update — the rule on
            // the I-pre-spike side already covers both LTP (silent E)
            // and LTD (recently-fired E) cases.
            if let Some(p) = stdp {
                let n_in = self.incoming[src].len();
                for i in 0..n_in {
                    let eid = self.incoming[src][i] as usize;
                    let pre = self.synapses[eid].pre;
                    if self.neurons[pre].kind != NeuronKind::Excitatory {
                        continue;
                    }
                    let w = self.synapses[eid].weight;
                    // Triplet LTP: `(a_plus + a3_plus * post_trace2[post])`
                    // read *before* the slow post-trace tick below, per
                    // Pfister-Gerstner.
                    let aplu = if triplet_active {
                        p.a_plus + p.a3_plus * self.post_trace2[src]
                    } else {
                        p.a_plus
                    };
                    let modulator = if meta_active {
                        meta_params
                            .map(|m| m.modulator(self.rate_trace[src], self.theta_trace[src]))
                            .unwrap_or(1.0)
                    } else {
                        1.0
                    };
                    let dw_post = aplu * self.pre_trace[pre] * modulator;
                    let new_w = if p.soft_bounds {
                        w + dw_post * (p.w_max - w)
                    } else {
                        (w + dw_post).clamp(p.w_min, p.w_max)
                    };
                    self.synapses[eid].weight = new_w;
                    // Reward eligibility: post-spike contribution.
                    if reward_active {
                        if let Some(rp) = reward_params {
                            self.eligibility[eid] += rp.a_plus * self.pre_trace[pre];
                        }
                    }
                }
            }

            // Triplet slow-trace tick happens *after* both directions
            // read the previous value — Pfister-Gerstner ordering.
            if triplet_active {
                self.pre_trace2[src] += 1.0;
                self.post_trace2[src] += 1.0;
            }

            // iter-67 BTSP plateau detector + one-shot potentiation.
            // Treats the firing `src` as a post-cell. Off path:
            // single boolean check, zero work.
            if btsp_active && self.btsp_post_mask[src] {
                self.btsp_post_burst_trace[src] += 1.0;
                let bp = btsp_params.expect("btsp_active checked above");
                let was_armed = self.btsp_post_armed_until[src] > t;
                if self.btsp_post_burst_trace[src] >= bp.plateau_threshold_spikes {
                    // (Re-)arm or extend the disarm timer on every
                    // post-spike while above threshold. The one-shot
                    // potentiation only fires on the disarmed → armed
                    // transition; subsequent post-spikes during the
                    // armed window only refresh the disarm time.
                    self.btsp_post_armed_until[src] = t + bp.post_plateau_decay_ms;
                    if !was_armed {
                        self.btsp_plateau_events = self.btsp_plateau_events.wrapping_add(1);
                        if bp.target_gated {
                            // Per-post-cell credit assignment: scan
                            // only this cell's incoming.
                            let n_in = self.incoming[src].len();
                            for i in 0..n_in {
                                let eid = self.incoming[src][i] as usize;
                                let tag = self.btsp_synapse_tag[eid];
                                if tag <= 0.0 {
                                    continue;
                                }
                                let dw = bp.potentiation_strength * tag;
                                if dw == 0.0 {
                                    continue;
                                }
                                let w = self.synapses[eid].weight;
                                let new_w = (w + dw).clamp(bp.w_min, bp.w_max);
                                self.synapses[eid].weight = new_w;
                                self.btsp_synapse_tag[eid] = 0.0;
                                self.btsp_potentiation_events =
                                    self.btsp_potentiation_events.wrapping_add(1);
                            }
                        } else {
                            // Ablation: any-cell-plateau → network-wide
                            // potentiation. O(N_synapses) per event;
                            // only used as a control to verify that
                            // per-post locality is what makes the rule
                            // work.
                            for eid in 0..self.synapses.len() {
                                let tag = self.btsp_synapse_tag[eid];
                                if tag <= 0.0 {
                                    continue;
                                }
                                let dw = bp.potentiation_strength * tag;
                                if dw == 0.0 {
                                    continue;
                                }
                                let w = self.synapses[eid].weight;
                                let new_w = (w + dw).clamp(bp.w_min, bp.w_max);
                                self.synapses[eid].weight = new_w;
                                self.btsp_synapse_tag[eid] = 0.0;
                                self.btsp_potentiation_events =
                                    self.btsp_potentiation_events.wrapping_add(1);
                            }
                        }
                    }
                }
            }
        }

        // 4b) Apply reward-gated weight update on every synapse with
        //     non-zero eligibility, every step. Cheap when the
        //     eligibility vec is zero (skip via early-out).
        if reward_active && self.neuromodulator.abs() > 0.0 {
            if let Some(rp) = reward_params {
                let scale = rp.eta * self.neuromodulator * dt;
                if scale != 0.0 {
                    for eid in 0..self.synapses.len() {
                        let elig = self.eligibility[eid];
                        if elig == 0.0 {
                            continue;
                        }
                        if rp.excitatory_only {
                            let pre = self.synapses[eid].pre;
                            if self.neurons[pre].kind != NeuronKind::Excitatory {
                                continue;
                            }
                        }
                        let w = self.synapses[eid].weight;
                        let new_w = (w + scale * elig).clamp(rp.w_min, rp.w_max);
                        self.synapses[eid].weight = new_w;
                    }
                }
            }
        }

        // 5) Periodic homeostatic synaptic scaling.
        self.step_counter = self.step_counter.wrapping_add(1);
        if let Some(h) = self.homeostasis {
            if h.eta_scale != 0.0
                && h.apply_every > 0
                && self.step_counter % (h.apply_every as u64) == 0
            {
                self.apply_synaptic_scaling(&h);
            }
        }

        // 5b) Periodic heterosynaptic normalisation.
        if let Some(hs) = self.heterosynaptic {
            if hs.enabled && hs.apply_every > 0 && self.step_counter % (hs.apply_every as u64) == 0
            {
                self.apply_heterosynaptic_norm(&hs);
            }
        }

        // 5c) Periodic structural plasticity — sprout + prune.
        if let Some(sp) = self.structural {
            if sp.enabled && sp.apply_every > 0 && self.step_counter % (sp.apply_every as u64) == 0
            {
                self.apply_structural_pass(&sp);
            }
        }

        self.time += dt;
        fired
    }

    /// Heterosynaptic normalisation pass — caps the per-post-neuron
    /// L1 / L2 norm of incoming excitatory weights at `target`.
    fn apply_heterosynaptic_norm(&mut self, hs: &HeterosynapticParams) {
        let n = self.neurons.len();
        for post in 0..n {
            // Only excitatory targets carry a meaningful "incoming
            // excitatory budget" — inhibitory cells are normally
            // shaped by iSTDP alone.
            if self.neurons[post].kind != NeuronKind::Excitatory {
                continue;
            }
            let n_in = self.incoming[post].len();
            if n_in == 0 {
                continue;
            }
            let mut acc = 0.0_f32;
            for i in 0..n_in {
                let eid = self.incoming[post][i] as usize;
                let pre = self.synapses[eid].pre;
                if self.neurons[pre].kind != NeuronKind::Excitatory {
                    continue;
                }
                let w = self.synapses[eid].weight;
                acc += match hs.kind {
                    NormKind::L1 => w.abs(),
                    NormKind::L2 => w * w,
                };
            }
            if acc < hs.min_active_sum {
                continue;
            }
            let norm = match hs.kind {
                NormKind::L1 => acc,
                NormKind::L2 => acc.sqrt(),
            };
            if norm <= hs.target {
                continue;
            }
            let factor = hs.target / norm;
            for i in 0..n_in {
                let eid = self.incoming[post][i] as usize;
                let pre = self.synapses[eid].pre;
                if self.neurons[pre].kind != NeuronKind::Excitatory {
                    continue;
                }
                self.synapses[eid].weight *= factor;
            }
        }
    }

    /// Structural pass — prune dormant E→E synapses and (optionally)
    /// sprout new ones between consistently co-active E cells.
    fn apply_structural_pass(&mut self, sp: &StructuralParams) {
        let m = self.synapses.len();
        if self.prune_counters.len() != m {
            self.prune_counters.resize_with(m, PruneCounter::default);
        }
        // ----- Pruning -----
        let mut to_prune: Vec<usize> = Vec::new();
        for eid in 0..m {
            if self.prune_counters[eid].dead {
                continue;
            }
            // Limit pruning to E→E synapses for safety — we don't want
            // to silently retract iSTDP-managed I→E inhibition.
            let pre = self.synapses[eid].pre;
            let post = self.synapses[eid].post;
            if self.neurons[pre].kind != NeuronKind::Excitatory
                || self.neurons[post].kind != NeuronKind::Excitatory
            {
                continue;
            }
            if self.synapses[eid].weight < sp.prune_threshold {
                self.prune_counters[eid].age = self.prune_counters[eid].age.saturating_add(1);
                if self.prune_counters[eid].age >= sp.prune_age_steps {
                    to_prune.push(eid);
                }
            } else {
                self.prune_counters[eid].age = 0;
            }
        }
        for eid in to_prune {
            self.synapses[eid].weight = 0.0;
            self.prune_counters[eid].dead = true;
            // Remove from adjacency buckets so the integration loop
            // never visits this slot again. The vector indices stay
            // stable; the `synapses` slot lingers as a tombstone.
            let pre = self.synapses[eid].pre;
            let post = self.synapses[eid].post;
            self.outgoing[pre].retain(|&e| e as usize != eid);
            self.incoming[post].retain(|&e| e as usize != eid);
            self.dead_synapses += 1;
        }

        // ----- Sprouting -----
        let n = self.neurons.len();
        // Candidate sources / targets from the recent activity traces.
        let mut hot_pre: Vec<usize> = Vec::new();
        let mut hot_post: Vec<usize> = Vec::new();
        for idx in 0..n {
            if self.neurons[idx].kind != NeuronKind::Excitatory {
                continue;
            }
            if self.pre_trace[idx] >= sp.sprout_pre_trace {
                hot_pre.push(idx);
            }
            if self.post_trace[idx] >= sp.sprout_post_trace {
                hot_post.push(idx);
            }
        }
        let mut sprouted: u32 = 0;
        // Cheap deterministic pairing: walk hot_pre × hot_post, skip
        // existing edges and self-loops, cap at `max_new_per_step`.
        'outer: for &pre in &hot_pre {
            for &post in &hot_post {
                if sprouted >= sp.max_new_per_step {
                    break 'outer;
                }
                if pre == post {
                    continue;
                }
                let exists = self.outgoing[pre]
                    .iter()
                    .any(|&eid| self.synapses[eid as usize].post == post);
                if exists {
                    continue;
                }
                let _new_eid = self.connect(pre, post, sp.sprout_initial);
                sprouted += 1;
            }
        }
    }

    /// Drive a brief offline replay/consolidation round through this
    /// network, biasing it towards the strongest already-formed
    /// engrams. Plasticity stays on, so weights consolidate exactly
    /// the way they would during a waking re-experience of the same
    /// activity pattern.
    pub fn consolidate(&mut self, params: &crate::replay::ReplayParams) {
        let n = self.neurons.len();
        if n == 0 {
            return;
        }
        // Rank E neurons by incoming excitatory weight sum.
        let mut order: Vec<(usize, f32)> = (0..n)
            .filter(|&i| self.neurons[i].kind == NeuronKind::Excitatory)
            .map(|i| {
                let s: f32 = self.incoming[i]
                    .iter()
                    .map(|&eid| {
                        let s = &self.synapses[eid as usize];
                        if self.neurons[s.pre].kind == NeuronKind::Excitatory {
                            s.weight
                        } else {
                            0.0
                        }
                    })
                    .sum();
                (i, s)
            })
            .collect();
        order.sort_by(|a, b| b.1.total_cmp(&a.1));
        let k = (params.top_k as usize).min(order.len());
        if k == 0 {
            return;
        }
        let mut chosen: Vec<usize> = order.into_iter().take(k).map(|(i, _)| i).collect();
        if params.alternate_reverse && self.replay_flip {
            chosen.reverse();
        }
        self.replay_flip = !self.replay_flip;

        let prev_modulator = self.neuromodulator;
        if params.neuromod_during != 0.0 {
            self.neuromodulator = params.neuromod_during;
        }

        let pulse_steps = (params.pulse_ms / self.dt).max(1.0) as usize;
        let gap_steps = (params.gap_ms / self.dt).max(0.0) as usize;
        let mut external = vec![0.0_f32; n];
        let zero = vec![0.0_f32; n];
        for &idx in &chosen {
            external[idx] = params.drive_current;
            for _ in 0..pulse_steps {
                let _ = self.step(&external);
            }
            external[idx] = 0.0;
            for _ in 0..gap_steps {
                let _ = self.step(&zero);
            }
        }

        self.neuromodulator = prev_modulator;
    }

    /// Compact the `synapses` vector by removing every dead slot. Every
    /// `usize` / `u32` index that previously pointed into `synapses`
    /// gets rewritten — this is the *only* operation in the crate that
    /// invalidates pre-existing synapse IDs, so callers must drop any
    /// stale handles before calling. Returns the number of synapses
    /// that were dropped.
    pub fn compact_synapses(&mut self) -> usize {
        if self.dead_synapses == 0 {
            return 0;
        }
        let m = self.synapses.len();
        let mut new_id: Vec<i32> = vec![-1; m];
        let mut new_synapses: Vec<Synapse> = Vec::with_capacity(m);
        let mut new_eligibility: Vec<f32> = if !self.eligibility.is_empty() {
            Vec::with_capacity(m)
        } else {
            Vec::new()
        };
        let mut new_counters: Vec<PruneCounter> = if !self.prune_counters.is_empty() {
            Vec::with_capacity(m)
        } else {
            Vec::new()
        };
        for (eid, slot) in new_id.iter_mut().enumerate().take(m) {
            let alive = self
                .prune_counters
                .get(eid)
                .map(|c| !c.dead)
                .unwrap_or(true);
            if alive {
                *slot = new_synapses.len() as i32;
                new_synapses.push(self.synapses[eid]);
                if !self.eligibility.is_empty() {
                    new_eligibility.push(self.eligibility[eid]);
                }
                if !self.prune_counters.is_empty() {
                    new_counters.push(self.prune_counters[eid]);
                }
            }
        }
        let dropped = m - new_synapses.len();
        self.synapses = new_synapses;
        if !self.eligibility.is_empty() {
            self.eligibility = new_eligibility;
        }
        if !self.prune_counters.is_empty() {
            self.prune_counters = new_counters;
        }
        // Rewrite adjacency buckets to point at the new indices.
        for bucket in self.outgoing.iter_mut() {
            bucket.retain_mut(|eid| {
                let mapped = new_id[*eid as usize];
                if mapped < 0 {
                    return false;
                }
                *eid = mapped as u32;
                true
            });
        }
        for bucket in self.incoming.iter_mut() {
            bucket.retain_mut(|eid| {
                let mapped = new_id[*eid as usize];
                if mapped < 0 {
                    return false;
                }
                *eid = mapped as u32;
                true
            });
        }
        self.dead_synapses = 0;
        dropped
    }

    /// Build a fresh, at-rest [`NetworkState`] sized for this network.
    /// Membrane potentials sit at `v_rest`, all refractory clocks at
    /// `-inf`, every channel and trace at zero. NMDA/GABA buffers are
    /// allocated only if any synapse uses that kind — same lazy
    /// policy as the in-place [`Self::step`] path.
    pub fn fresh_state(&self) -> NetworkState {
        let n = self.neurons.len();
        let need_nmda = self.synapses.iter().any(|s| s.kind == SynapseKind::Nmda);
        let need_gaba = self.synapses.iter().any(|s| s.kind == SynapseKind::Gaba);
        NetworkState {
            v: self.neurons.iter().map(|n| n.params.v_rest).collect(),
            refractory_until: vec![f32::NEG_INFINITY; n],
            last_spike: vec![f32::NEG_INFINITY; n],
            i_syn: vec![0.0; n],
            i_syn_nmda: if need_nmda { vec![0.0; n] } else { Vec::new() },
            i_syn_gaba: if need_gaba { vec![0.0; n] } else { Vec::new() },
            total_input: vec![0.0; n],
            time: 0.0,
            step_counter: 0,
            synapse_events: 0,
        }
    }

    /// Plasticity-free read-only step.
    ///
    /// Mathematically equivalent to [`Self::step`] when STDP / iSTDP /
    /// homeostasis are all disabled, but reads weights from `&self`
    /// without ever mutating them. All transient state lives in
    /// `state`, so a single `Network` can be stepped from multiple
    /// concurrent contexts as long as each holds its own
    /// [`NetworkState`].
    ///
    /// Plasticity is *unconditionally* skipped here regardless of
    /// what the parent `Network` has enabled — recall paths get the
    /// same dynamics whether the brain is in a "training" or
    /// "frozen" configuration.
    pub fn step_immutable(&self, state: &mut NetworkState, external: &[f32]) -> Vec<usize> {
        let dt = self.dt;
        let t = state.time;
        let n = self.neurons.len();

        // 1) Decay synaptic channels. Each loop is a flat `*x *= scalar`
        //    walk that the autovectoriser turns into AVX2/AVX-512 multiplies.
        let decay_ampa = (-dt / self.tau_syn_ms.max(1e-3)).exp();
        for x in state.i_syn.iter_mut() {
            *x *= decay_ampa;
        }
        let nmda_active = !state.i_syn_nmda.is_empty();
        if nmda_active {
            let decay = (-dt / self.tau_nmda_ms.max(1e-3)).exp();
            for x in state.i_syn_nmda.iter_mut() {
                *x *= decay;
            }
        }
        let gaba_active = !state.i_syn_gaba.is_empty();
        if gaba_active {
            let decay = (-dt / self.tau_gaba_ms.max(1e-3)).exp();
            for x in state.i_syn_gaba.iter_mut() {
                *x *= decay;
            }
        }

        // 1b) Pre-sum the input channels into `total_input` so the
        //     LIF loop iterates over a single contiguous slice. Hoists
        //     the NMDA/GABA presence checks out of the per-neuron
        //     inner loop and gives the autovectoriser a clean
        //     fused-multiply-add target on three input streams.
        //
        //     The `external` slice is allowed to be shorter than `n`
        //     (callers pass `&[]` for "no drive"); we resolve that
        //     once into a "fully sized or short" branch so the inner
        //     loop has no per-iteration bounds checks.
        if state.total_input.len() != n {
            state.total_input.resize(n, 0.0);
        }
        let i_syn = state.i_syn.as_slice();
        let nmda = state.i_syn_nmda.as_slice();
        let gaba = state.i_syn_gaba.as_slice();
        let total_input = state.total_input.as_mut_slice();
        let ext_full = external.len() >= n;
        match (ext_full, nmda_active, gaba_active) {
            (true, false, false) => {
                for i in 0..n {
                    total_input[i] = external[i] + i_syn[i];
                }
            }
            (true, true, false) => {
                for i in 0..n {
                    total_input[i] = external[i] + i_syn[i] + nmda[i];
                }
            }
            (true, false, true) => {
                for i in 0..n {
                    total_input[i] = external[i] + i_syn[i] + gaba[i];
                }
            }
            (true, true, true) => {
                for i in 0..n {
                    total_input[i] = external[i] + i_syn[i] + nmda[i] + gaba[i];
                }
            }
            (false, _, _) => {
                // Slow path: external too short. Fall back to bounds-
                // checked access; only happens during snapshot-load
                // smoke runs and similar.
                for i in 0..n {
                    let mut total = i_syn[i] + external.get(i).copied().unwrap_or(0.0);
                    if nmda_active {
                        total += nmda[i];
                    }
                    if gaba_active {
                        total += gaba[i];
                    }
                    total_input[i] = total;
                }
            }
        }

        // 2) LIF integration. Inputs all live in one slice now; the
        //    inner loop is a single straight-line walk over the
        //    per-neuron state vectors. Branches are limited to the
        //    refractory and threshold checks, both well-predicted
        //    when the network is in its asynchronous-irregular regime.
        let v_buf = state.v.as_mut_slice();
        let refr_buf = state.refractory_until.as_mut_slice();
        let last_buf = state.last_spike.as_mut_slice();
        let mut fired: Vec<usize> = Vec::with_capacity(64);
        for idx in 0..n {
            let p = &self.neurons[idx].params;
            let v = &mut v_buf[idx];
            let refr = &mut refr_buf[idx];
            if t < *refr {
                *v = p.v_reset;
                continue;
            }
            let dv = dt / p.tau_m * (-(*v - p.v_rest) + p.r_m * total_input[idx]);
            *v += dv;
            if *v >= p.v_threshold {
                *v = p.v_reset;
                *refr = t + p.refractory;
                last_buf[idx] = t;
                fired.push(idx);
            }
        }

        // 3) Deliver synaptic effects. No plasticity, no homeostasis,
        //    no traces. Lazy NMDA/GABA buffer allocation just like
        //    `step` — `state.ensure_channel` mirrors `Self::ensure_channel`.
        for &src in &fired {
            let src_kind = self.neurons[src].kind;
            let sign: f32 = match src_kind {
                NeuronKind::Excitatory => 1.0,
                NeuronKind::Inhibitory => -1.0,
            };
            for &eid in &self.outgoing[src] {
                let s = &self.synapses[eid as usize];
                let channel = state.ensure_channel(s.kind, n);
                channel[s.post] += sign * s.weight;
                state.synapse_events += 1;
            }
        }

        state.step_counter = state.step_counter.wrapping_add(1);
        state.time += dt;
        fired
    }

    /// Multiplicative homeostatic scaling of every excitatory incoming
    /// synapse, per post-neuron. Pure scalar multiplication preserves
    /// the relative weight pattern shaped by STDP.
    ///
    /// `factor_i = 1 + eta * (A_target - A_trace_i)`, then
    /// `w_ij = clamp(w_ij * factor_i)` for every excitatory pre `j`.
    fn apply_synaptic_scaling(&mut self, h: &HomeostasisParams) {
        let (w_min, w_max) = match self.stdp {
            Some(s) => (s.w_min, s.w_max),
            None => (0.0, f32::MAX),
        };

        let n = self.neurons.len();
        for post in 0..n {
            let trace = self.activity_trace[post];
            // Guard against very hyperactive neurons producing a negative
            // `factor` — that would push w * factor below zero and the
            // clamp to [w_min, …] would zero out *all* of the post's
            // incoming weights uniformly, destroying their relative
            // pattern. Clamping the factor to be non-negative keeps the
            // scaling well-defined even in extreme regimes.
            let factor_raw = 1.0 + h.eta_scale * (h.a_target - trace);
            let factor = if h.scale_only_down {
                factor_raw.clamp(0.0, 1.0)
            } else {
                factor_raw.max(0.0)
            };
            // Skip if no-op — saves the inner loop entirely.
            if factor == 1.0 {
                continue;
            }
            let n_in = self.incoming[post].len();
            for i in 0..n_in {
                let eid = self.incoming[post][i] as usize;
                let pre = self.synapses[eid].pre;
                if self.neurons[pre].kind != NeuronKind::Excitatory {
                    continue;
                }
                let new_w = (self.synapses[eid].weight * factor).clamp(w_min, w_max);
                self.synapses[eid].weight = new_w;
            }
        }
    }
}

// ------------------------------------------------------------------------
// Read-only step support: NetworkState carries every field that `step()`
// mutates, so the same `Network` can be stepped concurrently by giving
// each caller its own state.
// ------------------------------------------------------------------------

/// Per-step transient state, owned by the caller. Has the same shape
/// as the `#[serde(skip)]` fields of [`Network`] plus the per-neuron
/// transient state that normally lives inside [`LifNeuron`]. Allocate
/// one with [`Network::fresh_state`].
#[derive(Clone, Debug)]
pub struct NetworkState {
    /// Per-neuron membrane potential.
    pub v: Vec<f32>,
    pub refractory_until: Vec<f32>,
    pub last_spike: Vec<f32>,
    /// AMPA synaptic current channel.
    pub i_syn: Vec<f32>,
    /// NMDA channel — empty until first NMDA delivery.
    pub i_syn_nmda: Vec<f32>,
    /// GABA channel — empty until first GABA delivery.
    pub i_syn_gaba: Vec<f32>,
    /// Scratch buffer holding `external + i_syn (+ i_syn_nmda) (+ i_syn_gaba)`
    /// for the duration of one [`Network::step_immutable`] call. Pre-summing
    /// the input channels lets the LIF integration loop iterate over a
    /// single contiguous slice — autovectoriser-friendly, and removes
    /// branchy `Option::get` accesses from the inner loop.
    pub total_input: Vec<f32>,
    /// Sim clock.
    pub time: f32,
    pub step_counter: u64,
    pub synapse_events: u64,
}

impl NetworkState {
    /// Borrow the channel slot for `kind`, allocating on first touch.
    /// Mirrors [`Network::ensure_channel`] but operates on the
    /// caller-owned buffers.
    fn ensure_channel(&mut self, kind: SynapseKind, n: usize) -> &mut Vec<f32> {
        match kind {
            SynapseKind::Ampa => &mut self.i_syn,
            SynapseKind::Nmda => {
                if self.i_syn_nmda.len() != n {
                    self.i_syn_nmda = vec![0.0; n];
                }
                &mut self.i_syn_nmda
            }
            SynapseKind::Gaba => {
                if self.i_syn_gaba.len() != n {
                    self.i_syn_gaba = vec![0.0; n];
                }
                &mut self.i_syn_gaba
            }
        }
    }
}
