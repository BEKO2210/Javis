//! Reward-aware pair-association benchmark.
//!
//! The classic Javis benchmark (`scale_bench`) measures correlation-only
//! retrieval: train each sentence once, fingerprint every word, compare
//! decoded engrams to ground-truth co-occurrence. Pure STDP is enough
//! for that task because the network never has to *prefer* one
//! association over another — it just stores whatever co-occurrences it
//! sees.
//!
//! The iter-44 reward stack only earns its keep on a different kind of
//! problem: the network sees both *correct* and *distractor* pairings,
//! and a global modulator tells it which were good. Three-factor
//! learning (Frémaux & Gerstner 2016) is the only mechanism in the
//! current Javis stack that can use that signal.
//!
//! ## Task
//!
//! - Build a fixed list of `(cue, target)` pairs — one association per
//!   row.
//! - Optionally inject **noise pairs**: swapped or randomised cue/target
//!   pairings that pure STDP would happily memorise alongside the real
//!   ones.
//! - Train the brain by presenting `cue + target` SDRs simultaneously
//!   (paired-input training). STDP + iSTDP shape the connection from
//!   the cue assembly to the target assembly.
//! - **For each trial**, measure: drive the cue alone, decode top-3,
//!   check whether the *correct* target made it into the result.
//! - **If reward learning is on**: set `Brain::set_neuromodulator(±1)`
//!   based on whether the trial was correct (or, for noise pairs,
//!   whether the brain *did not* fall for the distractor), then run a
//!   short consolidation window so the eligibility tag turns into an
//!   actual weight update.
//!
//! ## What the benchmark expects to show
//!
//! - Pure STDP: the brain learns *every* association it sees, including
//!   the noise pairs, because the rule is purely correlational.
//! - R-STDP: the dopamine signal is positive on correct trials,
//!   negative on noise-driven trials → eligibility-tagged weights from
//!   the noise pairs get suppressed → the test accuracy on correct
//!   pairs grows over epochs while the noise associations fade.
//!
//! The harness logs per-epoch top-1 / top-3 accuracy and mean reward
//! so the two regimes are directly comparable.

use std::collections::BTreeSet;

use encoders::{EngramDictionary, TextEncoder};
use snn_core::{
    Brain, HomeostasisParams, IStdpParams, IntrinsicParams, LifNeuron, LifParams, NeuronKind,
    Region, RewardParams, Rng, StdpParams,
};

/// Topology — a fast 2 000-neuron R2 keeps the benchmark wall-time
/// under a minute per epoch on a release build, while still being
/// representative of the production R2.
const DT: f32 = 0.1;
const R1_N: usize = 1000;
const R2_N: usize = 2_000;
/// Iter-48: 0.20 → 0.30. Fraction of R2 cells that are inhibitory.
/// Iter-47a postmortem found oscillatory recurrent bursting at the
/// "interesting" sweep point and 0/30 canonical-target hits in
/// 12 001 R2-E spikes — classic Vogels-Sprekeler 2011 EI-imbalance
/// signature. Increasing the inhibitory pool from 20 % to 30 %
/// gives iSTDP roughly 1.5× the inhibitory budget to dampen the
/// avalanche; combined with the iSTDP timing fix below.
const R2_INH_FRAC: f32 = 0.30;
const R2_P_CONNECT: f32 = 0.05;
const FAN_OUT: usize = 12;
/// Iter-47a-2 sweep finding (notes/47a + notes/47a-postmortem):
///
/// - 2.0 (iter-46 baseline): 90–180 active R2 cells, margin negative.
/// - 0.5: signal-death, 0.8–3 cells, no learning substrate.
/// - 1.0: 96–139 cells, monotone learning signal in epochs 0-3
///   (target_hit 1.16 → 2.59, selectivity -0.022 → -0.0005), then
///   COLLAPSED in epochs 5–15 (selectivity → -0.045, target_hit →
///   0.08). Postmortem (b)+(c): adaptive θ effect size is tiny
///   (mean = 0.05 mV, max = 0.86 mV, frac > 1mV = 0 — << LIF
///   15 mV swing); the collapse is driven by STDP/iSTDP imbalance,
///   not by θ over-correction.
/// - 0.7: BISTABLE — epochs 0-2 stable at ~10 cells, epoch 3
///   explodes (mean 507 / p90 1599). Postmortem (b)+(c): per-step
///   trace shows synchronised oscillatory bursting (10 / 54 / …
///   / 189 / 10 / …, early-vs-late ratio 0.97) with 0 canonical-
///   target hits; θ jumps reactively from 0.03 → 2.84 mV in the
///   cascade epoch (99.9 % of cells > 1 mV) but only AFTER the
///   blow-up — Diehl-Cook is reactive, not preventive.
///
/// Held at 1.0 because that is the most stable sweep point with
/// at least transient learning. iter-48 entry from postmortem
/// data: tighter iSTDP (Vogels 2011) for fast EI balance, NOT
/// per-step k-WTA — the failure mode is oscillatory bursting,
/// not onset-burst.
const INTER_WEIGHT: f32 = 1.0;
const INTER_DELAY_MS: f32 = 2.0;
const ENC_N: u32 = R1_N as u32;
const ENC_K: u32 = 20;
const DRIVE_NA: f32 = 200.0;
/// Cue is presented alone for `CUE_LEAD_MS`, then cue + target
/// for `OVERLAP_MS`, then target alone for `TARGET_TAIL_MS`. This
/// staggered schedule gives STDP a clean pre-before-post timing
/// asymmetry — cue cells lead, target cells follow — so the rule
/// grows `cue → target` synapses preferentially.
const CUE_LEAD_MS: f32 = 40.0;
const OVERLAP_MS: f32 = 30.0;
const TARGET_TAIL_MS: f32 = 30.0;
const RECALL_MS: f32 = 40.0;
const CONSOLIDATION_MS: f32 = 80.0;
const COOLDOWN_MS: f32 = 20.0;
const KWTA_K: usize = 60;

/// One supervised cue → target association.
#[derive(Debug, Clone)]
pub struct RewardPair {
    pub cue: String,
    pub target: String,
}

/// Curriculum for the reward benchmark.
#[derive(Debug, Clone)]
pub struct RewardCorpus {
    /// The "correct" associations the brain is supposed to learn.
    pub pairs: Vec<RewardPair>,
    /// Distractor associations injected into training (typically
    /// shuffles of the real cues / targets). Pure STDP would memorise
    /// these alongside the real pairs; R-STDP with negative reward on
    /// noise trials should suppress them.
    pub noise_pairs: Vec<RewardPair>,
    /// Every word that participates in any pair. The benchmark scores
    /// against this vocabulary.
    pub vocab: BTreeSet<String>,
}

/// Per-epoch readout. Iter-45 fields are the ones the original
/// table used; iter-46 adds the diagnostic metrics needed to
/// distinguish "the brain learnt the association" from "the
/// teacher activated the target during training" — the
/// distinction the original harness could not make.
#[derive(Debug, Clone, Default)]
pub struct RewardEpochMetrics {
    pub epoch: usize,
    /// Fraction of `pairs` whose target is the *single* top-decoded
    /// engram when the cue is presented alone.
    pub top1_accuracy: f32,
    /// Fraction whose target is in the top-3 decoded engrams.
    pub top3_accuracy: f32,
    /// Average dopamine value emitted across the epoch (only
    /// meaningful when `use_reward = true`; included as 0.0 for the
    /// pure-STDP arm so the columns line up).
    pub mean_reward: f32,
    /// How many of the `noise_pairs` distractor targets ended up in
    /// the top-3 decoded set when the noise *cue* was presented.
    /// Lower is better — the brain should *not* learn the distractor
    /// associations.
    pub noise_top3_rate: f32,

    // ---- iter-46 diagnostics ------------------------------------
    /// Random top-3 baseline — `decode_k / vocab_size`. Anything
    /// at or below this is statistically indistinguishable from
    /// chance.
    pub random_top3_baseline: f32,
    /// Mean rank of the correct target across all real pairs.
    /// 1.0 = perfect, larger = worse. NaN if no decoder hits.
    pub mean_rank: f32,
    /// Mean reciprocal rank — the right metric when "is the answer
    /// somewhere in the list" matters more than top-k binary.
    pub mrr: f32,
    /// Fraction of teacher-clamped target neurons that actually
    /// fired during the teacher phase. < 0.5 means the clamp is
    /// not strong enough to override R1's forward drive.
    pub target_clamp_hit_rate: f32,
    /// Top-3 fraction measured *during the prediction phase*,
    /// before teacher firing. The cleanest signal of "the brain
    /// learnt the association without help".
    pub prediction_top3_before_teacher: f32,
    /// Number of synapses with a non-zero eligibility tag at the
    /// end of the epoch. > 0 means R-STDP machinery is alive.
    pub eligibility_nonzero_count: u32,
    /// Mean R2 → R2 weight at the end of the epoch.
    pub r2_recurrent_weight_mean: f32,
    /// Max R2 → R2 weight (a runaway-LTP early-warning).
    pub r2_recurrent_weight_max: f32,
    /// Mean active R2-E units when *cue alone* is presented at
    /// final eval. Compared against KWTA_K to spot runaway
    /// activity.
    pub active_r2_units_per_cue: f32,
    /// `correct_pair_weight_mean − incorrect_pair_weight_mean` —
    /// the cleanest single number for "is the brain
    /// preferentially strengthening the *right* associations?".
    pub correct_minus_incorrect_margin: f32,
    /// Wall-time of the final-eval decoder pass (microseconds).
    /// Useful to see whether iter-46 is slower than iter-45's
    /// dictionary-rebuild.
    pub decoder_micros: u128,

    // ---- iter-47 sparsity diagnostics ----------------------------
    /// Mean number of R2-E cells active during the prediction
    /// phase (per real pair, averaged over the epoch). Used by
    /// the iter-47 acceptance band [25, 70].
    pub r2_active_pre_teacher_mean: f32,
    /// 10th percentile across the epoch's real-pair trials.
    /// `< 5` is the iter-47 "signal-death" warning.
    pub r2_active_pre_teacher_p10: u32,
    /// 90th percentile. `> 100` is the iter-47 "drive-too-strong"
    /// warning.
    pub r2_active_pre_teacher_p90: u32,
    /// Mean count of canonical-target neurons that fired during
    /// the prediction phase, before the clamp. Direct readout of
    /// "did the cue alone reach the right cells?".
    pub target_hit_pre_teacher_mean: f32,
    /// Selectivity index — Diehl-Cook-style normalisation:
    /// `target_hit / |target| − non_target_active / (|R2_E| − |target|)`
    /// `> 0` ⇒ targets fire over-proportionally;
    /// `= 0` ⇒ uniform across the population;
    /// `< 0` ⇒ targets fire under-proportionally (the iter-46
    ///         symptom that iter-47 must flip).
    pub selectivity_index: f32,

    // ---- iter-48 cascade-stability diagnostics ------------------
    /// 99th percentile of `prediction_active_count` across the
    /// epoch's real-pair trials. Iter-47a postmortem identified
    /// avalanches as the dominant failure mode — p99 is a tight
    /// upper bound that catches them while p90 still looks clean.
    /// Iter-48 acceptance criterion: `< 50`.
    pub r2_active_pre_teacher_p99: u32,
    /// Mean threshold offset (`v_thresh_offset`) over R2-E
    /// **inhibitory** cells at the end of the epoch. Iter-48
    /// early warning: if iSTDP over-corrects, the inhibitory
    /// pool will silence itself faster than the excitatory pool,
    /// and θ_inh will drift toward zero (or the floor) faster
    /// than θ_exc. Read together with `r2_recurrent_weight_mean`
    /// to spot the iSTDP runaway.
    pub theta_inh_mean: f32,
    /// Mean threshold offset over R2-E **excitatory** cells —
    /// the comparison reference for `theta_inh_mean`. The diff
    /// (`theta_inh_mean - theta_exc_mean`) is the iter-48 EI-
    /// balance early-warning number.
    pub theta_exc_mean: f32,
}

/// Iter-53: trial-to-trial decoder consistency / specificity
/// metrics. Two orthogonal measurements on a 32-cue × 3-trial
/// matrix (one row per vocab entry, three reps with full
/// `brain.reset_state()` between):
///
/// - `same_cue_*`  — for each cue, Jaccard between Trial 2 and
///   Trial 3's top-3 word sets. Trial 1 is dropped as burn-in
///   per the iter-53.0 smoke (the first trial after the
///   dictionary build sees residual R1 / pending-queue state).
///   High = the engram triggered by this cue is consistent.
/// - `cross_cue_*` — for every cue pair `(i, j), i < j`, Jaccard
///   between `matrix[i][1]` and `matrix[j][1]` (post-burn-in
///   trials). Low = each cue activates a *cue-specific* pattern.
///   High = mode collapse (all cues point at the same engram).
///
/// Acceptance criterion for iter-53 is the **difference of
/// differences**: trained `same_cue_mean` must rise *and*
/// trained `cross_cue_mean` must fall, both significantly
/// versus untrained. Single-axis improvement is ambiguous
/// (high same-cue alone is satisfied by mode collapse).
#[derive(Debug, Clone, Default)]
pub struct JaccardMetrics {
    /// Mean Jaccard across cues of `Jaccard(trial[1], trial[2])`.
    pub same_cue_mean: f32,
    /// Sample standard deviation of the per-cue Jaccards.
    pub same_cue_std: f32,
    /// Mean Jaccard across cue pairs of
    /// `Jaccard(matrix[i][1], matrix[j][1])` for `i < j`.
    pub cross_cue_mean: f32,
    /// Sample standard deviation of the per-pair Jaccards.
    pub cross_cue_std: f32,
    /// Number of cues with a non-empty 3-trial row.
    pub n_cues: usize,
    /// Number of unique cue pairs sampled by the cross-cue mean.
    pub n_pairs: usize,
}

/// Iter-53 per-arm result. `arm` is `"trained"` or `"untrained"`.
#[derive(Debug, Clone)]
pub struct JaccardArmResult {
    pub seed: u64,
    pub arm: &'static str,
    pub jaccard: JaccardMetrics,
}

/// Iter-53 sweep result aggregating both arms over a list of
/// seeds. The two `Vec`s are indexed in lock-step by seed order.
#[derive(Debug, Default, Clone)]
pub struct JaccardSweepResult {
    pub trained: Vec<JaccardArmResult>,
    pub untrained: Vec<JaccardArmResult>,
}

/// Iter-58 per-cue-pair Jaccard sample. One entry per
/// `(cue_a, cue_b)` pair with `i < j` in the vocab order, plus
/// the post-burn-in trial\[1\] decoded top-3 sets so the floor-
/// diagnosis can spot encoder collisions ("which cues share
/// most of their top-3 lists").
#[derive(Debug, Clone)]
pub struct JaccardPairSample {
    pub cue_a: String,
    pub cue_b: String,
    pub jaccard: f32,
    pub top_a: Vec<String>,
    pub top_b: Vec<String>,
}

/// Iter-58 floor-diagnosis report for one (seed, trained-arm)
/// run at a fixed config. Emits the full per-pair Jaccard list
/// alongside the standard aggregate metrics so the caller can
/// inspect the *distribution* (min/p25/median/p75/p90/p95/max)
/// and identify the cue pairs that drive the residual cross-cue
/// floor.
#[derive(Debug, Clone)]
pub struct JaccardFloorReport {
    pub seed: u64,
    pub n_cues: usize,
    pub n_pairs: usize,
    pub same_cue_mean: f32,
    pub cross_cue_mean: f32,
    pub per_pair: Vec<JaccardPairSample>,
}

/// Iter-49 sweep modes — three orthogonal interventions on the
/// iter-48 iSTDP collapse mechanism (notes/48-saturation.md). All
/// modes share Config 2 as base: `--istdp-during-prediction = true`.
///
/// - `WmaxCap`     — symptom: cap inhibitory weight at `w_max = 2.0`
///   (vs iter-48's 8.0). Removes the structural ceiling against
///   which iSTDP LTP runs unopposed.
/// - `APlusHalf`   — dynamic: halve `a_plus` from 0.30 to 0.20
///   (half-way back to iter-47a's 0.10). Slows wall growth so the
///   collapse threshold is reached later (or not at all).
/// - `ActivityGated` — temporal: `a_plus = 0` for the first
///   `gated_warmup_epochs` (default 2), then ramp linearly to the
///   full 0.30 over `gated_ramp_epochs` (default 2), full thereafter.
///   Match the literature pattern "consolidate first, balance later".
///
/// `None` reproduces the iter-48 iSTDP exactly (committed defaults).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Iter49Mode {
    None,
    WmaxCap,
    APlusHalf,
    ActivityGated,
}

impl Iter49Mode {
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "iter48-baseline",
            Self::WmaxCap => "wmax-cap-2.0",
            Self::APlusHalf => "a_plus-half-0.20",
            Self::ActivityGated => "activity-gated-ramp",
        }
    }
}

/// Iter-46 teacher-forcing configuration. When attached to a
/// [`RewardConfig`], the trial schedule switches from "drive cue
/// then target through R1" (iter-45) to a clean six-phase cycle:
///
/// 1. **cue** (`cue_ms`): cue SDR drives R1 → forward R1→R2 fires.
/// 2. **delay** (`delay_ms`): no input.
/// 3. **prediction** (`prediction_ms`): cue alone again, plasticity
///    OFF (controlled by `plasticity_during_prediction`); the
///    network's spontaneous top-k is captured for diagnostics and
///    reward gating.
/// 4. **teacher** (`teacher_ms`): cue on R1 *plus* the canonical
///    target SDR clamped *directly into R2-E* with current
///    `target_clamp_strength`. Plasticity ON
///    (`plasticity_during_teacher`) — STDP / R-STDP get clean
///    pre→post coincidences between cue-driven R2 cells and
///    teacher-clamped target R2 cells.

// ----------------------------------------------------------------
// Iter-60 — DG pattern-separation bridge config.
// See `TeacherForcingConfig::dg` for the full description.
// ----------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct DgConfig {
    /// `true` enables the DG bridge (adds a third region named
    /// "DG" to the brain, computes per-cue k-of-n SDRs, drives
    /// DG cells alongside R1 on every cue presentation).
    pub enabled: bool,
    /// Number of DG cells. Default 4000 — gives ample room for
    /// orthogonal per-cue addresses at vocab=64 (62 cells/cue
    /// at the default `k = 80`, 2 % sparsity).
    pub size: u32,
    /// Active DG cells per cue (the "k" of the k-of-n hash).
    /// Default 80, ≈ 2 % of `size = 4000`.
    pub k: u32,
    /// Each DG cell's fan-out into R2-E cells. Default 30 —
    /// dense enough to drive R2 reliably but bounded so the
    /// total DG → R2 edge count stays manageable
    /// (size × fanout = 120 000 edges at defaults).
    pub to_r2_fanout: u32,
    /// Weight on every DG → R2 connection. Default 1.0 —
    /// mossy-fibre-style strong projection.
    pub to_r2_weight: f32,
    /// Multiplier on the existing R1 → R2 connection weights.
    /// `0.0` disables the direct path entirely (DG becomes the
    /// only R2 input). `1.0` keeps the iter-46…59 baseline.
    /// Default 0.0 — the iter-60 smoke wants DG as the primary
    /// cue-routing path.
    pub direct_r1r2_weight_scale: f32,
    /// External current injected into DG SDR cells when a cue
    /// is presented. Default `DRIVE_NA = 200.0`, matching the
    /// R1-side cue injection.
    pub drive_strength: f32,
}

impl Default for DgConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            size: 4000,
            k: 80,
            to_r2_fanout: 30,
            to_r2_weight: 1.0,
            direct_r1r2_weight_scale: 0.0,
            drive_strength: DRIVE_NA,
        }
    }
}

// ----------------------------------------------------------------
// Iter-66 — CA1-equivalent C1 readout config (Mechanism M1).
// See `notes/66-ca1-heteroassoc-readout.md` for the locked
// pre-registration. C1 cells live as an *appended* index range
// inside the R2 region's `Network` (indices
// `[r2_n_used, r2_n_used + size)`); this keeps the existing
// R-STDP plumbing applicable to R2-E → C1 synapses without
// adding plastic inter-region edges to snn-core. The "C1 region"
// terminology in the iter-66 ENTRY refers to this logical sub-
// region; physically there is one extended R2 `Network`. The
// `r2_e` set is captured *before* C1 cells are appended, so all
// existing R2-side metrics keep their iter-46/63 numerics
// bit-identically when `c1_readout = false`.
// ----------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct C1Config {
    /// `true` enables the C1 readout layer (appends `size` LIF
    /// excitatory neurons to the R2 region's `Network` and wires
    /// R2-E → C1 with sparse fan-out plastic synapses).
    pub enabled: bool,
    /// Number of C1 cells. iter-66 ENTRY locked at 1000.
    pub size: u32,
    /// kWTA window for C1 fingerprints (matches R1/target SDR k).
    /// iter-66 ENTRY locked at 20.
    pub sparsity_k: u32,
    /// Per-R2-E source cell, how many C1 targets to wire.
    /// iter-66 ENTRY locked at 30.
    pub from_r2_fanout: u32,
    /// R2-E → C1 initial weight is sampled uniformly from
    /// `(0, init_w_max)`. iter-66 ENTRY locked at 0.5
    /// (= `w_max / 2` with `w_max = 1.0`).
    pub init_w_max: f32,
    /// Strength of the M_target neuromodulator pulse during the
    /// teacher phase. iter-66 ENTRY default 1.0 (CLI knob
    /// `--c1-teacher-strength`).
    pub teacher_strength: f32,
    /// Iter-66 step 7.5: emit per-epoch diagnostic logs that
    /// discriminate (A) insufficient training / (B) silent C1 /
    /// (C) non-discriminative fingerprints. Logs include:
    /// pre/post R2→C1 weight L2 norm, per-epoch C1 spike count
    /// during the teacher and eval phases, the fraction of
    /// trials with nonzero C1 activity, mean rank / MRR of the
    /// canonical target word in the C1 readout, and the raw
    /// overlap of the eval kWTA with the canonical-target C1
    /// SDR. Off by default (CLI flag `--c1-diagnostic`).
    pub diagnostic: bool,
    /// Iter-66.5 Path-1 fix (`notes/66.5-eval-aligned-c1-rstdp.md`).
    /// When `true`, the teacher Phase 4 omits the canonical R2
    /// target SDR from the clamp (R2 fires its natural
    /// cue-driven response instead). The C1 target SDR clamp
    /// and the M_target = `c1.teacher_strength` modulator pulse
    /// stay active. R-STDP on R2-E → C1 then aligns
    /// (eval-time R2 cue pattern) → (canonical C1 target),
    /// instead of iter-66's
    /// (canonical R2 target SDR) → (canonical C1 target SDR)
    /// which iter-66 step-7.5 falsified at recall time.
    /// Off by default ⇒ iter-66 behaviour bit-identical;
    /// CLI flag `--c1-eval-aligned-rstdp`.
    pub eval_aligned_rstdp: bool,
    /// Iter-67 BTSP plateau-eligibility rule on R2-E → C1.
    /// When `true`, `enable_btsp` is called on R2's network with
    /// the C1 cell index range as the participation filter
    /// (R2-R2 R-STDP / STDP / etc stay untouched). See
    /// `notes/67-btsp-tagged-eligibility-c1.md` for the locked
    /// pre-registration. Off by default ⇒ iter-66.5 behaviour
    /// bit-identical; CLI flag `--c1-btsp`.
    pub btsp: bool,
    /// Iter-67: per-synapse eligibility tag decay (ms). Locked
    /// default 200 ms covers the iter-46 6-phase cue + delay +
    /// prediction + teacher lead-in interval (~ 80 ms) with
    /// safety margin. CLI flag `--c1-btsp-window-ms`.
    pub btsp_window_ms: f32,
    /// Iter-67: per-tagged-pre-spike weight increment applied at
    /// the plateau-arm transition (`Δw = strength × tag`).
    /// Default 0.4 — two pre-spikes during the eligibility window
    /// saturate the synapse to `w_max = 0.8`. CLI flag
    /// `--c1-btsp-strength`.
    pub btsp_strength: f32,
    /// Iter-67: per-post-cell credit-assignment toggle. Default
    /// `true` — only the post-cell that crossed plateau receives
    /// retroactive potentiation on its incoming synapses.
    /// `false` (ablation) — ANY post-cell's plateau triggers
    /// potentiation on every tagged synapse network-wide. Use
    /// the ablation only to verify per-post-cell locality is
    /// the binding mechanism. CLI flag `--c1-btsp-target-gated`.
    pub btsp_target_gated: bool,
    /// Iter-67-γ.1: R2-R2 recurrent E-cell synapse delivery scale
    /// during the teacher Phase 4 clamp window.  Default `1.0` =
    /// full strength (iter-67-α v3 / iter-67-β at e=1.0).
    /// Applied via `Network::set_recurrent_e_i_scales` with
    /// `pre_max = r2_n_used` so R2-E → C1 synapses (post >=
    /// r2_n_used) are NOT scaled.  CLI flag
    /// `--c1-btsp-teacher-recurrent-e-scale`.
    pub btsp_teacher_recurrent_e_scale: f32,
    /// Iter-67-γ.1: R2-R2 recurrent I-cell (inhibitory) synapse
    /// delivery scale during the teacher Phase 4 clamp window.
    /// Default `0.3` — Bekos's locked γ.1 default per the prompt:
    /// reduce I-suppression while keeping E recurrent at full
    /// strength so the strongest cue-engram E-cells dominate
    /// without the recurrent attractor saturating uniformly.
    /// `1.0` = uniform with `e_scale = 1.0` (= iter-67-α v3 verbatim).
    /// `0.0` = inhibition off entirely (risk: runaway E firing).
    /// Plasticity rules read the un-scaled stored weight, so this
    /// only attenuates inhibitory current delivery — STDP / R-STDP /
    /// BTSP / iSTDP all see the architectural weight.  CLI flag
    /// `--c1-btsp-teacher-recurrent-i-scale`.
    pub btsp_teacher_recurrent_i_scale: f32,
    /// Iter-67-γ.1.1: opt-out switch for the iter-67-α2 R2-isolation
    /// (cue + DG drive cut to 0 during teacher).  Default `false`
    /// (= iter-67-α2 isolation ON, matches iter-67/γ.1 v4 baseline).
    /// `true`: keep cue + DG drive at full strength during teacher
    /// Phase 4 clamp window — R2 fires its natural cue-driven
    /// response so γ.1's E/I-split has an active substrate to
    /// expose.  Tests Bekos's actual locked γ.1 hypothesis where
    /// cue-engram E-cells fire under reduced inhibition.  CLI flag
    /// `--c1-btsp-no-r2-isolation`.
    pub btsp_no_r2_isolation: bool,
}

impl Default for C1Config {
    fn default() -> Self {
        Self {
            enabled: false,
            size: 1000,
            sparsity_k: 20,
            from_r2_fanout: 30,
            init_w_max: 0.5,
            teacher_strength: 1.0,
            diagnostic: false,
            eval_aligned_rstdp: false,
            btsp: false,
            btsp_window_ms: 200.0,
            btsp_strength: 0.4,
            btsp_target_gated: true,
            btsp_teacher_recurrent_e_scale: 1.0,
            btsp_teacher_recurrent_i_scale: 0.3,
            btsp_no_r2_isolation: false,
        }
    }
}

/// 5. **reward** (`reward_after_teacher`): the modulator is set
///    based on the *prediction* (not teacher). Teacher cells alone
///    must never count as a recall success.
/// 6. **tail** (`tail_ms`): no input, traces decay.
#[derive(Debug, Clone, Copy)]
pub struct TeacherForcingConfig {
    pub enabled: bool,
    pub cue_ms: u32,
    pub delay_ms: u32,
    pub prediction_ms: u32,
    pub teacher_ms: u32,
    pub tail_ms: u32,
    /// External current injected into each clamped R2-E neuron during
    /// the teacher phase (nA). Higher values → harder forcing, lower
    /// values let the network's own dynamics co-determine which
    /// target cells fire. ~ 200 nA is the same scale as the existing
    /// R1 forward drive.
    pub target_clamp_strength: f32,
    /// Optional refractory-style limit on the clamp: if non-zero,
    /// only every Nth ms gets the high current (lets target cells
    /// recover between forced spikes).
    pub target_clamp_spike_interval_ms: u32,
    /// Disable STDP / iSTDP during the prediction phase so the test
    /// does not contaminate the weights.
    pub plasticity_during_prediction: bool,
    /// Enable plasticity during the teacher phase. Almost always
    /// `true` — that's the whole point of teacher-forcing.
    pub plasticity_during_teacher: bool,
    /// Run reward delivery only after the teacher phase. Default
    /// `true`; setting it to `false` falls back to iter-45 timing.
    pub reward_after_teacher: bool,
    /// How many top-scoring decoded engrams the prediction phase
    /// reads. Default 3 — same k the epoch readout uses.
    pub wta_k: usize,
    /// Negative reward applied when the *wrong* target sneaks into
    /// the prediction's top-k. Drives R-STDP to actively suppress
    /// false candidates.
    pub negative_reward_for_false_topk: f32,
    pub positive_reward_for_correct: f32,
    pub noise_reward: f32,
    /// Re-normalise R2 → R2 weights to a fixed L2 norm at the end of
    /// every epoch. Helps stop weight blow-up under repeated
    /// teacher-forcing.
    pub homeostatic_normalization: bool,
    /// Print debug info for the first 1–3 example pairs of each
    /// arm. Set to 0 to disable debug output.
    pub debug_trials: u32,
    /// Iter-48 A/B knob: keep iSTDP active during the prediction
    /// phase. Default `false` preserves the iter-46 invariant
    /// "evaluation does not modify weights"; setting it `true`
    /// lets inhibitory plasticity respond to the cue *during*
    /// the read-out, which the iter-47a postmortem identified as
    /// the natural test for "does iSTDP catch the cascade in time
    /// or only after the trial". Excitatory STDP and R-STDP are
    /// still gated by `plasticity_during_prediction`; this flag
    /// only opens iSTDP separately.
    pub istdp_during_prediction: bool,
    /// Iter-49 sweep mode. Default `None` reproduces iter-48
    /// behaviour exactly. See [`Iter49Mode`] for the three
    /// candidate interventions on the iter-48 iSTDP collapse
    /// (notes/48-saturation.md).
    pub iter49_mode: Iter49Mode,
    /// Iter-49 ActivityGated mode: number of warmup epochs in
    /// which `a_plus = 0`. Default 2.
    pub gated_warmup_epochs: u32,
    /// Iter-49 ActivityGated mode: number of ramp epochs after
    /// warmup, during which `a_plus` rises linearly from 0 to
    /// the full value. Default 2.
    pub gated_ramp_epochs: u32,
    /// Iter-50 diagnostic: revert all iter-47/48/49 drift to
    /// reproduce the iter-46 Arm B baseline (R-STDP only, no
    /// teacher, no iSTDP-tightening, no Diehl-Cook, original
    /// INTER_WEIGHT 2.0 and R2_INH_FRAC 0.20). Used to answer
    /// the open question why iter-46 Arm B reported top-3 = 0.19
    /// while every iter-47/48/49 selectivity stays under 0.02.
    /// When `true`, also forces `iter49_mode = None`,
    /// `enabled = false`, and skips `enable_intrinsic_plasticity`.
    pub iter46_baseline: bool,
    /// Iter-52 untrained control: skip every `enable_*` plasticity
    /// call so the brain runs as a pure random-weight forward
    /// projection — no STDP, no R-STDP, no iSTDP, no homeostasis,
    /// no intrinsic plasticity. Forward LIF dynamics, recurrent
    /// spike propagation, and the decoder all stay live; only
    /// weight updates are gated. Used to answer "is any of the
    /// chain's measured top-3 actually a learning signal, or is
    /// it the forward-projection baseline plus noise?". Combine
    /// with `iter46_baseline = true` for the strict iter-46-Arm-B
    /// untrained variant.
    pub no_plasticity: bool,
    /// Iter-62 recall-mode: when `true`, every plasticity rule
    /// (STDP, iSTDP, homeostasis, intrinsic, reward learning,
    /// structural) is disabled *between training and the
    /// jaccard-matrix eval phase*. Training itself is unchanged.
    /// The dictionary build + jaccard matrix + every drive after
    /// training run on a frozen-weight brain. Equivalent in role
    /// to iter-52's `no_plasticity` but applied only to the eval
    /// half of the run; the iter-52 L2 bit-identity invariant is
    /// re-asserted across the eval phase when this is on.
    ///
    /// Tests whether the iter-61 trained-same erosion (2 of 4
    /// seeds < 0.90) and the +/-0.9 to +4.6 eval-drift L2 are
    /// caused by plasticity acting during recall, or by the
    /// trained dynamics themselves.
    pub recall_mode_eval: bool,
    /// Iter-54 hard-decorrelated R1 → R2 wiring. When `true`,
    /// `wire_forward_decorrelated` replaces the standard random
    /// wire-up: each vocab word's R1 SDR cells project *only* into
    /// a disjoint block of R2-E cells, with shared cells (R1 cells
    /// that appear in multiple cue SDRs) dropped from the
    /// connectivity graph. The result is mechanically pairwise-
    /// disjoint R2 reachability per cue — the iter-54 invariant
    /// `assert_decorrelated_disjoint` asserts this end-to-end.
    /// Default `false` keeps the iter-46/53 random-FAN_OUT topology.
    pub decorrelated_init: bool,
    /// During the prediction phase only, scale the R1 cue current
    /// by this factor. `1.0` = no gating (default); `0.3` halves
    /// the forward drive so the cue's R2 response is more easily
    /// dominated by recurrent associations.
    ///
    /// This addresses the iter-46 finding that even with a clean
    /// pre-before-post teacher schedule, the dominant R1 → R2
    /// forward path keeps the cue's R2 representation almost
    /// entirely random — recurrent learning has no room to bias
    /// it. The gate is a *training-only* knob (it's never applied
    /// during evaluation), so it stays an honest measurement.
    pub r1r2_prediction_gate: f32,
    /// Iter-60 DG pattern-separation bridge (R1 → DG → R2). When
    /// [`DgConfig::enabled`] is `true`, brain construction adds a
    /// third region (DG) and wires DG → R2 with a sparse random
    /// projection. Each vocab cue gets a deterministic k-of-n
    /// hashed SDR over DG, injected as external current alongside
    /// the cue's R1 SDR during every drive. The direct R1 → R2
    /// projection weight is multiplied by
    /// [`DgConfig::direct_r1r2_weight_scale`] so the DG path can
    /// be made primary (or sole) cue-routing path.
    ///
    /// Smoke-level minimal viable design: NO kWTA / recurrent
    /// inhibition in DG, NO learning on R1 → DG. Each cue's DG
    /// SDR is fully precomputed and driven directly. Tests
    /// whether *perfect* upstream pattern separation breaks the
    /// iter-58 / iter-59 cross-cue floor; a positive result
    /// motivates a real DG region with kWTA dynamics in iter-61.
    pub dg: DgConfig,
    /// Iter-59 capacity override: number of R2 neurons to allocate.
    /// `0` (the default) means "use the compile-time `R2_N`
    /// constant" (2000), preserving iter-46…58 numerics. A positive
    /// value rebuilds the R2 region at the requested size on every
    /// fresh brain, giving the iter-59 capacity-scaling sweep a
    /// runtime knob without forking the file. Recurrent R2→R2
    /// connectivity scales as `r2_n² × R2_P_CONNECT`; expect
    /// quadratic compute cost.
    pub r2_n: u32,
    /// Iter-64 axis B override. `None` = use the compile-time
    /// `R2_P_CONNECT = 0.05` (iter-46 / iter-63 default). `Some(p)`
    /// overrides the recurrent E↔E connectivity probability for
    /// the duration of this run (axis B mechanism diagnosis;
    /// notes/64-mechanism-diagnosis.md). Range: `(0.0, 1.0]`.
    pub r2_p_connect_override: Option<f32>,
    /// Iter-66: CA1-equivalent C1 readout (Mechanism M1). When
    /// [`C1Config::enabled`] is `true`, brain construction appends
    /// `c1.size` excitatory LIF cells to the R2 region's network
    /// and wires R2-E → C1 with sparse plastic synapses. The
    /// existing R-STDP rule on the R2 region applies to these
    /// new edges automatically; gating is via `M_target` set
    /// (= `c1.teacher_strength`) during the teacher phase and
    /// reset to 0 during eval. The `target_top3_overlap` metric
    /// is reported as before (R2 readout) and a new
    /// `c1_target_top3_overlap` is reported alongside (C1
    /// readout). See `notes/66-ca1-heteroassoc-readout.md` for
    /// the locked pre-registration.
    pub c1: C1Config,
}

impl Default for TeacherForcingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cue_ms: 40,
            delay_ms: 10,
            prediction_ms: 20,
            teacher_ms: 40,
            tail_ms: 20,
            target_clamp_strength: 250.0,
            target_clamp_spike_interval_ms: 0,
            plasticity_during_prediction: false,
            plasticity_during_teacher: true,
            reward_after_teacher: true,
            wta_k: 3,
            negative_reward_for_false_topk: -0.5,
            positive_reward_for_correct: 1.0,
            noise_reward: -1.0,
            homeostatic_normalization: true,
            debug_trials: 0,
            r1r2_prediction_gate: 1.0,
            istdp_during_prediction: false,
            iter49_mode: Iter49Mode::None,
            gated_warmup_epochs: 2,
            gated_ramp_epochs: 2,
            iter46_baseline: false,
            no_plasticity: false,
            recall_mode_eval: false,
            decorrelated_init: false,
            r2_n: 0,
            r2_p_connect_override: None,
            dg: DgConfig {
                enabled: false,
                size: 4000,
                k: 80,
                to_r2_fanout: 30,
                to_r2_weight: 1.0,
                direct_r1r2_weight_scale: 0.0,
                drive_strength: DRIVE_NA,
            },
            c1: C1Config {
                enabled: false,
                size: 1000,
                sparsity_k: 20,
                from_r2_fanout: 30,
                init_w_max: 0.5,
                teacher_strength: 1.0,
                diagnostic: false,
                eval_aligned_rstdp: false,
                btsp: false,
                btsp_window_ms: 200.0,
                btsp_strength: 0.4,
                btsp_target_gated: true,
                btsp_teacher_recurrent_e_scale: 1.0,
                btsp_teacher_recurrent_i_scale: 0.3,
                btsp_no_r2_isolation: false,
            },
        }
    }
}

impl TeacherForcingConfig {
    pub const fn off() -> Self {
        Self {
            enabled: false,
            cue_ms: 40,
            delay_ms: 10,
            prediction_ms: 20,
            teacher_ms: 40,
            tail_ms: 20,
            target_clamp_strength: 250.0,
            target_clamp_spike_interval_ms: 0,
            plasticity_during_prediction: false,
            plasticity_during_teacher: true,
            reward_after_teacher: true,
            wta_k: 3,
            negative_reward_for_false_topk: -0.5,
            positive_reward_for_correct: 1.0,
            noise_reward: -1.0,
            homeostatic_normalization: false,
            debug_trials: 0,
            r1r2_prediction_gate: 1.0,
            istdp_during_prediction: false,
            iter49_mode: Iter49Mode::None,
            gated_warmup_epochs: 2,
            gated_ramp_epochs: 2,
            iter46_baseline: false,
            no_plasticity: false,
            recall_mode_eval: false,
            decorrelated_init: false,
            r2_n: 0,
            r2_p_connect_override: None,
            dg: DgConfig {
                enabled: false,
                size: 4000,
                k: 80,
                to_r2_fanout: 30,
                to_r2_weight: 1.0,
                direct_r1r2_weight_scale: 0.0,
                drive_strength: DRIVE_NA,
            },
            c1: C1Config {
                enabled: false,
                size: 1000,
                sparsity_k: 20,
                from_r2_fanout: 30,
                init_w_max: 0.5,
                teacher_strength: 1.0,
                diagnostic: false,
                eval_aligned_rstdp: false,
                btsp: false,
                btsp_window_ms: 200.0,
                btsp_strength: 0.4,
                btsp_target_gated: true,
                btsp_teacher_recurrent_e_scale: 1.0,
                btsp_teacher_recurrent_i_scale: 0.3,
                btsp_no_r2_isolation: false,
            },
        }
    }
    pub const fn enabled() -> Self {
        Self {
            enabled: true,
            ..Self::off()
        }
    }
}

/// Configuration for one benchmark run.
#[derive(Debug, Clone, Copy)]
pub struct RewardConfig {
    pub epochs: usize,
    /// `true` enables `Brain::set_neuromodulator(...)` and the
    /// reward-modulated STDP rule. `false` is the pure-STDP baseline.
    pub use_reward: bool,
    /// RNG seed for shuffling the trial order each epoch.
    pub seed: u64,
    /// How many times each pair is presented within a single epoch.
    /// More reps → stronger LTP per pass. Default 4 — the recurrent
    /// path needs several presentations to build up weights against
    /// the dominant R1 → R2 forward drive.
    pub reps_per_pair: u32,
    /// Iter-46 teacher-forcing schedule. Default `off` — the
    /// pre-iter-46 cue+target-through-R1 path is preserved.
    pub teacher: TeacherForcingConfig,
}

impl RewardConfig {
    pub const fn baseline(epochs: usize) -> Self {
        Self {
            epochs,
            use_reward: false,
            seed: 42,
            reps_per_pair: 4,
            teacher: TeacherForcingConfig::off(),
        }
    }
    pub const fn with_reward(epochs: usize) -> Self {
        Self {
            epochs,
            use_reward: true,
            seed: 42,
            reps_per_pair: 4,
            teacher: TeacherForcingConfig::off(),
        }
    }
    pub const fn with_teacher(epochs: usize) -> Self {
        Self {
            epochs,
            use_reward: true,
            seed: 42,
            reps_per_pair: 4,
            teacher: TeacherForcingConfig::enabled(),
        }
    }
}

/// A small fixed corpus of word associations. Two-syllable English
/// words chosen so that the encoder produces well-separated SDRs.
pub fn default_corpus() -> RewardCorpus {
    let raw_pairs = [
        ("rust", "ownership"),
        ("python", "dynamic"),
        ("cpp", "pointer"),
        ("java", "jvm"),
        ("haskell", "functional"),
        ("ocaml", "inference"),
        ("go", "channels"),
        ("scala", "tuple"),
        ("kotlin", "coroutine"),
        ("ruby", "block"),
        ("erlang", "actor"),
        ("clojure", "macro"),
        ("swift", "optional"),
        ("zig", "comptime"),
        ("lisp", "lambda"),
        ("scheme", "continuation"),
    ];
    let pairs: Vec<RewardPair> = raw_pairs
        .iter()
        .map(|(c, t)| RewardPair {
            cue: c.to_string(),
            target: t.to_string(),
        })
        .collect();
    // Distractor pairs: shift the targets by 1 so each cue is paired
    // with the *wrong* target. Pure STDP that sees these alongside the
    // real pairs will end up partially associating the wrong target.
    let n = raw_pairs.len();
    let noise_pairs: Vec<RewardPair> = (0..n)
        .map(|i| RewardPair {
            cue: raw_pairs[i].0.to_string(),
            target: raw_pairs[(i + 1) % n].1.to_string(),
        })
        .collect();
    let mut vocab = BTreeSet::new();
    for p in pairs.iter().chain(noise_pairs.iter()) {
        vocab.insert(p.cue.clone());
        vocab.insert(p.target.clone());
    }
    RewardCorpus {
        pairs,
        noise_pairs,
        vocab,
    }
}

/// Iter-58 vocab=64 stress-test corpus. Same construction as
/// [`default_corpus`] but with 32 real (cue, target) pairs
/// instead of 16, doubling the resulting vocabulary to 64
/// distinct words. The extra 16 pairs are programming-language
/// associations chosen to be lexically and phonetically
/// distinct from the original 16, so the encoder produces
/// well-separated SDRs at the larger size.
pub fn default_corpus_v64() -> RewardCorpus {
    let raw_pairs = [
        // -- Original iter-46/53/54/55/56/57 set (16 pairs) --
        ("rust", "ownership"),
        ("python", "dynamic"),
        ("cpp", "pointer"),
        ("java", "jvm"),
        ("haskell", "functional"),
        ("ocaml", "inference"),
        ("go", "channels"),
        ("scala", "tuple"),
        ("kotlin", "coroutine"),
        ("ruby", "block"),
        ("erlang", "actor"),
        ("clojure", "macro"),
        ("swift", "optional"),
        ("zig", "comptime"),
        ("lisp", "lambda"),
        ("scheme", "continuation"),
        // -- Iter-58 extension to vocab=64 (16 more pairs) --
        ("typescript", "generics"),
        ("perl", "regex"),
        ("php", "include"),
        ("lua", "table"),
        ("crystal", "fiber"),
        ("nim", "compile"),
        ("dart", "isolate"),
        ("racket", "syntax"),
        ("elixir", "supervisor"),
        ("groovy", "closure"),
        ("julia", "broadcast"),
        ("matlab", "matrix"),
        ("fortran", "array"),
        ("cobol", "division"),
        ("ada", "task"),
        ("prolog", "unify"),
    ];
    let pairs: Vec<RewardPair> = raw_pairs
        .iter()
        .map(|(c, t)| RewardPair {
            cue: c.to_string(),
            target: t.to_string(),
        })
        .collect();
    let n = raw_pairs.len();
    let noise_pairs: Vec<RewardPair> = (0..n)
        .map(|i| RewardPair {
            cue: raw_pairs[i].0.to_string(),
            target: raw_pairs[(i + 1) % n].1.to_string(),
        })
        .collect();
    let mut vocab = BTreeSet::new();
    for p in pairs.iter().chain(noise_pairs.iter()) {
        vocab.insert(p.cue.clone());
        vocab.insert(p.target.clone());
    }
    RewardCorpus {
        pairs,
        noise_pairs,
        vocab,
    }
}

// ----------------------------------------------------------------------
// Brain construction. Same shape as `scale_bench` but smaller so the
// reward benchmark runs in seconds, not minutes — the goal here is to
// see the *learning curve*, not the absolute accuracy ceiling.
// ----------------------------------------------------------------------

fn build_input_region() -> Region {
    let mut region = Region::new("R1", DT);
    for _ in 0..R1_N {
        region
            .network
            .add_neuron(LifNeuron::excitatory(LifParams::default()));
    }
    region
}

/// Build the R2 (memory) region. `r2_n` is the requested neuron
/// count — pass [`R2_N`] to reproduce the iter-46…58 baseline,
/// or any other positive value to scale the recurrent network up
/// or down. `inh_frac` is the inhibitory fraction; `r2_p_connect`
/// is the recurrent E↔E connection probability — pass
/// [`R2_P_CONNECT`] (0.05) for the iter-46/iter-63 default, or
/// any other positive value in `(0.0, 1.0]` for the iter-64
/// axis B mechanism diagnosis sweep.
fn build_memory_region(seed: u64, inh_frac: f32, r2_n: usize, r2_p_connect: f32) -> Region {
    let mut rng = Rng::new(seed);
    let mut region = Region::new("R2", DT);
    let net = &mut region.network;
    let n_inh = (r2_n as f32 * inh_frac) as usize;
    let n_exc = r2_n - n_inh;
    for _ in 0..n_exc {
        net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    }
    for _ in 0..n_inh {
        net.add_neuron(LifNeuron::inhibitory(LifParams::default()));
    }
    let g_exc = 0.20_f32;
    let g_inh = 0.80_f32;
    for pre in 0..r2_n {
        let g = match net.neurons[pre].kind {
            NeuronKind::Excitatory => g_exc,
            NeuronKind::Inhibitory => g_inh,
        };
        for post in 0..r2_n {
            if pre == post {
                continue;
            }
            if rng.bernoulli(r2_p_connect) {
                let w = rng.range_f32(0.5 * g, 1.0 * g);
                net.connect(pre, post, w);
            }
        }
    }
    region
}

/// Iter-64 axis B helper: returns the effective R2 recurrent
/// connection probability — `r2_p_connect_override` if set,
/// otherwise the compile-time [`R2_P_CONNECT`] (0.05).
fn effective_r2_p_connect(cfg: &TeacherForcingConfig) -> f32 {
    cfg.r2_p_connect_override.unwrap_or(R2_P_CONNECT)
}

fn wire_forward(brain: &mut Brain, seed: u64, inter_weight: f32) {
    let mut rng = Rng::new(seed);
    let r2_size = brain.regions[1].num_neurons();
    for src in 0..R1_N {
        for _ in 0..FAN_OUT {
            let dst = (rng.next_u64() as usize) % r2_size;
            brain.connect(0, src, 1, dst, inter_weight, INTER_DELAY_MS);
        }
    }
}

/// Iter-54 decorrelated R1 → R2 wiring. Partition the R2-E pool
/// into `vocab.len()` disjoint blocks; for each vocab word, count
/// every R1 cell across all encoded SDRs and connect ONLY R1 cells
/// that appear in exactly one cue's SDR — those cells fan out
/// `FAN_OUT` times (with replacement) into their owner-word's
/// block. R1 cells that are shared across multiple cue SDRs are
/// dropped from connectivity entirely. The mechanical invariant:
/// the set of R2 cells reachable from cue *i*'s R1 SDR is disjoint
/// from cue *j*'s for every pair (i, j); `assert_decorrelated_
/// disjoint` enforces this end-to-end.
///
/// Returns the per-word block (R2-E indices) so the caller can
/// audit / log the allocation. Any R1 cell appearing in zero cue
/// SDRs is similarly skipped — only cells with exactly one
/// vocab membership contribute edges.
fn wire_forward_decorrelated(
    brain: &mut Brain,
    encoder: &TextEncoder,
    vocab: &[String],
    seed: u64,
    inter_weight: f32,
) -> Vec<Vec<usize>> {
    use std::collections::BTreeMap;

    // R2-E pool — same convention as canonical_target_r2_sdr.
    let r2_e: Vec<usize> = {
        let net = &brain.regions[1].network;
        (0..net.neurons.len())
            .filter(|&i| matches!(net.neurons[i].kind, NeuronKind::Excitatory))
            .collect()
    };
    let n_words = vocab.len();
    assert!(n_words > 0, "iter-54 decorrelated init: empty vocab");
    let block_size = r2_e.len() / n_words;
    assert!(
        block_size > 0,
        "iter-54 decorrelated init: R2-E pool ({}) too small for vocab size ({})",
        r2_e.len(),
        n_words,
    );

    let mut blocks: Vec<Vec<usize>> = Vec::with_capacity(n_words);
    for word_idx in 0..n_words {
        let start = word_idx * block_size;
        let end = if word_idx == n_words - 1 {
            r2_e.len()
        } else {
            (word_idx + 1) * block_size
        };
        blocks.push(r2_e[start..end].to_vec());
    }

    // Count + first-owner pass over all cue SDRs.
    let mut count: BTreeMap<u32, usize> = BTreeMap::new();
    let mut first_owner: BTreeMap<u32, usize> = BTreeMap::new();
    for (word_idx, word) in vocab.iter().enumerate() {
        let r1_sdr = encoder.encode_word(word);
        for &r1_idx in &r1_sdr.indices {
            *count.entry(r1_idx).or_insert(0) += 1;
            first_owner.entry(r1_idx).or_insert(word_idx);
        }
    }

    // Wire only the R1 cells with exactly one cue membership.
    // Iterate `count` in BTreeMap order so the wiring is
    // deterministic given the seed.
    let mut rng = Rng::new(seed);
    for (&r1_idx, &c) in &count {
        if c != 1 {
            continue;
        }
        let word_idx = first_owner[&r1_idx];
        let block = &blocks[word_idx];
        if block.is_empty() {
            continue;
        }
        for _ in 0..FAN_OUT {
            let dst = block[(rng.next_u64() as usize) % block.len()];
            brain.connect(0, r1_idx as usize, 1, dst, inter_weight, INTER_DELAY_MS);
        }
    }

    blocks
}

/// Iter-54 mechanical invariant. Verifies that for every pair of
/// vocab words `(i, j), i < j`, the set of R2 cells reachable
/// from cue *i*'s R1 SDR (via any direct R1 → R2 edge) is
/// disjoint from cue *j*'s. Implemented end-to-end against the
/// post-wiring brain state — no internal short-cut to the block
/// allocation. Panics with the offending cue pair and the shared
/// indices on the first violation.
fn assert_decorrelated_disjoint(brain: &Brain, encoder: &TextEncoder, vocab: &[String]) {
    let mut targets: Vec<BTreeSet<u32>> = Vec::with_capacity(vocab.len());
    for word in vocab {
        let r1_sdr = encoder.encode_word(word);
        let mut t: BTreeSet<u32> = BTreeSet::new();
        for &r1_idx in &r1_sdr.indices {
            let outgoing = &brain.outgoing[0][r1_idx as usize];
            for &edge_id in outgoing {
                let edge = brain.inter_edges[edge_id as usize];
                if edge.dst_region == 1 {
                    t.insert(edge.dst_neuron);
                }
            }
        }
        targets.push(t);
    }
    for i in 0..targets.len() {
        for j in (i + 1)..targets.len() {
            let inter: Vec<u32> = targets[i].intersection(&targets[j]).copied().collect();
            assert!(
                inter.is_empty(),
                "iter-54 decorrelated invariant violated: cues '{}' (idx {}) and '{}' (idx {}) \
                 share {} R2 target(s) (first 10: {:?}). Either an R1 cell with multi-cue \
                 membership leaked into wiring, or the blocks were not partitioned cleanly.",
                vocab[i],
                i,
                vocab[j],
                j,
                inter.len(),
                inter.iter().take(10).collect::<Vec<_>>(),
            );
        }
    }
}

fn fresh_brain(seed: u64) -> Brain {
    fresh_brain_with(seed, INTER_WEIGHT, R2_INH_FRAC, R2_N, R2_P_CONNECT)
}

/// Iter-50 variant: build the brain with explicit override values
/// for `INTER_WEIGHT` and `R2_INH_FRAC`. Used by the
/// `--iter46-baseline` diagnostic to reproduce the original
/// iter-46 topology (2.0 / 0.20) instead of the iter-47/49
/// drift values (1.0 / 0.30).
///
/// Iter-59: `r2_n` is now an explicit parameter so the capacity
/// sweep can rebuild R2 at any positive size. Pass [`R2_N`] to
/// reproduce iter-46…58 numerics exactly.
///
/// Iter-64: `r2_p_connect` is the recurrent E↔E connectivity
/// probability — pass [`R2_P_CONNECT`] (0.05) for the iter-46/
/// iter-63 default, or any value in `(0.0, 1.0]` for axis B.
fn fresh_brain_with(
    seed: u64,
    inter_weight: f32,
    inh_frac: f32,
    r2_n: usize,
    r2_p_connect: f32,
) -> Brain {
    let mut brain = Brain::new(DT);
    brain.add_region(build_input_region());
    brain.add_region(build_memory_region(
        seed.wrapping_add(1),
        inh_frac,
        r2_n,
        r2_p_connect,
    ));
    wire_forward(&mut brain, seed.wrapping_add(2), inter_weight);
    brain
}

/// Iter-59 helper: extract the effective R2 neuron count from the
/// teacher config. `0` means "use the [`R2_N`] compile-time
/// default"; any positive value overrides it.
fn effective_r2_n(cfg: &TeacherForcingConfig) -> usize {
    if cfg.r2_n == 0 {
        R2_N
    } else {
        cfg.r2_n as usize
    }
}

/// Iter-60: build the DG region. All cells excitatory; no
/// recurrent connectivity inside DG (the smoke test deliberately
/// uses fully-precomputed k-of-n SDRs driven externally per cue,
/// so no kWTA / inhibitory dynamics needed in DG itself).
fn build_dg_region(size: usize) -> Region {
    let mut region = Region::new("DG", DT);
    for _ in 0..size {
        region
            .network
            .add_neuron(LifNeuron::excitatory(LifParams::default()));
    }
    region
}

/// Iter-60: random sparse DG → R2 projection (mossy-fibre-style).
/// Each DG cell projects to `fanout` random R2 cells with the
/// configured weight. R2 cells receiving DG inputs are not
/// constrained to the E-pool — inhibitory R2 cells can also
/// receive mossy-fibre drive (matches the biology where mossy
/// fibres synapse on both pyramidal cells and interneurons).
fn wire_dg_to_r2(brain: &mut Brain, cfg: &DgConfig, seed: u64) {
    let mut rng = Rng::new(seed);
    let dg_size = brain.regions[2].num_neurons();
    let r2_size = brain.regions[1].num_neurons();
    for src in 0..dg_size {
        for _ in 0..cfg.to_r2_fanout {
            let dst = (rng.next_u64() as usize) % r2_size;
            brain.connect(2, src, 1, dst, cfg.to_r2_weight, INTER_DELAY_MS);
        }
    }
}

/// Iter-66: extend the R2 region's `Network` with C1 cells and
/// wire R2-E → C1 plastic synapses (Mechanism M1, CA1-equivalent
/// readout layer). Returns the C1 cell index range
/// `[c1_start, c1_end)` inside R2's network so the caller can
/// build a `c1_set` and pass it to the readout dictionary.
///
/// Index layout after this call:
///   `0 .. r2_n_used`              → original R2 cells (E + I)
///   `r2_n_used .. r2_n_used + N`  → new C1 cells (all excitatory)
///
/// The R2-E → C1 edges are *intra-network* synapses on R2's
/// `Network`, so they are subject to the existing R-STDP rule
/// gated by the global neuromodulator (set to
/// `c1.teacher_strength` during the teacher phase via
/// `Brain::set_neuromodulator` in the trial schedule, and
/// implicitly 0 during eval). This is the minimum-scope way to
/// expose the readout projection to a target-presence-gated
/// three-factor learning rule without adding plastic inter-region
/// edges to snn-core.
///
/// `r2_e` is the original R2-E set captured *before* this
/// function is called; it is NOT recomputed because we want to
/// avoid C1 cells (also excitatory) appearing as R2-E in the
/// readout. Each R2-E source cell projects to
/// `cfg.from_r2_fanout` random C1 cells with weight uniform on
/// `(0, cfg.init_w_max)`.
fn append_c1_to_r2(
    brain: &mut Brain,
    cfg: &C1Config,
    r2_e: &BTreeSet<usize>,
    r2_n_used: usize,
    seed: u64,
) -> (usize, usize) {
    let net = &mut brain.regions[1].network;
    let c1_start = net.neurons.len();
    debug_assert_eq!(
        c1_start, r2_n_used,
        "iter-66: C1 must be appended directly after the original R2 cells",
    );
    for _ in 0..cfg.size {
        net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    }
    let c1_end = net.neurons.len();

    // Brain bookkeeping: outgoing buckets must grow with the new
    // post-cells so future synapses can reference them as src.
    brain.outgoing[1].resize_with(c1_end, Vec::new);

    // R2-E → C1 plastic synapses, fan-out per source.
    let mut rng = Rng::new(seed);
    let c1_count = (c1_end - c1_start).max(1);
    let net = &mut brain.regions[1].network;
    let r2_e_vec: Vec<usize> = r2_e.iter().copied().collect();
    for &src in &r2_e_vec {
        for _ in 0..cfg.from_r2_fanout {
            let dst = c1_start + (rng.next_u64() as usize) % c1_count;
            let w = rng.range_f32(0.0, cfg.init_w_max.max(f32::EPSILON));
            net.connect(src, dst, w);
        }
    }
    (c1_start, c1_end)
}

/// Iter-66: helper — return the C1 cell index set as a
/// `BTreeSet<usize>` matching the shape of `r2_e_set` so existing
/// `drive_for_with_counts` / `build_vocab_dictionary` /
/// `evaluate_with_dict` paths accept it without modification.
fn c1_set(c1_range: (usize, usize)) -> BTreeSet<usize> {
    (c1_range.0..c1_range.1).collect()
}

/// Iter-66: deterministic k-of-n hashed C1 target SDR for a target
/// word. Same hash structure as `dg_sdr_for_cue` /
/// `canonical_target_r2_sdr`; the SDR is a pure function of
/// `(salt, word, c1_size, k)` so every brain at the same seed
/// learns to bind the same R2-E cue spikes to the same C1 cell
/// pattern. Indices returned are *absolute* into R2's network
/// (i.e. shifted by `c1_start`) so the existing R2 clamp path
/// can drive them without index-space translation.
fn canonical_target_c1_sdr(
    word: &str,
    c1_start: usize,
    c1_size: usize,
    k: usize,
    salt: u64,
) -> Vec<u32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut chosen: BTreeSet<u32> = BTreeSet::new();
    let mut counter: u64 = 0;
    while chosen.len() < k && counter < (k as u64) * 32 {
        let mut hasher = DefaultHasher::new();
        salt.hash(&mut hasher);
        word.hash(&mut hasher);
        counter.hash(&mut hasher);
        let h = hasher.finish();
        let idx = c1_start + ((h as usize) % c1_size);
        chosen.insert(idx as u32);
        counter = counter.wrapping_add(1);
    }
    chosen.into_iter().collect()
}

/// Iter-60: deterministic k-of-n hashed DG SDR for a cue word.
/// Same hash structure as `canonical_target_r2_sdr`; the SDR is a
/// pure function of `(salt, word, dg_size, k)` so every brain at
/// the same seed sees identical DG addresses.
fn dg_sdr_for_cue(word: &str, dg_size: usize, k: usize, salt: u64) -> Vec<u32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut chosen: BTreeSet<u32> = BTreeSet::new();
    let mut counter: u64 = 0;
    while chosen.len() < k && counter < (k as u64) * 32 {
        let mut hasher = DefaultHasher::new();
        salt.hash(&mut hasher);
        word.hash(&mut hasher);
        counter.hash(&mut hasher);
        let h = hasher.finish();
        let idx = (h as usize) % dg_size;
        chosen.insert(idx as u32);
        counter = counter.wrapping_add(1);
    }
    chosen.into_iter().collect()
}

/// Iter-60: drive primitive aware of the DG region. When DG is
/// active (3rd region present), injects external current into
/// the cue's R1 SDR + DG SDR cells simultaneously. When `dg_sdr`
/// is empty *and* the brain has 3 regions, the DG region
/// receives zeros — used during phases that should not stimulate
/// the cue (idle, prediction without cue).
///
/// `r1_strength = 0.0` skips the R1-side injection (useful when
/// DG should be the sole driver, even though R1 SDR is non-empty);
/// `dg_strength = 0.0` likewise skips the DG-side injection.
fn drive_with_dg(
    brain: &mut Brain,
    r1_sdr: &[u32],
    dg_sdr: &[u32],
    r1_strength: f32,
    dg_strength: f32,
    duration_ms: f32,
) {
    let r2_size = brain.regions[1].num_neurons();
    let dg_size = brain.regions.get(2).map(|r| r.num_neurons()).unwrap_or(0);
    let mut ext_r1 = vec![0.0_f32; R1_N];
    if r1_strength != 0.0 {
        for &idx in r1_sdr {
            if (idx as usize) < R1_N {
                ext_r1[idx as usize] = r1_strength;
            }
        }
    }
    let ext_r2 = vec![0.0_f32; r2_size];
    let mut externals: Vec<Vec<f32>> = vec![ext_r1, ext_r2];
    if dg_size > 0 {
        let mut ext_dg = vec![0.0_f32; dg_size];
        if dg_strength != 0.0 {
            for &idx in dg_sdr {
                if (idx as usize) < dg_size {
                    ext_dg[idx as usize] = dg_strength;
                }
            }
        }
        externals.push(ext_dg);
    }
    let steps = (duration_ms / DT) as usize;
    for _ in 0..steps {
        brain.step(&externals);
    }
}

/// Iter-60: per-step R2-E spike-counting drive primitive aware
/// of the DG region. Shape matches `drive_for_with_counts`.
#[allow(clippy::too_many_arguments)]
fn drive_with_dg_counts(
    brain: &mut Brain,
    r1_sdr: &[u32],
    dg_sdr: &[u32],
    r1_strength: f32,
    dg_strength: f32,
    duration_ms: f32,
    r2_e_set: &BTreeSet<usize>,
) -> Vec<u32> {
    let r2_size = brain.regions[1].num_neurons();
    let dg_size = brain.regions.get(2).map(|r| r.num_neurons()).unwrap_or(0);
    let mut counts = vec![0u32; r2_size];
    let mut ext_r1 = vec![0.0_f32; R1_N];
    if r1_strength != 0.0 {
        for &idx in r1_sdr {
            if (idx as usize) < R1_N {
                ext_r1[idx as usize] = r1_strength;
            }
        }
    }
    let ext_r2 = vec![0.0_f32; r2_size];
    let mut externals: Vec<Vec<f32>> = vec![ext_r1, ext_r2];
    if dg_size > 0 {
        let mut ext_dg = vec![0.0_f32; dg_size];
        if dg_strength != 0.0 {
            for &idx in dg_sdr {
                if (idx as usize) < dg_size {
                    ext_dg[idx as usize] = dg_strength;
                }
            }
        }
        externals.push(ext_dg);
    }
    let steps = (duration_ms / DT) as usize;
    for _ in 0..steps {
        let spikes = brain.step(&externals);
        for &id in &spikes[1] {
            if r2_e_set.contains(&id) {
                counts[id] += 1;
            }
        }
    }
    counts
}

/// Iter-60: idle helper aware of the DG region. Currently
/// unused at the call sites (the legacy `idle` already auto-
/// zero-pads DG when present), kept for symmetry with
/// `drive_with_dg` / `drive_with_dg_counts` and as a building
/// block for any iter-61+ DG-specific idle phase.
#[allow(dead_code)]
fn idle_with_dg(brain: &mut Brain, duration_ms: f32) {
    let r2_size = brain.regions[1].num_neurons();
    let dg_size = brain.regions.get(2).map(|r| r.num_neurons()).unwrap_or(0);
    let mut externals: Vec<Vec<f32>> = vec![vec![0.0_f32; R1_N], vec![0.0_f32; r2_size]];
    if dg_size > 0 {
        externals.push(vec![0.0_f32; dg_size]);
    }
    let steps = (duration_ms / DT) as usize;
    for _ in 0..steps {
        brain.step(&externals);
    }
}

fn stdp() -> StdpParams {
    StdpParams {
        a_plus: 0.020,
        a_minus: 0.015,
        w_max: 0.8,
        ..StdpParams::default()
    }
}

/// Iter-50 diagnostic: the *original* iter-46 iSTDP parameter set.
/// Used by `--iter46-baseline` to reproduce the iter-46 Arm B
/// configuration verbatim, *before* the iter-48 tightening
/// (`a_plus 0.10 → 0.30`, `tau_minus 30 → 8 ms`).
fn istdp_iter46_baseline() -> IStdpParams {
    IStdpParams {
        a_plus: 0.10,
        a_minus: 1.10,
        tau_minus: 30.0,
        w_min: 0.0,
        w_max: 8.0,
    }
}

/// Iter-49 mode-aware iSTDP builder. Takes the iter-48 params as
/// the baseline and applies the per-mode + per-epoch override:
///
/// - `None`           → iter-48 params verbatim.
/// - `WmaxCap`        → `w_max = 2.0` (vs iter-48's 8.0).
/// - `APlusHalf`      → `a_plus = 0.20` (vs iter-48's 0.30).
/// - `ActivityGated`  → `a_plus` ramped over `gated_warmup_epochs +
///   gated_ramp_epochs`. During warmup `a_plus = 0`; during ramp it
///   rises linearly from 0 to the iter-48 baseline (0.30); thereafter
///   it stays at the baseline.
fn istdp_iter49(cfg: &TeacherForcingConfig, epoch: usize) -> IStdpParams {
    let mut p = istdp(); // iter-48 baseline (a_plus 0.30, tau_minus 8, w_max 8.0)
    match cfg.iter49_mode {
        Iter49Mode::None => {}
        Iter49Mode::WmaxCap => {
            p.w_max = 2.0;
        }
        Iter49Mode::APlusHalf => {
            p.a_plus = 0.20;
        }
        Iter49Mode::ActivityGated => {
            let warmup = cfg.gated_warmup_epochs as usize;
            let ramp = cfg.gated_ramp_epochs.max(1) as usize;
            let baseline = 0.30_f32;
            p.a_plus = if epoch < warmup {
                0.0
            } else if epoch < warmup + ramp {
                let r = (epoch - warmup) as f32 + 1.0;
                baseline * r / ramp as f32
            } else {
                baseline
            };
        }
    }
    p
}

/// Iter-48 iSTDP retune (Vogels et al. 2011 fast-EI-balance):
///
/// - `tau_minus`: 30 → 8 ms. The original 30 ms post-trace is
///   appropriate for slow homeostasis but is *longer* than the
///   AMPA → GABA latency budget needed to suppress a cascade
///   while it forms. 8 ms is in the same regime as the AMPA τ
///   used elsewhere in snn-core, so an inhibitory cell that
///   fires in response to a recurrent E avalanche has a chance
///   of catching the *same* cascade instead of one trial later.
///
/// - `a_plus`: 0.10 → 0.30. Larger LTP per pre-only I-spike so
///   silenced E-targets accumulate inhibitory weight faster.
///   In the iter-47a-pm cascade trace, the post-cascade epoch
///   showed θ jumping 95× — that's the rate the iSTDP has to
///   match if the inhibitory walls are to grow during, not after.
///
/// - `a_minus`: 1.10 unchanged — the LTD-on-coactivity term is
///   still the right magnitude (and it carries the full ratio
///   responsibility for engram selectivity, untouched here).
fn istdp() -> IStdpParams {
    IStdpParams {
        a_plus: 0.30,
        a_minus: 1.10,
        tau_minus: 8.0,
        w_min: 0.0,
        w_max: 8.0,
    }
}

fn homeostasis() -> HomeostasisParams {
    HomeostasisParams {
        eta_scale: 0.002,
        a_target: 2.0,
        tau_homeo_ms: 30.0,
        apply_every: 8,
        scale_only_down: true,
    }
}

/// Iter-47a-2 — Diehl & Cook (2015) adaptive threshold per R2-E
/// neuron. The iter-44 IntrinsicParams already implements the rule:
///
/// ```text
///   adapt(t+dt) = adapt(t) * exp(-dt / tau_adapt)
///   adapt      += alpha_spike on every post-spike
///   v_thresh_eff = v_threshold_base + beta * (adapt - a_target)
/// ```
///
/// With `a_target = 0`, `beta = 1`, this collapses to a pure
/// per-spike threshold offset (Δθ ≈ alpha_spike), which is exactly
/// Diehl-Cook. The cap on `offset_max` stops the threshold from
/// climbing so high that a neuron stops firing entirely (dead-cell
/// failure mode).
///
/// Key insight: this is the *cheapest* fix for the iter-46 finding
/// that R2 had 90–180 active cells per cue (vs. 30 canonical
/// targets). Cells that fire too often raise their own threshold,
/// quieting the runaway forward population without touching the
/// recurrent path.
fn intrinsic() -> IntrinsicParams {
    IntrinsicParams {
        alpha_spike: 0.05,
        tau_adapt_ms: 2000.0,
        a_target: 0.0,
        beta: 1.0,
        offset_min: 0.0,
        offset_max: 5.0,
        enabled: true,
    }
}

fn reward_params() -> RewardParams {
    // η large enough that 80 ms of consolidation moves a tag of order
    // 1.0 by ~ 0.1 in weight space — comparable to a single STDP pair
    // event. tau_eligibility = 1 s so the tag built up during phase 1
    // (~80 ms training) survives the brief test-and-decode in phase 2.
    RewardParams {
        eta: 0.05,
        tau_eligibility_ms: 1000.0,
        a_plus: 0.02,
        a_minus: 0.02,
        w_min: 0.0,
        w_max: 0.8,
        excitatory_only: true,
    }
}

// ----------------------------------------------------------------------
// Driver primitives.
// ----------------------------------------------------------------------

fn drive_for(brain: &mut Brain, sdr_indices: &[u32], duration_ms: f32) {
    let r2_size = brain.regions[1].num_neurons();
    let mut ext1 = vec![0.0_f32; R1_N];
    for &idx in sdr_indices {
        if (idx as usize) < R1_N {
            ext1[idx as usize] = DRIVE_NA;
        }
    }
    let ext2 = vec![0.0_f32; r2_size];
    let mut externals = vec![ext1, ext2];
    // Iter-60: zero-pad DG region (region 2) if present. Existing
    // call sites stay numerically unchanged when no DG is wired
    // up — DG just receives no drive.
    if let Some(r) = brain.regions.get(2) {
        externals.push(vec![0.0_f32; r.num_neurons()]);
    }
    let steps = (duration_ms / DT) as usize;
    for _ in 0..steps {
        brain.step(&externals);
    }
}

fn drive_for_with_counts(
    brain: &mut Brain,
    sdr_indices: &[u32],
    duration_ms: f32,
    r2_e_set: &BTreeSet<usize>,
) -> Vec<u32> {
    let r2_size = brain.regions[1].num_neurons();
    let mut counts = vec![0u32; r2_size];
    let mut ext1 = vec![0.0_f32; R1_N];
    for &idx in sdr_indices {
        if (idx as usize) < R1_N {
            ext1[idx as usize] = DRIVE_NA;
        }
    }
    let ext2 = vec![0.0_f32; r2_size];
    let mut externals = vec![ext1, ext2];
    if let Some(r) = brain.regions.get(2) {
        externals.push(vec![0.0_f32; r.num_neurons()]);
    }
    let steps = (duration_ms / DT) as usize;
    for _ in 0..steps {
        let spikes = brain.step(&externals);
        for &id in &spikes[1] {
            if r2_e_set.contains(&id) {
                counts[id] += 1;
            }
        }
    }
    counts
}

fn idle(brain: &mut Brain, duration_ms: f32) {
    let r2_size = brain.regions[1].num_neurons();
    let mut zeros = vec![vec![0.0_f32; R1_N], vec![0.0_f32; r2_size]];
    if let Some(r) = brain.regions.get(2) {
        zeros.push(vec![0.0_f32; r.num_neurons()]);
    }
    let steps = (duration_ms / DT) as usize;
    for _ in 0..steps {
        brain.step(&zeros);
    }
}

// ----------------------------------------------------------------------
// Iter-46 teacher-forcing primitives. Step 2 lands the helpers; step
// 3 plumbs them into the trial schedule. The `#[allow(dead_code)]`
// is on the unused-as-of-step-2 leaves so clippy stays clean across
// the intermediate commit.
// ----------------------------------------------------------------------

/// How many R2-E neurons we earmark per target word for the teacher
/// clamp. Roughly matches the kWTA size used elsewhere in the
/// harness so the clamped pattern stays in the same density regime
/// as a normal recall response.
#[allow(dead_code)]
const TARGET_R2_K: usize = 30;

/// Deterministic, stable mapping from a target word to a fixed set
/// of R2-E neuron indices. Implemented with the same DefaultHasher
/// the encoder uses, so the mapping is a pure function of the word
/// + a per-corpus salt — no random drift between epochs and no
///   trial-to-trial fingerprint variance.
///
/// The result is sorted, deduplicated, and constrained to the
/// excitatory portion of R2 (the `e_pool` slice). Picking from the
/// E-only pool keeps the clamp from accidentally pinning inhibitory
/// cells, which would invert the rule's intended sign.
#[allow(dead_code)]
fn canonical_target_r2_sdr(word: &str, e_pool: &[usize], k: usize, salt: u64) -> Vec<u32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut chosen: BTreeSet<u32> = BTreeSet::new();
    let mut counter: u64 = 0;
    while chosen.len() < k && counter < (k as u64) * 16 {
        let mut hasher = DefaultHasher::new();
        salt.hash(&mut hasher);
        word.hash(&mut hasher);
        counter.hash(&mut hasher);
        let h = hasher.finish();
        let idx = e_pool[(h as usize) % e_pool.len()] as u32;
        chosen.insert(idx);
        counter = counter.wrapping_add(1);
    }
    chosen.into_iter().collect()
}

/// Drive the brain for `duration_ms` with `cue_indices` clamped to
/// R1 *and* `r2_target_indices` clamped directly into R2 with
/// strength `r2_strength`. The R1 forward path runs as usual;
/// the R2 clamp is the iter-46 teacher signal.
///
/// Set `r2_strength = 0.0` to disable the clamp (then the function
/// behaves like [`drive_for`]). Pass `cue_indices = &[]` to clamp
/// R2 in isolation — useful for debug instrumentation.
///
/// Returns per-R2-E spike counts so callers can assess how reliably
/// the clamp produced the intended pattern.
#[allow(dead_code)]
fn drive_with_r2_clamp(
    brain: &mut Brain,
    cue_indices: &[u32],
    r2_target_indices: &[u32],
    r1_strength: f32,
    r2_strength: f32,
    duration_ms: f32,
    r2_e_set: &BTreeSet<usize>,
) -> Vec<u32> {
    let r2_size = brain.regions[1].num_neurons();
    let mut counts = vec![0u32; r2_size];

    let mut ext1 = vec![0.0_f32; R1_N];
    for &idx in cue_indices {
        if (idx as usize) < R1_N {
            ext1[idx as usize] = r1_strength;
        }
    }
    let mut ext2 = vec![0.0_f32; r2_size];
    if r2_strength != 0.0 {
        for &idx in r2_target_indices {
            if (idx as usize) < r2_size {
                ext2[idx as usize] = r2_strength;
            }
        }
    }
    let mut externals = vec![ext1, ext2];
    if let Some(r) = brain.regions.get(2) {
        externals.push(vec![0.0_f32; r.num_neurons()]);
    }
    let steps = (duration_ms / DT) as usize;
    for _ in 0..steps {
        let spikes = brain.step(&externals);
        for &id in &spikes[1] {
            if r2_e_set.contains(&id) {
                counts[id] += 1;
            }
        }
    }
    counts
}

/// Iter-60: DG-aware variant of `drive_with_r2_clamp`. Drives the
/// cue's R1 SDR (with `r1_strength`) AND its DG SDR (with
/// `dg_strength`) AND the canonical-target R2 clamp (with
/// `r2_strength`) for `duration_ms` ms. Returns per-R2-E spike
/// counts identically to `drive_with_r2_clamp`.
#[allow(clippy::too_many_arguments)]
fn drive_with_r2_clamp_dg(
    brain: &mut Brain,
    cue_indices: &[u32],
    dg_indices: &[u32],
    r2_target_indices: &[u32],
    r1_strength: f32,
    dg_strength: f32,
    r2_strength: f32,
    duration_ms: f32,
    r2_e_set: &BTreeSet<usize>,
) -> Vec<u32> {
    let r2_size = brain.regions[1].num_neurons();
    let dg_size = brain.regions.get(2).map(|r| r.num_neurons()).unwrap_or(0);
    let mut counts = vec![0u32; r2_size];

    let mut ext1 = vec![0.0_f32; R1_N];
    if r1_strength != 0.0 {
        for &idx in cue_indices {
            if (idx as usize) < R1_N {
                ext1[idx as usize] = r1_strength;
            }
        }
    }
    let mut ext2 = vec![0.0_f32; r2_size];
    if r2_strength != 0.0 {
        for &idx in r2_target_indices {
            if (idx as usize) < r2_size {
                ext2[idx as usize] = r2_strength;
            }
        }
    }
    let mut externals: Vec<Vec<f32>> = vec![ext1, ext2];
    if dg_size > 0 {
        let mut ext_dg = vec![0.0_f32; dg_size];
        if dg_strength != 0.0 {
            for &idx in dg_indices {
                if (idx as usize) < dg_size {
                    ext_dg[idx as usize] = dg_strength;
                }
            }
        }
        externals.push(ext_dg);
    }
    let steps = (duration_ms / DT) as usize;
    for _ in 0..steps {
        let spikes = brain.step(&externals);
        for &id in &spikes[1] {
            if r2_e_set.contains(&id) {
                counts[id] += 1;
            }
        }
    }
    counts
}

/// Iter-47a postmortem (b)+(c): same as `drive_with_r2_clamp` but
/// also records, per simulation step:
///   - how many R2-E cells fired in that step (active count),
///   - how many of those were canonical-target cells.
///
/// Used by the cascade diagnostic to distinguish onset-burst vs.
/// sustained-runaway (Litwin-Kumar 2014). The function is *not*
/// on the hot path — only called when the operator passes
/// `--debug-cascade`.
#[allow(clippy::too_many_arguments)]
fn drive_with_r2_clamp_traced(
    brain: &mut Brain,
    cue_indices: &[u32],
    r2_target_indices: &[u32],
    r1_strength: f32,
    r2_strength: f32,
    duration_ms: f32,
    r2_e_set: &BTreeSet<usize>,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let target_set: BTreeSet<u32> = r2_target_indices.iter().copied().collect();
    let r2_size = brain.regions[1].num_neurons();
    let mut counts = vec![0u32; r2_size];
    let mut per_step_active: Vec<u32> = Vec::new();
    let mut per_step_target: Vec<u32> = Vec::new();
    let mut ext1 = vec![0.0_f32; R1_N];
    for &idx in cue_indices {
        if (idx as usize) < R1_N {
            ext1[idx as usize] = r1_strength;
        }
    }
    let mut ext2 = vec![0.0_f32; r2_size];
    if r2_strength != 0.0 {
        for &idx in r2_target_indices {
            if (idx as usize) < r2_size {
                ext2[idx as usize] = r2_strength;
            }
        }
    }
    let mut externals = vec![ext1, ext2];
    if let Some(r) = brain.regions.get(2) {
        externals.push(vec![0.0_f32; r.num_neurons()]);
    }
    let steps = (duration_ms / DT) as usize;
    for _ in 0..steps {
        let spikes = brain.step(&externals);
        let mut step_active: u32 = 0;
        let mut step_target: u32 = 0;
        for &id in &spikes[1] {
            if r2_e_set.contains(&id) {
                counts[id] += 1;
                step_active += 1;
                if target_set.contains(&(id as u32)) {
                    step_target += 1;
                }
            }
        }
        per_step_active.push(step_active);
        per_step_target.push(step_target);
    }
    (counts, per_step_active, per_step_target)
}

/// Iter-48: split the adaptive-threshold mean by neuron kind so
/// the E/I balance asymmetry is readable. Returns
/// `(mean_over_E, mean_over_I)` of `Network.v_thresh_offset`.
/// If the offset buffer is empty (intrinsic plasticity off) both
/// values are 0.
fn intrinsic_mean_by_kind(brain: &Brain) -> (f32, f32) {
    let net = &brain.regions[1].network;
    if net.v_thresh_offset.is_empty() {
        return (0.0, 0.0);
    }
    let (mut e_sum, mut e_n, mut i_sum, mut i_n) = (0.0_f64, 0_u32, 0.0_f64, 0_u32);
    for (i, n) in net.neurons.iter().enumerate() {
        let v = net.v_thresh_offset[i] as f64;
        match n.kind {
            NeuronKind::Excitatory => {
                e_sum += v;
                e_n += 1;
            }
            NeuronKind::Inhibitory => {
                i_sum += v;
                i_n += 1;
            }
        }
    }
    let e = if e_n > 0 {
        (e_sum / e_n as f64) as f32
    } else {
        0.0
    };
    let i = if i_n > 0 {
        (i_sum / i_n as f64) as f32
    } else {
        0.0
    };
    (e, i)
}

/// Iter-47a postmortem (c): summary statistics for the per-neuron
/// adaptive-threshold offset (Diehl-Cook). Returns
/// `(mean, std, min, max, frac_above_1mV)` over the R2-E pool.
fn intrinsic_stats(brain: &Brain, r2_e_set: &BTreeSet<usize>) -> (f32, f32, f32, f32, f32) {
    let net = &brain.regions[1].network;
    if net.v_thresh_offset.is_empty() || r2_e_set.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let xs: Vec<f32> = r2_e_set.iter().map(|&i| net.v_thresh_offset[i]).collect();
    let n = xs.len() as f32;
    let mean: f32 = xs.iter().sum::<f32>() / n;
    let var: f32 = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let min = xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let above = xs.iter().filter(|&&x| x > 1.0).count() as f32 / n;
    (mean, std, min, max, above)
}

/// Subset of the R2-E neurons in their topological order — the pool
/// from which `canonical_target_r2_sdr` picks. Cached at brain build
/// time so we don't filter the neuron list on every call.
#[allow(dead_code)]
fn r2_e_pool(r2_e_set: &BTreeSet<usize>) -> Vec<usize> {
    r2_e_set.iter().copied().collect()
}

fn top_k_indices(counts: &[u32], k: usize) -> Vec<u32> {
    let mut paired: Vec<(u32, usize)> = counts
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, &c)| (c, i))
        .collect();
    paired.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    paired.truncate(k);
    let mut idx: Vec<u32> = paired.into_iter().map(|(_, i)| i as u32).collect();
    idx.sort_unstable();
    idx
}

fn r2_e_set(brain: &Brain) -> BTreeSet<usize> {
    brain.regions[1]
        .network
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.kind == NeuronKind::Excitatory)
        .map(|(i, _)| i)
        .collect()
}

/// Iter-52: per-region L2 norm of all synapse weights. Used as a
/// bit-identity sanity check under `--no-plasticity`: if the gate
/// is tight, the post-training norms must equal the pre-training
/// norms exactly (up to f32 stability of repeated `.weight` reads).
/// Returns one f64 per region.
fn brain_synapse_l2_norms(brain: &Brain) -> Vec<f64> {
    brain
        .regions
        .iter()
        .map(|r| {
            r.network
                .synapses
                .iter()
                .map(|s| (s.weight as f64) * (s.weight as f64))
                .sum::<f64>()
                .sqrt()
        })
        .collect()
}

fn shuffle<T>(rng: &mut Rng, v: &mut [T]) {
    // Fisher-Yates using the deterministic in-house RNG.
    for i in (1..v.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        v.swap(i, j);
    }
}

/// Iter-47: linear-interpolated percentile of a small sample
/// (clamped to integer u32 because all our active-cell counts are
/// integers). Empty sample → 0.
fn percentile_u32(sorted_samples: &[u32], pct: f32) -> u32 {
    if sorted_samples.is_empty() {
        return 0;
    }
    let n = sorted_samples.len();
    let idx = ((pct.clamp(0.0, 1.0)) * (n - 1) as f32).round() as usize;
    sorted_samples[idx.min(n - 1)]
}

/// Iter-46 six-phase trial. Returns the prediction-phase top-k
/// (computed *before* teacher-forcing) so the caller can score
/// recall and feed reward without contaminating the eval signal.
///
/// Plasticity is silenced during the prediction phase by toggling
/// the `Network`'s STDP / iSTDP / R-STDP setters around the
/// prediction window. The teacher phase clamps the canonical
/// target SDR directly into R2 so STDP picks up clean
/// pre→post coincidences between cue-driven cells and the
/// teacher-clamped target cells.
///
/// Iter-66 (M1): the new `c1_target_sdr` parameter — absolute
/// indices into R2's network in `[r2_n_used, r2_n_used + c1.size)`
/// — is empty when `cfg.c1.enabled = false`, preserving iter-46/65
/// numerics bit-identically. When non-empty, C1 cells in this SDR
/// are clamped during the Phase 4 teacher window with the same
/// current as `cfg.target_clamp_strength`, *and* the global
/// neuromodulator is set to `cfg.c1.teacher_strength` for the
/// duration of the clamp window so the existing R-STDP rule on
/// R2's network treats the supervised teacher epoch as the
/// `M_target = +1` gating window. The modulator is reset to 0
/// immediately after the clamp window so Phase 5's reward
/// delivery starts from a clean slate.
#[allow(clippy::too_many_arguments)]
fn run_teacher_trial(
    brain: &mut Brain,
    cfg: &TeacherForcingConfig,
    use_reward: bool,
    is_noise: bool,
    cue_sdr: &[u32],
    target_r2_sdr: &[u32],
    r2_e: &BTreeSet<usize>,
    rest_stdp: StdpParams,
    rest_istdp: IStdpParams,
    dg_sdr: &[u32],
    c1_target_sdr: &[u32],
) -> TrialOutcome {
    let mut outcome = TrialOutcome::default();
    let dg_active = !dg_sdr.is_empty() && cfg.dg.enabled;
    let dg_strength = if dg_active {
        cfg.dg.drive_strength
    } else {
        0.0
    };

    // ---- Phase 1: cue alone (R1 forward path + DG when enabled).
    if dg_active {
        drive_with_dg(
            brain,
            cue_sdr,
            dg_sdr,
            DRIVE_NA,
            dg_strength,
            cfg.cue_ms as f32,
        );
    } else {
        drive_for(brain, cue_sdr, cfg.cue_ms as f32);
    }

    // ---- Phase 2: delay (no input).
    idle(brain, cfg.delay_ms as f32);

    // ---- Phase 3: prediction. Plasticity off so the test does not
    //      mutate weights. We toggle STDP / iSTDP off, then back on
    //      after the prediction window. R-STDP eligibility decay
    //      continues normally — that's fine, it's the *gated*
    //      update we want to keep silenced (modulator stays 0).
    //
    //      `r1r2_prediction_gate` < 1.0 attenuates the cue current
    //      so the recurrent path (whatever STDP has shaped so far)
    //      gets a fair shot at biasing R2 against the otherwise-
    //      dominant R1 forward drive.
    // Iter-48 A/B: iSTDP is gated separately from STDP / R-STDP.
    // The default (`istdp_during_prediction = false`) preserves
    // the iter-46 invariant. Setting it `true` lets inhibitory
    // plasticity respond to the cue *during* the read-out — the
    // natural test for "does iSTDP catch the cascade in time"
    // identified by the iter-47a postmortem.
    let suppress_stdp = !cfg.plasticity_during_prediction;
    let suppress_istdp = !cfg.plasticity_during_prediction && !cfg.istdp_during_prediction;
    // Iter-63 plumbing fix: save the *current* plasticity state and
    // restore it at the end. The previous "always enable_*(rest_*)"
    // behaviour silently re-enabled STDP / iSTDP even when the
    // caller had them disabled (e.g. iter-63 untrained mode under
    // disable_all_plasticity), turning a "no plasticity" run into
    // a partial-plasticity run mid-trial. Bit-identity for
    // run_jaccard_arm / run_reward_benchmark is preserved because
    // those callers always pass `rest_stdp` / `rest_istdp` matching
    // the prior state when plasticity is on.
    let prior_stdp = brain.regions[1].network.stdp;
    let prior_istdp = brain.regions[1].network.istdp;
    if suppress_stdp {
        brain.regions[1].network.disable_stdp();
    }
    if suppress_istdp {
        brain.regions[1].network.disable_istdp();
    }
    let pred_counts = if dg_active {
        drive_with_r2_clamp_dg(
            brain,
            cue_sdr,
            dg_sdr,
            &[],
            DRIVE_NA * cfg.r1r2_prediction_gate,
            dg_strength * cfg.r1r2_prediction_gate,
            0.0,
            cfg.prediction_ms as f32,
            r2_e,
        )
    } else {
        drive_with_r2_clamp(
            brain,
            cue_sdr,
            &[],
            DRIVE_NA * cfg.r1r2_prediction_gate,
            0.0,
            cfg.prediction_ms as f32,
            r2_e,
        )
    };
    if suppress_stdp {
        match prior_stdp {
            Some(p) => brain.regions[1].network.enable_stdp(p),
            None => brain.regions[1].network.disable_stdp(),
        }
    }
    if suppress_istdp {
        match prior_istdp {
            Some(p) => brain.regions[1].network.enable_istdp(p),
            None => brain.regions[1].network.disable_istdp(),
        }
    }
    let pred_topk = top_k_indices(&pred_counts, cfg.wta_k.max(1));
    outcome.prediction_topk = pred_topk.clone();
    outcome.prediction_active_count = pred_counts.iter().filter(|&&c| c > 0).count() as u32;
    // Iter-47: count how many of the canonical-target neurons fired
    // during the prediction phase. This is the numerator of the
    // selectivity index — "did the cue alone reach the right
    // cells, before the clamp helped?".
    let target_set: BTreeSet<u32> = target_r2_sdr.iter().copied().collect();
    outcome.pred_target_hits = pred_counts
        .iter()
        .enumerate()
        .filter(|(i, &c)| c > 0 && target_set.contains(&(*i as u32)))
        .count() as u32;

    // ---- Phase 4: teacher. Cue + R2 target clamp. Plasticity ON
    //      (it is by default; this matches the iter-45 reps loop).
    //
    //      CRITICAL: drive cue alone for `cue_lead_in_teacher` ms
    //      *before* applying the clamp. The R1 → R2 inter-region
    //      delay is 2 ms, so without a lead-in the clamp would
    //      activate the target cells *before* the cue's R2 cells
    //      fire — STDP would then learn target → cue (anti-causal)
    //      instead of cue → target. The lead-in lets cue cells
    //      establish a pre-spike pattern that the clamped target
    //      cells then *follow*, giving STDP the right timing
    //      asymmetry.
    // Iter-63 plumbing fix: same save/restore as the prediction
    // phase above. Save *current* state; restore to the same
    // Option<…> value so callers with plasticity off don't get it
    // turned back on mid-trial.
    let prior_stdp_t = brain.regions[1].network.stdp;
    let prior_istdp_t = brain.regions[1].network.istdp;
    if !cfg.plasticity_during_teacher {
        brain.regions[1].network.disable_stdp();
        brain.regions[1].network.disable_istdp();
    }
    // Lead-in: cue alone, no clamp. ~ ¼ of the teacher window or
    // 8 ms, whichever is shorter, but always at least 4 ms so the
    // R1 → R2 delay can complete and pre-traces have time to
    // build before the post-spikes arrive.
    let lead_in_ms: u32 = (cfg.teacher_ms / 4).clamp(4, 12);
    let clamp_ms: u32 = cfg.teacher_ms.saturating_sub(lead_in_ms).max(1);
    let _lead_counts = if dg_active {
        drive_with_r2_clamp_dg(
            brain,
            cue_sdr,
            dg_sdr,
            &[],
            DRIVE_NA,
            dg_strength,
            0.0,
            lead_in_ms as f32,
            r2_e,
        )
    } else {
        drive_with_r2_clamp(brain, cue_sdr, &[], DRIVE_NA, 0.0, lead_in_ms as f32, r2_e)
    };
    // Iter-66: combine R2-target clamp indices with the C1-target
    // clamp indices. Both live inside R2's network at non-overlapping
    // ranges, so the existing R2-clamp helpers can drive both with
    // a single external-current vector. When `c1_target_sdr` is
    // empty (iter-46/65 path), `combined_clamp` borrows
    // `target_r2_sdr` as-is — no allocation, bit-identical to the
    // pre-iter-66 control flow.
    //
    // Iter-66.5 Path-1 fix (`notes/66.5-eval-aligned-c1-rstdp.md`):
    // when `cfg.c1.eval_aligned_rstdp` is set AND C1 is active
    // for this trial, the canonical R2 target SDR is dropped from
    // the clamp so R2 fires its natural cue-driven response. The
    // C1 target SDR clamp stays. R-STDP then aligns
    // (eval-time R2 cue pattern) → (canonical C1 target) instead
    // of iter-66's (canonical R2 target) → (canonical C1 target).
    // When the flag is off (default), this branch is skipped and
    // the iter-66 / iter-65 / iter-46 numerics are bit-identical.
    let drop_r2_clamp_for_c1 =
        cfg.c1.enabled && cfg.c1.eval_aligned_rstdp && !c1_target_sdr.is_empty();
    let combined_clamp: std::borrow::Cow<'_, [u32]> = if drop_r2_clamp_for_c1 {
        std::borrow::Cow::Borrowed(c1_target_sdr)
    } else if c1_target_sdr.is_empty() {
        std::borrow::Cow::Borrowed(target_r2_sdr)
    } else {
        let mut v = Vec::with_capacity(target_r2_sdr.len() + c1_target_sdr.len());
        v.extend_from_slice(target_r2_sdr);
        v.extend_from_slice(c1_target_sdr);
        std::borrow::Cow::Owned(v)
    };
    // Iter-66: target-presence-gated three-factor R-STDP. Save the
    // current modulator, set it to `cfg.c1.teacher_strength` for
    // the duration of the Phase 4 clamp window, then restore. When
    // `c1.enabled = false` this is a no-op.
    let prior_modulator = brain.regions[1].network.neuromodulator;
    let c1_active = cfg.c1.enabled && !c1_target_sdr.is_empty();
    if c1_active {
        brain.set_neuromodulator(cfg.c1.teacher_strength);
    }
    // Iter-67-α (post-Step-6 fix per Bekos's homeostasis-catch-22
    // diagnosis): when BTSP is on, hard-gate homeostasis OFF for
    // the duration of the Phase 4 clamp window. The 500 nA C1
    // target clamp drives canonical-target C1 cells to saturation
    // firing (~14 spikes / 30 ms). Homeostatic synaptic scaling
    // (`scale_only_down = true, a_target = 2.0`) then measures
    // that activity as far above target and scales the cells'
    // incoming R2-E → C1-target weights DOWN — directly cancelling
    // the BTSP plateau-gated potentiation. iter-67's first smoke
    // (notes/67-step-6-smoke-seed42-ep32-v2.log) confirmed the
    // catch-22: w_ratio = 0.42 oscillating, target-mean weight
    // 41 % of non-target, with weight movement of 0.0014/epoch
    // (≈ 700 epochs to reach K4's ratio ≥ 1.5). Disabling
    // homeostasis only during the clamp window — and only when
    // BTSP is on — preserves iter-46 / iter-65 / iter-66 numerics
    // bit-identically when c1.btsp = false. Same save/restore
    // pattern as the existing STDP / iSTDP gating above.
    let prior_homeostasis_t = brain.regions[1].network.homeostasis;
    if c1_active && cfg.c1.btsp {
        brain.regions[1].network.disable_homeostasis();
    }
    // Iter-67-γ.1 (E/I-split partial echo-state per Bekos's
    // locked γ.1 prompt): when BTSP is on, scale R2-R2 recurrent
    // synapse delivery SEPARATELY for E and I pre-cells.
    // Defaults `e = 1.0, i = 0.3` keep E recurrent at full
    // strength while reducing I-suppression — the strongest
    // cue-engram E-cells dominate without uniform attractor
    // saturation.  iter-67-β's uniform-scale sweep (notes/67
    // §"Step 7 — iter-67-β verdict") proved no scalar between
    // 0.0 and 0.80 produces both selectivity AND gain; γ.1
    // decouples E and I to address the architectural E/I
    // imbalance directly.  Combined with the iter-67-α2 R1+DG
    // drive cut above, this exposes cue-engram E-cells under
    // reduced inhibition.  R2-E → C1 (post >= r2_n_used) are
    // NOT scaled (pre_max = r2_n_used isolates the recurrent
    // block).  Stored synapse weight unchanged; STDP / iSTDP /
    // R-STDP / BTSP read un-scaled.
    let prior_recurrent_e_scale = brain.regions[1].network.recurrent_e_scale;
    let prior_recurrent_i_scale = brain.regions[1].network.recurrent_i_scale;
    let prior_recurrent_scale_pre_max = brain.regions[1].network.recurrent_scale_pre_max;
    if c1_active && cfg.c1.btsp {
        let r2_n = effective_r2_n(cfg);
        brain.regions[1].network.set_recurrent_e_i_scales(
            cfg.c1.btsp_teacher_recurrent_e_scale,
            cfg.c1.btsp_teacher_recurrent_i_scale,
            r2_n,
        );
    }
    // Iter-66: when C1 is active, augment the spike-tracking set
    // with the C1 cell index range so step-7.5 diagnostics can
    // read C1 spike counts out of `teacher_counts`. The original
    // `r2_e` slice covers only R2-E cells; C1 cells are at
    // `[r2_n_used, r2_n_used + cfg.c1.size)` and are *also*
    // excitatory but not in `r2_e_set`. The R2-side metrics
    // (`outcome.target_clamp_hits`) check `target_set` indices
    // (∈ R2 range only) so adding C1 indices to the tracking set
    // does NOT change R2 numerics — kwta on the returned counts
    // is not used in run_teacher_trial.
    let track_set_owned: BTreeSet<usize>;
    let track_set: &BTreeSet<usize> = if c1_active {
        let r2_n = effective_r2_n(cfg);
        let mut s = r2_e.clone();
        s.extend(r2_n..(r2_n + cfg.c1.size as usize));
        track_set_owned = s;
        &track_set_owned
    } else {
        r2_e
    };
    // Iter-67-α2 (R2-isolation per Bekos's selectivity-fix prompt):
    // when BTSP is on, gate the upstream cue drive (R1 → R2 +
    // DG → R2) to ZERO during the Phase 4 clamp window. The
    // motivation, post the homeostasis-gating fix
    // (notes/67-step-6-smoke-seed42-ep32-v3-homeostasis-gated.log
    // confirmed BTSP saturates indiscriminately when R2 keeps
    // firing during teacher): without this isolation, R2 fires
    // its full cue + recurrent + DG response throughout teacher,
    // BTSP tags ALL active R2-E synapses (engram + noise), and
    // the plateau-arm event potentiates them all uniformly →
    // w_ratio ≈ 1.0 (target ≈ non-target). With cue input cut
    // during teacher, only the residual cue-engram membrane
    // potentials carry over from cue/delay/prediction/lead-in;
    // those decay quickly under tau_m = 20 ms, leaving the
    // C1-target clamp as the dominant post-side driver. BTSP
    // tags accumulated DURING the cue/delay/prediction substrate
    // (via the long eligibility_window_ms = 200) capture the
    // engram cells; tags during teacher add only a small residual
    // from membrane decay. Plateau-arm potentiates these
    // engram-biased tagged synapses → target/non-target weight
    // separation. iter-66/iter-66.5 path bit-identical when
    // c1.btsp = false (gate is `c1_active && cfg.c1.btsp`).
    // Iter-67-γ.1.1: opt-out switch.  When `cfg.c1.no_r2_isolation`
    // is set, keep cue + DG drive at full strength during teacher
    // (default behaviour pre-iter-67-α2).  Tests Bekos's actual
    // locked γ.1 hypothesis: cue-engram E-cells fire under
    // reduced inhibition.  When the flag is off (default),
    // iter-67-α2's R2-isolation stays ON ⇒ bit-identical to
    // iter-67-γ.1 / iter-67-α2 baselines.
    let isolate_r2_for_btsp = c1_active && cfg.c1.btsp && !cfg.c1.btsp_no_r2_isolation;
    let teacher_r1_strength = if isolate_r2_for_btsp { 0.0 } else { DRIVE_NA };
    let teacher_dg_strength = if isolate_r2_for_btsp {
        0.0
    } else {
        dg_strength
    };
    let teacher_counts = if dg_active {
        drive_with_r2_clamp_dg(
            brain,
            cue_sdr,
            dg_sdr,
            &combined_clamp,
            teacher_r1_strength,
            teacher_dg_strength,
            cfg.target_clamp_strength,
            clamp_ms as f32,
            track_set,
        )
    } else {
        drive_with_r2_clamp(
            brain,
            cue_sdr,
            &combined_clamp,
            teacher_r1_strength,
            cfg.target_clamp_strength,
            clamp_ms as f32,
            track_set,
        )
    };
    if c1_active {
        brain.set_neuromodulator(prior_modulator);
    }
    // Iter-67-α: restore homeostasis state to whatever it was
    // before the clamp window. When `cfg.c1.btsp = false` this is
    // a pure no-op (we never disabled it).
    if c1_active && cfg.c1.btsp {
        if let Some(h) = prior_homeostasis_t {
            brain.regions[1].network.enable_homeostasis(h);
        }
    }
    // Iter-67-γ.1: restore E and I recurrent scales to their
    // prior values.  When `cfg.c1.btsp = false` this is a pure
    // no-op (we never touched them; the saved values are the
    // defaults 1.0 / 1.0 / u32::MAX).
    if c1_active && cfg.c1.btsp {
        brain.regions[1].network.set_recurrent_e_i_scales(
            prior_recurrent_e_scale,
            prior_recurrent_i_scale,
            prior_recurrent_scale_pre_max as usize,
        );
    }
    if !cfg.plasticity_during_teacher {
        match prior_stdp_t {
            Some(p) => brain.regions[1].network.enable_stdp(p),
            None => brain.regions[1].network.disable_stdp(),
        }
        match prior_istdp_t {
            Some(p) => brain.regions[1].network.enable_istdp(p),
            None => brain.regions[1].network.disable_istdp(),
        }
    }
    // Acknowledge the legacy `rest_stdp` / `rest_istdp` parameters —
    // they are preserved on the public function signature for
    // caller-compat but no longer drive the plasticity restore (the
    // saved-state mechanism above does, which is what makes
    // iter-63's untrained-mode iter-52 invariant hold).
    let _ = (rest_stdp, rest_istdp);
    // Diagnostic: how reliably did the clamp drive the intended
    // set? Reuses `target_set` already built in the prediction
    // phase above (iter-47 selectivity readout).
    let clamp_hit = target_set
        .iter()
        .filter(|&&i| teacher_counts[i as usize] > 0)
        .count();
    outcome.target_clamp_hits = clamp_hit as u32;
    outcome.target_clamp_size = target_r2_sdr.len() as u32;

    // Iter-66 step 7.5 diagnostic: capture C1 spike statistics
    // during the teacher clamp window. The C1 cell index range
    // is `[r2_n_used, r2_n_used + cfg.c1.size)` (when c1.enabled).
    // Cheap when c1 is off — early-return without iteration.
    if c1_active {
        let r2_n_used = effective_r2_n(cfg);
        let c1_start = r2_n_used;
        let c1_end = r2_n_used + cfg.c1.size as usize;
        let c1_total: u32 = teacher_counts
            .get(c1_start..c1_end)
            .map(|s| s.iter().sum())
            .unwrap_or(0);
        outcome.c1_teacher_spikes = c1_total;
        let c1_target_set: BTreeSet<u32> = c1_target_sdr.iter().copied().collect();
        let c1_clamp_hit = c1_target_set
            .iter()
            .filter(|&&i| {
                let idx = i as usize;
                idx < teacher_counts.len() && teacher_counts[idx] > 0
            })
            .count();
        outcome.c1_target_clamp_hits = c1_clamp_hit as u32;
        outcome.c1_target_clamp_size = c1_target_sdr.len() as u32;
    }

    // ---- Phase 5: reward. Score the *prediction* against the
    //      canonical target — teacher activation does NOT count.
    let target_in_topk = pred_topk.iter().any(|i| target_set.contains(i));
    let any_wrong_topk = pred_topk.iter().any(|i| !target_set.contains(i));
    let reward = if !use_reward {
        0.0
    } else if is_noise {
        cfg.noise_reward
    } else if target_in_topk {
        cfg.positive_reward_for_correct
    } else if any_wrong_topk {
        cfg.negative_reward_for_false_topk
    } else {
        // Empty prediction or all targets — neutral.
        0.0
    };
    outcome.prediction_correct = target_in_topk && !is_noise;
    outcome.reward_value = reward;

    if use_reward && reward != 0.0 && cfg.reward_after_teacher {
        brain.set_neuromodulator(reward);
        // Brief consolidation drive — same length as the teacher
        // phase, cue alone (no clamp) so the modulator gates the
        // eligibility tag built up during the teacher window.
        if dg_active {
            drive_with_dg(
                brain,
                cue_sdr,
                dg_sdr,
                DRIVE_NA,
                dg_strength,
                cfg.teacher_ms as f32,
            );
        } else {
            drive_for(brain, cue_sdr, cfg.teacher_ms as f32);
        }
        brain.set_neuromodulator(0.0);
    }

    // ---- Phase 6: tail (let traces decay).
    idle(brain, cfg.tail_ms as f32);

    outcome
}

/// Per-trial diagnostics that the teacher schedule emits.
/// All fields default to zero so the caller can accumulate them
/// across a whole epoch without special cases.
#[derive(Debug, Default, Clone)]
struct TrialOutcome {
    prediction_topk: Vec<u32>,
    prediction_active_count: u32,
    target_clamp_hits: u32,
    target_clamp_size: u32,
    prediction_correct: bool,
    reward_value: f32,
    /// Iter-47: number of canonical-target neurons that fired at
    /// least once during the prediction phase (before the clamp).
    /// Lets the epoch loop compute the selectivity index without
    /// re-running R2 step.
    pred_target_hits: u32,
    /// Iter-66 step 7.5 diagnostic: total C1 spike count across
    /// the teacher (Phase 4) clamp window. Zero when c1.enabled
    /// is false. Used by the per-epoch diagnostic accumulator
    /// to discriminate "silent C1" from "C1 active but not
    /// discriminative".
    c1_teacher_spikes: u32,
    /// Iter-66 step 7.5 diagnostic: count of canonical-target C1
    /// cells that fired at least once during the teacher clamp.
    /// Discriminates "C1 target SDR clamp ineffective" from "C1
    /// fires but R-STDP not gated to canonical cells".
    c1_target_clamp_hits: u32,
    /// Iter-66 step 7.5 diagnostic: clamp window size in cells
    /// (denominator for `c1_target_clamp_hits / size` ratio).
    c1_target_clamp_size: u32,
}

// ----------------------------------------------------------------------
// The harness. Three phases per trial: paired training, evaluation,
// optional reward delivery.
// ----------------------------------------------------------------------

/// Run the pair-association benchmark for `cfg.epochs` epochs and
/// return per-epoch metrics. The harness builds its own `Brain` and
/// dictionary so the call is self-contained.
/// Iter-47a postmortem driver. Trains for `train_epochs` epochs on
/// the corpus (using the supplied teacher config), then for the
/// first real pair runs ONE additional teacher trial in which the
/// prediction phase is captured step-by-step. Prints:
///   - per-step R2-E active count (onset-burst vs sustained-runaway)
///   - per-step canonical-target hit count
///   - final intrinsic-θ distribution stats over R2-E
///   - mean / max R2 → R2 weight
///
/// Read-only after the configured training; does not mutate the
/// epoch metric stream. Returns the final intrinsic stats as a
/// quintuple `(theta_mean, theta_std, theta_min, theta_max, frac>1mV)`
/// so callers can render them in a table.
/// Iter-53 determinism smoke (Bekos's pre-implementation gate
/// for the Jaccard-consistency metric path B). Builds an
/// untrained brain, builds the per-vocab fingerprint dictionary
/// once, then presents the first real-pair cue 3× with
/// `reset_state` between trials. Reports the per-trial top-3
/// sets and the pairwise Jaccard similarity.
///
/// Decision matrix (Bekos):
/// - mean Jaccard ≈ 1.0 → system is deterministic per cue,
///   Jaccard metric trivial, fall back to Option A
///   (per-trial trained-minus-untrained Δ).
/// - mean Jaccard < 1.0 but > 0 → Jaccard is informative,
///   proceed to Option B (full implementation).
/// - mean Jaccard ≈ 0 → too random, edge case.
///
/// Uses no plasticity by construction (forced regardless of
/// `cfg.teacher.no_plasticity`) so the determinism observed
/// is the pure brain dynamics, not a side effect of any
/// learning rule.
pub fn run_determinism_smoke(corpus: &RewardCorpus, cfg: &RewardConfig) {
    let (inter_weight_used, inh_frac_used) = if cfg.teacher.iter46_baseline {
        (2.0_f32, 0.20_f32)
    } else {
        (INTER_WEIGHT, R2_INH_FRAC)
    };
    let r2_n_used = effective_r2_n(&cfg.teacher);
    let r2_p_connect_used = effective_r2_p_connect(&cfg.teacher);
    let mut brain = fresh_brain_with(
        cfg.seed,
        inter_weight_used,
        inh_frac_used,
        r2_n_used,
        r2_p_connect_used,
    );
    let encoder = TextEncoder::with_stopwords(ENC_N, ENC_K, std::iter::empty::<&str>());
    let r2_e = r2_e_set(&brain);

    eprintln!(
        "[iter-53 smoke] determinism test — seed={}, INTER_WEIGHT={}, R2_INH_FRAC={}",
        cfg.seed, inter_weight_used, inh_frac_used,
    );
    eprintln!("[iter-53 smoke] plasticity: ALL OFF (forced for determinism check)");

    // No enable_* calls — pure forward + recurrent dynamics, no
    // plasticity. This is the cleanest baseline for "does the
    // same cue produce the same top-3 across reset_state cycles".
    let pre_l2 = brain_synapse_l2_norms(&brain);

    let vocab: Vec<String> = corpus.vocab.iter().cloned().collect();
    // Build the fingerprint dictionary once. Each call to
    // `drive_for_with_counts` does reset_state internally inside
    // build_vocab_dictionary so dictionary construction itself
    // is deterministic given the seed.
    let empty_dg: std::collections::HashMap<String, Vec<u32>> = std::collections::HashMap::new();
    let dict = build_vocab_dictionary(&mut brain, &encoder, &r2_e, &vocab, &empty_dg, 0.0);

    // Pick the first real-pair cue.
    let pair = &corpus.pairs[0];
    let cue_sdr = encoder.encode_word(&pair.cue);
    eprintln!(
        "[iter-53 smoke] cue = '{}' target = '{}' (vocab size = {}, decode k = 3)",
        pair.cue,
        pair.target,
        vocab.len(),
    );

    let mut trials: Vec<Vec<String>> = Vec::with_capacity(3);
    for trial_i in 0..3 {
        brain.regions[1].network.reset_state();
        let counts = drive_for_with_counts(&mut brain, &cue_sdr.indices, RECALL_MS, &r2_e);
        let kwta = top_k_indices(&counts, KWTA_K);
        let active = counts.iter().filter(|&&c| c > 0).count();
        let decoded = dict.decode_top(&kwta, 3);
        let top3_words: Vec<String> = decoded.iter().map(|(w, _)| w.clone()).collect();
        let target_in = top3_words.iter().any(|w| w == &pair.target);
        eprintln!(
            "[iter-53 smoke] trial {trial_i}: r2_active={active} top-3={top3_words:?} target_in={target_in}",
        );
        trials.push(top3_words);
    }

    // Pairwise Jaccard on top-3 word sets.
    let jaccard = |a: &[String], b: &[String]| -> f32 {
        let sa: BTreeSet<&String> = a.iter().collect();
        let sb: BTreeSet<&String> = b.iter().collect();
        let inter = sa.intersection(&sb).count();
        let uni = sa.union(&sb).count();
        if uni == 0 {
            0.0
        } else {
            inter as f32 / uni as f32
        }
    };
    let j12 = jaccard(&trials[0], &trials[1]);
    let j23 = jaccard(&trials[1], &trials[2]);
    let j13 = jaccard(&trials[0], &trials[2]);
    let mean_j = (j12 + j23 + j13) / 3.0;

    eprintln!("[iter-53 smoke] pairwise Jaccard: 1↔2 = {j12:.3} | 2↔3 = {j23:.3} | 1↔3 = {j13:.3}",);
    eprintln!("[iter-53 smoke] mean Jaccard = {mean_j:.3}");

    // L2 sanity assertion (Layer-4 lesson from iter-52).
    let post_l2 = brain_synapse_l2_norms(&brain);
    let identical = pre_l2.iter().zip(post_l2.iter()).all(|(a, b)| a == b);
    eprintln!(
        "[iter-53 smoke] L2 invariant: pre={pre_l2:?} post={post_l2:?} bit-identical={identical}",
    );
    assert!(
        identical,
        "iter-53 smoke: weights changed under no-plasticity path",
    );

    eprintln!("[iter-53 smoke] Bekos decision matrix:");
    eprintln!("  mean Jaccard ≈ 1.0  → deterministic, Option B trivial, fall back to Option A");
    eprintln!("  0 < mean J < 1.0    → Jaccard informative, proceed to Option B");
    eprintln!("  mean Jaccard ≈ 0    → too random, edge case");
}

pub fn run_postmortem_diagnostic(
    corpus: &RewardCorpus,
    cfg: &RewardConfig,
    train_epochs: usize,
) -> (f32, f32, f32, f32, f32) {
    // Mirror run_reward_benchmark's brain construction, train for
    // train_epochs, then run one diagnostic trial on the first
    // real pair.
    let mut brain = fresh_brain(cfg.seed);
    let encoder = TextEncoder::with_stopwords(ENC_N, ENC_K, std::iter::empty::<&str>());
    let r2_e = r2_e_set(&brain);
    let stdp_params = stdp();
    let istdp_params = istdp();
    brain.regions[1].network.enable_stdp(stdp_params);
    brain.regions[1].network.enable_istdp(istdp_params);
    brain.regions[1].network.enable_homeostasis(homeostasis());
    brain.regions[1]
        .network
        .enable_intrinsic_plasticity(intrinsic());
    if cfg.use_reward {
        brain.regions[1]
            .network
            .enable_reward_learning(reward_params());
    }
    let mut rng = Rng::new(cfg.seed);
    let pool = r2_e_pool(&r2_e);
    let salt = cfg.seed ^ 0xCAFE_F00D_DEAD_BEEFu64;
    let target_r2_map: std::collections::HashMap<String, Vec<u32>> = corpus
        .vocab
        .iter()
        .map(|w| {
            (
                w.clone(),
                canonical_target_r2_sdr(w, &pool, TARGET_R2_K, salt),
            )
        })
        .collect();

    eprintln!(
        "[postmortem] train_epochs={train_epochs} pairs={} noise={} INTER_WEIGHT={}",
        corpus.pairs.len(),
        corpus.noise_pairs.len(),
        INTER_WEIGHT,
    );
    for epoch in 0..train_epochs {
        let mut schedule: Vec<(RewardPair, bool)> = corpus
            .pairs
            .iter()
            .map(|p| (p.clone(), false))
            .chain(corpus.noise_pairs.iter().map(|p| (p.clone(), true)))
            .collect();
        shuffle(&mut rng, &mut schedule);
        for (pair, is_noise) in &schedule {
            let cue_sdr = encoder.encode_word(&pair.cue);
            let canonical = target_r2_map.get(&pair.target).cloned().unwrap_or_default();
            for _ in 0..cfg.reps_per_pair.max(1) {
                brain.regions[1].network.reset_state();
                let _ = run_teacher_trial(
                    &mut brain,
                    &cfg.teacher,
                    cfg.use_reward,
                    *is_noise,
                    &cue_sdr.indices,
                    &canonical,
                    &r2_e,
                    stdp_params,
                    istdp_params,
                    &[],
                    &[],
                );
                idle(&mut brain, COOLDOWN_MS);
            }
        }
        let (m, s, mn, mx, fr) = intrinsic_stats(&brain, &r2_e);
        let net = &brain.regions[1].network;
        let w_max = net
            .synapses
            .iter()
            .map(|s| s.weight)
            .fold(0.0_f32, f32::max);
        let w_sum: f64 = net.synapses.iter().map(|s| s.weight as f64).sum();
        let w_mean = (w_sum / net.synapses.len().max(1) as f64) as f32;
        eprintln!(
            "[postmortem epoch {epoch}] θ mean={:.3} std={:.3} min={:.3} max={:.3} frac>1mV={:.3} | w̄={:.3} wmax={:.3}",
            m, s, mn, mx, fr, w_mean, w_max,
        );
    }
    eprintln!("[postmortem] training done; capturing per-step prediction trace …");

    // Diagnostic trial — first real pair, plasticity OFF on all
    // rules so the trace is a pure read-out.
    let pair = &corpus.pairs[0];
    let cue_sdr = encoder.encode_word(&pair.cue);
    let _canonical = target_r2_map.get(&pair.target).cloned().unwrap_or_default();
    brain.regions[1].network.disable_stdp();
    brain.regions[1].network.disable_istdp();
    brain.set_neuromodulator(0.0);
    brain.regions[1].network.reset_state();

    // Phase-1 cue, phase-2 delay, phase-3 prediction TRACED.
    drive_for(&mut brain, &cue_sdr.indices, cfg.teacher.cue_ms as f32);
    idle(&mut brain, cfg.teacher.delay_ms as f32);
    let (_counts, per_step_active, per_step_target) = drive_with_r2_clamp_traced(
        &mut brain,
        &cue_sdr.indices,
        &[],
        DRIVE_NA * cfg.teacher.r1r2_prediction_gate,
        0.0,
        cfg.teacher.prediction_ms as f32,
        &r2_e,
    );

    eprintln!(
        "[postmortem] prediction phase per-step trace ({} steps × {:.2} ms = {:.0} ms)",
        per_step_active.len(),
        DT,
        cfg.teacher.prediction_ms,
    );
    eprintln!("  step | active | target");
    for (i, (&a, &t)) in per_step_active
        .iter()
        .zip(per_step_target.iter())
        .enumerate()
    {
        // Print every 10th step plus last to stay readable.
        if i % 10 == 0 || i + 1 == per_step_active.len() {
            eprintln!("  {:>4} | {:>5} | {:>5}", i, a, t);
        }
    }
    let total_active: u64 = per_step_active.iter().map(|&x| x as u64).sum();
    let total_target: u64 = per_step_target.iter().map(|&x| x as u64).sum();
    let max_active = per_step_active.iter().max().copied().unwrap_or(0);
    let max_target = per_step_target.iter().max().copied().unwrap_or(0);
    let early = per_step_active
        .iter()
        .take(per_step_active.len() / 4)
        .map(|&x| x as u64)
        .sum::<u64>();
    let late = per_step_active
        .iter()
        .skip(per_step_active.len() * 3 / 4)
        .map(|&x| x as u64)
        .sum::<u64>();
    let onset_dominance = if late > 0 {
        early as f32 / late as f32
    } else {
        f32::INFINITY
    };
    eprintln!(
        "[postmortem] sums: total_active={total_active} total_target={total_target} max_active/step={max_active} max_target/step={max_target}",
    );
    eprintln!(
        "[postmortem] early-vs-late dominance ratio (sum first 25% / sum last 25%): {onset_dominance:.2}",
    );
    eprintln!(
        "[postmortem]   > 2.0 ⇒ onset-burst pattern (k-WTA fix); ≤ 1.5 ⇒ sustained drive (iSTDP fix)",
    );

    intrinsic_stats(&brain, &r2_e)
}

/// **Legacy iter-44 / iter-46 reward-benchmark runner.** Do **not**
/// extend with new architectural axes (DG bridge, decorrelated
/// init, recall-mode eval, future CA3/CA1 split). The iter-63
/// plumbing fix lifted the brain-build code into
/// `build_benchmark_brain` and the plasticity gating into
/// `disable_all_plasticity`; new runners must consume those
/// helpers instead of going through this function.
///
/// This function is kept verbatim because its numerical output
/// is the **calibrated baseline** behind iter-46's 0.19 reading
/// and iter-51's 0.107 stable estimator — both of which iter-63
/// uses as the positive-control band. Refactoring this code path
/// would invalidate those calibrations.
///
/// TODO(iter-64+): if a future iteration needs this runner on a
/// DG-enabled brain, build a v2 alongside (mirroring
/// `run_target_overlap_arm`'s helper-based structure) rather than
/// modifying this function.
pub fn run_reward_benchmark(corpus: &RewardCorpus, cfg: &RewardConfig) -> Vec<RewardEpochMetrics> {
    // Iter-50 diagnostic: when iter46_baseline is set, build the
    // brain with the original iter-46 INTER_WEIGHT (2.0) and
    // R2_INH_FRAC (0.20) instead of the iter-47/49 drift values
    // (1.0 / 0.30). This is the topology Arm B saw when it
    // reported top-3 = 0.19 in iter-46.
    let (inter_weight_used, inh_frac_used) = if cfg.teacher.iter46_baseline {
        (2.0_f32, 0.20_f32)
    } else {
        (INTER_WEIGHT, R2_INH_FRAC)
    };
    let r2_n_used = effective_r2_n(&cfg.teacher);
    let r2_p_connect_used = effective_r2_p_connect(&cfg.teacher);
    let mut brain = fresh_brain_with(
        cfg.seed,
        inter_weight_used,
        inh_frac_used,
        r2_n_used,
        r2_p_connect_used,
    );
    let encoder = TextEncoder::with_stopwords(ENC_N, ENC_K, std::iter::empty::<&str>());
    let r2_e = r2_e_set(&brain);

    // Plasticity setup. `enable_reward_learning` allocates the per-
    // synapse eligibility tag; without it the modulator setter is a
    // no-op even if the caller flips it on.
    let stdp_params = stdp();
    // Iter-49: iSTDP params now mode-aware. Base = iter-48 baseline
    // (`istdp()`); each mode overrides per-epoch via `istdp_iter49`.
    // ActivityGated additionally re-enables iSTDP at the start of
    // each epoch with the current ramp value.
    //
    // Iter-50 diagnostic: when iter46_baseline is set, use the
    // original iter-46 iSTDP parameter set (a_plus 0.10,
    // tau_minus 30 ms) instead of iter-48's tightened values.
    let mut istdp_params = if cfg.teacher.iter46_baseline {
        istdp_iter46_baseline()
    } else {
        istdp_iter49(&cfg.teacher, 0)
    };
    // Iter-52 untrained control: gate every plasticity enable.
    // The brain becomes a pure random-weight forward projection
    // with recurrent spike propagation but no weight updates.
    // L2-norm sanity asserts at the end of the run prove the
    // gate is tight (post-training norms must equal pre-training).
    let no_plasticity = cfg.teacher.no_plasticity;
    if !no_plasticity {
        brain.regions[1].network.enable_stdp(stdp_params);
        brain.regions[1].network.enable_istdp(istdp_params);
        brain.regions[1].network.enable_homeostasis(homeostasis());
    }
    // Iter-47a-2: Diehl-Cook adaptive threshold to keep R2 active
    // cell count in a sparse band per cue. Without this the reduced
    // INTER_WEIGHT alone risks variance blow-up (some cues over,
    // some under-active). Tracked in metrics:
    // r2_active_pre_teacher (mean, p10, p90), selectivity_index.
    //
    // Iter-50 diagnostic: skip when iter46_baseline is set, since
    // iter-46 Arm B did not enable intrinsic plasticity.
    // Iter-52: also skip under no_plasticity.
    if !cfg.teacher.iter46_baseline && !no_plasticity {
        brain.regions[1]
            .network
            .enable_intrinsic_plasticity(intrinsic());
    }
    if cfg.use_reward && !no_plasticity {
        brain.regions[1]
            .network
            .enable_reward_learning(reward_params());
    }
    // Iter-52 sanity: capture the L2 norm of every region's
    // synapse weights at run start. End-of-run norms must equal
    // these bit-for-bit when no_plasticity is true.
    let pre_norms = if no_plasticity {
        Some(brain_synapse_l2_norms(&brain))
    } else {
        None
    };
    if no_plasticity {
        eprintln!(
            "[iter-52] Plasticity gated: STDP=off iSTDP=off Homeostasis=off \
             AdaptiveTheta=off R-STDP=off | initial L2 (R1→R2, R2→R2)={:?}",
            pre_norms.as_ref().unwrap(),
        );
    }

    // Iter-46: pre-compute the canonical target-R2 SDR for every
    // word in the corpus. The mapping is stable per (word, salt)
    // and never recomputed per trial.
    //
    // Iter-50 diagnostic: built unconditionally (cheap hash op)
    // so the iter-47/48/49 sparsity metrics
    // (selectivity_index, target_hit_pre_teacher_mean) are also
    // populated in the iter-46 Arm B path. The map is used only
    // for clamp/scoring; it does not change the trial schedule.
    let teacher_active = cfg.teacher.enabled;
    let target_r2_map: std::collections::HashMap<String, Vec<u32>> = {
        let pool = r2_e_pool(&r2_e);
        let salt = cfg.seed ^ 0xCAFE_F00D_DEAD_BEEFu64;
        corpus
            .vocab
            .iter()
            .map(|w| {
                (
                    w.clone(),
                    canonical_target_r2_sdr(w, &pool, TARGET_R2_K, salt),
                )
            })
            .collect()
    };

    // Fingerprint the vocabulary once *before* any training so the
    // dictionary contains the encoder's address-side mapping. The
    // R2-side engrams will be rebuilt at the end of every epoch
    // (because the recurrent weights drift). The fingerprint is a
    // brief, plasticity-free recall pass.
    let vocab: Vec<String> = corpus.vocab.iter().cloned().collect();

    // Build a per-epoch RNG stream — Rng is Copy, so derived RNGs are
    // cheap and reproducible.
    let mut rng = Rng::new(cfg.seed);

    let mut metrics: Vec<RewardEpochMetrics> = Vec::with_capacity(cfg.epochs);

    for epoch in 0..cfg.epochs {
        // Iter-49 ActivityGated: re-enable iSTDP at the start of the
        // epoch with the current ramp value. The other modes (None /
        // WmaxCap / APlusHalf) return the same params on every epoch
        // so the call is a no-op for them — but it is cheap enough
        // (one parameter copy) that we don't bother branching.
        //
        // Iter-50 diagnostic: when iter46_baseline is set, use the
        // original iter-46 iSTDP every epoch (no ramp).
        // Iter-52: gate under no_plasticity.
        istdp_params = if cfg.teacher.iter46_baseline {
            istdp_iter46_baseline()
        } else {
            istdp_iter49(&cfg.teacher, epoch)
        };
        if !no_plasticity {
            brain.regions[1].network.enable_istdp(istdp_params);
        }

        // Build the trial schedule for this epoch: every real pair +
        // every noise pair, shuffled.
        let mut schedule: Vec<(RewardPair, bool)> = corpus
            .pairs
            .iter()
            .map(|p| (p.clone(), false))
            .chain(corpus.noise_pairs.iter().map(|p| (p.clone(), true)))
            .collect();
        shuffle(&mut rng, &mut schedule);

        let mut total_reward = 0.0_f32;
        let mut reward_trials = 0usize;
        // Iter-46 per-epoch accumulators.
        let mut clamp_hit_total: u64 = 0;
        let mut clamp_size_total: u64 = 0;
        let mut prediction_top3_hits: u32 = 0;
        let mut prediction_top3_total: u32 = 0;
        let mut debug_emitted: u32 = 0;
        // Iter-47 sparsity samples — collected per real-pair trial,
        // percentiles taken at the end of the epoch.
        let mut active_samples: Vec<u32> = Vec::new();
        let mut target_hit_samples: Vec<u32> = Vec::new();

        for (pair, is_noise) in &schedule {
            let cue_sdr = encoder.encode_word(&pair.cue);
            let tgt_sdr = encoder.encode_word(&pair.target);

            if teacher_active {
                // -- Iter-46: six-phase trial with R2 target clamp.
                //    See `run_teacher_trial`. The schedule cleanly
                //    separates training (teacher phase, plasticity
                //    on) from evaluation (prediction phase,
                //    plasticity off). The reward is computed from
                //    the prediction *before* the teacher fires, so
                //    teacher-induced activation never counts as
                //    recall success.
                let canonical = target_r2_map.get(&pair.target).cloned().unwrap_or_default();
                for rep in 0..cfg.reps_per_pair.max(1) {
                    brain.regions[1].network.reset_state();
                    let outcome = run_teacher_trial(
                        &mut brain,
                        &cfg.teacher,
                        cfg.use_reward,
                        *is_noise,
                        &cue_sdr.indices,
                        &canonical,
                        &r2_e,
                        stdp_params,
                        istdp_params,
                        &[],
                        &[],
                    );
                    if cfg.use_reward {
                        total_reward += outcome.reward_value;
                        reward_trials += 1;
                    }
                    // Iter-46 trial diagnostics.
                    clamp_hit_total += outcome.target_clamp_hits as u64;
                    clamp_size_total += outcome.target_clamp_size as u64;
                    if !*is_noise {
                        // Did the prediction-phase top-k contain *any*
                        // canonical target neuron? (top3 in name —
                        // wta_k controls actual k.)
                        let target_set: BTreeSet<u32> = canonical.iter().copied().collect();
                        if outcome
                            .prediction_topk
                            .iter()
                            .any(|i| target_set.contains(i))
                        {
                            prediction_top3_hits += 1;
                        }
                        prediction_top3_total += 1;
                        // Iter-47: per-trial sparsity samples.
                        active_samples.push(outcome.prediction_active_count);
                        target_hit_samples.push(outcome.pred_target_hits);
                    }
                    // Debug: print the first `debug_trials` real-pair
                    // diagnostics each epoch so the operator can see
                    // the clamp / prediction / reward chain.
                    if cfg.teacher.debug_trials > 0
                        && debug_emitted < cfg.teacher.debug_trials
                        && !*is_noise
                        && rep == 0
                    {
                        eprintln!(
                            "  [debug epoch={epoch} pair=({}, {})] \
                             clamp={}/{} pred_active={} pred_topk={:?} \
                             pred_correct={} reward={:+.2}",
                            pair.cue,
                            pair.target,
                            outcome.target_clamp_hits,
                            outcome.target_clamp_size,
                            outcome.prediction_active_count,
                            outcome.prediction_topk,
                            outcome.prediction_correct,
                            outcome.reward_value,
                        );
                        debug_emitted += 1;
                    }
                    idle(&mut brain, COOLDOWN_MS);
                }
            } else {
                // -- Iter-45 fallback: cue-then-target staggered
                //    training, no R2 clamp.
                let mut combined: Vec<u32> = cue_sdr
                    .indices
                    .iter()
                    .chain(tgt_sdr.indices.iter())
                    .copied()
                    .collect();
                combined.sort_unstable();
                combined.dedup();
                for _ in 0..cfg.reps_per_pair.max(1) {
                    brain.regions[1].network.reset_state();
                    drive_for(&mut brain, &cue_sdr.indices, CUE_LEAD_MS);
                    drive_for(&mut brain, &combined, OVERLAP_MS);
                    drive_for(&mut brain, &tgt_sdr.indices, TARGET_TAIL_MS);
                }

                // Iter-45 reward: cue-vs-target overlap proxy, then
                // brief consolidation drive with the modulator set.
                if cfg.use_reward {
                    let reward = if *is_noise {
                        -1.0_f32
                    } else {
                        brain.regions[1].network.reset_state();
                        let counts =
                            drive_for_with_counts(&mut brain, &cue_sdr.indices, RECALL_MS, &r2_e);
                        let kwta = top_k_indices(&counts, KWTA_K);
                        let target_kwta = {
                            brain.regions[1].network.reset_state();
                            let cs = drive_for_with_counts(
                                &mut brain,
                                &tgt_sdr.indices,
                                RECALL_MS,
                                &r2_e,
                            );
                            top_k_indices(&cs, KWTA_K)
                        };
                        let overlap = kwta.iter().filter(|i| target_kwta.contains(i)).count();
                        let ratio = overlap as f32 / target_kwta.len().max(1) as f32;
                        if ratio >= 0.30 {
                            1.0
                        } else if ratio >= 0.15 {
                            0.0
                        } else {
                            -0.5
                        }
                    };
                    if reward != 0.0 {
                        brain.set_neuromodulator(reward);
                        drive_for(&mut brain, &cue_sdr.indices, CONSOLIDATION_MS);
                        brain.set_neuromodulator(0.0);
                    }
                    total_reward += reward;
                    reward_trials += 1;
                }

                // Iter-50 diagnostic: collect the same per-trial
                // sparsity samples the teacher path collects, so
                // selectivity_index / target_hit_pre_teacher_mean
                // are populated for the iter-46 Arm B path too.
                // Plasticity-OFF read using step_immutable would be
                // cleaner; here we toggle STDP/iSTDP off for the
                // brief sample to keep weights stable, then back on.
                if !*is_noise {
                    let canonical = target_r2_map.get(&pair.target).cloned().unwrap_or_default();
                    // Iter-52: skip the disable/enable cycle when
                    // no_plasticity is true — STDP / iSTDP were
                    // never enabled, so re-enabling them mid-run
                    // would silently turn plasticity on (the leak
                    // the L2-norm sanity assert caught on first
                    // run).
                    if !no_plasticity {
                        brain.regions[1].network.disable_stdp();
                        brain.regions[1].network.disable_istdp();
                    }
                    brain.regions[1].network.reset_state();
                    let pred_counts =
                        drive_for_with_counts(&mut brain, &cue_sdr.indices, RECALL_MS, &r2_e);
                    if !no_plasticity {
                        brain.regions[1].network.enable_stdp(stdp_params);
                        brain.regions[1].network.enable_istdp(istdp_params);
                    }
                    let target_set: BTreeSet<u32> = canonical.iter().copied().collect();
                    let active = pred_counts.iter().filter(|&&c| c > 0).count() as u32;
                    let target_hits = pred_counts
                        .iter()
                        .enumerate()
                        .filter(|(i, &c)| c > 0 && target_set.contains(&(*i as u32)))
                        .count() as u32;
                    active_samples.push(active);
                    target_hit_samples.push(target_hits);
                }

                idle(&mut brain, COOLDOWN_MS);
            }
        }

        // ---- Epoch readout. Two parallel evaluations:
        //
        //   * iter-45 path (always run): build a dictionary by
        //     fingerprinting every vocab word against the current
        //     state, score top-1/top-3 against that dictionary.
        //
        //   * iter-46 path (teacher_active only): score against the
        //     *canonical* target SDR map. A real pair is "correct"
        //     iff cue-only recall has ≥ ε overlap with the canonical
        //     target — that's the cleanest "did the brain learn the
        //     association" question, untouched by per-epoch
        //     fingerprint drift.
        //
        // Plasticity is silenced for the duration of the readout
        // so the test does not contaminate the next epoch's
        // training. Iter-52: skip when no_plasticity (the call
        // would not change behaviour, but the matching enable_*
        // below would silently turn plasticity on).
        let saved_modulator = brain.regions[1].network.neuromodulator;
        if !no_plasticity {
            brain.regions[1].network.disable_stdp();
            brain.regions[1].network.disable_istdp();
        }
        brain.set_neuromodulator(0.0);

        let dec_t0 = std::time::Instant::now();
        let empty_dg: std::collections::HashMap<String, Vec<u32>> =
            std::collections::HashMap::new();
        let dict = build_vocab_dictionary(&mut brain, &encoder, &r2_e, &vocab, &empty_dg, 0.0);
        let mut top1 = 0usize;
        let mut top3 = 0usize;
        let mut rank_sum = 0.0_f32;
        let mut rr_sum = 0.0_f32;
        let mut active_total: u64 = 0;
        let mut active_n: u32 = 0;
        let mut correct_pair_w_sum = 0.0_f64;
        let mut correct_pair_w_count = 0_usize;
        let mut incorrect_pair_w_sum = 0.0_f64;
        let mut incorrect_pair_w_count = 0_usize;
        for pair in &corpus.pairs {
            let cue_sdr = encoder.encode_word(&pair.cue);
            if cue_sdr.indices.is_empty() {
                continue;
            }
            brain.regions[1].network.reset_state();
            let counts = drive_for_with_counts(&mut brain, &cue_sdr.indices, RECALL_MS, &r2_e);
            active_total += counts.iter().filter(|&&c| c > 0).count() as u64;
            active_n += 1;
            let kwta = top_k_indices(&counts, KWTA_K);
            if kwta.is_empty() {
                continue;
            }
            // Iter-45 dictionary scoring (rank etc.).
            let decoded = dict.decode_top(&kwta, 16);
            let rank = decoded
                .iter()
                .position(|(w, _)| w == &pair.target)
                .map(|p| p + 1);
            if let Some(r) = rank {
                rank_sum += r as f32;
                rr_sum += 1.0 / r as f32;
                if r == 1 {
                    top1 += 1;
                }
                if r <= 3 {
                    top3 += 1;
                }
            } else {
                // Missing — count as last-rank for the average.
                rank_sum += vocab.len() as f32;
            }
            // Iter-46 canonical-target margin: the cue-driven
            // R2 spike count averaged over correct vs incorrect
            // target neurons.
            if teacher_active {
                if let Some(canonical) = target_r2_map.get(&pair.target) {
                    let target_set: BTreeSet<u32> = canonical.iter().copied().collect();
                    let (mut correct_sum, mut correct_n, mut incorrect_sum, mut incorrect_n) =
                        (0u64, 0u32, 0u64, 0u32);
                    for (i, &c) in counts.iter().enumerate() {
                        if !r2_e.contains(&i) {
                            continue;
                        }
                        if target_set.contains(&(i as u32)) {
                            correct_sum += c as u64;
                            correct_n += 1;
                        } else {
                            incorrect_sum += c as u64;
                            incorrect_n += 1;
                        }
                    }
                    if correct_n > 0 {
                        correct_pair_w_sum += correct_sum as f64 / correct_n as f64;
                        correct_pair_w_count += 1;
                    }
                    if incorrect_n > 0 {
                        incorrect_pair_w_sum += incorrect_sum as f64 / incorrect_n as f64;
                        incorrect_pair_w_count += 1;
                    }
                }
            }
        }
        let mut noise_hits = 0usize;
        for pair in &corpus.noise_pairs {
            let (_, distractor_in_top3) =
                evaluate_with_dict(&mut brain, &encoder, &r2_e, &dict, &pair.cue, &pair.target);
            if distractor_in_top3 {
                noise_hits += 1;
            }
        }
        let decoder_micros = dec_t0.elapsed().as_micros();

        // Restore plasticity for the next epoch. Iter-52: skip
        // when no_plasticity — STDP / iSTDP were never enabled,
        // so calling enable_*() here would turn them on
        // mid-run.
        if !no_plasticity {
            brain.regions[1].network.enable_stdp(stdp_params);
            brain.regions[1].network.enable_istdp(istdp_params);
        }
        brain.set_neuromodulator(saved_modulator);

        // Optional iter-46 homeostatic normalisation: re-scale
        // R2 → R2 weights so their L2 norm matches a fixed
        // budget. Stops repeated teacher-forcing from blowing up
        // a few super-weights.
        if cfg.teacher.enabled && cfg.teacher.homeostatic_normalization {
            let net = &mut brain.regions[1].network;
            let mut sumsq: f64 = 0.0;
            for s in &net.synapses {
                sumsq += (s.weight as f64) * (s.weight as f64);
            }
            let norm = sumsq.sqrt() as f32;
            // Target = max of the initial random weight L2 (computed
            // empirically as ≈ √(num_synapses · mean_w²) ≈ 12 — but
            // we don't recompute it; just cap drift via factor < 1).
            let target = 1.5_f32 * norm.min(20.0);
            if norm > target && norm > 0.0 {
                let factor = target / norm;
                for s in net.synapses.iter_mut() {
                    s.weight *= factor;
                }
            }
        }

        let pairs_n = corpus.pairs.len() as f32;
        let noise_n = corpus.noise_pairs.len() as f32;
        let mean_reward = if reward_trials == 0 {
            0.0
        } else {
            total_reward / reward_trials as f32
        };
        // R2 → R2 weight stats — only synapses inside region 1.
        let net = &brain.regions[1].network;
        let mut w_sum = 0.0_f64;
        let mut w_max = 0.0_f32;
        for s in &net.synapses {
            w_sum += s.weight as f64;
            if s.weight > w_max {
                w_max = s.weight;
            }
        }
        let w_mean = if net.synapses.is_empty() {
            0.0
        } else {
            (w_sum / net.synapses.len() as f64) as f32
        };
        let elig_nonzero = net.eligibility.iter().filter(|&&e| e != 0.0).count() as u32;
        let correct_w = if correct_pair_w_count > 0 {
            (correct_pair_w_sum / correct_pair_w_count as f64) as f32
        } else {
            0.0
        };
        let incorrect_w = if incorrect_pair_w_count > 0 {
            (incorrect_pair_w_sum / incorrect_pair_w_count as f64) as f32
        } else {
            0.0
        };
        let pred_top3 = if prediction_top3_total > 0 {
            prediction_top3_hits as f32 / prediction_top3_total as f32
        } else {
            0.0
        };
        let clamp_rate = if clamp_size_total > 0 {
            clamp_hit_total as f32 / clamp_size_total as f32
        } else {
            0.0
        };

        metrics.push(RewardEpochMetrics {
            epoch,
            top1_accuracy: if pairs_n > 0.0 {
                top1 as f32 / pairs_n
            } else {
                0.0
            },
            top3_accuracy: if pairs_n > 0.0 {
                top3 as f32 / pairs_n
            } else {
                0.0
            },
            mean_reward,
            noise_top3_rate: if noise_n > 0.0 {
                noise_hits as f32 / noise_n
            } else {
                0.0
            },
            random_top3_baseline: 3.0 / vocab.len().max(1) as f32,
            mean_rank: if pairs_n > 0.0 {
                rank_sum / pairs_n
            } else {
                f32::NAN
            },
            mrr: if pairs_n > 0.0 { rr_sum / pairs_n } else { 0.0 },
            target_clamp_hit_rate: clamp_rate,
            prediction_top3_before_teacher: pred_top3,
            eligibility_nonzero_count: elig_nonzero,
            r2_recurrent_weight_mean: w_mean,
            r2_recurrent_weight_max: w_max,
            active_r2_units_per_cue: if active_n > 0 {
                active_total as f32 / active_n as f32
            } else {
                0.0
            },
            correct_minus_incorrect_margin: correct_w - incorrect_w,
            decoder_micros,
            // ---- iter-47 sparsity readout ----
            r2_active_pre_teacher_mean: {
                if active_samples.is_empty() {
                    0.0
                } else {
                    active_samples.iter().sum::<u32>() as f32 / active_samples.len() as f32
                }
            },
            r2_active_pre_teacher_p10: {
                let mut s = active_samples.clone();
                s.sort_unstable();
                percentile_u32(&s, 0.10)
            },
            r2_active_pre_teacher_p90: {
                let mut s = active_samples.clone();
                s.sort_unstable();
                percentile_u32(&s, 0.90)
            },
            target_hit_pre_teacher_mean: {
                if target_hit_samples.is_empty() {
                    0.0
                } else {
                    target_hit_samples.iter().sum::<u32>() as f32 / target_hit_samples.len() as f32
                }
            },
            selectivity_index: {
                // Per-trial selectivity, then mean. Each trial's
                // selectivity is target_hit / |target| − non_target_active /
                // (|R2_E| − |target|).
                let target_size = TARGET_R2_K as f32;
                let r2_e_size = r2_e.len() as f32;
                let bg_size = (r2_e_size - target_size).max(1.0);
                if active_samples.is_empty() {
                    0.0
                } else {
                    let mut sum = 0.0_f32;
                    for (i, &active) in active_samples.iter().enumerate() {
                        let hit = target_hit_samples[i] as f32;
                        let non = (active as f32 - hit).max(0.0);
                        sum += hit / target_size - non / bg_size;
                    }
                    sum / active_samples.len() as f32
                }
            },
            // ---- iter-48 cascade-stability ----
            r2_active_pre_teacher_p99: {
                let mut s = active_samples.clone();
                s.sort_unstable();
                percentile_u32(&s, 0.99)
            },
            theta_inh_mean: {
                let (_e, i) = intrinsic_mean_by_kind(&brain);
                i
            },
            theta_exc_mean: {
                let (e, _i) = intrinsic_mean_by_kind(&brain);
                e
            },
        });
    }

    // Iter-52 sanity assertion: when no_plasticity is true, post-
    // training synapse L2 norms must equal the pre-training values
    // bit-for-bit. If they differ by even one ulp, a plasticity
    // path is leaking.
    if let Some(pre) = pre_norms {
        let post = brain_synapse_l2_norms(&brain);
        let identical = pre.iter().zip(post.iter()).all(|(a, b)| a == b);
        eprintln!(
            "[iter-52] post-training L2 norms = {:?} | bit-identical to pre = {}",
            post, identical,
        );
        assert!(
            identical,
            "iter-52 invariant violated: weights changed under --no-plasticity. \
             pre = {:?}, post = {:?}",
            pre, post,
        );
    }

    metrics
}

// ----------------------------------------------------------------------
// Iter-53: decoder-relative Jaccard benchmark.
//
// The post-iter-52 problem: top-k accuracy against canonical-hash
// targets is structurally biased by R1 forward drive (the untrained
// brain hits 0.039 — significantly *below* the 0.094 random baseline
// — because forward projections asymmetrically push the decoder
// toward a small subset of vocab words, regardless of plasticity).
// To answer "did Plastizität *learn* anything", we need a metric
// computed *relative to the same brain's own decoder*, not against
// a canonical hash.
//
// The Jaccard pair gives that, on two orthogonal axes:
//
//   Same-cue  — Jaccard(trial[1], trial[2]) for one cue, averaged
//               over the vocab. High = consistent engram per cue.
//   Cross-cue — Jaccard(matrix[i][1], matrix[j][1]) over all cue
//               pairs `i < j`. Low = each cue is *cue-specific*.
//
// Acceptance is a **difference of differences**: trained same-cue
// must rise *and* trained cross-cue must fall, both versus
// untrained. Either alone is satisfied by trivial pathologies
// (high same-cue + high cross-cue = mode collapse; low cross-cue
// + low same-cue = pure noise).
// ----------------------------------------------------------------------

/// Iter-53 helper: in-place training loop. Mirrors
/// `run_reward_benchmark`'s inner schedule (teacher arm + iter-45
/// fallback) without metrics collection. Used by `run_jaccard_arm`
/// to re-use the existing trial machinery without going through
/// `run_reward_benchmark` (which would consume its own brain).
#[allow(clippy::too_many_arguments)]
fn train_brain_inplace(
    brain: &mut Brain,
    corpus: &RewardCorpus,
    cfg: &RewardConfig,
    target_r2_map: &std::collections::HashMap<String, Vec<u32>>,
    encoder: &TextEncoder,
    r2_e: &BTreeSet<usize>,
    stdp_params: StdpParams,
    initial_istdp: IStdpParams,
    dg_sdr_map: &std::collections::HashMap<String, Vec<u32>>,
) {
    // initial_istdp is only used to satisfy the type of the loop-
    // local `istdp_params`; the per-epoch reassignment below
    // overwrites it on every iteration. The let-binding stays so
    // the variable's type is fixed before the loop body.
    let mut istdp_params;
    let _ = initial_istdp;
    let mut rng = Rng::new(cfg.seed);
    let no_plasticity = cfg.teacher.no_plasticity;
    let teacher_active = cfg.teacher.enabled;

    for epoch in 0..cfg.epochs {
        istdp_params = if cfg.teacher.iter46_baseline {
            istdp_iter46_baseline()
        } else {
            istdp_iter49(&cfg.teacher, epoch)
        };
        if !no_plasticity {
            brain.regions[1].network.enable_istdp(istdp_params);
        }

        let mut schedule: Vec<(RewardPair, bool)> = corpus
            .pairs
            .iter()
            .map(|p| (p.clone(), false))
            .chain(corpus.noise_pairs.iter().map(|p| (p.clone(), true)))
            .collect();
        shuffle(&mut rng, &mut schedule);

        for (pair, is_noise) in &schedule {
            let cue_sdr = encoder.encode_word(&pair.cue);
            let tgt_sdr = encoder.encode_word(&pair.target);

            if teacher_active {
                let canonical = target_r2_map.get(&pair.target).cloned().unwrap_or_default();
                let dg_sdr = dg_sdr_map.get(&pair.cue).cloned().unwrap_or_default();
                for _rep in 0..cfg.reps_per_pair.max(1) {
                    brain.regions[1].network.reset_state();
                    let _ = run_teacher_trial(
                        brain,
                        &cfg.teacher,
                        cfg.use_reward,
                        *is_noise,
                        &cue_sdr.indices,
                        &canonical,
                        r2_e,
                        stdp_params,
                        istdp_params,
                        &dg_sdr,
                        &[],
                    );
                    idle(brain, COOLDOWN_MS);
                }
            } else {
                let mut combined: Vec<u32> = cue_sdr
                    .indices
                    .iter()
                    .chain(tgt_sdr.indices.iter())
                    .copied()
                    .collect();
                combined.sort_unstable();
                combined.dedup();
                for _ in 0..cfg.reps_per_pair.max(1) {
                    brain.regions[1].network.reset_state();
                    drive_for(brain, &cue_sdr.indices, CUE_LEAD_MS);
                    drive_for(brain, &combined, OVERLAP_MS);
                    drive_for(brain, &tgt_sdr.indices, TARGET_TAIL_MS);
                }
                if cfg.use_reward {
                    let reward = if *is_noise {
                        -1.0_f32
                    } else {
                        brain.regions[1].network.reset_state();
                        let counts =
                            drive_for_with_counts(brain, &cue_sdr.indices, RECALL_MS, r2_e);
                        let kwta = top_k_indices(&counts, KWTA_K);
                        let target_kwta = {
                            brain.regions[1].network.reset_state();
                            let cs =
                                drive_for_with_counts(brain, &tgt_sdr.indices, RECALL_MS, r2_e);
                            top_k_indices(&cs, KWTA_K)
                        };
                        let overlap = kwta.iter().filter(|i| target_kwta.contains(i)).count();
                        let ratio = overlap as f32 / target_kwta.len().max(1) as f32;
                        if ratio >= 0.30 {
                            1.0
                        } else if ratio >= 0.15 {
                            0.0
                        } else {
                            -0.5
                        }
                    };
                    if reward != 0.0 {
                        brain.set_neuromodulator(reward);
                        drive_for(brain, &cue_sdr.indices, CONSOLIDATION_MS);
                        brain.set_neuromodulator(0.0);
                    }
                }
            }
        }
    }
}

/// Iter-53 core: collect the 32-cue × 3-trial decoder matrix and
/// compute same-cue + cross-cue Jaccard. Each trial does a *full*
/// `brain.reset_state()` (R1 + R2 + cross-region pending queue +
/// clock + traces + eligibility), unlike the existing dictionary
/// build / evaluate_with_dict path which only resets region 1.
/// Without the cross-region reset, R1 carry-over and queued spikes
/// from trial N contaminate trial N+1 — the iter-53.0 smoke
/// reproduced this with mean Jaccard = 0.667 on a frozen brain.
///
/// Trial 0 is dropped as burn-in: even with full reset, the *very*
/// first trial after `build_vocab_dictionary` sees a brain whose
/// `time` was advanced by 32 dictionary drives, and intrinsic
/// plasticity offsets (if active) accumulated across them. Trial
/// 1 and Trial 2 are the actual measurement.
fn evaluate_jaccard_matrix(
    brain: &mut Brain,
    encoder: &TextEncoder,
    r2_e: &BTreeSet<usize>,
    dict: &EngramDictionary,
    vocab: &[String],
    dg_sdr_map: &std::collections::HashMap<String, Vec<u32>>,
    dg_drive_strength: f32,
) -> JaccardMetrics {
    const N_TRIALS: usize = 3;
    const TOP_K: usize = 3;

    // matrix[cue_idx][trial_idx] = top-3 decoded word list.
    let mut matrix: Vec<Vec<Vec<String>>> = Vec::with_capacity(vocab.len());
    for cue_word in vocab {
        let cue_sdr = encoder.encode_word(cue_word);
        if cue_sdr.indices.is_empty() {
            matrix.push(Vec::new());
            continue;
        }
        let dg_sdr = dg_sdr_map.get(cue_word);
        let mut trials: Vec<Vec<String>> = Vec::with_capacity(N_TRIALS);
        for _trial_i in 0..N_TRIALS {
            // Full reset — R1 + R2 + pending queue + brain.time.
            brain.reset_state();
            let counts = if let Some(dg) = dg_sdr {
                drive_with_dg_counts(
                    brain,
                    &cue_sdr.indices,
                    dg,
                    DRIVE_NA,
                    dg_drive_strength,
                    RECALL_MS,
                    r2_e,
                )
            } else {
                drive_for_with_counts(brain, &cue_sdr.indices, RECALL_MS, r2_e)
            };
            let kwta = top_k_indices(&counts, KWTA_K);
            let decoded = dict.decode_top(&kwta, TOP_K);
            let top: Vec<String> = decoded.iter().map(|(w, _)| w.clone()).collect();
            trials.push(top);
        }
        matrix.push(trials);
    }

    let jaccard = |a: &[String], b: &[String]| -> f32 {
        let sa: BTreeSet<&String> = a.iter().collect();
        let sb: BTreeSet<&String> = b.iter().collect();
        let inter = sa.intersection(&sb).count();
        let uni = sa.union(&sb).count();
        if uni == 0 {
            1.0
        } else {
            inter as f32 / uni as f32
        }
    };

    // Same-cue: post-burn-in trials of the *same* cue.
    let mut same_vals: Vec<f32> = Vec::new();
    for trials in &matrix {
        if trials.len() == N_TRIALS {
            same_vals.push(jaccard(&trials[1], &trials[2]));
        }
    }

    // Cross-cue: post-burn-in trials of *different* cues.
    let mut cross_vals: Vec<f32> = Vec::new();
    for i in 0..matrix.len() {
        if matrix[i].len() < N_TRIALS {
            continue;
        }
        for j in (i + 1)..matrix.len() {
            if matrix[j].len() < N_TRIALS {
                continue;
            }
            cross_vals.push(jaccard(&matrix[i][1], &matrix[j][1]));
        }
    }

    let mean = |v: &[f32]| {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f32>() / v.len() as f32
        }
    };
    let std_dev = |v: &[f32], m: f32| {
        if v.len() < 2 {
            0.0
        } else {
            let var = v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / (v.len() - 1) as f32;
            var.sqrt()
        }
    };

    let same_cue_mean = mean(&same_vals);
    let same_cue_std = std_dev(&same_vals, same_cue_mean);
    let cross_cue_mean = mean(&cross_vals);
    let cross_cue_std = std_dev(&cross_vals, cross_cue_mean);

    JaccardMetrics {
        same_cue_mean,
        same_cue_std,
        cross_cue_mean,
        cross_cue_std,
        n_cues: same_vals.len(),
        n_pairs: cross_vals.len(),
    }
}

/// Iter-58 floor-diagnosis variant of `evaluate_jaccard_matrix`.
/// Identical 32-cue × 3-trial matrix collection (same full
/// `brain.reset_state()` per trial, same trial-1-vs-trial-2
/// post-burn-in protocol), but additionally emits the per-cue-
/// pair Jaccard list with cue names + decoded top-3 sets so the
/// caller can inspect the *shape* of the cross-cue distribution
/// (concentrated on a few SDR-collision pairs vs uniform across
/// the vocab).
fn evaluate_jaccard_matrix_with_pairs(
    brain: &mut Brain,
    encoder: &TextEncoder,
    r2_e: &BTreeSet<usize>,
    dict: &EngramDictionary,
    vocab: &[String],
    dg_sdr_map: &std::collections::HashMap<String, Vec<u32>>,
    dg_drive_strength: f32,
) -> (JaccardMetrics, Vec<JaccardPairSample>) {
    const N_TRIALS: usize = 3;
    const TOP_K: usize = 3;

    let mut matrix: Vec<Vec<Vec<String>>> = Vec::with_capacity(vocab.len());
    for cue_word in vocab {
        let cue_sdr = encoder.encode_word(cue_word);
        if cue_sdr.indices.is_empty() {
            matrix.push(Vec::new());
            continue;
        }
        let dg_sdr = dg_sdr_map.get(cue_word);
        let mut trials: Vec<Vec<String>> = Vec::with_capacity(N_TRIALS);
        for _trial_i in 0..N_TRIALS {
            brain.reset_state();
            let counts = if let Some(dg) = dg_sdr {
                drive_with_dg_counts(
                    brain,
                    &cue_sdr.indices,
                    dg,
                    DRIVE_NA,
                    dg_drive_strength,
                    RECALL_MS,
                    r2_e,
                )
            } else {
                drive_for_with_counts(brain, &cue_sdr.indices, RECALL_MS, r2_e)
            };
            let kwta = top_k_indices(&counts, KWTA_K);
            let decoded = dict.decode_top(&kwta, TOP_K);
            let top: Vec<String> = decoded.iter().map(|(w, _)| w.clone()).collect();
            trials.push(top);
        }
        matrix.push(trials);
    }

    let jaccard = |a: &[String], b: &[String]| -> f32 {
        let sa: BTreeSet<&String> = a.iter().collect();
        let sb: BTreeSet<&String> = b.iter().collect();
        let inter = sa.intersection(&sb).count();
        let uni = sa.union(&sb).count();
        if uni == 0 {
            1.0
        } else {
            inter as f32 / uni as f32
        }
    };

    let mut same_vals: Vec<f32> = Vec::new();
    for trials in &matrix {
        if trials.len() == N_TRIALS {
            same_vals.push(jaccard(&trials[1], &trials[2]));
        }
    }

    let mut cross_vals: Vec<f32> = Vec::new();
    let mut per_pair: Vec<JaccardPairSample> = Vec::new();
    for i in 0..matrix.len() {
        if matrix[i].len() < N_TRIALS {
            continue;
        }
        for j in (i + 1)..matrix.len() {
            if matrix[j].len() < N_TRIALS {
                continue;
            }
            let j_val = jaccard(&matrix[i][1], &matrix[j][1]);
            cross_vals.push(j_val);
            per_pair.push(JaccardPairSample {
                cue_a: vocab[i].clone(),
                cue_b: vocab[j].clone(),
                jaccard: j_val,
                top_a: matrix[i][1].clone(),
                top_b: matrix[j][1].clone(),
            });
        }
    }

    let mean = |v: &[f32]| {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f32>() / v.len() as f32
        }
    };
    let std_dev = |v: &[f32], m: f32| {
        if v.len() < 2 {
            0.0
        } else {
            let var = v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / (v.len() - 1) as f32;
            var.sqrt()
        }
    };

    let same_cue_mean = mean(&same_vals);
    let same_cue_std = std_dev(&same_vals, same_cue_mean);
    let cross_cue_mean = mean(&cross_vals);
    let cross_cue_std = std_dev(&cross_vals, cross_cue_mean);

    let metrics = JaccardMetrics {
        same_cue_mean,
        same_cue_std,
        cross_cue_mean,
        cross_cue_std,
        n_cues: same_vals.len(),
        n_pairs: cross_vals.len(),
    };
    (metrics, per_pair)
}

// =====================================================================
// Iter-63 plumbing-fix shared helpers.
//
// Single source of truth for benchmark-brain construction and for
// the plasticity-disable stack. iter-63 caught the kind of silent-
// wiring-gap bug iter-52 was designed to prevent: brain-build code
// was duplicated across run_jaccard_arm and run_jaccard_floor_
// diagnosis, and a third caller (run_target_overlap_arm) silently
// went through run_reward_benchmark which ignores the iter-54 /
// iter-60 flags entirely. This produced numbers that *looked* like
// the iter-63 config but actually came from a vanilla random-wired
// non-DG brain.
//
// Going forward, every new runner that needs a "benchmark brain"
// must call `build_benchmark_brain` and `disable_all_plasticity`
// instead of building one inline. The legacy `run_reward_benchmark`
// path is intentionally NOT refactored — its numerics are the iter-46
// / iter-51 baselines that the iter-63 positive control verified
// minutes before this refactor, and breaking that contract would
// invalidate the calibration.
// =====================================================================

/// Output of `build_benchmark_brain`. Caller-friendly metadata keeps
/// the helper pure (no `eprintln!`) so the legacy log strings — which
/// differ per caller (`[iter-54 …]` for jaccard arm, `[iter-58 floor]`
/// for floor diagnosis) — can stay at the call site verbatim.
struct BrainBuild {
    brain: Brain,
    encoder: TextEncoder,
    r2_e: BTreeSet<usize>,
    target_r2_map: std::collections::HashMap<String, Vec<u32>>,
    dg_sdr_map: std::collections::HashMap<String, Vec<u32>>,
    /// `effective_r2_n(&cfg.teacher)` — the R2 size used for this
    /// build. Returned so callers can log it without re-computing.
    r2_n_used: usize,
    /// `Some(block_size)` when `decorrelated_init` was used; `None`
    /// otherwise. `block_size = r2_e.len() / vocab.len()`.
    decorrelated_block_size: Option<usize>,
    /// Iter-66: `Some(set)` when `cfg.teacher.c1.enabled` is true,
    /// containing the appended C1 cell indices inside R2's
    /// network (range `[r2_n_used, r2_n_used + cfg.c1.size)`).
    /// `None` when the C1 readout is disabled — preserves the
    /// iter-46/63 numerics bit-identically for every callsite that
    /// ignores `c1_e`.
    c1_e: Option<BTreeSet<usize>>,
    /// Iter-66: per-target canonical C1 SDR used by the teacher
    /// schedule to clamp C1 cells in the canonical-target pattern
    /// during the encoding phase. Indices are absolute into R2's
    /// network (shifted by `c1_start`). Empty map when
    /// `c1.enabled = false`.
    target_c1_map: std::collections::HashMap<String, Vec<u32>>,
    /// Iter-66 step 7.5 diagnostic: index of the first R2-E → C1
    /// synapse in R2's `network.synapses` vec. Synapses
    /// `[c1_synapse_start..]` are the new R2-E → C1 edges; the
    /// pre-suffix synapses are the original R2-R2 connectivity.
    /// `None` when `c1.enabled = false`.
    c1_synapse_start: Option<usize>,
}

/// Single source of truth for benchmark-brain construction. Replaces
/// ~70 lines of duplicated brain-build code in `run_jaccard_arm` and
/// `run_jaccard_floor_diagnosis`. Adding a new architectural axis
/// (e.g. CA3/CA1 split, perforant-path re-introduction) means
/// touching this function only.
///
/// Behaviour mirrors the existing inline code exactly — RNG seeds,
/// salt constants, region order, wiring sequence — so the iter-63
/// snapshot tests
/// (`jaccard_bench_snapshot_{vanilla,decorrelated,decorrelated_dg_recall}`)
/// still pass bit-identically.
fn build_benchmark_brain(corpus: &RewardCorpus, cfg: &RewardConfig) -> BrainBuild {
    let (inter_weight, inh_frac) = if cfg.teacher.iter46_baseline {
        (2.0_f32, 0.20_f32)
    } else {
        (INTER_WEIGHT, R2_INH_FRAC)
    };
    let r2_n_used = effective_r2_n(&cfg.teacher);
    let r2_p_connect_used = effective_r2_p_connect(&cfg.teacher);
    let encoder = TextEncoder::with_stopwords(ENC_N, ENC_K, std::iter::empty::<&str>());
    let effective_inter_weight = if cfg.teacher.dg.enabled {
        inter_weight * cfg.teacher.dg.direct_r1r2_weight_scale
    } else {
        inter_weight
    };

    let (mut brain, decorrelated_block_size) = if cfg.teacher.decorrelated_init {
        let mut b = Brain::new(DT);
        b.add_region(build_input_region());
        b.add_region(build_memory_region(
            cfg.seed.wrapping_add(1),
            inh_frac,
            r2_n_used,
            r2_p_connect_used,
        ));
        let vocab_vec: Vec<String> = corpus.vocab.iter().cloned().collect();
        let _blocks = wire_forward_decorrelated(
            &mut b,
            &encoder,
            &vocab_vec,
            cfg.seed.wrapping_add(2),
            effective_inter_weight,
        );
        assert_decorrelated_disjoint(&b, &encoder, &vocab_vec);
        let r2_e_size = r2_e_set(&b).len();
        let block_size = r2_e_size / corpus.vocab.len().max(1);
        (b, Some(block_size))
    } else {
        (
            fresh_brain_with(
                cfg.seed,
                effective_inter_weight,
                inh_frac,
                r2_n_used,
                r2_p_connect_used,
            ),
            None,
        )
    };

    if cfg.teacher.dg.enabled {
        brain.add_region(build_dg_region(cfg.teacher.dg.size as usize));
        wire_dg_to_r2(&mut brain, &cfg.teacher.dg, cfg.seed.wrapping_add(0xD9));
    }

    // R2-E set — captured *before* the iter-66 C1 cells are
    // appended so C1 (also excitatory) does not leak into R2-E.
    let r2_e = r2_e_set(&brain);

    // Iter-66 (M1): append C1 cells + R2-E → C1 plastic synapses.
    // No-op when `c1.enabled = false` ⇒ bit-identical to iter-65
    // for every existing call site. The seed salt
    // `0xC1A1_5EE9_..` is distinct from DG's `0xD9..` so two
    // distinct architectures at the same `cfg.seed` get distinct
    // wiring RNG streams.
    let (c1_e, target_c1_map, c1_synapse_start) = if cfg.teacher.c1.enabled {
        // Iter-66 step 7.5 diagnostic: capture the boundary index
        // *before* appending C1, so the post-suffix synapse range
        // is identifiable as "R2-E → C1 only".
        let r2_r2_synapse_count = brain.regions[1].network.synapses.len();
        let (c1_start, c1_end) = append_c1_to_r2(
            &mut brain,
            &cfg.teacher.c1,
            &r2_e,
            r2_n_used,
            cfg.seed.wrapping_add(0xC1A1_5EE9),
        );
        let c1_size = c1_end - c1_start;
        let salt_c1 = cfg.seed ^ 0xC1A1_BABE_F00D_CA1Eu64;
        let map: std::collections::HashMap<String, Vec<u32>> = corpus
            .vocab
            .iter()
            .map(|w| {
                (
                    w.clone(),
                    canonical_target_c1_sdr(
                        w,
                        c1_start,
                        c1_size,
                        cfg.teacher.c1.sparsity_k as usize,
                        salt_c1,
                    ),
                )
            })
            .collect();
        (
            Some(c1_set((c1_start, c1_end))),
            map,
            Some(r2_r2_synapse_count),
        )
    } else {
        (None, std::collections::HashMap::new(), None)
    };

    let target_r2_map: std::collections::HashMap<String, Vec<u32>> = {
        let pool = r2_e_pool(&r2_e);
        let salt = cfg.seed ^ 0xCAFE_F00D_DEAD_BEEFu64;
        corpus
            .vocab
            .iter()
            .map(|w| {
                (
                    w.clone(),
                    canonical_target_r2_sdr(w, &pool, TARGET_R2_K, salt),
                )
            })
            .collect()
    };

    let dg_sdr_map: std::collections::HashMap<String, Vec<u32>> = if cfg.teacher.dg.enabled {
        let salt = cfg.seed ^ 0xDEAD_BEEF_F00D_BABEu64;
        corpus
            .vocab
            .iter()
            .map(|w| {
                (
                    w.clone(),
                    dg_sdr_for_cue(
                        w,
                        cfg.teacher.dg.size as usize,
                        cfg.teacher.dg.k as usize,
                        salt,
                    ),
                )
            })
            .collect()
    } else {
        std::collections::HashMap::new()
    };

    BrainBuild {
        brain,
        encoder,
        r2_e,
        target_r2_map,
        dg_sdr_map,
        r2_n_used,
        decorrelated_block_size,
        c1_e,
        target_c1_map,
        c1_synapse_start,
    }
}

/// Single source of truth for plasticity gating. Disables every
/// plasticity rule that can mutate weights — STDP, iSTDP,
/// homeostasis, intrinsic plasticity, reward learning,
/// metaplasticity, heterosynaptic scaling, structural plasticity —
/// on the given region's `Network`. Adding a new plasticity
/// mechanism in `snn-core` means touching this function only.
///
/// Iter-62 introduced this exact stack inline in `run_jaccard_arm`'s
/// recall-mode branch. Iter-63 lifts it to a top-level helper so
/// every runner that needs a "frozen weights" guarantee can call it
/// without forgetting a rule.
fn disable_all_plasticity(brain: &mut Brain, region_idx: usize) {
    let net = &mut brain.regions[region_idx].network;
    net.disable_stdp();
    net.disable_istdp();
    net.disable_homeostasis();
    net.disable_intrinsic_plasticity();
    net.disable_reward_learning();
    net.disable_metaplasticity();
    net.disable_heterosynaptic();
    net.disable_structural();
}

/// Per-region L2 norms of synapse weights — companion to the iter-52
/// invariant check. Type alias so callers can pass these around
/// without committing to a concrete container.
type WeightSnapshot = Vec<f64>;

/// Capture the current weight state of every region as L2 norms.
/// Trivial wrapper over `brain_synapse_l2_norms`; the named alias
/// makes the iter-52 invariant intent explicit at call sites.
fn snapshot_weights(brain: &Brain) -> WeightSnapshot {
    brain_synapse_l2_norms(brain)
}

/// Iter-52 / iter-62 invariant: bit-identical weight L2 norms
/// pre/post a "frozen weights" phase. Caller passes a `context`
/// string (e.g. "iter-63 untrained calibration") so the panic
/// message identifies which guarantee was violated.
fn assert_no_weight_drift(pre: &WeightSnapshot, post: &WeightSnapshot, context: &str) {
    assert_eq!(
        pre.len(),
        post.len(),
        "{context}: weight snapshot region count changed ({} → {})",
        pre.len(),
        post.len(),
    );
    let identical = pre.iter().zip(post.iter()).all(|(a, b)| a == b);
    assert!(
        identical,
        "{context}: weight invariant violated (pre/post not bit-identical). \
         pre={pre:?} post={post:?}"
    );
}

/// Iter-53 single-arm runner. Builds the brain, optionally trains
/// it (a no-op when `cfg.epochs == 0` or `cfg.teacher.no_plasticity`),
/// freezes plasticity for the eval phase, builds the vocabulary
/// fingerprint dictionary against the post-training brain, then
/// computes the Jaccard matrix. Asserts the iter-52 invariant on
/// `no_plasticity` arms: pre-run L2 norms must equal post-eval L2
/// norms bit-for-bit.
fn run_jaccard_arm(
    corpus: &RewardCorpus,
    cfg: &RewardConfig,
    arm: &'static str,
) -> JaccardArmResult {
    let BrainBuild {
        mut brain,
        encoder,
        r2_e,
        target_r2_map,
        dg_sdr_map,
        r2_n_used,
        decorrelated_block_size,
        c1_e: _,
        target_c1_map: _,
        c1_synapse_start: _,
    } = build_benchmark_brain(corpus, cfg);

    if let Some(block_size) = decorrelated_block_size {
        eprintln!(
            "[iter-54 {arm}] seed={} decorrelated init: vocab={} R2_N={} R2-E={} block_size={} (disjoint invariant ✓)",
            cfg.seed,
            corpus.vocab.len(),
            r2_n_used,
            r2_e.len(),
            block_size,
        );
    }
    if cfg.teacher.dg.enabled {
        eprintln!(
            "[iter-60 {arm}] seed={} DG bridge: dg_size={} dg_k={} dg_to_r2_fanout={} dg_to_r2_weight={:.2} direct_r1r2_scale={:.2}",
            cfg.seed,
            cfg.teacher.dg.size,
            cfg.teacher.dg.k,
            cfg.teacher.dg.to_r2_fanout,
            cfg.teacher.dg.to_r2_weight,
            cfg.teacher.dg.direct_r1r2_weight_scale,
        );
    }

    let stdp_params = stdp();
    let initial_istdp = if cfg.teacher.iter46_baseline {
        istdp_iter46_baseline()
    } else {
        istdp_iter49(&cfg.teacher, 0)
    };

    let no_plasticity = cfg.teacher.no_plasticity;
    if !no_plasticity {
        brain.regions[1].network.enable_stdp(stdp_params);
        brain.regions[1].network.enable_istdp(initial_istdp);
        brain.regions[1].network.enable_homeostasis(homeostasis());
    }
    if !cfg.teacher.iter46_baseline && !no_plasticity {
        brain.regions[1]
            .network
            .enable_intrinsic_plasticity(intrinsic());
    }
    if cfg.use_reward && !no_plasticity {
        brain.regions[1]
            .network
            .enable_reward_learning(reward_params());
    }

    let pre_l2 = snapshot_weights(&brain);

    train_brain_inplace(
        &mut brain,
        corpus,
        cfg,
        &target_r2_map,
        &encoder,
        &r2_e,
        stdp_params,
        initial_istdp,
        &dg_sdr_map,
    );

    // Iter-62 recall-mode: when --plasticity-off-during-eval is set
    // on a trained arm, disable every plasticity rule before the
    // eval phase. Training stays unchanged; only the dictionary
    // build + jaccard matrix collection run on a frozen-weight
    // brain. The post-eval L2 invariant below catches any plasticity
    // path that escaped the gate. Iter-63 lifted the disable stack
    // into `disable_all_plasticity` so a missing rule cannot leak
    // through.
    let recall_mode_active = cfg.teacher.recall_mode_eval && !no_plasticity;
    if recall_mode_active {
        disable_all_plasticity(&mut brain, 1);
        eprintln!(
            "[iter-62 {arm}] seed={} recall-mode: every plasticity rule disabled before eval (STDP / iSTDP / homeostasis / intrinsic / reward / metaplasticity / heterosynaptic / structural)",
            cfg.seed,
        );
    }

    let l2_pre_eval = snapshot_weights(&brain);

    let vocab: Vec<String> = corpus.vocab.iter().cloned().collect();
    let dict = build_vocab_dictionary(
        &mut brain,
        &encoder,
        &r2_e,
        &vocab,
        &dg_sdr_map,
        cfg.teacher.dg.drive_strength,
    );
    let jaccard = evaluate_jaccard_matrix(
        &mut brain,
        &encoder,
        &r2_e,
        &dict,
        &vocab,
        &dg_sdr_map,
        cfg.teacher.dg.drive_strength,
    );

    let l2_post_eval = snapshot_weights(&brain);

    if no_plasticity {
        // Untrained arm: plasticity was never enabled; eval must be
        // a pure read. Pre-train and post-eval L2 must match
        // bit-for-bit (the iter-52 invariant carried forward).
        assert_no_weight_drift(
            &pre_l2,
            &l2_post_eval,
            &format!(
                "iter-53: --no-plasticity arm changed weights (seed={}, arm={arm})",
                cfg.seed
            ),
        );
    } else if recall_mode_active {
        // Iter-62 recall-mode invariant: with every plasticity rule
        // disabled before eval, the dictionary build + jaccard
        // matrix must NOT change synapse weights. Pre-eval and
        // post-eval L2 norms must match bit-for-bit. Catches any
        // plasticity path the disable_* calls missed.
        assert_no_weight_drift(
            &l2_pre_eval,
            &l2_post_eval,
            &format!(
                "iter-62 recall-mode invariant violated: weights changed during eval (seed={}, arm={arm})",
                cfg.seed
            ),
        );
        eprintln!(
            "[iter-62 trained recall-mode] seed={} pre={l2_pre_eval:?} post={l2_post_eval:?} (bit-identical ✓)",
            cfg.seed,
        );
    } else {
        // Trained arm: log eval-phase L2 drift so the magnitude is
        // visible. Drift > 0 is expected; drift > training drift
        // would be a configuration smell.
        let drift_l2: Vec<f64> = l2_pre_eval
            .iter()
            .zip(l2_post_eval.iter())
            .map(|(a, b)| (b - a).abs())
            .collect();
        eprintln!(
            "[iter-53 trained eval-drift] seed={} pre={l2_pre_eval:?} post={l2_post_eval:?} |Δ|={drift_l2:?}",
            cfg.seed,
        );
    }

    eprintln!(
        "[iter-53 {arm}] seed={} same={:.3}±{:.3} cross={:.3}±{:.3} (n_cues={}, n_pairs={})",
        cfg.seed,
        jaccard.same_cue_mean,
        jaccard.same_cue_std,
        jaccard.cross_cue_mean,
        jaccard.cross_cue_std,
        jaccard.n_cues,
        jaccard.n_pairs,
    );

    JaccardArmResult {
        seed: cfg.seed,
        arm,
        jaccard,
    }
}

/// Iter-53 public entry point. Runs both arms (untrained + trained)
/// for each seed in `seeds` and returns the aggregated sweep.
///
/// The untrained arm forces `epochs = 0`, `use_reward = false`,
/// and `teacher.no_plasticity = true`. After its eval, asserts
/// `same_cue_mean == 1.0` exactly: with no plasticity AND a full
/// `brain.reset_state()` between trials, identical input must
/// produce identical output. If this assertion fails, either
/// `Network::reset_state` is incomplete or there is a hidden
/// source of trial-to-trial variance — the metric is not
/// trustworthy until the assertion holds.
///
/// The trained arm uses `cfg` as given (including `epochs`,
/// `teacher`, `use_reward`).
pub fn run_jaccard_bench(
    corpus: &RewardCorpus,
    cfg: &RewardConfig,
    seeds: &[u64],
) -> JaccardSweepResult {
    let mut sweep = JaccardSweepResult::default();
    for &seed in seeds {
        let mut cfg_seeded = *cfg;
        cfg_seeded.seed = seed;

        // Untrained: zero epochs, no plasticity, no reward.
        let mut cfg_untrained = cfg_seeded;
        cfg_untrained.epochs = 0;
        cfg_untrained.use_reward = false;
        cfg_untrained.teacher.no_plasticity = true;
        let untrained_arm = run_jaccard_arm(corpus, &cfg_untrained, "untrained");
        assert!(
            (untrained_arm.jaccard.same_cue_mean - 1.0).abs() < 1e-6,
            "iter-53 state-reset assertion FAILED: untrained arm seed={seed} \
             produced same_cue_mean={} std={} (expected exactly 1.0). \
             brain.reset_state() is incomplete or some hidden source of \
             trial-to-trial variance is leaking through.",
            untrained_arm.jaccard.same_cue_mean,
            untrained_arm.jaccard.same_cue_std,
        );
        sweep.untrained.push(untrained_arm);

        // Trained: cfg as given.
        let trained_arm = run_jaccard_arm(corpus, &cfg_seeded, "trained");
        sweep.trained.push(trained_arm);
    }
    sweep
}

/// Iter-58 floor-diagnosis runner. Builds the *trained* arm at the
/// passed config (mirroring `run_jaccard_arm`'s setup verbatim),
/// then uses the per-pair-emitting evaluator to return one
/// [`JaccardFloorReport`] per seed. The aggregate diagnostic
/// (cross-seed averaged distribution, top-N high-overlap pairs,
/// per-cue frequency) is computed by the caller from the per-seed
/// reports — keeps this function side-effect-light and lets the
/// CLI render whichever cuts of the data Bekos's spec asks for.
///
/// `cfg.epochs == 0` would degenerate to the untrained arm, which
/// is *not* the floor-diagnosis target — the function panics in
/// that case so a misconfigured CLI invocation fails loud.
pub fn run_jaccard_floor_diagnosis(
    corpus: &RewardCorpus,
    cfg: &RewardConfig,
    seeds: &[u64],
) -> Vec<JaccardFloorReport> {
    assert!(
        cfg.epochs > 0 && !cfg.teacher.no_plasticity,
        "iter-58 floor diagnosis requires a trained arm (epochs > 0 \
         and not no_plasticity)",
    );
    let mut reports: Vec<JaccardFloorReport> = Vec::with_capacity(seeds.len());
    for &seed in seeds {
        let mut cfg_seeded = *cfg;
        cfg_seeded.seed = seed;

        let BrainBuild {
            mut brain,
            encoder,
            r2_e,
            target_r2_map,
            dg_sdr_map,
            r2_n_used,
            decorrelated_block_size,
            c1_e: _,
            target_c1_map: _,
            c1_synapse_start: _,
        } = build_benchmark_brain(corpus, &cfg_seeded);

        if let Some(block_size) = decorrelated_block_size {
            eprintln!(
                "[iter-58 floor] seed={} decorrelated init: vocab={} R2_N={} R2-E={} block_size={} (disjoint invariant ✓)",
                cfg_seeded.seed,
                corpus.vocab.len(),
                r2_n_used,
                r2_e.len(),
                block_size,
            );
        }

        let stdp_params = stdp();
        let initial_istdp = if cfg_seeded.teacher.iter46_baseline {
            istdp_iter46_baseline()
        } else {
            istdp_iter49(&cfg_seeded.teacher, 0)
        };
        brain.regions[1].network.enable_stdp(stdp_params);
        brain.regions[1].network.enable_istdp(initial_istdp);
        brain.regions[1].network.enable_homeostasis(homeostasis());
        if !cfg_seeded.teacher.iter46_baseline {
            brain.regions[1]
                .network
                .enable_intrinsic_plasticity(intrinsic());
        }
        if cfg_seeded.use_reward {
            brain.regions[1]
                .network
                .enable_reward_learning(reward_params());
        }

        train_brain_inplace(
            &mut brain,
            corpus,
            &cfg_seeded,
            &target_r2_map,
            &encoder,
            &r2_e,
            stdp_params,
            initial_istdp,
            &dg_sdr_map,
        );

        // Iter-62 recall-mode: same protocol as run_jaccard_arm —
        // freeze every plasticity rule before the eval phase so the
        // floor-diagnosis numbers come from the post-training weight
        // state, not from a continuation of training during recall.
        // Iter-63: lifted to `disable_all_plasticity` helper.
        let recall_mode_active = cfg_seeded.teacher.recall_mode_eval;
        let l2_pre_eval = snapshot_weights(&brain);
        if recall_mode_active {
            disable_all_plasticity(&mut brain, 1);
            eprintln!(
                "[iter-62 floor] seed={} recall-mode: every plasticity rule disabled before eval",
                cfg_seeded.seed,
            );
        }

        let vocab: Vec<String> = corpus.vocab.iter().cloned().collect();
        let dict = build_vocab_dictionary(
            &mut brain,
            &encoder,
            &r2_e,
            &vocab,
            &dg_sdr_map,
            cfg_seeded.teacher.dg.drive_strength,
        );
        let (metrics, per_pair) = evaluate_jaccard_matrix_with_pairs(
            &mut brain,
            &encoder,
            &r2_e,
            &dict,
            &vocab,
            &dg_sdr_map,
            cfg_seeded.teacher.dg.drive_strength,
        );

        if recall_mode_active {
            let l2_post_eval = snapshot_weights(&brain);
            assert_no_weight_drift(
                &l2_pre_eval,
                &l2_post_eval,
                &format!(
                    "iter-62 recall-mode invariant violated in floor diagnosis (seed={})",
                    cfg_seeded.seed
                ),
            );
        }

        eprintln!(
            "[iter-58 floor] seed={} same={:.3}±{:.3} cross={:.3}±{:.3} (n_cues={}, n_pairs={})",
            cfg_seeded.seed,
            metrics.same_cue_mean,
            metrics.same_cue_std,
            metrics.cross_cue_mean,
            metrics.cross_cue_std,
            metrics.n_cues,
            metrics.n_pairs,
        );

        reports.push(JaccardFloorReport {
            seed: cfg_seeded.seed,
            n_cues: metrics.n_cues,
            n_pairs: metrics.n_pairs,
            same_cue_mean: metrics.same_cue_mean,
            cross_cue_mean: metrics.cross_cue_mean,
            per_pair,
        });
    }
    reports
}

/// Iter-58 floor-diagnosis renderer. Takes a slice of per-seed
/// reports and emits a Markdown report covering the three cuts
/// Bekos's spec asks for:
/// 1. **Distribution stats** of the cross-seed averaged per-pair
///    Jaccard (min / p25 / median / p75 / p90 / p95 / max).
/// 2. **Top-N high-overlap pairs** (default N = 10).
/// 3. **Per-cue frequency in high-overlap pairs**: how many of
///    each cue's `vocab − 1` partners exceed the threshold.
pub fn render_jaccard_floor_diagnosis(
    reports: &[JaccardFloorReport],
    threshold: f32,
    top_n: usize,
) -> String {
    use std::collections::BTreeMap;
    use std::fmt::Write;

    let mut s = String::new();
    s.push_str("### Iter-58: Jaccard floor diagnosis\n\n");
    if reports.is_empty() {
        s.push_str("_no reports — empty seeds list?_\n");
        return s;
    }

    // Group per-pair samples by (cue_a, cue_b) and average across seeds.
    type PairKey = (String, String);
    let mut sums: BTreeMap<PairKey, (f32, u32)> = BTreeMap::new();
    for r in reports {
        for sample in &r.per_pair {
            let key = (sample.cue_a.clone(), sample.cue_b.clone());
            let entry = sums.entry(key).or_insert((0.0, 0));
            entry.0 += sample.jaccard;
            entry.1 += 1;
        }
    }
    let averaged: Vec<(PairKey, f32)> = sums
        .into_iter()
        .map(|(k, (sum, n))| (k, sum / n as f32))
        .collect();
    let n_pairs = averaged.len();

    let _ = writeln!(
        s,
        "_n_seeds = {}, n_pairs (averaged) = {}, threshold for high-overlap = {:.2}_\n",
        reports.len(),
        n_pairs,
        threshold,
    );

    let _ = writeln!(s, "**Per-seed aggregate:**\n");
    let _ = writeln!(
        s,
        "| Seed | same_cue_mean | cross_cue_mean | n_cues | n_pairs |"
    );
    let _ = writeln!(s, "| ---: | ---: | ---: | ---: | ---: |");
    for r in reports {
        let _ = writeln!(
            s,
            "| {} | {:.3} | {:.3} | {} | {} |",
            r.seed, r.same_cue_mean, r.cross_cue_mean, r.n_cues, r.n_pairs,
        );
    }
    s.push('\n');

    // -- (1) Distribution of cross-seed-averaged per-pair Jaccards.
    let mut sorted: Vec<f32> = averaged.iter().map(|(_, j)| *j).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pick = |frac: f32| -> f32 {
        if sorted.is_empty() {
            return 0.0;
        }
        let idx = ((sorted.len() as f32 - 1.0) * frac).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    };
    let _ = writeln!(s, "**Cross-seed averaged per-pair distribution:**\n");
    let _ = writeln!(s, "| min | p25 | median | p75 | p90 | p95 | max |",);
    let _ = writeln!(s, "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    let _ = writeln!(
        s,
        "| {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
        pick(0.0),
        pick(0.25),
        pick(0.50),
        pick(0.75),
        pick(0.90),
        pick(0.95),
        pick(1.0),
    );
    s.push('\n');

    // -- (2) Top-N high-overlap pairs (cross-seed averaged).
    let mut top: Vec<&(PairKey, f32)> = averaged.iter().collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_n_actual = top_n.min(top.len());
    let _ = writeln!(
        s,
        "**Top-{top_n_actual} high-overlap pairs (averaged across seeds):**\n"
    );
    let _ = writeln!(s, "| Rank | cue_a | cue_b | mean Jaccard |");
    let _ = writeln!(s, "| ---: | :--- | :--- | ---: |");
    for (rank, ((a, b), j)) in top.iter().take(top_n_actual).enumerate() {
        let _ = writeln!(s, "| {} | {} | {} | {:.3} |", rank + 1, a, b, j);
    }
    s.push('\n');

    // -- (3) Per-cue frequency in pairs above threshold.
    let mut high_count: BTreeMap<String, u32> = BTreeMap::new();
    for ((a, b), j) in averaged.iter() {
        if *j >= threshold {
            *high_count.entry(a.clone()).or_insert(0) += 1;
            *high_count.entry(b.clone()).or_insert(0) += 1;
        }
    }
    if high_count.is_empty() {
        let _ = writeln!(
            s,
            "**Cue frequency in pairs ≥ {threshold:.2}:** no pairs cross the threshold.\n",
        );
    } else {
        let mut counts: Vec<(String, u32)> = high_count.into_iter().collect();
        counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let _ = writeln!(
            s,
            "**Cue frequency in pairs ≥ {threshold:.2} (out of vocab−1 = {} possible partners):**\n",
            reports[0].n_cues.saturating_sub(1).max(1),
        );
        let _ = writeln!(s, "| Cue | high-overlap partners |");
        let _ = writeln!(s, "| :--- | ---: |");
        let show = counts.len().min(20);
        for (cue, n) in counts.into_iter().take(show) {
            let _ = writeln!(s, "| {cue} | {n} |");
        }
        s.push('\n');
    }

    s
}

/// Render an iter-53 jaccard sweep as a Markdown report. Per-seed
/// rows + an aggregate panel that surfaces the difference of
/// differences, which is the single number that decides whether
/// engrams are cue-specific (positive Δ-of-Δ) or whether trained
/// gains are mode collapse (negative or zero Δ-of-Δ).
pub fn render_jaccard_sweep(sweep: &JaccardSweepResult) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    s.push_str("### Iter-53: Decoder-relative Jaccard sweep\n\n");
    s.push_str("| Seed | Arm | Same-Cue | Cross-Cue | n_cues | n_pairs |\n");
    s.push_str("| ---: | :--- | ---: | ---: | ---: | ---: |\n");
    let n = sweep.untrained.len().max(sweep.trained.len());
    for i in 0..n {
        for (label, arms) in [("untrained", &sweep.untrained), ("trained", &sweep.trained)] {
            if let Some(r) = arms.get(i) {
                let _ = writeln!(
                    s,
                    "| {} | {label} | {:.3}±{:.3} | {:.3}±{:.3} | {} | {} |",
                    r.seed,
                    r.jaccard.same_cue_mean,
                    r.jaccard.same_cue_std,
                    r.jaccard.cross_cue_mean,
                    r.jaccard.cross_cue_std,
                    r.jaccard.n_cues,
                    r.jaccard.n_pairs,
                );
            }
        }
    }

    let mean_of = |v: &[f32]| -> f32 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f32>() / v.len() as f32
        }
    };
    let std_of = |v: &[f32], m: f32| -> f32 {
        if v.len() < 2 {
            0.0
        } else {
            let var = v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / (v.len() - 1) as f32;
            var.sqrt()
        }
    };

    let us: Vec<f32> = sweep
        .untrained
        .iter()
        .map(|r| r.jaccard.same_cue_mean)
        .collect();
    let uc: Vec<f32> = sweep
        .untrained
        .iter()
        .map(|r| r.jaccard.cross_cue_mean)
        .collect();
    let ts: Vec<f32> = sweep
        .trained
        .iter()
        .map(|r| r.jaccard.same_cue_mean)
        .collect();
    let tc: Vec<f32> = sweep
        .trained
        .iter()
        .map(|r| r.jaccard.cross_cue_mean)
        .collect();

    let us_m = mean_of(&us);
    let us_s = std_of(&us, us_m);
    let uc_m = mean_of(&uc);
    let uc_s = std_of(&uc, uc_m);
    let ts_m = mean_of(&ts);
    let ts_s = std_of(&ts, ts_m);
    let tc_m = mean_of(&tc);
    let tc_s = std_of(&tc, tc_m);

    let _ = writeln!(s, "\n**Aggregate (n={} seeds):**\n", sweep.untrained.len(),);
    let _ = writeln!(
        s,
        "- Untrained: same={us_m:.3}±{us_s:.3} cross={uc_m:.3}±{uc_s:.3}",
    );
    let _ = writeln!(
        s,
        "- Trained:   same={ts_m:.3}±{ts_s:.3} cross={tc_m:.3}±{tc_s:.3}",
    );
    let d_same = ts_m - us_m;
    let d_cross = tc_m - uc_m;
    let _ = writeln!(
        s,
        "- Δ same  = {d_same:+.3} (trained − untrained, target > 0: consistency rises)",
    );
    let _ = writeln!(
        s,
        "- Δ cross = {d_cross:+.3} (trained − untrained, target < 0: specificity rises)",
    );
    let _ = writeln!(
        s,
        "- Δ-of-Δ  = {:+.3} (Δ same − Δ cross; > 0 ⇒ engrams form *and* are cue-specific)",
        d_same - d_cross,
    );

    s
}

/// Build a fingerprint for every word in `vocab` against the
/// *current* brain state. Plasticity is expected to be off (or at
/// least the caller has accepted that the fingerprint pass will
/// drift weights slightly — empirically negligible at the cue
/// drives this benchmark uses).
fn build_vocab_dictionary(
    brain: &mut Brain,
    encoder: &TextEncoder,
    r2_e: &BTreeSet<usize>,
    vocab: &[String],
    dg_sdr_map: &std::collections::HashMap<String, Vec<u32>>,
    dg_drive_strength: f32,
) -> EngramDictionary {
    let mut dict = EngramDictionary::new();
    for word in vocab {
        let sdr = encoder.encode_word(word);
        if sdr.indices.is_empty() {
            continue;
        }
        brain.regions[1].network.reset_state();
        let cs = if let Some(dg_sdr) = dg_sdr_map.get(word) {
            drive_with_dg_counts(
                brain,
                &sdr.indices,
                dg_sdr,
                DRIVE_NA,
                dg_drive_strength,
                RECALL_MS,
                r2_e,
            )
        } else {
            drive_for_with_counts(brain, &sdr.indices, RECALL_MS, r2_e)
        };
        let kw = top_k_indices(&cs, KWTA_K);
        if !kw.is_empty() {
            dict.learn_concept(word, &kw);
        }
    }
    dict
}

/// Read-only-ish evaluation pass against a pre-built dictionary.
/// Drives the cue, kWTAs the R2 activity, decodes against `dict`,
/// and returns `(target in top-1, target in top-3)`.
fn evaluate_with_dict(
    brain: &mut Brain,
    encoder: &TextEncoder,
    r2_e: &BTreeSet<usize>,
    dict: &EngramDictionary,
    cue: &str,
    target: &str,
) -> (bool, bool) {
    let cue_sdr = encoder.encode_word(cue);
    if cue_sdr.indices.is_empty() {
        return (false, false);
    }
    brain.regions[1].network.reset_state();
    let counts = drive_for_with_counts(brain, &cue_sdr.indices, RECALL_MS, r2_e);
    let kwta = top_k_indices(&counts, KWTA_K);
    if kwta.is_empty() {
        return (false, false);
    }
    let decoded = dict.decode_top(&kwta, 3);
    let in_top1 = decoded.first().is_some_and(|(w, _)| w == target);
    let in_top3 = decoded.iter().any(|(w, _)| w == target);
    (in_top1, in_top3)
}

// =====================================================================
// Iter-63 — direct cue → target metric on the DG-enabled brain.
//
// Pre-registered single metric (per `notes/63-cue-target-metric.md`):
// `target_top3_overlap` is operationally identical to iter-46's
// `prediction_top3_before_teacher` — the fraction of real-pair
// prediction-phase trials whose top-k contained any neuron from the
// canonical-target SDR. The new wiring exposes that same number on
// a DG-enabled brain through an explicit `--mode {untrained,trained}`
// CLI surface and a paired-seed renderer.
// =====================================================================

/// Iter-63 explicit arm selector. The CLI requires `--mode <X>`; no
/// implicit code path exists. Untrained ⇒ `no_plasticity = true`,
/// trained ⇒ `no_plasticity = false`. The asymmetric defaults from
/// iter-52 (`--no-plasticity` flag flipping the bit) made the
/// iter-50 "wrong-arm-by-accident" class of bug possible; iter-63
/// forces the operator to state the arm out loud.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmMode {
    Untrained,
    Trained,
}

impl ArmMode {
    pub fn label(self) -> &'static str {
        match self {
            ArmMode::Untrained => "untrained",
            ArmMode::Trained => "trained",
        }
    }
}

/// Iter-63 target_top3_overlap result for a single arm. The `seeds`
/// vector is preserved verbatim so the paired-seed invariant in
/// [`render_target_overlap_sweep`] can verify position-by-position
/// equality across arms (paired t(n−1) on independently-drawn seed
/// lists is structurally invalid).
#[derive(Debug, Clone)]
pub struct TargetOverlapMetrics {
    pub mode: ArmMode,
    pub seeds: Vec<u64>,
    pub per_seed: Vec<f32>,
    pub mean: f32,
    pub std: f32,
    /// Iter-66: per-seed C1 readout (`c1_target_top3_overlap`).
    /// Empty when `cfg.teacher.c1.enabled = false`. When non-empty,
    /// the vector is parallel to `per_seed` (`per_seed[i]` = R2
    /// metric, `c1_per_seed[i]` = C1 metric, both at `seeds[i]`).
    pub c1_per_seed: Vec<f32>,
    /// Iter-66: mean of `c1_per_seed`. `0.0` when the C1 readout
    /// is disabled.
    pub c1_mean: f32,
    /// Iter-66: sample-std of `c1_per_seed`. `0.0` when the C1
    /// readout is disabled or only one seed was used.
    pub c1_std: f32,
}

impl TargetOverlapMetrics {
    fn new(mode: ArmMode, seeds: Vec<u64>, per_seed: Vec<f32>, c1_per_seed: Vec<f32>) -> Self {
        let n = per_seed.len();
        let mean = if n == 0 {
            0.0
        } else {
            per_seed.iter().sum::<f32>() / n as f32
        };
        let std = if n > 1 {
            let var = per_seed.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (n - 1) as f32;
            var.sqrt()
        } else {
            0.0
        };
        let c1_n = c1_per_seed.len();
        let c1_mean = if c1_n == 0 {
            0.0
        } else {
            c1_per_seed.iter().sum::<f32>() / c1_n as f32
        };
        let c1_std = if c1_n > 1 {
            let var = c1_per_seed
                .iter()
                .map(|v| (v - c1_mean).powi(2))
                .sum::<f32>()
                / (c1_n - 1) as f32;
            var.sqrt()
        } else {
            0.0
        };
        Self {
            mode,
            seeds,
            per_seed,
            mean,
            std,
            c1_per_seed,
            c1_mean,
            c1_std,
        }
    }
}

/// Iter-66: per-seed return value of `run_target_overlap_one_seed`.
/// `r2` is the legacy R2-readout `target_top3_overlap` (mean over
/// epochs). `c1` is the C1-readout `c1_target_top3_overlap` and is
/// `Some` only when the brain was built with `cfg.teacher.c1.enabled
/// = true`. The outer `TargetOverlapMetrics` aggregates these
/// across seeds so the iter-66 verdict block can present R2 and C1
/// arms side-by-side.
#[derive(Debug, Clone, Copy)]
pub struct OverlapSeedResult {
    pub r2: f32,
    pub c1: Option<f32>,
}

/// Iter-63 single-arm runner — v2, post plumbing-fix refactor.
///
/// For each seed, builds the iter-63-config brain via the shared
/// `build_benchmark_brain` helper (which honours `decorrelated_init`
/// and `dg.enabled` — the bug v1 had was silently dropping these
/// because it routed through `run_reward_benchmark`, which ignores
/// both flags). Trains in place using the same per-trial schedule
/// as `train_brain_inplace` / `run_reward_benchmark`, computes the
/// iter-44/45 `top3_accuracy` after every epoch, and returns the
/// mean across epochs as the per-seed `target_top3_overlap`.
///
/// Plasticity gating is centralised: untrained mode calls
/// `disable_all_plasticity` (the iter-62 8-rule stack) before
/// training begins, then asserts `assert_no_weight_drift` after
/// the run completes — closing the iter-52 invariant gap that v1
/// hit at vocab=64 + epochs=32 (R2-recurrent L2 went 159.87 →
/// 1606.87 because metaplasticity / heterosynaptic / structural
/// were not gated). Trained mode enables the iter-46 plasticity
/// stack (STDP + iSTDP + homeostasis + intrinsic + reward), and
/// the per-epoch eval phase temporarily disables every rule
/// (full 8-rule stack when `recall_mode_eval` is on, just
/// STDP + iSTDP otherwise — matching the iter-46 read-only-eval
/// convention).
///
/// The metric is the iter-44/45 `top3_accuracy` (cue → R2 →
/// decoder dictionary → top-3 vs target word). iter-46 / iter-50 /
/// iter-51 calibrated this metric: positive-control band [0.07,
/// 0.15] (iter-51 mean = 0.107, 95 % CI [0.069, 0.145]).
/// Pre-measurement-correction notes in `notes/63-cue-target-
/// metric.md`.
pub fn run_target_overlap_arm(
    corpus: &RewardCorpus,
    cfg: &RewardConfig,
    seeds: &[u64],
    mode: ArmMode,
) -> TargetOverlapMetrics {
    match mode {
        ArmMode::Untrained => assert!(
            cfg.teacher.no_plasticity,
            "iter-63 --mode untrained requires no_plasticity = true (operational \
             definition (a) per ENTRY note: fresh init weights, no plasticity \
             ever applied)"
        ),
        ArmMode::Trained => assert!(
            !cfg.teacher.no_plasticity,
            "iter-63 --mode trained requires plasticity enabled (no_plasticity \
             = false)"
        ),
    }
    assert!(
        !seeds.is_empty(),
        "iter-63 target_top3_overlap requires at least one seed"
    );

    let mut per_seed: Vec<f32> = Vec::with_capacity(seeds.len());
    let mut c1_per_seed: Vec<f32> = Vec::with_capacity(seeds.len());
    for &seed in seeds {
        let mut cfg_seeded = *cfg;
        cfg_seeded.seed = seed;

        if matches!(mode, ArmMode::Untrained) {
            cfg_seeded.use_reward = false;
            cfg_seeded.epochs = cfg_seeded.epochs.max(1);
        } else {
            assert!(
                cfg_seeded.epochs > 0,
                "iter-63 trained mode requires --epochs > 0"
            );
        }

        let result = run_target_overlap_one_seed(corpus, &cfg_seeded, mode);
        if let Some(c1) = result.c1 {
            eprintln!(
                "[iter-63 {arm}] seed={seed} target_top3_overlap={value:.4} \
                 c1_target_top3_overlap={c1:.4} \
                 (mean top3_accuracy over {n_ep} epochs, vocab={vocab}, dg={dg}, \
                 c1={c1on}, decorrelated={dec}, recall_mode={rec})",
                arm = mode.label(),
                value = result.r2,
                n_ep = cfg_seeded.epochs,
                vocab = corpus.vocab.len(),
                dg = cfg_seeded.teacher.dg.enabled,
                c1on = cfg_seeded.teacher.c1.enabled,
                dec = cfg_seeded.teacher.decorrelated_init,
                rec = cfg_seeded.teacher.recall_mode_eval,
            );
            c1_per_seed.push(c1);
        } else {
            eprintln!(
                "[iter-63 {arm}] seed={seed} target_top3_overlap={value:.4} \
                 (mean top3_accuracy over {n_ep} epochs, vocab={vocab}, dg={dg}, \
                 decorrelated={dec}, recall_mode={rec})",
                arm = mode.label(),
                value = result.r2,
                n_ep = cfg_seeded.epochs,
                vocab = corpus.vocab.len(),
                dg = cfg_seeded.teacher.dg.enabled,
                dec = cfg_seeded.teacher.decorrelated_init,
                rec = cfg_seeded.teacher.recall_mode_eval,
            );
        }
        per_seed.push(result.r2);
    }
    TargetOverlapMetrics::new(mode, seeds.to_vec(), per_seed, c1_per_seed)
}

/// Runs the iter-63 schedule for a single seed and returns the
/// mean of per-epoch `top3_accuracy`. Internal helper — the public
/// `run_target_overlap_arm` is the one tests / CLI exercises.
fn run_target_overlap_one_seed(
    corpus: &RewardCorpus,
    cfg: &RewardConfig,
    mode: ArmMode,
) -> OverlapSeedResult {
    let BrainBuild {
        mut brain,
        encoder,
        r2_e,
        target_r2_map,
        dg_sdr_map,
        r2_n_used,
        decorrelated_block_size,
        c1_e,
        target_c1_map,
        c1_synapse_start,
    } = build_benchmark_brain(corpus, cfg);

    if let Some(block_size) = decorrelated_block_size {
        eprintln!(
            "[iter-63 {arm}] seed={} decorrelated init: vocab={} R2_N={} R2-E={} block_size={} (disjoint invariant ✓)",
            cfg.seed,
            corpus.vocab.len(),
            r2_n_used,
            r2_e.len(),
            block_size,
            arm = mode.label(),
        );
    }
    if cfg.teacher.dg.enabled {
        eprintln!(
            "[iter-63 {arm}] seed={} DG bridge: dg_size={} dg_k={} dg_to_r2_fanout={} dg_to_r2_weight={:.2} direct_r1r2_scale={:.2}",
            cfg.seed,
            cfg.teacher.dg.size,
            cfg.teacher.dg.k,
            cfg.teacher.dg.to_r2_fanout,
            cfg.teacher.dg.to_r2_weight,
            cfg.teacher.dg.direct_r1r2_weight_scale,
            arm = mode.label(),
        );
    }
    if let Some(ref c1_e_set) = c1_e {
        eprintln!(
            "[iter-66 {arm}] seed={} C1 readout: c1_size={} sparsity_k={} from_r2_fanout={} init_w_max={:.2} teacher_strength={:.2} (R2-E→C1 plastic, M_target gated)",
            cfg.seed,
            c1_e_set.len(),
            cfg.teacher.c1.sparsity_k,
            cfg.teacher.c1.from_r2_fanout,
            cfg.teacher.c1.init_w_max,
            cfg.teacher.c1.teacher_strength,
            arm = mode.label(),
        );
    }

    let no_plasticity = cfg.teacher.no_plasticity;
    let teacher_active = cfg.teacher.enabled;
    let recall_mode_active = cfg.teacher.recall_mode_eval && !no_plasticity;
    let stdp_params = stdp();

    if no_plasticity {
        // Iter-63 v2 fix #2: full 8-rule disable closes the iter-52
        // invariant gap that iter-52 itself never exercised because
        // its untrained arm used epochs=0 (no training loop ⇒ no
        // plasticity events). iter-63 untrained mode runs the full
        // training loop so any default-on rule (metaplasticity,
        // heterosynaptic, structural) would mutate weights without
        // this gate.
        disable_all_plasticity(&mut brain, 1);
    } else {
        let initial_istdp = if cfg.teacher.iter46_baseline {
            istdp_iter46_baseline()
        } else {
            istdp_iter49(&cfg.teacher, 0)
        };
        brain.regions[1].network.enable_stdp(stdp_params);
        brain.regions[1].network.enable_istdp(initial_istdp);
        brain.regions[1].network.enable_homeostasis(homeostasis());
        if !cfg.teacher.iter46_baseline {
            brain.regions[1]
                .network
                .enable_intrinsic_plasticity(intrinsic());
        }
        if cfg.use_reward || cfg.teacher.c1.enabled {
            // Iter-66: C1 readout requires R-STDP enabled so the
            // teacher-phase M_target modulator can drive plastic
            // updates on the new R2-E → C1 synapses. Defensive:
            // iter-46 / iter-65 paths set `use_reward = true` via
            // `with_teacher`, but explicit OR keeps the invariant.
            brain.regions[1]
                .network
                .enable_reward_learning(reward_params());
        }
        // Iter-67 BTSP plateau-eligibility on R2-E → C1.  Restricted
        // to the C1 cell index range via `post_filter` so R2-R2
        // synapses (also in this network) are NOT subject to BTSP —
        // only the new R2-E → C1 sub-pathway gets the plateau-gated
        // retroactive potentiation rule.  R-STDP stays alive on the
        // R2-R2 synapses (handled above by enable_reward_learning).
        // No-op when c1.btsp = false ⇒ iter-66 / iter-66.5 numerics
        // bit-identical.
        if cfg.teacher.c1.enabled && cfg.teacher.c1.btsp {
            if let Some(ref c1_e_set) = c1_e {
                let bp = snn_core::BtspParams {
                    eligibility_window_ms: cfg.teacher.c1.btsp_window_ms,
                    plateau_window_ms: 30.0,
                    plateau_threshold_spikes: 5.0,
                    potentiation_strength: cfg.teacher.c1.btsp_strength,
                    post_plateau_decay_ms: 50.0,
                    w_min: 0.0,
                    w_max: 0.8,
                    target_gated: cfg.teacher.c1.btsp_target_gated,
                };
                let post_filter: Vec<usize> = c1_e_set.iter().copied().collect();
                brain.regions[1].network.enable_btsp(bp, Some(&post_filter));
            }
        }
    }

    let pre_l2 = snapshot_weights(&brain);

    // Iter-66 step 7.5 diagnostic: pre-train snapshot of the
    // R2-E → C1 plastic suffix. Captures raw weight magnitudes so
    // post-train deltas can be computed (mean / max change,
    // nonzero-update count). Cheap when c1.diagnostic = false —
    // single boolean check.
    let c1_diag = cfg.teacher.c1.enabled && cfg.teacher.c1.diagnostic;
    let r2c1_pre_weights: Vec<f32> = if let (true, Some(start)) = (c1_diag, c1_synapse_start) {
        brain.regions[1].network.synapses[start..]
            .iter()
            .map(|s| s.weight)
            .collect()
    } else {
        Vec::new()
    };
    if c1_diag {
        let l2_pre: f64 = r2c1_pre_weights
            .iter()
            .map(|&w| (w as f64) * (w as f64))
            .sum::<f64>()
            .sqrt();
        let mean_pre: f64 = if r2c1_pre_weights.is_empty() {
            0.0
        } else {
            r2c1_pre_weights.iter().map(|&w| w as f64).sum::<f64>() / r2c1_pre_weights.len() as f64
        };
        eprintln!(
            "[iter-66 diag] seed={} pre-train: r2_r2_synapses={} r2c1_synapses={} \
             r2c1_l2={l2_pre:.4} r2c1_mean_w={mean_pre:.4} r2c1_init_w_max={iwm:.2}",
            cfg.seed,
            c1_synapse_start.unwrap_or(0),
            r2c1_pre_weights.len(),
            iwm = cfg.teacher.c1.init_w_max,
        );
    }

    let vocab: Vec<String> = corpus.vocab.iter().cloned().collect();
    let mut rng = Rng::new(cfg.seed);
    let mut per_epoch_top3: Vec<f32> = Vec::with_capacity(cfg.epochs);
    // Iter-66: parallel per-epoch C1-readout `top3_accuracy`.
    // Empty when c1.enabled = false; pushed in lock-step with
    // `per_epoch_top3` otherwise so a per-epoch zip(R2, C1) is
    // always position-aligned by epoch index.
    let mut per_epoch_top3_c1: Vec<f32> = Vec::with_capacity(cfg.epochs);

    for epoch in 0..cfg.epochs {
        // Per-epoch iSTDP refresh (iter-49 ramp / iter-46 baseline).
        let istdp_params = if cfg.teacher.iter46_baseline {
            istdp_iter46_baseline()
        } else {
            istdp_iter49(&cfg.teacher, epoch)
        };
        if !no_plasticity {
            brain.regions[1].network.enable_istdp(istdp_params);
        }

        // Iter-66 step 7.5 diagnostic: per-epoch C1 spike accumulators
        // populated from `TrialOutcome`'s c1_*  fields. Cheap when
        // c1.diagnostic = false (just unused locals).
        let mut diag_train_trials: u32 = 0;
        let mut diag_train_c1_active: u32 = 0;
        let mut diag_train_c1_total_spikes: u64 = 0;
        let mut diag_train_c1_clamp_hits: u64 = 0;
        let mut diag_train_c1_clamp_size: u64 = 0;
        // Iter-67: snapshot BTSP counters at start of training so the
        // per-epoch diff captures only training-phase events. The
        // counters persist across reset_state (intentional — they
        // accumulate across trials within a single epoch's training);
        // eval-phase trials don't fire BTSP (recall-mode plasticity
        // off) so the diff = training-only events.
        let btsp_pe_at_train_start = brain.regions[1].network.btsp_plateau_events;
        let btsp_pot_at_train_start = brain.regions[1].network.btsp_potentiation_events;

        // -- Training: cue+target presentations for this epoch --
        let mut schedule: Vec<(RewardPair, bool)> = corpus
            .pairs
            .iter()
            .map(|p| (p.clone(), false))
            .chain(corpus.noise_pairs.iter().map(|p| (p.clone(), true)))
            .collect();
        shuffle(&mut rng, &mut schedule);

        for (pair, is_noise) in &schedule {
            let cue_sdr = encoder.encode_word(&pair.cue);
            let tgt_sdr = encoder.encode_word(&pair.target);

            if teacher_active {
                let canonical = target_r2_map.get(&pair.target).cloned().unwrap_or_default();
                let dg_sdr = dg_sdr_map.get(&pair.cue).cloned().unwrap_or_default();
                // Iter-66: pass the per-target C1 SDR so the teacher
                // schedule can clamp C1 cells alongside R2 and gate
                // R-STDP via M_target during the encoding window.
                let c1_sdr = target_c1_map.get(&pair.target).cloned().unwrap_or_default();
                for _rep in 0..cfg.reps_per_pair.max(1) {
                    brain.regions[1].network.reset_state();
                    let outcome = run_teacher_trial(
                        &mut brain,
                        &cfg.teacher,
                        cfg.use_reward,
                        *is_noise,
                        &cue_sdr.indices,
                        &canonical,
                        &r2_e,
                        stdp_params,
                        istdp_params,
                        &dg_sdr,
                        &c1_sdr,
                    );
                    if c1_diag {
                        diag_train_trials += 1;
                        diag_train_c1_total_spikes += outcome.c1_teacher_spikes as u64;
                        if outcome.c1_teacher_spikes > 0 {
                            diag_train_c1_active += 1;
                        }
                        diag_train_c1_clamp_hits += outcome.c1_target_clamp_hits as u64;
                        diag_train_c1_clamp_size += outcome.c1_target_clamp_size as u64;
                    }
                    idle(&mut brain, COOLDOWN_MS);
                }
            } else {
                let mut combined: Vec<u32> = cue_sdr
                    .indices
                    .iter()
                    .chain(tgt_sdr.indices.iter())
                    .copied()
                    .collect();
                combined.sort_unstable();
                combined.dedup();
                for _ in 0..cfg.reps_per_pair.max(1) {
                    brain.regions[1].network.reset_state();
                    drive_for(&mut brain, &cue_sdr.indices, CUE_LEAD_MS);
                    drive_for(&mut brain, &combined, OVERLAP_MS);
                    drive_for(&mut brain, &tgt_sdr.indices, TARGET_TAIL_MS);
                    idle(&mut brain, COOLDOWN_MS);
                }
            }
        }

        // -- Eval: build dictionary, present each cue alone, decode
        //    top-3 vs target word. iter-46 read-only-eval convention:
        //    silence STDP / iSTDP for the duration. Iter-62 recall
        //    mode extends this to the full 8-rule stack so any rule
        //    that could mutate weights during eval is gated.
        let saved_modulator = brain.regions[1].network.neuromodulator;
        if !no_plasticity {
            if recall_mode_active {
                disable_all_plasticity(&mut brain, 1);
            } else {
                brain.regions[1].network.disable_stdp();
                brain.regions[1].network.disable_istdp();
            }
        }
        brain.set_neuromodulator(0.0);

        let dict = build_vocab_dictionary(
            &mut brain,
            &encoder,
            &r2_e,
            &vocab,
            &dg_sdr_map,
            cfg.teacher.dg.drive_strength,
        );
        let mut top3 = 0usize;
        let mut pairs_n = 0usize;
        for pair in &corpus.pairs {
            let cue_sdr = encoder.encode_word(&pair.cue);
            if cue_sdr.indices.is_empty() {
                continue;
            }
            brain.regions[1].network.reset_state();
            let counts = if let Some(dg_sdr) = dg_sdr_map.get(&pair.cue) {
                drive_with_dg_counts(
                    &mut brain,
                    &cue_sdr.indices,
                    dg_sdr,
                    DRIVE_NA,
                    cfg.teacher.dg.drive_strength,
                    RECALL_MS,
                    &r2_e,
                )
            } else {
                drive_for_with_counts(&mut brain, &cue_sdr.indices, RECALL_MS, &r2_e)
            };
            let kwta = top_k_indices(&counts, KWTA_K);
            if kwta.is_empty() {
                pairs_n += 1;
                continue;
            }
            let decoded = dict.decode_top(&kwta, 16);
            let rank = decoded
                .iter()
                .position(|(w, _)| w == &pair.target)
                .map(|p| p + 1);
            if let Some(r) = rank {
                if r <= 3 {
                    top3 += 1;
                }
            }
            pairs_n += 1;
        }
        let top3_accuracy = if pairs_n > 0 {
            top3 as f32 / pairs_n as f32
        } else {
            0.0
        };
        per_epoch_top3.push(top3_accuracy);

        // Iter-66 (M1): C1 readout. Re-uses the same eval-phase
        // gating (plasticity already disabled above), builds a
        // C1-fingerprint dictionary by reading C1 cell spikes
        // during each cue-only presentation, and decodes top-3
        // against the canonical-target word. Empty when C1 is
        // disabled — keeps the iter-46/65 path bit-identical.
        if let Some(ref c1_e_set) = c1_e {
            let c1_dict = build_vocab_dictionary(
                &mut brain,
                &encoder,
                c1_e_set,
                &vocab,
                &dg_sdr_map,
                cfg.teacher.dg.drive_strength,
            );
            let mut c1_top3 = 0usize;
            let mut c1_pairs_n = 0usize;
            // Iter-66 step 7.5 diagnostic accumulators.
            let mut diag_eval_spikes_total: u64 = 0;
            let mut diag_eval_kwta_empty: u32 = 0;
            let mut diag_eval_target_in_dict: u32 = 0;
            let mut diag_eval_mrr_sum: f64 = 0.0;
            let mut diag_eval_raw_overlap_sum: u64 = 0;
            let mut diag_eval_raw_overlap_denom: u64 = 0;
            let dict_n_concepts = c1_dict.len();
            for pair in &corpus.pairs {
                let cue_sdr = encoder.encode_word(&pair.cue);
                if cue_sdr.indices.is_empty() {
                    continue;
                }
                brain.regions[1].network.reset_state();
                let c1_counts = if let Some(dg_sdr) = dg_sdr_map.get(&pair.cue) {
                    drive_with_dg_counts(
                        &mut brain,
                        &cue_sdr.indices,
                        dg_sdr,
                        DRIVE_NA,
                        cfg.teacher.dg.drive_strength,
                        RECALL_MS,
                        c1_e_set,
                    )
                } else {
                    drive_for_with_counts(&mut brain, &cue_sdr.indices, RECALL_MS, c1_e_set)
                };
                if c1_diag {
                    let total: u64 = c1_counts.iter().map(|&c| c as u64).sum();
                    diag_eval_spikes_total += total;
                    if let Some(canonical_c1) = target_c1_map.get(&pair.target) {
                        let kw_top: BTreeSet<u32> =
                            top_k_indices(&c1_counts, cfg.teacher.c1.sparsity_k as usize)
                                .into_iter()
                                .collect();
                        let canon_set: BTreeSet<u32> = canonical_c1.iter().copied().collect();
                        let inter = kw_top.intersection(&canon_set).count() as u64;
                        diag_eval_raw_overlap_sum += inter;
                        diag_eval_raw_overlap_denom += canon_set.len() as u64;
                    }
                }
                let c1_kwta = top_k_indices(&c1_counts, cfg.teacher.c1.sparsity_k as usize);
                if c1_kwta.is_empty() {
                    if c1_diag {
                        diag_eval_kwta_empty += 1;
                    }
                    c1_pairs_n += 1;
                    continue;
                }
                let c1_decoded = c1_dict.decode_top(&c1_kwta, 16);
                let c1_rank = c1_decoded
                    .iter()
                    .position(|(w, _)| w == &pair.target)
                    .map(|p| p + 1);
                if c1_diag {
                    if c1_rank.is_some() {
                        diag_eval_target_in_dict += 1;
                    }
                    if let Some(r) = c1_rank {
                        diag_eval_mrr_sum += 1.0 / r as f64;
                    }
                }
                if let Some(r) = c1_rank {
                    if r <= 3 {
                        c1_top3 += 1;
                    }
                }
                c1_pairs_n += 1;
            }
            let c1_acc = if c1_pairs_n > 0 {
                c1_top3 as f32 / c1_pairs_n as f32
            } else {
                0.0
            };
            per_epoch_top3_c1.push(c1_acc);

            if c1_diag {
                // Per-epoch R2→C1 weight stats. Slice the synapses
                // suffix and compare to the pre-train snapshot.
                let l2_now: f64 = if let Some(start) = c1_synapse_start {
                    brain.regions[1].network.synapses[start..]
                        .iter()
                        .map(|s| (s.weight as f64) * (s.weight as f64))
                        .sum::<f64>()
                        .sqrt()
                } else {
                    0.0
                };
                let mut nonzero_updates: u64 = 0;
                let mut max_abs_delta: f32 = 0.0;
                let mut sum_abs_delta: f64 = 0.0;
                if let Some(start) = c1_synapse_start {
                    let now_slice = &brain.regions[1].network.synapses[start..];
                    for (i, syn) in now_slice.iter().enumerate() {
                        let pre = r2c1_pre_weights.get(i).copied().unwrap_or(0.0);
                        let d = syn.weight - pre;
                        let absd = d.abs();
                        if absd > 1e-7 {
                            nonzero_updates += 1;
                        }
                        if absd > max_abs_delta {
                            max_abs_delta = absd;
                        }
                        sum_abs_delta += absd as f64;
                    }
                }
                let n_train = diag_train_trials.max(1) as f64;
                let train_clamp_eff = if diag_train_c1_clamp_size > 0 {
                    diag_train_c1_clamp_hits as f64 / diag_train_c1_clamp_size as f64
                } else {
                    0.0
                };
                let eval_n = corpus.pairs.len().max(1) as f64;
                let eval_mean_spikes = diag_eval_spikes_total as f64 / eval_n;
                let mrr = diag_eval_mrr_sum / eval_n;
                let raw_overlap_ratio = if diag_eval_raw_overlap_denom > 0 {
                    diag_eval_raw_overlap_sum as f64 / diag_eval_raw_overlap_denom as f64
                } else {
                    0.0
                };
                // Iter-67: BTSP per-epoch counters as DELTAS vs the
                // start-of-training snapshot, so the diagnostic shows
                // per-epoch totals (not cumulative-since-enable).
                // When c1.btsp = false both deltas stay 0.  The
                // per-class mean weights are computed below and are
                // still informative on the iter-66.5 R-STDP path.
                let btsp_pe = brain.regions[1]
                    .network
                    .btsp_plateau_events
                    .wrapping_sub(btsp_pe_at_train_start);
                let btsp_pot = brain.regions[1]
                    .network
                    .btsp_potentiation_events
                    .wrapping_sub(btsp_pot_at_train_start);
                let (r2c1_target_w, r2c1_nontarget_w) = if let Some(start) = c1_synapse_start {
                    let mut tgt_sum: f64 = 0.0;
                    let mut tgt_n: u64 = 0;
                    let mut non_sum: f64 = 0.0;
                    let mut non_n: u64 = 0;
                    // Build a per-post-cell membership flag: true
                    // ⇔ this C1 cell is in *any* word's canonical
                    // C1 target SDR.  R2-E → C1 synapses then
                    // sort into target / non-target buckets by
                    // their post-cell.
                    let n_neurons = brain.regions[1].network.neurons.len();
                    let mut is_target_cell = vec![false; n_neurons];
                    for sdr in target_c1_map.values() {
                        for &idx in sdr {
                            let i = idx as usize;
                            if i < n_neurons {
                                is_target_cell[i] = true;
                            }
                        }
                    }
                    let synapses = &brain.regions[1].network.synapses;
                    for syn in synapses[start..].iter() {
                        let w = syn.weight as f64;
                        let post = syn.post;
                        if is_target_cell.get(post).copied().unwrap_or(false) {
                            tgt_sum += w;
                            tgt_n += 1;
                        } else {
                            non_sum += w;
                            non_n += 1;
                        }
                    }
                    let tgt_mean = if tgt_n > 0 {
                        tgt_sum / tgt_n as f64
                    } else {
                        0.0
                    };
                    let non_mean = if non_n > 0 {
                        non_sum / non_n as f64
                    } else {
                        0.0
                    };
                    (tgt_mean, non_mean)
                } else {
                    (0.0, 0.0)
                };
                let r2c1_w_ratio = if r2c1_nontarget_w > 1e-9 {
                    r2c1_target_w / r2c1_nontarget_w
                } else {
                    0.0
                };
                eprintln!(
                    "[iter-66 diag] seed={} epoch={epoch}/{ep_total} \
                     teacher: trials={tt} c1_active_frac={af:.3} c1_spikes_mean={sm:.2} \
                     clamp_eff={ce:.3} | eval: kwta_empty={ke}/{ep} target_in_dict={td}/{ep} \
                     spikes_mean={es:.2} top3_r2={r2:.4} top3_c1={c1:.4} mrr_c1={mrr:.4} \
                     raw_overlap={ro:.3} dict_concepts={dc} | r2c1: l2={l2:.4} nz_upd={nu} \
                     max|Δw|={mx:.4} sum|Δw|={sd:.4} tgt_w={tw:.4} non_w={nw:.4} \
                     w_ratio={wr:.3} | btsp: plateau_events={pe} potentiation_events={pot}",
                    cfg.seed,
                    ep_total = cfg.epochs,
                    tt = diag_train_trials,
                    af = (diag_train_c1_active as f64) / n_train,
                    sm = (diag_train_c1_total_spikes as f64) / n_train,
                    ce = train_clamp_eff,
                    ke = diag_eval_kwta_empty,
                    ep = corpus.pairs.len(),
                    td = diag_eval_target_in_dict,
                    es = eval_mean_spikes,
                    r2 = top3_accuracy,
                    c1 = c1_acc,
                    mrr = mrr,
                    ro = raw_overlap_ratio,
                    dc = dict_n_concepts,
                    l2 = l2_now,
                    nu = nonzero_updates,
                    mx = max_abs_delta,
                    sd = sum_abs_delta,
                    tw = r2c1_target_w,
                    nw = r2c1_nontarget_w,
                    wr = r2c1_w_ratio,
                    pe = btsp_pe,
                    pot = btsp_pot,
                );
            }
        }

        // Restore plasticity state for the next epoch's training. If
        // recall_mode_active was used to disable everything, only STDP /
        // iSTDP need re-enabling here — homeostasis / intrinsic /
        // reward / metaplasticity / heterosynaptic / structural were
        // not enabled by the run_target_overlap_one_seed setup beyond
        // the original initial enable, and the iter-46 convention is
        // to leave them as the post-disable_all state. For non-recall-
        // mode trained runs, the iter-46 pattern (re-enable STDP +
        // iSTDP) preserves bit-identity to run_reward_benchmark's
        // eval-restore.
        if !no_plasticity {
            brain.regions[1].network.enable_stdp(stdp_params);
            brain.regions[1].network.enable_istdp(istdp_params);
            if recall_mode_active {
                // Recall mode disabled everything; bring the rest
                // back so the next epoch trains as expected.
                brain.regions[1].network.enable_homeostasis(homeostasis());
                if !cfg.teacher.iter46_baseline {
                    brain.regions[1]
                        .network
                        .enable_intrinsic_plasticity(intrinsic());
                }
                if cfg.use_reward {
                    brain.regions[1]
                        .network
                        .enable_reward_learning(reward_params());
                }
            }
        }
        brain.set_neuromodulator(saved_modulator);
    }

    if no_plasticity {
        let post_l2 = snapshot_weights(&brain);
        assert_no_weight_drift(
            &pre_l2,
            &post_l2,
            &format!(
                "iter-63 untrained arm changed weights (seed={}, vocab={}, ep={}, dg={}, c1={})",
                cfg.seed,
                corpus.vocab.len(),
                cfg.epochs,
                cfg.teacher.dg.enabled,
                cfg.teacher.c1.enabled,
            ),
        );
    }

    let r2_mean = if per_epoch_top3.is_empty() {
        0.0
    } else {
        per_epoch_top3.iter().sum::<f32>() / per_epoch_top3.len() as f32
    };
    let c1_mean = if per_epoch_top3_c1.is_empty() {
        None
    } else {
        Some(per_epoch_top3_c1.iter().sum::<f32>() / per_epoch_top3_c1.len() as f32)
    };
    OverlapSeedResult {
        r2: r2_mean,
        c1: c1_mean,
    }
}

// =====================================================================
// Iter-64 — mechanism diagnosis: axis sweep infrastructure.
//
// Three isolated diagnostic axes per `notes/64-mechanism-diagnosis.md`:
//   A — DG → R2 drive scale (`dg_to_r2_weight`)
//   B — R2 recurrent connectivity (`r2_p_connect_override`)
//   C — direct (perforant) R1 → R2 path (`direct_r1r2_weight_scale`)
//
// Each axis is swept over a small list of values, paired by seed,
// and classified per the locked iter-64 acceptance matrix
// (Alpha / Beta / Gamma / Delta). The untrained arm at every
// (config, seed) tuple is deterministic — the cache short-circuits
// repeat computations when an axis value happens to coincide with
// the iter-63 baseline configuration. Pre-seeded with the iter-63
// calibration values for the four locked seeds.
// =====================================================================

/// Iter-64 axis under test. Each variant binds to exactly one
/// runtime-overrideable parameter on `TeacherForcingConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepAxis {
    /// Axis A — DG mossy-fibre weight scale into R2.
    DgToR2Weight,
    /// Axis B — R2 recurrent E↔E connectivity probability.
    R2PConnect,
    /// Axis C — direct (perforant) R1 → R2 weight scale.
    DirectR1R2WeightScale,
}

impl SweepAxis {
    pub fn label(self) -> &'static str {
        match self {
            SweepAxis::DgToR2Weight => "dg_to_r2_weight",
            SweepAxis::R2PConnect => "r2_p_connect",
            SweepAxis::DirectR1R2WeightScale => "direct_r1r2_weight_scale",
        }
    }

    pub fn cli_arg(self) -> &'static str {
        match self {
            SweepAxis::DgToR2Weight => "dg-to-r2-weight",
            SweepAxis::R2PConnect => "r2-p-connect",
            SweepAxis::DirectR1R2WeightScale => "direct-r1r2-weight-scale",
        }
    }

    /// iter-63 baseline value for this axis. The configuration
    /// `(dg=1.0, r2_p=0.05, direct=0.0)` reproduces the iter-63
    /// trained main run; any axis value matching its baseline
    /// short-circuits to the cached iter-63 calibration value.
    pub fn iter63_baseline(self) -> f32 {
        match self {
            SweepAxis::DgToR2Weight => 1.0,
            SweepAxis::R2PConnect => R2_P_CONNECT,
            SweepAxis::DirectR1R2WeightScale => 0.0,
        }
    }

    pub fn parse_cli(s: &str) -> Option<Self> {
        match s {
            "dg-to-r2-weight" | "dg_to_r2_weight" => Some(SweepAxis::DgToR2Weight),
            "r2-p-connect" | "r2_p_connect" => Some(SweepAxis::R2PConnect),
            "direct-r1r2-weight-scale" | "direct_r1r2_weight_scale" => {
                Some(SweepAxis::DirectR1R2WeightScale)
            }
            _ => None,
        }
    }
}

/// Iter-64 acceptance classification (locked in
/// `notes/64-mechanism-diagnosis.md`):
///
/// - **(α) Alpha:** `Δ̄ > 0` AND `n_pos ≥ ⌈3n/4⌉` AND `t > 0`
/// - **(β) Beta:** `|Δ̄| ≤ σ_untrained_iter63` (≈ 0.0213)
/// - **(γ) Gamma:** `Δ̄ < 0` AND `n_pos ≤ ⌊n/4⌋` AND `t < −1.0`
/// - **(δ) Delta:** anything else (mixed / inconclusive — needs
///   more seeds before verdict)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisClassification {
    Alpha,
    Beta,
    Gamma,
    Delta,
}

impl AxisClassification {
    pub fn label(self) -> &'static str {
        match self {
            AxisClassification::Alpha => "(α) Alpha — positive trend, deepen in iter-65",
            AxisClassification::Beta => "(β) Beta — no effect, rule out",
            AxisClassification::Gamma => "(γ) Gamma — negative trend, rule out (degrading)",
            AxisClassification::Delta => {
                "(δ) Delta — mixed / inconclusive, more seeds before verdict"
            }
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            AxisClassification::Alpha => "α",
            AxisClassification::Beta => "β",
            AxisClassification::Gamma => "γ",
            AxisClassification::Delta => "δ",
        }
    }
}

/// Two-phase run signal. Smoke phase = 16 epochs, full phase =
/// 32 epochs. Caller decides epoch count via `cfg.epochs`; this
/// enum just tags the result for the renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepPhase {
    Smoke,
    Full,
    /// Custom epoch count that doesn't match either canonical phase.
    Other(usize),
}

impl SweepPhase {
    pub fn from_epochs(epochs: usize) -> Self {
        match epochs {
            16 => SweepPhase::Smoke,
            32 => SweepPhase::Full,
            other => SweepPhase::Other(other),
        }
    }

    pub fn label(self) -> String {
        match self {
            SweepPhase::Smoke => "smoke (16 ep)".to_string(),
            SweepPhase::Full => "full (32 ep)".to_string(),
            SweepPhase::Other(n) => format!("other ({n} ep)"),
        }
    }
}

/// Iter-64 σ_untrained from the iter-63 calibration commit
/// (`a08a117` on `claude/iter63-calibration` → merged via PR #38).
/// Sample std of the four per-seed untrained values
/// (0.0127, 0.0000, 0.0498, 0.0156) at the iter-63 baseline
/// configuration. Used by the (β) classification band and as the
/// reference noise scale for axis verdict thresholds.
const SIGMA_UNTRAINED_ITER63: f32 = 0.0213;

/// One value-point in an axis sweep — both arms + paired stats +
/// classification.
#[derive(Debug, Clone)]
pub struct AxisSweepPoint {
    pub axis: SweepAxis,
    pub value: f32,
    pub trained_per_seed: Vec<f32>,
    pub untrained_per_seed: Vec<f32>,
    pub deltas: Vec<f32>,
    pub mean_trained: f32,
    pub mean_untrained: f32,
    pub mean_delta: f32,
    pub sd_delta: f32,
    pub t_stat: f32,
    pub df: usize,
    pub n_pos: usize,
    /// Count of seeds where Δ ≥ iter-63 locked threshold (0.0621).
    /// Reference only — iter-64's acceptance does not require this.
    pub n_pass_iter63: usize,
    pub classification: AxisClassification,
}

/// Iter-64 axis sweep result — one [`AxisSweepPoint`] per requested
/// value, paired-by-seed across the whole sweep.
#[derive(Debug, Clone)]
pub struct AxisSweepResult {
    pub axis: SweepAxis,
    pub seeds: Vec<u64>,
    pub epochs: usize,
    pub phase: SweepPhase,
    pub points: Vec<AxisSweepPoint>,
}

/// Iter-64 untrained-arm cache key. Hash is stable across runs
/// because `f32::to_bits` gives a deterministic 32-bit integer.
/// All three iter-64-axis-affecting parameters are part of the
/// key so an untrained value computed at one axis can be looked
/// up by any other axis sweep that lands on the same configuration.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct UntrainedCacheKey {
    seed: u64,
    r2_p_connect_bits: u32,
    dg_to_r2_weight_bits: u32,
    direct_r1r2_weight_scale_bits: u32,
}

impl UntrainedCacheKey {
    fn from_cfg(cfg: &RewardConfig, seed: u64) -> Self {
        Self {
            seed,
            r2_p_connect_bits: effective_r2_p_connect(&cfg.teacher).to_bits(),
            dg_to_r2_weight_bits: cfg.teacher.dg.to_r2_weight.to_bits(),
            direct_r1r2_weight_scale_bits: cfg.teacher.dg.direct_r1r2_weight_scale.to_bits(),
        }
    }
}

/// Process-local untrained-arm cache. Seeded with the iter-63
/// calibration values on first access; all four locked seeds at
/// the iter-63 baseline configuration are pre-populated so that
/// any axis sweep landing on the baseline value short-circuits.
fn untrained_cache() -> &'static std::sync::Mutex<std::collections::HashMap<UntrainedCacheKey, f32>>
{
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<UntrainedCacheKey, f32>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        let mut map = std::collections::HashMap::new();
        // iter-63 calibration values (commit a08a117 / PR #38),
        // measured at vocab=64 + DG bridge + recall-mode-eval +
        // decorrelated_init at the configuration tuple
        // (R2_P_CONNECT=0.05, dg_to_r2_weight=1.0,
        //  direct_r1r2_weight_scale=0.0). Seed/value mapping per
        // notes/63-cue-target-metric.md "Calibration result".
        let baseline_r2_p_bits = R2_P_CONNECT.to_bits();
        let baseline_dg_w_bits: u32 = 1.0_f32.to_bits();
        let baseline_direct_bits: u32 = 0.0_f32.to_bits();
        for (seed, value) in [
            (42_u64, 0.0127_f32),
            (7_u64, 0.0000_f32),
            (13_u64, 0.0498_f32),
            (99_u64, 0.0156_f32),
        ] {
            map.insert(
                UntrainedCacheKey {
                    seed,
                    r2_p_connect_bits: baseline_r2_p_bits,
                    dg_to_r2_weight_bits: baseline_dg_w_bits,
                    direct_r1r2_weight_scale_bits: baseline_direct_bits,
                },
                value,
            );
        }
        std::sync::Mutex::new(map)
    })
}

/// Look up the untrained `target_top3_overlap` for one
/// `(cfg, seed)` tuple — cache hit if the configuration matches
/// the iter-63 baseline or any previously-computed point;
/// otherwise the brain is built and the untrained run executed
/// via [`run_target_overlap_one_seed`] under [`ArmMode::Untrained`].
///
/// Result is cached for subsequent lookups in the same process.
/// The cache is process-local; cross-process persistence is out
/// of scope for iter-64 (each `cargo run` rebuilds the cache from
/// the iter-63 baseline pre-seed).
fn cached_untrained_target_top3(corpus: &RewardCorpus, cfg: &RewardConfig, seed: u64) -> f32 {
    let key = UntrainedCacheKey::from_cfg(cfg, seed);
    if let Ok(guard) = untrained_cache().lock() {
        if let Some(&v) = guard.get(&key) {
            return v;
        }
    }
    // Cache miss — build cfg as untrained and compute.
    let mut cfg_un = *cfg;
    cfg_un.seed = seed;
    cfg_un.teacher.no_plasticity = true;
    cfg_un.use_reward = false;
    cfg_un.epochs = cfg_un.epochs.max(1);
    let value = run_target_overlap_one_seed(corpus, &cfg_un, ArmMode::Untrained).r2;
    if let Ok(mut guard) = untrained_cache().lock() {
        guard.entry(key).or_insert(value);
    }
    value
}

/// Sample standard deviation (n − 1 denominator). Returns 0.0
/// for `xs.len() <= 1`.
fn sample_std(xs: &[f32], mean: f32) -> f32 {
    if xs.len() <= 1 {
        return 0.0;
    }
    let v = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (xs.len() - 1) as f32;
    v.sqrt()
}

/// Iter-64 acceptance classification per the locked matrix. The
/// `n` parameter is the seed count; the `n_pos / Δ̄ / t` triple
/// is the per-axis-value paired-stat. Edge cases that fit neither
/// (α) nor (γ) and fall outside the (β) band collapse to (δ).
fn classify_axis_point(n: usize, n_pos: usize, mean_delta: f32, t_stat: f32) -> AxisClassification {
    let three_quarters = n.saturating_mul(3).div_ceil(4);
    let one_quarter = n / 4;
    if mean_delta > 0.0 && n_pos >= three_quarters && t_stat > 0.0 {
        AxisClassification::Alpha
    } else if mean_delta.abs() <= SIGMA_UNTRAINED_ITER63 {
        AxisClassification::Beta
    } else if mean_delta < 0.0 && n_pos <= one_quarter && t_stat < -1.0 {
        AxisClassification::Gamma
    } else {
        AxisClassification::Delta
    }
}

/// Apply an axis value to a configuration. Mutates the relevant
/// `cfg.teacher` field per axis; leaves all other axis-related
/// parameters untouched (axes are isolated per iter-64 spec).
fn apply_axis_value(cfg: &mut RewardConfig, axis: SweepAxis, value: f32) {
    match axis {
        SweepAxis::DgToR2Weight => {
            cfg.teacher.dg.to_r2_weight = value;
        }
        SweepAxis::R2PConnect => {
            cfg.teacher.r2_p_connect_override = Some(value);
        }
        SweepAxis::DirectR1R2WeightScale => {
            cfg.teacher.dg.direct_r1r2_weight_scale = value;
        }
    }
}

/// Iter-64 axis sweep — runs the trained arm at every value in
/// `values` (paired against the cached untrained arm at the same
/// `(config, seed)` tuple), collects per-value paired stats, and
/// classifies each value per the locked iter-64 acceptance matrix.
///
/// `cfg` should carry the iter-63 baseline configuration except
/// for the swept axis: `vocab=64`, `--decorrelated-init`,
/// `--teacher-forcing`, `--target-clamp-strength 500`,
/// `--teacher-ms 40`, `--corpus-vocab 64`, `--dg-bridge`,
/// `--plasticity-off-during-eval`, `epochs=16` (smoke) or `32`
/// (full).
///
/// Untrained values are cached by configuration tuple; multiple
/// axis values that land on the iter-63 baseline configuration
/// share the iter-63-locked untrained calibration value.
pub fn run_axis_sweep(
    corpus: &RewardCorpus,
    cfg: &RewardConfig,
    seeds: &[u64],
    axis: SweepAxis,
    values: &[f32],
) -> AxisSweepResult {
    assert!(cfg.epochs > 0, "iter-64 axis sweep requires --epochs > 0");
    assert!(
        !seeds.is_empty(),
        "iter-64 axis sweep requires at least one seed"
    );
    assert!(
        !values.is_empty(),
        "iter-64 axis sweep requires at least one --values entry"
    );

    let mut points: Vec<AxisSweepPoint> = Vec::with_capacity(values.len());
    for &value in values {
        let mut cfg_axis = *cfg;
        apply_axis_value(&mut cfg_axis, axis, value);

        let mut trained_per_seed: Vec<f32> = Vec::with_capacity(seeds.len());
        let mut untrained_per_seed: Vec<f32> = Vec::with_capacity(seeds.len());

        for &seed in seeds {
            // Untrained — cache lookup, deterministic.
            let untrained = cached_untrained_target_top3(corpus, &cfg_axis, seed);
            untrained_per_seed.push(untrained);

            // Trained — fresh run with full plasticity.
            let mut cfg_t = cfg_axis;
            cfg_t.seed = seed;
            cfg_t.teacher.no_plasticity = false;
            assert!(cfg_t.epochs > 0, "iter-64 trained arm requires epochs > 0");
            let trained = run_target_overlap_one_seed(corpus, &cfg_t, ArmMode::Trained).r2;
            trained_per_seed.push(trained);

            eprintln!(
                "[iter-64 sweep axis={ax} value={value:.4}] seed={seed} \
                 untrained={untrained:.4} trained={trained:.4} Δ={:+.4}",
                trained - untrained,
                ax = axis.label(),
            );
        }

        let deltas: Vec<f32> = trained_per_seed
            .iter()
            .zip(untrained_per_seed.iter())
            .map(|(t, u)| t - u)
            .collect();

        let n = deltas.len();
        let n_pos = deltas.iter().filter(|&&d| d > 0.0).count();
        let n_pass_iter63 = deltas.iter().filter(|&&d| d >= 0.0621).count();
        let mean_delta = if n == 0 {
            0.0
        } else {
            deltas.iter().sum::<f32>() / n as f32
        };
        let sd_delta = sample_std(&deltas, mean_delta);
        let mean_trained = if n == 0 {
            0.0
        } else {
            trained_per_seed.iter().sum::<f32>() / n as f32
        };
        let mean_untrained = if n == 0 {
            0.0
        } else {
            untrained_per_seed.iter().sum::<f32>() / n as f32
        };
        let df = n.saturating_sub(1);
        let t_stat = if sd_delta > 0.0 {
            mean_delta / (sd_delta / (n as f32).sqrt())
        } else if mean_delta > 0.0 {
            f32::INFINITY
        } else if mean_delta < 0.0 {
            f32::NEG_INFINITY
        } else {
            0.0
        };

        let classification = classify_axis_point(n, n_pos, mean_delta, t_stat);

        points.push(AxisSweepPoint {
            axis,
            value,
            trained_per_seed,
            untrained_per_seed,
            deltas,
            mean_trained,
            mean_untrained,
            mean_delta,
            sd_delta,
            t_stat,
            df,
            n_pos,
            n_pass_iter63,
            classification,
        });
    }

    AxisSweepResult {
        axis,
        seeds: seeds.to_vec(),
        epochs: cfg.epochs,
        phase: SweepPhase::from_epochs(cfg.epochs),
        points,
    }
}

/// Iter-64 axis-sweep markdown renderer.
pub fn render_axis_sweep(result: &AxisSweepResult) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "### Iter-64 Axis Sweep — {} ({})\n\n",
        result.axis.label(),
        result.phase.label(),
    ));
    s.push_str(&format!(
        "_n_seeds = {}, seeds = {:?}, σ_untrained_iter63 = {:.4}, iter-63 baseline value = {}._\n\n",
        result.seeds.len(),
        result.seeds,
        SIGMA_UNTRAINED_ITER63,
        result.axis.iter63_baseline(),
    ));
    s.push_str(
        "| value | μ_untrained | μ_trained | Δ̄ | σ_Δ | n_pos | n_pass(0.0621) | t(df) | classification |\n",
    );
    s.push_str("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |\n");
    for p in &result.points {
        s.push_str(&format!(
            "| {:.3} | {:.4} | {:.4} | {:+.4} | {:.4} | {}/{} | {}/{} | {:+.3} (df={}) | {} {} |\n",
            p.value,
            p.mean_untrained,
            p.mean_trained,
            p.mean_delta,
            p.sd_delta,
            p.n_pos,
            result.seeds.len(),
            p.n_pass_iter63,
            result.seeds.len(),
            p.t_stat,
            p.df,
            p.classification.short(),
            p.classification.label(),
        ));
    }
    s.push('\n');
    let n_alpha = result
        .points
        .iter()
        .filter(|p| p.classification == AxisClassification::Alpha)
        .count();
    let n_gamma = result
        .points
        .iter()
        .filter(|p| p.classification == AxisClassification::Gamma)
        .count();
    let n_delta = result
        .points
        .iter()
        .filter(|p| p.classification == AxisClassification::Delta)
        .count();
    let n_beta = result.points.len() - n_alpha - n_gamma - n_delta;
    s.push_str(&format!(
        "**Per-value verdict tally:** α = {n_alpha}, β = {n_beta}, γ = {n_gamma}, δ = {n_delta}.\n",
    ));
    s
}

/// One-sided critical-t value at α=0.05 for the small-n cases iter-63
/// and the canonical iter-65 escalations need. Falls back to the
/// asymptotic z=1.96 for unsupported df. Lookup, not interpolation —
/// keeps the verdict deterministic and inspectable.
fn t_critical_05(df: usize) -> f32 {
    match df {
        1 => 6.314,
        2 => 2.920,
        3 => 2.353,
        4 => 2.132,
        5 => 2.015,
        6 => 1.943,
        7 => 1.895,
        8 => 1.860,
        9 => 1.833,
        10 => 1.812,
        15 => 1.753,
        20 => 1.725,
        30 => 1.697,
        _ => 1.96,
    }
}

/// One-sided critical-t value at α=0.15. Used by the iter-63 (C)
/// branch — Δ > 0 on ≥ 3/4 seeds plus 0.05 ≤ p < 0.15 ⇒ underpowered
/// at n=4, escalate to more seeds.
fn t_critical_15(df: usize) -> f32 {
    match df {
        1 => 1.963,
        2 => 1.386,
        3 => 1.250,
        4 => 1.190,
        5 => 1.156,
        6 => 1.134,
        7 => 1.119,
        8 => 1.108,
        9 => 1.100,
        10 => 1.093,
        15 => 1.074,
        20 => 1.064,
        30 => 1.055,
        _ => 1.04,
    }
}

/// Iter-63 paired renderer. Asserts the paired-seed invariant
/// (untrained.seeds == trained.seeds, position-by-position) and
/// emits a markdown block with the per-seed table, paired t(n−1),
/// and the locked branching verdict from the iter-63 ENTRY note.
///
/// Verdict logic (per the ENTRY's pre-registered branching matrix):
/// - **(A)** Δ ≥ threshold on **all** seeds AND `t > t_crit(0.05)`.
/// - **(B)** Δ < 0 on any seed, OR n_pos ≤ n/2, OR any condition not
///   covered by (A) / (C). Edge cases collapse here.
/// - **(C)** n_pos ≥ ⌈3n/4⌉ AND `t > t_crit(0.15)` AND `t ≤
///   t_crit(0.05)` (underpowered, escalate seeds).
pub fn render_target_overlap_sweep(
    untrained: &TargetOverlapMetrics,
    trained: &TargetOverlapMetrics,
    threshold: f32,
) -> String {
    assert!(
        matches!(untrained.mode, ArmMode::Untrained),
        "render_target_overlap_sweep: first argument must be ArmMode::Untrained"
    );
    assert!(
        matches!(trained.mode, ArmMode::Trained),
        "render_target_overlap_sweep: second argument must be ArmMode::Trained"
    );
    assert_eq!(
        untrained.seeds, trained.seeds,
        "iter-63 paired-seed invariant violated: untrained seeds {:?} vs \
         trained seeds {:?}. Paired t(n−1) requires position-by-position \
         identical seed lists; independently-drawn lists are structurally \
         invalid for paired analysis.",
        untrained.seeds, trained.seeds,
    );

    let n = untrained.seeds.len();
    let deltas: Vec<f32> = trained
        .per_seed
        .iter()
        .zip(untrained.per_seed.iter())
        .map(|(t, u)| t - u)
        .collect();
    let n_pos = deltas.iter().filter(|&&d| d > 0.0).count();
    let n_pass = deltas.iter().filter(|&&d| d >= threshold).count();
    let mean_d = if n == 0 {
        0.0
    } else {
        deltas.iter().sum::<f32>() / n as f32
    };
    let sd_d = if n > 1 {
        let v = deltas.iter().map(|d| (d - mean_d).powi(2)).sum::<f32>() / (n - 1) as f32;
        v.sqrt()
    } else {
        0.0
    };
    let df = n.saturating_sub(1);
    let t_stat = if sd_d > 0.0 {
        mean_d / (sd_d / (n as f32).sqrt())
    } else if mean_d > 0.0 {
        f32::INFINITY
    } else if mean_d < 0.0 {
        f32::NEG_INFINITY
    } else {
        0.0
    };
    let t_c05 = t_critical_05(df);
    let t_c15 = t_critical_15(df);
    let p_lt_05 = t_stat > t_c05;
    let p_lt_15 = t_stat > t_c15;

    // Three-way verdict per ENTRY note. (B) is the catch-all so any
    // edge case lands there, never silently in (A) or (C).
    let three_quarters = n.saturating_mul(3).div_ceil(4);
    let branch = if n_pass == n && p_lt_05 {
        "(A) PASS — Δ ≥ threshold on n/n seeds AND p < 0.05. iter-64 entry: \
         CA3/CA1 split on the verified DG read-out."
    } else if mean_d < 0.0 || n_pos <= n / 2 {
        "(B) FAIL — Δ negative or directional support on ≤ n/2 seeds. \
         iter-64 entry: mechanism question first (DG→R2 lr, R2 recurrent \
         strength, perforant-path re-introduction)."
    } else if n_pos >= three_quarters && p_lt_15 && !p_lt_05 {
        "(C) MIXED — n_pos ≥ ⌈3n/4⌉ AND 0.05 ≤ p < 0.15. iter-64 entry: \
         more seeds at the same architecture; no escalation."
    } else {
        "(B) FAIL — collapse case (per ENTRY: edge cases collapse to B, \
         not (A) and not (C))."
    };

    let mut s = String::new();
    s.push_str("### Iter-63: target_top3_overlap (cue → canonical target SDR)\n\n");
    s.push_str(&format!(
        "_n_seeds = {n}, threshold = {threshold:.4} (locked in iter-63 ENTRY note before main run)._\n\n"
    ));
    s.push_str("**Per-seed:**\n\n");
    s.push_str("| Seed | untrained | trained | Δ | Δ ≥ threshold |\n");
    s.push_str("| ---: | ---: | ---: | ---: | :---: |\n");
    for (i, &d) in deltas.iter().enumerate() {
        s.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:+.4} | {} |\n",
            untrained.seeds[i],
            untrained.per_seed[i],
            trained.per_seed[i],
            d,
            if d >= threshold { "✓" } else { "✗" },
        ));
    }
    s.push('\n');
    s.push_str(&format!(
        "**Aggregate:** μ_untrained = {:.4} ± {:.4}, μ_trained = {:.4} ± {:.4}, \
         Δ = {:+.4} ± {:.4}, n_pos = {}/{}, n_pass = {}/{}.\n\n",
        untrained.mean, untrained.std, trained.mean, trained.std, mean_d, sd_d, n_pos, n, n_pass, n,
    ));
    s.push_str(&format!(
        "**Paired t(df={df}):** t = {t_stat:+.3}, t_crit(α=0.05) = {t_c05:.3}, \
         t_crit(α=0.15) = {t_c15:.3}. p < 0.05 ⇒ {}, p < 0.15 ⇒ {}.\n\n",
        if p_lt_05 { "✓" } else { "✗" },
        if p_lt_15 { "✓" } else { "✗" },
    ));
    s.push_str(&format!("**Branching verdict:** {branch}\n"));
    s
}

/// Render the per-epoch table as a single Markdown block — useful
/// for pasting into release notes. Includes the iter-46
/// diagnostics (clamp hit rate, prediction-before-teacher,
/// margin, eligibility count, weight stats).
pub fn render_markdown(label: &str, metrics: &[RewardEpochMetrics]) -> String {
    let mut s = String::new();
    s.push_str(&format!("### {label}\n\n"));
    if let Some(first) = metrics.first() {
        s.push_str(&format!(
            "_random top-3 baseline: {:.3}_\n\n",
            first.random_top3_baseline,
        ));
    }
    // Table 1 — outcome / accuracy / weight stats (iter-44…46).
    s.push_str(
        "| Epoch | top-1 | top-3 | MRR | mean rwd | noise t3 | \
         pred-t3 | clamp | margin | elig | w̄ | wmax | dec µs |\n",
    );
    s.push_str(
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    );
    for m in metrics {
        s.push_str(&format!(
            "| {:>3} | {:.2} | {:.2} | {:.2} | {:>+5.2} | {:.2} | {:.2} | {:.2} | {:>+5.2} | {:>5} | {:.2} | {:.2} | {:>5} |\n",
            m.epoch,
            m.top1_accuracy,
            m.top3_accuracy,
            m.mrr,
            m.mean_reward,
            m.noise_top3_rate,
            m.prediction_top3_before_teacher,
            m.target_clamp_hit_rate,
            m.correct_minus_incorrect_margin,
            m.eligibility_nonzero_count,
            m.r2_recurrent_weight_mean,
            m.r2_recurrent_weight_max,
            m.decoder_micros,
        ));
    }
    // Table 2 — iter-47 sparsity diagnostics. Surfaces the
    // forward-drive vs target-population balance per epoch so the
    // [25, 70] acceptance band and the selectivity sign are
    // immediately readable.
    s.push('\n');
    s.push_str(
        "| Epoch | r2_act mean | p10 | p90 | p99 | tgt_hit mean | selectivity | θ_E | θ_I |\n",
    );
    s.push_str("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for m in metrics {
        s.push_str(&format!(
            "| {:>3} | {:>5.1} | {:>4} | {:>4} | {:>4} | {:>5.2} | {:>+6.4} | {:>5.3} | {:>5.3} |\n",
            m.epoch,
            m.r2_active_pre_teacher_mean,
            m.r2_active_pre_teacher_p10,
            m.r2_active_pre_teacher_p90,
            m.r2_active_pre_teacher_p99,
            m.target_hit_pre_teacher_mean,
            m.selectivity_index,
            m.theta_exc_mean,
            m.theta_inh_mean,
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `canonical_target_r2_sdr` must be (a) deterministic given
    /// the same word + salt, (b) constrained to the supplied E-pool,
    /// (c) sorted + deduplicated, (d) of size at most `k`.
    #[test]
    fn canonical_target_r2_sdr_is_stable_and_in_pool() {
        let pool: Vec<usize> = (0..2_000usize).step_by(2).collect(); // 1000 indices
        let a1 = canonical_target_r2_sdr("rust", &pool, 30, 0xCAFE);
        let a2 = canonical_target_r2_sdr("rust", &pool, 30, 0xCAFE);
        assert_eq!(a1, a2, "same word + salt must give the same SDR");
        assert!(!a1.is_empty());
        assert!(a1.len() <= 30);
        // Sorted-unique invariant.
        for w in a1.windows(2) {
            assert!(w[0] < w[1]);
        }
        // Every index must come from the pool.
        let pool_set: BTreeSet<u32> = pool.iter().map(|&i| i as u32).collect();
        for idx in &a1 {
            assert!(pool_set.contains(idx));
        }
        // Different word → different SDR (with overwhelming probability
        // for a 30-of-1000 hash).
        let b = canonical_target_r2_sdr("python", &pool, 30, 0xCAFE);
        assert_ne!(a1, b);
    }

    /// `drive_with_r2_clamp` must actually make the clamped neurons
    /// fire when the strength is high enough — that's the
    /// architectural prerequisite for teacher-forcing to work at all.
    #[test]
    fn drive_with_r2_clamp_makes_target_neurons_fire() {
        let mut brain = fresh_brain(7);
        let r2_e = r2_e_set(&brain);
        let pool = r2_e_pool(&r2_e);
        let target = canonical_target_r2_sdr("zebra", &pool, 16, 0xCAFE);
        // No cue, just the clamp at high strength.
        let counts = drive_with_r2_clamp(&mut brain, &[], &target, 0.0, 500.0, 20.0, &r2_e);
        let target_set: BTreeSet<u32> = target.iter().copied().collect();
        let fired_in_target = target_set
            .iter()
            .filter(|&&i| counts[i as usize] > 0)
            .count();
        assert!(
            fired_in_target as f32 / target.len() as f32 >= 0.5,
            "at least half of the clamped target neurons must fire — \
             clamp is broken if not. fired = {fired_in_target}/{}",
            target.len(),
        );
    }

    /// Smoke test: the harness must run two epochs without panicking
    /// in either configuration, and each epoch metric block must
    /// have well-formed numeric fields.
    #[test]
    fn reward_benchmark_smoke() {
        let corpus = default_corpus();
        // Trim the corpus so the test stays under 60 s in CI.
        let small = RewardCorpus {
            pairs: corpus.pairs.into_iter().take(4).collect(),
            noise_pairs: corpus.noise_pairs.into_iter().take(4).collect(),
            vocab: corpus.vocab,
        };

        for use_reward in [false, true] {
            let cfg = RewardConfig {
                epochs: 2,
                use_reward,
                seed: 42,
                // Tiny reps_per_pair so the smoke test stays fast.
                reps_per_pair: 1,
                teacher: TeacherForcingConfig::off(),
            };
            let metrics = run_reward_benchmark(&small, &cfg);
            assert_eq!(metrics.len(), 2);
            for m in &metrics {
                assert!(m.top1_accuracy >= 0.0 && m.top1_accuracy <= 1.0);
                assert!(m.top3_accuracy >= 0.0 && m.top3_accuracy <= 1.0);
                assert!(m.noise_top3_rate >= 0.0 && m.noise_top3_rate <= 1.0);
                assert!(m.mean_reward.is_finite());
            }
        }
    }

    /// Iter-54: the decorrelated init must produce pairwise-disjoint
    /// R2 reach across the full vocab. This test runs the wiring
    /// against the real default corpus + encoder and lets
    /// `assert_decorrelated_disjoint` panic if the invariant is
    /// violated. It is the unit-test mirror of the runtime
    /// assertion in `run_jaccard_arm` — both must agree.
    #[test]
    fn decorrelated_init_is_pairwise_disjoint() {
        let corpus = default_corpus();
        let vocab_vec: Vec<String> = corpus.vocab.iter().cloned().collect();
        let encoder = TextEncoder::with_stopwords(ENC_N, ENC_K, std::iter::empty::<&str>());
        let mut brain = Brain::new(DT);
        brain.add_region(build_input_region());
        brain.add_region(build_memory_region(43, R2_INH_FRAC, R2_N, R2_P_CONNECT));
        let blocks = wire_forward_decorrelated(&mut brain, &encoder, &vocab_vec, 123, INTER_WEIGHT);
        assert_eq!(blocks.len(), vocab_vec.len());
        // Block-to-block disjointness (precondition).
        for i in 0..blocks.len() {
            for j in (i + 1)..blocks.len() {
                let bi: BTreeSet<usize> = blocks[i].iter().copied().collect();
                let bj: BTreeSet<usize> = blocks[j].iter().copied().collect();
                assert!(
                    bi.is_disjoint(&bj),
                    "block {i} and block {j} overlap: {:?}",
                    bi.intersection(&bj).take(5).collect::<Vec<_>>(),
                );
            }
        }
        // End-to-end reachability disjointness (the real iter-54 invariant).
        assert_decorrelated_disjoint(&brain, &encoder, &vocab_vec);
    }

    // ====================================================================
    // Iter-63 refactor snapshot tests
    //
    // These tests lock the pre-refactor numerics of `run_jaccard_bench`
    // for three configurations that exercise every brain-build branch:
    //   1. vanilla — random R1→R2 wiring, no DG
    //   2. decorrelated — iter-54 disjoint-block R1→R2 wiring, no DG
    //   3. decorrelated + DG + recall — iter-60+ pattern-separation bridge
    //      with iter-62 plasticity-off-during-eval
    //
    // The `build_benchmark_brain` + `disable_all_plasticity` refactor
    // must produce **bit-identical** JaccardMetrics for every snapshot —
    // that is the contract the iter-63 plumbing-fix spec demands.
    // Expected values are captured by the `jaccard_bench_capture_snapshot`
    // harness (run once before the refactor, output pasted into the
    // const block below).
    // ====================================================================

    fn jaccard_bench_snapshot_cfg(decorrelated: bool, dg: bool, recall_mode: bool) -> RewardConfig {
        let teacher = TeacherForcingConfig {
            enabled: true,
            decorrelated_init: decorrelated,
            dg: DgConfig {
                enabled: dg,
                ..DgConfig::default()
            },
            recall_mode_eval: recall_mode,
            target_clamp_strength: 250.0,
            ..TeacherForcingConfig::default()
        };
        RewardConfig {
            epochs: 2,
            use_reward: true,
            seed: 42,
            reps_per_pair: 2,
            teacher,
        }
    }

    fn approx_eq_or_panic(label: &str, expected: f32, actual: f32) {
        // Bit-identity is the contract; `to_bits` so NaN handling
        // does not silently pass.
        assert_eq!(
            expected.to_bits(),
            actual.to_bits(),
            "{label}: expected={expected} ({:#x}) actual={actual} ({:#x})",
            expected.to_bits(),
            actual.to_bits(),
        );
    }

    #[test]
    // (un-ignored: snapshot constants captured pre-refactor)
    fn jaccard_bench_snapshot_vanilla() {
        let cfg = jaccard_bench_snapshot_cfg(false, false, false);
        let corpus = default_corpus();
        let sweep = run_jaccard_bench(&corpus, &cfg, &[42]);
        let u = &sweep.untrained[0].jaccard;
        approx_eq_or_panic(
            "vanilla untrained same_cue_mean",
            __SNAPSHOT_VANILLA_UNTRAINED_SAME,
            u.same_cue_mean,
        );
        approx_eq_or_panic(
            "vanilla untrained cross_cue_mean",
            __SNAPSHOT_VANILLA_UNTRAINED_CROSS,
            u.cross_cue_mean,
        );
        let t = &sweep.trained[0].jaccard;
        approx_eq_or_panic(
            "vanilla trained same_cue_mean",
            __SNAPSHOT_VANILLA_TRAINED_SAME,
            t.same_cue_mean,
        );
        approx_eq_or_panic(
            "vanilla trained cross_cue_mean",
            __SNAPSHOT_VANILLA_TRAINED_CROSS,
            t.cross_cue_mean,
        );
    }

    #[test]
    // (un-ignored: snapshot constants captured pre-refactor)
    fn jaccard_bench_snapshot_decorrelated() {
        let cfg = jaccard_bench_snapshot_cfg(true, false, false);
        let corpus = default_corpus();
        let sweep = run_jaccard_bench(&corpus, &cfg, &[42]);
        let u = &sweep.untrained[0].jaccard;
        approx_eq_or_panic(
            "decorrelated untrained same_cue_mean",
            __SNAPSHOT_DECORRELATED_UNTRAINED_SAME,
            u.same_cue_mean,
        );
        approx_eq_or_panic(
            "decorrelated untrained cross_cue_mean",
            __SNAPSHOT_DECORRELATED_UNTRAINED_CROSS,
            u.cross_cue_mean,
        );
        let t = &sweep.trained[0].jaccard;
        approx_eq_or_panic(
            "decorrelated trained same_cue_mean",
            __SNAPSHOT_DECORRELATED_TRAINED_SAME,
            t.same_cue_mean,
        );
        approx_eq_or_panic(
            "decorrelated trained cross_cue_mean",
            __SNAPSHOT_DECORRELATED_TRAINED_CROSS,
            t.cross_cue_mean,
        );
    }

    #[test]
    // (un-ignored: snapshot constants captured pre-refactor)
    fn jaccard_bench_snapshot_decorrelated_dg_recall() {
        let cfg = jaccard_bench_snapshot_cfg(true, true, true);
        let corpus = default_corpus();
        let sweep = run_jaccard_bench(&corpus, &cfg, &[42]);
        let u = &sweep.untrained[0].jaccard;
        approx_eq_or_panic(
            "dg+recall untrained same_cue_mean",
            __SNAPSHOT_DGRECALL_UNTRAINED_SAME,
            u.same_cue_mean,
        );
        approx_eq_or_panic(
            "dg+recall untrained cross_cue_mean",
            __SNAPSHOT_DGRECALL_UNTRAINED_CROSS,
            u.cross_cue_mean,
        );
        let t = &sweep.trained[0].jaccard;
        approx_eq_or_panic(
            "dg+recall trained same_cue_mean",
            __SNAPSHOT_DGRECALL_TRAINED_SAME,
            t.same_cue_mean,
        );
        approx_eq_or_panic(
            "dg+recall trained cross_cue_mean",
            __SNAPSHOT_DGRECALL_TRAINED_CROSS,
            t.cross_cue_mean,
        );
    }

    // Snapshot constants — captured pre-refactor at seed=42, ep=2,
    // reps=2, vocab=32, target_clamp_strength=250.0. Bit patterns
    // are exact; the bit-identity contract for the iter-63 plumbing
    // refactor demands these values match post-refactor too.
    const __SNAPSHOT_VANILLA_UNTRAINED_SAME: f32 = 1.0; // bits 0x3f800000
    const __SNAPSHOT_VANILLA_UNTRAINED_CROSS: f32 = f32::from_bits(0x3d6e_a884); // 0.058266178
    const __SNAPSHOT_VANILLA_TRAINED_SAME: f32 = 0.890625; // bits 0x3f640000
    const __SNAPSHOT_VANILLA_TRAINED_CROSS: f32 = f32::from_bits(0x3d62_456a); // 0.05524198
    const __SNAPSHOT_DECORRELATED_UNTRAINED_SAME: f32 = 1.0;
    const __SNAPSHOT_DECORRELATED_UNTRAINED_CROSS: f32 = f32::from_bits(0x3eef_46fb); // 0.4673384
    const __SNAPSHOT_DECORRELATED_TRAINED_SAME: f32 = 1.0;
    const __SNAPSHOT_DECORRELATED_TRAINED_CROSS: f32 = f32::from_bits(0x3eef_46fb); // 0.4673384
    const __SNAPSHOT_DGRECALL_UNTRAINED_SAME: f32 = 1.0;
    const __SNAPSHOT_DGRECALL_UNTRAINED_CROSS: f32 = f32::from_bits(0x3d7a_3839); // 0.061088774
    const __SNAPSHOT_DGRECALL_TRAINED_SAME: f32 = 1.0;
    const __SNAPSHOT_DGRECALL_TRAINED_CROSS: f32 = f32::from_bits(0x3d49_7f35); // 0.04919358

    /// One-shot helper to capture the actual values for the snapshot
    /// constants. Run with
    /// `cargo test -p eval --release jaccard_bench_capture_snapshot
    /// -- --nocapture --ignored`, paste the printed values into the
    /// constants above, then remove `#[ignore]` from the snapshot
    /// tests.
    #[test]
    #[ignore = "manual capture harness, not a regression test"]
    fn jaccard_bench_capture_snapshot() {
        let corpus = default_corpus();
        for (label, decorrelated, dg, recall) in [
            ("vanilla", false, false, false),
            ("decorrelated", true, false, false),
            ("dg_recall", true, true, true),
        ] {
            let cfg = jaccard_bench_snapshot_cfg(decorrelated, dg, recall);
            let sweep = run_jaccard_bench(&corpus, &cfg, &[42]);
            let u = &sweep.untrained[0].jaccard;
            let t = &sweep.trained[0].jaccard;
            eprintln!(
                "[snapshot {label}] untrained: same=f32::from_bits({:#x})_/* {} */ cross=f32::from_bits({:#x})_/* {} */",
                u.same_cue_mean.to_bits(),
                u.same_cue_mean,
                u.cross_cue_mean.to_bits(),
                u.cross_cue_mean,
            );
            eprintln!(
                "[snapshot {label}] trained:   same=f32::from_bits({:#x})_/* {} */ cross=f32::from_bits({:#x})_/* {} */",
                t.same_cue_mean.to_bits(),
                t.same_cue_mean,
                t.cross_cue_mean.to_bits(),
                t.cross_cue_mean,
            );
        }
    }

    // ====================================================================
    // Iter-63 plumbing-fix: shared helper unit tests.
    // ====================================================================

    /// `build_benchmark_brain` must be deterministic: identical
    /// `(cfg, seed)` produces identical brain structure (region
    /// count, R2-E indices, dg_sdr_map keys, target_r2_map keys).
    /// Catches any non-determinism introduced by future refactors.
    #[test]
    fn build_benchmark_brain_is_deterministic() {
        let cfg = jaccard_bench_snapshot_cfg(true, true, true);
        let corpus = default_corpus();
        let a = build_benchmark_brain(&corpus, &cfg);
        let b = build_benchmark_brain(&corpus, &cfg);
        assert_eq!(a.brain.regions.len(), b.brain.regions.len());
        assert_eq!(a.r2_e, b.r2_e);
        assert_eq!(a.r2_n_used, b.r2_n_used);
        assert_eq!(a.decorrelated_block_size, b.decorrelated_block_size);
        for word in &corpus.vocab {
            assert_eq!(
                a.target_r2_map.get(word),
                b.target_r2_map.get(word),
                "target_r2_map differs for {word}"
            );
            assert_eq!(
                a.dg_sdr_map.get(word),
                b.dg_sdr_map.get(word),
                "dg_sdr_map differs for {word}"
            );
        }
        // Synapse L2 norms must match per region.
        let la = brain_synapse_l2_norms(&a.brain);
        let lb = brain_synapse_l2_norms(&b.brain);
        assert_eq!(la, lb, "synapse L2 norms differ between identical builds");
    }

    /// `build_benchmark_brain` must honour the iter-54 / iter-60
    /// flags. Vanilla → 2 regions (R1 + R2), no DG. Decorrelated →
    /// `Some(block_size)`. DG → 3 regions. Catches the iter-63 v1
    /// bug where these flags were silently ignored.
    #[test]
    fn build_benchmark_brain_honours_dg_and_decorrelated_flags() {
        let corpus = default_corpus();

        let vanilla =
            build_benchmark_brain(&corpus, &jaccard_bench_snapshot_cfg(false, false, false));
        assert_eq!(vanilla.brain.regions.len(), 2);
        assert!(vanilla.decorrelated_block_size.is_none());
        assert!(vanilla.dg_sdr_map.is_empty());

        let dec = build_benchmark_brain(&corpus, &jaccard_bench_snapshot_cfg(true, false, false));
        assert_eq!(dec.brain.regions.len(), 2);
        assert!(
            dec.decorrelated_block_size.is_some(),
            "decorrelated_init=true must yield Some(block_size)"
        );
        assert!(dec.dg_sdr_map.is_empty());

        let dg = build_benchmark_brain(&corpus, &jaccard_bench_snapshot_cfg(true, true, true));
        assert_eq!(
            dg.brain.regions.len(),
            3,
            "dg.enabled=true must add a DG region"
        );
        assert_eq!(
            dg.dg_sdr_map.len(),
            corpus.vocab.len(),
            "dg_sdr_map must have one entry per vocab word"
        );
    }

    /// `disable_all_plasticity` must clear every plasticity-rule
    /// `Option<…>` field in the target region's `Network`. Catches
    /// any future plasticity mechanism added to `snn-core` that the
    /// disable helper forgets to gate.
    #[test]
    fn disable_all_plasticity_clears_every_rule() {
        let cfg = jaccard_bench_snapshot_cfg(false, false, false);
        let corpus = default_corpus();
        let mut bb = build_benchmark_brain(&corpus, &cfg);
        // Enable everything we have control over, so the disable
        // call has something to turn off.
        bb.brain.regions[1].network.enable_stdp(stdp());
        bb.brain.regions[1]
            .network
            .enable_istdp(istdp_iter46_baseline());
        bb.brain.regions[1]
            .network
            .enable_homeostasis(homeostasis());
        bb.brain.regions[1]
            .network
            .enable_intrinsic_plasticity(intrinsic());
        bb.brain.regions[1]
            .network
            .enable_reward_learning(reward_params());
        let net_before = &bb.brain.regions[1].network;
        assert!(net_before.stdp.is_some());
        assert!(net_before.istdp.is_some());
        assert!(net_before.homeostasis.is_some());
        assert!(net_before.intrinsic.is_some());
        assert!(net_before.reward.is_some());

        disable_all_plasticity(&mut bb.brain, 1);

        let net = &bb.brain.regions[1].network;
        assert!(net.stdp.is_none(), "stdp not disabled");
        assert!(net.istdp.is_none(), "istdp not disabled");
        assert!(net.homeostasis.is_none(), "homeostasis not disabled");
        assert!(net.intrinsic.is_none(), "intrinsic not disabled");
        assert!(net.reward.is_none(), "reward learning not disabled");
        assert!(net.metaplasticity.is_none(), "metaplasticity not disabled");
        assert!(net.heterosynaptic.is_none(), "heterosynaptic not disabled");
        assert!(
            net.structural.is_none(),
            "structural plasticity not disabled"
        );
    }

    /// `run_target_overlap_arm` in untrained mode must complete the
    /// full training-loop schedule (epochs > 0, no_plasticity = true)
    /// without violating the iter-52 weight-invariant. This is the
    /// regression test for the bug iter-63 v1 had: silent plasticity
    /// rules (metaplasticity / heterosynaptic / structural) leaked
    /// through `run_reward_benchmark`'s 5-rule disable gate.
    #[test]
    fn target_overlap_arm_untrained_passes_iter52_invariant() {
        let mut cfg = jaccard_bench_snapshot_cfg(true, true, true);
        cfg.epochs = 2;
        cfg.teacher.no_plasticity = true;
        cfg.use_reward = false;
        let corpus = default_corpus();
        // The function asserts no_weight_drift internally; if any
        // rule leaks through disable_all_plasticity, this panics.
        let result = run_target_overlap_arm(&corpus, &cfg, &[42], ArmMode::Untrained);
        assert_eq!(result.per_seed.len(), 1);
        assert!(result.mean.is_finite(), "mean must be finite");
        assert!(
            result.mean >= 0.0 && result.mean <= 1.0,
            "mean target_top3_overlap out of range: {}",
            result.mean
        );
    }
}
