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
    Brain, HomeostasisParams, IStdpParams, LifNeuron, LifParams, NeuronKind, Region, RewardParams,
    Rng, StdpParams,
};

/// Topology — a fast 2 000-neuron R2 keeps the benchmark wall-time
/// under a minute per epoch on a release build, while still being
/// representative of the production R2.
const DT: f32 = 0.1;
const R1_N: usize = 1000;
const R2_N: usize = 2_000;
const R2_INH_FRAC: f32 = 0.20;
const R2_P_CONNECT: f32 = 0.05;
const FAN_OUT: usize = 12;
const INTER_WEIGHT: f32 = 2.0;
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
    /// arm. See [`RewardConfig::debug_trials`].
    pub debug_trials: u32,
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

fn build_memory_region(seed: u64) -> Region {
    let mut rng = Rng::new(seed);
    let mut region = Region::new("R2", DT);
    let net = &mut region.network;
    let n_inh = (R2_N as f32 * R2_INH_FRAC) as usize;
    let n_exc = R2_N - n_inh;
    for _ in 0..n_exc {
        net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    }
    for _ in 0..n_inh {
        net.add_neuron(LifNeuron::inhibitory(LifParams::default()));
    }
    let g_exc = 0.20_f32;
    let g_inh = 0.80_f32;
    for pre in 0..R2_N {
        let g = match net.neurons[pre].kind {
            NeuronKind::Excitatory => g_exc,
            NeuronKind::Inhibitory => g_inh,
        };
        for post in 0..R2_N {
            if pre == post {
                continue;
            }
            if rng.bernoulli(R2_P_CONNECT) {
                let w = rng.range_f32(0.5 * g, 1.0 * g);
                net.connect(pre, post, w);
            }
        }
    }
    region
}

fn wire_forward(brain: &mut Brain, seed: u64) {
    let mut rng = Rng::new(seed);
    let r2_size = brain.regions[1].num_neurons();
    for src in 0..R1_N {
        for _ in 0..FAN_OUT {
            let dst = (rng.next_u64() as usize) % r2_size;
            brain.connect(0, src, 1, dst, INTER_WEIGHT, INTER_DELAY_MS);
        }
    }
}

fn fresh_brain(seed: u64) -> Brain {
    let mut brain = Brain::new(DT);
    brain.add_region(build_input_region());
    brain.add_region(build_memory_region(seed.wrapping_add(1)));
    wire_forward(&mut brain, seed.wrapping_add(2));
    brain
}

fn stdp() -> StdpParams {
    StdpParams {
        a_plus: 0.020,
        a_minus: 0.015,
        w_max: 0.8,
        ..StdpParams::default()
    }
}

fn istdp() -> IStdpParams {
    IStdpParams {
        a_plus: 0.10,
        a_minus: 1.10,
        tau_minus: 30.0,
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
    let mut ext1 = vec![0.0_f32; R1_N];
    for &idx in sdr_indices {
        if (idx as usize) < R1_N {
            ext1[idx as usize] = DRIVE_NA;
        }
    }
    let ext2 = vec![0.0_f32; R2_N];
    let externals = vec![ext1, ext2];
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
    let ext2 = vec![0.0_f32; R2_N];
    let externals = vec![ext1, ext2];
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
    let zeros = vec![vec![0.0_f32; R1_N], vec![0.0_f32; R2_N]];
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
    let mut ext2 = vec![0.0_f32; R2_N];
    if r2_strength != 0.0 {
        for &idx in r2_target_indices {
            if (idx as usize) < R2_N {
                ext2[idx as usize] = r2_strength;
            }
        }
    }
    let externals = vec![ext1, ext2];
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

fn shuffle<T>(rng: &mut Rng, v: &mut [T]) {
    // Fisher-Yates using the deterministic in-house RNG.
    for i in (1..v.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        v.swap(i, j);
    }
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
) -> TrialOutcome {
    let mut outcome = TrialOutcome::default();

    // ---- Phase 1: cue alone (R1 forward path).
    drive_for(brain, cue_sdr, cfg.cue_ms as f32);

    // ---- Phase 2: delay (no input).
    idle(brain, cfg.delay_ms as f32);

    // ---- Phase 3: prediction. Plasticity off so the test does not
    //      mutate weights. We toggle STDP / iSTDP off, then back on
    //      after the prediction window. R-STDP eligibility decay
    //      continues normally — that's fine, it's the *gated*
    //      update we want to keep silenced (modulator stays 0).
    if !cfg.plasticity_during_prediction {
        brain.regions[1].network.disable_stdp();
        brain.regions[1].network.disable_istdp();
    }
    let pred_counts = drive_for_with_counts(
        brain,
        cue_sdr,
        cfg.prediction_ms as f32,
        r2_e,
    );
    if !cfg.plasticity_during_prediction {
        brain.regions[1].network.enable_stdp(rest_stdp);
        brain.regions[1].network.enable_istdp(rest_istdp);
    }
    let pred_topk = top_k_indices(&pred_counts, cfg.wta_k.max(1));
    outcome.prediction_topk = pred_topk.clone();
    outcome.prediction_active_count =
        pred_counts.iter().filter(|&&c| c > 0).count() as u32;

    // ---- Phase 4: teacher. Cue + R2 target clamp. Plasticity ON
    //      (it is by default; this matches the iter-45 reps loop).
    if !cfg.plasticity_during_teacher {
        brain.regions[1].network.disable_stdp();
        brain.regions[1].network.disable_istdp();
    }
    let teacher_counts = drive_with_r2_clamp(
        brain,
        cue_sdr,
        target_r2_sdr,
        DRIVE_NA,
        cfg.target_clamp_strength,
        cfg.teacher_ms as f32,
        r2_e,
    );
    if !cfg.plasticity_during_teacher {
        brain.regions[1].network.enable_stdp(rest_stdp);
        brain.regions[1].network.enable_istdp(rest_istdp);
    }
    // Diagnostic: how reliably did the clamp drive the intended set?
    let target_set: BTreeSet<u32> = target_r2_sdr.iter().copied().collect();
    let clamp_hit = target_set
        .iter()
        .filter(|&&i| teacher_counts[i as usize] > 0)
        .count();
    outcome.target_clamp_hits = clamp_hit as u32;
    outcome.target_clamp_size = target_r2_sdr.len() as u32;

    // ---- Phase 5: reward. Score the *prediction* against the
    //      canonical target — teacher activation does NOT count.
    let target_in_topk = pred_topk
        .iter()
        .any(|i| target_set.contains(i));
    let any_wrong_topk = pred_topk
        .iter()
        .any(|i| !target_set.contains(i));
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
        drive_for(brain, cue_sdr, cfg.teacher_ms as f32);
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
}

// ----------------------------------------------------------------------
// The harness. Three phases per trial: paired training, evaluation,
// optional reward delivery.
// ----------------------------------------------------------------------

/// Run the pair-association benchmark for `cfg.epochs` epochs and
/// return per-epoch metrics. The harness builds its own `Brain` and
/// dictionary so the call is self-contained.
pub fn run_reward_benchmark(corpus: &RewardCorpus, cfg: &RewardConfig) -> Vec<RewardEpochMetrics> {
    let mut brain = fresh_brain(cfg.seed);
    let encoder = TextEncoder::with_stopwords(ENC_N, ENC_K, std::iter::empty::<&str>());
    let r2_e = r2_e_set(&brain);

    // Plasticity setup. `enable_reward_learning` allocates the per-
    // synapse eligibility tag; without it the modulator setter is a
    // no-op even if the caller flips it on.
    let stdp_params = stdp();
    let istdp_params = istdp();
    brain.regions[1].network.enable_stdp(stdp_params);
    brain.regions[1].network.enable_istdp(istdp_params);
    brain.regions[1].network.enable_homeostasis(homeostasis());
    if cfg.use_reward {
        brain.regions[1]
            .network
            .enable_reward_learning(reward_params());
    }

    // Iter-46: pre-compute the canonical target-R2 SDR for every
    // word in the corpus. The mapping is stable per (word, salt)
    // and never recomputed per trial.
    let teacher_active = cfg.teacher.enabled;
    let target_r2_map: std::collections::HashMap<String, Vec<u32>> = if teacher_active {
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
    } else {
        std::collections::HashMap::new()
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
                        let target_set: BTreeSet<u32> =
                            canonical.iter().copied().collect();
                        if outcome
                            .prediction_topk
                            .iter()
                            .any(|i| target_set.contains(i))
                        {
                            prediction_top3_hits += 1;
                        }
                        prediction_top3_total += 1;
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
                        let counts = drive_for_with_counts(
                            &mut brain,
                            &cue_sdr.indices,
                            RECALL_MS,
                            &r2_e,
                        );
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
                        let overlap =
                            kwta.iter().filter(|i| target_kwta.contains(i)).count();
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
        // training.
        let saved_modulator = brain.regions[1].network.neuromodulator;
        brain.regions[1].network.disable_stdp();
        brain.regions[1].network.disable_istdp();
        brain.set_neuromodulator(0.0);

        let dec_t0 = std::time::Instant::now();
        let dict = build_vocab_dictionary(&mut brain, &encoder, &r2_e, &vocab);
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
            let counts =
                drive_for_with_counts(&mut brain, &cue_sdr.indices, RECALL_MS, &r2_e);
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
                    let target_set: BTreeSet<u32> =
                        canonical.iter().copied().collect();
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

        // Restore plasticity for the next epoch.
        brain.regions[1].network.enable_stdp(stdp_params);
        brain.regions[1].network.enable_istdp(istdp_params);
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
        });
    }

    metrics
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
) -> EngramDictionary {
    let mut dict = EngramDictionary::new();
    for word in vocab {
        let sdr = encoder.encode_word(word);
        if sdr.indices.is_empty() {
            continue;
        }
        brain.regions[1].network.reset_state();
        let cs = drive_for_with_counts(brain, &sdr.indices, RECALL_MS, r2_e);
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
        let counts = drive_with_r2_clamp(
            &mut brain,
            &[],
            &target,
            0.0,
            500.0,
            20.0,
            &r2_e,
        );
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
}
