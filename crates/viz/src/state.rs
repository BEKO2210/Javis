//! Persistent application state.
//!
//! A single brain lives in the server for the lifetime of the process.
//! Train operations append to it, recall operations read from it. The
//! whole thing is wrapped in a `tokio::Mutex` so concurrent requests
//! are serialised cleanly — typical use is one human at a time.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use encoders::{EngramDictionary, TextEncoder};
use snn_core::{
    Brain, BrainState, HomeostasisParams, IStdpParams, LifNeuron, LifParams, NeuronKind, Region,
    Rng, StdpParams,
};
use tokio::sync::{mpsc::Sender, RwLock, Semaphore};
use tracing::{debug, info, warn};

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::events::{DecodedWord, Event, LlmReply};
use llm::LlmClient;

// ----------------------------------------------------------------------
// Topology + plasticity (sweet spot from notes/11–12)
// ----------------------------------------------------------------------

pub const DT: f32 = 0.1;
pub const R1_N: usize = 1000;
/// R2 layer size — bumped from 2000 (iter ≤24) to 10 000 (iter 25)
/// to give engrams a much larger orthogonal space. See notes/43 for
/// the cross-bleed problem this addresses.
pub const R2_N: usize = 10_000;
pub const R2_INH_FRAC: f32 = 0.20;
/// Sparser recurrent connectivity (was 0.10). Keeps the synapse
/// count under control at the new R2 size: 10_000² × 0.03 = 3 M
/// synapses, vs 10_000² × 0.10 = 10 M which would dominate memory
/// and snapshot file size.
pub const R2_P_CONNECT: f32 = 0.03;
/// Forward fan-out R1 → R2. Bumped from 10 (iter ≤24) so the input
/// drive still excites a ≥ 1 % slice of the new R2.
pub const FAN_OUT: usize = 30;
pub const INTER_WEIGHT: f32 = 2.0;
pub const INTER_DELAY_MS: f32 = 2.0;
pub const ENC_N: u32 = R1_N as u32;
pub const ENC_K: u32 = 20;
pub const DRIVE_NA: f32 = 200.0;
pub const TRAINING_MS: f32 = 150.0;
pub const COOLDOWN_MS: f32 = 50.0;
pub const RECALL_MS: f32 = 30.0;
/// kWTA cap for engram fingerprints. 100 / 10 000 = 1 % sparsity,
/// down from 11 % (220 / 2000) — engrams are now mathematically
/// far less likely to overlap by chance, which is the whole point
/// of this iteration.
pub const KWTA_K: usize = 100;
pub const DECODE_THRESHOLD: f32 = 0.50;

const BATCH_MS: f32 = 1.0;

const STOPWORDS: &[&str] = &[
    "is", "a", "the", "an", "on", "at", "of", "in", "to", "and", "or", "for", "with", "by", "from",
    "but", "as", "it", "its", "this", "that", "these", "those", "be", "are", "was", "were", "like",
];

pub fn default_corpus() -> Vec<&'static str> {
    vec![
        "Rust is a systems programming language focused on memory safety \
         and ownership; the borrow checker prevents data races at compile time.",
        "Python dominates data science with libraries like numpy and pandas; \
         dynamic typing makes prototyping fast at the cost of runtime speed.",
        "Cpp gives raw control over memory through pointers and manual \
         allocation; templates and zero cost abstractions enable performance.",
    ]
}

fn stdp() -> StdpParams {
    StdpParams {
        a_plus: 0.015,
        a_minus: 0.012,
        w_max: 0.8,
        ..StdpParams::default()
    }
}

fn istdp() -> IStdpParams {
    // Aggressive LTD on co-active E-targets: with R2 = 10 000 and
    // K = 100 the inhibitory layer has to silence a much larger sea
    // of neurons that should *not* be part of the engram. a_minus
    // bumped from 0.55 (iter ≤24) to 1.10; a_plus also raised so
    // pairs of (silent E, firing I) carve baseline-suppressing
    // weights faster.
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

// ----------------------------------------------------------------------
// Brain construction
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

fn fresh_brain() -> Brain {
    let mut brain = Brain::new(DT);
    brain.add_region(build_input_region());
    brain.add_region(build_memory_region(2027));
    wire_forward(&mut brain, 2028);
    brain
}

fn fresh_encoder() -> TextEncoder {
    TextEncoder::with_stopwords(ENC_N, ENC_K, STOPWORDS.iter().copied())
}

fn r2_excitatory_indices(brain: &Brain) -> HashSet<usize> {
    brain.regions[1]
        .network
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.kind == NeuronKind::Excitatory)
        .map(|(i, _)| i)
        .collect()
}

fn count_neurons(brain: &Brain) -> (u32, u32, u32) {
    let r1 = brain.regions[0].num_neurons() as u32;
    let r2_e = brain.regions[1]
        .network
        .neurons
        .iter()
        .filter(|n| n.kind == NeuronKind::Excitatory)
        .count() as u32;
    let r2_i = brain.regions[1].num_neurons() as u32 - r2_e;
    (r1, r2_e, r2_i)
}

// ----------------------------------------------------------------------
// Driver primitives
// ----------------------------------------------------------------------

/// Drive a cue, stream batched spike events, *and* return per-R2-E
/// spike counts so the caller can run kWTA without simulating twice.
async fn run_with_cue_streaming(
    brain: &mut Brain,
    drive: &[u32],
    duration_ms: f32,
    tx: &Sender<Event>,
) -> Vec<u32> {
    let r2_e_set = r2_excitatory_indices(brain);
    let r2_size = brain.regions[1].num_neurons();
    let mut counts = vec![0u32; r2_size];

    let mut ext1 = vec![0.0_f32; R1_N];
    for &idx in drive {
        if (idx as usize) < R1_N {
            ext1[idx as usize] = DRIVE_NA;
        }
    }
    let ext2 = vec![0.0_f32; R2_N];
    let externals = vec![ext1, ext2];

    let total_steps = (duration_ms / DT) as usize;
    let batch_steps = (BATCH_MS / DT) as usize;
    let mut batch_r1: Vec<u32> = Vec::new();
    let mut batch_r2: Vec<u32> = Vec::new();
    let mut t_ms: f32 = 0.0;

    for step in 0..total_steps {
        let spikes = brain.step(&externals);
        for &id in &spikes[0] {
            batch_r1.push(id as u32);
        }
        for &id in &spikes[1] {
            batch_r2.push(id as u32);
            if r2_e_set.contains(&id) {
                counts[id] += 1;
            }
        }
        if (step + 1) % batch_steps == 0 || step + 1 == total_steps {
            t_ms += BATCH_MS;
            let _ = tx
                .send(Event::Step {
                    t_ms,
                    r1: std::mem::take(&mut batch_r1),
                    r2: std::mem::take(&mut batch_r2),
                })
                .await;
        }
    }
    counts
}

/// Read-only variant of [`run_with_cue_streaming`] for the recall
/// path. Same dynamics (plasticity is off either way during recall),
/// but the brain is read-only — every concurrent caller drives its
/// own [`BrainState`] against the same shared `Brain`. This is what
/// frees the recall path from the `RwLock` write-side bottleneck.
async fn run_with_cue_streaming_immutable(
    brain: &Brain,
    state: &mut BrainState,
    drive: &[u32],
    duration_ms: f32,
    tx: &Sender<Event>,
) -> Vec<u32> {
    let r2_e_set = r2_excitatory_indices(brain);
    let r2_size = brain.regions[1].num_neurons();
    let mut counts = vec![0u32; r2_size];

    let mut ext1 = vec![0.0_f32; R1_N];
    for &idx in drive {
        if (idx as usize) < R1_N {
            ext1[idx as usize] = DRIVE_NA;
        }
    }
    let ext2 = vec![0.0_f32; R2_N];
    let externals = vec![ext1, ext2];

    let total_steps = (duration_ms / DT) as usize;
    let batch_steps = (BATCH_MS / DT) as usize;
    let mut batch_r1: Vec<u32> = Vec::new();
    let mut batch_r2: Vec<u32> = Vec::new();
    let mut t_ms: f32 = 0.0;

    // Sub-phase timers: distinguish raw brain compute from the
    // WS-channel send overhead that interleaves with it. Reported via
    // `javis_recall_subphase_seconds{phase}` so the existing pipeline
    // profile script can break the SNN-compute number down further.
    let mut compute_s: f64 = 0.0;
    let mut stream_s: f64 = 0.0;
    let mut dropped_step_events: u64 = 0;

    for step in 0..total_steps {
        let t0 = Instant::now();
        let spikes = brain.step_immutable(state, &externals);
        for &id in &spikes[0] {
            batch_r1.push(id as u32);
        }
        for &id in &spikes[1] {
            batch_r2.push(id as u32);
            if r2_e_set.contains(&id) {
                counts[id] += 1;
            }
        }
        compute_s += t0.elapsed().as_secs_f64();

        if (step + 1) % batch_steps == 0 || step + 1 == total_steps {
            t_ms += BATCH_MS;
            let t1 = Instant::now();
            // Fire-and-forget: `Event::Step` is a visualisation
            // breadcrumb. If the WS consumer is behind we drop the
            // event rather than awaiting backpressure into the
            // simulation loop. The non-async `try_send` removes the
            // tokio yield that was costing ~6.5 % of the recall
            // pipeline (notes/40); the trade-off is that a slow
            // browser may miss step batches, which the frontend
            // handles gracefully because each batch is independent.
            match tx.try_send(Event::Step {
                t_ms,
                r1: std::mem::take(&mut batch_r1),
                r2: std::mem::take(&mut batch_r2),
            }) {
                Ok(()) => {}
                Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                    dropped_step_events += 1;
                }
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => break,
            }
            stream_s += t1.elapsed().as_secs_f64();
        }
    }

    metrics::histogram!("javis_recall_subphase_seconds", "phase" => "brain_compute")
        .record(compute_s);
    metrics::histogram!("javis_recall_subphase_seconds", "phase" => "ws_stream").record(stream_s);
    if dropped_step_events > 0 {
        metrics::counter!("javis_ws_step_dropped_total").increment(dropped_step_events);
    }

    counts
}

fn run_with_cue_counts(brain: &mut Brain, drive: &[u32], duration_ms: f32) -> Vec<u32> {
    let r2_e_set = r2_excitatory_indices(brain);
    let r2_size = brain.regions[1].num_neurons();
    let mut counts = vec![0u32; r2_size];

    let mut ext1 = vec![0.0_f32; R1_N];
    for &idx in drive {
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

fn top_k(counts: &[u32], k: usize) -> Vec<u32> {
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

// ----------------------------------------------------------------------
// AppState: the shared, persistent brain
// ----------------------------------------------------------------------

pub struct AppState {
    /// Inner state behind a `RwLock`: train / reset / load take the
    /// write lock; recall, stats, snapshot save take the read lock
    /// so they run concurrently. The plasticity-free recall path uses
    /// [`snn_core::Brain::step_immutable`] with a per-task
    /// [`BrainState`] so multiple readers can drive simulations on
    /// the same shared `Brain` without contention.
    inner: Arc<RwLock<Inner>>,
    llm: Arc<LlmClient>,
    /// Bounded WebSocket-session counter. Each session acquires one
    /// permit on accept and releases it on close. When the cap is
    /// reached the upgrade handler rejects new connections with
    /// `503 Service Unavailable + Retry-After`. Configurable via
    /// `JAVIS_MAX_CONCURRENT_SESSIONS` (default 32).
    sessions: Arc<Semaphore>,
}

/// Default ceiling on simultaneous WebSocket sessions. Chosen by
/// reading `notes/35-load-test.md`: throughput plateaus at ~141 ops/s
/// regardless of concurrency, but p99 latency stays under ~100 ms only
/// up to roughly 10 parallel clients. 32 leaves headroom for short
/// bursts while still bounding worst-case queue depth.
pub const DEFAULT_MAX_CONCURRENT_SESSIONS: usize = 32;

/// Resolve the session cap from the `JAVIS_MAX_CONCURRENT_SESSIONS`
/// env var. Falls back to [`DEFAULT_MAX_CONCURRENT_SESSIONS`] if unset
/// or unparseable; logs a warning on bad input rather than panicking
/// because a server should not refuse to start over an env-var typo.
fn resolve_session_cap() -> usize {
    match std::env::var("JAVIS_MAX_CONCURRENT_SESSIONS") {
        Err(_) => DEFAULT_MAX_CONCURRENT_SESSIONS,
        Ok(s) => match s.parse::<usize>() {
            Ok(n) if n > 0 => n,
            _ => {
                tracing::warn!(
                    value = %s,
                    default = DEFAULT_MAX_CONCURRENT_SESSIONS,
                    "JAVIS_MAX_CONCURRENT_SESSIONS not a positive integer; using default",
                );
                DEFAULT_MAX_CONCURRENT_SESSIONS
            }
        },
    }
}

struct Inner {
    brain: Brain,
    dict: EngramDictionary,
    encoder: TextEncoder,
    known_words: HashSet<String>,
    trained_sentences: Vec<String>,
}

/// On-disk wire format for an Javis snapshot. Matches `Inner` plus a
/// small block of metadata so a future operator can tell when and by
/// which build a snapshot was written. The schema is explicitly
/// versioned; older formats are upgraded through the migration chain
/// in [`migrate_snapshot`] before deserialisation.
#[derive(Serialize, Deserialize)]
struct Snapshot {
    /// Bumped any time the schema becomes incompatible. Loading a
    /// snapshot with `version != SNAPSHOT_VERSION` triggers the
    /// migration chain; loading a snapshot with `version >
    /// SNAPSHOT_VERSION` is a hard error (we cannot downgrade).
    version: u32,
    /// Provenance information; introduced in v2.
    metadata: SnapshotMetadata,
    brain: Brain,
    dict: EngramDictionary,
    encoder: TextEncoder,
    known_words: HashSet<String>,
    trained_sentences: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    /// UNIX seconds at which the snapshot was first written; 0 means
    /// the snapshot was migrated from a pre-v2 file that did not
    /// record this.
    pub created_at_unix: u64,
    /// Free-form identifier of the Javis build that wrote the
    /// snapshot. For migrated snapshots this is the literal string
    /// `"migrated-from-v1"`.
    pub javis_version: String,
}

/// Current schema version. Incremented in lock-step with the
/// `migrate_snapshot` table in this module.
pub const SNAPSHOT_VERSION: u32 = 2;

/// Type-erased migration step. Takes a `Value` describing a snapshot
/// at version `N` and returns the same logical snapshot at version
/// `N+1`. Failing with `String` keeps error reporting cheap; the
/// caller wraps it in an `io::Error` for the load API.
type MigrationFn = fn(serde_json::Value) -> Result<serde_json::Value, String>;

/// Migration registry. Each entry is `(from_version, fn)`. The chain
/// is walked top-to-bottom by [`migrate_snapshot`], applying each
/// step exactly once until the value is at [`SNAPSHOT_VERSION`]. To
/// add a v3 schema later, append `(2, migrate_v2_to_v3)` and bump
/// `SNAPSHOT_VERSION` to 3.
const MIGRATIONS: &[(u32, MigrationFn)] = &[(1, migrate_v1_to_v2)];

/// Walk the migration chain on `value` from `from_version` up to
/// [`SNAPSHOT_VERSION`]. Returns the migrated value or a string
/// describing the first failure.
fn migrate_snapshot(
    mut value: serde_json::Value,
    from_version: u32,
) -> Result<serde_json::Value, String> {
    if from_version > SNAPSHOT_VERSION {
        return Err(format!(
            "snapshot version {from_version} is newer than this build supports \
             (max {SNAPSHOT_VERSION}); refusing to downgrade",
        ));
    }
    let mut current = from_version;
    while current < SNAPSHOT_VERSION {
        let step = MIGRATIONS
            .iter()
            .find(|(from, _)| *from == current)
            .map(|(_, f)| f)
            .ok_or_else(|| format!("no migration registered for version {current}"))?;
        value = step(value)?;
        current += 1;
        // Bump the embedded version so the post-loop deserialise sees
        // a self-consistent document.
        if let serde_json::Value::Object(ref mut map) = value {
            map.insert("version".into(), serde_json::Value::from(current));
        }
    }
    Ok(value)
}

/// v1 → v2: introduce the mandatory `metadata` block.
///
/// v1 had no metadata; we backfill `created_at_unix = 0` and
/// `javis_version = "migrated-from-v1"` so an operator can spot in
/// `/ready` (or by inspecting the file) that this snapshot did not
/// originate from a v2-or-later writer.
fn migrate_v1_to_v2(mut value: serde_json::Value) -> Result<serde_json::Value, String> {
    let map = value
        .as_object_mut()
        .ok_or_else(|| "v1 snapshot root is not a JSON object".to_string())?;
    map.insert(
        "metadata".into(),
        serde_json::json!({
            "created_at_unix": 0,
            "javis_version": "migrated-from-v1",
        }),
    );
    Ok(value)
}

impl AppState {
    pub fn new() -> Self {
        Self::with_session_cap(resolve_session_cap())
    }

    /// Like [`AppState::new`] but with an explicit cap. Useful for
    /// integration tests that want to exercise the rejection path.
    pub fn with_session_cap(cap: usize) -> Self {
        let inner = Inner {
            brain: fresh_brain(),
            dict: EngramDictionary::new(),
            encoder: fresh_encoder(),
            known_words: HashSet::new(),
            trained_sentences: Vec::new(),
        };
        Self {
            inner: Arc::new(RwLock::new(inner)),
            llm: Arc::new(LlmClient::from_env()),
            sessions: Arc::new(Semaphore::new(cap)),
        }
    }

    /// Same brain, but with the LLM forced into mock mode — useful for
    /// tests and offline demos.
    pub fn new_with_mock_llm() -> Self {
        let mut s = Self::new();
        s.llm = Arc::new(LlmClient::mock());
        s
    }

    /// Hand out a clone of the session-permit semaphore so the
    /// HTTP-layer can `try_acquire_owned()` before upgrading the
    /// WebSocket. Returns the same `Arc` every call.
    pub fn sessions(&self) -> Arc<Semaphore> {
        Arc::clone(&self.sessions)
    }

    pub fn handle(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            llm: Arc::clone(&self.llm),
            sessions: Arc::clone(&self.sessions),
        }
    }

    pub fn llm_is_real(&self) -> bool {
        self.llm.is_real()
    }

    /// Reset to a freshly-initialised brain. Loses every learnt engram.
    pub async fn reset(&self) {
        let mut g = self.inner.write().await;
        let dropped_sentences = g.trained_sentences.len();
        let dropped_words = g.known_words.len();
        g.brain = fresh_brain();
        g.dict = EngramDictionary::new();
        g.known_words.clear();
        g.trained_sentences.clear();
        metrics::gauge!("javis_brain_sentences").set(0.0);
        metrics::gauge!("javis_brain_words").set(0.0);
        info!(
            dropped_sentences,
            dropped_words, "brain reset to fresh state",
        );
    }

    /// Train the brain on every default-corpus sentence at startup so
    /// the first recall already has something to surface.
    pub async fn bootstrap_default_corpus(&self, tx: Option<Sender<Event>>) {
        for sentence in default_corpus() {
            self.run_train(sentence.to_string(), tx.clone()).await;
        }
    }

    /// Apply STDP/iSTDP/homeostasis on `sentence`, then fingerprint
    /// every newly-seen word into the dictionary. Streams init/phase/
    /// step events on `tx` if provided.
    pub async fn run_train(&self, sentence: String, tx: Option<Sender<Event>>) {
        let started = Instant::now();
        let sentence_len = sentence.len();
        let mut g = self.inner.write().await;

        if let Some(tx) = &tx {
            let (r1, r2_e, r2_i) = count_neurons(&g.brain);
            let _ = tx
                .send(Event::Init {
                    r1_size: r1,
                    r2_size: r2_e + r2_i,
                    r2_excitatory: r2_e,
                    r2_inhibitory: r2_i,
                })
                .await;
            let _ = tx
                .send(Event::Phase {
                    name: "training".into(),
                    detail: format!("\"{}\"", short(&sentence)),
                })
                .await;
        }

        // Engage plasticity for the duration of this training cue.
        g.brain.regions[1].network.enable_stdp(stdp());
        g.brain.regions[1].network.enable_istdp(istdp());
        g.brain.regions[1].network.enable_homeostasis(homeostasis());
        // Iter-44 breakthrough stack: opt-in via `JAVIS_ITER44=1`. The
        // env switch is read once per training cue so a running viz
        // session can be flipped without restart.
        if std::env::var("JAVIS_ITER44").is_ok_and(|v| v == "1") {
            use snn_core::{
                HeterosynapticParams, IntrinsicParams, MetaplasticityParams, RewardParams,
                StructuralParams,
            };
            g.brain.regions[1]
                .network
                .enable_metaplasticity(MetaplasticityParams::enabled());
            g.brain.regions[1]
                .network
                .enable_intrinsic_plasticity(IntrinsicParams::enabled());
            g.brain.regions[1]
                .network
                .enable_heterosynaptic(HeterosynapticParams::l2());
            g.brain.regions[1]
                .network
                .enable_structural(StructuralParams::enabled());
            g.brain.regions[1]
                .network
                .enable_reward_learning(RewardParams::enabled());
        }

        let sdr = g.encoder.encode(&sentence);
        g.brain.reset_state();
        if let Some(tx) = &tx {
            let _ = run_with_cue_streaming(&mut g.brain, &sdr.indices, TRAINING_MS, tx).await;
        } else {
            // Headless: same sim, no event emission.
            let mut ext1 = vec![0.0_f32; R1_N];
            for &i in &sdr.indices {
                if (i as usize) < R1_N {
                    ext1[i as usize] = DRIVE_NA;
                }
            }
            let ext2 = vec![0.0_f32; R2_N];
            let externals = vec![ext1, ext2];
            let steps = (TRAINING_MS / DT) as usize;
            for _ in 0..steps {
                g.brain.step(&externals);
            }
        }

        // Cool-down (no streaming — silence anyway).
        let zeros = vec![vec![0.0_f32; R1_N], vec![0.0_f32; R2_N]];
        let cd_steps = (COOLDOWN_MS / DT) as usize;
        for _ in 0..cd_steps {
            g.brain.step(&zeros);
        }

        // Fingerprint new words, with plasticity frozen so we don't
        // perturb the freshly-shaped engram.
        if let Some(tx) = &tx {
            let _ = tx
                .send(Event::Phase {
                    name: "fingerprint".into(),
                    detail: "learning new vocabulary".into(),
                })
                .await;
        }
        g.brain.disable_stdp_all();
        g.brain.disable_istdp_all();
        g.brain.disable_homeostasis_all();

        let tokens = g.encoder.tokenize(&sentence);
        let new_words: Vec<String> = tokens
            .into_iter()
            .filter(|t| !g.known_words.contains(t))
            .collect();
        for word in &new_words {
            let sdr = g.encoder.encode_word(word);
            if sdr.indices.is_empty() {
                continue;
            }
            g.brain.reset_state();
            let counts = run_with_cue_counts(&mut g.brain, &sdr.indices, RECALL_MS);
            let kwta = top_k(&counts, KWTA_K);
            if !kwta.is_empty() {
                g.dict.learn_concept(word, &kwta);
                g.known_words.insert(word.clone());
            }
        }

        g.trained_sentences.push(sentence);
        let total_sentences = g.trained_sentences.len();
        let total_words = g.known_words.len();
        let new_words_count = new_words.len();
        let elapsed = started.elapsed();
        let elapsed_ms = elapsed.as_secs_f32() * 1000.0;

        if let Some(tx) = &tx {
            let _ = tx
                .send(Event::Phase {
                    name: "ready".into(),
                    detail: format!("{total_sentences} sentences, {total_words} concepts learnt",),
                })
                .await;
            let _ = tx.send(Event::Done).await;
        }

        metrics::histogram!("javis_train_duration_seconds").record(elapsed.as_secs_f64());
        metrics::gauge!("javis_brain_sentences").set(total_sentences as f64);
        metrics::gauge!("javis_brain_words").set(total_words as f64);

        info!(
            sentence_len,
            new_words = new_words_count,
            total_sentences,
            total_words,
            elapsed_ms,
            "train completed",
        );
    }

    /// Push the query into R1 and decode the resulting R2 pattern
    /// against the current dictionary. Streams the spike activity and
    /// the final Decoded event.
    pub async fn run_recall(&self, query: String, tx: Sender<Event>) {
        let started = Instant::now();

        // Phase 1: lock acquisition + init/phase event sends. This is
        // the "request handling overhead" that sits between the WS
        // upgrade and the first useful brain work.
        let p_lock_t0 = Instant::now();
        let g = self.inner.read().await;
        let (r1, r2_e, r2_i) = count_neurons(&g.brain);
        let _ = tx
            .send(Event::Init {
                r1_size: r1,
                r2_size: r2_e + r2_i,
                r2_excitatory: r2_e,
                r2_inhibitory: r2_i,
            })
            .await;
        let _ = tx
            .send(Event::Phase {
                name: "recall".into(),
                detail: format!("query: '{query}'"),
            })
            .await;
        let phase_lock_s = p_lock_t0.elapsed().as_secs_f64();

        // Phase 2: text → SDR via the deterministic-hash encoder.
        let p_enc_t0 = Instant::now();
        let sdr = g.encoder.encode(&query);
        let phase_encode_s = p_enc_t0.elapsed().as_secs_f64();
        if sdr.indices.is_empty() {
            warn!(%query, "recall: query produced empty SDR (unknown vocabulary)");
            let _ = tx.send(Event::Done).await;
            return;
        }

        // Phase 3: SNN compute. `step_immutable` runs RECALL_MS of
        // simulated time and emits Step events to the WS channel as
        // it goes — channel-send latency is *included* in this phase
        // because it interleaves with the brain step.
        let p_sim_t0 = Instant::now();
        let mut state = g.brain.fresh_state();
        let counts =
            run_with_cue_streaming_immutable(&g.brain, &mut state, &sdr.indices, RECALL_MS, &tx)
                .await;
        let phase_sim_s = p_sim_t0.elapsed().as_secs_f64();

        // Phase 4: SNN-output → text. kWTA top-k plus the
        // EngramDictionary scan that compares the recalled SDR
        // against every learnt concept.
        let p_dec_t0 = Instant::now();
        let recall_indices = top_k(&counts, KWTA_K);
        let candidates = g.dict.decode(&recall_indices, DECODE_THRESHOLD);
        let phase_decode_s = p_dec_t0.elapsed().as_secs_f64();

        let javis_payload = candidates
            .iter()
            .map(|(w, _)| w.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        // Phase 5: naïve-RAG mock. Linear scan over every trained
        // sentence; included here as the comparison baseline that
        // motivates the whole brain-side pipeline.
        let p_rag_t0 = Instant::now();
        let q_lc = query.to_lowercase();
        let rag_payload = g
            .trained_sentences
            .iter()
            .find(|s| s.to_lowercase().contains(&q_lc))
            .cloned()
            .unwrap_or_default();
        let phase_rag_s = p_rag_t0.elapsed().as_secs_f64();

        let rag_tokens = eval::count_tokens(&rag_payload) as u32;
        let javis_tokens = eval::count_tokens(&javis_payload) as u32;
        let reduction = if rag_tokens > 0 {
            (1.0 - javis_tokens as f32 / rag_tokens as f32) * 100.0
        } else {
            0.0
        };

        let candidate_count = candidates.len();
        let elapsed = started.elapsed();
        let elapsed_ms = elapsed.as_secs_f32() * 1000.0;
        metrics::histogram!("javis_recall_duration_seconds").record(elapsed.as_secs_f64());
        metrics::counter!("javis_recall_tokens_rag_total").increment(rag_tokens as u64);
        metrics::counter!("javis_recall_tokens_javis_total").increment(javis_tokens as u64);

        // Phase 6: Decoded-event build + send. The actual JSON
        // serialisation happens later in `run_session_inner`; what
        // we capture here is the cost of constructing the payload
        // and pushing it onto the channel.
        let p_build_t0 = Instant::now();
        let decoded_event = Event::Decoded {
            query: query.clone(),
            candidates: candidates
                .into_iter()
                .map(|(word, score)| DecodedWord { word, score })
                .collect(),
            rag_tokens,
            javis_tokens,
            reduction_pct: reduction,
            rag_payload,
            javis_payload,
        };
        let _ = tx.send(decoded_event).await;
        let phase_build_s = p_build_t0.elapsed().as_secs_f64();

        // Per-phase histograms (Prometheus). Same `_seconds` suffix
        // as the existing duration metrics so they share the bucket
        // configuration from `viz::metrics::init`.
        for (phase, value) in [
            ("lock_overhead", phase_lock_s),
            ("encode", phase_encode_s),
            ("snn_compute", phase_sim_s),
            ("decode", phase_decode_s),
            ("rag_search", phase_rag_s),
            ("response_build", phase_build_s),
        ] {
            metrics::histogram!("javis_recall_phase_seconds", "phase" => phase).record(value);
        }

        info!(
            %query,
            candidates = candidate_count,
            rag_tokens,
            javis_tokens,
            reduction_pct = reduction,
            elapsed_ms,
            phase_lock_ms = phase_lock_s * 1000.0,
            phase_encode_ms = phase_encode_s * 1000.0,
            phase_snn_ms = phase_sim_s * 1000.0,
            phase_decode_ms = phase_decode_s * 1000.0,
            phase_rag_ms = phase_rag_s * 1000.0,
            phase_build_ms = phase_build_s * 1000.0,
            "recall completed",
        );

        let _ = tx.send(Event::Done).await;
    }

    /// Snapshot for the UI: how many sentences / concepts have been
    /// learnt so far.
    pub async fn stats(&self) -> (usize, usize) {
        let g = self.inner.read().await;
        (g.trained_sentences.len(), g.known_words.len())
    }

    /// Serialise the entire learnt state to a JSON file. Transient
    /// buffers (membrane potentials, traces, scheduled spike events,
    /// the global clock) are not written — they reset to zero on load
    /// and rebuild within milliseconds of simulation.
    pub async fn save_to_file(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let started = Instant::now();
        let path_ref = path.as_ref().to_path_buf();
        let g = self.inner.read().await;
        let snap = Snapshot {
            version: SNAPSHOT_VERSION,
            metadata: SnapshotMetadata {
                created_at_unix: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                javis_version: env!("CARGO_PKG_VERSION").to_string(),
            },
            brain: clone_brain(&g.brain),
            dict: g.dict.clone(),
            encoder: g.encoder.clone(),
            known_words: g.known_words.clone(),
            trained_sentences: g.trained_sentences.clone(),
        };
        let bytes = serde_json::to_vec(&snap).map_err(io_err)?;
        let bytes_len = bytes.len();
        tokio::fs::write(path, bytes).await?;
        let elapsed = started.elapsed();
        metrics::histogram!("javis_snapshot_duration_seconds", "op" => "save")
            .record(elapsed.as_secs_f64());
        info!(
            path = %path_ref.display(),
            bytes = bytes_len,
            elapsed_ms = elapsed.as_secs_f32() * 1000.0,
            "snapshot saved",
        );
        Ok(())
    }

    /// Replace the in-memory state with a snapshot read from disk.
    /// Transient buffers are re-initialised so the brain is immediately
    /// runnable.
    pub async fn load_from_file(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let started = Instant::now();
        let path_ref = path.as_ref().to_path_buf();
        let bytes = tokio::fs::read(&path_ref).await?;
        let bytes_len = bytes.len();

        // Two-step parse: first into a generic Value so we can read
        // the version field without binding the schema, then through
        // the migration chain, then into the canonical Snapshot
        // struct. This is what lets a v1 file load on a v2 build.
        let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(io_err)?;
        let from_version = value
            .get("version")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "snapshot is missing required `version` field",
                )
            })?;
        let migrated_from_version = if from_version != SNAPSHOT_VERSION {
            Some(from_version)
        } else {
            None
        };
        let value = migrate_snapshot(value, from_version)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut snap: Snapshot = serde_json::from_value(value).map_err(io_err)?;
        snap.brain.ensure_transient_state();

        let sentences = snap.trained_sentences.len();
        let words = snap.known_words.len();

        let mut g = self.inner.write().await;
        g.brain = snap.brain;
        g.dict = snap.dict;
        g.encoder = snap.encoder;
        g.known_words = snap.known_words;
        g.trained_sentences = snap.trained_sentences;

        let elapsed = started.elapsed();
        metrics::histogram!("javis_snapshot_duration_seconds", "op" => "load")
            .record(elapsed.as_secs_f64());
        metrics::gauge!("javis_brain_sentences").set(sentences as f64);
        metrics::gauge!("javis_brain_words").set(words as f64);

        if let Some(from) = migrated_from_version {
            info!(
                path = %path_ref.display(),
                from_version = from,
                to_version = SNAPSHOT_VERSION,
                bytes = bytes_len,
                sentences,
                words,
                elapsed_ms = elapsed.as_secs_f32() * 1000.0,
                "snapshot loaded after schema migration",
            );
        } else {
            info!(
                path = %path_ref.display(),
                bytes = bytes_len,
                sentences,
                words,
                elapsed_ms = elapsed.as_secs_f32() * 1000.0,
                "snapshot loaded",
            );
        }
        Ok(())
    }

    /// Send the question to the LLM twice in parallel — once with the
    /// full RAG payload as context, once with the compact Javis payload.
    /// Streams the answers back as a single `Asked` event so the UI can
    /// show them side by side.
    pub async fn run_ask(
        &self,
        question: String,
        rag_payload: String,
        javis_payload: String,
        tx: Sender<Event>,
    ) {
        let started = Instant::now();
        let real = self.llm.is_real();
        debug!(
            real,
            rag_len = rag_payload.len(),
            javis_len = javis_payload.len(),
            "ask: dispatching parallel LLM calls",
        );
        let _ = tx
            .send(Event::Phase {
                name: "asking".into(),
                detail: if real {
                    "calling Claude (real) with both payloads".into()
                } else {
                    "calling Claude (mock) with both payloads".into()
                },
            })
            .await;

        let llm = self.llm.clone();
        let llm_b = self.llm.clone();
        let q1 = question.clone();
        let q2 = question.clone();
        let ctx_rag = rag_payload.clone();
        let ctx_jvs = javis_payload.clone();

        let (rag_ans, jvs_ans) =
            tokio::join!(async move { llm.ask(&q1, &ctx_rag).await }, async move {
                llm_b.ask(&q2, &ctx_jvs).await
            },);

        let elapsed = started.elapsed();
        let elapsed_ms = elapsed.as_secs_f32() * 1000.0;
        metrics::histogram!(
            "javis_ask_duration_seconds",
            "real" => if real { "true" } else { "false" },
        )
        .record(elapsed.as_secs_f64());
        info!(
            real,
            rag_input_tokens = rag_ans.input_tokens,
            rag_output_tokens = rag_ans.output_tokens,
            javis_input_tokens = jvs_ans.input_tokens,
            javis_output_tokens = jvs_ans.output_tokens,
            elapsed_ms,
            "ask completed",
        );

        let _ = tx
            .send(Event::Asked {
                question,
                rag: LlmReply {
                    text: rag_ans.text,
                    input_tokens: rag_ans.input_tokens,
                    output_tokens: rag_ans.output_tokens,
                    real: rag_ans.real,
                },
                javis: LlmReply {
                    text: jvs_ans.text,
                    input_tokens: jvs_ans.input_tokens,
                    output_tokens: jvs_ans.output_tokens,
                    real: jvs_ans.real,
                },
            })
            .await;
        let _ = tx.send(Event::Done).await;
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// Brain isn't `Copy`, but we need a snapshot-friendly clone — `Clone`
/// is now derived on Brain/Region/Network so this is a real clone, not
/// a serde round-trip.
fn clone_brain(brain: &Brain) -> Brain {
    brain.clone()
}

fn io_err<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
}

fn short(s: &str) -> String {
    if s.chars().count() > 60 {
        let truncated: String = s.chars().take(57).collect();
        format!("{truncated}…")
    } else {
        s.to_string()
    }
}
