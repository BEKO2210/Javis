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
    Brain, HomeostasisParams, IStdpParams, LifNeuron, LifParams, NeuronKind, Region, Rng,
    StdpParams,
};
use tokio::sync::{mpsc::Sender, Mutex};
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
pub const R2_N: usize = 2000;
pub const R2_INH_FRAC: f32 = 0.20;
pub const R2_P_CONNECT: f32 = 0.10;
pub const FAN_OUT: usize = 10;
pub const INTER_WEIGHT: f32 = 2.0;
pub const INTER_DELAY_MS: f32 = 2.0;
pub const ENC_N: u32 = R1_N as u32;
pub const ENC_K: u32 = 20;
pub const DRIVE_NA: f32 = 200.0;
pub const TRAINING_MS: f32 = 150.0;
pub const COOLDOWN_MS: f32 = 50.0;
pub const RECALL_MS: f32 = 30.0;
pub const KWTA_K: usize = 220;
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
    IStdpParams {
        a_plus: 0.05,
        a_minus: 0.55,
        tau_minus: 30.0,
        w_min: 0.0,
        w_max: 5.0,
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
    inner: Arc<Mutex<Inner>>,
    llm: Arc<LlmClient>,
}

struct Inner {
    brain: Brain,
    dict: EngramDictionary,
    encoder: TextEncoder,
    known_words: HashSet<String>,
    trained_sentences: Vec<String>,
}

/// On-disk wire format for an Javis snapshot. Matches `Inner` exactly
/// so the round-trip is just `serde_json::{from,to}_writer`. Schema is
/// versioned so a future format change can fail loudly rather than
/// silently reading stale data.
#[derive(Serialize, Deserialize)]
struct Snapshot {
    /// Bumped any time the schema becomes incompatible.
    version: u32,
    brain: Brain,
    dict: EngramDictionary,
    encoder: TextEncoder,
    known_words: HashSet<String>,
    trained_sentences: Vec<String>,
}

const SNAPSHOT_VERSION: u32 = 1;

impl AppState {
    pub fn new() -> Self {
        let inner = Inner {
            brain: fresh_brain(),
            dict: EngramDictionary::new(),
            encoder: fresh_encoder(),
            known_words: HashSet::new(),
            trained_sentences: Vec::new(),
        };
        Self {
            inner: Arc::new(Mutex::new(inner)),
            llm: Arc::new(LlmClient::from_env()),
        }
    }

    /// Same brain, but with the LLM forced into mock mode — useful for
    /// tests and offline demos.
    pub fn new_with_mock_llm() -> Self {
        let mut s = Self::new();
        s.llm = Arc::new(LlmClient::mock());
        s
    }

    pub fn handle(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            llm: Arc::clone(&self.llm),
        }
    }

    pub fn llm_is_real(&self) -> bool {
        self.llm.is_real()
    }

    /// Reset to a freshly-initialised brain. Loses every learnt engram.
    pub async fn reset(&self) {
        let mut g = self.inner.lock().await;
        let dropped_sentences = g.trained_sentences.len();
        let dropped_words = g.known_words.len();
        g.brain = fresh_brain();
        g.dict = EngramDictionary::new();
        g.known_words.clear();
        g.trained_sentences.clear();
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
        let mut g = self.inner.lock().await;

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
        let elapsed_ms = started.elapsed().as_secs_f32() * 1000.0;

        if let Some(tx) = &tx {
            let _ = tx
                .send(Event::Phase {
                    name: "ready".into(),
                    detail: format!("{total_sentences} sentences, {total_words} concepts learnt",),
                })
                .await;
            let _ = tx.send(Event::Done).await;
        }

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
        let mut g = self.inner.lock().await;

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

        // Plasticity stays frozen for recall.
        g.brain.disable_stdp_all();
        g.brain.disable_istdp_all();
        g.brain.disable_homeostasis_all();

        let sdr = g.encoder.encode(&query);
        if sdr.indices.is_empty() {
            warn!(%query, "recall: query produced empty SDR (unknown vocabulary)");
            let _ = tx.send(Event::Done).await;
            return;
        }
        g.brain.reset_state();
        let counts = run_with_cue_streaming(&mut g.brain, &sdr.indices, RECALL_MS, &tx).await;
        let recall_indices = top_k(&counts, KWTA_K);

        let candidates = g.dict.decode(&recall_indices, DECODE_THRESHOLD);
        let javis_payload = candidates
            .iter()
            .map(|(w, _)| w.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        // RAG search across the sentences trained so far.
        let q_lc = query.to_lowercase();
        let rag_payload = g
            .trained_sentences
            .iter()
            .find(|s| s.to_lowercase().contains(&q_lc))
            .cloned()
            .unwrap_or_default();

        let rag_tokens = eval::count_tokens(&rag_payload) as u32;
        let javis_tokens = eval::count_tokens(&javis_payload) as u32;
        let reduction = if rag_tokens > 0 {
            (1.0 - javis_tokens as f32 / rag_tokens as f32) * 100.0
        } else {
            0.0
        };

        let candidate_count = candidates.len();
        let elapsed_ms = started.elapsed().as_secs_f32() * 1000.0;
        info!(
            %query,
            candidates = candidate_count,
            rag_tokens,
            javis_tokens,
            reduction_pct = reduction,
            elapsed_ms,
            "recall completed",
        );

        let _ = tx
            .send(Event::Decoded {
                query,
                candidates: candidates
                    .into_iter()
                    .map(|(word, score)| DecodedWord { word, score })
                    .collect(),
                rag_tokens,
                javis_tokens,
                reduction_pct: reduction,
                rag_payload,
                javis_payload,
            })
            .await;
        let _ = tx.send(Event::Done).await;
    }

    /// Snapshot for the UI: how many sentences / concepts have been
    /// learnt so far.
    pub async fn stats(&self) -> (usize, usize) {
        let g = self.inner.lock().await;
        (g.trained_sentences.len(), g.known_words.len())
    }

    /// Serialise the entire learnt state to a JSON file. Transient
    /// buffers (membrane potentials, traces, scheduled spike events,
    /// the global clock) are not written — they reset to zero on load
    /// and rebuild within milliseconds of simulation.
    pub async fn save_to_file(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let started = Instant::now();
        let path_ref = path.as_ref().to_path_buf();
        let g = self.inner.lock().await;
        let snap = Snapshot {
            version: SNAPSHOT_VERSION,
            brain: clone_brain(&g.brain),
            dict: g.dict.clone(),
            encoder: g.encoder.clone(),
            known_words: g.known_words.clone(),
            trained_sentences: g.trained_sentences.clone(),
        };
        let bytes = serde_json::to_vec(&snap).map_err(io_err)?;
        let bytes_len = bytes.len();
        tokio::fs::write(path, bytes).await?;
        info!(
            path = %path_ref.display(),
            bytes = bytes_len,
            elapsed_ms = started.elapsed().as_secs_f32() * 1000.0,
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
        let mut snap: Snapshot = serde_json::from_slice(&bytes).map_err(io_err)?;
        if snap.version != SNAPSHOT_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "snapshot version {} unsupported (expected {})",
                    snap.version, SNAPSHOT_VERSION
                ),
            ));
        }
        snap.brain.ensure_transient_state();

        let sentences = snap.trained_sentences.len();
        let words = snap.known_words.len();

        let mut g = self.inner.lock().await;
        g.brain = snap.brain;
        g.dict = snap.dict;
        g.encoder = snap.encoder;
        g.known_words = snap.known_words;
        g.trained_sentences = snap.trained_sentences;

        info!(
            path = %path_ref.display(),
            bytes = bytes_len,
            sentences,
            words,
            elapsed_ms = started.elapsed().as_secs_f32() * 1000.0,
            "snapshot loaded",
        );
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

        let elapsed_ms = started.elapsed().as_secs_f32() * 1000.0;
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
