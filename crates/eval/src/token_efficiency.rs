//! Token-efficiency benchmark: Javis vs. naive RAG.
//!
//! For a given query keyword, the **naive RAG** simulator returns the
//! whole paragraph that contains the keyword — that is the entire
//! chunk an LLM would have to read. **Javis** returns only the
//! concepts its `EngramDictionary` decoded from the post-recall R2
//! activity. Both payloads are scored with the same word-count
//! heuristic, so their ratio is the apples-to-apples token saving.
//!
//! All SNN parameters here mirror the sweet spot found in
//! `notes/11-istdp-intrinsische-selektivitaet.md`.

use std::collections::HashSet;

use encoders::{inject_sdr, EngramDictionary, TextEncoder};
use snn_core::{
    poisson, Brain, HomeostasisParams, IStdpParams, LifNeuron, LifParams, NeuronKind,
    PoissonInput, Region, Rng, StdpParams,
};

use crate::count_tokens;

// ----------------------------------------------------------------------
// Mock corpus
// ----------------------------------------------------------------------

/// Three short paragraphs about three different programming languages.
/// Realistic-enough English, short enough that a single training cue
/// per paragraph fits inside R1's address space.
pub fn corpus() -> Vec<&'static str> {
    vec![
        "Rust is a systems programming language focused on memory safety \
         and ownership; the borrow checker prevents data races at compile time.",
        "Python dominates data science with libraries like numpy and pandas; \
         dynamic typing makes prototyping fast at the cost of runtime speed.",
        "Cpp gives raw control over memory through pointers and manual \
         allocation; templates and zero cost abstractions enable performance.",
    ]
}

const STOPWORDS: &[&str] = &[
    "is", "a", "the", "an", "on", "at", "of", "in", "to", "and", "or",
    "for", "with", "by", "from", "but", "as", "it", "its", "this", "that",
    "these", "those", "be", "are", "was", "were", "like",
];

// ----------------------------------------------------------------------
// Naive RAG simulator
// ----------------------------------------------------------------------

/// Simulate a vector search: return the entire paragraph whose lowercase
/// form contains the (lowercase) query. First match wins. The full
/// paragraph is what naive RAG would hand to the LLM as context.
pub fn naive_rag_lookup(corpus: &[&str], query: &str) -> Option<String> {
    let q = query.to_lowercase();
    for chunk in corpus {
        if chunk.to_lowercase().contains(&q) {
            return Some(chunk.to_string());
        }
    }
    None
}

// ----------------------------------------------------------------------
// SNN parameters — sweet spot from notes/11
// ----------------------------------------------------------------------

const DT: f32 = 0.1;
const R1_N: usize = 1000;
const R2_N: usize = 2000;
const R2_INH_FRAC: f32 = 0.20;
const R2_P_CONNECT: f32 = 0.10;
const FAN_OUT: usize = 10;
const INTER_WEIGHT: f32 = 2.0;
const INTER_DELAY_MS: f32 = 2.0;
const ENC_N: u32 = R1_N as u32;
const ENC_K: u32 = 20;
const DRIVE_NA: f32 = 200.0;
const TRAINING_MS: f32 = 150.0;
const COOLDOWN_MS: f32 = 50.0;
const RECALL_MS: f32 = 30.0;
/// kWTA selectivity at readout. With a large vocabulary the recurrent
/// network is dense enough that "every neuron that fired at least once"
/// covers most of R2 — every fingerprint then matches by sheer
/// cardinality. Top-K filtering keeps both fingerprints and recall
/// patterns sparse so containment scores stay meaningful.
///
/// `KWTA_K` is sized to cover roughly one paragraph's worth of
/// associated concepts (~10 words × 20 fingerprint bits = 200), so
/// the partial query cue completes the rest of its paragraph.
const KWTA_K: usize = 220;
/// Containment-score threshold for the dictionary readout.
const DECODE_THRESHOLD: f32 = 0.50;

fn r2_stdp() -> StdpParams {
    let mut s = StdpParams::default();
    s.a_plus = 0.015;
    s.a_minus = 0.012;
    s.w_max = 0.8;
    s
}

fn r2_istdp() -> IStdpParams {
    IStdpParams {
        a_plus: 0.05,
        a_minus: 0.55,
        tau_minus: 30.0,
        w_min: 0.0,
        w_max: 5.0,
    }
}

fn r2_homeostasis() -> HomeostasisParams {
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
    let net = &mut region.network;
    for _ in 0..R1_N {
        net.add_neuron(LifNeuron::excitatory(LifParams::default()));
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

// ----------------------------------------------------------------------
// Drive helpers
// ----------------------------------------------------------------------

fn run_with_cue(brain: &mut Brain, drive: &[u32], duration_ms: f32) -> HashSet<usize> {
    let r2_e = r2_excitatory_indices(brain);
    let mut ext1 = vec![0.0_f32; R1_N];
    for &idx in drive {
        if (idx as usize) < R1_N {
            ext1[idx as usize] = DRIVE_NA;
        }
    }
    let ext2 = vec![0.0_f32; R2_N];
    let externals = vec![ext1, ext2];
    let steps = (duration_ms / DT) as usize;
    let mut fired = HashSet::new();
    for _ in 0..steps {
        let spikes = brain.step(&externals);
        for &id in &spikes[1] {
            if r2_e.contains(&id) {
                fired.insert(id);
            }
        }
    }
    fired
}

/// Run a cue and return per-R2-E-neuron spike counts over the window.
fn run_with_cue_counts(brain: &mut Brain, drive: &[u32], duration_ms: f32) -> Vec<u32> {
    let r2_e = r2_excitatory_indices(brain);
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
            if r2_e.contains(&id) {
                counts[id] += 1;
            }
        }
    }
    counts
}

/// kWTA: keep the indices of the `k` highest-count neurons (ties broken
/// by lower neuron index). Result is sorted ascending so containment
/// scoring runs in linear merge time.
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

fn idle(brain: &mut Brain, duration_ms: f32) {
    let zeros = vec![vec![0.0_f32; R1_N], vec![0.0_f32; R2_N]];
    let steps = (duration_ms / DT) as usize;
    for _ in 0..steps {
        brain.step(&zeros);
    }
}

// ----------------------------------------------------------------------
// Vocabulary extraction
// ----------------------------------------------------------------------

fn vocabulary(corpus: &[&str], enc: &TextEncoder) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for chunk in corpus {
        for tok in enc.tokenize(chunk) {
            if seen.insert(tok.clone()) {
                out.push(tok);
            }
        }
    }
    out
}

// ----------------------------------------------------------------------
// Javis pipeline
// ----------------------------------------------------------------------

/// Result of one decoded recall: the words the dictionary surfaced
/// above the relevance threshold, sorted by score descending.
pub type DecodedWords = Vec<(String, f32)>;

/// Train R2 on `corpus`, fingerprint every vocabulary word as an
/// engram in the dictionary, then recall with `query` and decode.
pub fn run_javis_pipeline(corpus: &[&str], query: &str) -> DecodedWords {
    let enc = TextEncoder::with_stopwords(ENC_N, ENC_K, STOPWORDS.iter().copied());
    let vocab = vocabulary(corpus, &enc);

    let mut brain = Brain::new(DT);
    brain.add_region(build_input_region());
    brain.add_region(build_memory_region(2027));
    wire_forward(&mut brain, 2028);

    // -------- Phase 1: training --------
    brain.regions[1].network.enable_stdp(r2_stdp());
    brain.regions[1].network.enable_istdp(r2_istdp());
    brain.regions[1].network.enable_homeostasis(r2_homeostasis());

    for chunk in corpus {
        let sdr = enc.encode(chunk);
        brain.reset_state();
        let _ = run_with_cue(&mut brain, &sdr.indices, TRAINING_MS);
    }
    idle(&mut brain, COOLDOWN_MS);

    // -------- Phase 2: fingerprint vocabulary --------
    brain.disable_stdp_all();
    brain.disable_istdp_all();
    brain.disable_homeostasis_all();

    let mut dict = EngramDictionary::new();
    for word in &vocab {
        let sdr = enc.encode_word(word);
        if sdr.indices.is_empty() {
            continue;
        }
        brain.reset_state();
        let counts = run_with_cue_counts(&mut brain, &sdr.indices, RECALL_MS);
        let kwta = top_k_indices(&counts, KWTA_K);
        if !kwta.is_empty() {
            dict.learn_concept(word, &kwta);
        }
    }

    // -------- Phase 3: query recall --------
    let query_sdr = enc.encode(query);
    if query_sdr.indices.is_empty() {
        return Vec::new();
    }
    brain.reset_state();
    let counts = run_with_cue_counts(&mut brain, &query_sdr.indices, RECALL_MS);
    let recall_indices = top_k_indices(&counts, KWTA_K);

    // -------- Phase 4: decode --------
    // The asymmetric containment score is `|recall ∩ stored| / |stored|`,
    // i.e. "how much of this concept's engram is in the recall pattern".
    // The threshold trades off precision (high → only the strongest
    // associates) vs. recall (low → more co-paragraph concepts).
    dict.decode(&recall_indices, DECODE_THRESHOLD)
}

// ----------------------------------------------------------------------
// Benchmark harness
// ----------------------------------------------------------------------

#[derive(Debug)]
pub struct BenchmarkResult {
    pub query: String,
    pub rag_payload: String,
    pub rag_tokens: usize,
    pub javis_words: DecodedWords,
    pub javis_payload: String,
    pub javis_tokens: usize,
    pub token_reduction_pct: f32,
}

impl BenchmarkResult {
    pub fn report(&self) -> String {
        let words: Vec<String> = self
            .javis_words
            .iter()
            .map(|(w, s)| format!("{w}({s:.2})"))
            .collect();
        format!(
            "\n— query  : '{}'\n— RAG    : {} tokens — \"{}\"\n— Javis  : {} tokens — \"{}\"\n— decoded: {}\n— saving : {:.1}%\n",
            self.query,
            self.rag_tokens,
            self.rag_payload,
            self.javis_tokens,
            self.javis_payload,
            words.join(", "),
            self.token_reduction_pct,
        )
    }
}

pub fn run_benchmark(query: &str) -> BenchmarkResult {
    run_benchmark_on(&corpus(), query)
}

/// Same as [`run_benchmark`] but lets the caller supply any corpus —
/// used by the Wikipedia-scale test in `eval/tests/wiki_benchmark.rs`.
pub fn run_benchmark_on(corpus: &[&str], query: &str) -> BenchmarkResult {
    let rag_payload = naive_rag_lookup(corpus, query).unwrap_or_default();
    let rag_tokens = count_tokens(&rag_payload);

    let javis_words = run_javis_pipeline(corpus, query);
    let javis_payload = javis_words
        .iter()
        .map(|(w, _)| w.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let javis_tokens = count_tokens(&javis_payload);

    let token_reduction_pct = if rag_tokens > 0 {
        (1.0 - javis_tokens as f32 / rag_tokens as f32) * 100.0
    } else {
        0.0
    };

    BenchmarkResult {
        query: query.to_string(),
        rag_payload,
        rag_tokens,
        javis_words,
        javis_payload,
        javis_tokens,
        token_reduction_pct,
    }
}

// Suppress the "unused import" warning when the file is built alone —
// `inject_sdr` and `PoissonInput`/`poisson` are exported for future
// experiments and kept here on purpose.
#[allow(dead_code)]
fn _keep_imports_alive(net: &mut snn_core::Network, sdr: &[u32], gens: &[PoissonInput]) {
    inject_sdr(net, sdr, 0.0);
    let mut ext = Vec::<f32>::new();
    let mut rng = Rng::new(0);
    poisson::drive(gens, &mut ext, DT, &mut rng);
}
