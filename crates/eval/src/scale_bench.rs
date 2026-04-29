//! Train-once, query-many harness for the validation-at-scale
//! benchmark.
//!
//! `token_efficiency::run_javis_pipeline` rebuilds and retrains the
//! brain on every query — fine for a 5-paragraph corpus, fatal for
//! anything bigger. [`ScaleBrain`] trains the brain *once* on a
//! corpus, fingerprints every vocabulary word into the dictionary
//! once, and then runs as many queries as the caller wants against
//! the persistent state.
//!
//! Per-query metrics:
//! - **token reduction**: same comparison as the small-corpus
//!   benchmark. RAG payload = first sentence containing the query
//!   verbatim; Javis payload = decoded engram words joined.
//! - **decoder latency**: wall-time for `dict.decode_top(...)` only,
//!   so we can bound the linear-scan cost as the vocabulary grows
//!   without folding in the brain-step cost.
//! - **precision**: did the decoded set contain the query word
//!   itself? (The strongest single-bit answer to "does Javis remember
//!   this concept?")
//! - **false positives**: how many decoded words have *no* sentence-
//!   level co-occurrence with the query, i.e. cross-domain bleed.
//! - **recall**: of every word that genuinely co-occurs with the
//!   query in the corpus, how many made it into the decoded set?

use std::collections::{BTreeSet, HashMap};
use std::time::Instant;

use encoders::{EngramDictionary, TextEncoder};
use snn_core::{
    Brain, HomeostasisParams, IStdpParams, LifNeuron, LifParams, NeuronKind, Region, Rng,
    StdpParams,
};

use crate::count_tokens;
use crate::scale_corpus::ScaleCorpus;

// Iter 26: R1 / encoder grown 1 000 → 4 000 to break the encoder-
// collision side of cross-bleed. See notes/44.
const DT: f32 = 0.1;
const R1_N: usize = 4000;
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
const KWTA_K: usize = 220;

/// Single query result with everything the report needs.
#[derive(Debug, Clone)]
pub struct ScaleQueryResult {
    pub query: String,
    pub rag_tokens: usize,
    pub javis_tokens: usize,
    pub token_reduction_pct: f32,
    /// Words decoded by the engram dictionary (sorted by score, top
    /// `decode_k` only).
    pub decoded: Vec<(String, f32)>,
    /// Wall-time of the `EngramDictionary::decode_top` call only.
    pub decode_micros: u128,
    /// Wall-time from start of `query` to decoded result.
    pub total_micros: u128,
    /// Was the query word itself in the decoded set?
    pub has_self: bool,
    /// Decoded words that have no overlap with the query's
    /// ground-truth sentence set — cross-domain bleed.
    pub false_positives: Vec<String>,
    /// Words that genuinely co-occur with the query in the corpus
    /// but did *not* make it into the decoded set.
    pub false_negatives: Vec<String>,
}

impl ScaleReport {
    /// Render the summary as a Markdown table block plus a short
    /// outlier list (the queries with the most false positives).
    /// Designed to be appended to a benchmark report file or pasted
    /// into a release-notes section.
    pub fn render_markdown(&self) -> String {
        let s = &self.summary;
        let mut out = String::new();
        out.push_str("### Scale benchmark summary\n\n");
        out.push_str(&format!(
            "- Sentences: **{}** · Vocabulary: **{}** · Queries evaluated: **{}**\n",
            s.n_sentences, s.vocab_size, s.n_queries,
        ));
        out.push_str(&format!(
            "- Training wall-time: **{:.1} s**\n\n",
            s.training_secs,
        ));

        out.push_str("| Metric | Value |\n");
        out.push_str("| --- | ---: |\n");
        out.push_str(&format!(
            "| Mean RAG tokens / query | {:.2} |\n",
            s.mean_rag_tokens,
        ));
        out.push_str(&format!(
            "| Mean Javis tokens / query | {:.2} |\n",
            s.mean_javis_tokens,
        ));
        out.push_str(&format!(
            "| Mean token reduction | **{:.1} %** |\n",
            s.mean_reduction_pct,
        ));
        out.push_str(&format!(
            "| Median token reduction | {:.1} % |\n",
            s.median_reduction_pct,
        ));
        out.push_str(&format!(
            "| Decoder precision (query in result) | {:.3} |\n",
            s.precision,
        ));
        out.push_str(&format!(
            "| Decoder recall (expected neighbours) | {:.3} |\n",
            s.recall,
        ));
        out.push_str(&format!(
            "| Mean false positives / query | {:.2} |\n",
            s.mean_false_positives,
        ));
        out.push_str(&format!(
            "| Mean false negatives / query | {:.2} |\n",
            s.mean_false_negatives,
        ));
        out.push_str(&format!(
            "| Mean decoder latency | {:.0} µs |\n",
            s.mean_decode_micros,
        ));
        out.push_str(&format!(
            "| p99 decoder latency | {:.0} µs |\n",
            s.p99_decode_micros,
        ));
        out.push_str(&format!(
            "| Mean total query latency | {:.2} ms |\n\n",
            s.mean_total_micros / 1000.0,
        ));

        // Top-5 noisiest queries by FP count, useful for the writeup
        // when we need a concrete cross-bleed example.
        let mut noisy: Vec<&ScaleQueryResult> = self.per_query.iter().collect();
        noisy.sort_by(|a, b| b.false_positives.len().cmp(&a.false_positives.len()));
        let top: Vec<&&ScaleQueryResult> = noisy.iter().take(5).collect();
        if top.iter().any(|r| !r.false_positives.is_empty()) {
            out.push_str("Top-5 cross-bleed examples (most false positives):\n\n");
            out.push_str("| Query | FP count | Sample FP words |\n");
            out.push_str("| --- | ---: | --- |\n");
            for r in top {
                let sample: Vec<&str> = r
                    .false_positives
                    .iter()
                    .take(4)
                    .map(|s| s.as_str())
                    .collect();
                out.push_str(&format!(
                    "| `{}` | {} | {} |\n",
                    r.query,
                    r.false_positives.len(),
                    if sample.is_empty() {
                        "—".to_string()
                    } else {
                        sample.join(", ")
                    },
                ));
            }
            out.push('\n');
        }
        out
    }
}

/// Aggregated stats over all queries in a benchmark run.
#[derive(Debug, Clone)]
pub struct ScaleSummary {
    pub n_sentences: usize,
    pub vocab_size: usize,
    pub n_queries: usize,
    pub training_secs: f64,
    pub mean_rag_tokens: f64,
    pub mean_javis_tokens: f64,
    pub mean_reduction_pct: f64,
    pub median_reduction_pct: f64,
    pub mean_decode_micros: f64,
    pub p99_decode_micros: f64,
    pub mean_total_micros: f64,
    pub precision: f64,
    pub recall: f64,
    pub mean_false_positives: f64,
    pub mean_false_negatives: f64,
}

/// Full report bundle: per-corpus-size summary plus the raw
/// per-query results so the runner can drill into outliers.
#[derive(Debug)]
pub struct ScaleReport {
    pub summary: ScaleSummary,
    pub per_query: Vec<ScaleQueryResult>,
}

/// Persistent trained brain, ready to answer many queries.
pub struct ScaleBrain {
    brain: Brain,
    dict: EngramDictionary,
    encoder: TextEncoder,
    corpus_sentences: Vec<String>,
    /// Per-content-word: the set of sentence indices it appears in.
    /// Loaded straight from `ScaleCorpus::ground_truth`. Used to
    /// compute false-positive / false-negative classifications at
    /// query time.
    cooccurrence: HashMap<String, BTreeSet<usize>>,
    sentence_to_words: Vec<BTreeSet<String>>,
    pub n_sentences: usize,
    pub vocab_size: usize,
    pub training_secs: f64,
}

impl ScaleBrain {
    /// Train the SNN on the corpus and fingerprint every vocabulary
    /// word as an engram. Equivalent to the
    /// `run_javis_recall_inner` setup in `token_efficiency`, except
    /// it returns the brain so the caller can run many queries
    /// without retraining. Logs progress to stderr every 50
    /// sentences so the operator can watch a long run.
    pub fn train_on(corpus: &ScaleCorpus) -> Self {
        let started = Instant::now();
        let stopwords: &[&str] = &[
            "is", "a", "the", "an", "on", "at", "of", "in", "to", "and", "or", "for", "with", "by",
            "from", "but", "as", "it", "its", "this", "that", "these", "those", "be", "are", "was",
            "were", "like",
        ];
        let encoder = TextEncoder::with_stopwords(ENC_N, ENC_K, stopwords.iter().copied());

        let mut brain = Brain::new(DT);
        brain.add_region(build_input_region());
        brain.add_region(build_memory_region(2027));
        wire_forward(&mut brain, 2028);

        // Phase 1: training pass over every sentence with plasticity on.
        brain.regions[1].network.enable_stdp(stdp());
        brain.regions[1].network.enable_istdp(istdp());
        brain.regions[1].network.enable_homeostasis(homeostasis());

        let n = corpus.sentences.len();
        for (i, sentence) in corpus.sentences.iter().enumerate() {
            let sdr = encoder.encode(sentence);
            brain.regions[1].network.reset_state();
            // Drive R1 with the sentence SDR for TRAINING_MS,
            // letting STDP / iSTDP shape the engram.
            run_with_cue(&mut brain, &sdr.indices, TRAINING_MS);
            if (i + 1) % 50 == 0 || i + 1 == n {
                eprintln!("  trained {}/{} sentences", i + 1, n);
            }
        }

        // Cool-down — let traces decay before fingerprinting.
        idle(&mut brain, COOLDOWN_MS);

        // Phase 2: freeze plasticity, fingerprint every vocab word
        // by re-driving its SDR and grabbing the kWTA pattern.
        brain.disable_stdp_all();
        brain.disable_istdp_all();
        brain.disable_homeostasis_all();

        let mut dict = EngramDictionary::new();
        let vocab_total = corpus.vocabulary.len();
        for (i, word) in corpus.vocabulary.iter().enumerate() {
            let sdr = encoder.encode_word(word);
            if sdr.indices.is_empty() {
                continue;
            }
            brain.regions[1].network.reset_state();
            let counts = run_with_cue_counts(&mut brain, &sdr.indices, RECALL_MS);
            let kwta = top_k_indices(&counts, KWTA_K);
            if !kwta.is_empty() {
                dict.learn_concept(word, &kwta);
            }
            if (i + 1) % 100 == 0 || i + 1 == vocab_total {
                eprintln!(
                    "  fingerprinted {}/{} vocabulary words ({} engrams)",
                    i + 1,
                    vocab_total,
                    dict.len(),
                );
            }
        }

        // Cache: per-sentence content-word set. Lets `query` do an
        // O(|S|) RAG lookup without re-tokenising every time.
        let sentence_to_words: Vec<BTreeSet<String>> = corpus
            .sentences
            .iter()
            .map(|s| {
                s.split_whitespace()
                    .map(normalise_word)
                    .filter(|w| w.len() >= 3)
                    .collect()
            })
            .collect();
        let cooccurrence: HashMap<String, BTreeSet<usize>> = corpus
            .ground_truth
            .iter()
            .map(|(w, idxs)| (w.clone(), idxs.iter().copied().collect()))
            .collect();

        Self {
            brain,
            dict,
            encoder,
            corpus_sentences: corpus.sentences.clone(),
            cooccurrence,
            sentence_to_words,
            n_sentences: corpus.sentences.len(),
            vocab_size: corpus.vocabulary.len(),
            training_secs: started.elapsed().as_secs_f64(),
        }
    }

    /// Run one query against the trained state. `decode_k` is the
    /// top-k cap on the engram dictionary scan.
    pub fn query(&mut self, query: &str, decode_k: usize) -> ScaleQueryResult {
        let total_t0 = Instant::now();
        let q_lc = query.to_lowercase();

        // Phase A: brain step on the query SDR (recall path, plasticity
        // already off from the training fixture).
        let qsdr = self.encoder.encode(&q_lc);
        let counts = if qsdr.indices.is_empty() {
            Vec::new()
        } else {
            self.brain.regions[1].network.reset_state();
            run_with_cue_counts(&mut self.brain, &qsdr.indices, RECALL_MS)
        };
        let recall_indices = if counts.is_empty() {
            Vec::new()
        } else {
            top_k_indices(&counts, KWTA_K)
        };

        // Phase B: dictionary scan — measured separately so we can
        // chart it against vocab size.
        let dec_t0 = Instant::now();
        let decoded = self.dict.decode_top(&recall_indices, decode_k);
        let decode_micros = dec_t0.elapsed().as_micros();

        // RAG payload = first sentence whose lowercased form contains
        // the literal query word. Same baseline that the small-corpus
        // benchmark uses.
        let rag_payload = self
            .corpus_sentences
            .iter()
            .find(|s| s.to_lowercase().contains(&q_lc))
            .cloned()
            .unwrap_or_default();
        let rag_tokens = count_tokens(&rag_payload);

        let javis_payload = decoded
            .iter()
            .map(|(w, _)| w.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let javis_tokens = count_tokens(&javis_payload);
        let reduction = if rag_tokens > 0 {
            (1.0 - javis_tokens as f32 / rag_tokens as f32) * 100.0
        } else {
            0.0
        };

        // Quality classification.
        let decoded_set: BTreeSet<String> = decoded.iter().map(|(w, _)| w.clone()).collect();
        let has_self = decoded_set.contains(&q_lc);

        // Ground-truth co-occurrences for the query: every word that
        // appears together with the query in at least one sentence.
        let expected_neighbours = self.expected_neighbours(&q_lc);
        let false_negatives: Vec<String> = expected_neighbours
            .iter()
            .filter(|w| !decoded_set.contains(*w))
            .cloned()
            .collect();
        // FP: a decoded word that never co-occurred with the query in
        // any training sentence. The query itself is exempt.
        let false_positives: Vec<String> = decoded_set
            .iter()
            .filter(|w| **w != q_lc && !expected_neighbours.contains(*w))
            .cloned()
            .collect();

        ScaleQueryResult {
            query: query.to_string(),
            rag_tokens,
            javis_tokens,
            token_reduction_pct: reduction,
            decoded,
            decode_micros,
            total_micros: total_t0.elapsed().as_micros(),
            has_self,
            false_positives,
            false_negatives,
        }
    }

    /// Run the full set of queries the corpus declared and aggregate.
    pub fn evaluate(&mut self, queries: &[String], decode_k: usize) -> ScaleReport {
        let mut per_query: Vec<ScaleQueryResult> = Vec::with_capacity(queries.len());
        for (i, q) in queries.iter().enumerate() {
            per_query.push(self.query(q, decode_k));
            if (i + 1) % 25 == 0 || i + 1 == queries.len() {
                eprintln!("  evaluated {}/{} queries", i + 1, queries.len());
            }
        }
        let summary = aggregate(&per_query, self);
        ScaleReport { summary, per_query }
    }

    /// Set of words that genuinely co-occur with `query` in the
    /// corpus (excluding the query itself). Reused by `query` for
    /// FP / FN classification.
    fn expected_neighbours(&self, query: &str) -> BTreeSet<String> {
        let mut neighbours = BTreeSet::new();
        if let Some(sentence_idxs) = self.cooccurrence.get(query) {
            for &si in sentence_idxs {
                if let Some(words) = self.sentence_to_words.get(si) {
                    for w in words {
                        if w != query {
                            neighbours.insert(w.clone());
                        }
                    }
                }
            }
        }
        neighbours
    }
}

fn aggregate(results: &[ScaleQueryResult], brain: &ScaleBrain) -> ScaleSummary {
    let n = results.len() as f64;
    if n == 0.0 {
        return ScaleSummary {
            n_sentences: brain.n_sentences,
            vocab_size: brain.vocab_size,
            n_queries: 0,
            training_secs: brain.training_secs,
            mean_rag_tokens: 0.0,
            mean_javis_tokens: 0.0,
            mean_reduction_pct: 0.0,
            median_reduction_pct: 0.0,
            mean_decode_micros: 0.0,
            p99_decode_micros: 0.0,
            mean_total_micros: 0.0,
            precision: 0.0,
            recall: 0.0,
            mean_false_positives: 0.0,
            mean_false_negatives: 0.0,
        };
    }

    let sum_rag: f64 = results.iter().map(|r| r.rag_tokens as f64).sum();
    let sum_jav: f64 = results.iter().map(|r| r.javis_tokens as f64).sum();
    let sum_red: f64 = results.iter().map(|r| r.token_reduction_pct as f64).sum();
    let sum_decode: f64 = results.iter().map(|r| r.decode_micros as f64).sum();
    let sum_total: f64 = results.iter().map(|r| r.total_micros as f64).sum();
    let n_self_hit: f64 = results.iter().filter(|r| r.has_self).count() as f64;
    let sum_fp: f64 = results.iter().map(|r| r.false_positives.len() as f64).sum();
    let sum_fn: f64 = results.iter().map(|r| r.false_negatives.len() as f64).sum();

    let mut decode_sorted: Vec<u128> = results.iter().map(|r| r.decode_micros).collect();
    decode_sorted.sort_unstable();
    let p99 = decode_sorted
        .get(((decode_sorted.len() as f64 * 0.99) as usize).min(decode_sorted.len() - 1))
        .copied()
        .unwrap_or(0) as f64;

    let mut red_sorted: Vec<f32> = results.iter().map(|r| r.token_reduction_pct).collect();
    red_sorted.sort_by(|a, b| a.total_cmp(b));
    let median = red_sorted[red_sorted.len() / 2] as f64;

    // Recall = expected-neighbours-actually-decoded / expected-total.
    let mut tp_sum = 0.0_f64;
    let mut expected_sum = 0.0_f64;
    for r in results {
        let expected_total = (r.false_negatives.len()
            + r.decoded
                .iter()
                .filter(|(w, _)| {
                    *w != r.query.to_lowercase() && !r.false_positives.iter().any(|fp| fp == w)
                })
                .count()) as f64;
        let tp = (r.decoded.len() as f64)
            - (r.false_positives.len() as f64)
            - if r.has_self { 1.0 } else { 0.0 };
        tp_sum += tp.max(0.0);
        expected_sum += expected_total;
    }
    let recall = if expected_sum > 0.0 {
        tp_sum / expected_sum
    } else {
        0.0
    };

    ScaleSummary {
        n_sentences: brain.n_sentences,
        vocab_size: brain.vocab_size,
        n_queries: results.len(),
        training_secs: brain.training_secs,
        mean_rag_tokens: sum_rag / n,
        mean_javis_tokens: sum_jav / n,
        mean_reduction_pct: sum_red / n,
        median_reduction_pct: median,
        mean_decode_micros: sum_decode / n,
        p99_decode_micros: p99,
        mean_total_micros: sum_total / n,
        precision: n_self_hit / n,
        recall,
        mean_false_positives: sum_fp / n,
        mean_false_negatives: sum_fn / n,
    }
}

// ---------------------------------------------------------------
// Brain construction (mirrors viz::state). Inlined here so the
// benchmark is self-contained.
// ---------------------------------------------------------------

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

/// Balanced-distinct R1 → R2 wiring (iter 26). Same scheme as
/// `viz::state::wire_forward` and `token_efficiency::wire_forward`.
fn wire_forward(brain: &mut Brain, seed: u64) {
    let mut rng = Rng::new(seed);
    let r2_size = brain.regions[1].num_neurons();
    let target_in_degree = (R1_N * FAN_OUT) / r2_size;
    let mut in_degree = vec![0u32; r2_size];
    let mut taken = vec![false; r2_size];
    let cap = target_in_degree as u32 + 1;
    let attempt_cap = FAN_OUT * 32;

    for src in 0..R1_N {
        for slot in taken.iter_mut() {
            *slot = false;
        }
        let mut picked = 0;
        let mut attempts = 0;
        while picked < FAN_OUT && attempts < attempt_cap {
            let dst = (rng.next_u64() as usize) % r2_size;
            if !taken[dst] && in_degree[dst] < cap {
                taken[dst] = true;
                in_degree[dst] += 1;
                brain.connect(0, src, 1, dst, INTER_WEIGHT, INTER_DELAY_MS);
                picked += 1;
            }
            attempts += 1;
        }
        if picked < FAN_OUT {
            for dst in 0..r2_size {
                if picked >= FAN_OUT {
                    break;
                }
                if !taken[dst] {
                    taken[dst] = true;
                    in_degree[dst] += 1;
                    brain.connect(0, src, 1, dst, INTER_WEIGHT, INTER_DELAY_MS);
                    picked += 1;
                }
            }
        }
    }
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

fn run_with_cue(brain: &mut Brain, drive: &[u32], duration_ms: f32) {
    let mut ext1 = vec![0.0_f32; R1_N];
    for &i in drive {
        if (i as usize) < R1_N {
            ext1[i as usize] = DRIVE_NA;
        }
    }
    let ext2 = vec![0.0_f32; R2_N];
    let externals = vec![ext1, ext2];
    let steps = (duration_ms / DT) as usize;
    for _ in 0..steps {
        brain.step(&externals);
    }
}

fn run_with_cue_counts(brain: &mut Brain, drive: &[u32], duration_ms: f32) -> Vec<u32> {
    let r2_size = brain.regions[1].num_neurons();
    let mut counts = vec![0u32; r2_size];
    let r2_e_set: BTreeSet<usize> = brain.regions[1]
        .network
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.kind == NeuronKind::Excitatory)
        .map(|(i, _)| i)
        .collect();
    let mut ext1 = vec![0.0_f32; R1_N];
    for &i in drive {
        if (i as usize) < R1_N {
            ext1[i as usize] = DRIVE_NA;
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

fn normalise_word(raw: &str) -> String {
    raw.chars()
        .filter(|c| c.is_alphabetic() || *c == '-')
        .collect::<String>()
        .to_lowercase()
}
