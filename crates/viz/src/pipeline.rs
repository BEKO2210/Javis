//! Streaming version of the Javis pipeline.
//!
//! Same SNN setup as `eval::token_efficiency`, but every simulation
//! step pushes a batched spike event into a channel so a frontend can
//! render the activity in real time. Phase markers (training / recall /
//! decode) flow through the same channel.

use std::collections::HashSet;

use encoders::{EngramDictionary, TextEncoder};
use snn_core::{
    Brain, HomeostasisParams, IStdpParams, LifNeuron, LifParams, NeuronKind, Region, Rng,
    StdpParams,
};
use tokio::sync::mpsc::Sender;

use crate::events::{DecodedWord, Event};

// SNN sweet-spot, mirrors notes/11 + notes/12.
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
const KWTA_K: usize = 220;
const DECODE_THRESHOLD: f32 = 0.50;

/// One Step event per simulated millisecond — keeps wire traffic
/// roughly constant regardless of `dt`.
const BATCH_MS: f32 = 1.0;

const STOPWORDS: &[&str] = &[
    "is", "a", "the", "an", "on", "at", "of", "in", "to", "and", "or",
    "for", "with", "by", "from", "but", "as", "it", "its", "this", "that",
    "these", "those", "be", "are", "was", "were", "like",
];

fn stdp() -> StdpParams {
    let mut s = StdpParams::default();
    s.a_plus = 0.015;
    s.a_minus = 0.012;
    s.w_max = 0.8;
    s
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

/// Drive R1 with a constant SDR-shaped current and emit per-millisecond
/// spike events. Returns the union of R2-E neurons that fired during
/// the run (usable as a count for downstream readout).
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
        .into_iter()
        .enumerate()
        .filter(|(_, c)| *c > 0)
        .map(|(i, _)| i as u32)
        .collect()
}

/// Helper that returns the per-R2-E spike counts, used for kWTA at
/// fingerprint and recall time.
async fn run_with_cue_counts_streaming(
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
                counts[id as usize] += 1;
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
    counts.iter().map(|&c| c as u32).collect()
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

fn vocabulary(corpus: &[&str], enc: &TextEncoder) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for chunk in corpus {
        for tok in enc.tokenize(chunk) {
            if seen.insert(tok.clone()) {
                out.push(tok);
            }
        }
    }
    out
}

fn corpus() -> Vec<&'static str> {
    vec![
        "Rust is a systems programming language focused on memory safety \
         and ownership; the borrow checker prevents data races at compile time.",
        "Python dominates data science with libraries like numpy and pandas; \
         dynamic typing makes prototyping fast at the cost of runtime speed.",
        "Cpp gives raw control over memory through pointers and manual \
         allocation; templates and zero cost abstractions enable performance.",
    ]
}

/// Run a complete demo session: build brain, train on the corpus,
/// fingerprint vocabulary, recall a single query, decode. All spike
/// activity and phase changes are pushed through `tx`.
pub async fn run_demo_session(query: String, tx: Sender<Event>) {
    let enc = TextEncoder::with_stopwords(ENC_N, ENC_K, STOPWORDS.iter().copied());
    let vocab = vocabulary(&corpus(), &enc);

    let mut brain = Brain::new(DT);
    brain.add_region(build_input_region());
    brain.add_region(build_memory_region(2027));
    wire_forward(&mut brain, 2028);

    let (r1, r2_e, r2_i) = count_neurons(&brain);
    let _ = tx
        .send(Event::Init {
            r1_size: r1,
            r2_size: r2_e + r2_i,
            r2_excitatory: r2_e,
            r2_inhibitory: r2_i,
        })
        .await;

    // Phase 1 — training.
    brain.regions[1].network.enable_stdp(stdp());
    brain.regions[1].network.enable_istdp(istdp());
    brain.regions[1].network.enable_homeostasis(homeostasis());

    for (i, chunk) in corpus().iter().enumerate() {
        let _ = tx
            .send(Event::Phase {
                name: "training".into(),
                detail: format!("paragraph {}/{}", i + 1, corpus().len()),
            })
            .await;
        let sdr = enc.encode(chunk);
        brain.reset_state();
        let _ = run_with_cue_streaming(&mut brain, &sdr.indices, TRAINING_MS, &tx).await;
    }

    // Cool-down.
    let _ = tx
        .send(Event::Phase {
            name: "cooldown".into(),
            detail: format!("{} ms", COOLDOWN_MS),
        })
        .await;
    {
        let zeros = vec![vec![0.0_f32; R1_N], vec![0.0_f32; R2_N]];
        let steps = (COOLDOWN_MS / DT) as usize;
        for _ in 0..steps {
            brain.step(&zeros);
        }
    }

    // Phase 2 — fingerprint vocabulary (no streaming during this phase
    // to keep the wire quiet; it's structural, not interesting visually).
    let _ = tx
        .send(Event::Phase {
            name: "fingerprint".into(),
            detail: format!("{} concepts", vocab.len()),
        })
        .await;
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
        // Run cue without emitting per-step events.
        let mut ext1 = vec![0.0_f32; R1_N];
        for &i in &sdr.indices {
            if (i as usize) < R1_N {
                ext1[i as usize] = DRIVE_NA;
            }
        }
        let ext2 = vec![0.0_f32; R2_N];
        let externals = vec![ext1, ext2];
        let r2_size = brain.regions[1].num_neurons();
        let mut counts = vec![0u32; r2_size];
        let r2_e_set = r2_excitatory_indices(&brain);
        let steps = (RECALL_MS / DT) as usize;
        for _ in 0..steps {
            let spikes = brain.step(&externals);
            for &id in &spikes[1] {
                if r2_e_set.contains(&id) {
                    counts[id] += 1;
                }
            }
        }
        let kwta = top_k(&counts, KWTA_K);
        if !kwta.is_empty() {
            dict.learn_concept(word, &kwta);
        }
    }

    // Phase 3 — recall the query, this time *do* stream.
    let _ = tx
        .send(Event::Phase {
            name: "recall".into(),
            detail: format!("query: '{query}'"),
        })
        .await;
    let query_sdr = enc.encode(&query);
    if query_sdr.indices.is_empty() {
        let _ = tx.send(Event::Done).await;
        return;
    }
    brain.reset_state();
    let counts = run_with_cue_counts_streaming(&mut brain, &query_sdr.indices, RECALL_MS, &tx).await;
    let recall_indices = top_k(&counts, KWTA_K);

    // Phase 4 — decode + RAG comparison.
    let candidates = dict.decode(&recall_indices, DECODE_THRESHOLD);
    let javis_payload = candidates
        .iter()
        .map(|(w, _)| w.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let rag_payload = eval::token_efficiency::naive_rag_lookup(&corpus(), &query)
        .unwrap_or_default();
    let rag_tokens = eval::count_tokens(&rag_payload) as u32;
    let javis_tokens = eval::count_tokens(&javis_payload) as u32;
    let reduction = if rag_tokens > 0 {
        (1.0 - javis_tokens as f32 / rag_tokens as f32) * 100.0
    } else {
        0.0
    };

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
