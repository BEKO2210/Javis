//! End-to-end recall: SDR → R1 → R2 → spike set → text candidates.
//!
//! Closes the read path of Javis. The setup is the same R1 → R2 brain
//! used in the homeostasis-aware pattern-completion test, plus an
//! `EngramDictionary` that learns each word's R2 fingerprint *before*
//! training. After training "hello rust" together the partial cue
//! "hello" should pull "rust" out of the dictionary, even though the
//! word "rust" is not present in the input at recall time — that's the
//! associative read we promised.

use std::collections::HashSet;

use encoders::{EngramDictionary, TextEncoder};
use snn_core::{
    Brain, HomeostasisParams, IStdpParams, LifNeuron, LifParams, NeuronKind, Region, Rng,
    StdpParams,
};

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

const FINGERPRINT_MS: f32 = 100.0;
const TRAINING_MS: f32 = 250.0;
const TARGET_WINDOW_MS: f32 = 20.0;
const COOLDOWN_MS: f32 = 50.0;
const RECALL_MS: f32 = 100.0;

fn build_input_region() -> Region {
    let mut region = Region::new("R1", DT);
    let net = &mut region.network;
    for _ in 0..R1_N {
        net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    }
    region
}

fn r2_stdp() -> StdpParams {
    let mut stdp = StdpParams::default();
    stdp.a_plus = 0.04;
    stdp.a_minus = 0.025;
    stdp.w_max = 2.0;
    stdp
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

fn build_memory_region(seed: u64) -> Region {
    let mut rng = Rng::new(seed);
    let mut region = Region::new("R2", DT);
    let net = &mut region.network;

    net.enable_stdp(r2_stdp());

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

fn run_with_cue(
    brain: &mut Brain,
    drive_indices: &[u32],
    duration_ms: f32,
    record_window_ms: f32,
) -> HashSet<usize> {
    let r2_e: HashSet<usize> = r2_excitatory_indices(brain);

    let total_steps = (duration_ms / DT) as usize;
    let record_start = total_steps.saturating_sub((record_window_ms / DT) as usize);

    let mut ext1 = vec![0.0_f32; R1_N];
    for &idx in drive_indices {
        if (idx as usize) < R1_N {
            ext1[idx as usize] = DRIVE_NA;
        }
    }
    let ext2 = vec![0.0_f32; R2_N];
    let externals = vec![ext1, ext2];

    let mut recorded: HashSet<usize> = HashSet::new();
    for s in 0..total_steps {
        let spikes = brain.step(&externals);
        if s >= record_start {
            for &id in &spikes[1] {
                if r2_e.contains(&id) {
                    recorded.insert(id);
                }
            }
        }
    }
    recorded
}

fn idle(brain: &mut Brain, duration_ms: f32) {
    let zeros = vec![vec![0.0_f32; R1_N], vec![0.0_f32; R2_N]];
    let steps = (duration_ms / DT) as usize;
    for _ in 0..steps {
        brain.step(&zeros);
    }
}

fn record_fingerprint(
    brain: &mut Brain,
    cue_indices: &[u32],
) -> Vec<u32> {
    brain.disable_stdp_all();
    brain.disable_homeostasis_all();
    brain.reset_state();
    let fired = run_with_cue(brain, cue_indices, FINGERPRINT_MS, FINGERPRINT_MS);
    let mut indices: Vec<u32> = fired.into_iter().map(|i| i as u32).collect();
    indices.sort_unstable();
    indices
}

#[test]
fn decoder_retrieves_completed_pattern() {
    let stops = ["the", "is", "on", "a", "der", "die", "und"];
    let enc = TextEncoder::with_stopwords(ENC_N, ENC_K, stops);

    let cue_full = enc.encode("hello rust");
    let cue_hello = enc.encode("hello");
    let cue_rust = enc.encode("rust");

    let mut brain = Brain::new(DT);
    brain.add_region(build_input_region());
    brain.add_region(build_memory_region(2027));
    wire_forward(&mut brain, 2028);

    // Phase 1 — fingerprint each concept before training.
    // STDP and homeostasis are off so the fingerprint reflects only
    // the forward path R1 → R2; nothing in the network changes.
    let mut dict = EngramDictionary::new();
    let fp_hello = record_fingerprint(&mut brain, &cue_hello.indices);
    let fp_rust = record_fingerprint(&mut brain, &cue_rust.indices);
    dict.learn_concept("hello", &fp_hello);
    dict.learn_concept("rust", &fp_rust);

    eprintln!(
        "fingerprints: hello={} bits, rust={} bits",
        fp_hello.len(),
        fp_rust.len(),
    );

    assert!(
        fp_hello.len() >= 30 && fp_rust.len() >= 30,
        "forward path produced no fingerprint (hello={}, rust={})",
        fp_hello.len(),
        fp_rust.len(),
    );

    // Phase 2 — associative training. Both cues together inside R2,
    // STDP shapes the joint engram, asymmetric homeostasis stops
    // bleeding (parameters validated in notes/08).
    brain.regions[1].network.enable_stdp(r2_stdp());
    brain.regions[1].network.enable_homeostasis(r2_homeostasis());
    brain.reset_state();
    let _target_assembly = run_with_cue(
        &mut brain,
        &cue_full.indices,
        TRAINING_MS,
        TARGET_WINDOW_MS,
    );
    idle(&mut brain, COOLDOWN_MS);

    // Phase 3 — partial-cue recall. Both forms of plasticity frozen.
    brain.disable_stdp_all();
    brain.disable_homeostasis_all();
    brain.reset_state();
    let post_recall = run_with_cue(&mut brain, &cue_hello.indices, RECALL_MS, RECALL_MS);
    let post_indices: Vec<u32> = {
        let mut v: Vec<u32> = post_recall.iter().map(|&i| i as u32).collect();
        v.sort_unstable();
        v
    };

    // Phase 4 — decode. Only "hello" was actually presented; if the
    // associative engram is real, "rust" must come back too.
    let candidates = dict.decode(&post_indices, 0.5);

    eprintln!(
        "recall set = {} bits   candidates = {:?}",
        post_indices.len(),
        candidates,
    );

    assert!(
        !candidates.is_empty(),
        "decoder returned no candidates for the partial cue",
    );

    let rust_score = candidates
        .iter()
        .find(|(w, _)| w == "rust")
        .map(|(_, s)| *s)
        .expect("decoder did not surface the absent word 'rust'");

    let hello_score = candidates
        .iter()
        .find(|(w, _)| w == "hello")
        .map(|(_, s)| *s);

    // Core assertion: "rust" must be retrieved with high containment
    // even though the input only carried "hello".
    assert!(
        rust_score >= 0.70,
        "associative recall too weak: rust score {:.2} (want ≥ 0.70)",
        rust_score,
    );

    // And the directly-cued concept should obviously also come back.
    if let Some(s) = hello_score {
        assert!(
            s >= 0.70,
            "directly-cued 'hello' should score well: got {:.2}",
            s,
        );
    }
}

// ----------------------------------------------------------------------
// Multi-concept coexistence test
// ----------------------------------------------------------------------
//
// Selectivity from intrinsic iSTDP, not a kWTA post-filter
// ---------------------------------------------------------
// The previous version of this test used a k-Winners-Take-All filter
// to approximate competitive inhibition at recall. Now the network
// learns the inhibition itself: I→E synapses are governed by an
// anti-Hebbian rule (`Network.istdp`) that potentiates the I→E weight
// when E was silent during the I-pre-spike (build a wall around
// competing engrams) and depresses it when E co-fires (free your
// own assembly from self-inhibition).
//
// `MULTI_TOPK` is set to the entire R2-E population, so the readout
// is the raw recall set — no kWTA. Selectivity must therefore come
// from the network state itself, exactly as the brief calls for.

const MULTI_TRAINING_MS: f32 = 150.0;
const MULTI_RECALL_MS: f32 = 30.0;
/// Effectively no kWTA — accept every neuron that fired at least
/// once during the recall window. Selectivity comes from iSTDP.
const MULTI_TOPK: usize = R2_N;

fn r2_homeostasis_multi() -> HomeostasisParams {
    HomeostasisParams {
        eta_scale: 0.002,
        a_target: 2.0,
        tau_homeo_ms: 30.0,
        apply_every: 8,
        scale_only_down: true,
    }
}

fn r2_stdp_multi() -> StdpParams {
    let mut stdp = StdpParams::default();
    stdp.a_plus = 0.015;
    stdp.a_minus = 0.012;
    stdp.w_max = 0.8;
    stdp
}

/// iSTDP tuned for the multi-concept regime. `a_plus` is small —
/// each I-pre-spike on a silent E only mildly grows the wall —
/// while `a_minus` is large enough that any co-firing immediately
/// frees the inhibition. The asymmetry steers the network toward
/// "inhibition where E is silent, none where E co-fires", which is
/// exactly competing-assembly selectivity.
fn r2_istdp_multi() -> IStdpParams {
    IStdpParams {
        a_plus: 0.05,
        a_minus: 0.55,
        tau_minus: 30.0,
        w_min: 0.0,
        w_max: 5.0,
    }
}

fn score_for(candidates: &[(String, f32)], word: &str) -> f32 {
    candidates
        .iter()
        .find(|(w, _)| w == word)
        .map(|(_, s)| *s)
        .unwrap_or(0.0)
}

/// Run a cue and record spike *counts* per excitatory R2 neuron over
/// the entire recall window. Returns the per-neuron spike count vector
/// (length = number of excitatory neurons in R2, indexed by neuron id).
fn run_with_cue_counts(brain: &mut Brain, cue_indices: &[u32], duration_ms: f32) -> Vec<u32> {
    let r2_e = r2_excitatory_indices(brain);
    let r2_size = brain.regions[1].num_neurons();
    let mut counts = vec![0u32; r2_size];

    let mut ext1 = vec![0.0_f32; R1_N];
    for &idx in cue_indices {
        if (idx as usize) < R1_N {
            ext1[idx as usize] = DRIVE_NA;
        }
    }
    let ext2 = vec![0.0_f32; R2_N];
    let externals = vec![ext1, ext2];

    let total_steps = (duration_ms / DT) as usize;
    for _ in 0..total_steps {
        let spikes = brain.step(&externals);
        for &id in &spikes[1] {
            if r2_e.contains(&id) {
                counts[id] += 1;
            }
        }
    }
    counts
}

/// kWTA filter: pick the indices of the `k` highest-count neurons.
/// Ties broken by lower neuron index. Sorted ascending.
fn top_k_indices(counts: &[u32], k: usize) -> Vec<u32> {
    let mut paired: Vec<(u32, usize)> = counts
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, &c)| (c, i))
        .collect();
    // Sort descending by count, then ascending by index.
    paired.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    paired.truncate(k);
    let mut idx: Vec<u32> = paired.into_iter().map(|(_, i)| i as u32).collect();
    idx.sort_unstable();
    idx
}

fn run_recall(
    brain: &mut Brain,
    cue_indices: &[u32],
    dict: &EngramDictionary,
) -> (Vec<(String, f32)>, usize) {
    brain.reset_state();
    let counts = run_with_cue_counts(brain, cue_indices, MULTI_RECALL_MS);
    let kwta = top_k_indices(&counts, MULTI_TOPK);
    (dict.decode(&kwta, 0.0), kwta.len())
}

#[test]
fn multiple_overlapping_concepts_coexist() {
    let stops = ["the", "is", "on", "a", "der", "die", "und"];
    let enc = TextEncoder::with_stopwords(ENC_N, ENC_K, stops);

    let cue_hello = enc.encode("hello");
    let cue_rust = enc.encode("rust");
    let cue_world = enc.encode("world");
    let cue_hello_rust = enc.encode("hello rust");
    let cue_hello_world = enc.encode("hello world");

    let mut brain = Brain::new(DT);
    brain.add_region(build_input_region());
    brain.add_region(build_memory_region(31415));
    wire_forward(&mut brain, 27182);

    // We deliberately do NOT fingerprint here. The dictionary will be
    // populated *after* training, from the engram pattern that each
    // concept-cue actually evokes through its STDP-shaped recurrent
    // structure. That is what we'll later try to recognise during
    // pattern completion, so that's the representation the dictionary
    // should hold.

    // Phase 2 — interleaved associative training with all three forms
    // of plasticity active in R2. `reset_state` clears the transient
    // state between sentences but leaves synapse weights, so each
    // round builds on (rather than replaces) the previous.
    brain.regions[1].network.enable_stdp(r2_stdp_multi());
    brain.regions[1].network.enable_homeostasis(r2_homeostasis_multi());
    brain.regions[1].network.enable_istdp(r2_istdp_multi());

    // One interleaved pair. With moderate STDP rates a single round
    // builds each association without saturating the network; iSTDP
    // walls off cross-engram I→E weights in the same pass.
    for _ in 0..1 {
        brain.reset_state();
        let _ = run_with_cue(
            &mut brain,
            &cue_hello_rust.indices,
            MULTI_TRAINING_MS,
            TARGET_WINDOW_MS,
        );
        brain.reset_state();
        let _ = run_with_cue(
            &mut brain,
            &cue_hello_world.indices,
            MULTI_TRAINING_MS,
            TARGET_WINDOW_MS,
        );
    }

    idle(&mut brain, COOLDOWN_MS);

    // Phase 3 — register engram fingerprints *after* training with
    // every form of plasticity frozen, so the act of fingerprinting
    // does not change the network. Each concept-cue is run alone and
    // its raw recall pattern goes into the dictionary as the engram
    // for that concept. No kWTA — selectivity must come from the
    // network's own learnt I→E topology.
    brain.disable_stdp_all();
    brain.disable_homeostasis_all();
    brain.disable_istdp_all();

    let mut dict = EngramDictionary::new();
    let cues: [(&str, &[u32]); 3] = [
        ("hello", &cue_hello.indices),
        ("rust", &cue_rust.indices),
        ("world", &cue_world.indices),
    ];
    for (word, cue) in cues {
        brain.reset_state();
        let counts = run_with_cue_counts(&mut brain, cue, MULTI_RECALL_MS);
        let kwta = top_k_indices(&counts, MULTI_TOPK);
        eprintln!("engram '{word}': {} bits", kwta.len());
        dict.learn_concept(word, &kwta);
    }

    let (cands_rust, n_rust) = run_recall(&mut brain, &cue_rust.indices, &dict);
    let (cands_world, n_world) = run_recall(&mut brain, &cue_world.indices, &dict);
    let (cands_hello, n_hello) = run_recall(&mut brain, &cue_hello.indices, &dict);
    eprintln!(
        "recall set sizes: rust={} world={} hello={}",
        n_rust, n_world, n_hello,
    );

    let s_rust_hello = score_for(&cands_rust, "hello");
    let s_rust_rust = score_for(&cands_rust, "rust");
    let s_rust_world = score_for(&cands_rust, "world");

    let s_world_hello = score_for(&cands_world, "hello");
    let s_world_world = score_for(&cands_world, "world");
    let s_world_rust = score_for(&cands_world, "rust");

    let s_hello_hello = score_for(&cands_hello, "hello");
    let s_hello_rust = score_for(&cands_hello, "rust");
    let s_hello_world = score_for(&cands_hello, "world");

    eprintln!(
        "recall(rust)  → hello={:.2} rust={:.2} world={:.2}",
        s_rust_hello, s_rust_rust, s_rust_world,
    );
    eprintln!(
        "recall(world) → hello={:.2} rust={:.2} world={:.2}",
        s_world_hello, s_world_rust, s_world_world,
    );
    eprintln!(
        "recall(hello) → hello={:.2} rust={:.2} world={:.2}",
        s_hello_hello, s_hello_rust, s_hello_world,
    );

    // Direct concept retrieval: the recall pattern is built from the
    // same firing distribution as the engram, so the score sits at 1.0.
    assert!(
        s_rust_rust >= 0.95 && s_world_world >= 0.95 && s_hello_hello >= 0.95,
        "direct retrieval too weak: rust={:.2} world={:.2} hello={:.2}",
        s_rust_rust, s_world_world, s_hello_hello,
    );

    // Hub-recall: cueing the shared hub "hello" must pull both
    // associated concepts back strongly. This is the proof that
    // both engrams survived sequential training (no catastrophic
    // forgetting) AND that the recurrent network still completes
    // patterns through the hub even though iSTDP put walls around
    // the cross-engram I→E paths.
    assert!(
        s_hello_rust >= 0.70,
        "hello→rust hub recall too weak: {:.2} (want ≥ 0.70)",
        s_hello_rust,
    );
    assert!(
        s_hello_world >= 0.70,
        "hello→world hub recall too weak: {:.2} (want ≥ 0.70)",
        s_hello_world,
    );

    // Cross-concept selectivity — the bleeding cure, now intrinsic.
    // rust and world share only the hello hub; their post-training
    // engrams must stay mutually distinguishable below the relevance
    // threshold. With iSTDP enabled this happens without any kWTA
    // post-filter.
    assert!(
        s_rust_world < 0.40,
        "rust→world contamination ({:.2}, want < 0.40) — iSTDP wall too weak?",
        s_rust_world,
    );
    assert!(
        s_world_rust < 0.40,
        "world→rust contamination ({:.2}, want < 0.40) — iSTDP wall too weak?",
        s_world_rust,
    );

    // Direct cue dominates the indirect cross-concept activation.
    assert!(
        s_rust_rust > s_rust_world && s_world_world > s_world_rust,
        "direct cue did not dominate over cross-concept",
    );
}
