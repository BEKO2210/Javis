//! Multi-Region pattern completion: Javis's first associative-recall test.
//!
//! Two regions live in one `Brain`:
//! - **R1 (input)** — 1000 purely excitatory LIF neurons, no internal
//!   recurrent connections. R1 is a relay: SDR-addressed neurons receive
//!   a constant external drive and feed their spikes through long-range
//!   axons into R2.
//! - **R2 (memory)** — 2000 neurons, 80/20 E/I, p=0.1 random recurrent
//!   wiring, STDP enabled. Internal inhibition gives R2 the
//!   winner-takes-all dynamics that make recalled patterns sparse and
//!   stable instead of running away.
//!
//! Forward routing R1 → R2 is implemented through the existing
//! address-event router with a 2 ms delay and a fan-out of `FAN_OUT` per
//! R1 neuron. R1 senders may target both R2-E and R2-I cells (sampling
//! is random, so the 80/20 split happens automatically).
//!
//! Protocol
//! --------
//! 1. **Pre-recall** (STDP off): drive the partial cue "hello" into R1
//!    for `RECALL_MS`. Record which R2-E neurons fire — this is what the
//!    *forward path alone* can produce.
//! 2. **Training** (STDP on): drive the full cue "hello rust" into R1
//!    for `TRAINING_MS`. The R2 neurons co-active in the last
//!    `TARGET_WINDOW_MS` form `target_assembly`. Cool down for
//!    `COOLDOWN_MS`.
//! 3. **Post-recall** (STDP off, fresh state): drive the partial cue
//!    "hello" again. Record the firing R2-E set.
//!
//! Pass conditions
//! ---------------
//! - `target_assembly` is non-trivial (≥ 30 neurons).
//! - The post-recall ∩ target_assembly cardinality grows by at least
//!   `assembly_size / 5` over the pre-recall baseline. Only the
//!   STDP-shaped recurrent weights inside R2 changed between the two
//!   measurements, so any gain is *associative completion*.
//! - At least 30 % of the original assembly is recalled by the partial
//!   cue.

use std::collections::HashSet;

use encoders::TextEncoder;
use snn_core::{Brain, LifNeuron, LifParams, NeuronKind, Region, Rng, StdpParams};

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
    StdpParams {
        a_plus: 0.04,
        a_minus: 0.025,
        w_max: 2.0,
        ..StdpParams::default()
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

/// Run `duration_ms` of simulation with a constant SDR-shaped external
/// drive on R1 and zero external on R2. Returns the R2-E neurons that
/// fired during the last `record_window_ms`.
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

#[test]
fn r2_completes_partial_cue_after_training() {
    let stops = ["the", "is", "on", "a", "der", "die", "und"];
    let enc = TextEncoder::with_stopwords(ENC_N, ENC_K, stops);

    let cue_full = enc.encode("hello rust");
    let cue_partial = enc.encode("hello");

    let mut brain = Brain::new(DT);
    brain.add_region(build_input_region());
    brain.add_region(build_memory_region(2027));
    wire_forward(&mut brain, 2028);

    // Phase 1 — pre-recall: forward-path-only baseline.
    brain.disable_stdp_all();
    brain.reset_state();
    let pre_recall = run_with_cue(&mut brain, &cue_partial.indices, RECALL_MS, RECALL_MS);

    // Phase 2 — training with the full cue, STDP on inside R2.
    brain.regions[1].network.enable_stdp(r2_stdp());
    brain.reset_state();
    let target_assembly = run_with_cue(
        &mut brain,
        &cue_full.indices,
        TRAINING_MS,
        TARGET_WINDOW_MS,
    );
    idle(&mut brain, COOLDOWN_MS);

    // Phase 3 — post-recall with the partial cue, STDP frozen, fresh state.
    brain.disable_stdp_all();
    brain.reset_state();
    let post_recall = run_with_cue(&mut brain, &cue_partial.indices, RECALL_MS, RECALL_MS);

    let assembly_size = target_assembly.len();
    let pre_overlap = pre_recall.intersection(&target_assembly).count();
    let post_overlap = post_recall.intersection(&target_assembly).count();

    eprintln!(
        "target_assembly={assembly_size}  pre_recall={}  post_recall={}  \
         pre∩target={pre_overlap}  post∩target={post_overlap}  \
         coverage={:.0}%",
        pre_recall.len(),
        post_recall.len(),
        100.0 * post_overlap as f32 / assembly_size.max(1) as f32,
    );

    assert!(
        assembly_size >= 30,
        "training did not produce a usable assembly (size={assembly_size})",
    );

    let gain = post_overlap.saturating_sub(pre_overlap);
    assert!(
        gain >= assembly_size / 5,
        "no associative gain over forward baseline: \
         pre∩target={pre_overlap}, post∩target={post_overlap}, \
         assembly={assembly_size}",
    );

    let coverage = post_overlap as f32 / assembly_size as f32;
    assert!(
        coverage >= 0.30,
        "pattern completion too weak: {:.0} % of assembly recalled (want ≥ 30 %)",
        coverage * 100.0,
    );
}
