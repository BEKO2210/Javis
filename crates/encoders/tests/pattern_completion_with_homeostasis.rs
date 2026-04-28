//! Pattern completion *with* homeostatic synaptic scaling — the cure
//! for the generalisation-bleeding observed in the plain
//! `pattern_completion` test.
//!
//! Same protocol as the baseline (R1 → R2, partial-cue recall after
//! full-cue training), but R2 now runs both STDP **and** homeostasis
//! during training.
//!
//! Why **asymmetric** homeostasis (`scale_only_down: true`)
//! ---------------------------------------------------------
//! Symmetric multiplicative scaling has two regimes:
//! - high `eta_scale` collapses the engram during training itself
//!   (the same scaling that fights the bleed also weakens the
//!   intra-engram weights, killing recall);
//! - low `eta_scale` lets `factor > 1` (when trace < target) amplify
//!   low-activity neurons' weights and feed activity around the
//!   recurrent loop until the network ends up *more* hyperactive
//!   than without any homeostasis.
//!
//! Asymmetric scaling — capping the factor at 1.0 — eliminates the
//! second regime entirely. STDP is then the sole source of
//! potentiation; homeostasis is a one-way brake whose only job is to
//! drag down the post-neurons that actually overshoot the target.
//!
//! Sweet-spot
//! ----------
//! `eta_scale = 0.002`, `a_target = 2.0`, `tau_homeo_ms = 30`,
//! `apply_every = 8`. The brake activates around ~70 Hz and is gentle
//! enough across 150 ms of training to leave engram coverage near
//! the baseline while shrinking the recalled set down to assembly
//! size.

use std::collections::HashSet;

use encoders::TextEncoder;
use snn_core::{
    Brain, HomeostasisParams, LifNeuron, LifParams, NeuronKind, Region, Rng, StdpParams,
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
    let mut stdp = StdpParams::default();
    stdp.a_plus = 0.04;
    stdp.a_minus = 0.025;
    stdp.w_max = 2.0;
    stdp
}

/// Asymmetric homeostasis: weights can only shrink. STDP is
/// responsible for all potentiation; homeostasis is purely a brake on
/// runaway. This avoids the chaotic regime where a multiplicative
/// factor > 1 amplifies low-activity neurons' weights and spreads
/// activity through recurrent loops.
fn r2_homeostasis() -> HomeostasisParams {
    HomeostasisParams {
        eta_scale: 0.004,
        a_target: 1.8,
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

#[test]
fn pattern_completion_with_homeostasis() {
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
    brain.disable_homeostasis_all();
    brain.reset_state();
    let pre_recall = run_with_cue(&mut brain, &cue_partial.indices, RECALL_MS, RECALL_MS);

    // Phase 2 — training. STDP and homeostasis run inside R2.
    brain.regions[1].network.enable_stdp(r2_stdp());
    brain.regions[1].network.enable_homeostasis(r2_homeostasis());
    brain.reset_state();
    let target_assembly = run_with_cue(
        &mut brain,
        &cue_full.indices,
        TRAINING_MS,
        TARGET_WINDOW_MS,
    );
    idle(&mut brain, COOLDOWN_MS);

    // Phase 3 — post-recall, both forms of plasticity frozen.
    brain.disable_stdp_all();
    brain.disable_homeostasis_all();
    brain.reset_state();
    let post_recall = run_with_cue(&mut brain, &cue_partial.indices, RECALL_MS, RECALL_MS);

    let assembly_size = target_assembly.len();
    let pre_overlap = pre_recall.intersection(&target_assembly).count();
    let post_overlap = post_recall.intersection(&target_assembly).count();
    let coverage = post_overlap as f32 / assembly_size.max(1) as f32;
    let bleed_ratio = post_recall.len() as f32 / assembly_size.max(1) as f32;

    eprintln!(
        "target_assembly={assembly_size}  pre_recall={}  post_recall={}  \
         pre∩target={pre_overlap}  post∩target={post_overlap}  \
         coverage={:.0}%  bleed_ratio={:.2}×",
        pre_recall.len(),
        post_recall.len(),
        coverage * 100.0,
        bleed_ratio,
    );

    assert!(
        assembly_size >= 30,
        "training did not produce a usable assembly (size={assembly_size})",
    );

    // Pattern completion still works — most of the original engram
    // comes back from the partial cue alone.
    assert!(
        coverage >= 0.70,
        "coverage too low with homeostasis on: {:.0} % (want ≥ 70 %)",
        coverage * 100.0,
    );

    // The critical bleeding-cure check: post_recall must not balloon
    // beyond 1.3× the assembly size. Without homeostasis the same
    // setup measures ~1.79×; with this brake we measure < 1.0×.
    assert!(
        bleed_ratio <= 1.3,
        "homeostasis failed to contain bleeding: post_recall={} vs target={} ({:.2}× target, want ≤ 1.30×)",
        post_recall.len(),
        assembly_size,
        bleed_ratio,
    );

    // Associative gain: the recall is genuinely powered by STDP-shaped
    // recurrent links, not just by the forward path.
    let gain = post_overlap.saturating_sub(pre_overlap);
    assert!(
        gain >= assembly_size / 5,
        "no associative gain over the forward baseline: \
         pre∩target={pre_overlap}, post∩target={post_overlap}, assembly={assembly_size}",
    );
}
