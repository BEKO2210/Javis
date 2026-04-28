//! Test 6: signal hand-off between two E/I-balanced regions.
//!
//! Setup
//! -----
//! - Brain holds two 1000-neuron regions, each 80/20 E/I, p=0.1 random
//!   recurrent connectivity, STDP enabled.
//! - 1 % of region 1's excitatory neurons are designated "projection
//!   neurons". Each one sends a sparse fan-out of `FAN_OUT` edges to
//!   randomly chosen excitatory targets in region 2, with a per-edge
//!   delay drawn uniformly in [2, 5] ms.
//! - Region 1 gets a Poisson background drive; region 2 receives no
//!   external input — its only stimulus is the inter-region traffic.
//!
//! Hypotheses
//! ----------
//! 1. With inter-region edges present, region 2 fires noticeably more
//!    than the no-edges control (signal transfer works).
//! 2. Region 2's mean rate stays in the asynchronous-irregular regime
//!    (< 30 Hz), proving its internal inhibition contains the imported
//!    activity rather than going into runaway.

use snn_core::{
    Brain, LifNeuron, LifParams, NeuronKind, PoissonInput, Region, Rng, StdpParams,
};

const DT: f32 = 0.1;
const N: usize = 1000;
const P_CONNECT: f32 = 0.1;
const INH_FRACTION: f32 = 0.20;
const SIM_MS: f32 = 1000.0;

const PROJ_FRACTION: f32 = 0.01;
const FAN_OUT: usize = 15;
const INTER_WEIGHT: f32 = 12.0;
const DELAY_LO: f32 = 2.0;
const DELAY_HI: f32 = 5.0;

fn build_region(name: &str, seed: u64) -> Region {
    let mut rng = Rng::new(seed);
    let mut region = Region::new(name, DT);
    let net = &mut region.network;

    let mut stdp = StdpParams::default();
    stdp.a_plus = 0.05;
    stdp.a_minus = 0.025;
    net.enable_stdp(stdp);

    let n_inh = (N as f32 * INH_FRACTION) as usize;
    let n_exc = N - n_inh;
    for _ in 0..n_exc {
        net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    }
    for _ in 0..n_inh {
        net.add_neuron(LifNeuron::inhibitory(LifParams::default()));
    }

    let g_exc = 0.20_f32;
    let g_inh = 0.80_f32;
    for pre in 0..N {
        let g = match net.neurons[pre].kind {
            NeuronKind::Excitatory => g_exc,
            NeuronKind::Inhibitory => g_inh,
        };
        for post in 0..N {
            if pre == post {
                continue;
            }
            if rng.bernoulli(P_CONNECT) {
                let w = rng.range_f32(0.5 * g, 1.0 * g);
                net.connect(pre, post, w);
            }
        }
    }
    region
}

fn excitatory_indices(region: &Region) -> Vec<usize> {
    region
        .network
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.kind == NeuronKind::Excitatory)
        .map(|(i, _)| i)
        .collect()
}

/// Build a two-region brain. If `wire_inter` is false, the brain has no
/// long-range edges — the control condition.
fn build_brain(seed: u64, wire_inter: bool) -> Brain {
    let mut brain = Brain::new(DT);
    brain.add_region(build_region("R1", seed));
    brain.add_region(build_region("R2", seed.wrapping_add(101)));

    if wire_inter {
        let mut rng = Rng::new(seed.wrapping_add(7777));
        let r1_exc = excitatory_indices(&brain.regions[0]);
        let r2_exc = excitatory_indices(&brain.regions[1]);

        let n_proj = (r1_exc.len() as f32 * PROJ_FRACTION).max(1.0) as usize;
        for k in 0..n_proj {
            let src = r1_exc[k];
            for _ in 0..FAN_OUT {
                let pick = (rng.next_u64() as usize) % r2_exc.len();
                let dst = r2_exc[pick];
                let delay = rng.range_f32(DELAY_LO, DELAY_HI);
                brain.connect(0, src, 1, dst, INTER_WEIGHT, delay);
            }
        }
    }

    brain
}

fn run(brain: &mut Brain, drive_rng_seed: u64) -> (u64, u64, u64, u64) {
    let r1_n = brain.regions[0].num_neurons();
    let r2_n = brain.regions[1].num_neurons();

    // Region 1 gets a Poisson background; region 2 gets nothing.
    let r1_gens: Vec<PoissonInput> = (0..r1_n)
        .map(|i| PoissonInput { target: i, rate_hz: 80.0, current_per_spike: 80.0 })
        .collect();

    let mut rng = Rng::new(drive_rng_seed);
    let steps = (SIM_MS / DT) as usize;
    let mut ext1 = vec![0.0_f32; r1_n];
    let ext2 = vec![0.0_f32; r2_n];

    let mut r1_e = 0u64;
    let mut r1_i = 0u64;
    let mut r2_e = 0u64;
    let mut r2_i = 0u64;

    for _ in 0..steps {
        ext1.iter_mut().for_each(|x| *x = 0.0);
        snn_core::poisson::drive(&r1_gens, &mut ext1, DT, &mut rng);
        let externals = vec![ext1.clone(), ext2.clone()];
        let spikes = brain.step(&externals);

        for &id in &spikes[0] {
            if brain.regions[0].network.neurons[id].kind == NeuronKind::Excitatory {
                r1_e += 1;
            } else {
                r1_i += 1;
            }
        }
        for &id in &spikes[1] {
            if brain.regions[1].network.neurons[id].kind == NeuronKind::Excitatory {
                r2_e += 1;
            } else {
                r2_i += 1;
            }
        }
    }

    (r1_e, r1_i, r2_e, r2_i)
}

#[test]
fn signal_transfers_between_regions_without_runaway() {
    // Control: no inter-region edges.
    let mut brain_ctrl = build_brain(2026, false);
    let (c_r1e, c_r1i, c_r2e, c_r2i) = run(&mut brain_ctrl, 11);

    // Wired: 1 % of R1's E neurons project sparsely into R2.
    let mut brain = build_brain(2026, true);
    let (r1e, r1i, r2e, r2i) = run(&mut brain, 11);

    let n_exc = (N as f32 * (1.0 - INH_FRACTION)) as f32;
    let n_inh = N as f32 - n_exc;

    let rate_r1e = r1e as f32 / n_exc / (SIM_MS / 1000.0);
    let rate_r2e = r2e as f32 / n_exc / (SIM_MS / 1000.0);
    let rate_r2i = r2i as f32 / n_inh / (SIM_MS / 1000.0);
    let rate_ctrl_r2e = c_r2e as f32 / n_exc / (SIM_MS / 1000.0);

    eprintln!(
        "control  R1 spikes={c_r1e}+{c_r1i}  R2 spikes={c_r2e}+{c_r2i}  R2 rate_E={rate_ctrl_r2e:.2} Hz",
    );
    eprintln!(
        "wired    R1 spikes={r1e}+{r1i}  R2 spikes={r2e}+{r2i}  \
         R1 rate_E={rate_r1e:.2} Hz  R2 rate_E={rate_r2e:.2} Hz  R2 rate_I={rate_r2i:.2} Hz  \
         events_delivered={}",
        brain.events_delivered,
    );

    // Hypothesis 1: signal actually crosses.
    assert!(
        r2e + r2i > c_r2e + c_r2i + 50,
        "R2 should fire noticeably more with inter-region edges (ctrl={}, wired={})",
        c_r2e + c_r2i,
        r2e + r2i,
    );

    // Hypothesis 2: R2 stays in the asynchronous-irregular regime.
    assert!(rate_r2e < 30.0, "R2 E-rate runaway: {rate_r2e} Hz");
    assert!(rate_r2i < 30.0, "R2 I-rate runaway: {rate_r2i} Hz");

    // And R1 itself should be active under its drive.
    assert!(r1e + r1i > 100, "R1 looks dead under drive");
}
