//! Test 4: assembly formation via STDP — the core "memory" demonstration.
//!
//! Setup
//! -----
//! - 100 LIF neurons, randomly recurrently connected (p_connect = 0.1)
//! - STDP enabled with slightly amplified `a_plus` so changes are visible
//! - Pattern A := neurons 0..10, Pattern B := neurons 10..20
//! - Pattern C := neurons 50..60   (control, never paired with B)
//!
//! Procedure
//! ---------
//! 1. Pre-test recall: drive only A, count spikes by B neurons. STDP off.
//! 2. Pre-test control: drive only C, count spikes by B neurons. STDP off.
//! 3. Train: 100 trials of "A then B" stimulation with overlap. STDP on.
//! 4. Post-test recall: drive only A, count B spikes. STDP off.
//! 5. Post-test control: drive only C, count B spikes. STDP off.
//!
//! Pass condition: post-A B-spikes are noticeably more than pre-A B-spikes
//! AND noticeably more than post-C B-spikes (true association, not
//! generic hyperactivity).

use snn_core::{LifNeuron, LifParams, Network, PoissonInput, Rng, StdpParams};

const N: usize = 100;
const PATTERN_A: std::ops::Range<usize> = 0..10;
const PATTERN_B: std::ops::Range<usize> = 10..20;
const PATTERN_C: std::ops::Range<usize> = 50..60;
const DT: f32 = 0.1;

fn build_region(seed: u64) -> Network {
    let mut rng = Rng::new(seed);
    let mut net = Network::new(DT);

    for _ in 0..N {
        net.add_neuron(LifNeuron::new(LifParams::default()));
    }

    let p_connect = 0.1_f32;
    let w_lo = 0.05_f32;
    let w_hi = 0.30_f32;
    for pre in 0..N {
        for post in 0..N {
            if pre == post {
                continue;
            }
            if rng.bernoulli(p_connect) {
                let w = rng.range_f32(w_lo, w_hi);
                net.connect(pre, post, w);
            }
        }
    }
    net
}

fn idle(net: &mut Network, ms: f32) {
    let zeros = vec![0.0_f32; net.neurons.len()];
    let steps = (ms / net.dt) as usize;
    for _ in 0..steps {
        net.step(&zeros);
    }
}

/// Drive a single pattern via Poisson input at high rate. Returns spike
/// counts per index range (drive-pattern, target-pattern).
fn drive_and_count(
    net: &mut Network,
    drive: std::ops::Range<usize>,
    target: std::ops::Range<usize>,
    rng: &mut Rng,
    duration_ms: f32,
    rate_hz: f32,
    current_per_spike: f32,
) -> (usize, usize) {
    let gens: Vec<PoissonInput> = drive
        .clone()
        .map(|i| PoissonInput {
            target: i,
            rate_hz,
            current_per_spike,
        })
        .collect();
    let steps = (duration_ms / net.dt) as usize;
    let n = net.neurons.len();
    let mut external = vec![0.0_f32; n];
    let mut drive_spikes = 0usize;
    let mut target_spikes = 0usize;
    for _ in 0..steps {
        external.iter_mut().for_each(|x| *x = 0.0);
        snn_core::poisson::drive(&gens, &mut external, net.dt, rng);
        let fired = net.step(&external);
        for &id in &fired {
            if drive.contains(&id) {
                drive_spikes += 1;
            }
            if target.contains(&id) {
                target_spikes += 1;
            }
        }
    }
    (drive_spikes, target_spikes)
}

/// Training trial: drive A then B with overlap so A fires before B.
/// STDP must be enabled on the caller side.
fn training_trial(net: &mut Network, rng: &mut Rng) {
    let a_gen: Vec<PoissonInput> = PATTERN_A
        .map(|i| PoissonInput {
            target: i,
            rate_hz: 500.0,
            current_per_spike: 80.0,
        })
        .collect();
    let b_gen: Vec<PoissonInput> = PATTERN_B
        .map(|i| PoissonInput {
            target: i,
            rate_hz: 500.0,
            current_per_spike: 80.0,
        })
        .collect();

    let n = net.neurons.len();
    let mut external = vec![0.0_f32; n];

    // Phase 1: A only, 0..10 ms.
    let s = (10.0 / net.dt) as usize;
    for _ in 0..s {
        external.iter_mut().for_each(|x| *x = 0.0);
        snn_core::poisson::drive(&a_gen, &mut external, net.dt, rng);
        net.step(&external);
    }
    // Phase 2: A + B, 10..20 ms.
    for _ in 0..s {
        external.iter_mut().for_each(|x| *x = 0.0);
        snn_core::poisson::drive(&a_gen, &mut external, net.dt, rng);
        snn_core::poisson::drive(&b_gen, &mut external, net.dt, rng);
        net.step(&external);
    }
    // Phase 3: B only, 20..30 ms.
    for _ in 0..s {
        external.iter_mut().for_each(|x| *x = 0.0);
        snn_core::poisson::drive(&b_gen, &mut external, net.dt, rng);
        net.step(&external);
    }
    // Idle 70 ms between trials so STDP traces decay.
    idle(net, 70.0);
}

#[test]
fn pattern_b_recalls_after_training_via_pattern_a() {
    let mut net = build_region(2026);
    let mut rng = Rng::new(7);

    // 1) Pre-test recall.
    net.reset_state();
    let (pre_a_drv, pre_a_b) =
        drive_and_count(&mut net, PATTERN_A, PATTERN_B, &mut rng, 50.0, 500.0, 80.0);

    // 2) Pre-test control (different pattern → B).
    net.reset_state();
    let (pre_c_drv, pre_c_b) =
        drive_and_count(&mut net, PATTERN_C, PATTERN_B, &mut rng, 50.0, 500.0, 80.0);

    // 3) Train.
    net.enable_stdp(StdpParams {
        a_plus: 0.05,
        a_minus: 0.025,
        ..StdpParams::default()
    });
    net.reset_state();
    for _ in 0..100 {
        training_trial(&mut net, &mut rng);
    }
    net.disable_stdp();

    // 4) Post-test recall.
    net.reset_state();
    let (post_a_drv, post_a_b) =
        drive_and_count(&mut net, PATTERN_A, PATTERN_B, &mut rng, 50.0, 500.0, 80.0);

    // 5) Post-test control.
    net.reset_state();
    let (post_c_drv, post_c_b) =
        drive_and_count(&mut net, PATTERN_C, PATTERN_B, &mut rng, 50.0, 500.0, 80.0);

    eprintln!("drive  pre_A={pre_a_drv} pre_C={pre_c_drv} post_A={post_a_drv} post_C={post_c_drv}",);
    eprintln!("B-out  pre_A={pre_a_b}  pre_C={pre_c_b}  post_A={post_a_b}  post_C={post_c_b}",);

    assert!(
        post_a_b >= pre_a_b + 5,
        "expected post-training A→B recall to grow noticeably (pre={pre_a_b}, post={post_a_b})",
    );
    assert!(
        post_a_b > post_c_b,
        "expected A→B recall stronger than control C→B (A={post_a_b}, C={post_c_b})",
    );
}
