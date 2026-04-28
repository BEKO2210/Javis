//! Homeostatic synaptic scaling.
//!
//! Two properties to lock in:
//! 1. A neuron driven well above its activity target has its incoming
//!    excitatory weights actively scaled *down*.
//! 2. The scaling is purely multiplicative — the *ratio* between two
//!    incoming weights is preserved across the scaling pass, even as
//!    their absolute magnitudes shrink dramatically.

use snn_core::{HomeostasisParams, LifNeuron, LifParams, Network};

const DT: f32 = 0.1;

fn aggressive_homeostasis() -> HomeostasisParams {
    let mut h = HomeostasisParams::default();
    h.eta_scale = 0.01;        // 1% of the gap per scaling pass
    h.a_target = 1.0;          // very low — easy to overshoot
    h.tau_homeo_ms = 500.0;    // shorter than default so the test runs fast
    h.apply_every = 100;       // every 10 ms
    h
}

/// Smaller `eta_scale` so the multiplicative factor stays clearly above
/// zero across the full simulation and weights end up in a numerically
/// well-behaved range — necessary when we care about *relative* weights
/// surviving f32 round-off across thousands of multiplications.
fn gentle_homeostasis() -> HomeostasisParams {
    let mut h = aggressive_homeostasis();
    h.eta_scale = 0.0001;
    h
}

fn run_steps(net: &mut Network, external: &[f32], steps: usize) {
    for _ in 0..steps {
        net.step(external);
    }
}

#[test]
fn homeostasis_scales_down_hyperactive_neuron() {
    let mut net = Network::new(DT);
    net.enable_homeostasis(aggressive_homeostasis());

    // One silent presynaptic neuron and one strongly-driven post.
    let _pre = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    let post = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    let eid = net.connect(0, 1, 1.0);
    let initial_w = net.synapses[eid].weight;

    // Drive `post` directly via external current. A 5 nA constant input
    // pushes the LIF default neuron well above rheobase (~1.5 nA), so it
    // fires steadily — way above the homeostatic target of 1.0 trace
    // units (≈ 2 Hz with tau_homeo = 500 ms).
    let mut external = vec![0.0_f32; 2];
    external[post] = 5.0;

    run_steps(&mut net, &external, 50_000); // 5 simulated seconds

    let final_w = net.synapses[eid].weight;
    let trace = net.neurons[post].activity_trace;

    eprintln!("scale-down: w {initial_w} -> {final_w}, post.activity_trace = {trace}");

    assert!(
        trace > 5.0,
        "post never became hyperactive (trace={trace}); test premise broken",
    );
    assert!(
        final_w < initial_w * 0.5,
        "weight did not scale down enough: {initial_w} -> {final_w}",
    );
    assert!(
        final_w > 0.0,
        "weight collapsed to zero — scaling overshot",
    );
}

#[test]
fn homeostasis_preserves_relative_weights() {
    let mut net = Network::new(DT);
    net.enable_homeostasis(gentle_homeostasis());

    // Two silent pre-neurons feeding the same hyperactive post-neuron.
    let _pre_strong = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    let _pre_weak = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    let post = net.add_neuron(LifNeuron::excitatory(LifParams::default()));

    let w_strong_initial = 2.0_f32;
    let w_weak_initial = 0.5_f32;
    let eid_strong = net.connect(0, post, w_strong_initial);
    let eid_weak = net.connect(1, post, w_weak_initial);

    let initial_ratio = w_strong_initial / w_weak_initial; // 4.0

    let mut external = vec![0.0_f32; 3];
    external[post] = 5.0;

    run_steps(&mut net, &external, 50_000); // 5 simulated seconds

    let w_strong_final = net.synapses[eid_strong].weight;
    let w_weak_final = net.synapses[eid_weak].weight;
    let final_ratio = w_strong_final / w_weak_final;

    let drift = (final_ratio - initial_ratio).abs() / initial_ratio;
    eprintln!(
        "preserve-ratio: strong {w_strong_initial} -> {w_strong_final}, \
         weak {w_weak_initial} -> {w_weak_final}, ratio {initial_ratio} -> {final_ratio} (drift {drift:.2e})",
    );

    // Both must have actually scaled down.
    assert!(
        w_strong_final < w_strong_initial * 0.5,
        "strong synapse did not scale down: {w_strong_initial} -> {w_strong_final}",
    );
    assert!(
        w_weak_final < w_weak_initial * 0.5,
        "weak synapse did not scale down: {w_weak_initial} -> {w_weak_final}",
    );

    // And their ratio must survive the scaling almost exactly. We allow
    // a tiny numerical drift (1e-3) for f32 round-off across thousands of
    // multiplications.
    assert!(
        drift < 1e-3,
        "ratio drifted: {initial_ratio} -> {final_ratio} (rel. drift {drift:.3e})",
    );
}

#[test]
fn homeostasis_off_by_default_does_not_touch_weights() {
    // The defining contract: without `enable_homeostasis`, weights move
    // only because of STDP (or not at all). This is what protects all
    // older tests from regression.
    let mut net = Network::new(DT);

    let _pre = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    let post = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    let eid = net.connect(0, post, 1.0);
    let initial_w = net.synapses[eid].weight;

    let mut external = vec![0.0_f32; 2];
    external[post] = 5.0;

    run_steps(&mut net, &external, 50_000);

    assert!(net.homeostasis.is_none());
    let final_w = net.synapses[eid].weight;
    assert_eq!(
        final_w, initial_w,
        "weight changed without homeostasis (or STDP) enabled",
    );
}
