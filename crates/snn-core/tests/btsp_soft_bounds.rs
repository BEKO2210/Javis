//! BTSP-style soft-bounded STDP.
//!
//! With `soft_bounds = true` the weight update factors `(w_max - w)`
//! and `(w - w_min)` shrink as `w` approaches each bound. The result
//! is that weights settle into a smooth distribution between `w_min`
//! and `w_max` instead of piling up at the clamp edges. Default
//! behaviour (hard clamp) is unchanged so all existing tests stay
//! grün.

use snn_core::{LifNeuron, LifParams, Network, StdpParams};

fn pulse(net: &mut Network, neuron: usize, ms: f32) -> bool {
    let n = net.neurons.len();
    let mut ext = vec![0.0_f32; n];
    ext[neuron] = 100.0;
    let steps = (ms / net.dt).max(1.0) as usize;
    let mut fired = false;
    for _ in 0..steps {
        if net.step(&ext).contains(&neuron) {
            fired = true;
        }
    }
    fired
}

fn idle(net: &mut Network, ms: f32) {
    let n = net.neurons.len();
    let z = vec![0.0_f32; n];
    let steps = (ms / net.dt) as usize;
    for _ in 0..steps {
        net.step(&z);
    }
}

fn build_pair(initial_weight: f32, soft_bounds: bool) -> (Network, usize, usize) {
    let mut net = Network::new(0.1);
    let mut p = StdpParams::default();
    p.a_plus = 0.05;
    p.a_minus = 0.025;
    p.w_min = 0.0;
    p.w_max = 1.0;
    p.soft_bounds = soft_bounds;
    net.enable_stdp(p);
    let pre = net.add_neuron(LifNeuron::new(LifParams::default()));
    let post = net.add_neuron(LifNeuron::new(LifParams::default()));
    net.connect(pre, post, initial_weight);
    (net, pre, post)
}

/// Soft-bounded LTP must converge from below w_max but never overshoot
/// it, even under heavy pre-before-post repetition.
#[test]
fn soft_bounds_ltp_does_not_exceed_w_max() {
    let (mut net, pre, post) = build_pair(0.5, true);
    for _ in 0..200 {
        assert!(pulse(&mut net, pre, 1.0));
        idle(&mut net, 9.0);
        assert!(pulse(&mut net, post, 1.0));
        idle(&mut net, 90.0);
    }
    let final_w = net.synapses[0].weight;
    let p = net.stdp.unwrap();
    assert!(
        final_w > 0.5 && final_w < p.w_max,
        "soft-bound LTP should converge below w_max: {final_w} (w_max = {})",
        p.w_max,
    );
    let _ = (pre, post);
}

/// Soft-bounded LTD must drive the weight towards w_min but never
/// below it, regardless of how many post-before-pre pairs we feed.
#[test]
fn soft_bounds_ltd_does_not_undershoot_w_min() {
    let (mut net, pre, post) = build_pair(0.9, true);
    for _ in 0..200 {
        assert!(pulse(&mut net, post, 1.0));
        idle(&mut net, 9.0);
        assert!(pulse(&mut net, pre, 1.0));
        idle(&mut net, 90.0);
    }
    let final_w = net.synapses[0].weight;
    let p = net.stdp.unwrap();
    assert!(
        final_w < 0.9 && final_w > p.w_min,
        "soft-bound LTD should converge above w_min: {final_w} (w_min = {})",
        p.w_min,
    );
    let _ = (pre, post);
}

/// Soft-bound LTP should settle below w_max while hard-bound LTP
/// pinned at the same a_plus / a_minus eventually saturates *at*
/// w_max. The soft-bound run must end strictly below the hard-bound
/// run on the same training schedule.
#[test]
fn soft_bounds_settle_lower_than_hard_clamp() {
    fn run(soft: bool) -> f32 {
        let (mut net, pre, post) = build_pair(0.4, soft);
        for _ in 0..400 {
            let _ = pulse(&mut net, pre, 1.0);
            idle(&mut net, 9.0);
            let _ = pulse(&mut net, post, 1.0);
            idle(&mut net, 90.0);
        }
        let _ = (pre, post);
        net.synapses[0].weight
    }
    let hard = run(false);
    let soft = run(true);
    eprintln!("hard-bound final w = {hard:.4} | soft-bound final w = {soft:.4}");
    assert!(
        soft < hard,
        "soft-bound LTP should settle strictly below hard-clamped LTP: hard={hard} soft={soft}",
    );
}
