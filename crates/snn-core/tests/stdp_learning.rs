//! Test 3: classical STDP — when pre fires consistently *before* post,
//! the connecting weight should grow (LTP). When pre fires *after* post,
//! the weight should shrink (LTD).

use snn_core::{LifNeuron, LifParams, Network, StdpParams};

fn build_pair(initial_weight: f32) -> (Network, usize, usize) {
    let mut net = Network::new(0.1);
    net.enable_stdp(StdpParams::default());
    let pre = net.add_neuron(LifNeuron::new(LifParams::default()));
    let post = net.add_neuron(LifNeuron::new(LifParams::default()));
    net.connect(pre, post, initial_weight);
    (net, pre, post)
}

/// Drive a neuron to fire on a single timestep with a strong, brief pulse.
fn pulse(net: &mut Network, neuron: usize, ms: f32) -> bool {
    let n = net.neurons.len();
    let mut ext = vec![0.0_f32; n];
    ext[neuron] = 100.0;
    let steps = (ms / net.dt).max(1.0) as usize;
    let mut fired_target = false;
    for _ in 0..steps {
        let fired = net.step(&ext);
        if fired.contains(&neuron) {
            fired_target = true;
        }
    }
    fired_target
}

fn idle(net: &mut Network, ms: f32) {
    let n = net.neurons.len();
    let zeros = vec![0.0_f32; n];
    let steps = (ms / net.dt) as usize;
    for _ in 0..steps {
        net.step(&zeros);
    }
}

#[test]
fn pre_before_post_strengthens_weight() {
    let (mut net, pre, post) = build_pair(0.5);
    let initial = net.synapses[0].weight;

    for _ in 0..40 {
        assert!(pulse(&mut net, pre, 1.0));
        idle(&mut net, 9.0);
        assert!(pulse(&mut net, post, 1.0));
        idle(&mut net, 90.0);
    }

    let final_w = net.synapses[0].weight;
    assert!(
        final_w > initial,
        "expected LTP: weight should grow from {initial}, got {final_w}",
    );
    let _ = (pre, post);
}

#[test]
fn post_before_pre_weakens_weight() {
    let (mut net, pre, post) = build_pair(2.0);
    let initial = net.synapses[0].weight;

    for _ in 0..40 {
        assert!(pulse(&mut net, post, 1.0));
        idle(&mut net, 9.0);
        assert!(pulse(&mut net, pre, 1.0));
        idle(&mut net, 90.0);
    }

    let final_w = net.synapses[0].weight;
    assert!(
        final_w < initial,
        "expected LTD: weight should shrink from {initial}, got {final_w}",
    );
    let _ = (pre, post);
}
