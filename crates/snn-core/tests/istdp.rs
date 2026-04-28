//! Inhibitory STDP rule unit tests.
//!
//! Two minimal pair-firing experiments verify the two regimes of the
//! anti-Hebbian update applied on every I-pre-spike:
//!     dw = a_plus - a_minus * post_trace_e[post]
//!
//! 1. **Co-activation → LTD.** When the post E neuron has just fired
//!    and its post-trace is high, the second term dominates and the
//!    I→E weight shrinks below its initial value.
//!
//! 2. **Pre-only → LTP.** When the post E neuron stays silent during
//!    the I pre-spike, the post-trace is ≈ 0 and only the baseline
//!    `a_plus` term applies — the I→E weight grows above its initial
//!    value.

use snn_core::{IStdpParams, LifNeuron, LifParams, Network};

const DT: f32 = 0.1;

fn istdp_params() -> IStdpParams {
    IStdpParams {
        a_plus: 0.02,
        a_minus: 0.05,
        tau_minus: 20.0,
        w_min: 0.0,
        w_max: 5.0,
    }
}

/// Drive a target neuron to fire on this very step using a strong
/// external pulse. Returns `true` if the neuron actually fired.
fn pulse(net: &mut Network, neuron: usize) -> bool {
    let n = net.neurons.len();
    let mut ext = vec![0.0_f32; n];
    ext[neuron] = 800.0;
    let fired = net.step(&ext);
    fired.contains(&neuron)
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
fn istdp_weakens_synapse_on_coactivation() {
    let mut net = Network::new(DT);
    net.enable_istdp(istdp_params());

    let i = net.add_neuron(LifNeuron::inhibitory(LifParams::default()));
    let e = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    let eid = net.connect(i, e, 1.0);
    let initial = net.synapses[eid].weight;

    // Repeat: fire E first, then I shortly after — E's post-trace is
    // still elevated when the I-pre-spike triggers the iSTDP update,
    // so the LTD term dominates and the weight drops.
    for _ in 0..50 {
        assert!(pulse(&mut net, e));
        idle(&mut net, 1.0);
        assert!(pulse(&mut net, i));
        idle(&mut net, 80.0);
    }

    let final_w = net.synapses[eid].weight;
    assert!(
        final_w < initial,
        "expected LTD on co-activation: weight should shrink, got {initial} -> {final_w}",
    );
}

#[test]
fn istdp_strengthens_synapse_on_pre_only() {
    let mut net = Network::new(DT);
    net.enable_istdp(istdp_params());

    let i = net.add_neuron(LifNeuron::inhibitory(LifParams::default()));
    let _e = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    let eid = net.connect(i, 1, 1.0);
    let initial = net.synapses[eid].weight;

    // Fire I alone; E never spikes. post_trace stays at 0, so the
    // update reduces to dw = a_plus per pre-spike — pure LTP.
    for _ in 0..50 {
        assert!(pulse(&mut net, i));
        idle(&mut net, 80.0);
    }

    let final_w = net.synapses[eid].weight;
    assert!(
        final_w > initial,
        "expected LTP on pre-only: weight should grow, got {initial} -> {final_w}",
    );
}

#[test]
fn istdp_off_by_default_does_not_touch_weights() {
    let mut net = Network::new(DT);

    let i = net.add_neuron(LifNeuron::inhibitory(LifParams::default()));
    let _e = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    let eid = net.connect(i, 1, 1.0);
    let initial = net.synapses[eid].weight;

    for _ in 0..50 {
        let _ = pulse(&mut net, i);
        idle(&mut net, 80.0);
    }

    assert!(net.istdp.is_none());
    assert_eq!(net.synapses[eid].weight, initial);
}
