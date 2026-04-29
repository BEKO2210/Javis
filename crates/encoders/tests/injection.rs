//! SDR → SNN injection.
//!
//! After injecting an SDR with sufficiently strong drive into a fresh
//! network and running one timestep, the firing set must equal exactly
//! the SDR index set: the encoder addresses the network by neuron index,
//! nothing else fires, and every targeted neuron does fire.

use encoders::{inject_sdr, TextEncoder};
use snn_core::{LifNeuron, LifParams, Network};

const N: u32 = 2048;
const K: u32 = 20;
const DT: f32 = 0.1;

/// Build a fresh, unconnected network with `n` excitatory LIF neurons.
fn fresh_network(n: u32) -> Network {
    let mut net = Network::new(DT);
    for _ in 0..n {
        net.add_neuron(LifNeuron::new(LifParams::default()));
    }
    net
}

#[test]
fn injected_sdr_fires_exactly_its_neurons() {
    let mut net = fresh_network(N);
    let enc = TextEncoder::new(N, K);

    let sdr = enc.encode("hello world");
    assert_eq!(sdr.len(), 2 * K as usize, "two distinct words → 2·k bits");

    // Drive strong enough to fire in a single 0.1 ms step.
    // dV per step ≈ (dt/τ_m) · R · drive  =  0.005 · 10 · 700  = 35 mV
    // → V crosses threshold (15 mV from rest) in this step.
    inject_sdr(&mut net, &sdr.indices, 700.0);

    let fired = net.step(&[]);

    let mut fired_sorted = fired.iter().map(|&i| i as u32).collect::<Vec<_>>();
    fired_sorted.sort_unstable();

    assert_eq!(
        fired_sorted, sdr.indices,
        "firing set must equal the SDR exactly",
    );
}

#[test]
fn untargeted_neurons_remain_at_rest() {
    let mut net = fresh_network(N);
    let enc = TextEncoder::new(N, K);

    let sdr = enc.encode_word("hello");
    inject_sdr(&mut net, &sdr.indices, 400.0);
    let _fired = net.step(&[]);

    let active: std::collections::HashSet<u32> = sdr.indices.iter().copied().collect();
    let v_rest = LifParams::default().v_rest;

    let mut wrong = 0;
    for i in 0..net.neurons.len() {
        if active.contains(&(i as u32)) {
            continue;
        }
        // Untargeted neurons must be exactly at V_rest after one step
        // (no external input, no synapses, no Poisson noise).
        if (net.v[i] - v_rest).abs() > 1e-4 {
            wrong += 1;
        }
    }
    assert_eq!(wrong, 0, "{wrong} untargeted neurons drifted from rest");
}

#[test]
fn weak_drive_only_depolarises() {
    let mut net = fresh_network(N);
    let enc = TextEncoder::new(N, K);

    let sdr = enc.encode_word("hello");
    // 30 nA: dV ≈ 0.005 · 10 · 30 = 1.5 mV per step → depolarised, not firing.
    inject_sdr(&mut net, &sdr.indices, 30.0);
    let fired = net.step(&[]);
    assert!(fired.is_empty(), "weak drive should not fire any neuron");

    let v_rest = LifParams::default().v_rest;
    let v_th = LifParams::default().v_threshold;
    for &idx in &sdr.indices {
        let v = net.v[idx as usize];
        assert!(v > v_rest + 0.1, "neuron {idx} not depolarised: V={v}");
        assert!(
            v < v_th,
            "neuron {idx} should not have crossed threshold: V={v}"
        );
    }
}
