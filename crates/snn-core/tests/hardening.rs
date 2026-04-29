//! Architecture hardening regression tests.
//!
//! Pin down the contracts introduced in iteration 9: per-network
//! `tau_syn_ms`, bounds checks on `Network::connect` and
//! `Brain::connect`, and snapshot-compatible defaults.

use snn_core::{Brain, LifNeuron, LifParams, Network, NeuronKind, Region};

// ----------------------------------------------------------------------
// tau_syn_ms — moved from Synapse to Network
// ----------------------------------------------------------------------

#[test]
fn default_tau_syn_is_5_ms() {
    let net = Network::new(0.1);
    assert!((net.tau_syn_ms - 5.0).abs() < 1e-6);
}

#[test]
fn tau_syn_setter_changes_psc_decay() {
    // A single delivered EPSP at t=0 should decay slower under a
    // longer τ_syn. We measure i_syn at t=10 ms after a one-shot
    // injection.
    fn residual_after_ten_ms(tau_syn_ms: f32) -> f32 {
        let mut net = Network::new(0.1);
        net.set_tau_syn_ms(tau_syn_ms);
        let n = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
        net.i_syn[n] = 1.0;
        // 100 steps of 0.1 ms = 10 ms.
        for _ in 0..100 {
            net.step(&[]);
        }
        net.i_syn[n]
    }

    let fast = residual_after_ten_ms(5.0);
    let slow = residual_after_ten_ms(20.0);

    eprintln!("residual after 10ms: τ_syn=5 → {fast:.4}  τ_syn=20 → {slow:.4}");
    assert!(
        slow > fast,
        "longer τ_syn must decay slower (got {fast} vs {slow})"
    );
    // Analytical sanity: with τ=20 ms, residual after 10 ms ≈ exp(-0.5) ≈ 0.607.
    assert!((slow - (-0.5_f32).exp()).abs() < 0.05);
    assert!((fast - (-2.0_f32).exp()).abs() < 0.05);
}

#[test]
#[should_panic(expected = "tau_syn_ms must be positive")]
fn set_tau_syn_rejects_zero() {
    let mut net = Network::new(0.1);
    net.set_tau_syn_ms(0.0);
}

#[test]
#[should_panic(expected = "tau_syn_ms must be positive")]
fn set_tau_syn_rejects_negative() {
    let mut net = Network::new(0.1);
    net.set_tau_syn_ms(-1.0);
}

// ----------------------------------------------------------------------
// Network::connect bounds checks
// ----------------------------------------------------------------------

#[test]
#[should_panic(expected = "out of bounds")]
fn network_connect_rejects_pre_out_of_bounds() {
    let mut net = Network::new(0.1);
    net.add_neuron(LifNeuron::new(LifParams::default()));
    // Only 1 neuron — pre = 1 is out of bounds.
    net.connect(1, 0, 0.5);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn network_connect_rejects_post_out_of_bounds() {
    let mut net = Network::new(0.1);
    net.add_neuron(LifNeuron::new(LifParams::default()));
    net.connect(0, 1, 0.5);
}

#[test]
#[should_panic(expected = "weight must be finite")]
fn network_connect_rejects_nan_weight() {
    let mut net = Network::new(0.1);
    net.add_neuron(LifNeuron::new(LifParams::default()));
    net.add_neuron(LifNeuron::new(LifParams::default()));
    net.connect(0, 1, f32::NAN);
}

#[test]
#[should_panic(expected = "weight must be finite")]
fn network_connect_rejects_inf_weight() {
    let mut net = Network::new(0.1);
    net.add_neuron(LifNeuron::new(LifParams::default()));
    net.add_neuron(LifNeuron::new(LifParams::default()));
    net.connect(0, 1, f32::INFINITY);
}

// ----------------------------------------------------------------------
// Brain::connect bounds checks
// ----------------------------------------------------------------------

fn build_two_regions() -> Brain {
    let mut brain = Brain::new(0.1);
    let mut r1 = Region::new("R1", 0.1);
    for _ in 0..3 {
        r1.network
            .add_neuron(LifNeuron::excitatory(LifParams::default()));
    }
    let mut r2 = Region::new("R2", 0.1);
    for _ in 0..5 {
        r2.network
            .add_neuron(LifNeuron::excitatory(LifParams::default()));
    }
    brain.add_region(r1);
    brain.add_region(r2);
    brain
}

#[test]
#[should_panic(expected = "src_region")]
fn brain_connect_rejects_unknown_src_region() {
    let mut b = build_two_regions();
    b.connect(7, 0, 1, 0, 1.0, 2.0);
}

#[test]
#[should_panic(expected = "dst_region")]
fn brain_connect_rejects_unknown_dst_region() {
    let mut b = build_two_regions();
    b.connect(0, 0, 7, 0, 1.0, 2.0);
}

#[test]
#[should_panic(expected = "src_neuron")]
fn brain_connect_rejects_oob_src_neuron() {
    let mut b = build_two_regions();
    b.connect(0, 99, 1, 0, 1.0, 2.0);
}

#[test]
#[should_panic(expected = "dst_neuron")]
fn brain_connect_rejects_oob_dst_neuron() {
    let mut b = build_two_regions();
    b.connect(0, 0, 1, 99, 1.0, 2.0);
}

#[test]
#[should_panic(expected = "delay_ms must be positive")]
fn brain_connect_rejects_zero_delay() {
    let mut b = build_two_regions();
    b.connect(0, 0, 1, 0, 1.0, 0.0);
}

#[test]
#[should_panic(expected = "weight must be finite")]
fn brain_connect_rejects_nan_weight() {
    let mut b = build_two_regions();
    b.connect(0, 0, 1, 0, f32::NAN, 2.0);
}

#[test]
fn brain_connect_accepts_valid_input() {
    let mut b = build_two_regions();
    let id1 = b.connect(0, 0, 1, 0, 1.5, 2.0);
    let id2 = b.connect(0, 1, 1, 4, 0.5, 3.0);
    assert_eq!(id1, 0);
    assert_eq!(id2, 1);
    assert_eq!(b.inter_edges.len(), 2);
    let _ = NeuronKind::Excitatory; // silence import warning
}

// ----------------------------------------------------------------------
// Snapshot back-compat: missing tau_syn_ms field deserialises with default
// ----------------------------------------------------------------------

#[test]
fn snapshot_without_tau_syn_uses_default() {
    // Build a network, serialise it, strip the tau_syn_ms field from
    // the JSON, deserialise back. The default function on the field
    // must restore it to 5.0 — guarantees pre-iteration-9 snapshots
    // still load.
    let mut net = Network::new(0.1);
    net.add_neuron(LifNeuron::new(LifParams::default()));
    net.add_neuron(LifNeuron::new(LifParams::default()));
    net.connect(0, 1, 0.5);

    let json = serde_json::to_string(&net).unwrap();
    let stripped = json.replace(",\"tau_syn_ms\":5.0", "");
    assert!(!stripped.contains("tau_syn_ms"));

    let mut restored: Network = serde_json::from_str(&stripped).unwrap();
    restored.ensure_transient_state();
    assert!((restored.tau_syn_ms - 5.0).abs() < 1e-6);
}
