//! Read-only step path equivalence tests.
//!
//! When plasticity (STDP / iSTDP / homeostasis) is disabled, the two
//! step paths must produce identical spike outputs and identical
//! transient state — same membrane potentials, same channel currents,
//! same delivery counters. The read-only path is what the recall
//! pipeline will use to drop the `Mutex<Brain>` bottleneck identified
//! in `notes/35-load-test.md`.

use snn_core::{
    Brain, LifNeuron, LifParams, Network, NetworkState, NeuronKind, Region, Rng, StdpParams,
};

const DT: f32 = 0.1;

/// Build a small E/I network with sparse recurrent connectivity. STDP
/// is intentionally NOT enabled — the immutable path skips plasticity
/// unconditionally, so equivalence with the mutating path only holds
/// when the mutating path also has plasticity off.
fn build_network(n: usize, seed: u64) -> Network {
    let mut net = Network::new(DT);
    let mut rng = Rng::new(seed);
    let n_inh = (n as f32 * 0.2) as usize;
    let n_exc = n - n_inh;
    for _ in 0..n_exc {
        net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    }
    for _ in 0..n_inh {
        net.add_neuron(LifNeuron::inhibitory(LifParams::default()));
    }
    let g_exc = 0.20_f32;
    let g_inh = 0.80_f32;
    for pre in 0..n {
        let g = match net.neurons[pre].kind {
            NeuronKind::Excitatory => g_exc,
            NeuronKind::Inhibitory => g_inh,
        };
        for post in 0..n {
            if pre == post {
                continue;
            }
            if rng.bernoulli(0.1) {
                let w = rng.range_f32(0.5 * g, 1.0 * g);
                net.connect(pre, post, w);
            }
        }
    }
    net
}

#[test]
fn network_step_immutable_matches_step_when_plasticity_off() {
    let net_mutating = build_network(200, 2027);
    let net_readonly = build_network(200, 2027); // identical seed → identical topology
    let mut net_mut = net_mutating.clone();
    let net_imm = net_readonly;
    let mut state = net_imm.fresh_state();

    // Same external drive on both: small steady current to a handful
    // of neurons. Run 200 steps and compare per-step spike vectors.
    let mut external = vec![0.0_f32; 200];
    for slot in external.iter_mut().take(30) {
        *slot = 0.6;
    }

    for step_idx in 0..200 {
        let fired_mut = net_mut.step(&external);
        let fired_imm = net_imm.step_immutable(&mut state, &external);
        assert_eq!(
            fired_mut, fired_imm,
            "spike divergence at step {step_idx}: mut={fired_mut:?} imm={fired_imm:?}",
        );
    }

    // Final transient state must match too. We compare the
    // mutating-path's `Network` fields against the read-only state.
    for i in 0..net_mut.neurons.len() {
        assert!(
            (net_mut.neurons[i].v - state.v[i]).abs() < 1e-4,
            "membrane v mismatch at neuron {i}: {} vs {}",
            net_mut.neurons[i].v,
            state.v[i],
        );
    }
    for (idx, (&a, &b)) in net_mut.i_syn.iter().zip(state.i_syn.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "i_syn mismatch at neuron {idx}: {a} vs {b}",
        );
    }
    assert_eq!(net_mut.synapse_events, state.synapse_events);
}

#[test]
fn network_state_lazy_nmda_gaba_allocation() {
    // A network that only uses AMPA should never allocate the NMDA or
    // GABA buffers in the read-only state. Same lazy policy as the
    // in-place path, so memory stays cheap for typical recall sims.
    let net = build_network(100, 2027);
    let state = net.fresh_state();
    assert!(state.i_syn_nmda.is_empty());
    assert!(state.i_syn_gaba.is_empty());
    assert_eq!(state.i_syn.len(), 100);
    assert_eq!(state.v.len(), 100);
}

#[test]
fn brain_step_immutable_matches_step_when_plasticity_off() {
    // Two-region brain, no STDP. Must produce identical spike vectors
    // through 100 steps via both paths.
    let mut brain_mut = build_two_region_brain(2027);
    let brain_imm = build_two_region_brain(2027);
    let mut state = brain_imm.fresh_state();

    let mut ext0 = vec![0.0_f32; 200];
    for slot in ext0.iter_mut().take(30) {
        *slot = 0.6;
    }
    let externals = vec![ext0, vec![0.0_f32; 200]];

    for step_idx in 0..100 {
        let fired_mut = brain_mut.step(&externals);
        let fired_imm = brain_imm.step_immutable(&mut state, &externals);
        assert_eq!(
            fired_mut, fired_imm,
            "spike divergence at step {step_idx}: mut={fired_mut:?} imm={fired_imm:?}",
        );
    }
    assert_eq!(brain_mut.events_delivered, state.events_delivered);
}

#[test]
fn brain_step_immutable_does_not_mutate_brain() {
    // Sanity: after running the read-only path, the original Brain's
    // synapse weights and brain-clock are unchanged. This is the
    // property that lets multiple concurrent recalls share an
    // `Arc<Brain>`.
    let brain = build_two_region_brain(2027);
    let weights_before: Vec<f32> = brain.regions[0]
        .network
        .synapses
        .iter()
        .map(|s| s.weight)
        .collect();
    let inter_before: Vec<f32> = brain.inter_edges.iter().map(|e| e.weight).collect();
    let time_before = brain.time;

    let mut state = brain.fresh_state();
    let externals = vec![vec![0.6_f32; 200], vec![0.0_f32; 200]];
    for _ in 0..200 {
        brain.step_immutable(&mut state, &externals);
    }

    let weights_after: Vec<f32> = brain.regions[0]
        .network
        .synapses
        .iter()
        .map(|s| s.weight)
        .collect();
    let inter_after: Vec<f32> = brain.inter_edges.iter().map(|e| e.weight).collect();
    assert_eq!(weights_before, weights_after, "synapse weights changed");
    assert_eq!(inter_before, inter_after, "inter-region weights changed");
    assert_eq!(brain.time, time_before, "brain.time advanced");
    assert!(state.time > 0.0, "but the state-side clock did advance");
}

fn build_region(n: usize, seed: u64) -> Region {
    let mut region = Region::new("R", DT);
    let mut rng = Rng::new(seed);
    let n_inh = (n as f32 * 0.2) as usize;
    let n_exc = n - n_inh;
    for _ in 0..n_exc {
        region
            .network
            .add_neuron(LifNeuron::excitatory(LifParams::default()));
    }
    for _ in 0..n_inh {
        region
            .network
            .add_neuron(LifNeuron::inhibitory(LifParams::default()));
    }
    let g_exc = 0.20_f32;
    let g_inh = 0.80_f32;
    for pre in 0..n {
        let g = match region.network.neurons[pre].kind {
            NeuronKind::Excitatory => g_exc,
            NeuronKind::Inhibitory => g_inh,
        };
        for post in 0..n {
            if pre == post {
                continue;
            }
            if rng.bernoulli(0.1) {
                let w = rng.range_f32(0.5 * g, 1.0 * g);
                region.network.connect(pre, post, w);
            }
        }
    }
    region
}

fn build_two_region_brain(seed: u64) -> Brain {
    let mut brain = Brain::new(DT);
    brain.add_region(build_region(200, seed));
    brain.add_region(build_region(200, seed.wrapping_add(101)));
    let mut rng = Rng::new(seed.wrapping_add(7777));
    for src in 0..20 {
        for _ in 0..10 {
            let dst = (rng.next_u64() as usize) % 200;
            let delay = 1.0 + rng.range_f32(0.0, 2.0);
            brain.connect(0, src, 1, dst, 1.5, delay);
        }
    }
    brain
}

// Quiet the unused-import lint when StdpParams is left out of the
// active surface — re-export via a no-op binding so future tests in
// this file can flip plasticity on without re-importing.
#[allow(dead_code)]
fn _stdp_unused() -> StdpParams {
    StdpParams::default()
}

// Bring `NetworkState` into the test compilation unit explicitly so it
// is checked as part of the public API surface even if a future
// refactor moves it.
#[allow(dead_code)]
fn _ns_type_check(_: &NetworkState) {}
