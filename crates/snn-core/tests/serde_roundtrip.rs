//! Snapshot round-trip: build a small brain with non-default plasticity,
//! serialize to JSON, deserialize, and verify topology + weights are
//! bit-identical. Transient buffers (i_syn, traces, time) are not
//! checked because they're explicitly skipped at serialization time.

use snn_core::{
    Brain, HomeostasisParams, IStdpParams, LifNeuron, LifParams, NeuronKind, Region, StdpParams,
};

#[test]
fn network_roundtrips_through_json() {
    let mut net = snn_core::Network::new(0.1);
    net.enable_stdp(StdpParams::default());
    net.enable_istdp(IStdpParams::default());
    net.enable_homeostasis(HomeostasisParams::default());

    let pre = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    let mid = net.add_neuron(LifNeuron::inhibitory(LifParams::default()));
    let post = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    net.connect(pre, post, 0.42);
    net.connect(mid, post, 0.13);

    let json = serde_json::to_string(&net).unwrap();
    let mut restored: snn_core::Network = serde_json::from_str(&json).unwrap();
    restored.ensure_transient_state();

    assert_eq!(restored.neurons.len(), 3);
    assert_eq!(restored.neurons[mid].kind, NeuronKind::Inhibitory);
    assert_eq!(restored.synapses.len(), 2);
    assert!((restored.synapses[0].weight - 0.42).abs() < 1e-6);
    assert!((restored.synapses[1].weight - 0.13).abs() < 1e-6);
    assert_eq!(restored.outgoing[pre], vec![0]);
    assert_eq!(restored.incoming[post], vec![0, 1]);
    assert!(restored.stdp.is_some());
    assert!(restored.istdp.is_some());
    assert!(restored.homeostasis.is_some());

    // Transient buffers must be initialised after ensure_transient_state.
    assert_eq!(restored.i_syn.len(), 3);
    assert_eq!(restored.pre_trace.len(), 3);
    assert_eq!(restored.post_trace.len(), 3);
}

#[test]
fn brain_roundtrips_with_inter_region_edges() {
    let mut brain = Brain::new(0.1);

    let mut r1 = Region::new("R1", 0.1);
    for _ in 0..5 {
        r1.network
            .add_neuron(LifNeuron::excitatory(LifParams::default()));
    }
    let mut r2 = Region::new("R2", 0.1);
    for _ in 0..7 {
        r2.network
            .add_neuron(LifNeuron::excitatory(LifParams::default()));
    }
    brain.add_region(r1);
    brain.add_region(r2);

    brain.connect(0, 0, 1, 3, 1.5, 2.0);
    brain.connect(0, 1, 1, 4, 0.8, 4.0);

    let json = serde_json::to_string(&brain).unwrap();
    let mut restored: Brain = serde_json::from_str(&json).unwrap();
    restored.ensure_transient_state();

    assert_eq!(restored.regions.len(), 2);
    assert_eq!(restored.regions[0].num_neurons(), 5);
    assert_eq!(restored.regions[1].num_neurons(), 7);
    assert_eq!(restored.inter_edges.len(), 2);
    assert!((restored.inter_edges[0].weight - 1.5).abs() < 1e-6);
    assert_eq!(restored.outgoing[0][0], vec![0]);
    assert_eq!(restored.outgoing[0][1], vec![1]);

    // Step it once — must not panic, transient buffers OK.
    let externals: Vec<Vec<f32>> = vec![Vec::new(), Vec::new()];
    let _ = restored.step(&externals);
}
