//! Iteration 10 hardening: BinaryHeap-backed `PendingQueue` and
//! heterogeneous synaptic channels (AMPA/NMDA/GABA).

use snn_core::{
    Brain, LifNeuron, LifParams, Network, PendingEvent, PendingQueue, Region, SynapseKind,
};

// ----------------------------------------------------------------------
// PendingQueue contract
// ----------------------------------------------------------------------

fn ev(arrive_at: f32, dst_neuron: u32, weight: f32) -> PendingEvent {
    PendingEvent {
        arrive_at,
        dst_region: 0,
        dst_neuron,
        weight,
    }
}

#[test]
fn pending_queue_drains_in_chronological_order() {
    let mut q = PendingQueue::new();
    q.push(ev(5.0, 0, 1.0));
    q.push(ev(2.0, 1, 1.0));
    q.push(ev(7.0, 2, 1.0));
    q.push(ev(2.0, 3, 1.0)); // tied with idx 1, FIFO inserts later

    // Cutoff t = 4.0 lets through both arrive_at=2.0 events.
    let due: Vec<_> = q.drain_due(4.0).collect();
    assert_eq!(due.len(), 2);
    assert_eq!(due[0].dst_neuron, 1, "older tied event first");
    assert_eq!(due[1].dst_neuron, 3);
    assert_eq!(q.len(), 2);
}

#[test]
fn pending_queue_respects_cutoff() {
    let mut q = PendingQueue::new();
    for i in 0..10u32 {
        q.push(ev(i as f32 + 1.0, i, 1.0));
    }
    let due: Vec<_> = q.drain_due(3.5).collect();
    assert_eq!(due.len(), 3); // arrival times 1, 2, 3
    assert_eq!(q.len(), 7);
    let due_ids: Vec<_> = due.iter().map(|e| e.dst_neuron).collect();
    assert_eq!(due_ids, vec![0, 1, 2]);
}

#[test]
fn pending_queue_clear_resets_sequence() {
    let mut q = PendingQueue::new();
    q.push(ev(1.0, 0, 1.0));
    q.push(ev(1.0, 1, 1.0));
    q.clear();
    assert_eq!(q.len(), 0);
    assert!(q.is_empty());
    // Re-pushing tied events still pops in FIFO.
    q.push(ev(1.0, 7, 1.0));
    q.push(ev(1.0, 8, 1.0));
    let due: Vec<_> = q.drain_due(2.0).collect();
    assert_eq!(due[0].dst_neuron, 7);
    assert_eq!(due[1].dst_neuron, 8);
}

#[test]
fn pending_queue_handles_many_events_efficiently() {
    // Smoke test that the heap correctly handles a large number of
    // events with varied arrival times. With a Vec-based queue this
    // would still terminate but in O(N²); with the heap it's
    // O(N log N).
    let mut q = PendingQueue::new();
    for i in 0..1000 {
        // Varying arrival times in [0, 10) so plenty interleave.
        let t = ((i * 31) % 1000) as f32 * 0.01;
        q.push(ev(t, i as u32, 1.0));
    }
    let due: Vec<_> = q.drain_due(5.0).collect();
    // All events with arrive_at <= 5.0 must come out in monotone order.
    for w in due.windows(2) {
        assert!(w[0].arrive_at <= w[1].arrive_at, "drain not chronological");
    }
    // And the queue still holds the rest.
    assert_eq!(q.len() + due.len(), 1000);
}

#[test]
fn brain_uses_heap_pending_queue_underneath() {
    // Smoke test that the queue change didn't break Brain wiring.
    let mut brain = Brain::new(0.1);
    let mut r1 = Region::new("R1", 0.1);
    r1.network
        .add_neuron(LifNeuron::excitatory(LifParams::default()));
    let mut r2 = Region::new("R2", 0.1);
    r2.network
        .add_neuron(LifNeuron::excitatory(LifParams::default()));
    brain.add_region(r1);
    brain.add_region(r2);

    brain.connect(0, 0, 1, 0, 5.0, 2.0);
    let mut ext1 = vec![100.0_f32; 1];
    let ext2 = vec![0.0_f32; 1];
    let externals = vec![ext1.clone(), ext2.clone()];

    let mut delivered_steps: Vec<f32> = Vec::new();
    let mut last_events_delivered = 0u64;
    for step in 0..50 {
        let _ = brain.step(&externals);
        if brain.events_delivered > last_events_delivered {
            delivered_steps.push(step as f32 * 0.1);
            last_events_delivered = brain.events_delivered;
        }
        ext1[0] = 0.0; // pulse only on first step
        let _ = ext1;
    }
    // We pulsed R1 → expected at least one delivery in R2.
    assert!(brain.events_delivered > 0, "no events delivered");
}

// ----------------------------------------------------------------------
// SynapseKind / heterogeneous channels
// ----------------------------------------------------------------------

#[test]
fn default_synapse_is_ampa() {
    let s = snn_core::Synapse::new(0, 1, 0.5);
    assert_eq!(s.kind, SynapseKind::Ampa);
}

#[test]
fn nmda_channel_decays_slower_than_ampa() {
    fn residual_after_ten_ms(kind: SynapseKind) -> f32 {
        let mut net = Network::new(0.1);
        let n = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
        match kind {
            SynapseKind::Ampa => net.i_syn[n] = 1.0,
            SynapseKind::Nmda => {
                net.i_syn_nmda = vec![1.0];
            }
            SynapseKind::Gaba => {
                net.i_syn_gaba = vec![1.0];
            }
        }
        for _ in 0..100 {
            net.step(&[]);
        }
        match kind {
            SynapseKind::Ampa => net.i_syn[n],
            SynapseKind::Nmda => net.i_syn_nmda[n],
            SynapseKind::Gaba => net.i_syn_gaba[n],
        }
    }
    let ampa = residual_after_ten_ms(SynapseKind::Ampa);
    let nmda = residual_after_ten_ms(SynapseKind::Nmda);
    let gaba = residual_after_ten_ms(SynapseKind::Gaba);
    eprintln!("residual after 10ms: ampa={ampa:.3} gaba={gaba:.3} nmda={nmda:.3}");
    // NMDA τ=100 ms → exp(-0.1) ≈ 0.905
    // GABA τ=10 ms  → exp(-1.0) ≈ 0.368
    // AMPA τ=5 ms   → exp(-2.0) ≈ 0.135
    assert!(
        nmda > gaba && gaba > ampa,
        "ordering must be NMDA > GABA > AMPA"
    );
    assert!((nmda - (-0.1_f32).exp()).abs() < 0.05);
    assert!((gaba - (-1.0_f32).exp()).abs() < 0.05);
    assert!((ampa - (-2.0_f32).exp()).abs() < 0.05);
}

#[test]
fn nmda_synapse_routes_to_nmda_channel() {
    let mut net = Network::new(0.1);
    let pre = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    let post = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    net.synapses.push(snn_core::Synapse::with_kind(
        pre,
        post,
        0.5,
        SynapseKind::Nmda,
    ));
    let id = (net.synapses.len() - 1) as u32;
    net.outgoing[pre].push(id);
    net.incoming[post].push(id);

    // Drive the pre-neuron with a strong current so it spikes.
    let mut ext = vec![0.0_f32; 2];
    ext[pre] = 50.0;
    for _ in 0..50 {
        let _ = net.step(&ext);
    }

    // After at least one pre-spike, the NMDA channel must be populated
    // with a non-zero current; the AMPA channel for the post-neuron
    // must be (near-)zero.
    assert!(
        net.i_syn_nmda.len() == 2 && net.i_syn_nmda[post] > 0.0,
        "NMDA delivery did not populate i_syn_nmda",
    );
    assert!(
        net.i_syn[post].abs() < 1e-6,
        "AMPA channel should not have received the NMDA-kind weight",
    );
}

#[test]
fn snapshot_round_trip_preserves_synapse_kind() {
    let mut net = Network::new(0.1);
    let a = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    let b = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    net.synapses
        .push(snn_core::Synapse::with_kind(a, b, 0.4, SynapseKind::Nmda));
    let id = (net.synapses.len() - 1) as u32;
    net.outgoing[a].push(id);
    net.incoming[b].push(id);

    let json = serde_json::to_string(&net).unwrap();
    let restored: Network = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.synapses[0].kind, SynapseKind::Nmda);
}

#[test]
fn snapshot_without_kind_field_defaults_to_ampa() {
    let mut net = Network::new(0.1);
    let a = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    let b = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    net.connect(a, b, 0.4);
    // Build a synapse JSON without the `kind` field — this is what
    // pre-iteration-10 snapshots looked like.
    let synapse_only = r#"{"pre":0,"post":1,"weight":0.4}"#;
    let restored: snn_core::Synapse = serde_json::from_str(synapse_only).unwrap();
    assert_eq!(restored.kind, SynapseKind::Ampa);

    // And the network-level round trip with explicit kind also works.
    let json = serde_json::to_string(&net).unwrap();
    let _: Network = serde_json::from_str(&json).unwrap();
}

#[test]
fn set_synaptic_taus_validates_inputs() {
    let mut net = Network::new(0.1);
    net.set_synaptic_taus(2.0, 80.0, 8.0);
    assert!((net.tau_syn_ms - 2.0).abs() < 1e-6);
    assert!((net.tau_nmda_ms - 80.0).abs() < 1e-6);
    assert!((net.tau_gaba_ms - 8.0).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "tau_nmda must be positive")]
fn set_synaptic_taus_rejects_negative_nmda() {
    let mut net = Network::new(0.1);
    net.set_synaptic_taus(2.0, -1.0, 8.0);
}
