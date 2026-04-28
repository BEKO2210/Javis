//! Test 2: a pre-synaptic spike, delivered with a strong enough weight,
//! makes the post-synaptic neuron fire shortly after.

use snn_core::{LifNeuron, LifParams, Network};

#[test]
fn presynaptic_spike_drives_postsynaptic_spike() {
    let mut net = Network::new(0.1);
    let pre = net.add_neuron(LifNeuron::new(LifParams::default()));
    let post = net.add_neuron(LifNeuron::new(LifParams::default()));
    net.connect(pre, post, 30.0);

    let mut external = vec![0.0_f32; 2];
    external[pre] = 3.0;
    external[post] = 0.0;

    let mut pre_spikes = 0usize;
    let mut post_spikes = 0usize;
    let mut first_pre_at: Option<f32> = None;
    let mut first_post_at: Option<f32> = None;

    for _ in 0..2000 {
        let t = net.time;
        let fired = net.step(&external);
        for &id in &fired {
            if id == pre {
                pre_spikes += 1;
                first_pre_at.get_or_insert(t);
            }
            if id == post {
                post_spikes += 1;
                first_post_at.get_or_insert(t);
            }
        }
    }

    assert!(pre_spikes > 0, "pre never fired");
    assert!(post_spikes > 0, "post never fired despite strong synapse");
    let tp = first_pre_at.unwrap();
    let tq = first_post_at.unwrap();
    assert!(tq >= tp, "post should fire at or after the first pre spike");
    assert!(
        tq - tp < 10.0,
        "post should follow pre quickly, gap was {} ms",
        tq - tp,
    );
}

#[test]
fn isolated_neuron_without_input_stays_silent() {
    let mut net = Network::new(0.1);
    let _ = net.add_neuron(LifNeuron::new(LifParams::default()));

    for _ in 0..1000 {
        let fired = net.step(&[]);
        assert!(fired.is_empty(), "no input should yield no spikes");
    }
}
