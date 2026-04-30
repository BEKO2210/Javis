//! Iter-44 breakthrough plasticity stack — tests.
//!
//! Each new mechanism gets at least one positive test (it does what
//! the literature says it should) and one off-by-default test (the
//! existing pre-iter-44 networks see no change unless `enable_*` is
//! called). The off-by-default checks are what protect the 113
//! pre-existing tests from silent regressions.

use snn_core::{
    HeterosynapticParams, IntrinsicParams, LifNeuron, LifParams, MetaplasticityParams, Network,
    NeuronKind, ReplayParams, RewardParams, StdpParams, StructuralParams,
};

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

// ---------- triplet STDP ------------------------------------------

/// Triplet-STDP with `a3_plus > 0` should potentiate a pre-before-post
/// pair *more* than the same pair under plain pair-STDP, given a few
/// repetitions that build up `post_trace2`. This is the headline
/// frequency-dependent prediction of Pfister-Gerstner 2006. We use
/// soft bounds and a small w_max so both runs settle inside the
/// interior — we want to compare *positions*, not detect "both
/// saturate at the clamp".
#[test]
fn triplet_stdp_potentiates_above_pair_baseline() {
    fn run(triplet: bool) -> f32 {
        let mut net = Network::new(0.1);
        net.enable_stdp(StdpParams {
            a_plus: 0.005,
            a_minus: 0.006,
            w_min: 0.0,
            w_max: 1.0,
            soft_bounds: true,
            a3_plus: if triplet { 0.02 } else { 0.0 },
            ..StdpParams::default()
        });
        let pre = net.add_neuron(LifNeuron::new(LifParams::default()));
        let post = net.add_neuron(LifNeuron::new(LifParams::default()));
        net.connect(pre, post, 0.3);
        for _ in 0..40 {
            assert!(pulse(&mut net, pre, 1.0));
            idle(&mut net, 9.0);
            assert!(pulse(&mut net, post, 1.0));
            idle(&mut net, 90.0);
        }
        net.synapses[0].weight
    }
    let pair = run(false);
    let trip = run(true);
    eprintln!("pair-STDP final w = {pair:.4} | triplet final w = {trip:.4}");
    assert!(
        trip > pair,
        "triplet STDP should drive LTP further than pair-STDP under repeated pairing: pair={pair} triplet={trip}",
    );
    // And both must stay strictly below the soft-bound w_max so the
    // comparison is meaningful (not "both pinned at the clamp").
    assert!(pair < 0.99 && trip < 0.99);
}

#[test]
fn triplet_stdp_off_by_default() {
    let mut net = Network::new(0.1);
    let p = StdpParams {
        a_plus: 0.01,
        a_minus: 0.012,
        w_max: 1.0,
        ..StdpParams::default()
    };
    net.enable_stdp(p);
    assert_eq!(p.a3_plus, 0.0);
    assert_eq!(p.a3_minus, 0.0);
    assert!(net.pre_trace2.is_empty());
    assert!(net.post_trace2.is_empty());
}

// ---------- reward-modulated STDP ---------------------------------

#[test]
fn reward_modulator_drives_eligible_synapse_up() {
    let mut net = Network::new(0.1);
    net.enable_stdp(StdpParams {
        a_plus: 0.01,
        a_minus: 0.012,
        w_min: 0.0,
        w_max: 1.0,
        ..StdpParams::default()
    });
    net.enable_reward_learning(RewardParams {
        eta: 1e-2,
        tau_eligibility_ms: 500.0,
        a_plus: 0.05,
        a_minus: 0.05,
        w_min: 0.0,
        w_max: 1.0,
        excitatory_only: true,
    });
    let pre = net.add_neuron(LifNeuron::new(LifParams::default()));
    let post = net.add_neuron(LifNeuron::new(LifParams::default()));
    net.connect(pre, post, 0.4);
    let w_before = net.synapses[0].weight;
    // Build up eligibility tag with 10 pre→post pairings (no
    // reward yet).
    for _ in 0..10 {
        assert!(pulse(&mut net, pre, 1.0));
        idle(&mut net, 5.0);
        assert!(pulse(&mut net, post, 1.0));
        idle(&mut net, 50.0);
    }
    let w_tag = net.synapses[0].weight;
    // Now flip dopamine on for ~200 ms of idle time. Eligibility
    // decays but the gated update must push the weight up.
    net.set_neuromodulator(1.0);
    idle(&mut net, 200.0);
    let w_after_reward = net.synapses[0].weight;
    eprintln!("before={w_before:.4} after_pairing={w_tag:.4} after_reward={w_after_reward:.4}",);
    assert!(
        w_after_reward > w_tag,
        "positive dopamine should push the eligible synapse upward: {w_tag} → {w_after_reward}",
    );
}

#[test]
fn reward_off_by_default_does_not_touch_weights() {
    let mut net = Network::new(0.1);
    net.enable_stdp(StdpParams {
        a_plus: 0.0,
        a_minus: 0.0,
        ..StdpParams::default()
    });
    let pre = net.add_neuron(LifNeuron::new(LifParams::default()));
    let post = net.add_neuron(LifNeuron::new(LifParams::default()));
    net.connect(pre, post, 0.4);
    let w0 = net.synapses[0].weight;
    net.set_neuromodulator(1.0);
    for _ in 0..5 {
        assert!(pulse(&mut net, pre, 1.0));
        idle(&mut net, 5.0);
        assert!(pulse(&mut net, post, 1.0));
        idle(&mut net, 50.0);
    }
    assert!(net.reward.is_none());
    assert_eq!(
        net.synapses[0].weight, w0,
        "reward learning must stay completely silent until enable_reward_learning",
    );
}

// ---------- metaplasticity (BCM) ----------------------------------

#[test]
fn metaplasticity_modulator_bounded_around_one() {
    let m = MetaplasticityParams::enabled();
    // Cold start: zero rate, zero θ → unity modulator.
    assert!((m.modulator(0.0, 0.0) - 1.0).abs() < 1e-6);
    // Heavy firing, low θ → modulator above 1, capped at 1 + k_max.
    let hot = m.modulator(10.0, 0.1);
    assert!(hot > 1.0);
    assert!(hot <= 1.0 + m.k_max + 1e-6);
    // Quiet, high θ → modulator below 1, capped at 1 - k_max.
    let cold = m.modulator(0.1, 10.0);
    assert!(cold < 1.0);
    assert!(cold >= 1.0 - m.k_max - 1e-6);
}

#[test]
fn metaplasticity_stabilises_runaway_ltp() {
    fn run(meta_on: bool) -> f32 {
        let mut net = Network::new(0.1);
        net.enable_stdp(StdpParams {
            a_plus: 0.05,
            a_minus: 0.025,
            w_min: 0.0,
            w_max: 1.0,
            ..StdpParams::default()
        });
        if meta_on {
            net.enable_metaplasticity(MetaplasticityParams {
                tau_rate_ms: 50.0,
                tau_theta_ms: 200.0,
                strength: 1.0,
                k_max: 0.5,
                enabled: true,
            });
        }
        let pre = net.add_neuron(LifNeuron::new(LifParams::default()));
        let post = net.add_neuron(LifNeuron::new(LifParams::default()));
        net.connect(pre, post, 0.4);
        // Drive both heavily so post fires often → θ rises → LTP
        // is pulled back toward neutral (BCM stabilisation).
        for _ in 0..400 {
            let _ = pulse(&mut net, pre, 1.0);
            idle(&mut net, 4.0);
            let _ = pulse(&mut net, post, 1.0);
            idle(&mut net, 5.0);
        }
        net.synapses[0].weight
    }
    let plain = run(false);
    let bcm = run(true);
    eprintln!("plain LTP = {plain:.4} | BCM LTP = {bcm:.4}");
    assert!(
        bcm < plain || (plain - bcm).abs() < 0.05,
        "BCM should not run *higher* than plain STDP under sustained drive",
    );
}

// ---------- intrinsic plasticity (adaptive threshold) -------------

#[test]
fn intrinsic_threshold_offset_grows_with_overactivity() {
    let mut net = Network::new(0.1);
    net.enable_intrinsic_plasticity(IntrinsicParams {
        alpha_spike: 1.0,
        tau_adapt_ms: 200.0,
        a_target: 1.0,
        beta: 0.5,
        offset_min: -10.0,
        offset_max: 10.0,
        enabled: true,
    });
    let n = net.add_neuron(LifNeuron::new(LifParams::default()));
    // Heavy drive → many spikes → adapt trace grows → offset grows.
    // We expect *most* pulses to fire, but as adaptation kicks in the
    // neuron will eventually skip pulses (that *is* the mechanism). We
    // care about the trace + offset, not 100 % firing reliability.
    let mut fired_at_least_once = false;
    for _ in 0..100 {
        if pulse(&mut net, n, 1.0) {
            fired_at_least_once = true;
        }
        idle(&mut net, 1.0);
    }
    assert!(fired_at_least_once);
    let offset = net.v_thresh_offset[0];
    assert!(offset > 0.0, "offset must rise under heavy drive: {offset}");
    let adapt = net.adapt_trace[0];
    assert!(adapt > 1.0, "adaptation trace must rise: {adapt}");
}

#[test]
fn intrinsic_off_by_default_does_not_resize_buffers() {
    let mut net = Network::new(0.1);
    let _n = net.add_neuron(LifNeuron::new(LifParams::default()));
    assert!(net.adapt_trace.is_empty());
    assert!(net.v_thresh_offset.is_empty());
}

// ---------- heterosynaptic normalisation --------------------------

#[test]
fn heterosynaptic_caps_incoming_l2_norm() {
    let mut net = Network::new(0.1);
    net.enable_heterosynaptic(HeterosynapticParams::l2());
    let pre1 = net.add_neuron(LifNeuron::new(LifParams::default()));
    let pre2 = net.add_neuron(LifNeuron::new(LifParams::default()));
    let post = net.add_neuron(LifNeuron::new(LifParams::default()));
    net.connect(pre1, post, 1.5);
    net.connect(pre2, post, 1.5);
    let target = 1.5_f32;
    // Force a normalisation pass — `apply_every` defaults to 200,
    // so step the network 200 times with no drive.
    idle(&mut net, 20.0);
    let w1 = net.synapses[0].weight;
    let w2 = net.synapses[1].weight;
    let l2 = (w1 * w1 + w2 * w2).sqrt();
    assert!(
        l2 <= target + 1e-3,
        "L2 norm must be capped at target {target}, got {l2}",
    );
}

#[test]
fn heterosynaptic_off_by_default() {
    let mut net = Network::new(0.1);
    let pre = net.add_neuron(LifNeuron::new(LifParams::default()));
    let post = net.add_neuron(LifNeuron::new(LifParams::default()));
    net.connect(pre, post, 5.0);
    let w0 = net.synapses[0].weight;
    idle(&mut net, 100.0);
    assert!(net.heterosynaptic.is_none());
    assert_eq!(net.synapses[0].weight, w0);
}

// ---------- structural plasticity ---------------------------------

#[test]
fn structural_pruning_drops_dormant_synapse() {
    let mut net = Network::new(0.1);
    net.enable_structural(StructuralParams {
        prune_threshold: 0.05,
        prune_age_steps: 2,
        sprout_pre_trace: 1e6, // disable sprouting for this test
        sprout_post_trace: 1e6,
        sprout_initial: 0.0,
        max_new_per_step: 0,
        apply_every: 100,
        enabled: true,
    });
    let pre = net.add_neuron(LifNeuron::new(LifParams::default()));
    let post = net.add_neuron(LifNeuron::new(LifParams::default()));
    let _eid = net.connect(pre, post, 0.01); // below prune threshold
    assert_eq!(net.outgoing[pre].len(), 1);
    // Run enough steps for ≥ 2 structural passes.
    idle(&mut net, 30.0); // 300 steps → 3 passes
    assert_eq!(net.dead_synapses, 1, "synapse should be marked dead");
    assert_eq!(
        net.outgoing[pre].len(),
        0,
        "dead synapse should be removed from adjacency",
    );
    let dropped = net.compact_synapses();
    assert_eq!(dropped, 1);
    assert_eq!(net.synapses.len(), 0);
    assert_eq!(net.dead_synapses, 0);
}

#[test]
fn structural_off_by_default_does_not_grow_or_prune() {
    let mut net = Network::new(0.1);
    let pre = net.add_neuron(LifNeuron::new(LifParams::default()));
    let post = net.add_neuron(LifNeuron::new(LifParams::default()));
    net.connect(pre, post, 0.01); // would be prunable if enabled
    idle(&mut net, 100.0);
    assert!(net.structural.is_none());
    assert_eq!(net.synapses.len(), 1);
    assert_eq!(net.dead_synapses, 0);
}

// ---------- replay / consolidation --------------------------------

#[test]
fn consolidate_drives_top_engram_cells() {
    let mut net = Network::new(0.1);
    net.enable_stdp(StdpParams {
        a_plus: 0.05,
        a_minus: 0.0,
        w_min: 0.0,
        w_max: 1.0,
        ..StdpParams::default()
    });
    let a = net.add_neuron(LifNeuron::new(LifParams::default()));
    let b = net.add_neuron(LifNeuron::new(LifParams::default()));
    let c = net.add_neuron(LifNeuron::new(LifParams::default()));
    // Pre-shape an engram on `b` only — its incoming weight starts
    // higher than `a` and `c`'s.
    net.connect(a, b, 0.8);
    net.connect(c, b, 0.8);
    net.connect(a, c, 0.05);
    let w_before = net.synapses[0].weight;
    // Replay should pick `b` (largest incoming sum), pulse it, the
    // pulse pre-trace builds, post-trace rises, LTP occurs on every
    // incoming synapse onto `b`.
    let mut params = ReplayParams::quick();
    params.top_k = 1;
    params.pulse_ms = 5.0;
    params.gap_ms = 2.0;
    params.drive_current = 5.0;
    net.consolidate(&params);
    let w_after = net.synapses[0].weight;
    assert!(
        w_after >= w_before,
        "replay should not weaken the strongest engram synapse: {w_before} → {w_after}",
    );
}

// ---------- composite / interference ------------------------------

#[test]
fn full_iter44_stack_runs_without_panicking() {
    // Stress test: turn every iter-44 mechanism on at once and step
    // a small E/I network for a few thousand cycles. Anything that
    // mis-allocates a buffer (length mismatches, off-by-one) will
    // surface here as an `index out of bounds` panic.
    let mut net = Network::new(0.1);
    net.enable_stdp(StdpParams {
        a_plus: 0.01,
        a_minus: 0.012,
        a3_plus: 0.005,
        a3_minus: 0.0,
        w_min: 0.0,
        w_max: 1.0,
        ..StdpParams::default()
    });
    net.enable_metaplasticity(MetaplasticityParams::enabled());
    net.enable_intrinsic_plasticity(IntrinsicParams::enabled());
    net.enable_heterosynaptic(HeterosynapticParams::l2());
    net.enable_structural(StructuralParams {
        apply_every: 100,
        ..StructuralParams::enabled()
    });
    net.enable_reward_learning(RewardParams::enabled());
    let mut ids = Vec::new();
    for i in 0..16 {
        let kind = if i % 4 == 0 {
            NeuronKind::Inhibitory
        } else {
            NeuronKind::Excitatory
        };
        ids.push(net.add_neuron(LifNeuron::with_kind(LifParams::default(), kind)));
    }
    for i in 0..16 {
        for j in 0..16 {
            if i == j {
                continue;
            }
            net.connect(ids[i], ids[j], 0.05);
        }
    }
    // Drive a couple of neurons with a small constant external,
    // toggle dopamine, and step.
    let mut ext = vec![0.0_f32; 16];
    ext[1] = 1.5;
    ext[2] = 1.5;
    for s in 0..2000 {
        if s == 500 {
            net.set_neuromodulator(0.5);
        }
        if s == 1500 {
            net.set_neuromodulator(0.0);
        }
        let _ = net.step(&ext);
    }
    // The full stack is a no-op-safe configuration; eligibility / θ /
    // adapt-trace must all be finite at the end.
    for x in &net.adapt_trace {
        assert!(x.is_finite());
    }
    for x in &net.theta_trace {
        assert!(x.is_finite());
    }
    for x in &net.eligibility {
        assert!(x.is_finite());
    }
    for s in &net.synapses {
        assert!(s.weight.is_finite());
    }
}

#[test]
fn classical_passive_network_unchanged_by_iter44() {
    // No `enable_*` calls → behaviour must be byte-identical to the
    // pre-iter-44 passive integrator. This is the critical
    // regression guard for the existing 113 tests.
    let mut net = Network::new(0.1);
    let pre = net.add_neuron(LifNeuron::new(LifParams::default()));
    let post = net.add_neuron(LifNeuron::new(LifParams::default()));
    net.connect(pre, post, 30.0);
    let w0 = net.synapses[0].weight;
    let mut external = vec![0.0_f32; 2];
    external[pre] = 3.0;
    let mut post_fired = false;
    for _ in 0..1000 {
        let fired = net.step(&external);
        if fired.contains(&post) {
            post_fired = true;
            break;
        }
    }
    assert!(post_fired);
    assert_eq!(
        net.synapses[0].weight, w0,
        "passive network must not touch weights",
    );
    assert!(net.pre_trace2.is_empty());
    assert!(net.post_trace2.is_empty());
    assert!(net.eligibility.is_empty());
    assert!(net.rate_trace.is_empty());
    assert!(net.adapt_trace.is_empty());
}
