//! Iter-67 BTSP plateau-eligibility unit tests.
//!
//! Five tests cover the locked rule semantics from notes/67:
//!
//! 1. `plateau_arms_after_threshold_post_spikes_within_window`
//! 2. `tag_accumulates_on_pre_spikes_within_eligibility_window`
//! 3. `potentiation_only_when_plateau_armed_AND_tag_nonzero`
//! 4. `plateau_disarms_after_post_silence`
//! 5. `weight_capped_at_w_max`
//!
//! Plus a sanity test that BTSP off ⇒ snn-core numerics are
//! bit-identical to the pre-iter-67 path.

use snn_core::{BtspParams, LifNeuron, LifParams, Network, NeuronKind};

const DT_MS: f32 = 1.0;

fn fresh_network(n: usize) -> Network {
    let mut net = Network::new(DT_MS);
    let p = LifParams {
        v_rest: -70.0,
        v_reset: -75.0,
        v_threshold: -55.0,
        tau_m: 20.0,
        r_m: 10.0,
        refractory: 2.0,
    };
    for _ in 0..n {
        net.add_neuron(LifNeuron {
            params: p,
            kind: NeuronKind::Excitatory,
        });
    }
    net
}

/// Step the network until neuron `idx` fires once, with strong
/// external drive on `idx`. LIF refractory (default 2 ms with
/// dt = 1 ms) means a spike happens at most every 2 steps; this
/// helper steps until the spike actually lands so test bookkeeping
/// is by *spike count*, not by *step count*.
fn force_spike(net: &mut Network, idx: usize) -> Vec<usize> {
    let mut ext = vec![0.0_f32; net.neurons.len()];
    ext[idx] = 1000.0;
    for _ in 0..16 {
        let fired = net.step(&ext);
        if fired.contains(&idx) {
            return fired;
        }
    }
    panic!(
        "force_spike: neuron {} did not fire within 16 steps despite \
         1000 nA external — fix the LIF params in fresh_network",
        idx
    );
}

fn idle_one_step(net: &mut Network) -> Vec<usize> {
    let zeros = vec![0.0_f32; net.neurons.len()];
    net.step(&zeros)
}

#[test]
fn plateau_arms_after_threshold_post_spikes_within_window() {
    // Two neurons, one synapse 0 → 1. Force post-cell (1) to spike
    // 5 times within 5 ms. With plateau_threshold_spikes = 5 and
    // plateau_window_ms = 30, the burst trace should cross threshold
    // on the 5th spike → plateau arms → btsp_plateau_events == 1.
    let mut net = fresh_network(2);
    net.connect(0, 1, 0.1);
    // Use a long plateau_window_ms (1000) so burst-trace decay is
    // essentially zero over the few-ms test interval; the test then
    // checks the threshold-crossing logic in isolation, not the
    // decay kinetics (which `plateau_disarms_after_post_silence`
    // covers separately).
    let bp = BtspParams {
        plateau_threshold_spikes: 5.0,
        plateau_window_ms: 1000.0,
        eligibility_window_ms: 100.0,
        potentiation_strength: 0.4,
        post_plateau_decay_ms: 50.0,
        w_min: 0.0,
        w_max: 0.8,
        target_gated: true,
        non_target_depression_strength: 0.0,
        heterosynaptic_strength: 0.0,
    };
    net.enable_btsp(bp, None);
    // Fire 8 post-spikes — well above the threshold of 5 to clear
    // the refractory-induced decay between spikes (each spike adds
    // 1.0 to the trace but the next spike is ≥ 2 ms later, so the
    // trace decays slightly between adds; 5 spikes give trace
    // ≈ 4.98, only the 6th crosses ≥ 5.0).
    for _ in 0..8 {
        let _ = force_spike(&mut net, 1);
    }
    assert_eq!(
        net.btsp_plateau_events, 1,
        "expected exactly one plateau-arm event after 8 post-spikes \
         (well above threshold 5); got {}",
        net.btsp_plateau_events,
    );
    // Plateau is armed; armed_until should be in the future.
    assert!(
        net.btsp_post_armed_until[1] > net.time,
        "post-cell 1 should be armed; armed_until = {} but time = {}",
        net.btsp_post_armed_until[1],
        net.time,
    );
}

#[test]
fn tag_accumulates_on_pre_spikes_within_eligibility_window() {
    // 0 → 1 synapse. Fire pre (0) three times. Tag on the synapse
    // should accumulate to ~3.0 (minus tiny exp-decay from
    // intermediate steps). No plateau ever fires (post never spikes).
    let mut net = fresh_network(2);
    let eid = net.connect(0, 1, 0.1);
    let bp = BtspParams {
        eligibility_window_ms: 1000.0, // very long, minimal decay over a few ms
        ..BtspParams::default()
    };
    net.enable_btsp(bp, None);
    for _ in 0..3 {
        let _ = force_spike(&mut net, 0);
    }
    let tag = net.btsp_synapse_tag[eid];
    assert!(
        tag > 2.5 && tag < 3.05,
        "expected tag ≈ 3.0 (post 3 pre-spikes, minimal decay); got {}",
        tag,
    );
    assert_eq!(
        net.btsp_plateau_events, 0,
        "no plateau event should have fired (post never spikes)",
    );
    assert_eq!(net.btsp_potentiation_events, 0);
}

#[test]
fn potentiation_only_when_plateau_armed_and_tag_nonzero() {
    // 0 → 1 synapse, initial weight 0.1. Drive 3 pre-spikes so the
    // tag accumulates to ~3.0. Then force the post-cell (1) to fire
    // 5 times so plateau arms. On plateau-arm, expect:
    //   Δw = potentiation_strength × tag = 0.4 × 3.0 = 1.2 → clamped
    //   to w_max = 0.8. New weight = (0.1 + 1.2).clamp(0, 0.8) = 0.8.
    //   Tag consumed (= 0).
    let mut net = fresh_network(2);
    let eid = net.connect(0, 1, 0.1);
    // Long burst window so 5 post-spikes reliably cross the
    // threshold of 5; long eligibility so 3 pre-spike tag survives
    // until the post-arm event.
    let bp = BtspParams {
        plateau_window_ms: 1000.0,
        eligibility_window_ms: 1000.0,
        ..BtspParams::iter67_smoke()
    };
    net.enable_btsp(bp, None);
    for _ in 0..3 {
        let _ = force_spike(&mut net, 0);
    }
    let tag_before_plateau = net.btsp_synapse_tag[eid];
    assert!(tag_before_plateau > 2.5);
    // 8 post-spikes — clears the refractory-decay margin (5 alone
    // give trace ≈ 4.98 < 5.0).
    for _ in 0..8 {
        let _ = force_spike(&mut net, 1);
    }
    assert_eq!(net.btsp_plateau_events, 1);
    // Weight should have jumped to w_max (0.8).
    let w_after = net.synapses[eid].weight;
    assert!(
        (w_after - bp.w_max).abs() < 1e-5,
        "expected weight saturated at w_max = {}; got {}",
        bp.w_max,
        w_after,
    );
    // Tag consumed.
    assert!(
        net.btsp_synapse_tag[eid].abs() < 1e-5,
        "expected tag = 0 after consumption; got {}",
        net.btsp_synapse_tag[eid],
    );
    assert!(net.btsp_potentiation_events >= 1);
}

#[test]
fn plateau_disarms_after_post_silence() {
    // Arm plateau via post-spike burst. Idle for long enough to
    // disarm AND for burst trace to decay back below threshold.
    // Then fire another burst → expect a SECOND plateau-arm event
    // (the disarm-then-rearm cycle).
    let mut net = fresh_network(2);
    net.connect(0, 1, 0.1);
    // Long burst window so 8 post-spikes reliably arm; short
    // post-plateau decay so the auto-disarm fires fast on silence;
    // short eligibility so tags clear between bursts (we're testing
    // arming dynamics, not tag persistence).
    let bp = BtspParams {
        plateau_threshold_spikes: 5.0,
        plateau_window_ms: 1000.0,
        post_plateau_decay_ms: 10.0,
        eligibility_window_ms: 5.0,
        ..BtspParams::default()
    };
    net.enable_btsp(bp, None);
    for _ in 0..8 {
        let _ = force_spike(&mut net, 1);
    }
    assert_eq!(net.btsp_plateau_events, 1);
    // Idle long enough for plateau to disarm AND for the burst
    // trace to drop well below the 5.0 threshold under
    // plateau_window_ms = 1000 (need lots of idle time).
    for _ in 0..3000 {
        let _ = idle_one_step(&mut net);
    }
    assert!(
        net.btsp_post_armed_until[1] < net.time,
        "expected disarmed after silence; armed_until = {} time = {}",
        net.btsp_post_armed_until[1],
        net.time,
    );
    assert!(
        net.btsp_post_burst_trace[1] < 1.0,
        "expected burst trace decayed below threshold; got {}",
        net.btsp_post_burst_trace[1],
    );
    // Now fire another burst → expect a second plateau-arm.
    for _ in 0..8 {
        let _ = force_spike(&mut net, 1);
    }
    assert_eq!(
        net.btsp_plateau_events, 2,
        "expected two plateau events (one before idle, one after); \
         got {}",
        net.btsp_plateau_events,
    );
}

#[test]
fn weight_capped_at_w_max() {
    // Tag is 10× the saturation point; potentiation must clamp at
    // w_max (no overshoot).
    let mut net = fresh_network(2);
    let eid = net.connect(0, 1, 0.1);
    let bp = BtspParams {
        eligibility_window_ms: 1000.0, // long, minimal decay
        potentiation_strength: 1.0,    // huge per-spike potentiation
        plateau_threshold_spikes: 5.0,
        plateau_window_ms: 1000.0,
        post_plateau_decay_ms: 50.0,
        w_min: 0.0,
        w_max: 0.8,
        target_gated: true,
        non_target_depression_strength: 0.0,
        heterosynaptic_strength: 0.0,
    };
    net.enable_btsp(bp, None);
    // Accumulate a huge tag.
    for _ in 0..20 {
        let _ = force_spike(&mut net, 0);
    }
    // Plateau-arm (8 spikes for refractory-decay margin).
    for _ in 0..8 {
        let _ = force_spike(&mut net, 1);
    }
    let w = net.synapses[eid].weight;
    assert!(
        w <= bp.w_max + 1e-5,
        "expected weight clamped at w_max = {}; got {}",
        bp.w_max,
        w,
    );
    assert!(
        w >= bp.w_max - 1e-5,
        "expected weight saturated at w_max = {}; got {}",
        bp.w_max,
        w,
    );
}

#[test]
fn btsp_off_path_is_bit_identical() {
    // Sanity: with btsp = None (default), the spike loop must produce
    // identical synapse weights vs running without ever calling
    // enable_btsp. This is the contract that lets iter-67 land
    // without breaking iter-65 / iter-66 / iter-66.5 numerics when
    // c1.btsp = false.
    let mut net_a = fresh_network(2);
    let _ = net_a.connect(0, 1, 0.1);
    let mut net_b = fresh_network(2);
    let _ = net_b.connect(0, 1, 0.1);
    // Drive identical spike trains in both. Net A never touches BTSP
    // API; Net B has BTSP allocated then explicitly disabled (the
    // common case where a downstream caller turns the flag off after
    // experimentation).
    net_b.enable_btsp(BtspParams::default(), None);
    net_b.disable_btsp();
    for _ in 0..10 {
        let _ = force_spike(&mut net_a, 0);
        let _ = force_spike(&mut net_b, 0);
    }
    for _ in 0..5 {
        let _ = force_spike(&mut net_a, 1);
        let _ = force_spike(&mut net_b, 1);
    }
    assert_eq!(
        net_a.synapses[0].weight, net_b.synapses[0].weight,
        "BTSP off path must be bit-identical: net_a w = {}, net_b w = {}",
        net_a.synapses[0].weight, net_b.synapses[0].weight,
    );
    // Net B's BTSP buffers were allocated but the rule didn't fire
    // (disabled). Net A's BTSP buffers are empty. Both networks have
    // the same weight — the contract is on the synapse weight, not
    // on the BTSP buffer state.
}

// ===== Iter-67-γ.5 heterosynaptic competition tests =====
//
// Four tests cover the locked γ.5 rule semantics from
// reports/gate_b_gamma_5_entry.md:
//
//   gamma5_off_path_is_bit_identical
//   gamma5_ltp_couples_to_heterosynaptic_ltd
//   gamma5_skips_self_post_cell
//   gamma5_skips_btsp_inactive_post_cells

/// γ.5 contract: when `heterosynaptic_strength = 0.0`, a γ.5-enabled
/// run produces bit-identical synapse weights to a γ.1.1 run with
/// the same protocol. Off-path bit-identity is the regression guard
/// that lets γ.5 land without breaking γ.1.1's Class (C) Partial
/// verdict.
#[test]
fn gamma5_off_path_is_bit_identical() {
    let mut net_a = fresh_network(2);
    let _ = net_a.connect(0, 1, 0.1);
    let mut net_b = fresh_network(2);
    let _ = net_b.connect(0, 1, 0.1);

    // Net A: γ.1.1 — BTSP on with default params
    // (heterosynaptic_strength = 0.0 by Default).
    net_a.enable_btsp(BtspParams::default(), None);

    // Net B: γ.5-flag-explicit — same as γ.1.1 except
    // heterosynaptic_strength is explicitly written as 0.0 (no host
    // call to set_btsp_target_post — the off-path gate
    // `!self.btsp_target_post.is_empty()` fails so γ.5 is inactive).
    net_b.enable_btsp(
        BtspParams {
            heterosynaptic_strength: 0.0,
            ..BtspParams::default()
        },
        None,
    );

    // Drive identical pre / post spike trains.
    for _ in 0..10 {
        let _ = force_spike(&mut net_a, 0);
        let _ = force_spike(&mut net_b, 0);
    }
    for _ in 0..6 {
        let _ = force_spike(&mut net_a, 1);
        let _ = force_spike(&mut net_b, 1);
    }

    assert_eq!(
        net_a.synapses[0].weight, net_b.synapses[0].weight,
        "γ.5 off path must be bit-identical to γ.1.1: \
         net_a w = {}, net_b w = {}",
        net_a.synapses[0].weight, net_b.synapses[0].weight,
    );
    assert_eq!(
        net_a.btsp_potentiation_events, net_b.btsp_potentiation_events,
        "γ.5 off path: potentiation event counters must match",
    );
    assert_eq!(
        net_b.btsp_heterosynaptic_ltd_events, 0,
        "γ.5 off path: heterosynaptic_ltd_events must remain 0",
    );
}

/// γ.5 core mechanism: a target-cell LTP event triggers bounded LTD
/// on every non-target post-cell (in the BTSP mask) that receives
/// the same tagged pre-cell. Tags on heterosynaptic synapses are
/// NOT consumed (only the LTP path consumes tags).
///
/// Setup: 1 pre cell + 3 post cells. Pre 0 connects to all 3 posts.
/// Post 1 = target; posts 2, 3 = non-target. All 3 synapses planted
/// with tag = 1.0. Plateau-arm post 1.
#[test]
fn gamma5_ltp_couples_to_heterosynaptic_ltd() {
    let mut net = fresh_network(4);
    let eid_p1 = net.connect(0, 1, 0.1); // pre0 → post1 (target)
    let eid_p2 = net.connect(0, 2, 0.5); // pre0 → post2 (non-target)
    let eid_p3 = net.connect(0, 3, 0.5); // pre0 → post3 (non-target)

    let bp = BtspParams {
        plateau_threshold_spikes: 5.0,
        plateau_window_ms: 1.0e6, // → no decay over test span
        eligibility_window_ms: 1.0e6,
        potentiation_strength: 0.4,
        post_plateau_decay_ms: 50.0,
        w_min: 0.0,
        w_max: 0.8,
        target_gated: true,
        non_target_depression_strength: 0.0, // γ.4 OFF
        heterosynaptic_strength: 0.2,        // γ.5 ON
    };
    // BTSP mask covers all 3 post cells (post 1, 2, 3).
    net.enable_btsp(bp, Some(&[1, 2, 3]));

    // Plant identical tags directly on the 3 synapses (bypass the
    // pre-spike accumulation path so we can isolate the LTP/LTD
    // arithmetic).
    net.btsp_synapse_tag[eid_p1] = 1.0;
    net.btsp_synapse_tag[eid_p2] = 1.0;
    net.btsp_synapse_tag[eid_p3] = 1.0;

    // Mark post 1 as the only target. Posts 2, 3 are non-target.
    net.set_btsp_target_post(&[1]);

    let w1_before = net.synapses[eid_p1].weight;
    let w2_before = net.synapses[eid_p2].weight;
    let w3_before = net.synapses[eid_p3].weight;

    // Force post 1 to spike 6 times — well above plateau_threshold = 5.
    // Plateau arms on the 5th spike; one-shot LTP fires; subsequent
    // spikes do not re-trigger.
    for _ in 0..6 {
        let _ = force_spike(&mut net, 1);
    }

    // Synapse 1 (target post): LTP'd by +0.4 * 1.0 = +0.4 (clamp to
    // w_max = 0.8 means 0.1 + 0.4 = 0.5 ≤ 0.8 → exact).
    assert!(
        (net.synapses[eid_p1].weight - (w1_before + 0.4)).abs() < 1e-3,
        "target synapse should be LTP'd: before {}, after {}, expected {}",
        w1_before,
        net.synapses[eid_p1].weight,
        w1_before + 0.4,
    );
    // Synapse 1 tag: consumed (LTP path consumes).
    assert_eq!(
        net.btsp_synapse_tag[eid_p1], 0.0,
        "target synapse tag must be consumed on LTP",
    );

    // Synapses 2, 3 (non-target): LTD'd by -0.2 * 1.0 = -0.2 (clamp
    // [0.0, 0.8] means 0.5 - 0.2 = 0.3 → exact).
    assert!(
        (net.synapses[eid_p2].weight - (w2_before - 0.2)).abs() < 1e-3,
        "non-target synapse 2 should receive heterosynaptic LTD: \
         before {}, after {}, expected {}",
        w2_before,
        net.synapses[eid_p2].weight,
        w2_before - 0.2,
    );
    assert!(
        (net.synapses[eid_p3].weight - (w3_before - 0.2)).abs() < 1e-3,
        "non-target synapse 3 should receive heterosynaptic LTD: \
         before {}, after {}, expected {}",
        w3_before,
        net.synapses[eid_p3].weight,
        w3_before - 0.2,
    );

    // Synapses 2, 3 tags: NOT consumed (heterosynaptic LTD does not
    // consume; natural eligibility-window decay handles cleanup).
    assert!(
        net.btsp_synapse_tag[eid_p2] > 0.0,
        "non-target synapse 2 tag must NOT be consumed by heterosynaptic LTD",
    );
    assert!(
        net.btsp_synapse_tag[eid_p3] > 0.0,
        "non-target synapse 3 tag must NOT be consumed by heterosynaptic LTD",
    );

    // Counters: 1 LTP event, 2 heterosynaptic-LTD events.
    assert_eq!(
        net.btsp_potentiation_events, 1,
        "expected exactly 1 potentiation event (1 target with 1 incoming)",
    );
    assert_eq!(
        net.btsp_heterosynaptic_ltd_events, 2,
        "expected exactly 2 heterosynaptic-LTD events (2 non-target neighbours)",
    );
}

/// γ.5 must skip the target post-cell itself in the heterosynaptic
/// loop. If the skip-self check were missing, the target synapse
/// would receive both LTP (`+strength`) and immediate LTD
/// (`-hetero_strength`), producing a smaller net weight change.
///
/// Setup: 1 pre cell + 1 post cell (only the target). Synapse
/// pre0→post1 only. Plant tag, plateau post 1, verify weight is
/// exactly LTP'd (not LTP-then-LTD).
#[test]
fn gamma5_skips_self_post_cell() {
    let mut net = fresh_network(2);
    let eid = net.connect(0, 1, 0.1);

    let bp = BtspParams {
        plateau_threshold_spikes: 5.0,
        plateau_window_ms: 1.0e6, // → no decay over test span
        eligibility_window_ms: 1.0e6,
        potentiation_strength: 0.4,
        post_plateau_decay_ms: 50.0,
        w_min: 0.0,
        w_max: 0.8,
        target_gated: true,
        non_target_depression_strength: 0.0,
        heterosynaptic_strength: 0.2, // γ.5 ON
    };
    net.enable_btsp(bp, Some(&[1]));
    net.btsp_synapse_tag[eid] = 1.0;
    net.set_btsp_target_post(&[1]);

    let w_before = net.synapses[eid].weight;
    for _ in 0..6 {
        let _ = force_spike(&mut net, 1);
    }

    // Net change must be exactly +0.4 (LTP only). If skip-self were
    // missing, the same synapse would also receive a -0.2 hetero-LTD
    // → net change = +0.2, which is wrong.
    assert!(
        (net.synapses[eid].weight - (w_before + 0.4)).abs() < 1e-3,
        "target's own synapse must receive LTP only, not LTP+LTD: \
         before {}, after {}, expected {}",
        w_before,
        net.synapses[eid].weight,
        w_before + 0.4,
    );
    assert_eq!(
        net.btsp_potentiation_events, 1,
        "exactly 1 LTP event expected",
    );
    assert_eq!(
        net.btsp_heterosynaptic_ltd_events, 0,
        "no heterosynaptic-LTD when no non-target neighbours exist; \
         skip-self must skip the target itself",
    );
}

/// γ.5 must scope heterosynaptic LTD to the BTSP mask only. Synapses
/// outgoing from the same pre-cell to post-cells *outside* the C1
/// readout layer (i.e. R2-E recurrent synapses) must be unaffected
/// — the rule lives on R2-E → C1, not on R2-R2.
///
/// Setup: 1 pre + 2 posts in the BTSP mask + 1 post outside the mask.
/// Synapses pre0→post1 (target, in mask), pre0→post2 (non-target, in
/// mask), pre0→post3 (NOT in mask — represents R2-R2 recurrent).
/// Tags planted on all three. Plateau post 1.
#[test]
fn gamma5_skips_btsp_inactive_post_cells() {
    let mut net = fresh_network(4);
    let eid_p1 = net.connect(0, 1, 0.1); // pre0 → post1 (target, in mask)
    let eid_p2 = net.connect(0, 2, 0.5); // pre0 → post2 (non-target, in mask)
    let eid_p3 = net.connect(0, 3, 0.7); // pre0 → post3 (OUT of mask)

    let bp = BtspParams {
        plateau_threshold_spikes: 5.0,
        plateau_window_ms: 1.0e6, // → no decay over test span
        eligibility_window_ms: 1.0e6,
        potentiation_strength: 0.4,
        post_plateau_decay_ms: 50.0,
        w_min: 0.0,
        w_max: 0.8,
        target_gated: true,
        non_target_depression_strength: 0.0,
        heterosynaptic_strength: 0.2,
    };
    // BTSP mask: posts 1, 2 only. Post 3 is NOT in the mask.
    net.enable_btsp(bp, Some(&[1, 2]));
    net.btsp_synapse_tag[eid_p1] = 1.0;
    net.btsp_synapse_tag[eid_p2] = 1.0;
    net.btsp_synapse_tag[eid_p3] = 1.0;
    net.set_btsp_target_post(&[1]);

    let w3_before = net.synapses[eid_p3].weight;

    for _ in 0..6 {
        let _ = force_spike(&mut net, 1);
    }

    // Post 3 is outside the BTSP mask — synapse weight must be
    // unchanged.
    assert_eq!(
        net.synapses[eid_p3].weight, w3_before,
        "synapse to post outside BTSP mask must be unchanged: \
         before {}, after {}",
        w3_before, net.synapses[eid_p3].weight,
    );
    // Tag on the out-of-mask synapse: only natural decay, no LTP /
    // LTD consumption (rule didn't touch it).
    assert!(
        (net.btsp_synapse_tag[eid_p3] - 1.0).abs() < 1e-3,
        "tag on out-of-mask synapse must be unchanged (modulo natural decay): \
         observed {}",
        net.btsp_synapse_tag[eid_p3],
    );
    // Counters: exactly 1 LTP (post 1) + 1 hetero-LTD (post 2 only;
    // post 3 was skipped because it's not in the BTSP mask).
    assert_eq!(net.btsp_potentiation_events, 1);
    assert_eq!(net.btsp_heterosynaptic_ltd_events, 1);
}
