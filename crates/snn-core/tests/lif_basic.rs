//! Test 1: a single LIF neuron under constant supra-threshold input
//! fires at approximately the analytically expected rate.
//!
//! Analytical inter-spike interval for the LIF model with reset != rest:
//!   ISI = τ_m · ln((V_reset − V_ss) / (V_th − V_ss)) + refractory
//! where V_ss = V_rest + R·I.
//!
//! With defaults (τ_m=20, V_rest=−70, V_reset=−75, V_th=−55, R=10, ref=2)
//! and I=2 nA: V_ss=−50, ratio=0.2, ISI ≈ 32.19 + 2 = 34.19 ms,
//! so ≈ 29 spikes in 1000 ms.

use snn_core::{LifNeuron, LifParams};

#[test]
fn single_neuron_fires_at_expected_rate() {
    let mut n = LifNeuron::new(LifParams::default());
    let dt = 0.1_f32;
    let duration = 1000.0_f32;
    let i_input = 2.0_f32;

    let mut t = 0.0;
    let mut spikes = 0usize;
    while t < duration {
        if n.step(t, dt, i_input) {
            spikes += 1;
        }
        t += dt;
    }

    // Expected ~29 spikes. Allow generous tolerance for Euler integration.
    assert!(
        (25..=33).contains(&spikes),
        "expected ~29 spikes in 1s with I=2nA, got {spikes}",
    );
}

#[test]
fn subthreshold_input_does_not_fire() {
    let mut n = LifNeuron::new(LifParams::default());
    let dt = 0.1_f32;

    // Rheobase = (V_th − V_rest) / R_m = 15 / 10 = 1.5 nA.
    // 1.0 nA is sub-threshold → V settles at −60 mV, never reaches −55.
    let mut spiked = false;
    let mut t = 0.0;
    while t < 500.0 {
        if n.step(t, dt, 1.0) {
            spiked = true;
            break;
        }
        t += dt;
    }
    assert!(!spiked, "neuron should not fire below rheobase");
    assert!(
        n.v < -59.0 && n.v > -61.0,
        "V should approach −60 mV, got {}",
        n.v
    );
}
