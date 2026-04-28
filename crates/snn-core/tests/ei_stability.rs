//! Test 5: an 80/20 E/I balanced recurrent network with STDP and Poisson
//! drive stays in an asynchronous low-rate regime — no runaway. This is
//! the regression guard for Dale's-principle wiring.

use snn_core::{LifNeuron, LifParams, Network, NeuronKind, PoissonInput, Rng, StdpParams};

const DT: f32 = 0.1;
const N: usize = 500;
const P_CONNECT: f32 = 0.1;
const INH_FRACTION: f32 = 0.20;

fn build(seed: u64) -> Network {
    let mut rng = Rng::new(seed);
    let mut net = Network::new(DT);
    let mut stdp = StdpParams::default();
    stdp.a_plus = 0.05;
    stdp.a_minus = 0.025;
    net.enable_stdp(stdp);

    let n_inh = (N as f32 * INH_FRACTION) as usize;
    let n_exc = N - n_inh;
    for _ in 0..n_exc {
        net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    }
    for _ in 0..n_inh {
        net.add_neuron(LifNeuron::inhibitory(LifParams::default()));
    }

    let g_exc = 0.20_f32;
    let g_inh = 0.80_f32;
    for pre in 0..N {
        let g = match net.neurons[pre].kind {
            NeuronKind::Excitatory => g_exc,
            NeuronKind::Inhibitory => g_inh,
        };
        for post in 0..N {
            if pre == post {
                continue;
            }
            if rng.bernoulli(P_CONNECT) {
                let w = rng.range_f32(0.5 * g, 1.0 * g);
                net.connect(pre, post, w);
            }
        }
    }
    net
}

#[test]
fn balanced_network_stays_below_runaway() {
    let mut net = build(2026);

    let gens: Vec<PoissonInput> = (0..N)
        .map(|i| PoissonInput { target: i, rate_hz: 50.0, current_per_spike: 80.0 })
        .collect();

    let mut rng = Rng::new(11);
    let sim_ms = 1000.0_f32;
    let steps = (sim_ms / DT) as usize;
    let mut external = vec![0.0_f32; N];
    let mut spikes_e = 0u64;
    let mut spikes_i = 0u64;

    for _ in 0..steps {
        external.iter_mut().for_each(|x| *x = 0.0);
        snn_core::poisson::drive(&gens, &mut external, DT, &mut rng);
        let fired = net.step(&external);
        for &id in &fired {
            if net.neurons[id].kind == NeuronKind::Excitatory {
                spikes_e += 1;
            } else {
                spikes_i += 1;
            }
        }
    }

    let n_exc = net.neurons.iter().filter(|x| x.kind == NeuronKind::Excitatory).count() as f32;
    let n_inh = (N as f32) - n_exc;
    let rate_e = spikes_e as f32 / n_exc / (sim_ms / 1000.0);
    let rate_i = spikes_i as f32 / n_inh / (sim_ms / 1000.0);

    eprintln!("N={N} rate_E={rate_e:.2} Hz  rate_I={rate_i:.2} Hz  spikes={spikes_e}+{spikes_i}");

    // Hard regression guard: anything above 30 Hz mean is "runaway-ish"
    // for this drive level. Healthy balanced regime is single-digit Hz.
    assert!(rate_e < 30.0, "E rate runaway: {rate_e} Hz");
    assert!(rate_i < 30.0, "I rate runaway: {rate_i} Hz");
    // And we need *some* activity, otherwise the network is dead.
    assert!(spikes_e + spikes_i > 50, "network is silent");
}
