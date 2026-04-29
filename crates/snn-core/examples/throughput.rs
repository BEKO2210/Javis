//! Throughput micro-benchmark, single CPU thread.
//!
//! Builds a recurrent network of `N` LIF neurons with random sparse
//! connectivity (p = 0.1) and STDP enabled. 80% of the neurons are
//! excitatory, 20% inhibitory (Dale's principle). Inhibitory weights
//! are scaled up by `g_inh` to keep E/I balance against the larger
//! excitatory population. Drives every neuron with independent Poisson
//! background and simulates one second of network time.
//!
//! Run with:
//!     cargo run --release --example throughput

use std::time::Instant;

use snn_core::{LifNeuron, LifParams, Network, NeuronKind, PoissonInput, Rng, StdpParams};

const DT: f32 = 0.1;
const SIM_MS: f32 = 1000.0;
const P_CONNECT: f32 = 0.1;
const INH_FRACTION: f32 = 0.20;

fn build(n: usize, seed: u64) -> Network {
    let mut rng = Rng::new(seed);
    let mut net = Network::new(DT);
    net.enable_stdp(StdpParams {
        a_plus: 0.05,
        a_minus: 0.025,
        ..StdpParams::default()
    });

    // Assign neuron types: first 80% excitatory, last 20% inhibitory.
    let n_inh = (n as f32 * INH_FRACTION) as usize;
    let n_exc = n - n_inh;
    for _ in 0..n_exc {
        net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    }
    for _ in 0..n_inh {
        net.add_neuron(LifNeuron::inhibitory(LifParams::default()));
    }

    // Connect with p_connect, weights scaled by pre-neuron type.
    // Inhibitory weights are stronger to compensate for fewer I neurons.
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
            if rng.bernoulli(P_CONNECT) {
                let w = rng.range_f32(0.5 * g, 1.0 * g);
                net.connect(pre, post, w);
            }
        }
    }
    net
}

fn run_size(n: usize) {
    let mut net = build(n, 2026);
    let synapses = net.synapses.len();
    let n_exc = net
        .neurons
        .iter()
        .filter(|x| x.kind == NeuronKind::Excitatory)
        .count();
    let n_inh = net.neurons.len() - n_exc;
    let gens: Vec<PoissonInput> = (0..n)
        .map(|i| PoissonInput {
            target: i,
            rate_hz: 50.0,
            current_per_spike: 80.0,
        })
        .collect();

    let mut rng = Rng::new(11);
    let steps = (SIM_MS / DT) as usize;
    let mut external = vec![0.0_f32; n];
    let mut total_spikes = 0u64;
    let mut e_spikes = 0u64;
    let mut i_spikes = 0u64;

    let start = Instant::now();
    for _ in 0..steps {
        external.iter_mut().for_each(|x| *x = 0.0);
        snn_core::poisson::drive(&gens, &mut external, DT, &mut rng);
        let fired = net.step(&external);
        for &id in &fired {
            total_spikes += 1;
            if net.neurons[id].kind == NeuronKind::Excitatory {
                e_spikes += 1;
            } else {
                i_spikes += 1;
            }
        }
    }
    let wall = start.elapsed();

    let wall_ms = wall.as_secs_f64() * 1000.0;
    let rtf = SIM_MS as f64 / wall_ms;
    let neuron_steps = (n as u64) * (steps as u64);
    let neuron_steps_per_s = neuron_steps as f64 / wall.as_secs_f64();
    let syn_events_per_s = net.synapse_events as f64 / wall.as_secs_f64();
    let mean_rate_e = (e_spikes as f64 / n_exc.max(1) as f64) / (SIM_MS as f64 / 1000.0);
    let mean_rate_i = (i_spikes as f64 / n_inh.max(1) as f64) / (SIM_MS as f64 / 1000.0);

    println!(
        "N={n:5} (E={n_exc:5}, I={n_inh:5})  E={synapses:7}  wall={wall_ms:7.1}ms  RTF={rtf:7.2}x  \
         spikes={total_spikes:7}  ⟨rate_E⟩={mean_rate_e:5.1}Hz  ⟨rate_I⟩={mean_rate_i:5.1}Hz  \
         neuron-steps/s={neuron_steps_per_s:>11.0}  syn-events/s={syn_events_per_s:>11.0}",
    );
}

fn main() {
    println!(
        "Workload: 1 s simulated, dt={DT} ms, p_connect={P_CONNECT}, \
         STDP on, 80/20 E/I, Poisson background 50 Hz / 80 nA per neuron"
    );
    println!();
    for n in [100usize, 250, 500, 1000, 2000, 4000] {
        run_size(n);
    }
}
