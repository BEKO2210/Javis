//! Throughput micro-benchmark, single CPU thread.
//!
//! Builds a recurrent network of `N` LIF neurons with random sparse
//! connectivity (p = 0.1) and STDP enabled, drives every neuron with
//! independent Poisson background, and simulates one second of network
//! time. Reports wall-clock ratio and throughput numbers.
//!
//! Run with:
//!     cargo run --release --example throughput

use std::time::Instant;

use snn_core::{LifNeuron, LifParams, Network, PoissonInput, Rng, StdpParams};

const DT: f32 = 0.1;
const SIM_MS: f32 = 1000.0;
const P_CONNECT: f32 = 0.1;

fn build(n: usize, seed: u64) -> Network {
    let mut rng = Rng::new(seed);
    let mut net = Network::new(DT);
    let mut stdp = StdpParams::default();
    stdp.a_plus = 0.05;
    stdp.a_minus = 0.025;
    net.enable_stdp(stdp);

    for _ in 0..n {
        net.add_neuron(LifNeuron::new(LifParams::default()));
    }
    for pre in 0..n {
        for post in 0..n {
            if pre == post {
                continue;
            }
            if rng.bernoulli(P_CONNECT) {
                let w = rng.range_f32(0.05, 0.30);
                net.connect(pre, post, w);
            }
        }
    }
    net
}

fn run_size(n: usize) {
    let mut net = build(n, 2026);
    let synapses = net.synapses.len();
    let gens: Vec<PoissonInput> = (0..n)
        .map(|i| PoissonInput { target: i, rate_hz: 50.0, current_per_spike: 80.0 })
        .collect();

    let mut rng = Rng::new(11);
    let steps = (SIM_MS / DT) as usize;
    let mut external = vec![0.0_f32; n];
    let mut total_spikes = 0u64;

    let start = Instant::now();
    for _ in 0..steps {
        external.iter_mut().for_each(|x| *x = 0.0);
        snn_core::poisson::drive(&gens, &mut external, DT, &mut rng);
        let fired = net.step(&external);
        total_spikes += fired.len() as u64;
    }
    let wall = start.elapsed();

    let wall_ms = wall.as_secs_f64() * 1000.0;
    let rtf = SIM_MS as f64 / wall_ms;
    let neuron_steps = (n as u64) * (steps as u64);
    let neuron_steps_per_s = neuron_steps as f64 / wall.as_secs_f64();
    let syn_events_per_s = net.synapse_events as f64 / wall.as_secs_f64();
    let mean_rate = (total_spikes as f64 / n as f64) / (SIM_MS as f64 / 1000.0);

    println!(
        "N={n:5}  E={synapses:7}  wall={wall_ms:7.1}ms  RTF={rtf:6.2}x  \
         spikes={total_spikes:7}  ⟨rate⟩={mean_rate:5.1}Hz  \
         neuron-steps/s={neuron_steps_per_s:>11.0}  syn-events/s={syn_events_per_s:>11.0}",
    );
}

fn main() {
    println!(
        "Workload: 1 s simulated, dt={DT} ms, p_connect={P_CONNECT}, \
         STDP on, Poisson background 50 Hz / 80 nA per neuron"
    );
    println!();
    for n in [100usize, 250, 500, 1000, 2000] {
        run_size(n);
    }
}
