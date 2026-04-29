//! Benchmarks for `Network::step` — the inner loop of every Javis
//! operation. We benchmark three sizes (small / medium / large) so a
//! regression on either the per-neuron LIF integration or the
//! synapse-delivery loop shows up as a clear shift in one of the
//! curves.
//!
//! Run locally:
//!   `cargo bench -p snn-core --bench network_step`
//!
//! Criterion writes results to `target/criterion/` and produces an
//! HTML report at `target/criterion/report/index.html`.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use snn_core::{LifNeuron, LifParams, Network, NeuronKind, Rng, StdpParams};

const DT: f32 = 0.1;

fn build_network(n: usize, seed: u64, with_stdp: bool) -> Network {
    let mut net = Network::new(DT);
    if with_stdp {
        net.enable_stdp(StdpParams::default());
    }
    let mut rng = Rng::new(seed);
    let n_inh = (n as f32 * 0.2) as usize;
    let n_exc = n - n_inh;
    for _ in 0..n_exc {
        net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    }
    for _ in 0..n_inh {
        net.add_neuron(LifNeuron::inhibitory(LifParams::default()));
    }
    // p = 0.1 sparse Erdős–Rényi, exactly like the integration tests.
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

fn bench_step_passive(c: &mut Criterion) {
    let mut group = c.benchmark_group("network_step_passive");
    for &n in &[100usize, 500, 1000] {
        let mut net = build_network(n, 2027, false);
        let drive = vec![0.5_f32; n];
        // Warm up the network briefly so the membrane potentials are
        // not all at rest — closer to steady-state behaviour.
        for _ in 0..50 {
            let _ = net.step(&drive);
        }

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let _ = black_box(net.step(black_box(&drive)));
            });
        });
    }
    group.finish();
}

fn bench_step_with_stdp(c: &mut Criterion) {
    let mut group = c.benchmark_group("network_step_stdp");
    for &n in &[500usize, 1000] {
        let mut net = build_network(n, 2027, true);
        let drive = vec![0.5_f32; n];
        for _ in 0..50 {
            let _ = net.step(&drive);
        }

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let _ = black_box(net.step(black_box(&drive)));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_step_passive, bench_step_with_stdp);
criterion_main!(benches);
