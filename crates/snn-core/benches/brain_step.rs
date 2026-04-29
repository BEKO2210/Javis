//! Benchmarks for `Brain::step` — exercises the multi-region path
//! including the `PendingQueue` heap and inter-region delivery. The
//! single-region `Network::step` path has its own benchmark file
//! (`network_step.rs`); this one specifically stresses the inter-
//! region book-keeping that doesn't show up there.
//!
//! Run locally:
//!   `cargo bench -p snn-core --bench brain_step`

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use snn_core::{Brain, LifNeuron, LifParams, NeuronKind, Region, Rng};

const DT: f32 = 0.1;
const P_CONNECT: f32 = 0.1;
const FAN_OUT: usize = 10;

fn build_region(n: usize, seed: u64) -> Region {
    let mut rng = Rng::new(seed);
    let mut region = Region::new("R", DT);
    let net = &mut region.network;
    let n_inh = (n as f32 * 0.2) as usize;
    let n_exc = n - n_inh;
    for _ in 0..n_exc {
        net.add_neuron(LifNeuron::excitatory(LifParams::default()));
    }
    for _ in 0..n_inh {
        net.add_neuron(LifNeuron::inhibitory(LifParams::default()));
    }
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
    region
}

fn build_two_region_brain(n: usize) -> Brain {
    let mut brain = Brain::new(DT);
    brain.add_region(build_region(n, 2027));
    brain.add_region(build_region(n, 2028));

    let mut rng = Rng::new(2029);
    for src in 0..n {
        for _ in 0..FAN_OUT {
            let dst = (rng.next_u64() as usize) % n;
            // Vary the delay so the heap actually has work to do.
            let delay = 1.0 + rng.range_f32(0.0, 4.0);
            brain.connect(0, src, 1, dst, 1.5, delay);
        }
    }
    brain
}

fn bench_brain_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("brain_step_two_region");
    for &n in &[200usize, 500, 1000] {
        let mut brain = build_two_region_brain(n);
        let externals = vec![vec![0.4_f32; n], vec![0.0_f32; n]];
        // Warm-up: let the pending queue accumulate some events.
        for _ in 0..50 {
            let _ = brain.step(&externals);
        }
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let _ = black_box(brain.step(black_box(&externals)));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_brain_step);
criterion_main!(benches);
