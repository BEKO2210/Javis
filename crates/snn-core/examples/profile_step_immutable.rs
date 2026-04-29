//! Phase-by-phase profiler for `Network::step_immutable`.
//!
//! Builds the same R2 memory cortex as viz::state (sparse 2000-neuron
//! E/I network, p=0.1 connectivity) and runs the read-only step in a
//! tight loop. Each call is broken down into three phases via
//! per-call `Instant::now()` markers so we can see where the 9 ms
//! per-recall server time actually goes:
//!
//! 1. **decay**: scale every active synaptic channel by exp(-dt/tau)
//! 2. **lif**:   per-neuron integration + spike detection
//! 3. **deliver**: walk outgoing edges of each spiker, write into
//!    the post-synaptic channel buffer
//!
//! Run:
//!   cargo run --release -p snn-core --example profile_step_immutable
//!
//! Output is a single block of mean / p50 / p99 timings per phase
//! plus the share of total step time. Run twice with different
//! drive intensities to see how the decay-vs-deliver balance shifts.

use std::time::Instant;

use snn_core::{LifNeuron, LifParams, Network, NetworkState, NeuronKind, Rng};

const DT: f32 = 0.1;
const N: usize = 2000;
const P_CONNECT: f32 = 0.1;

fn build_network(seed: u64) -> Network {
    let mut net = Network::new(DT);
    let mut rng = Rng::new(seed);
    let n_inh = (N as f32 * 0.2) as usize;
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

/// Manually-phased re-implementation of `Network::step_immutable`.
/// Returns (decay_ns, lif_ns, deliver_ns, fired_count). Identical
/// math to the canonical method; the only difference is that each
/// phase is wrapped in its own `Instant::now()` pair.
fn step_phased(
    net: &Network,
    state: &mut NetworkState,
    external: &[f32],
) -> (u128, u128, u128, usize) {
    let dt = net.dt;
    let t = state.time;

    // ---------- phase 1: channel decay ----------
    let t0 = Instant::now();
    let decay_ampa = (-dt / net.tau_syn_ms.max(1e-3)).exp();
    for x in state.i_syn.iter_mut() {
        *x *= decay_ampa;
    }
    if !state.i_syn_nmda.is_empty() {
        let d = (-dt / net.tau_nmda_ms.max(1e-3)).exp();
        for x in state.i_syn_nmda.iter_mut() {
            *x *= d;
        }
    }
    if !state.i_syn_gaba.is_empty() {
        let d = (-dt / net.tau_gaba_ms.max(1e-3)).exp();
        for x in state.i_syn_gaba.iter_mut() {
            *x *= d;
        }
    }
    let decay_ns = t0.elapsed().as_nanos();

    // ---------- phase 2: LIF integration ----------
    let t1 = Instant::now();
    let mut fired: Vec<usize> = Vec::new();
    for (idx, neuron) in net.neurons.iter().enumerate() {
        let ext = external.get(idx).copied().unwrap_or(0.0);
        let mut total = ext + state.i_syn[idx];
        if let Some(v) = state.i_syn_nmda.get(idx) {
            total += *v;
        }
        if let Some(v) = state.i_syn_gaba.get(idx) {
            total += *v;
        }
        // Inline LIF math (identical to lif_step_state).
        let v = &mut state.v[idx];
        let refr = &mut state.refractory_until[idx];
        let last = &mut state.last_spike[idx];
        let p = &neuron.params;
        if t < *refr {
            *v = p.v_reset;
        } else {
            let dv = dt / p.tau_m * (-(*v - p.v_rest) + p.r_m * total);
            *v += dv;
            if *v >= p.v_threshold {
                *v = p.v_reset;
                *refr = t + p.refractory;
                *last = t;
                fired.push(idx);
            }
        }
    }
    let lif_ns = t1.elapsed().as_nanos();

    // ---------- phase 3: synaptic delivery ----------
    let t2 = Instant::now();
    for &src in &fired {
        let src_kind = net.neurons[src].kind;
        let sign: f32 = match src_kind {
            NeuronKind::Excitatory => 1.0,
            NeuronKind::Inhibitory => -1.0,
        };
        for &eid in &net.outgoing[src] {
            let s = &net.synapses[eid as usize];
            // We only test AMPA topology in this profile; bypass the
            // ensure_channel branch since we know the channel exists.
            state.i_syn[s.post] += sign * s.weight;
            state.synapse_events += 1;
        }
    }
    let deliver_ns = t2.elapsed().as_nanos();

    state.step_counter = state.step_counter.wrapping_add(1);
    state.time += dt;
    (decay_ns, lif_ns, deliver_ns, fired.len())
}

fn pct(values: &mut [u128], p: f64) -> u128 {
    values.sort_unstable();
    let k = ((values.len() - 1) as f64 * p) as usize;
    values[k]
}

fn main() {
    eprintln!("Building 2000-neuron network …");
    let net = build_network(2027);
    eprintln!(
        "  neurons={} synapses={} avg-degree={:.1}",
        net.neurons.len(),
        net.synapses.len(),
        net.synapses.len() as f32 / net.neurons.len() as f32,
    );

    let mut state = net.fresh_state();

    // Drive a chunk of R1-equivalent neurons so the network actually
    // fires — without spikes, phase 3 collapses to zero work and we
    // measure nothing useful. r_m=10 + tau_m=20 ms means a sustained
    // drive of ~2 nA depolarises by ~20 mV at steady state, well past
    // threshold (15 mV above rest). 200 driven neurons gives recurrent
    // activity that propagates through the rest of the network.
    let mut external = vec![0.0_f32; N];
    for slot in external.iter_mut().take(200) {
        *slot = 2.0;
    }

    // Warm up so traces and channels are at steady state, not all
    // zero (which would skew phase 3 toward zero spikes).
    for _ in 0..200 {
        let _ = net.step_immutable(&mut state, &external);
    }

    // Measurement: 5 000 phased steps. Collect ns per phase.
    const N_STEPS: usize = 5_000;
    let mut decays = Vec::with_capacity(N_STEPS);
    let mut lifs = Vec::with_capacity(N_STEPS);
    let mut delivers = Vec::with_capacity(N_STEPS);
    let mut fires = Vec::with_capacity(N_STEPS);

    for _ in 0..N_STEPS {
        let (d, l, dv, f) = step_phased(&net, &mut state, &external);
        decays.push(d);
        lifs.push(l);
        delivers.push(dv);
        fires.push(f as u128);
    }

    let n = N_STEPS as u128;
    let sum_d: u128 = decays.iter().sum();
    let sum_l: u128 = lifs.iter().sum();
    let sum_v: u128 = delivers.iter().sum();
    let sum_f: u128 = fires.iter().sum();
    let mean_d = sum_d / n;
    let mean_l = sum_l / n;
    let mean_v = sum_v / n;
    let mean_f = sum_f as f64 / n as f64;
    let total = mean_d + mean_l + mean_v;

    let p50_d = pct(&mut decays, 0.50);
    let p99_d = pct(&mut decays, 0.99);
    let p50_l = pct(&mut lifs, 0.50);
    let p99_l = pct(&mut lifs, 0.99);
    let p50_v = pct(&mut delivers, 0.50);
    let p99_v = pct(&mut delivers, 0.99);

    println!(
        "\n=== {}-step profile, {} neurons, {} synapses ===",
        N_STEPS,
        net.neurons.len(),
        net.synapses.len()
    );
    println!("                mean ns        p50         p99       share");
    println!(
        "  decay      {:>10}  {:>10}  {:>10}    {:>5.1}%",
        mean_d,
        p50_d,
        p99_d,
        100.0 * mean_d as f64 / total as f64,
    );
    println!(
        "  lif        {:>10}  {:>10}  {:>10}    {:>5.1}%",
        mean_l,
        p50_l,
        p99_l,
        100.0 * mean_l as f64 / total as f64,
    );
    println!(
        "  deliver    {:>10}  {:>10}  {:>10}    {:>5.1}%",
        mean_v,
        p50_v,
        p99_v,
        100.0 * mean_v as f64 / total as f64,
    );
    println!("  ----------------------------------------------------------------",);
    println!(
        "  step total {:>10} ns                                  100.0%",
        total,
    );
    println!("  spikes/step mean = {:.2} (drives phase 3 cost)", mean_f);
}
