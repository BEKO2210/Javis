//! Network of LIF neurons connected by synapses.
//!
//! Storage layout (sparse adjacency, GPU-portable):
//! - `synapses` — flat `Vec<Synapse>` of all edges.
//! - `outgoing[pre]` — edge indices whose pre-neuron is `pre`.
//! - `incoming[post]` — edge indices whose post-neuron is `post`.
//!
//! When a neuron fires we iterate only its `outgoing` (delivery + LTD) and
//! `incoming` (LTP) buckets — O(degree) instead of O(E) per spike.
//!
//! Excitatory pre-neurons add `+weight` to the post-synaptic current;
//! inhibitory pre-neurons subtract. Weights themselves are non-negative
//! magnitudes, clamped by STDP into `[w_min, w_max]`.

use crate::neuron::{LifNeuron, NeuronKind};
use crate::stdp::StdpParams;
use crate::synapse::Synapse;

pub struct Network {
    pub neurons: Vec<LifNeuron>,
    pub synapses: Vec<Synapse>,
    pub outgoing: Vec<Vec<u32>>,
    pub incoming: Vec<Vec<u32>>,
    pub i_syn: Vec<f32>,
    pub pre_trace: Vec<f32>,
    pub post_trace: Vec<f32>,
    pub time: f32,
    pub dt: f32,
    pub stdp: Option<StdpParams>,
    /// Cumulative count of synaptic deliveries since construction.
    /// Useful for benchmarking real work done.
    pub synapse_events: u64,
}

impl Network {
    pub fn new(dt: f32) -> Self {
        Self {
            neurons: Vec::new(),
            synapses: Vec::new(),
            outgoing: Vec::new(),
            incoming: Vec::new(),
            i_syn: Vec::new(),
            pre_trace: Vec::new(),
            post_trace: Vec::new(),
            time: 0.0,
            dt,
            stdp: None,
            synapse_events: 0,
        }
    }

    pub fn add_neuron(&mut self, n: LifNeuron) -> usize {
        let id = self.neurons.len();
        self.neurons.push(n);
        self.outgoing.push(Vec::new());
        self.incoming.push(Vec::new());
        self.i_syn.push(0.0);
        self.pre_trace.push(0.0);
        self.post_trace.push(0.0);
        id
    }

    pub fn connect(&mut self, pre: usize, post: usize, weight: f32) -> usize {
        let id = self.synapses.len();
        self.synapses.push(Synapse::new(pre, post, weight));
        self.outgoing[pre].push(id as u32);
        self.incoming[post].push(id as u32);
        id
    }

    pub fn enable_stdp(&mut self, params: StdpParams) {
        self.stdp = Some(params);
    }

    pub fn disable_stdp(&mut self) {
        self.stdp = None;
    }

    /// Reset transient state (membrane potentials, synaptic currents,
    /// STDP traces, refractory clocks, time, event counter). Synapse
    /// weights and network topology are preserved.
    pub fn reset_state(&mut self) {
        for (idx, n) in self.neurons.iter_mut().enumerate() {
            n.v = n.params.v_rest;
            n.refractory_until = f32::NEG_INFINITY;
            n.last_spike = f32::NEG_INFINITY;
            self.i_syn[idx] = 0.0;
            self.pre_trace[idx] = 0.0;
            self.post_trace[idx] = 0.0;
        }
        self.time = 0.0;
        self.synapse_events = 0;
    }

    /// Advance the network one timestep with optional external currents.
    /// `external` length must equal the number of neurons (or be empty,
    /// meaning zero external input). Returns indices that fired.
    pub fn step(&mut self, external: &[f32]) -> Vec<usize> {
        let dt = self.dt;
        let t = self.time;

        // 1) Decay synaptic currents (fixed τ_syn = 5 ms for now).
        let decay_psc = (-dt / 5.0_f32).exp();
        for x in self.i_syn.iter_mut() {
            *x *= decay_psc;
        }

        // 2) Decay STDP traces.
        if let Some(p) = self.stdp {
            let dp = (-dt / p.tau_plus).exp();
            let dm = (-dt / p.tau_minus).exp();
            for x in self.pre_trace.iter_mut() {
                *x *= dp;
            }
            for x in self.post_trace.iter_mut() {
                *x *= dm;
            }
        }

        // 3) Step every LIF, recording spikes.
        let mut fired: Vec<usize> = Vec::new();
        for (idx, n) in self.neurons.iter_mut().enumerate() {
            let ext = external.get(idx).copied().unwrap_or(0.0);
            if n.step(t, dt, ext + self.i_syn[idx]) {
                fired.push(idx);
            }
        }

        // 4) Deliver synaptic effects + STDP updates for spikes this step.
        let stdp = self.stdp;
        for &src in &fired {
            if stdp.is_some() {
                self.pre_trace[src] += 1.0;
                self.post_trace[src] += 1.0;
            }
            let sign: f32 = match self.neurons[src].kind {
                NeuronKind::Excitatory => 1.0,
                NeuronKind::Inhibitory => -1.0,
            };

            // Outgoing edges from `src`: deliver PSC, optional LTD update.
            // Iterate by index so we keep `self.synapses` mutable while the
            // adjacency vector itself is not touched.
            let n_out = self.outgoing[src].len();
            for i in 0..n_out {
                let eid = self.outgoing[src][i] as usize;
                let post = self.synapses[eid].post;
                let w = self.synapses[eid].weight;
                self.i_syn[post] += sign * w;
                self.synapse_events += 1;
                if let Some(p) = stdp {
                    let new_w = (w - p.a_minus * self.post_trace[post])
                        .clamp(p.w_min, p.w_max);
                    self.synapses[eid].weight = new_w;
                }
            }

            // Incoming edges to `src`: optional LTP update on each.
            if let Some(p) = stdp {
                let n_in = self.incoming[src].len();
                for i in 0..n_in {
                    let eid = self.incoming[src][i] as usize;
                    let pre = self.synapses[eid].pre;
                    let w = self.synapses[eid].weight;
                    let new_w = (w + p.a_plus * self.pre_trace[pre])
                        .clamp(p.w_min, p.w_max);
                    self.synapses[eid].weight = new_w;
                }
            }
        }

        self.time += dt;
        fired
    }
}
