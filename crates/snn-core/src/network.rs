//! Network of LIF neurons connected by synapses.
//!
//! Per-step update order:
//!   1. Decay each neuron's synaptic current (exp(-dt/τ_syn))
//!   2. Decay STDP traces
//!   3. Inject external + synaptic currents, advance LIF dynamics
//!   4. For every spike this step, deliver `weight` to post-synapse
//!      and apply STDP weight updates
//!
//! Storage is flat (`Vec<LifNeuron>`, `Vec<Synapse>`) so the same
//! algorithm can later run on GPU with identical memory layout.

use crate::neuron::LifNeuron;
use crate::stdp::StdpParams;
use crate::synapse::Synapse;

pub struct Network {
    pub neurons: Vec<LifNeuron>,
    pub synapses: Vec<Synapse>,
    pub i_syn: Vec<f32>,
    pub pre_trace: Vec<f32>,
    pub post_trace: Vec<f32>,
    pub time: f32,
    pub dt: f32,
    pub stdp: Option<StdpParams>,
}

impl Network {
    pub fn new(dt: f32) -> Self {
        Self {
            neurons: Vec::new(),
            synapses: Vec::new(),
            i_syn: Vec::new(),
            pre_trace: Vec::new(),
            post_trace: Vec::new(),
            time: 0.0,
            dt,
            stdp: None,
        }
    }

    pub fn add_neuron(&mut self, n: LifNeuron) -> usize {
        let id = self.neurons.len();
        self.neurons.push(n);
        self.i_syn.push(0.0);
        self.pre_trace.push(0.0);
        self.post_trace.push(0.0);
        id
    }

    pub fn connect(&mut self, pre: usize, post: usize, weight: f32) -> usize {
        let id = self.synapses.len();
        self.synapses.push(Synapse::new(pre, post, weight));
        id
    }

    pub fn enable_stdp(&mut self, params: StdpParams) {
        self.stdp = Some(params);
    }

    /// Advance the network one timestep with optional external currents.
    /// `external` length must equal the number of neurons (or be empty,
    /// meaning zero external input). Returns the indices that fired.
    pub fn step(&mut self, external: &[f32]) -> Vec<usize> {
        let dt = self.dt;
        let t = self.time;

        // 1) Decay synaptic currents and STDP traces.
        for (idx, n) in self.neurons.iter().enumerate() {
            let decay_syn = (-dt / n.params.tau_m.max(1e-6)).exp();
            // Use the synapse's own tau_syn — but since it's per-synapse and
            // i_syn is per-post-neuron, we approximate with a fixed 5 ms here.
            // (Same value as Synapse::new default; kept simple intentionally.)
            let decay_psc = (-dt / 5.0_f32).exp();
            self.i_syn[idx] *= decay_psc;
            let _ = decay_syn;
        }
        if let Some(p) = self.stdp {
            let dp = (-dt / p.tau_plus).exp();
            let dm = (-dt / p.tau_minus).exp();
            for x in self.pre_trace.iter_mut() { *x *= dp; }
            for x in self.post_trace.iter_mut() { *x *= dm; }
        }

        // 2) Step every LIF, recording spikes.
        let mut fired: Vec<usize> = Vec::new();
        for (idx, n) in self.neurons.iter_mut().enumerate() {
            let ext = external.get(idx).copied().unwrap_or(0.0);
            if n.step(t, dt, ext + self.i_syn[idx]) {
                fired.push(idx);
            }
        }

        // 3) Deliver synaptic effects + STDP updates for spikes this step.
        let stdp = self.stdp;
        for &src in &fired {
            // Pre-trace bump on the firing neuron (it's "pre" for outgoing edges).
            if stdp.is_some() {
                self.pre_trace[src] += 1.0;
                self.post_trace[src] += 1.0;
            }
            for syn in self.synapses.iter_mut() {
                if syn.pre == src {
                    self.i_syn[syn.post] += syn.weight;
                    if let Some(p) = stdp {
                        // Pre-before-... → LTD contribution: w -= A_minus * post_trace[post]
                        syn.weight -= p.a_minus * self.post_trace[syn.post];
                        syn.weight = syn.weight.clamp(p.w_min, p.w_max);
                    }
                }
                if syn.post == src {
                    if let Some(p) = stdp {
                        // ...-before-post → LTP contribution: w += A_plus * pre_trace[pre]
                        syn.weight += p.a_plus * self.pre_trace[syn.pre];
                        syn.weight = syn.weight.clamp(p.w_min, p.w_max);
                    }
                }
            }
        }

        self.time += dt;
        fired
    }
}
