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

use crate::homeostasis::HomeostasisParams;
use crate::istdp::IStdpParams;
use crate::neuron::{LifNeuron, NeuronKind};
use crate::stdp::StdpParams;
use crate::synapse::Synapse;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Network {
    pub neurons: Vec<LifNeuron>,
    pub synapses: Vec<Synapse>,
    pub outgoing: Vec<Vec<u32>>,
    pub incoming: Vec<Vec<u32>>,
    /// Synaptic current channel — transient, rebuilt on first step
    /// after a snapshot load.
    #[serde(skip, default)]
    pub i_syn: Vec<f32>,
    /// STDP pre-trace per neuron — transient.
    #[serde(skip, default)]
    pub pre_trace: Vec<f32>,
    /// STDP post-trace per neuron — transient.
    #[serde(skip, default)]
    pub post_trace: Vec<f32>,
    /// Simulation clock — transient.
    #[serde(skip, default)]
    pub time: f32,
    pub dt: f32,
    pub stdp: Option<StdpParams>,
    pub istdp: Option<IStdpParams>,
    pub homeostasis: Option<HomeostasisParams>,
    /// Step counter — transient.
    #[serde(skip, default)]
    pub step_counter: u64,
    /// Cumulative count of synaptic deliveries since construction —
    /// transient.
    #[serde(skip, default)]
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
            istdp: None,
            homeostasis: None,
            step_counter: 0,
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

    /// Ensure the transient buffers (`i_syn`, `pre_trace`, `post_trace`)
    /// have the right length for the current neuron count. Called after
    /// loading a snapshot, where `#[serde(skip)]` left them empty.
    pub fn ensure_transient_state(&mut self) {
        let n = self.neurons.len();
        if self.i_syn.len() != n {
            self.i_syn = vec![0.0; n];
        }
        if self.pre_trace.len() != n {
            self.pre_trace = vec![0.0; n];
        }
        if self.post_trace.len() != n {
            self.post_trace = vec![0.0; n];
        }
    }

    pub fn enable_stdp(&mut self, params: StdpParams) {
        self.stdp = Some(params);
    }

    pub fn disable_stdp(&mut self) {
        self.stdp = None;
    }

    pub fn enable_homeostasis(&mut self, params: HomeostasisParams) {
        self.homeostasis = Some(params);
    }

    pub fn disable_homeostasis(&mut self) {
        self.homeostasis = None;
    }

    pub fn enable_istdp(&mut self, params: IStdpParams) {
        self.istdp = Some(params);
    }

    pub fn disable_istdp(&mut self) {
        self.istdp = None;
    }

    /// Reset transient state (membrane potentials, synaptic currents,
    /// STDP traces, homeostatic activity traces, refractory clocks,
    /// time, step counter, event counter). Synapse weights and network
    /// topology are preserved.
    pub fn reset_state(&mut self) {
        for (idx, n) in self.neurons.iter_mut().enumerate() {
            n.v = n.params.v_rest;
            n.refractory_until = f32::NEG_INFINITY;
            n.last_spike = f32::NEG_INFINITY;
            n.activity_trace = 0.0;
            self.i_syn[idx] = 0.0;
            self.pre_trace[idx] = 0.0;
            self.post_trace[idx] = 0.0;
        }
        self.time = 0.0;
        self.step_counter = 0;
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

        // 2) Decay plasticity traces. `pre_trace` is only used by E-side
        //    STDP. `post_trace` is shared between STDP (E-side) and
        //    iSTDP (I→E side) — both interpret it as "this post-neuron
        //    has fired recently". When both are active we decay with
        //    the STDP `tau_minus` for compatibility.
        if let Some(p) = self.stdp {
            let dp = (-dt / p.tau_plus).exp();
            let dm = (-dt / p.tau_minus).exp();
            for x in self.pre_trace.iter_mut() {
                *x *= dp;
            }
            for x in self.post_trace.iter_mut() {
                *x *= dm;
            }
        } else if let Some(ip) = self.istdp {
            // iSTDP only — decay just the post-trace using its own tau.
            let dm = (-dt / ip.tau_minus).exp();
            for x in self.post_trace.iter_mut() {
                *x *= dm;
            }
        }

        // 2b) Decay homeostatic activity traces (long time constant).
        if let Some(h) = self.homeostasis {
            let decay = (-dt / h.tau_homeo_ms).exp();
            for n in self.neurons.iter_mut() {
                n.activity_trace *= decay;
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

        // 4) Deliver synaptic effects + plasticity updates for spikes this step.
        let stdp = self.stdp;
        let istdp = self.istdp;
        let any_post_trace = stdp.is_some() || istdp.is_some();
        let homeo_active = self.homeostasis.is_some();
        for &src in &fired {
            if stdp.is_some() {
                self.pre_trace[src] += 1.0;
            }
            if any_post_trace {
                self.post_trace[src] += 1.0;
            }
            if homeo_active {
                self.neurons[src].activity_trace += 1.0;
            }
            let src_kind = self.neurons[src].kind;
            let sign: f32 = match src_kind {
                NeuronKind::Excitatory => 1.0,
                NeuronKind::Inhibitory => -1.0,
            };

            // Outgoing edges from `src`: deliver PSC, optional plasticity.
            //
            // - E-pre + STDP: classical LTD using post_trace[post].
            // - I-pre + iSTDP: anti-Hebbian update on I→E edges using
            //   `dw = a_plus - a_minus * post_trace[post]`. Pre-only
            //   firing (silent E) drives LTP; co-activity (E recently
            //   fired) drives LTD. Magnitudes stay non-negative.
            let n_out = self.outgoing[src].len();
            for i in 0..n_out {
                let eid = self.outgoing[src][i] as usize;
                let post = self.synapses[eid].post;
                let w = self.synapses[eid].weight;
                self.i_syn[post] += sign * w;
                self.synapse_events += 1;
                match src_kind {
                    NeuronKind::Excitatory => {
                        if let Some(p) = stdp {
                            let new_w = (w - p.a_minus * self.post_trace[post])
                                .clamp(p.w_min, p.w_max);
                            self.synapses[eid].weight = new_w;
                        }
                    }
                    NeuronKind::Inhibitory => {
                        if let Some(ip) = istdp {
                            if self.neurons[post].kind == NeuronKind::Excitatory {
                                let dw = ip.a_plus - ip.a_minus * self.post_trace[post];
                                let new_w = (w + dw).clamp(ip.w_min, ip.w_max);
                                self.synapses[eid].weight = new_w;
                            }
                        }
                    }
                }
            }

            // Incoming edges to `src`: classical STDP LTP (E-side).
            // iSTDP does not need an incoming-side update — the rule on
            // the I-pre-spike side already covers both LTP (silent E)
            // and LTD (recently-fired E) cases.
            if let Some(p) = stdp {
                let n_in = self.incoming[src].len();
                for i in 0..n_in {
                    let eid = self.incoming[src][i] as usize;
                    let pre = self.synapses[eid].pre;
                    if self.neurons[pre].kind != NeuronKind::Excitatory {
                        continue;
                    }
                    let w = self.synapses[eid].weight;
                    let new_w = (w + p.a_plus * self.pre_trace[pre])
                        .clamp(p.w_min, p.w_max);
                    self.synapses[eid].weight = new_w;
                }
            }
        }

        // 5) Periodic homeostatic synaptic scaling.
        self.step_counter = self.step_counter.wrapping_add(1);
        if let Some(h) = self.homeostasis {
            if h.eta_scale != 0.0
                && h.apply_every > 0
                && self.step_counter % h.apply_every as u64 == 0
            {
                self.apply_synaptic_scaling(&h);
            }
        }

        self.time += dt;
        fired
    }

    /// Multiplicative homeostatic scaling of every excitatory incoming
    /// synapse, per post-neuron. Pure scalar multiplication preserves
    /// the relative weight pattern shaped by STDP.
    ///
    /// `factor_i = 1 + eta * (A_target - A_trace_i)`, then
    /// `w_ij = clamp(w_ij * factor_i)` for every excitatory pre `j`.
    fn apply_synaptic_scaling(&mut self, h: &HomeostasisParams) {
        let (w_min, w_max) = match self.stdp {
            Some(s) => (s.w_min, s.w_max),
            None => (0.0, f32::MAX),
        };

        let n = self.neurons.len();
        for post in 0..n {
            let trace = self.neurons[post].activity_trace;
            // Guard against very hyperactive neurons producing a negative
            // `factor` — that would push w * factor below zero and the
            // clamp to [w_min, …] would zero out *all* of the post's
            // incoming weights uniformly, destroying their relative
            // pattern. Clamping the factor to be non-negative keeps the
            // scaling well-defined even in extreme regimes.
            let factor_raw = 1.0 + h.eta_scale * (h.a_target - trace);
            let factor = if h.scale_only_down {
                factor_raw.clamp(0.0, 1.0)
            } else {
                factor_raw.max(0.0)
            };
            // Skip if no-op — saves the inner loop entirely.
            if factor == 1.0 {
                continue;
            }
            let n_in = self.incoming[post].len();
            for i in 0..n_in {
                let eid = self.incoming[post][i] as usize;
                let pre = self.synapses[eid].pre;
                if self.neurons[pre].kind != NeuronKind::Excitatory {
                    continue;
                }
                let new_w = (self.synapses[eid].weight * factor).clamp(w_min, w_max);
                self.synapses[eid].weight = new_w;
            }
        }
    }
}
