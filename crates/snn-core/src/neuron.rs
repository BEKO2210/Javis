//! Leaky Integrate-and-Fire neuron with Dale's-principle E/I labelling.
//!
//! Membrane dynamics:  τ_m · dV/dt = -(V - V_rest) + R_m · I(t)
//! Discretised with forward Euler at timestep `dt` (ms). The neuron's
//! `kind` (Excitatory or Inhibitory) determines the *sign* with which
//! its outgoing synapses act on post-synaptic membranes — synaptic
//! weights themselves stay non-negative.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuronKind {
    Excitatory,
    Inhibitory,
}

#[derive(Debug, Clone, Copy)]
pub struct LifParams {
    pub v_rest: f32,
    pub v_reset: f32,
    pub v_threshold: f32,
    pub tau_m: f32,
    pub r_m: f32,
    pub refractory: f32,
}

impl Default for LifParams {
    fn default() -> Self {
        Self {
            v_rest: -70.0,
            v_reset: -75.0,
            v_threshold: -55.0,
            tau_m: 20.0,
            r_m: 10.0,
            refractory: 2.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LifNeuron {
    pub params: LifParams,
    pub kind: NeuronKind,
    pub v: f32,
    pub refractory_until: f32,
    pub last_spike: f32,
    /// Exponentially-decaying spike counter used by homeostatic synaptic
    /// scaling. Each spike adds 1.0; each step decays by
    /// `exp(-dt / tau_homeo)`. The equilibrium value of the trace under
    /// a steady firing rate `r` (Hz) is approximately `r · tau_homeo / 1000`.
    pub activity_trace: f32,
}

impl LifNeuron {
    pub fn new(params: LifParams) -> Self {
        Self::with_kind(params, NeuronKind::Excitatory)
    }

    pub fn excitatory(params: LifParams) -> Self {
        Self::with_kind(params, NeuronKind::Excitatory)
    }

    pub fn inhibitory(params: LifParams) -> Self {
        Self::with_kind(params, NeuronKind::Inhibitory)
    }

    pub fn with_kind(params: LifParams, kind: NeuronKind) -> Self {
        Self {
            v: params.v_rest,
            refractory_until: f32::NEG_INFINITY,
            last_spike: f32::NEG_INFINITY,
            activity_trace: 0.0,
            kind,
            params,
        }
    }

    /// Advance one timestep with input current `i_input` (nA).
    /// Returns `true` if the neuron emitted a spike on this step.
    pub fn step(&mut self, t: f32, dt: f32, i_input: f32) -> bool {
        if t < self.refractory_until {
            self.v = self.params.v_reset;
            return false;
        }
        let p = &self.params;
        let dv = dt / p.tau_m * (-(self.v - p.v_rest) + p.r_m * i_input);
        self.v += dv;
        if self.v >= p.v_threshold {
            self.v = p.v_reset;
            self.refractory_until = t + p.refractory;
            self.last_spike = t;
            return true;
        }
        false
    }
}
