//! Leaky Integrate-and-Fire neuron.
//!
//! Membrane dynamics:  τ_m · dV/dt = -(V - V_rest) + R_m · I(t)
//! Discretised with forward Euler at timestep `dt` (ms).

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
    pub v: f32,
    pub refractory_until: f32,
    pub last_spike: f32,
}

impl LifNeuron {
    pub fn new(params: LifParams) -> Self {
        Self {
            v: params.v_rest,
            refractory_until: f32::NEG_INFINITY,
            last_spike: f32::NEG_INFINITY,
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
