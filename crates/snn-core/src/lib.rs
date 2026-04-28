//! Spiking neural network core for Javis.
//!
//! Biological-style Leaky Integrate-and-Fire neurons, exponential synaptic
//! currents, and pair-based STDP plasticity. Fixed-step Euler integration.
//! All state is held in flat `Vec`s so a future GPU port (wgpu / candle)
//! can mirror the layout without refactoring the algorithms.

pub mod neuron;
pub mod synapse;
pub mod network;
pub mod stdp;
pub mod rng;
pub mod poisson;

pub use neuron::{LifNeuron, LifParams};
pub use synapse::Synapse;
pub use network::Network;
pub use stdp::StdpParams;
pub use rng::Rng;
pub use poisson::PoissonInput;
