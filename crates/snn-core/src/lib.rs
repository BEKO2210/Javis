//! Spiking neural network core for Javis.
//!
//! Biological-style Leaky Integrate-and-Fire neurons, exponential synaptic
//! currents, and pair-based STDP plasticity. Fixed-step Euler integration.
//! All state is held in flat `Vec`s so a future GPU port (wgpu / candle)
//! can mirror the layout without refactoring the algorithms.
//!
//! # Quick start
//!
//! Build a tiny two-neuron network, wire one synapse, drive the
//! pre-neuron with a strong external current, and watch the
//! post-neuron eventually fire as the synaptic current accumulates:
//!
//! ```rust
//! use snn_core::{LifNeuron, LifParams, Network};
//!
//! let mut net = Network::new(0.1);              // 0.1 ms timestep
//! let pre  = net.add_neuron(LifNeuron::new(LifParams::default()));
//! let post = net.add_neuron(LifNeuron::new(LifParams::default()));
//! net.connect(pre, post, 30.0);                  // strong synapse
//!
//! // Drive `pre` directly with 3 nA — well above rheobase.
//! let mut external = vec![0.0_f32; 2];
//! external[pre] = 3.0;
//!
//! let mut post_fired = false;
//! for _ in 0..1000 {
//!     let fired = net.step(&external);
//!     if fired.contains(&post) {
//!         post_fired = true;
//!         break;
//!     }
//! }
//! assert!(post_fired, "post neuron should fire via the synapse");
//! ```
//!
//! Larger building blocks live one level up:
//!
//! - [`Brain`] — composes multiple [`Region`]s with delayed
//!   inter-region spike events.
//! - [`StdpParams`], [`IStdpParams`], [`HomeostasisParams`] — opt-in
//!   plasticity mechanisms; default is a passive integrate-and-fire
//!   network.

pub mod brain;
pub mod homeostasis;
pub mod istdp;
pub mod network;
pub mod neuron;
pub mod poisson;
pub mod region;
pub mod rng;
pub mod stdp;
pub mod synapse;

pub use brain::{Brain, BrainState, InterEdge, PendingEvent, PendingQueue};
pub use homeostasis::HomeostasisParams;
pub use istdp::IStdpParams;
pub use network::{Network, NetworkState};
pub use neuron::{LifNeuron, LifParams, NeuronKind};
pub use poisson::PoissonInput;
pub use region::Region;
pub use rng::Rng;
pub use stdp::StdpParams;
pub use synapse::{Synapse, SynapseKind};
