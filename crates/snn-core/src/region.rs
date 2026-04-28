//! A `Region` is a named, self-contained `Network`.
//!
//! Regions are the macro-level unit Javis composes the brain from. Each
//! region runs its own LIF dynamics, has its own E/I balance, and exposes
//! the underlying network for inspection. Long-range communication
//! between regions goes through `Brain` (see `brain.rs`).

use crate::network::Network;

pub struct Region {
    pub name: String,
    pub network: Network,
}

impl Region {
    pub fn new(name: impl Into<String>, dt: f32) -> Self {
        Self { name: name.into(), network: Network::new(dt) }
    }

    pub fn num_neurons(&self) -> usize {
        self.network.neurons.len()
    }
}
