//! Adapter from SDRs into a spiking `Network`.

use snn_core::Network;

/// Adds `drive_na` (nA) to the synaptic-current channel of every neuron
/// whose index appears in `sdr_indices`. The next call to `network.step`
/// will integrate this current under the normal LIF dynamics — strong
/// drives will fire those neurons, weaker drives will only depolarise.
///
/// Use `+=` semantics so multiple calls (e.g. multiple SDRs in the same
/// step) compose naturally. Indices outside the network are silently
/// skipped, which matches biology: a connection that does not exist
/// simply has no effect.
pub fn inject_sdr(network: &mut Network, sdr_indices: &[u32], drive_na: f32) {
    for &idx in sdr_indices {
        if let Some(slot) = network.i_syn.get_mut(idx as usize) {
            *slot += drive_na;
        }
    }
}
