//! Macro-level orchestration: a `Brain` is a set of `Region`s connected
//! by long-range axons that carry asynchronous spike events between them.
//!
//! Routing follows the AER (Address-Event Representation) idea common in
//! neuromorphic hardware: when a designated source neuron in region A
//! fires, every outgoing inter-region edge from that neuron emits a
//! `PendingEvent` carrying its weight, target address, and arrival time
//! `t + delay_ms`. On each `Brain::step` we deliver every event whose
//! arrival time has passed by adding its weight to the target neuron's
//! `i_syn`. Inhibitory pre-neurons send negative-weight events.
//!
//! All regions share the same `dt`. Per-region externals are passed as
//! `&[Vec<f32>]`; an empty/short slot means zero external input for that
//! region.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::network::NetworkState;
use crate::neuron::NeuronKind;
use crate::region::Region;

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct InterEdge {
    pub src_region: u32,
    pub src_neuron: u32,
    pub dst_region: u32,
    pub dst_neuron: u32,
    pub weight: f32,
    pub delay_ms: f32,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct PendingEvent {
    pub arrive_at: f32,
    pub dst_region: u32,
    pub dst_neuron: u32,
    pub weight: f32,
}

/// Internal heap entry. Keeps the original `f32 arrive_at` so the
/// `<=` check matches the previous `Vec`-based code's float
/// comparison bit-for-bit (no rounding drift). Total order comes
/// from `f32::total_cmp`. A monotonic `sequence` breaks ties so older
/// events pop first when their arrival times are exactly equal,
/// preserving the original FIFO-on-tie semantics.
#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    sequence: u64,
    event: PendingEvent,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.event.arrive_at.total_cmp(&other.event.arrive_at) == Ordering::Equal
            && self.sequence == other.sequence
    }
}
impl Eq for HeapEntry {}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse on arrive_at (min-heap), then reverse on sequence.
        other
            .event
            .arrive_at
            .total_cmp(&self.event.arrive_at)
            .then(other.sequence.cmp(&self.sequence))
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Priority queue of inter-region spike events sorted by arrival time.
/// `push` is `O(log P)`, `drain_due` pops everything with `arrive_at <= t`
/// in `O(k log P)` where `k` is the number of due events. The previous
/// `Vec`-based implementation was `O(P)` per step regardless of due
/// volume — fine for sparse traffic, slow for dense brains.
#[derive(Debug, Clone, Default)]
pub struct PendingQueue {
    heap: BinaryHeap<HeapEntry>,
    next_sequence: u64,
}

impl PendingQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    pub fn clear(&mut self) {
        self.heap.clear();
        self.next_sequence = 0;
    }

    pub fn push(&mut self, ev: PendingEvent) {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);
        self.heap.push(HeapEntry {
            sequence,
            event: ev,
        });
    }

    /// Iterate over every event whose arrival time is at or before `t`
    /// (in milliseconds), in chronological order. Removes them from
    /// the queue as it goes. Uses the same `f32 <=` comparison as the
    /// previous `Vec`-based code so behaviour is bit-identical.
    pub fn drain_due(&mut self, t_ms: f32) -> DrainDue<'_> {
        DrainDue {
            heap: &mut self.heap,
            cutoff: t_ms,
        }
    }
}

/// Iterator returned by [`PendingQueue::drain_due`]. Pops only the
/// events that are due, leaving future events on the heap.
pub struct DrainDue<'a> {
    heap: &'a mut BinaryHeap<HeapEntry>,
    cutoff: f32,
}

impl<'a> Iterator for DrainDue<'a> {
    type Item = PendingEvent;
    fn next(&mut self) -> Option<Self::Item> {
        match self.heap.peek() {
            Some(top) if top.event.arrive_at <= self.cutoff => {
                Some(self.heap.pop().expect("just peeked").event)
            }
            _ => None,
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Brain {
    pub regions: Vec<Region>,
    pub inter_edges: Vec<InterEdge>,
    /// `outgoing[region][neuron]` → indices into `inter_edges`.
    pub outgoing: Vec<Vec<Vec<u32>>>,
    /// Scheduled inter-region spike events — transient, rebuilt as
    /// neurons fire after a snapshot load. Backed by a min-heap on
    /// arrival time so delivery is `O(log P)` per due event.
    #[serde(skip, default)]
    pub pending: PendingQueue,
    /// Simulation clock — transient.
    #[serde(skip, default)]
    pub time: f32,
    pub dt: f32,
    /// Cumulative event counter — transient.
    #[serde(skip, default)]
    pub events_delivered: u64,
}

impl Brain {
    pub fn new(dt: f32) -> Self {
        Self {
            regions: Vec::new(),
            inter_edges: Vec::new(),
            outgoing: Vec::new(),
            pending: PendingQueue::new(),
            time: 0.0,
            dt,
            events_delivered: 0,
        }
    }

    pub fn add_region(&mut self, region: Region) -> usize {
        let id = self.regions.len();
        let n = region.num_neurons();
        self.outgoing.push(vec![Vec::new(); n]);
        self.regions.push(region);
        id
    }

    /// Adjust the per-neuron outgoing buckets if a region grew after
    /// being added (regions are normally fully built before being
    /// handed to the brain, but this keeps the data structures honest).
    fn ensure_outgoing_capacity(&mut self, region: usize) {
        let needed = self.regions[region].num_neurons();
        if self.outgoing[region].len() < needed {
            self.outgoing[region].resize_with(needed, Vec::new);
        }
    }

    /// Refresh every region's transient buffers (`i_syn`, traces) to
    /// match the current neuron count. Called after loading a snapshot;
    /// safe to call any time.
    pub fn ensure_transient_state(&mut self) {
        for region in &mut self.regions {
            region.network.ensure_transient_state();
        }
        // Outgoing per-neuron adjacency is part of topology, but if a
        // region was loaded with a stale shape we resize defensively.
        for (ri, region) in self.regions.iter_mut().enumerate() {
            let needed = region.num_neurons();
            if self.outgoing.len() <= ri {
                self.outgoing.push(vec![Vec::new(); needed]);
            } else if self.outgoing[ri].len() < needed {
                self.outgoing[ri].resize_with(needed, Vec::new);
            }
        }
    }

    /// Reset every region's transient state and clear the global event
    /// queue + clock. Region topology and synaptic weights survive.
    pub fn reset_state(&mut self) {
        for region in &mut self.regions {
            region.network.reset_state();
        }
        self.pending.clear();
        self.time = 0.0;
        self.events_delivered = 0;
    }

    /// Turn STDP off in every region. Useful before recall measurements
    /// so that the act of measuring does not itself modify weights.
    pub fn disable_stdp_all(&mut self) {
        for region in &mut self.regions {
            region.network.disable_stdp();
        }
    }

    /// Turn homeostatic synaptic scaling off in every region. Symmetric
    /// to `disable_stdp_all` — both forms of plasticity have to be
    /// frozen for a clean recall measurement.
    pub fn disable_homeostasis_all(&mut self) {
        for region in &mut self.regions {
            region.network.disable_homeostasis();
        }
    }

    /// Turn inhibitory STDP off in every region.
    pub fn disable_istdp_all(&mut self) {
        for region in &mut self.regions {
            region.network.disable_istdp();
        }
    }

    /// Broadcast a global neuromodulator value (dopamine surrogate)
    /// to every region. Reward-modulated STDP in any region picks it
    /// up on its next step.
    pub fn set_neuromodulator(&mut self, value: f32) {
        for region in &mut self.regions {
            region.network.set_neuromodulator(value);
        }
    }

    /// Run an offline consolidation round in every region, in turn.
    /// Engram-relevant cells are driven with brief pulses while
    /// plasticity stays on, so weights deepen along the same paths
    /// the network would re-experience during recall.
    pub fn consolidate(&mut self, params: &crate::replay::ReplayParams) {
        for region in &mut self.regions {
            region.network.consolidate(params);
        }
    }

    /// Compact every region's synapse vector — drops every slot that
    /// structural plasticity has marked dead. Returns the total
    /// number of synapses that were physically removed.
    pub fn compact_synapses(&mut self) -> usize {
        self.regions
            .iter_mut()
            .map(|r| r.network.compact_synapses())
            .sum()
    }

    /// Wire an inter-region axon from `(src_region, src_neuron)` to
    /// `(dst_region, dst_neuron)`. Bounds-checks every index, the
    /// weight (finite) and the delay (positive, finite). Panics with
    /// a clear message on violation rather than corrupting silently.
    pub fn connect(
        &mut self,
        src_region: usize,
        src_neuron: usize,
        dst_region: usize,
        dst_neuron: usize,
        weight: f32,
        delay_ms: f32,
    ) -> usize {
        let n_regions = self.regions.len();
        assert!(
            src_region < n_regions,
            "Brain::connect: src_region {src_region} out of bounds ({n_regions} regions)",
        );
        assert!(
            dst_region < n_regions,
            "Brain::connect: dst_region {dst_region} out of bounds ({n_regions} regions)",
        );
        let n_src = self.regions[src_region].num_neurons();
        let n_dst = self.regions[dst_region].num_neurons();
        assert!(
            src_neuron < n_src,
            "Brain::connect: src_neuron {src_neuron} out of bounds in region {src_region} ({n_src} neurons)",
        );
        assert!(
            dst_neuron < n_dst,
            "Brain::connect: dst_neuron {dst_neuron} out of bounds in region {dst_region} ({n_dst} neurons)",
        );
        assert!(
            weight.is_finite(),
            "Brain::connect: weight must be finite, got {weight}",
        );
        assert!(
            delay_ms > 0.0 && delay_ms.is_finite(),
            "Brain::connect: delay_ms must be positive and finite, got {delay_ms}",
        );
        self.ensure_outgoing_capacity(src_region);
        let id = self.inter_edges.len();
        assert!(
            id < u32::MAX as usize,
            "Brain::connect: inter_edge count exceeds u32 capacity",
        );
        self.inter_edges.push(InterEdge {
            src_region: src_region as u32,
            src_neuron: src_neuron as u32,
            dst_region: dst_region as u32,
            dst_neuron: dst_neuron as u32,
            weight,
            delay_ms,
        });
        self.outgoing[src_region][src_neuron].push(id as u32);
        id
    }

    /// Advance every region one timestep. `externals[r]` (if present)
    /// is the external current vector for region `r`. Returns the
    /// per-region spike index lists for this step.
    pub fn step(&mut self, externals: &[Vec<f32>]) -> Vec<Vec<usize>> {
        let t = self.time;

        // 1) Deliver every event whose arrival time has passed. The
        //    min-heap pops them in chronological order in
        //    O(k log P), where k = due events this step.
        let due: Vec<PendingEvent> = self.pending.drain_due(t).collect();
        for ev in due {
            let net = &mut self.regions[ev.dst_region as usize].network;
            let idx = ev.dst_neuron as usize;
            if let Some(slot) = net.i_syn.get_mut(idx) {
                *slot += ev.weight;
            }
            self.events_delivered += 1;
        }

        // 2) Step each region's local dynamics with its own externals.
        let mut spikes: Vec<Vec<usize>> = Vec::with_capacity(self.regions.len());
        for (ri, region) in self.regions.iter_mut().enumerate() {
            let ext = externals.get(ri).map(|v| v.as_slice()).unwrap_or(&[]);
            let fired = region.network.step(ext);
            spikes.push(fired);
        }

        // 3) Schedule new inter-region events for everyone who fired.
        for (ri, fired_local) in spikes.iter().enumerate() {
            for &src in fired_local {
                let edges = &self.outgoing[ri][src];
                if edges.is_empty() {
                    continue;
                }
                let sign = match self.regions[ri].network.neurons[src].kind {
                    NeuronKind::Excitatory => 1.0_f32,
                    NeuronKind::Inhibitory => -1.0_f32,
                };
                for &eid in edges {
                    let edge = self.inter_edges[eid as usize];
                    self.pending.push(PendingEvent {
                        arrive_at: t + edge.delay_ms,
                        dst_region: edge.dst_region,
                        dst_neuron: edge.dst_neuron,
                        weight: sign * edge.weight,
                    });
                }
            }
        }

        self.time += self.dt;
        spikes
    }

    /// Build a fresh, at-rest [`BrainState`] sized for this brain.
    /// Each region gets its own [`NetworkState`] from
    /// [`crate::Network::fresh_state`]; the pending event queue is
    /// empty.
    pub fn fresh_state(&self) -> BrainState {
        BrainState {
            regions: self
                .regions
                .iter()
                .map(|r| r.network.fresh_state())
                .collect(),
            pending: PendingQueue::new(),
            time: 0.0,
            events_delivered: 0,
        }
    }

    /// Read-only step: same orchestration as [`Self::step`] but every
    /// piece of mutable state lives in `state`. The brain's regions,
    /// inter-region edges, and synapse weights are untouched, so the
    /// same [`Brain`] can be stepped from multiple concurrent
    /// contexts as long as each holds its own [`BrainState`].
    pub fn step_immutable(
        &self,
        state: &mut BrainState,
        externals: &[Vec<f32>],
    ) -> Vec<Vec<usize>> {
        let t = state.time;

        // 1) Deliver due inter-region events into the corresponding
        //    region state's AMPA channel.
        let due: Vec<PendingEvent> = state.pending.drain_due(t).collect();
        for ev in due {
            let region_state = &mut state.regions[ev.dst_region as usize];
            let idx = ev.dst_neuron as usize;
            if let Some(slot) = region_state.i_syn.get_mut(idx) {
                *slot += ev.weight;
            }
            state.events_delivered += 1;
        }

        // 2) Step each region's network read-only.
        let mut spikes: Vec<Vec<usize>> = Vec::with_capacity(self.regions.len());
        for (ri, region) in self.regions.iter().enumerate() {
            let ext = externals.get(ri).map(|v| v.as_slice()).unwrap_or(&[]);
            let fired = region.network.step_immutable(&mut state.regions[ri], ext);
            spikes.push(fired);
        }

        // 3) Schedule new inter-region events for spikes this step.
        for (ri, fired_local) in spikes.iter().enumerate() {
            for &src in fired_local {
                let edges = &self.outgoing[ri][src];
                if edges.is_empty() {
                    continue;
                }
                let sign = match self.regions[ri].network.neurons[src].kind {
                    NeuronKind::Excitatory => 1.0_f32,
                    NeuronKind::Inhibitory => -1.0_f32,
                };
                for &eid in edges {
                    let edge = self.inter_edges[eid as usize];
                    state.pending.push(PendingEvent {
                        arrive_at: t + edge.delay_ms,
                        dst_region: edge.dst_region,
                        dst_neuron: edge.dst_neuron,
                        weight: sign * edge.weight,
                    });
                }
            }
        }

        state.time += self.dt;
        spikes
    }
}

/// Per-recall transient state. Mirrors the mutable surface of [`Brain`]
/// (the per-region [`NetworkState`]s, the pending event queue, the
/// global clock and event counter) so a single immutable [`Brain`]
/// can drive multiple concurrent recalls.
#[derive(Debug, Clone)]
pub struct BrainState {
    pub regions: Vec<NetworkState>,
    pub pending: PendingQueue,
    pub time: f32,
    pub events_delivered: u64,
}
