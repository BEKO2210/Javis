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

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Brain {
    pub regions: Vec<Region>,
    pub inter_edges: Vec<InterEdge>,
    /// `outgoing[region][neuron]` → indices into `inter_edges`.
    pub outgoing: Vec<Vec<Vec<u32>>>,
    /// Scheduled inter-region spike events — transient, rebuilt as
    /// neurons fire after a snapshot load.
    #[serde(skip, default)]
    pub pending: Vec<PendingEvent>,
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
            pending: Vec::new(),
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

    pub fn connect(
        &mut self,
        src_region: usize,
        src_neuron: usize,
        dst_region: usize,
        dst_neuron: usize,
        weight: f32,
        delay_ms: f32,
    ) -> usize {
        self.ensure_outgoing_capacity(src_region);
        let id = self.inter_edges.len();
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

        // 1) Deliver all pending events whose arrival time has passed.
        //    This is O(P); tolerable for sparse long-range traffic. A
        //    BinaryHeap can replace this once event volume justifies it.
        let mut keep: Vec<PendingEvent> = Vec::with_capacity(self.pending.len());
        for ev in self.pending.drain(..) {
            if ev.arrive_at <= t {
                let net = &mut self.regions[ev.dst_region as usize].network;
                let idx = ev.dst_neuron as usize;
                if let Some(slot) = net.i_syn.get_mut(idx) {
                    *slot += ev.weight;
                }
                self.events_delivered += 1;
            } else {
                keep.push(ev);
            }
        }
        self.pending = keep;

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
}
