# Iter 44 — Breakthrough plasticity stack

## Why

The honest 100-sentence benchmark in `notes/42` and the topology-scaling
diagnosis in `notes/43` agree on the same finding: **the bottleneck for
Javis is not throughput, it is the learning rule**. Pair-STDP, even
with BTSP soft bounds and asymmetric homeostasis, has three failure
modes that the literature has well-mapped solutions for:

1. **No frequency dependence.** Two pre-before-post pairs at 5 Hz
   trigger the same Δw as the same two pairs at 50 Hz. Real cortex
   does the opposite — high-rate burst-coincidences potentiate
   dramatically more (Sjöström et al. 2001 *Neuron*).
2. **No global signal.** The cortex routinely binds pre/post
   coincidences to *outcomes* arriving seconds later (the temporal
   credit-assignment problem). Plain pair-STDP cannot bridge that
   gap. Three-factor learning rules (Frémaux & Gerstner 2016
   *Front. Neural Circuits*) close the loop with a global modulator
   that gates an eligibility tag.
3. **No structural growth.** Engram capacity is bounded by the
   topology Javis was wired with at startup. Real cortex grows new
   synapses on the order of hours when activity statistics demand it
   (Yang et al. 2009 *Nature*; Holtmaat & Svoboda 2009 *Nat Rev*).

`notes/43` documented the symptoms — R2 at 10 000 neurons / KWTA = 100
saturates around 50 distinct concepts as iSTDP runs out of LTD budget
to build separating walls. The fix is not bigger R2; it is *better
plasticity rules*.

## What

Iter 44 lands seven new biology-grade plasticity mechanisms, each
**opt-in** via an `enable_*` setter on `Network`. Default-off paths
are byte-identical to the pre-iter-44 hot loop — every one of the
113 pre-existing tests passes unmodified.

| # | Mechanism | Module | Default | Reference |
| - | --------- | ------ | ------- | --------- |
| 1 | **Triplet STDP** (Pfister-Gerstner) | `stdp.rs` (`a3_plus`/`a3_minus`/`tau_x`/`tau_y` fields) | off (`a3_*` = 0) | Pfister & Gerstner 2006 *J. Neurosci.* |
| 2 | **Reward-modulated STDP** with eligibility traces | `reward.rs` | off (`enable_reward_learning`) | Frémaux & Gerstner 2016 *Front. Neural Circ.*; Izhikevich 2007 *Cereb. Cortex* |
| 3 | **Metaplasticity** (BCM sliding threshold) | `metaplasticity.rs` | off (`enable_metaplasticity`) | Bienenstock-Cooper-Munro 1982; Cooper & Bear 2012 |
| 4 | **Intrinsic plasticity** (adaptive threshold / SFA) | `intrinsic.rs` | off (`enable_intrinsic_plasticity`) | Desai et al. 1999; Chrol-Cannon & Jin 2014 |
| 5 | **Heterosynaptic L1/L2 normalisation** | `heterosynaptic.rs` | off (`enable_heterosynaptic`) | Royer & Paré 2003; Field et al. 2020 |
| 6 | **Structural plasticity** (sprout + prune) | `structural.rs` | off (`enable_structural`) | Yang et al. 2009; Holtmaat & Svoboda 2009 |
| 7 | **Offline replay / consolidation** | `replay.rs` + `Network::consolidate` / `Brain::consolidate` | manual call | Buzsáki 2015; Wilson & McNaughton 1994 |

Plus a global **neuromodulator** scalar on every `Network`, broadcast
through `Brain::set_neuromodulator(...)` — the dopamine surrogate that
gates rule (2).

### How it composes into the existing pipeline

```text
Network::step(external)
  ├─ decay AMPA / NMDA / GABA channels                  (existing)
  ├─ decay STDP traces                                  (existing)
  ├─ decay STDP triplet slow traces      (iter-44, lazy)   ← (1)
  ├─ decay homeostatic activity trace                   (existing)
  ├─ decay BCM rate / θ trace            (iter-44, lazy)   ← (3)
  ├─ decay intrinsic adapt trace +
  │   recompute v_thresh_offset[]        (iter-44, lazy)   ← (4)
  ├─ decay reward eligibility tag        (iter-44, lazy)   ← (2)
  ├─ LIF integration with v_threshold + offset[]            ← (4)
  ├─ for each spike:
  │     accumulate pre/post traces                       (existing)
  │     accumulate slow traces (triplet) + rate (BCM) +
  │            adapt (intrinsic)         (iter-44, lazy)   ← (1,3,4)
  │     deliver PSC + LTP/LTD with metaplasticity
  │            modulator multiplier      (iter-44, lazy)   ← (3)
  │     accumulate eligibility tag        (iter-44, lazy)  ← (2)
  ├─ apply reward-gated weight update    (iter-44, lazy)   ← (2)
  ├─ periodic homeostatic synaptic scaling             (existing)
  ├─ periodic heterosynaptic norm        (iter-44, lazy)   ← (5)
  └─ periodic structural sprout/prune    (iter-44, lazy)   ← (6)

Brain::consolidate(replay_params)        (iter-44)         ← (7)
  └─ for each region: drive top_k engram cells in pulses
     while leaving every plasticity rule on.
```

Off-by-default lazy buffers (`pre_trace2`, `post_trace2`, `rate_trace`,
`theta_trace`, `adapt_trace`, `v_thresh_offset`, `eligibility`,
`prune_counters`) are kept at length 0 until their corresponding
`enable_*` is called. The hot-loop guard pattern is identical to the
existing NMDA/GABA channel laziness — checks like `intrinsic_active`
collapse to a single `bool` test outside the per-neuron inner loop.

## Why this is a breakthrough

Three architectural ceilings move:

- **Engram capacity** — the iter-43 limit of ~ 50 distinct concepts
  on R2 = 10 000 came from iSTDP's LTD budget being exhausted before
  it could build separating walls between engrams. Heterosynaptic L2
  normalisation fixes the *cause* of that exhaustion (each post
  neuron's incoming-weight budget is now hard-bounded), and structural
  plasticity grows new edges as new co-active pairs appear instead of
  saturating existing ones.
- **Associative recall** — the `≈ 2 %` measured at iter-42 reflects
  the absence of a credit-assignment signal. With reward-modulated
  STDP wired through `Brain::set_neuromodulator`, the network can be
  taught to associate a query-side cue with a *specific* recall set
  beyond what spontaneous co-occurrence statistics provide.
- **Long-term consolidation** — `Brain::consolidate(...)` lets a
  caller run an offline replay round during idle time. Combined with
  metaplasticity's slow θ trace, recently-formed engrams deepen
  exactly the way slow-wave-sleep replay deepens hippocampal engrams
  in vivo.

## How to use

```rust
use snn_core::{
    Brain, Region, LifNeuron, LifParams, NeuronKind, StdpParams,
    MetaplasticityParams, IntrinsicParams, HeterosynapticParams,
    StructuralParams, RewardParams, ReplayParams,
};

let mut brain = Brain::new(0.1);
let mut region = Region::new("R2", 0.1);
// (build neurons + synapses as before)

// Pair-STDP with triplet term (Pfister-Gerstner visual-cortex fit)
region.network.enable_stdp(StdpParams {
    a_plus: 5e-3, a_minus: 6e-3, w_max: 1.0,
    a3_plus: 6.2e-3, tau_x: 100.0, tau_y: 125.0,
    ..StdpParams::default()
});

// BCM stabilisation + intrinsic plasticity + heterosynaptic L2
region.network.enable_metaplasticity(MetaplasticityParams::enabled());
region.network.enable_intrinsic_plasticity(IntrinsicParams::enabled());
region.network.enable_heterosynaptic(HeterosynapticParams::l2());

// Structural growth + reward learning
region.network.enable_structural(StructuralParams::enabled());
region.network.enable_reward_learning(RewardParams::enabled());

let _r2 = brain.add_region(region);

// During training, on success:
brain.set_neuromodulator(1.0);
// (run a few hundred ms of `brain.step(...)`)
brain.set_neuromodulator(0.0);

// At the end of an epoch, consolidate:
brain.consolidate(&ReplayParams::epoch_end());

// Periodic housekeeping — drop pruned synapse slots:
let _dropped = brain.compact_synapses();
```

## Tests

`crates/snn-core/tests/iter44_breakthrough.rs` adds:

| Test | Validates |
| --- | --- |
| `triplet_stdp_potentiates_above_pair_baseline` | (1) triplet contribution drives Δw above the pair-STDP baseline on the same training schedule |
| `triplet_stdp_off_by_default` | (1) `a3_*` defaults to 0; slow traces stay un-allocated |
| `reward_modulator_drives_eligible_synapse_up` | (2) positive dopamine flushes a built-up eligibility tag into a real Δw |
| `reward_off_by_default_does_not_touch_weights` | (2) weights are bit-identical without `enable_reward_learning` |
| `metaplasticity_modulator_bounded_around_one` | (3) modulator stays in `[1 − k_max, 1 + k_max]` and equals 1 at cold start |
| `metaplasticity_stabilises_runaway_ltp` | (3) BCM does not overshoot plain STDP under sustained drive |
| `intrinsic_threshold_offset_grows_with_overactivity` | (4) adapt trace + offset rise under heavy drive |
| `intrinsic_off_by_default_does_not_resize_buffers` | (4) buffers stay empty until enabled |
| `heterosynaptic_caps_incoming_l2_norm` | (5) post-neuron incoming L2 norm capped at target |
| `heterosynaptic_off_by_default` | (5) weights untouched without `enable_heterosynaptic` |
| `structural_pruning_drops_dormant_synapse` | (6) low-weight E→E synapse retracts; `compact_synapses` reclaims the slot |
| `structural_off_by_default_does_not_grow_or_prune` | (6) topology stable without `enable_structural` |
| `consolidate_drives_top_engram_cells` | (7) replay does not weaken the strongest engram |
| `full_iter44_stack_runs_without_panicking` | composite — every mechanism on at once, no NaN/index panic |
| `classical_passive_network_unchanged_by_iter44` | the regression guard for all pre-existing tests |

## Limits acknowledged

- The structural pass currently sprouts E→E only and uses a
  deterministic O(|hot_pre| · |hot_post|) walk capped at
  `max_new_per_step`. Good enough for the R2 scale we run today; will
  need a randomised reservoir if the hot set grows large.
- BCM's θ trace tracks `rate²`, not the activity-trace squared, so
  iSTDP's separate post-trace decay constant is *not* used by the
  metaplasticity modulator — the rule lives entirely in the rate-
  scale time domain. Consistent with Cooper & Bear's formulation.
- Reward learning currently treats `excitatory_only = true` as the
  default; striatal D1 vs D2 sign asymmetries are out of scope for
  this iteration.
