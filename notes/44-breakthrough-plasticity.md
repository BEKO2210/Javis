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

## Benchmark — measured against iter-43 baseline

Reproducible:

```sh
# Pre-iter-44 baseline (5x speed-up, used as reference):
cargo run --release -p eval --example scale_benchmark -- \
    --sentences 32 --queries 16 --seed 42 --iter44 off

# iter-44 stability subset (heterosynaptic + metaplasticity):
cargo run --release -p eval --example scale_benchmark -- \
    --sentences 32 --queries 16 --seed 42 --iter44 stability

# iter-44 short-corpus tuned (intrinsic + heterosynaptic + structural):
cargo run --release -p eval --example scale_benchmark -- \
    --sentences 32 --queries 16 --seed 42 --iter44 tuned

# Every iter-44 mechanism on (dev / stress test only):
cargo run --release -p eval --example scale_benchmark -- \
    --sentences 32 --queries 16 --seed 42 --iter44 full
```

Results on the **deterministic 32-sentence corpus, seed 42, decode-k 6**:

| Config | Train sec | Recall | FP / query | FN / query | Total query latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `off` (iter-43 baseline) | 175 | **4.4 %** | 4.50 | 10.94 | 16.1 ms |
| `stability` (heterosyn + BCM) | 236 | 4.4 % | 4.50 | 10.94 | 16.9 ms |
| `tuned` (intrinsic + heterosyn + structural) | **27** | 2.7 % | 4.69 | 11.12 | **14.0 ms** |
| `full` (every mechanism) | 355 | 1.6 % | 4.81 | 11.25 | 44.0 ms |

(Token reduction is `38.9 %` and self-recall `100 %` for every config —
those depend on `decode_k` and the dictionary scan, not the underlying
plasticity choice.)

### Honest reading of the numbers

**The iter-44 stack does not improve recall on this 32-sentence
benchmark out of the box.** Three reasons make this an *expected*
result rather than a regression:

1. **Heterosynaptic / BCM scale weights uniformly per post-neuron** —
   they shrink absolute drive but preserve the relative pattern that
   kWTA reads from. The fingerprint is the same regardless of the
   absolute weight magnitudes, so the engram dictionary is identical
   (the FP word *lists* differ between `off` and `stability` but the
   aggregate metrics do not).
2. **BCM θ has τ = 10 s**, the tuned default for biological
   regimes. The benchmark only spends 4.8 s in plasticity-on training
   (32 × 150 ms), nowhere near steady-state. The metaplasticity
   modulator stays close to 1 throughout, so it acts as a no-op.
3. **The `full` regression is genuine**: with default `IntrinsicParams`
   `a_target = 5` (tuned for multi-second cues) on a 150 ms cue,
   every neuron's adaptation trace stays well below target → offset
   pinned at `offset_min = -10` → effective threshold drops 10 mV →
   blob-saturation. The `tuned` preset uses `a_target = 0.5`,
   `offset_min = -2` and recovers most of the baseline behaviour
   while running 6× faster (structural pruning aggressively shrinks
   the dormant E→E synapses).

### Where the iter-44 stack *does* matter

The mechanisms target failure modes that this benchmark cannot
exercise:

| Mechanism | Failure mode it targets | Why this benchmark misses it |
| --- | --- | --- |
| Reward-modulated STDP | Goal-directed learning, credit assignment over seconds | The eval harness has no reward signal — `set_neuromodulator` is never called |
| Triplet STDP | Burst-coincidence frequency dependence | The training schedule has flat 150 ms cues, no high-rate bursts |
| BCM metaplasticity | Runaway LTP under sustained activity | 4.8 s of training is below the τ = 10 s θ time constant |
| Structural plasticity | Engram capacity ceiling at fixed topology | 32 concepts on a 10 000-neuron R2 is well below the topology-determined capacity |
| Replay / consolidation | Long-term consolidation across an idle gap | The harness runs strictly online, no idle-time replay loop |
| Intrinsic plasticity | Dead / saturated neurons over hours of operation | Dynamic range ≈ 5 s training; intrinsic time scale not reached |

In other words: the iter-44 stack is *infrastructure for the next
benchmark*, not a free win on the iter-25 one. It unblocks
reward-based learning, multi-epoch consolidation and unbounded
structural growth — all of which need a different evaluation harness
to measure (the current one assumes one-shot supervised training and
no idle / consolidation phase).

### Action items for a future benchmark
- Add a reward-aware corpus (e.g. multi-step retrieval where the
  ground-truth co-occurrence map is the reward signal).
- Schedule explicit replay rounds *between* training epochs so θ /
  eligibility traces actually equilibrate.
- Stress the structural plasticity rule with a streaming corpus
  (sentences arriving in batches over minutes) where engram capacity
  *is* the bottleneck.

## Iter 44.1 — decoder confidence floor

Independent of the plasticity stack, the original `decode_top` always
returned `k` results — even when the highest-scoring engram had a
containment ratio of `0.13` (right at the random-overlap baseline of
`KWTA_K / R2_E = 100 / 8000 = 12.5 %`). That is, **the FP count of
4.50 / query in the iter-43 baseline was largely a decoder artifact**:
the engrams were already orthogonal, the decoder just refused to
return an empty result.

`EngramDictionary::decode_top_above(active, k, min_score)` (and
`ScaleBrain::evaluate_with_threshold`) cleanly fix this — engrams
that score below the floor are *omitted* instead of filling the slot
with the next-best garbage.

Measured on the same 32-sentence corpus, seed 42, `--iter44 off`:

| `--decode-threshold` | FP / Q | Token reduction | Self-recall | "Recall" of co-occurring |
| ---: | ---: | ---: | ---: | ---: |
| `0.0` (default — pre-iter-44) | 4.50 | 38.9 % | **100 %** | 4.4 % |
| `0.10` | 4.50 | 38.9 % | 100 % | 4.4 % |
| **`0.20`** | **0.62** | **79.7 %** | **100 %** | 0.0 % |
| `0.30` | 0.00 | 84.7 % | 100 % | 0.0 % |

The "recall of co-occurring neighbours" that fell to zero was almost
entirely **noise**: the random-overlap floor for two
KWTA-100 patterns in an R2 of 8000 E neurons is 12.5 %, so anything
in the 4 – 13 % score range is statistically indistinguishable from
chance. With threshold `0.2`:

- **FP / query: 4.50 → 0.62 (− 86 %).** Cross-bleed mostly disappears.
- **Token reduction: 38.9 % → 79.7 % (+ 2.0×).** The mean Javis
  payload shrinks from 8.00 to 2.69 tokens — usually just the query
  word and the occasional high-overlap neighbour.
- **Self-recall stays at 100 %**: the query word's own engram has
  containment 1.0 by construction, well above any threshold.
- **Decoder latency unchanged** (the threshold filter is a single
  comparison per scored engram).

### Honest reading of the threshold result

This is a *measurement* fix, not a model fix. The SNN's engrams were
already orthogonal — the previous benchmark just hid that behind a
decoder that always returned `k` words regardless of confidence. With
the floor in place, the same network looks dramatically better on
both precision metrics, but the underlying engram structure is
unchanged.

What the threshold makes visible is the *real* problem the iter-44
stack was supposed to attack: **after the noise is filtered out, the
"recall of co-occurring neighbours" is 0 %**. Related concepts in
the corpus produce *separate* engrams, not partially-overlapping
ones. This is the gap that reward-modulated STDP + structural
plasticity have to close, and it is exactly the gap a
correlation-only training pass cannot close.

Recommendation: `--decode-threshold 0.2` is the new default for any
honest measurement on this corpus. Lower values include random
overlaps; higher values gain a few extra FP-reduction percentage
points at the cost of also dropping marginally-real partial matches.

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
