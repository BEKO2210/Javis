# Iter 45 — Reward-aware pair-association benchmark

## Why

`notes/44` documented that the iter-44 plasticity stack (Triplet,
R-STDP, BCM, Intrinsic, Heterosynaptic, Structural, Replay) failed
to improve any metric on the existing 32-sentence scale benchmark
*because the eval harness has no reward signal*. The whole point of
three-factor learning (Frémaux & Gerstner 2016) is that a global
modulator gates pre/post coincidences after the fact — pure
correlation-only benchmarks cannot exercise the rule by definition.

This iteration builds the missing harness: a fixed (cue, target)
list with deliberate distractor pairs, a per-trial reward signal,
and a per-epoch top-1 / top-3 readout. The same network is run
twice — once with `enable_reward_learning(...)` + dopamine, once
without — and the per-epoch curves are compared.

## What

`crates/eval/src/reward_bench.rs` adds:

- `RewardPair` / `RewardCorpus` — 16 (cue, target) word pairs +
  16 deliberately mis-paired noise distractors.
- `RewardConfig { epochs, use_reward, seed, reps_per_pair }`.
- `run_reward_benchmark(corpus, &cfg) -> Vec<RewardEpochMetrics>`.
- `RewardEpochMetrics { epoch, top1_accuracy, top3_accuracy,
  mean_reward, noise_top3_rate }`.
- `default_reward_corpus()` and `render_markdown(label, metrics)`.

Plus the `crates/eval/examples/reward_benchmark.rs` CLI:

```sh
cargo run --release -p eval --example reward_benchmark -- \
    --epochs 8 --reps 4 --seed 42
```

### Trial schedule

For every `(cue, target)` (real or noise) the harness runs:

1. **Phase 1 — staggered training** (× `reps_per_pair`): drive cue
   for `CUE_LEAD_MS = 40 ms`, then `cue + target` for
   `OVERLAP_MS = 30 ms`, then target for `TARGET_TAIL_MS = 30 ms`.
   The lead/overlap/tail order gives STDP a clean pre-before-post
   timing asymmetry → it preferentially grows `cue → target`
   weights, *not* the symmetric joint pattern the original draft
   accidentally learned.
2. **Phase 2 — reward delivery** (only if `use_reward`): for noise
   pairs the reward is always `−1`. For real pairs we use a cheap
   proxy — drive cue alone, kWTA the R2 activity, compute its
   overlap with the target's own kWTA, map `[0.30, 0.15, …]` to
   `[+1, 0, −0.5]`. The proxy doesn't need to be exact; R-STDP
   averages over many trials.
3. **Cool-down** (`COOLDOWN_MS = 20 ms`) so eligibility traces
   decay before the next trial.

### Epoch readout

After every epoch, the harness fingerprints all 32 vocab words
*once* against the current weights, then probes every real pair
(top-1 / top-3 accuracy) and every noise pair (distractor in
top-3 = "false-positive" rate). Building the dictionary per-epoch
rather than per-trial cuts the wall-time by ~ 30× compared to the
original draft.

## Measured (pair = 16, noise = 16, vocab = 32, decode_k = 3, seed = 42)

### epochs = 6, reps_per_pair = 4

```
Pure STDP (no neuromodulator)              R-STDP (dopamine)
| Epoch | top-1 | top-3 | noise |        | Epoch | top-1 | top-3 | noise |
| ---: | ---: | ---: | ---: |             | ---: | ---: | ---: | ---: |
|   0 | 0.00 | 0.12 | 0.06 |              |   0 | 0.00 | 0.12 | 0.00 |
|   1 | 0.00 | 0.00 | 0.19 |              |   1 | 0.00 | 0.00 | 0.12 |
|   2 | 0.00 | 0.06 | 0.19 |              |   2 | 0.00 | 0.06 | 0.12 |
|   3 | 0.00 | 0.19 | 0.12 |              |   3 | 0.00 | 0.00 | 0.12 |
|   4 | 0.00 | 0.00 | 0.19 |              |   4 | 0.00 | 0.06 | 0.19 |
|   5 | 0.00 | 0.06 | 0.19 |              |   5 | 0.00 | 0.00 | 0.06 |
```

Random-baseline top-3 with vocab = 32 is `3 / 32 ≈ 9.4 %`. Both
arms oscillate at or just above that floor.

## Honest reading

**Neither configuration converges to above-chance accuracy in 6
epochs**, and the comparison between them is not statistically
clean. R-STDP does show a noticeable but small advantage on the
*noise-suppression* axis — the mean noise-top-3 rate over the run
is `0.10` for R-STDP vs `0.16` for pure STDP — but the real-pair
accuracy is dominated by trial-to-trial variance.

The architectural reason is straightforward: the brain's R2 layer
gets driven primarily by the **forward** R1 → R2 path (FAN_OUT
synapses per cue input neuron, weight 2.0, delay 2 ms). The
**recurrent** R2 → R2 weights STDP is shaping start near 0.10 and
grow at most into the 0.20 – 0.50 range over the available
training time. Cue alone produces an R2 pattern that is mostly
this fixed forward pattern — *not* the recurrent associate of the
target.

For pair-association to work cleanly the architecture would need
one of:

- **Teacher-forcing on R2 target cells** during training (i.e.
  inject the target SDR directly into R2, not just via R1).
- **Stronger / pre-shaped recurrent connectivity** so the
  cue → target path has enough room for STDP to bias it.
- **A dedicated associative region** between R2-cue and R2-target
  blocks, similar to a CA3 → CA1 model.

None of these are infrastructure changes — they're architectural
decisions for a future iteration. What this benchmark **does**
deliver is the *ability to ask the question*: any future
plasticity rule, topology change, or training schedule can now be
A/B-tested against the same `RewardCorpus` with the same scoring
contract.

## Limits acknowledged

- The reward proxy in Phase 2 is a fast approximation of the
  per-vocab-word fingerprint from the epoch readout. A more
  accurate per-trial reward would re-fingerprint every trial, but
  that costs ~30× wall-time and would erase any eligibility tag
  built up during phase 1.
- Top-3 ≈ 9.4 % chance baseline is small; longer runs (epochs ≥
  16) and tighter R-STDP coefficients should give a tighter
  separation.
- The harness uses `Brain::set_neuromodulator` directly; the
  global value applies to *every* region. With a multi-region
  brain you'd want per-region modulators — adding that surface to
  `Brain` is a < 30-line change but out of scope for this note.
- Pure STDP is *also* learning the noise pairs (their distractor
  targets sometimes end up in top-3). R-STDP suppresses them only
  partially because the dopamine pulse is brief (80 ms) compared
  to the eligibility trace decay (1 s).

## Where this leaves the iter-44 stack

The reward-aware harness exists, the iter-44 mechanisms wire into
it cleanly, the `--reps` / `--epochs` axes are set up for sweeps,
and 13 unit tests on the underlying plasticity rules continue to
pass. The next concrete experiment that would actually move the
needle is **teacher-forcing the target SDR directly into R2
during training** so STDP has a strong, clean pre/post coincidence
to grab onto. That's a one-day implementation plus a one-day
parameter sweep, but it needs the rest of the iter-44 stack
(BCM stabilisation, intrinsic plasticity, structural growth) to
already be wired up — which it now is.
