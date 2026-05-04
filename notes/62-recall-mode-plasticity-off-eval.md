# Iter 62 — Recall-mode (plasticity-off-during-eval)

## Why iter-62

iter-61's full replication of the iter-60 DG smoke produced a
clean per-axis split:

- **Separation: robust.** Cross-cue 0.025–0.033 across all 4
  seeds, per-pair median 0.000.
- **Learning: invisible at the floor.** Δ cross trained −
  untrained = −0.003, paired t(3) ≈ −2.07, **p ≈ 0.13** (NS).
- **Stability: heterogeneous.** Per-seed trained_same: 0.961,
  0.875, 0.898, 0.930. **2 of 4 below the 0.90 threshold.**
- **Drift: high and seed-dependent.** Eval-drift L2 (R2→R2)
  per seed: +4.60, +1.65, **−0.89** (sign flip), +3.73 — all
  10×–100× the no-DG baseline.

The mechanistic reading: under DG, cue-driven R2 traffic is
~10–100× denser than no-DG. The same eval-time plasticity
rate eats more engram per trial during recall. The instability
is not in the trained weights; it is in the eval phase
modifying them.

iter-62 tests exactly that hypothesis with the simplest
possible intervention: **disable every plasticity rule between
training and the jaccard-matrix eval phase.** Training stays
unchanged. If the iter-61 erosion was caused by recall-time
drift, recall-mode eliminates it; if not, the erosion is
encoded in the trained weights themselves.

## What iter-61 proved

```text
                    iter-58 no-DG ep32   iter-60 smoke ep16    iter-61 full ep32
Untrained cross     0.448 ± 0.012        0.028                 0.029 ± 0.002
Trained cross       0.422 ± 0.017        0.026                 0.026 ± 0.001
Δ cross             −0.025 (sig)         −0.002                −0.003 (NS)
Trained same        1.000                0.922 ± 0.011         0.916 ± 0.037
Eval-drift L2       +0.04                +3.3 to +4.4          −0.9 to +4.6
```

## Hypothesis

If the iter-61 erosion is *eval-phase plasticity drift*,
recall-mode (plasticity off during eval) gives:

- trained_same → 1.000 across all 4 seeds (deterministic LIF
  + no plasticity ⇒ trial-2 = trial-3),
- eval-drift L2 → 0 (asserted bit-identical),
- trained_cross stays near the iter-61 floor (~0.025–0.030).

If the erosion is *encoded in the trained weights*,
trained_same stays heterogeneous (some seeds < 0.90) even
without eval plasticity — it would mean the trained recurrent
attractor is itself unstable under repeated cue presentation,
not that plasticity is eroding it.

## Implementation

Single field on `TeacherForcingConfig`:

```rust
pub recall_mode_eval: bool,  // default false
```

Plumbing in `crates/eval/src/reward_bench.rs`:

- `run_jaccard_arm`: when
  `cfg.teacher.recall_mode_eval && !cfg.teacher.no_plasticity`,
  call `disable_stdp / disable_istdp / disable_homeostasis /
  disable_intrinsic_plasticity / disable_reward_learning /
  disable_metaplasticity / disable_heterosynaptic /
  disable_structural` on R2 *between training and the eval
  phase*. Post-eval L2 norms must equal pre-eval L2 norms
  bit-for-bit (asserted, panics if any plasticity path
  escaped the gate).
- `run_jaccard_floor_diagnosis`: same protocol mirrored so
  the per-pair distribution + top-N + cue-frequency reports
  measure the recall-mode brain.

CLI:

```sh
--plasticity-off-during-eval     # iter-62 recall-mode
--recall-mode-eval               # alias
```

Build / clippy / 10 tests still green.

## Run command

```sh
# Bench (untrained vs trained Δ).
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-bench --seeds 42,7,13,99 --epochs 32 \
  --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval

# Floor diagnosis (per-pair distribution).
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-floor-diagnosis --seeds 42,7,13,99 --epochs 32 \
  --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval \
  --floor-threshold 0.1 --floor-top-n 10
```

## Per-seed comparison (iter-61 vs iter-62)

**Per-seed table (the iter-55/56 lesson held — per-seed view
is decisive).**

| Seed | iter-61 same | **iter-62 same** | iter-61 eval-drift L2 | **iter-62 eval-drift L2** | iter-61 cross | **iter-62 cross** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.961 | **1.000** ✓ | +4.60 | **0.000** ✓ | 0.028 | 0.027 |
|  7 | 0.875 | **1.000** ✓ | +1.65 | **0.000** ✓ | 0.026 | 0.029 |
| 13 | 0.898 | **1.000** ✓ | −0.89 | **0.000** ✓ | 0.025 | 0.030 |
| 99 | 0.930 | **1.000** ✓ | +3.73 | **0.000** ✓ | 0.026 | 0.027 |

**Stability is the headline:** every seed went from
heterogeneous (4/4 below 1.000, 2/4 below 0.90) to **exactly
1.000** under recall-mode. Eval-drift L2 went from a 0.9–4.6
range (with seed 13's sign flip) to **bit-identical pre/post
on all 4 seeds**.

Cross-cue stayed in the same ~0.025–0.033 band. The
geometric floor reproduces to within seed-level noise.

The L2 bit-identity assertion fired on every trained arm
(printed in the log as `[iter-62 trained recall-mode] seed=N
pre=[..., X, ...] post=[..., X, ...] (bit-identical ✓)`) —
no plasticity path escaped the gate.

## Aggregate comparison

| | iter-58 no-DG ep32 (4 seeds) | iter-60 DG smoke ep16 (2 seeds) | iter-61 DG full ep32 (4 seeds) | **iter-62 DG + recall-mode ep32 (4 seeds)** |
| --- | ---: | ---: | ---: | ---: |
| Untrained cross | 0.448 ± 0.012 | 0.028 | 0.029 ± 0.002 | **0.029 ± 0.002** |
| Trained cross | 0.422 ± 0.017 | 0.026 | 0.026 ± 0.001 | **0.028 ± 0.001** |
| Δ cross | −0.025 (sig) | −0.002 | −0.003 (NS) | **−0.001 (NS)** |
| Trained same | 1.000 | 0.922 ± 0.011 | 0.916 ± 0.037 | **1.000 ± 0.000** |
| Eval-drift L2 | +0.04 | +3.3 to +4.4 | −0.9 to +4.6 | **0 (bit-identical, all 4 seeds)** |
| Δ same | 0.000 | −0.078 | −0.084 | **+0.000** |
| Δ-of-Δ | −0.025 | −0.076 | −0.081 | **+0.001** |

iter-62 collapses two of iter-61's open numbers to zero
(eval-drift L2 → 0; Δ same → 0) while keeping cross-cue at
the same floor. Δ cross stays NS (−0.001), Δ-of-Δ goes
positive (+0.001) but the magnitude is at the noise floor.

## Floor distribution diagnosis

A standalone `--jaccard-floor-diagnosis --plasticity-off-during-
eval` run with the same seeds was launched in parallel with the
bench. The bench-side per-seed table above carries the full
recall-mode verdict; the floor diagnosis serves as an independent
confirmation that the L2 invariant and same-cue stability hold in
the second eval path (the per-pair distribution + top-N + cue-
frequency render). Expectation at write-time: iter-61's
distribution (median 0.000, p95 0.10, max 0.275) holds within
±0.005 — recall-mode shifts the absolute floor by +0.002 in
aggregate so the distribution-level shift should be similarly
small. **Result: confirmed, see "Addendum: floor diagnosis path
confirmation" below.**

## Recall-mode stability result

**Recall-mode result: clean.**

- Trained same-cue = 1.000 on **4 of 4** seeds (vs iter-61's
  1 of 4 above 0.961, 2 of 4 below 0.90).
- Eval-drift L2 (R2→R2) = 0 bit-identical on **4 of 4** seeds
  (vs iter-61's 0.9–4.6 spread, with sign-flip on seed 13).
- Cross-cue stays in the iter-60 / iter-61 floor band
  (0.027–0.030 trained; 0.028–0.033 untrained).

**The iter-61 stability erosion was caused entirely by
plasticity acting during recall.** The trained weights
themselves were stable; the iter-61 same-cue heterogeneity
(0.875 / 0.898 / 0.930 / 0.961) was a recall-time artefact,
not a property of the post-training weight state.

The L2 bit-identity assertion is a hard structural check: it
fires on every brain step that touches a synapse weight.
Bit-identity across all 4 seeds means **zero** plasticity
paths leak through (all `disable_*` calls are working —
including the metaplasticity / heterosynaptic / structural
ones I wired alongside the standard STDP / iSTDP /
homeostasis / intrinsic / reward set).

Sub-observation: the iter-62 trained same-cue spread is
**zero** (std = 0.000, all 4 seeds = 1.000), much tighter
than iter-61's 0.037. Recall-mode also kills the seed-level
heterogeneity. Per the iter-55 / iter-56 lesson, this is
not a coincidence — without eval plasticity drift, the
deterministic LIF + frozen weights gives an identical
response to identical input across trials, so per-seed
spread is bounded by the fixed-point dynamics, not by
seed-dependent plasticity trajectories.

## Separation vs learning reading

**Separation: still robust, slightly higher floor under
recall-mode.** Per-seed cross-cue 0.027 / 0.029 / 0.030 /
0.027. Aggregate 0.028 ± 0.001. iter-61 (eval plasticity ON)
had 0.026 ± 0.001 — recall-mode adds ~0.002 to the absolute
floor. The reading: when plasticity acted during eval, it
was *very slightly* lowering the cross-cue floor (by ~0.002)
on top of the geometric floor — but at the cost of
heterogeneous same-cue erosion. The trade was bad. Recall-
mode keeps the floor at the geometry's natural value
(median 0.000 per-pair, with a thin tail).

**Learning: still invisible.** Δ cross trained-untrained =
−0.001, paired t(3) ≈ −0.6, p ≈ 0.6 NS. Compared to iter-61's
−0.003 (also NS). Recall-mode does not change this number —
plasticity remains unable to register cue-specific
improvement against the geometric floor. This is the
expected branch (D) reading: Jaccard is at the floor and
no longer measures learning. iter-63 needs a direct
cue → target metric to register plasticity-driven
association.

**Stability: solved.** All 4 seeds at same-cue = 1.000.
Recall-mode is the right intervention; the iter-61 erosion
was a recall artefact, not a trained-weight property.

**Drift: by construction zero.** L2 bit-identity asserted on
every trained arm. The iter-52 weight-stability invariant
(carried through iter-58 for the no-plasticity arm only)
now applies to the trained-arm eval phase too whenever
recall-mode is on.

## Acceptance per Bekos's iter-63 branching matrix

From Bekos's iter-62 spec, applied verbatim:

| Outcome | iter-63 branch | This data |
| --- | --- | :-: |
| (A) Recall-mode success: trained_same ≈ 1.000 + eval-drift ≈ 0 + cross-cue stays low across all 4 seeds | iter-63 = direct cue → target learning metric (separation + recall-stability now clean) | **✓ PRIMARY** — trained_same = 1.000 on 4/4 seeds; eval-drift L2 = 0 bit-identical on 4/4 seeds; cross-cue 0.027–0.030 trained, 0.028–0.033 untrained, all ≤ 0.05 |
| (B) Recall-mode stabilises same-cue but cross-cue rises | iter-63 = train-vs-recall dynamics analysis | ❌ — cross-cue did not rise (iter-61 0.026 → iter-62 0.028, +0.002 within noise); separation reading unchanged |
| (C) Recall-mode does not help: trained_same stays low even with plasticity disabled | iter-63 = DG → R2 weight / sparsity / recall dynamics (instability stored in trained weights, not eval drift) | ❌ — recall-mode fully restored same-cue to 1.000 on 4/4 seeds; the iter-61 erosion was a recall-time artefact, not a trained-weight property |
| (D) Recall-mode works, but learning still invisible (expected) | iter-63 = direct cue → target metric | **✓ secondary** — Δ cross trained-untrained = −0.001 (NS); the Jaccard metric is at the geometric floor and no longer measures plasticity-driven cue-specific learning. iter-63 = direct cue → target metric. |

**iter-63 entry: branch (A) primary + branch (D) secondary.**
Recall-mode is the right intervention; same-cue and eval-
drift are both clean. The Jaccard cross-cue metric has now
done its job — it surfaced separation, then surfaced the
recall-stability question, and is now at the floor. iter-63
needs a different metric to measure cue-specific learning on
top of the DG geometry. Most candidates are already
implemented in `RewardEpochMetrics` from iter-46 / 52
(canonical-target top-k overlap, target rank, MRR, correct-
minus-incorrect target overlap, per-pair target activation)
and just need re-wiring on the DG path.

## Methodological lesson

iter-61 surfaced two confounded numbers: same-cue heterogeneous
(2 of 4 seeds < 0.90) and eval-drift L2 high (0.9–4.6). The
mechanistic hypothesis — "DG produces denser cue-driven R2
activity, so the same eval-time plasticity rate eats more
engram per trial" — was directly testable with one bit:
disable plasticity during eval. iter-62 set that bit. Same-
cue went to 1.000 across all seeds and eval-drift to 0
bit-identical, both confirming the hypothesis. **The right
intervention often has fewer parameters than the wrong one
— in iter-62 it was a single boolean.** Whenever a sweep
isolates a clean mechanism, look for the simplest one-bit
intervention to test it; treat new architecture as a last
resort.

The corollary is that the iter-58 / iter-59 / iter-60 / iter-
61 same-cue numbers — including the carefully reported
heterogeneity in iter-61 — were measuring *recall-time
plasticity dynamics* on top of the engram, not the engram
itself. Under iter-53's "plasticity ON during eval" protocol
(chosen so the trained brain's same-cue would vary, per
Bekos's iter-53 design intent), DG's denser R2 traffic
amplified that variance. iter-62 disambiguates the two — and
the answer is that the trained engram alone is fully stable
(same-cue = 1.000); the variance was the eval phase, not the
training.

## Headline

**Recall-mode restores same-cue stability; DG separation is
now stable under read-only recall.**

## Preview — iter-63 cue→target metric

The Jaccard cross-cue metric measures *separation*, not
*learning*. With DG (iter-60+), the floor is ~0.025 in both
trained and untrained arms; plasticity gains no measurable
ground against this floor (iter-61 Δ cross −0.003, p ≈ 0.13
NS). For iter-63 we need a metric that activates plasticity-
driven cue → target association directly:

- **canonical target top-k overlap** — does the trained R2
  response to the cue contain cells from the canonical
  target SDR? (re-introduces iter-46/52's
  `target_clamp_hit_rate` / `prediction_top3_before_teacher`
  on the DG-enabled brain.)
- **target rank** in the per-cue R2 firing distribution.
- **MRR** across the vocab.
- **correct-minus-incorrect target overlap** — overlap with
  cue's *own* canonical target minus overlap with all other
  targets, averaged.
- **per-pair target activation score**.

Most are already implemented in `RewardEpochMetrics` from
iter-46 and just need re-wiring on the DG path. Keep iter-62
narrow: only recall-mode here. iter-63 = re-introduce the
direct metric.

## Addendum: floor diagnosis path confirmation

The standalone floor-diagnosis sweep (started in parallel with
the bench, see *Floor distribution diagnosis* section above)
completed for all four seeds. Raw artefact preserved at
`/tmp/iter62-recall-floor.log`. Run command identical to the
floor-diagnosis line in the *Run command* section.

**Per-seed floor result (vocab=64 + DG + recall-mode, ep32):**

| Seed | same | cross | n_pairs |
| ---: | ---: | ---: | ---: |
| 42 | **1.000** | 0.027 ± 0.076 | 2016 |
|  7 | **1.000** | 0.029 ± 0.076 | 2016 |
| 13 | **1.000** | 0.030 ± 0.080 | 2016 |
| 99 | **1.000** | 0.027 ± 0.078 | 2016 |

`[iter-62 floor] seed=N recall-mode: every plasticity rule
disabled before eval` printed for each seed; the L2 bit-identity
assertion held on every floor-arm too — no plasticity path
escaped the gate in the second eval path either.

**Cross-seed averaged per-pair distribution:**

| min | p25 | median | p75 | p90 | p95 | max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.000 | 0.000 | 0.050 | 0.100 | 0.100 | 0.300 |

Compared to the chain so far: iter-58 max = 1.000 (geometric
collisions on short cues, vocab=64, no DG); iter-61 max = 0.275
(DG + plasticity ON during eval); iter-62 max = 0.300 (DG +
recall-mode). The recall-mode floor is ~iter-61 + 0.002 in
aggregate (0.026 ± 0.001 → 0.028 ± 0.001) and ~+0.025 at the
extreme tail — a thin worst-case widening, not a regime shift.
Median per-pair stays at 0.000.

**Top high-overlap pair:** `optional` / `scala` at 0.300
(cross-seed averaged Jaccard).

**Cue promiscuity:** `fortran` is the most promiscuous cue,
13 of 63 possible partners ≥ 0.10 — followed by `cobol` /
`haskell` (10 each) and `block` / `coroutine` / `cpp` /
`dynamic` / `elixir` / `include` / `julia` / `lambda` /
`tuple` / `typescript` / `zig` (9 each). The "long tail" of
high-overlap partners is concentrated on a small set of
short / morphologically-shared cues, consistent with
iter-58's encoder-collision sub-effect.

**Interpretation.**

- **Recall-mode is now confirmed in both the normal benchmark
  path *and* the floor diagnosis path.** Same-cue = 1.000 on
  4/4 seeds in the floor render (independent of the
  `run_jaccard_arm` path). The earlier in-flight observation
  (seed 42 floor = 0.961 from a stale binary, before the floor
  recall-mode plumbing was rebuilt) was *not* the final iter-
  62 behavior; it was a binary-staleness artefact. The
  rebuilt floor binary reproduces the bench-side recall-mode
  invariant exactly.
- **The remaining measurable imperfection is the cross-cue
  residual floor (≈ 0.028), not same-cue recall.** Recall-mode
  has fully solved the iter-61 stability question; the
  Jaccard cross-cue metric is now operating purely against
  the geometric / encoder-collision floor that iter-58 first
  surfaced.
- **iter-63's framing is unchanged:** the residual issue is
  cross-cue promiscuity / encoder-collision tail, which
  Jaccard cannot resolve — the path forward is the direct
  cue → target metric on the DG-enabled brain (canonical-
  target top-k, target rank, MRR, correct-minus-incorrect),
  not further work on Jaccard itself.

## Files touched

- `crates/eval/src/reward_bench.rs` — `recall_mode_eval` field
  + plasticity-disable + L2 invariant in `run_jaccard_arm` +
  `run_jaccard_floor_diagnosis`.
- `crates/eval/examples/reward_benchmark.rs` —
  `--plasticity-off-during-eval` / `--recall-mode-eval`
  CLI flags.
- `notes/62-recall-mode-plasticity-off-eval.md` — this note.
- `CHANGELOG.md` — iter-62 section.
- `README.md` — iter-62 entry.
