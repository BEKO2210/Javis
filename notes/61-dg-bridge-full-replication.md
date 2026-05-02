# Iter 61 — DG-bridge full replication at 4 seeds × 32 epochs

## Why iter-61

iter-60 was the first architectural number-move in the
iter-46…60 chain by an order of magnitude: adding a DG-like
pattern-separation bridge dropped the vocab=64 cross-cue floor
from 0.448 (untrained) and 0.422 (trained) to **0.028 / 0.026**
— a ~94 % collapse. But the smoke that produced this was 2
seeds × 16 epochs, and the trained-vs-untrained Δ cross was
only −0.002 (i.e. the geometry alone explains nearly the whole
floor; plasticity adds almost nothing on top of it). Same-cue
also dropped from 1.000 to 0.922, eval-drift L2 jumped ~100×.

iter-61 is **not** a new architecture. It is the standard
iter-55 / iter-56 lesson applied to iter-60: per-seed view at
full training before declaring the pivot solved.

## What iter-60 proved

```text
Setup: vocab=64, c500, ep16, seeds 42 + 7, DG size=4000, k=80
                       no-DG (iter-58)        + DG (iter-60)
Untrained cross        0.448 ± 0.012          0.028
Trained cross          0.422 ± 0.017          0.026
Δ cross                −0.025                 −0.002
Trained same           1.000                  0.922
Eval-drift L2 (R2→R2)  +0.04                  +3.3 to +4.4
```

Three claims to falsify or confirm at full seeds × full epochs:

1. **Separation:** is the cross-cue floor ≤ 0.05 across all 4
   seeds at ep32?
2. **Learning:** is trained_cross meaningfully different from
   untrained_cross, or has the metric saturated against a
   geometric floor?
3. **Stability:** does same-cue stay near 1.0, or does it
   erode further at ep32?

## Hypothesis

If iter-60's smoke was a real architectural effect, the floor
holds at ep32 across all 4 seeds: cross-cue stays near 0.03,
Δ cross stays near zero (because the floor is geometric, not
plastic), same-cue may erode further (more epochs of eval-time
plasticity drift). If iter-60 was a smoke artefact, ep32 + new
seeds will surface the failure mode.

## Setup

| | iter-60 smoke | iter-61 full |
| --- | --- | --- |
| seeds | 42, 7 | 42, 7, 13, 99 |
| epochs | 16 | 32 |
| vocab | 64 | 64 |
| clamp | 500 nA | 500 nA |
| teacher_ms | 40 | 40 |
| DG | enabled | enabled |
| DG size | 4000 | 4000 |
| DG k | 80 | 80 |
| DG fanout | 30 | 30 |
| DG weight | 1.0 | 1.0 |
| direct R1→R2 scale | 0.0 | 0.0 |
| R2_N | 2000 | 2000 |

## Run command

```sh
# Standard bench (untrained vs trained Δ).
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-bench --seeds 42,7,13,99 --epochs 32 \
  --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge

# Floor diagnosis (per-pair distribution, top-N pairs, cue frequency).
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-floor-diagnosis --seeds 42,7,13,99 --epochs 32 \
  --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge \
  --floor-threshold 0.1 --floor-top-n 10
```

## Per-seed results

**Per-seed table (the iter-55/56 lesson — aggregate hides
heterogeneity).**

| Seed | Untrained cross | Trained cross | Δ cross | Untrained same | Trained same | Eval-drift L2 (R2→R2) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.028 ± 0.090 | 0.028 ± 0.074 | +0.000 | 1.000 ± 0.000 | **0.961** ± 0.135 | +4.60 |
|  7 | 0.029 ± 0.088 | 0.026 ± 0.076 | −0.003 | 1.000 ± 0.000 | **0.875** ± 0.218 | +1.65 |
| 13 | 0.028 ± 0.086 | 0.025 ± 0.070 | −0.003 | 1.000 ± 0.000 | **0.898** ± 0.203 | **−0.89** (post < pre) |
| 99 | 0.033 ± 0.087 | 0.026 ± 0.074 | −0.007 | 1.000 ± 0.000 | **0.930** ± 0.175 | +3.73 |

**Same-cue per seed against the 0.90 stability threshold:**

- seed 42: **0.961** ✓ (above)
- seed 7: **0.875** ✗ (below)
- seed 13: **0.898** ✗ (below — borderline but under)
- seed 99: **0.930** ✓ (above)

**2 of 4 seeds erode below 0.90.** A simple aggregate (mean
0.916) hides this completely.

Eval-drift L2 (R2→R2) per seed: +4.60, +1.65, −0.89 (sign
change!), +3.73. Range 0.9 to 4.6. All ~10× to ~100× higher
than the iter-58 no-DG baseline (~0.04). Seed 13's *negative*
sign means weights decreased during eval — net depression, not
potentiation.

## Aggregate

| | mean | std (4 seeds) | SE | comment |
| --- | ---: | ---: | ---: | --- |
| Untrained cross | 0.029 | 0.002 | 0.001 | tightly clustered, all ≤ 0.05 |
| Trained cross | 0.026 | 0.001 | 0.0005 | even tighter than untrained |
| Δ cross (paired) | −0.003 | 0.003 | 0.0014 | t(3) ≈ −2.07, **p ≈ 0.13 (NS)** — Δ cross is not statistically distinguishable from zero at n=4 |
| Untrained same | 1.000 | 0.000 | 0.0 | state-reset assertion held on all 4 seeds |
| Trained same | 0.916 | 0.037 | 0.019 | aggregate above 0.90, but per-seed half are below |
| Δ same | −0.084 | 0.037 | 0.019 | meaningful drop |
| Δ-of-Δ | −0.081 | — | — | negative: same-cue erosion exceeds cross-cue gain |

The aggregate-level reading is "trained cross 0.026, untrained
0.029, ~94 % below the iter-58 no-DG baseline" — but the
trained-vs-untrained difference (−0.003) is **not statistically
significant** at n=4. The Jaccard floor has saturated against
the geometric ceiling DG provides; plasticity adds no
measurable cross-cue improvement on top of it.

### Comparison

| | iter-58 no-DG ep32 (4 seeds) | iter-60 DG smoke ep16 (2 seeds) | iter-61 DG full ep32 (4 seeds) |
| --- | ---: | ---: | ---: |
| Untrained cross | 0.448 ± 0.012 | 0.028 ± 0.000 | 0.029 ± 0.002 |
| Trained cross | 0.422 ± 0.017 | 0.026 ± 0.000 | 0.026 ± 0.001 |
| Δ cross | −0.025 (sig) | −0.002 | −0.003 (NS) |
| Trained same | 1.000 | 0.922 ± 0.011 | **0.916 ± 0.037** |
| Eval-drift L2 | +0.04 | +3.3 to +4.4 | −0.9 to +4.6 |

Cross-cue floor reproduces bit-close to the smoke (0.026 vs
0.026); same-cue mean is similar in aggregate (0.922 → 0.916)
but std is **3× larger** (0.011 → 0.037) — the per-seed spread
opens up meaningfully at full epochs.

## Distribution

**Cross-seed averaged per-pair Jaccard distribution:**

| | min | p25 | median | p75 | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| iter-58 vocab=64 (no DG) | 0.100 | 0.350 | 0.425 | 0.500 | 0.500 | 0.500 | **1.000** |
| **iter-61 vocab=64 + DG** | **0.000** | **0.000** | **0.000** | **0.050** | **0.050** | **0.100** | **0.275** |

**Median per-pair Jaccard is exactly 0.000** — half the cue
pairs share *zero* top-3 words. p90 = 0.05, p95 = 0.10. Max is
0.275 (vs iter-58's 1.000). The whole distribution shifted
left by ~0.40, and the high-overlap tail collapsed.

**Top-10 high-overlap pairs (cross-seed averaged):**

| Rank | cue_a | cue_b | mean Jaccard |
| ---: | :--- | :--- | ---: |
|  1 | swift | syntax | 0.275 |
|  2 | array | groovy | 0.225 |
|  3 | broadcast | channels | 0.225 |
|  4 | channels | clojure | 0.225 |
|  5 | closure | haskell | 0.225 |
|  6 | division | typescript | 0.225 |
|  7 | inference | isolate | 0.225 |
|  8 | jvm | kotlin | 0.225 |
|  9 | continuation | coroutine | 0.200 |
| 10 | fiber | fortran | 0.200 |

vs iter-58's top: actor/ada/array trio at **1.000** each. The
DG hash spreads even short common words across mostly-disjoint
DG cells, so the iter-58 perfect-collision trio no longer
dominates.

**Cue frequency in pairs ≥ 0.10** (out of 63 possible partners
per cue): top is **9 partners** (closure, erlang, optional,
zig). At iter-58 vocab=64 the top was **59 partners** for
clojure / dynamic / fiber / ownership — i.e. nearly every cue
overlapped meaningfully with nearly every other cue. iter-61
collapses this to ~14 % (9 / 63).

## Separation vs learning vs stability

**1. Separation — robust.** Cross-cue stays at 0.025-0.033
across all 4 seeds, all under the 0.05 acceptance threshold.
Per-pair median is 0.000, p95 is 0.10. Max collision (swift /
syntax) is 0.275 vs iter-58's perfect-overlap 1.000 trio. The
DG geometry collapse from iter-60 reproduces at full epochs;
this is the most robust single signal of the sweep.

**2. Learning — invisible at the floor.** Δ cross
(trained − untrained) is **−0.003 mean, t(3) ≈ −2.07, p ≈
0.13** — *not* statistically distinguishable from zero at n=4.
Per-seed Δ cross is +0.000, −0.003, −0.003, −0.007 — three of
four show small negative Δ but the magnitude is below the
seed-to-seed noise. **DG solves separation geometrically;
plasticity does not yet add measurable cue-specific
improvement in the Jaccard metric.**

**3. Stability — heterogeneous, half the seeds erode.** Per-
seed trained_same: 0.961, 0.875, 0.898, 0.930. **2 of 4 seeds
fall below the 0.90 acceptance threshold** (seed 7 at 0.875,
seed 13 at 0.898). The aggregate (0.916 ± 0.037) hides this:
the std is 3× larger than the iter-60 smoke (0.011), and the
two sub-0.90 seeds are not noise — they are seeds where DG-
driven dense R2 activity + STDP/iSTDP eats more of the engram
per trial during eval. Per Bekos's iter-55/56 lesson: report
the per-seed view, not the aggregate.

**4. Drift — high and seed-dependent, including a sign flip.**
Eval-drift L2 (R2→R2): +4.60, +1.65, **−0.89** (seed 13: net
weight *depression* during eval), +3.73. Range 0.9 to 4.6.
All seeds 10× to 100× higher than the iter-58 no-DG baseline
(~0.04). Seed 13's negative sign is qualitatively different
— iSTDP / homeostasis doing net depression where the other
seeds see net potentiation. *DG increases spike coincidences
and therefore weight changes during eval — speaking for a
later separation of learn-mode and recall-mode (eval-time
plasticity off, or its rate decayed).*

## Acceptance per Bekos's iter-62 branching matrix

From Bekos's iter-61 spec, applied verbatim:

| Outcome | iter-62 branch | This data |
| --- | --- | :-: |
| (A) DG robust: trained_cross and untrained_cross both ≤ 0.05 across all 4 seeds; trained_same ≥ 0.90 | iter-62 = direct cue→target learning metric, since cross-cue Jaccard is at the floor | **partial** — cross-cue is at the floor (✓ all 4 seeds ≤ 0.05); same-cue heterogeneous (2 of 4 below 0.90) |
| (B) DG separates but stability erodes: cross-cue stays low but trained_same drops below 0.90 or eval-drift rises further | iter-62 = recall-mode / plasticity-off-during-eval / plasticity-rate decay / consolidation-freeze | **✓ PRIMARY** — cross-cue robust at the floor (all seeds ≤ 0.05) but trained_same drops below 0.90 on seeds 7 (0.875) and 13 (0.898); eval-drift L2 stays 10×–100× higher than no-DG baseline, with seed-13 sign-flip indicating heterogeneous plasticity dynamics during eval |
| (C) DG smoke does not replicate: cross-cue rises substantially at ep32 or on new seeds | iter-62 = DG parameter sweep, not new architecture | ❌ — cross-cue (0.026) replicates the iter-60 smoke (0.026) bit-close at 4 seeds and ep32; aggregate floor reproduces; only same-cue spread opens up (std 0.011 → 0.037) |
| (D) DG separates only untrained, trained gets worse | iter-62 = tame plasticity, do not enlarge DG | ❌ — trained cross is *slightly lower* than untrained on 3 of 4 seeds (Δ = −0.003 mean); training does not make cross-cue worse, it just doesn't help much beyond the geometric floor |

**iter-62 entry: branch (B) PRIMARY — DG separates but
stability erodes on half the seeds.** Two parallel iter-62
candidates, both directly from Bekos's branch (B) prescription:

- **Path 1 (recommended primary): plasticity-off-during-eval
  / recall-mode.** The iter-58 / iter-59 / iter-60 / iter-61
  sweeps all run plasticity *during the 32-cue × 3-trial Jaccard
  matrix collection* (per the iter-53 spec where Bekos chose
  "im trained Run würde Plastizität zwischen Trials variieren,
  was *gewollt* ist"). Under DG the cue-driven R2 traffic is
  ~10–100× denser than no-DG, so the same eval-time plasticity
  rate eats more of the engram per trial. Branch (B) explicitly
  proposes recall-mode = plasticity-off-during-eval. Under that
  protocol the iter-53 same-cue=1.000 deterministic-LIF
  invariant returns automatically; the eval-drift L2 question
  vanishes.

- **Path 2 (parallel): plasticity-rate decay or DG → R2 weight
  decay during eval.** Less drastic than full plasticity-off:
  scale STDP `a_plus`, iSTDP `a_plus`, or the DG → R2 weight
  toward zero over the eval window. Lets the engram be slowly
  refined during eval without erosion. New CLI knob,
  ~30 min code.

The iter-58/59/60 cross-cue floor is now fully characterised:
DG drives it to a geometric near-zero, plasticity does not
register in the Jaccard against that floor. **Bekos's
follow-up suggestion — direct cue→target learning metric — is
the natural iter-62 sub-question** alongside Path 1: with
cross-cue at the floor, top-3 against the canonical target
(the iter-52 metric) is the metric that can register
plasticity-driven cue-specific learning. Re-introducing it on
the DG-enabled brain would tell us whether plasticity *does*
learn the cue→target association when the metric isn't
floor-saturated.

## Methodological lesson

A 2-seed × 16-epoch smoke gave aggregate same-cue 0.922 ±
0.011. The 4-seed × 32-epoch full run gives aggregate same-cue
0.916 ± **0.037** — same mean to 0.006, *3× the std*. The
mean was right; the heterogeneity was not. The per-seed view
is what splits "DG robust" (branch A) from "DG separates but
erodes" (branch B). The aggregate alone — 0.916 ± 0.037 — even
stays above the 0.90 threshold on its mean, but two of four
seeds individually drop below. iter-55 / iter-56 lesson, again,
in the architecture iteration: per-seed *stability spread* is
a different signal from per-seed *mean*, and a smoke is too
small to measure spread reliably. **Always replicate at full
seeds × full epochs before declaring an architectural pivot
solved.** This is now the eighth consecutive iteration where
the per-seed view has produced a different verdict than the
aggregate would have produced alone.

## Headline

**DG robustly solves cross-cue separation, but Jaccard no
longer measures learning, and same-cue erodes on half the
seeds under continued plasticity.**

## Files touched (single commit)

- `notes/61-dg-bridge-full-replication.md` — this note.
- `CHANGELOG.md` — iter-61 section.
- `README.md` — iter-61 entry.
