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

<!-- @PER_SEED@ -->

## Aggregate

<!-- @AGGREGATE@ -->

## Distribution

<!-- @DISTRIBUTION@ -->

## Separation vs learning vs stability

<!-- @READING@ -->

## Acceptance per Bekos's iter-62 branching matrix

From Bekos's iter-61 spec, applied verbatim:

| Outcome | iter-62 branch | This data |
| --- | --- | :-: |
| (A) DG robust: trained_cross and untrained_cross both ≤ 0.05 across all 4 seeds; trained_same ≥ 0.90 | iter-62 = direct cue→target learning metric, since cross-cue Jaccard is at the floor | <!-- @A_ITER62@ --> |
| (B) DG separates but stability erodes: cross-cue stays low but trained_same drops below 0.90 or eval-drift rises further | iter-62 = recall-mode / plasticity-off-during-eval / plasticity-rate decay / consolidation-freeze | <!-- @B_ITER62@ --> |
| (C) DG smoke does not replicate: cross-cue rises substantially at ep32 or on new seeds | iter-62 = DG parameter sweep, not new architecture | <!-- @C_ITER62@ --> |
| (D) DG separates only untrained, trained gets worse | iter-62 = tame plasticity, do not enlarge DG | <!-- @D_ITER62@ --> |

## Methodological lesson

<!-- @LESSON@ -->

## Files touched (single commit)

- `notes/61-dg-bridge-full-replication.md` — this note.
- `CHANGELOG.md` — iter-61 section.
- `README.md` — iter-61 entry.
