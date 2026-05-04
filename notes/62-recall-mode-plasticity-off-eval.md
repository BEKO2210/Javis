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

<!-- @PER_SEED@ -->

## Aggregate comparison

<!-- @AGGREGATE@ -->

## Recall-mode stability result

<!-- @STABILITY@ -->

## Separation vs learning reading

<!-- @READING@ -->

## Acceptance per Bekos's iter-63 branching matrix

From Bekos's iter-62 spec, applied verbatim:

| Outcome | iter-63 branch | This data |
| --- | --- | :-: |
| (A) Recall-mode success: trained_same ≈ 1.000 + eval-drift ≈ 0 + cross-cue stays low across all 4 seeds | iter-63 = direct cue → target learning metric (separation + recall-stability now clean) | <!-- @A_ITER63@ --> |
| (B) Recall-mode stabilises same-cue but cross-cue rises | iter-63 = train-vs-recall dynamics analysis | <!-- @B_ITER63@ --> |
| (C) Recall-mode does not help: trained_same stays low even with plasticity disabled | iter-63 = DG → R2 weight / sparsity / recall dynamics (instability stored in trained weights, not eval drift) | <!-- @C_ITER63@ --> |
| (D) Recall-mode works, but learning still invisible (expected) | iter-63 = direct cue → target metric | <!-- @D_ITER63@ --> |

## Methodological lesson

<!-- @LESSON@ -->

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
