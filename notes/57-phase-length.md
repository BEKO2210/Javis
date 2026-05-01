# Iter 57 — Phase-length sweep on decorrelated + c500 architecture

## Why iter-57

iter-56 landed branch (α) of the iter-57 selector: clamp axis
is **magnitude-limited with diminishing returns**. Per-doubling
marginal Δ cross was −0.027 at 125 → 250 nA and −0.015 at 250 →
500 nA (ratio 0.55), giving a clamp-axis asymptote near 0.197.
Combined with iter-55's epoch-axis ceiling (~0.21), the
combined ceiling estimate is roughly trained cross **≈ 0.20** —
50 % below the untrained decorrelated baseline (0.459) but
flat against further intensity-axis tuning.

Per Bekos's iter-57 spec, the next un-swept axis is **phase
length** — the teacher / prediction / cue / tail / delay phase
durations in `TeacherForcingConfig` control *integration time*
under fixed clamp intensity. iter-57 sweeps `teacher_ms`
specifically, at the iter-56 winner clamp = 500 nA:

- **40 ms** (default; replication of iter-56 c500)
- **80 ms** (double)
- **120 ms** (triple)

Base config: `ep32 + --decorrelated-init + clamp 500` (the
combined iter-55 / iter-56 best); 4 seeds (42, 7, 13, 99).

## Implementation

No new code. Pure sweep-config exercise on the
`--teacher-ms` CLI flag that already exists on the example.

## Run

```sh
for tms in 40 80 120; do
  cargo run --release -p eval --example reward_benchmark -- \
    --jaccard-bench --seeds 42,7,13,99 --epochs 32 \
    --decorrelated-init --teacher-forcing \
    --target-clamp-strength 500 --teacher-ms $tms
done
```

<!-- @SWEEP_OUTPUT_40@ -->

<!-- @SWEEP_OUTPUT_80@ -->

<!-- @SWEEP_OUTPUT_120@ -->

## Phase-length curve

<!-- @PHASE_CURVE@ -->

## Per-seed view (iter-55/56 lesson applied — VERPFLICHTEND)

<!-- @PER_SEED@ -->

## Honest reading

<!-- @HONEST_READING@ -->

## Acceptance per Bekos's iter-58 branching matrix

From Bekos's iter-57 spec, applied verbatim:

| Sweep result | iter-58 branch | This data |
| --- | --- | :-: |
| (A) trained cross drops below 0.18 significantly in ≥ 1 config | Asymptote was schedule-specific, not architectural. iter-58 = continue phase-length tuning OR combine phase-length + clamp optimum | <!-- @A_ITER58@ --> |
| (B) trained cross reaches ≈ 0.20 in best config but no significant breakthrough | Architectural ceiling confirmed from a third axis. iter-58 = shift research question: "how to interpret the ceiling" — geometric limit diagnosis OR vocab scaling (32 → 64) | <!-- @B_ITER58@ --> |
| (C) trained cross stays > 0.23 in all configs | Phase-length is sub-effective lever. iter-58 = noise-injection (parallel candidate from iter-56) OR topology investigation | <!-- @C_ITER58@ --> |
| (D) Same-cue drops below 1.0 in ≥ 1 config (alongside any primary verdict) | Important secondary: longer teacher allows more eval plasticity, attractor-robustness becomes measurable. If Δ cross also good: iter-58 = noise-injection becomes test-ready | <!-- @D_ITER58@ --> |

## Methodological lesson

iter-50: save the simplest configuration as a regression guard.
iter-51: a guard is only a guard if its baseline excludes the null.
iter-52: an analytical null is not an empirical control.
iter-53: when the literal acceptance direction is bounded by
construction, derive it from the protocol's mathematical bounds.
iter-54: when the metric reports a "cleaner" number on a random
topology than on an architecturally cleaner one, the metric is
reading something else than what its name suggests.
iter-55: a learning curve is not a single number; per-seed
trajectories often reveal a saturation ceiling the aggregate
hides.
iter-56: aggregate monotonicity is not seed-level monotonicity;
half the seeds can break "higher = better".
**iter-57: <!-- @LESSON@ -->.**

## Files touched (single commit)

- `notes/57-phase-length.md` — this note.
- `CHANGELOG.md` — iter-57 section.
- `README.md` — iter-57 entry.
