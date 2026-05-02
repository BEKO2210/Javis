# Iter 59 — R2 capacity scaling for the vocab=64 floor

## Why iter-59

iter-58 closed the geometry-vs-architecture question with a
direction-of-change argument: doubling vocab (32 → 64) at the
otherwise-fixed iter-54 best config raised trained cross from
0.230 to 0.422 (Δ collapsed from −0.229 to −0.025, paired t(3)
fell to ≈ −2.58 / p ≈ 0.08). A pure geometry / encoder-collision
floor would predict trained_cross holds or drops with bigger
vocab; an architecture / per-cue-capacity floor predicts it
rises. Trained_cross rose by +0.192 — branch (B) primary.

The architecture-floor reading has a specific mechanistic form:
**block_size = R2-E / vocab** is the per-cue R2 budget under
the iter-54 decorrelated init. At vocab=32, block_size = 43
cells; at vocab=64 it halves to 21. The iter-58 verdict was
"the floor scales with cells-per-cue, not with vocab itself".

iter-59 is the corresponding *positive* control: hold vocab=64
fixed, scale R2_N up, and see whether trained cross falls back
toward the iter-54 vocab=32 best (~0.20-0.25). If yes,
"cells-per-cue is the binding constraint" is *empirically*
confirmed instead of just *direction-of-change* inferred. If
no, the architecture-floor model needs revision.

## What iter-58 proved

```
                     vocab=32          vocab=64
  block_size         43 cells/cue      21 cells/cue
  Untrained cross    0.459 ± 0.022     0.448 ± 0.012
  Trained cross      0.230 ± 0.020     0.422 ± 0.017
  Δ cross            −0.229            −0.025
  paired t(3)        ≈ −36.3           ≈ −2.58
  p                  ≪ 0.001           ≈ 0.08 (NS)
```

Direction-of-change disambiguates geometric vs architectural.
Trained_cross rose with vocab → architecture floor.

## Hypothesis

If the iter-58 architecture-floor reading is correct, then at
vocab=64 + R2_N = 4000 (block_size ≈ 42 cells/cue, matching
iter-54 vocab=32), trained cross should fall from 0.422 back
toward ~0.20-0.25. At R2_N = 8000 (block_size ≈ 84 cells/cue,
roughly double iter-54's budget), trained cross should fall
further or saturate near the floor.

If trained cross stays high under any R2_N, the architecture
limit is not "cells per cue" — it is something else, and the
iter-58 verdict needs to be re-examined.

## Implementation

No new metric, no new mechanism. R2_N becomes runtime-
configurable:

- `TeacherForcingConfig.r2_n: u32` (default 0 = use compile-
  time `R2_N` constant for backward compatibility).
- `effective_r2_n(cfg)` helper resolves 0 → R2_N constant.
- `build_memory_region` and `fresh_brain_with` take an explicit
  `r2_n: usize` parameter; the `drive_*` helpers replace
  hardcoded `R2_N` with `brain.regions[1].num_neurons()` so any
  R2 size flows through transparently.
- KWTA_K stays fixed at 60 — sparsity intentionally varies
  with R2_N (4.3 % at 2000 → 1.1 % at 8000). Part of the
  scaling test, not noise.

CLI:
- `--r2-n N` sets the effective R2 size for any single run.
- `--r2-capacity-sweep` runs `--jaccard-bench` once per size
  in `--r2-sizes 2000,4000,8000` (or whatever list is passed)
  and emits a single scaling table.

All re-exports unchanged. 10/10 eval lib tests stay green;
clippy `-D warnings` clean.

## Run command

```sh
cargo run --release -p eval --example reward_benchmark -- \
  --r2-capacity-sweep --r2-sizes 2000,4000,8000 \
  --seeds 42,7,13,99 --epochs 32 \
  --corpus-vocab 64 --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40
```

## Capacity sweep table

<!-- @SWEEP_TABLE@ -->

## Distribution / sanity notes

<!-- @SANITY@ -->

## Acceptance status

<!-- @ACCEPTANCE@ -->

## iter-60 branch decision

From Bekos's iter-59 spec, applied verbatim against the sweep
result:

| Outcome | iter-60 branch | This data |
| --- | --- | :-: |
| (A) R2_N=4000 or 8000 brings vocab64 trained_cross back to ~0.20-0.25 with strongly negative Δ cross | Architecture / capacity model confirmed; iter-60 = real scaling-law sweep over `R2-E / vocab` vs trained_cross | <!-- @A_ITER60@ --> |
| (B) R2_N improves only partially (e.g. trained 0.42 → 0.32, not back to 0.20) | Mixed limit; iter-60 = capacity *and* geometry cleanup combined | <!-- @B_ITER60@ --> |
| (C) R2_N barely helps | Not a simple capacity problem; iter-60 = learnable R1→R2 or association bridge | <!-- @C_ITER60@ --> |
| (D) Larger R2_N degrades or destabilises runs | Don't push further; iter-60 = topology / sparsity / connectivity scaling first | <!-- @D_ITER60@ --> |

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
trajectories often reveal a saturation ceiling the aggregate hides.
iter-56: aggregate monotonicity is not seed-level monotonicity.
iter-57: a 3-point sweep is the minimum for a non-monotonic axis.
iter-58: a saturation ceiling has a *direction*; pick a non-
training axis where competing models predict opposite signs.
**iter-59: <!-- @LESSON@ -->.**

## Files touched (single commit)

- `crates/eval/src/reward_bench.rs` — `r2_n` field on
  `TeacherForcingConfig`, `effective_r2_n` helper,
  `build_memory_region` / `fresh_brain_with` parameterised on
  `r2_n`, `drive_*` helpers use `brain.regions[1].num_neurons()`.
- `crates/eval/examples/reward_benchmark.rs` — `--r2-n`,
  `--r2-capacity-sweep`, `--r2-sizes` CLI flags + the sweep
  table renderer.
- `notes/59-r2-capacity-scaling.md` — this note.
- `CHANGELOG.md` — iter-59 section.
- `README.md` — iter-59 entry.
