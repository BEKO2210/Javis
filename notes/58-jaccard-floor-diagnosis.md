# Iter 58 — Jaccard floor geometry vs plasticity diagnosis

## Why iter-58

iter-55 / iter-56 / iter-57 swept three independent training
axes (epoch / clamp / phase-length) on the iter-54 decorrelated
+ teacher-forcing architecture. All three saturate near
trained cross **≈ 0.20** with diminishing returns on each axis:

- iter-55 epoch axis: per-doubling Δ cross −0.054 → −0.016
  (ratio 0.30, asymptote ≈ 0.21).
- iter-56 clamp axis: per-doubling Δ cross −0.027 → −0.015
  (ratio 0.55, asymptote ≈ 0.197).
- iter-57 phase-length axis: t40 best at 0.230, t80 catastrophic
  dip, t120 modest regression — non-monotonic, no break below
  0.23.

**The ≈ 0.20 cross-cue floor is robust across three orthogonal
training-axis sweeps.** It is no longer plausible that the
floor is "we just haven't trained enough" — it has held against
4× epochs (16 → 64), 4× clamp (125 → 500), 3× teacher_ms
(40 → 120), and the non-monotonic t80 catastrophe.

iter-58 is therefore *not* another optimisation iteration. It
is a **diagnosis**: what *is* the 0.20 floor?

Per Bekos's iter-58 spec, two diagnostics, both at the iter-57
best config (decorrelated + ep32 + clamp 500 + teacher_ms 40):

- **Path 1** — per-cue-pair Jaccard *distribution* analysis at
  vocab = 32. If the residual cross-cue is concentrated on a
  small fraction of cue pairs, those pairs flag an encoder /
  SDR / dictionary collision (geometric limit). If the
  residual is broadly uniform across all 496 pairs, the floor
  is a plasticity / architecture limit.
- **Path 2** — vocab = 64 stress test (same config). If trained
  cross scales with `1 / vocab` (uniform-noise model), the
  floor is vocab-specific (geometric). If trained cross stays
  near 0.20, the floor is architecture-specific.

## Implementation — single commit

New code in `crates/eval/src/reward_bench.rs`:

- `pub struct JaccardPairSample` — one entry per (cue_a, cue_b)
  with `i < j`, plus the post-burn-in trial-1 decoded top-3
  sets so the diagnosis can spot encoder collisions.
- `pub struct JaccardFloorReport` — per-(seed, trained-arm)
  per-pair list + standard aggregate.
- `evaluate_jaccard_matrix_with_pairs(brain, encoder, r2_e,
  dict, vocab) -> (JaccardMetrics, Vec<JaccardPairSample>)` —
  per-pair-emitting variant of the iter-53 evaluator.
- `pub fn run_jaccard_floor_diagnosis(corpus, cfg, seeds) ->
  Vec<JaccardFloorReport>` — runs the trained arm at the
  passed config across `seeds`, mirroring `run_jaccard_arm`'s
  brain construction + training but using the with-pairs
  evaluator.
- `pub fn render_jaccard_floor_diagnosis(reports, threshold,
  top_n) -> String` — Markdown report covering the three cuts
  Bekos's spec asks for: distribution stats / top-N high-
  overlap pairs / per-cue frequency in pairs above threshold.
- `pub fn default_corpus_v64()` — vocab = 64 corpus extending
  the iter-46…57 set with 16 more programming-language pairs
  (typescript / perl / php / lua / crystal / nim / dart /
  racket / elixir / groovy / julia / matlab / fortran / cobol
  / ada / prolog).

Two new CLI flags in
`crates/eval/examples/reward_benchmark.rs`:

- `--jaccard-floor-diagnosis` (mirror of `--jaccard-bench` but
  emits the per-pair distribution report).
- `--corpus-vocab 32 | 64` (default 32; selects the standard
  16-pair corpus or the iter-58 32-pair extension).
- `--floor-threshold` (default 0.5) and `--floor-top-n`
  (default 10) — controls the floor-diagnosis report
  thresholds.

All re-exports plumbed through `crates/eval/src/lib.rs`. 10/10
eval lib tests stay green; clippy `-D warnings` clean.

## Run

```sh
# Path 1 — vocab = 32 floor diagnosis
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-floor-diagnosis --seeds 42,7,13,99 --epochs 32 \
  --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 32 --floor-threshold 0.3 --floor-top-n 15

# Path 2 — vocab = 64 standard bench (trained vs untrained)
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-bench --seeds 42,7,13,99 --epochs 32 \
  --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64

# Path 2 — vocab = 64 floor diagnosis
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-floor-diagnosis --seeds 42,7,13,99 --epochs 32 \
  --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --floor-threshold 0.3 --floor-top-n 15
```

## Path 1 — vocab=32 distribution

<!-- @PATH1@ -->

## Path 2 — vocab=64 stress test

<!-- @PATH2@ -->

## Verdict per Bekos's iter-58 / iter-59 branching matrix

From Bekos's iter-58 spec, applied verbatim:

| Sweep result | iter-59 branch | This data |
| --- | --- | :-: |
| (A) Geometry floor: high cross-cue from few recurring cue pairs + vocab=64 lowers trained_cross substantially | iter-59 = encoder / dictionary / decode geometry fix | <!-- @A_VERDICT@ --> |
| (B) Plasticity / architecture floor: high cross-cue uniform across pairs + vocab=64 stays near 0.20 | iter-59 = real architecture question (bridge / learnable R1→R2 / contrastive objective) | <!-- @B_VERDICT@ --> |
| (C) Mixed: some collision pairs + global floor | iter-59 = geometry cleanup first, then architecture | <!-- @C_VERDICT@ --> |

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
**iter-58: <!-- @LESSON@ -->.**

## Files touched (single commit)

- `crates/eval/src/reward_bench.rs` — `JaccardPairSample` +
  `JaccardFloorReport` + `evaluate_jaccard_matrix_with_pairs` +
  `run_jaccard_floor_diagnosis` + `render_jaccard_floor_diagnosis` +
  `default_corpus_v64`.
- `crates/eval/src/lib.rs` — re-export new public surface.
- `crates/eval/examples/reward_benchmark.rs` —
  `--jaccard-floor-diagnosis` + `--corpus-vocab` +
  `--floor-threshold` + `--floor-top-n` CLI flags.
- `notes/58-jaccard-floor-diagnosis.md` — this note.
- `CHANGELOG.md` — iter-58 section.
- `README.md` — iter-58 entry.
