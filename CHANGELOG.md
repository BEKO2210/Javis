# Changelog

All notable changes to Javis. The version line follows the iteration
note that introduced the change — every iteration has a corresponding
`notes/NN-*.md` with the full reasoning, measurements, and references.

## Unreleased — Iteration 58 (Jaccard floor geometry vs plasticity diagnosis)

iter-55 / iter-56 / iter-57 swept three orthogonal training
axes (epoch / clamp / phase-length) on the iter-54 decorrelated
+ teacher-forcing architecture. All three saturate near
trained cross **≈ 0.20** with diminishing returns. The ≈ 0.20
cross-cue floor is no longer plausibly "we just haven't trained
enough" — it has held against 4× epochs, 4× clamp, 3×
teacher_ms, and the non-monotonic t80 catastrophe.

iter-58 is therefore *not* another optimisation iteration. It
is a **diagnosis**: what *is* the 0.20 floor? Geometric (encoder
/ SDR / dictionary collision artefact) or architectural
(plasticity / topology limit)?

### Added — single commit

- `pub struct JaccardPairSample` (cue_a, cue_b, jaccard, top_a,
  top_b) — one entry per (i < j) cue pair from the trained-arm
  trial-1 decoded top-3 sets.
- `pub struct JaccardFloorReport` (per-seed per-pair list +
  standard aggregate).
- `evaluate_jaccard_matrix_with_pairs` — per-pair-emitting
  variant of the iter-53 evaluator.
- `pub fn run_jaccard_floor_diagnosis(corpus, cfg, seeds)` —
  trained arm at the passed config × N seeds, mirroring
  `run_jaccard_arm`'s brain construction + training.
- `pub fn render_jaccard_floor_diagnosis(reports, threshold,
  top_n)` — Markdown report with the three cuts Bekos's spec
  asks for: distribution stats (min / p25 / median / p75 /
  p90 / p95 / max) + top-N high-overlap pairs + per-cue
  frequency in pairs ≥ threshold.
- `pub fn default_corpus_v64()` — vocab = 64 corpus extending
  the iter-46…57 set with 16 more programming-language pairs.
- CLI: `--jaccard-floor-diagnosis` + `--corpus-vocab 32 | 64`
  + `--floor-threshold` + `--floor-top-n`.

### Verified — Path 1 (vocab=32) and Path 2 (vocab=64) results

<!-- @CHANGELOG_RESULTS@ -->

### Honest reading

<!-- @CHANGELOG_HONEST_READING@ -->

### Methodological lesson

<!-- @CHANGELOG_LESSON@ -->

All eval lib tests still green (10/10); clippy `-D warnings`
clean.

## Unreleased — Iteration 57 (phase-length sweep on decorrelated + c500)

iter-56 landed branch (α) of the iter-57 selector — clamp axis
is magnitude-limited with diminishing returns (asymptote
estimate ≈ 0.197, combined-axes ceiling ~0.20). Per Bekos's
iter-57 spec, the next un-swept axis is **phase length** —
specifically `teacher_ms`, which controls integration time
under fixed clamp intensity. iter-57 sweeps three points at
clamp = 500 nA: 40 ms (default; iter-56 c500 replication),
80 ms (double), 120 ms (triple), 4 seeds × 32 epochs.

### Run

```sh
for tms in 40 80 120; do
  cargo run --release -p eval --example reward_benchmark -- \
    --jaccard-bench --seeds 42,7,13,99 --epochs 32 \
    --decorrelated-init --teacher-forcing \
    --target-clamp-strength 500 --teacher-ms $tms
done
```

### Verified — phase-length curve

Aggregate (n = 4 seeds, untrained baseline = 0.459 ± 0.022 in
all three runs; clamp = 500 nA throughout):

| teacher_ms | Trained cross | std | Δ cross | Δ-of-Δ | paired t(3) |
| ---: | ---: | ---: | ---: | ---: | ---: |
|  40 | 0.230 | ±0.020 | −0.229 | +0.229 | ≈ −36.3 |
|  80 | **0.408** | ±0.052 | **−0.051** | +0.051 | ≈ −3.01 (p ≈ 0.06) |
| 120 | 0.248 | ±0.051 | −0.211 | +0.211 | ≈ −10.05 |

t40 is bit-exact replication of iter-56 c500 (per-seed:
42=0.242, 7=0.250, 13=0.208, 99=0.220 — identical).

Per-seed trained cross trajectory:

| Seed | t40 | t80 | t120 | 40 → 80 | 80 → 120 | 40 → 120 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.242 | 0.434 | 0.317 | **+0.192** (worse) | −0.117 | +0.075 (worse) |
|  7 | 0.250 | 0.467 | 0.249 | **+0.217** (worse) | −0.218 | −0.001 (tied)  |
| 13 | 0.208 | 0.354 | 0.233 | **+0.146** (worse) | −0.121 | +0.025 (worse) |
| 99 | 0.220 | 0.375 | **0.194** | **+0.155** (worse) | −0.181 | **−0.026** (better) |

**Phase-length is a non-monotonic axis with a catastrophic
dip at t80.** Every seed sees t80 as substantially worse than
both t40 and t120; t40 vs t120 is a per-seed coin flip (2 seeds
prefer t40, 1 tied, 1 prefers t120 — seed 99 at t120 = 0.194
is the global best per-seed value across the entire iter-53 …
iter-57 chain). Lead-in / clamp ratios under the existing
`lead_in = (teacher_ms/4).clamp(4, 12)` formula:

| teacher_ms | lead_in | clamp_ms | lead:clamp |
| ---: | ---: | ---: | --- |
|  40 | 10 | 30 | 1 : 3   (uncapped)              |
|  80 | 12 | 68 | 1 : 5.7 (lead-in capped at 12)  |
| 120 | 12 | 108 | 1 : 9   (lead-in capped at 12)  |

t40 is the only config where lead-in is uncapped — gives STDP
the cue → target timing asymmetry. At t80 / t120 the lead-in
is the same (12 ms) but the clamp window stretches; t80 lands
in a "long enough to push iSTDP/homeostasis past stable, not
long enough to recover via consolidation" regime; t120's
longer consolidation (= teacher_ms) apparently lets the system
re-settle to roughly the t40 level.

State-reset assertion: PASSED on every untrained arm (12/12).
Decorrelated invariant: PASSED on every brain construction
(24/24).

### Honest reading

Three layered observations:

1. **Phase-length is non-monotonic with a catastrophic dip at
   t80.** Doubling teacher_ms (40 → 80) collapses Δ cross from
   −0.229 to −0.051 (78 % of the signal lost). Tripling
   (40 → 120) recovers most of t40 (Δ cross −0.211, only
   +0.018 worse than t40 in aggregate). t80 is uniformly bad
   on every seed.
2. **The dip mechanism is plausibly the lead-in / clamp ratio
   cap.** The lead-in formula caps at 12 ms; at teacher_ms ≥
   48 ms the lead-in stops scaling while the clamp window
   keeps growing. t80's 1:5.7 lead:clamp ratio appears to
   land in a "iSTDP/homeostasis pushed past stable but not
   long enough to consolidate back" regime; t120's 1:9 ratio
   plus longer consolidation phase recovers.
3. **Same-cue stays at exactly 1.000 in 12/12 trained arms;
   eval-drift L2 *decreases* at higher teacher_ms** (t40
   ~0.029 → t80 ~0.0022 → t120 ~0.0023). Branch (D) firmly
   REJECTED — phase length does not unlock attractor-
   plasticity at eval. Post-training R2 → R2 L2 norm scales
   with teacher_ms (seed 99: 339.78 → 481.92 → 564.85), but
   the buildup at t80 is in the *wrong place* (degrades
   cross-cue) while at t120 it's apparently in a more useful
   place.

Per Bekos's pre-fixed iter-58 branching matrix:
  - branch (A) trained cross < 0.18 anywhere: single-seed only (seed 99 t120 = 0.194)
  - branch (B) trained cross ≈ 0.20 in best, no breakthrough: ✓ secondary
  - branch (C) trained cross > 0.23 in all configs: ✓ PRIMARY
  - branch (D) same-cue drops below 1.0: ❌

iter-58 entry: **shift the research question.** All three
training-axes (epoch / clamp / phase-length) saturate or non-
monotonically dip near trained cross 0.20. iter-58 should
investigate what the ceiling *means*, not push it lower.

Recommended Path 1: **geometric vs plastic limit diagnosis.**
Compute per-cue-pair cross-cue Jaccard and inspect the
distribution. If concentrated on a small fraction of pairs
(encoder produces near-identical SDRs), ceiling is a vocab
artefact. If uniform, it's a plasticity-dynamics floor.
~5 min code, no new sweep.

Parallel Path 2: **vocab-scaling stress test** (vocab 32 → 64
at iter-54 best config) to test whether 0.20 is vocab-
specific. ~30 min.

Noise-injection / cross-topology stays valid as iter-59
candidate.

### Methodological lesson

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
**iter-57: a 3-point sweep is the minimum for a non-monotonic
axis. iter-57 swept teacher_ms at 40 / 80 / 120 specifically
because Bekos's spec required it; a more "efficient" 2-point
sweep (40 + 120 only) would have reported "phase-length is
roughly neutral, slight regression at 120" and never seen the
catastrophic dip at 80. Whenever an axis has a plausible
biological non-linearity (here: the STDP lead-in / clamp
ratio cap interacting with iSTDP recovery), the sweep needs
≥ 3 points or the axis's shape is unobservable.**

All eval lib tests still green; clippy `-D warnings` clean
(no code changes since iter-54).

## Unreleased — Iteration 56 (clamp-strength sweep on decorrelated + ep32)

iter-55 landed branch (ii) — Saturation — of the iter-56
selector. Per Bekos's iter-56 spec, the next un-swept axis with
high a priori sensitivity is **target-clamp-strength**: it
controls how hard the teacher signal overrides recurrent
dynamics during the teacher window, which is the layer where
cue-specific weight changes get written. iter-56 sweeps three
points around the iter-46/53/54/55 default of 250 nA: 125 nA
(half), 250 nA (default + iter-55 ep32 replication), and 500 nA
(double).

### Run — three configs × 4 seeds, ep = 32

```sh
for clamp in 125 250 500; do
  cargo run --release -p eval --example reward_benchmark -- \
    --jaccard-bench --seeds 42,7,13,99 --epochs 32 \
    --decorrelated-init --teacher-forcing \
    --target-clamp-strength $clamp
done
```

### Verified — clamp-strength curve

Aggregate (n = 4 seeds, untrained baseline = 0.459 ± 0.022 in
all three runs):

| Clamp (nA) | Trained cross | std | Δ cross | Δ-of-Δ | paired t(3) | per-doubling Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 125 | 0.272 | ±0.036 | −0.187 | +0.187 | ≈ −17.6 | (baseline)        |
| 250 | 0.245 | ±0.041 | −0.214 | +0.214 | ≈ −20.4 | −0.027 (125 → 250) |
| 500 | **0.230** | **±0.020** | **−0.229** | **+0.229** | ≈ **−36.3** | −0.015 (250 → 500) |

c250 is bit-exact replication of iter-55 ep32 (per-seed:
42=0.277, 7=0.281, 13=0.225, 99=0.196 — identical).

Per-seed trained cross trajectory:

| Seed | c125 | c250 | c500 | 125 → 250 | 250 → 500 | Trajectory |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 0.262 | 0.277 | 0.242 | **+0.015** | −0.035 | NON-MONOTONE  |
|  7 | 0.315 | 0.281 | 0.250 | −0.034 | −0.031 | monotone ↓     |
| 13 | 0.283 | 0.225 | 0.208 | −0.058 | −0.017 | monotone ↓     |
| 99 | 0.229 | 0.196 | 0.220 | −0.033 | **+0.024** | NON-MONOTONE   |

Aggregate is monotone in clamp strength, but **2 of 4 seeds
are non-monotone**. Per-doubling marginal gain: −0.027 →
−0.015 (ratio 0.55). Asymptote estimate ≈ 0.197.

c500 has 5× tighter seed-level std (0.020) vs c125 / c250
(0.036 / 0.041) — higher clamp not only moves the mean but
flattens the seed-level distribution.

State-reset assertion: PASSED on every untrained arm (12/12).
Decorrelated invariant: PASSED on every brain construction
(24/24).

### Honest reading

Three layered observations:

1. **Aggregate Δ cross is monotone in clamp strength, with
   diminishing returns at the same shape as the epoch axis.**
   Per-doubling Δ ratio = 0.55 (vs iter-55 epoch axis ratio
   0.30 — clamp axis "still has more room"). Combined ceiling
   estimate from both axes: trained cross ≈ 0.20, vs the
   untrained decorrelated baseline of 0.459 (specificity has
   dropped 50 %).
2. **Half the seeds are non-monotone in clamp strength.**
   Seeds 7 and 13: clean monotone decrease. Seeds 42 and 99:
   non-monotone — seed 42's c125 better than c250 then c500
   best; seed 99's c250 best with c500 *regressing*. The
   seed × clamp interaction is real; "always use higher
   clamp" is wrong for some seeds. iter-55 lesson echoed.
3. **c500 dramatically tightens the seed-level std (0.020
   vs 0.036/0.041).** Higher clamp doesn't just move the
   mean — it flattens the seed-level distribution. Even when
   c500 is *worse* than c250 on seed 99 (0.220 > 0.196), the
   spread across seeds is smaller. For deployments that
   value reliability, c500 is the right choice even though
   seed 99 alone prefers c250.

**Branch (δ) is REJECTED**: same-cue stays at 1.000 for every
seed × clamp combination (12/12 trained arms). Eval-drift L2
remains tiny (0.026–0.045 across the seeds where it was
logged). Higher clamp during *training* does not trigger
meaningful eval-time plasticity drift — the decorrelated
wiring still starves eval-phase plasticity, regardless of
training intensity.

Per Bekos's pre-fixed iter-57 branching matrix, this lands in
**branch (α) primary — clamp axis is magnitude-limited but
with diminishing returns**.

iter-57 entry: **Path B — Achse C (phase-length tuning)** as
the recommended critical path. Sweep `teacher_ms` (40 default
→ 80, 120) at fixed c500 to isolate whether longer integration
under c500 intensity helps beyond clamp magnitude alone. ~30-
min wallclock. Path A (clamp 500/1000/2000) is a safe but
diminishing extension of the current axis; Path B opens a new
axis with potentially un-diminished sensitivity. Noise-
injection / cross-topology stays parallel as iter-58
candidate.

### Methodological lesson

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
**iter-56: aggregate monotonicity is not seed-level
monotonicity. iter-56's Δ cross was monotone in clamp strength
at the aggregate level (−0.187 → −0.214 → −0.229), which the
literature would read as "higher clamp = better, period". Per-
seed, half the seeds are non-monotone: seed 99 has its global
best at c250 with c500 *worse*, seed 42 has c125 better than
c250 then c500 best. Aggregate monotonicity hides a real
seed × clamp interaction. The aggregate-only reading would
have set "always use c500" as a deployment recommendation;
the per-seed view shows that for half the seeds c500 is not
the global best.**

All eval lib tests still green; clippy `-D warnings` clean
(no code changes since iter-54).

## Unreleased — Iteration 55 (epoch sweep on decorrelated + plasticity)

iter-54 landed branch M1 of the iter-55 selector
(Δ-of-Δ = +0.160, ACCEPTANCE PASSED). Per Bekos's iter-55 spec,
M1 mandates **keep decorrelation + plasticity combined, sweep
training schedule** to characterise the learning curve. iter-55
is a pure sweep run: no code changes, three configs (16, 32,
64 epochs) × 4 seeds, all other parameters identical to iter-54.

### Run

Three configs, no new code:

```sh
# 16 epochs (replication of iter-54 — sanity check)
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-bench --seeds 42,7,13,99 --epochs 16 \
  --decorrelated-init --teacher-forcing

# 32 epochs
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-bench --seeds 42,7,13,99 --epochs 32 \
  --decorrelated-init --teacher-forcing

# 64 epochs
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-bench --seeds 42,7,13,99 --epochs 64 \
  --decorrelated-init --teacher-forcing
```

### Verified — learning curve

Aggregate (n = 4 seeds, untrained baseline = 0.459 ± 0.022 in
all three runs):

| Epochs | Trained same | Trained cross | Δ cross | Δ-of-Δ | paired t(3) | per-doubling Δ cross |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1.000 ± 0.000 | 0.299 ± 0.038 | −0.160 | +0.160 | ≈ −16.0 | (baseline)         |
| 32 | 1.000 ± 0.000 | 0.245 ± 0.041 | −0.214 | +0.214 | ≈ −20.4 | −0.054 (16 → 32)   |
| 64 | 0.996 ± 0.008 | 0.229 ± 0.058 | −0.230 | +0.226 | ≈ −11.6 | −0.016 (32 → 64)   |

Per-seed trained cross trajectory:

| Seed | ep16 | ep32 | ep64 | 16 → 32 | 32 → 64 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.329 | 0.277 | 0.246 | −0.052 | −0.031 |
|  7 | 0.326 | 0.281 | 0.281 | −0.045 | **±0.000** (flat)        |
| 13 | 0.295 | 0.225 | 0.245 | −0.070 | **+0.020** (regression!) |
| 99 | 0.247 | 0.196 | 0.146 | −0.051 | −0.050                   |

Per-doubling marginal gain ratio = 0.30 ⇒ asymptote estimate
trained cross ≈ 0.22 (geometric series). 16-epoch result is a
**bit-exact replication of iter-54** (same seeds, same code).

State-reset assertion: PASSED on every untrained arm (12/12).
Decorrelated invariant: PASSED on every brain construction (24/24).

### Honest reading

Three layered observations:

1. **Δ cross is monotone in the aggregate, with diminishing
   returns.** −0.054 per epoch-doubling between 16 → 32, then
   −0.016 between 32 → 64. The ratio of those (≈ 0.30)
   suggests an asymptote near trained cross ≈ 0.21 — roughly
   half the untrained baseline (0.459).
2. **The aggregate hides per-seed instability.** Seeds 42 and
   99 keep improving at every doubling; seed 7 plateaus at
   ep32 and stays flat at ep64; seed 13 *worsens* between
   ep32 and ep64 (0.225 → 0.245, single-seed catastrophic-
   interference signature). Saturation is the dominant
   pattern; small-scale interference is the secondary one.
3. **Same-cue stays at exactly 1.000 in 11/12 trained arms.**
   The single exception is seed 99 at 64 epochs (same =
   0.984, eval-drift L2 = 0.0015). Seed 99 is also the seed
   with the strongest specificity gain (0.146 trained cross
   at ep64) — when the engram becomes "alive enough" to
   perturb the eval response, it also becomes the most cue-
   specific. Constructive plasticity surviving past the
   saturation point on a single seed.

Per Bekos's iter-55 spec branching matrix, this lands in
**branch (ii) Saturation as primary**, with **branch (iv) as
secondary** (eval-phase plasticity essentially still under
decorrelated wiring). Branches (i) is partial (Δ cross
monotone but same-cue stays at 1.0); (iii) is single-seed-only.

iter-56 entry: **Achse B Clamp-Strength-Sweep** as the
critical path, with noise-injection eval / cross-topology as
a parallel sub-question for iter-57.

### Methodological lesson

iter-50: save the simplest configuration as a regression guard.
iter-51: a guard is only a guard if its baseline excludes the null.
iter-52: an analytical null is not an empirical control.
iter-53: when the literal acceptance direction is bounded by
construction, derive it from the protocol's mathematical bounds.
iter-54: when the metric reports a "cleaner" number on a random
topology than on an architecturally cleaner one, the metric is
reading something else than what its name suggests.
**iter-55: a learning curve is not a single number — per-
doubling marginal gain + per-seed regression cases together
identify saturation more reliably than the aggregate Δ alone.
The aggregate Δ-of-Δ improved monotonically across all three
configs (16: +0.160, 32: +0.214, 64: +0.226), which would
suggest "keep training". The per-seed view shows seed 13
regressed at ep64 and seed 7 plateaued at ep32, putting the
saturation ceiling where the aggregate alone would have hidden
it.**

All eval lib tests still green; clippy `-D warnings` clean
(no code changes since iter-54).

## Unreleased — Iteration 54 (hard-decorrelated R1 → R2 init)

iter-53 produced **Δ-of-Δ = −0.121, FAILED**: 16 epochs of
teacher-forcing on the random-FAN_OUT topology drifted weights
plenty (eval-phase L2 +25 to +29) but produced *zero* cross-cue
specificity gain (trained 0.058 ≈ untrained 0.058). The
diagnosis was that forward-drive bias from a uniform random
projection swamps any cue-specific routing plasticity might
build. Bekos's iter-54 spec attacks the bottleneck at the
architecture layer.

### Added — single commit

- `wire_forward_decorrelated(brain, encoder, vocab, seed,
  inter_weight) -> Vec<Vec<usize>>` — partition R2-E into
  `vocab.len()` disjoint blocks; for each R1 cell that appears
  in *exactly one* cue SDR, fan out `FAN_OUT` times into that
  cue's block. Shared R1 cells (multi-cue membership) are
  dropped from connectivity entirely — the only way to
  preserve pairwise R2-reach disjointness given a non-disjoint
  encoder.
- `assert_decorrelated_disjoint(brain, encoder, vocab)` —
  end-to-end mechanical invariant: for every cue pair the set
  of R2 cells reachable from cue *i*'s R1 SDR via any R1 → R2
  edge must be disjoint from cue *j*'s. Iterates
  `brain.outgoing` + `brain.inter_edges` directly (no shortcut
  to the block allocation). Called at run-start in
  `run_jaccard_arm` whenever `decorrelated_init = true`.
- `TeacherForcingConfig.decorrelated_init: bool` (default
  `false`).
- `--decorrelated-init` CLI flag in
  `crates/eval/examples/reward_benchmark.rs`.
- Unit test `decorrelated_init_is_pairwise_disjoint` —
  builds wiring against the real default corpus + encoder,
  asserts both block-level disjointness AND end-to-end
  reachability disjointness.

### Topology numbers (iter-46/53 defaults)

vocab = 32, R2-E = 1400, block_size = 43 cells per cue. The
encoder produces ~17 unique R1 cells per word (out of ENC_K =
20, with ~3 shared cells dropped) ⇒ ~17 × 12 = 204
connections per cue. Total R1 → R2 edges ~6500, vs the
random baseline's ~12000.

### Verified — 4 seeds × 16 epochs

| Seed | Untrained same | Untrained cross | Trained same | Trained cross | Δ cross | Eval-drift L2 (R2→R2) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1.000 ± 0.000 | 0.467 ± 0.106 | 1.000 ± 0.000 | 0.329 ± 0.207 | **−0.138** | +0.023 |
|  7 | 1.000 ± 0.000 | 0.485 ± 0.082 | 1.000 ± 0.000 | 0.326 ± 0.194 | **−0.159** | +0.028 |
| 13 | 1.000 ± 0.000 | 0.450 ± 0.122 | 1.000 ± 0.000 | 0.295 ± 0.190 | **−0.155** | +0.247 |
| 99 | 1.000 ± 0.000 | 0.433 ± 0.128 | 1.000 ± 0.000 | 0.247 ± 0.187 | **−0.186** | +0.042 |

Aggregate (n = 4 seeds):
- Untrained: same = 1.000 ± 0.000  cross = 0.459 ± 0.022
- Trained:   same = 1.000 ± 0.000  cross = 0.299 ± 0.038
- Δ same   = +0.000  (decorrelated wiring dampens eval-phase
                      plasticity to L2 0.02–0.25 vs iter-53's
                      +25 to +29 ⇒ eval is effectively
                      deterministic, no attractor erosion)
- Δ cross  = **−0.160**
- **Δ-of-Δ = +0.160 (acceptance PASSED)**

Paired t-test on per-seed Δ cross (n = 4): t(3) ≈ −16,
**p ≪ 0.001**. Bekos's primary acceptance ("cross-cue trained
< cross-cue untrained, p < 0.05") is met by a wide margin.

State-reset assertion: PASSED (4/4 seeds untrained
same_cue_mean = 1.000 ± 0.000). Decorrelated invariant: PASSED
(8/8 arm × seed combinations).

### Honest reading

**The decorrelation worked.** Three layered observations:

1. **Trained cross-cue dropped 35 % below untrained**
   (0.299 vs 0.459, p ≪ 0.001 paired). Distinct cues' top-3
   sets are now substantially less overlapping after 16
   epochs of teacher-forcing on the disjoint topology. Same
   protocol on iter-53's random topology produced
   Δ cross = +0.000.
2. **Trained same-cue stayed at 1.000 — no attractor erosion.**
   Eval-phase L2 drift collapsed from iter-53's +25 to +29
   down to +0.02 to +0.25 under decorrelated wiring.
   With ~17 unique R1 cells per cue × FAN_OUT 12 = 204
   directed connections (vs 12 000 random), the cue-driven
   spike traffic during eval is too sparse to drive
   meaningful plasticity. Plasticity at eval is *practically*
   off, even though it is *configurationally* on.
3. **Cross-cue absolute values are higher under decorrelated
   init (0.459 untrained vs iter-53's 0.058).** Not a
   regression — with sparser cue drive, the kWTA top-3 is
   dominated by cue-independent recurrent R2 dynamics. The
   right comparison is *within* the decorrelated arm
   (trained vs untrained), where Δ cross = −0.160 says
   training visibly re-routes the recurrent equilibrium
   toward cue-specific basins. Comparing absolute cross-cue
   across iter-53 and iter-54 conflates two different "noise
   floors" the metric reports.

iter-55 entry per Bekos's pre-fixed branching matrix is
**branch M1**: keep decorrelation + plasticity combined,
sweep training schedule / epochs / target_clamp_strength to
maximise the gain. (Branches M2 = consolidation and M3 =
deeper topology rejected — the cross-cue Δ is significant and
attractor erosion is zero.)

A cautious sub-question for iter-55: eval-phase plasticity
under decorrelated wiring is essentially off, so the same-cue
= 1.000 in the trained arm is "deterministic LIF + minimal
plasticity" rather than "engram robust under continued
plasticity". To probe attractor robustness specifically,
iter-55 could (a) add stochastic noise during eval, or (b)
re-run the iter-54 training scheme on iter-53's random
topology to isolate the plasticity-erosion-vs-specificity-
gain trade-off. Sub-experiment, not the critical path.

### Methodological lesson

iter-50: save the simplest configuration as a regression guard.
iter-51: a guard is only a guard if its baseline excludes the null.
iter-52: an analytical null is not an empirical control.
iter-53: when the literal acceptance direction is bounded by
construction, derive it from the protocol's mathematical bounds.
**iter-54: when the metric reports a "cleaner" number on a
random topology than on an architecturally cleaner one, the
metric is reading something else than what its name suggests.
Re-derive what the metric measures under the new topology
*before* claiming a result.**

The disjointness invariant is the iter-54 equivalent of iter-52's
L2 bit-identity check. Both catch a class of "the protocol
leaked" bug that the visible config would have hidden.

All eval lib tests still green (10/10 with the new
`decorrelated_init_is_pairwise_disjoint`); clippy `-D warnings`
clean.

## Unreleased — Iteration 53 (decoder-relative Jaccard, Voll-Implementation)

Bekos picked Option B Voll-Implementation off the iter-53.0 smoke
gate (mean Jaccard = 0.667, informative regime), with cross-cue
Jaccard added as the second axis. iter-53 is a decoder-relative
metric that bypasses the forward-drive bias iter-52 surfaced.

### Added — single commit

- `JaccardMetrics`, `JaccardArmResult`, `JaccardSweepResult`
  public types in `crates/eval/src/reward_bench.rs`.
- `pub fn run_jaccard_bench(corpus, cfg, seeds)` —
  trained + untrained × N seeds in one call.
- `pub fn render_jaccard_sweep(&JaccardSweepResult)` —
  per-seed table + aggregate Δ-of-Δ.
- Private helpers: `evaluate_jaccard_matrix` (32-cue × 3-trial
  collector), `train_brain_inplace` (training without metrics),
  `run_jaccard_arm` (one seed × one arm).
- `--jaccard-bench --seeds N1,N2,…` CLI flags in the example.
- All re-exports plumbed through `crates/eval/src/lib.rs`.

### Protocol — what the metric measures

For every cue in the 32-word vocab, present 3 trials.
**Full `brain.reset_state()`** between trials (R1 + R2 + cross-
region pending queue + traces + V + refractory + eligibility +
neuromodulator), not the previous R2-only reset that the
iter-53.0 smoke showed produced membrane carry-over.

Drop trial 0 as burn-in. Compute on trial[1] and trial[2]:

- `same_cue_mean` — mean over 32 cues of `Jaccard(trial[1], trial[2])`
- `cross_cue_mean` — mean over 496 cue pairs of
  `Jaccard(matrix[i][1], matrix[j][1])` for `i < j`

**Untrained arm** (`epochs = 0`, `no_plasticity = true`):
plasticity never enabled → deterministic LIF + full reset →
`same_cue_mean == 1.0` exactly. State-reset assertion panics if
this invariant is violated. iter-52's L2-norm bit-identity
check is preserved end-to-end on this arm.

**Trained arm** (cfg as given, plasticity enabled): plasticity
**stays ON during eval**, per Bekos's spec
("im trained Run würde Plastizität zwischen Trials variieren,
was *gewollt* ist"). Membrane state is reset per trial, but
synapse weights drift between trials → trial 2 depends on trial 1
*via plasticity, not via membrane state* — exactly the
dependency Bekos wants to measure. iter-52's L2 invariant is
deliberately dropped on this arm; pre/post L2 is logged as a
drift readout instead.

### Verified — 4 seeds × 16 epochs

| Seed | Untrained same | Untrained cross | Trained same | Trained cross | Eval-drift L2 (R2→R2) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1.000 ± 0.000 | 0.058 ± 0.137 | 0.828 ± 0.241 | 0.056 ± 0.123 | +27.08 |
|  7 | 1.000 ± 0.000 | 0.058 ± 0.134 | 0.891 ± 0.210 | 0.056 ± 0.114 | +24.50 |
| 13 | 1.000 ± 0.000 | 0.056 ± 0.121 | 0.875 ± 0.220 | 0.062 ± 0.134 | +25.54 |
| 99 | 1.000 ± 0.000 | 0.062 ± 0.126 | 0.922 ± 0.184 | 0.059 ± 0.121 | +28.55 |

Aggregate (n = 4 seeds):
- Untrained: same = 1.000 ± 0.000  cross = 0.058 ± 0.003
- Trained:   same = 0.879 ± 0.039  cross = 0.058 ± 0.003
- Δ same   = **−0.121** (attractor erosion)
- Δ cross  = **+0.000** (no specificity gain)
- **Δ-of-Δ = −0.121 (acceptance FAILED)**

State-reset assertion held on every untrained arm (4/4 seeds).
L2 bit-identity held on every untrained arm (4/4 seeds).

### Honest reading — direction of Δ same-cue

Bekos's literal acceptance criterion was *trained same-cue >
untrained same-cue, significant*. Under this protocol that
direction is upper-bounded by construction: untrained = 1.0,
plasticity-induced drift in trained → trained ≤ 1.0. Trained =
untrained is impossible to falsify as a learning signal.

The right reading of trained same-cue is therefore not "larger
than untrained" but **"how close to 1.0 the trained arm
remains"** — a direct measure of how attractor-like the
post-training engram is when plasticity continues to act on it.
Trained → 1.0 = stable engram, robust to continued plasticity.
Trained → 0 = engram fragile, plasticity drift dominates.

Cross-cue keeps the original direction: trained < untrained =
specificity rises with training.

The Δ-of-Δ summary number remains the single engram-formation
indicator:

```text
Δ-of-Δ = (trained_same − untrained_same) − (trained_cross − untrained_cross)
       = (specificity gain) − (attractor erosion)
```

Positive Δ-of-Δ ⇒ specificity gain outpaces erosion ⇒ engrams
form *and* are cue-specific. Zero or negative ⇒ erosion is
faster than specificity gain.

**Acceptance: FAILED** (Δ-of-Δ = −0.121).

Reading the two axes separately:

1. **Plasticity is alive in the trained brain.** Eval-phase L2
   drift is **+25 to +29 (R2→R2)** on every seed; plasticity
   actively modifies weights during the matrix collection.
   Same-cue at 0.879 (substantially below 1.000) confirms
   this drift translates into different decoder responses
   across trials.

2. **The trained brain has *some* attractor structure.** Same-
   cue at 0.88 ± 0.04 means trial 2 and trial 3 share ~88 %
   of their top-3 words, well above the random floor for two
   random 3-element samples from a 32-word vocab. So weights
   *did* learn something — cues land in *some* basin that's
   at least partially robust to continued plasticity.

3. **The attractors are NOT cue-specific.** Cross-cue is **flat
   at 0.058 ± 0.003** in both arms — distinct cues' top-3 sets
   overlap at exactly the same rate as on a fresh forward-only
   brain. Training redistributes weight mass and creates per-
   cue basins, but those basins are not aligned with cue
   identity at the decoder layer.

This is a clean negative result for "did teacher-forcing
produce cue-specific engrams in 16 epochs". Fully consistent
with iter-52: forward-drive bias dominates the decoder
geometry, and 16 epochs of teacher-forcing on the current
architecture has not broken that uniformity.

iter-54 has to address cue-specificity at the architecture or
schedule layer, not at the metric layer. Candidates documented
in `notes/53-decoder-relative-jaccard.md` (decorrelated initial
projections; reward cue-specificity directly; cue-only schedule
final phase via `--association-training-gate-r1r2`).

### Methodological lesson

iter-50: save the simplest configuration as a regression guard.
iter-51: a guard is only a guard if its baseline excludes the null.
iter-52: an analytical null is not an empirical control.
**iter-53: when the literal acceptance direction is bounded by
construction, derive the actual acceptance from the protocol's
mathematical bounds — and document the derivation.**

The state-reset assertion (untrained `same_cue_mean == 1.0`) is
the iter-53 equivalent of iter-52's L2-norm bit-identity check.
Both catch a class of "the protocol leaked" bug that the
visible config would have hidden.

All eval lib tests still green; clippy `-D warnings` clean.

## Unreleased — Iteration 52 (untrained-brain control)

Bekos's iter-52 spec: `--no-plasticity` toggle that gates every
plasticity enable, plus L2-norm bit-identity sanity assertion,
plus 4-seed × 16-epoch run on `--iter46-baseline --no-plasticity`,
plus pre-fixed iter-53 branching matrix.

### Added — single commit

- `TeacherForcingConfig.no_plasticity` field + CLI flag
  `--no-plasticity` (alias `--frozen-weights`).
- Plasticity-enable gate at three sites the L2 sanity caught:
  (1) run-time setup, (2) per-epoch ActivityGated re-enable,
  (3) mid-trial disable/enable cycles in the Arm-B-diagnostic
  block AND the epoch readout.
- `brain_synapse_l2_norms(brain)` helper.
- Run-start log + run-end `assert!` of bit-identical L2 norms
  under `no_plasticity = true`.

### Verified — 4 seeds × 16 epochs, all bit-identical L2

```
seed | initial L2  | post L2     | match | top-3 mean
 41  | 136.5766…   | 136.5766…   |  ✓    | 0.0413
 42  | 136.0980…   | 136.0980…   |  ✓    | 0.0225
 43  | 136.3604…   | 136.3604…   |  ✓    | 0.0375
 44  | 136.5731…   | 136.5731…   |  ✓    | 0.0563
```

Aggregated over 64 epoch-samples:
- Untrained top-3 mean: **0.039** (95 % CI [-0.008, 0.086])
- iter-51 trained top-3 mean: **0.107** (95 % CI [0.069, 0.145])
- Δ trained − untrained: **0.068, ~2.2 σ**
- Untrained vs random 0.094: **−0.055, ~2.3 σ — significantly BELOW random**

### Honest reading — Mess-Frage per Bekos's matrix

The first L2-norm assertion failure on the initial run caught a
9× weight blowup that the visible config would have hidden:
mid-run `disable_stdp` / `enable_stdp(stdp_params)` cycles in
the Arm-B-diagnostic and epoch-readout blocks were silently
turning plasticity back on. After all three gate sites closed,
all 4 seeds produced bit-identical pre/post L2 norms.

Two new statistical readings emerge:

1. **Plasticity is doing something measurable.** Trained 0.107
   vs untrained 0.039 is a real Δ at ~2.2 σ. iter-51's
   "indistinguishable from chance" was too conservative — it
   compared trained to an analytical random model, not to an
   empirical decoder-on-fresh-brain control.
2. **The decoder has a bias against the correct target on a
   fresh brain.** Untrained top-3 sits at 0.039 — significantly
   *below* the 0.094 random baseline. With `r2_active = 180`
   in the untrained brain (vs 145 trained), the fingerprint
   dictionary is dominated by a uniform forward-projection
   pattern; `decode_top` returns the same alphabetic-tiebreak
   "default" set on every cue, almost never including the
   correct target.

Per Bekos's pre-fixed branching matrix:

| Untrained top-3 | Branch | This data |
| --- | --- | :-: |
| ≈ 0.107 (CI overlap) | Architecture inert | ❌ |
| ≤ 0.085 | **Measurement question** | **✓** (0.039 ≪ 0.085) |
| ≥ 0.13 | Sign question | ❌ |

### iter-53 implication — decoder-relative readout, NOT new mechanism

iter-53 should NOT:
- Build new architecture (the brain is doing something).
- Sweep plasticity parameters (real but small lift; no
  parameter is going to 10× the 0.068 Δ).
- Multi-seed power-analyse trained alone (the trained-vs-
  untrained Δ is the relevant signal, not a tighter trained-
  alone CI).

iter-53 SHOULD: replace top-3-against-fingerprint with a
decoder-relative metric that doesn't degrade on highly
correlated dictionaries. Two candidates, both ~30 min code:
(a) per-trial trained-minus-untrained Δ; (b) trial-to-trial
Jaccard consistency on 3× repeated cues.

### Methodological lesson

iter-50: save the simplest configuration as a regression guard.
iter-51: a regression guard is only a guard if its baseline
excludes the null.
**iter-52: an analytical null hypothesis is not the same as
an empirical untrained control. Use the control whenever the
decoder can have a bias.**

The L2 sanity assertion that caught the 9× blowup on the first
run was the single most leveraged 10 lines of diagnostic code
in the whole iter-44…52 chain.

All 9 eval lib tests still green; clippy `-D warnings` clean.

## Unreleased — Iteration 51 Schritt 1 (Arm B saturation)

Bekos protocol for iter-51 had three steps; Schritt 1 was a
single 16-epoch run with `--iter46-baseline` to answer "is 0.19
a stable operating point, the start of a learning curve, or a
metastable transient?". No code change, no new commit
infrastructure beyond the iter-50 flag.

### Verified — `--epochs 16 --reps 4 --iter46-baseline`

```
Arm B (R-STDP) top-3 trajectory:
Epoch:   0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
top-3: 0.19 0.12 0.06 0.06 0.12 0.06 0.12 0.12 0.19 0.19 0.06 0.06 0.12 0.06 0.06 0.12
```

**top-3 mean = 0.107**, oscillates 0.06 ↔ 0.19 with no
monotone trend. `top-1 = 0.00 every epoch`.

### The hard read — `0.19` is noise, not a learning signal

- Random baseline (3/32) = **0.094**
- Arm B 16-epoch mean = **0.107**
- Per-epoch Bernoulli StdDev with `n = 16` ≈ **0.077**
- 95 % CI for the mean: **[0.069, 0.145]**
- **Random 0.094 sits inside the CI.**

Arm B is **statistically not distinguishable from chance** on
the top-3 metric. The 0.19 hits in epochs 0, 8, 9 are noise
peaks of a chance-level distribution.

Secondary metrics confirm:
- `mean reward ≈ −0.60` vs random expectation `−0.68` —
  marginally less negative, ≈ 5–10 % "doing something" gap.
- `r2_active = 145` stable across all 16 epochs (no learning
  trajectory in cell counts).
- `tgt_hit_mean = 2.4–2.8` vs random expectation
  `145 × 30/1400 = 3.10` — sub-random throughout.

### Implication — iter-52 entry, NOT Bekos's Schritt 2/3

The iter-51 plan presupposed `0.19` was a real signal and asked
"ceiling vs starting point". Data answers: **neither**. Sweeping
parameters against a chance-level baseline is a coin-flip with
extra steps.

iter-52 entry should **statistically validate that any learning
exists at all** before any mechanism work:

1. **Multi-seed Arm B** (~ 30 min): seeds 41–45, 16 epochs
   each. With 80 epoch-samples, SE of mean drops to ~0.009;
   `top-3 mean > 0.11` would be 2 σ above chance.
2. **Untrained-brain control** (~ 5 min): forward-projection
   + random recurrent, no plasticity. If `top-3 ≈ 0.107` too,
   the chain has been measuring the forward baseline since
   iter-44.
3. **Trial-to-trial consistency** (~ 30 min code): same cue
   3×, Jaccard similarity of top-3 sets. Under chance ≈ low,
   under learning ≈ high. One new metric, decoder-relative,
   meaningful in BOTH paths.

### Methodological lesson

iter-50: "save the simplest working configuration as a
regression guard."
iter-51: **"a regression guard is only a guard if its baseline
is statistically distinguishable from the null."**

The whole iter-44…50 chain measured against `top-3 = 0.19`
without ever testing whether that was above chance. Five
iterations of methodology against an unverified baseline.

This isn't a failure of methodology — Bekos's protocol forced
exactly the 16-epoch reproduction that revealed the statistical
situation. But it sharpens what "baseline" requires before it
counts as evidence: a confidence interval that excludes the
null.

All 9 eval lib tests still green; clippy `-D warnings` clean.
No code change in this iter — pure data.

## Unreleased — Iteration 50 (Arm B reproduction)

Bekos diagnostic: before the fourth consecutive parameter sweep
(STDP magnitude, by iter-49 elimination) or any architecture
extension (Pfad 2 bridge), reproduce iter-46 Arm B's reported
top-3 = 0.19 on the current branch code. The data point
unaddressed for 5 iterations: Arm B's 0.19 is **3× the random
baseline** and **3× higher than every selectivity-positive arm
in iter-47/48/49**.

### Added — single commit

- `--iter46-baseline` CLI flag + `TeacherForcingConfig.iter46_baseline`
  field. When set, simultaneously reverts at runtime:
  * `INTER_WEIGHT 1.0 → 2.0`
  * `R2_INH_FRAC 0.30 → 0.20`
  * `iSTDP a_plus 0.30 → 0.10`, `tau_minus 8 → 30 ms`
  * Skips `enable_intrinsic_plasticity` (Diehl-Cook OFF)
  * Forces `iter49_mode = None`
- `target_r2_map` is now built unconditionally so the iter-47/48/49
  sparsity metrics (`selectivity_index`, `target_hit_pre_teacher_*`)
  are populated in the no-teacher Arm B path too.
- Per-trial plasticity-OFF cue-only diagnostic sample after each
  rep in the no-teacher branch — collects `active_samples` and
  `target_hit_samples` for the sparsity-metric calculation.

### Verified — `--epochs 4 --reps 4 --iter46-baseline`

| Arm | top-3 epoch 0 | best top-3 | mean reward | selectivity |
| --- | ---: | ---: | ---: | ---: |
| **Arm B (R-STDP, no teacher)** | **0.19** ✓ | **0.19** | **−0.59** | −0.009 |
| Arm A (pure STDP) | 0.00 | 0.06 | 0.00 | −0.008 |

| Comparison: same 4 epochs, current iter-49 defaults |
| --- |
| iter-48 Config 1 (teacher on, current code): top-3 = 0.06 |
| iter-48 Config 2 (+ istdp-during-prediction):  top-3 = 0.06 |

### Outcome — (b) with critical nuance

**top-3 = 0.19 reproduces in epoch 0 of Arm B.** Code drift
hypothesis (c) **falsified**. But the iter-47/48/49 sparsity
metrics show negative selectivity in Arm B — and a numerical
sanity check shows why: the canonical hash R2 SDR (used as the
ground-truth set for selectivity) is **never causally activated
in the no-teacher path**. The "expected target_hit under uniform
random firing" with `r2_active = 173` is `173 × 30/1400 = 3.71`;
the observed 3.00 sits *below* random expectation. Negative
selectivity in Arm B is not a learning signal — it's the
chance result of comparing an arbitrary hash subset against
unrelated firing.

### Mechanistic explanation (why teacher-forcing is *worse* here)

Two contributing causes:

1. **Phase budget**: Arm B drives `cue + target` together for
   ~70 ms of overlapping pre/post coincidences per rep
   (`CUE_LEAD_MS = 40` + `OVERLAP_MS = 30` + `TARGET_TAIL_MS = 30`).
   The 6-phase teacher schedule has only `teacher_ms = 40 ms`
   of plasticity-on coincidence — less than half.
2. **Trivial-learning trap**: the 250 nA R2-clamp activates
   target cells *directly*. STDP then learns "when clamp is on,
   target cells fire" — a tautology, not a cue→target
   association.

### Implications for iter-51

**Ruled out by data:**
- "Raise STDP a_plus" sweep on iter-49 defaults (would optimise
  against a metric that's now known to be meaningless in the
  no-teacher path that actually performs better)
- Pfad-2 bridge architecture (premature; baseline is unsolved)

**Ruled in:**
- **Iter-51 = reductive Arm B parameter study**: vary
  `reps_per_pair`, `w_max`, R-STDP `eta` one-at-a-time, measure
  top-3 (the metric that reads in this path) over 16 epochs.
  Find out whether 0.19 is a ceiling or a starting point.
- **Replace `selectivity_index`** with a decoder-relative
  measurement that's meaningful in BOTH paths (e.g. "top-3
  against the per-epoch fingerprint dictionary") before
  trusting it for further architectural decisions.

### Methodological lesson (notes/50, full version)

Five iterations of clean methodology optimised against
`selectivity_index` in arms where the metric is structurally
meaningless. The simplest working configuration (Arm B) was
documented and discarded in iter-46 instead of being saved as a
regression guard. Top-3 quietly held at 0.06 across iter-47/48/49
while Arm B was always at 0.19 — invisible because we stopped
running it.

All 9 eval lib tests still green; clippy `-D warnings` clean.

## Unreleased — Iteration 49 (iSTDP bounds & schedule sweep)

3-point parallel sweep on the *under-tuned* side of the iter-48
collapse boundary, three orthogonal axes against the same
attractor (Bekos protocol from notes/48-saturation, end). All
three under Config 2 base (--istdp-during-prediction); same
pre-fixed acceptance as iter-48 phase A:

> sustained selectivity > 0 across epochs 4-16
> AND mean target_hit at epoch 16 > mean target_hit at epoch 4
> NO magnitude criterion — testing collapse-survival only.

### Added — commit 5ba5c25

- `Iter49Mode` enum (`None | WmaxCap | APlusHalf | ActivityGated`),
  re-exported.
- `TeacherForcingConfig.iter49_mode` + `gated_warmup_epochs` (2)
  + `gated_ramp_epochs` (2). Default `None` reproduces iter-48.
- `istdp_iter49(cfg, epoch)` — mode + epoch aware iSTDP builder.
- `run_reward_benchmark` calls `enable_istdp(...)` at the start
  of each epoch with the current value (no-op for stable modes,
  ramped for ActivityGated).
- CLI: `--iter49-mode {none|wmax-cap|a-plus-half|activity-gated}`.

### Verified — three sweeps × 16 epochs each, full per-epoch tables in notes/49

| Sweep | Mechanism | Peak sel | Collapse epoch | Steady-state | Acceptance |
| --- | --- | ---: | ---: | ---: | :-: |
| **A: WmaxCap** (`w_max 8.0 → 2.0`) | symptom | -0.0080 | n/a (never positive) | -0.038 | **0/3** |
| **B: APlusHalf** (`a_plus 0.30 → 0.20`) | dynamic | +0.0184 | epoch 5 | -0.014 | **0/3** |
| **C: ActivityGated** (a_plus = 0 first 2 ep, ramp 0→0.30) | temporal | +0.0029 | n/a (lock) | +0.003 | 3/3 (artifact) |

### Honest reading

**0 out of 3 sweeps produce a positive learning regime** by any
meaningful definition.

- A weakens iSTDP suppression structurally → r2_active blows up
  to 100-119 (over [25,70] band), tgt_hit absolute is HIGHER
  than iter-48 (2.31 vs 1.34) but selectivity is WORSE. **Bekos's
  60 %-confidence WmaxCap hypothesis falsified.**
- B has the same trajectory shape as iter-48 — slightly higher
  peak, IDENTICAL collapse epoch (5). Halving a_plus delays
  nothing.
- C produces an entirely DIFFERENT failure mode: warmup with
  no iSTDP lets STDP saturate E→E unopposed → r2_active locks
  at **1400 (= entire R2-E pool, every cell fires every trial)**
  → selectivity ≈ +0.003 trivially passes the magnitude-free
  acceptance but is pure noise around uniform activity.

The diagnostic value: **three distinct failure modes from three
orthogonal axes of iSTDP tuning. None work.** This is information
no single experiment could have produced.

### Synthesis — iter-50 hypothesis (by elimination)

**iSTDP is not the primary lever.** The 15× rate asymmetry
between STDP `a_plus = 0.020` and iSTDP `a_plus = 0.30` (and
10× in `w_max`: 0.8 vs 8.0) means excitatory plasticity cannot
form selective engrams faster than iSTDP suppresses or saturates
the network. iter-50 candidate: **raise STDP `a_plus` (0.020 →
0.060) and/or `w_max` (0.8 → 2.0)** so E→E selective growth
outpaces iSTDP inhibition.

### Methodological note

Sweep C revealed the iter-49 acceptance criteria need a
saturation guard for iter-50:
`r2_active_pre_teacher_mean < 0.5 × |R2_E|` at epoch 16
(< 700 cells). Catches "trivially saturated" without
re-introducing magnitude pressure.

iter-49 infrastructure (Iter49Mode + epoch-aware iSTDP) stays
in the repo as an A/B platform — the diagnostic value of
"which axis fails how" doesn't disappear because no axis
succeeded.

## Unreleased — Iteration 48 phase A (saturation postmortem)

Bekos protocol from notes/48-istdp-tightening: pre-fixed 3-of-3
saturation acceptance, 16 epochs, both configs, no code change,
~5 min wallclock per config. Question: is iter-48's selectivity
flip the start of a learning curve (⇒ B1: η-lift) or a
metastable transient (⇒ Postmortem)?

### Verified — full 16-epoch curves, both configs (notes/48-saturation.md)

Identical trajectory in both configs:
- Epochs 0–4: monotone selectivity rise to peak (Config 1
  +0.0172 epoch 2 / Config 2 +0.0144 epoch 4), target_hit
  reaches 1.0–1.3.
- Epoch 5: hard collapse — selectivity → −0.006 to −0.008,
  target_hit → 0.12, r2_active drops by ~50%.
- Epochs 6–15: stable negative steady-state (Config 1 ≈
  −0.012, Config 2 ≈ −0.008). w̄ stable at ~1.40 throughout,
  wmax = 8.00 throughout, θ_E ≈ 0.003 mV / θ_I ≈ 0.004 mV
  (operationally invisible — same as iter-47a-pm).

| Acceptance | Config 1 | Config 2 |
| A1: > 0 in ≥ 12/16 | 4/16 ❌ | 4/16 ❌ |
| A2: epoch 13–16 ≥ 1–4 mean | ❌ | ❌ |
| A3: target_hit > 1.5 at epoch 13–16 | 0.13 ❌ | 0.13 ❌ |
| Total | 0/3 | 0/3 |

### Honest reading

The iter-48 selectivity flip is a **metastable transient**, not
a learning curve. But it survives the postmortem as a real
qualitative finding: 4 consecutive positive-selectivity epochs
in BOTH configs at +0.012 to +0.017 (a regime change that no
iter-44/45/46/47a configuration ever produced).

The collapse mechanism is **iSTDP cumulative over-inhibition**,
diagnostically distinct from any prior failure mode:
- NOT weight runaway (w̄ stable across 16 epochs)
- NOT cascade (r2_active DROPS at collapse, doesn't explode)
- NOT Diehl-Cook over-correction (θ values flat through collapse)

The iter-48 iSTDP parameters sit on the **over-tuned side** of
a collapse boundary. iter-49 should explore the under-tuned
side, not push further along the iter-48 axis. Three small,
parallel candidate experiments (each ~5 min smoke), all in
notes/48-saturation.md:

1. iSTDP w_max 8.0 → 2.0 (cap the wall growth)
2. iSTDP a_plus 0.30 → 0.20 (half-way back to iter-47a)
3. Activity-gated iSTDP (a_plus = 0 for first 2 epochs, then
   ramp — match the literature pattern "consolidate first,
   balance later")

Pre-fixed iter-49 acceptance: sustained selectivity > 0 across
epochs 4–16, AND mean target_hit at epoch 16 > mean target_hit
at epoch 4. No magnitude criterion yet — we are testing for
collapse-survival, not for top-3 lift.

Anomaly note from iter-48 phase 1 also resolved: Phase 3
plasticity (--istdp-during-prediction) is *almost* but NOT
strictly redundant. Through epoch 3 both configs are identical
to 4 decimals; epoch 4+ they diverge slightly (Config 2 has
tighter sparsity and softer collapse). The CLI flag stays as
an A/B knob.

## Unreleased — Iteration 48 (iSTDP-tightening, Vogels 2011)

Direct response to the iter-47a postmortem (commit 432cbee),
which ruled out k-WTA / Diehl-Cook tuning and ruled in tighter
iSTDP. Three atomic commits + 4-epoch acceptance smoke; per
Bekos protocol acceptance criteria fixed before phase 1.

### Added

- Commit `4bfacc0` — iSTDP retune for fast EI balance:
  `R2_INH_FRAC: 0.20 → 0.30`, `IStdpParams.tau_minus: 30 → 8 ms`,
  `IStdpParams.a_plus: 0.10 → 0.30`. INTER_WEIGHT and
  IntrinsicParams unchanged so iter-48 isolates the iSTDP variable.
- Commit `bdee598` — `--istdp-during-prediction` A/B flag
  (`TeacherForcingConfig.istdp_during_prediction`, default `false`).
  Splits iter-46's plasticity gate so iSTDP can run during the
  prediction phase independently of STDP / R-STDP.
- Commit `de5771c` — three new `RewardEpochMetrics` fields:
  * `r2_active_pre_teacher_p99` — 99th percentile of per-trial
    prediction-phase active counts. Catches avalanche tail trials
    that p90 hides. Iter-48 acceptance criterion `< 50`.
  * `theta_inh_mean` / `theta_exc_mean` — Diehl-Cook θ split by
    neuron kind. Diff `θ_I − θ_E` is the iSTDP-over-correction
    early warning.
  * `intrinsic_mean_by_kind` helper, reuses `percentile_u32`
    from iter-47a.

### Verified — Phase 1 smoke, 4 epochs × 2 configs, seed 42

| Criterion | Config 1 (default) | Config 2 (`--istdp-during-prediction`) |
| --- | :-: | :-: |
| `selectivity_index > 0.0` | **+0.0142 ✅** | **+0.0142 ✅** |
| `target_hit_mean > 5` | 1.23 ❌ | 1.06 ❌ |
| `p99(active) < 50` | 79 ❌ | 61 ❌ |
| **Total** | **1.5 / 3** | **1.5 / 3** |

Per protocol: not 3-of-3 ⇒ no Phase 2 run, no speculative pivot,
pause and document.

### Honest reading

The iSTDP retuning hypothesis is **directionally confirmed by data**:
**selectivity flipped from negative to positive for the first time
in the iter-44/45/46/47/48 chain** (iter-46 sat at -0.04, iter-47a
at -0.045 after collapse, iter-48 at +0.014 stable across three
consecutive epochs in both configs). `r2_active_pre_teacher_mean`
is in the [25, 70] band across all four epochs in both configs;
no cascade, p99 below 110 even on worst trial vs. iter-47a's
1599. iSTDP successfully caught the runaway recurrent dynamic.

But target_hit_mean ≈ 1 (vs. criterion 5) and p99 ≈ 60–80 (vs.
criterion 50) both miss; top-1/top-3 stay at chance because
target_hit ≈ 1 cannot dominate a 32-entry decoder. The right
cells are biased correctly; their absolute amplitude is still
small.

Open follow-ups for iter-49 (deliberately unanswered, per
protocol):

- Saturation at 16 epochs — does the +0.014 selectivity stabilise,
  drift up, or collapse as iter-47a's transient peak did?
- Why is `target_hit_mean ≈ 1` so far below the 30-cell clamp
  target? R-STDP `eta` lift, longer teacher phase, or per-pair
  p99 to detect bistable pairs all fit the same diagnostic
  budget.

Full per-epoch tables, both configs, in
`notes/48-istdp-tightening.md`. All 9 eval lib tests still green;
clippy `-D warnings` clean across the workspace.

## Unreleased — Iteration 47a postmortem (saturation + cascade + θ effect size)

Three diagnostic questions left after iter-47a-2's acceptance
sweep (notes/47a) — answered with explicit instrumentation
*before* writing iter-48 code, per Bekos's protocol:
"lieber eine Iteration verlieren an saubere Diagnose, als drei
Iterationen an spekulative Architekturänderungen".

### Added (commit pending)

- `drive_with_r2_clamp_traced(...)` — parallels
  `drive_with_r2_clamp` but records per-step R2-E spike count
  and per-step canonical-target hit count. Used only on the
  postmortem path; hot training loop untouched.
- `intrinsic_stats(brain, r2_e_set)` returns
  `(mean, std, min, max, frac > 1 mV)` of `Network.v_thresh_offset`.
- `run_postmortem_diagnostic(corpus, cfg, train_epochs)` —
  trains for `train_epochs`, prints per-epoch θ + weight stats,
  then runs ONE read-only diagnostic trial that captures the
  prediction-phase per-step trace.
- CLI: `--debug-cascade` switches to the postmortem path,
  `--postmortem-train N` controls epoch count (default 4).

### Verified (3 questions, full data in notes/47a-postmortem.md)

**(a) Saturation test — INTER_WEIGHT = 1.0 over 16 epochs**:
selectivity does NOT asymptote at zero and does NOT pass through.
It approaches zero in epoch 3 (-0.0005, target_hit 2.59), then
**collapses** in epochs 5-15 to ≈ -0.045 with target_hit 0.08.
The mechanism is partially right then structurally unstable
(catastrophic interference; the very cells that learn get
suppressed).

**(b) Cascade pattern at INTER_WEIGHT = 0.7**:
12 001 R2-E spikes in 20 ms prediction window, max 219 cells/step;
**early-vs-late ratio = 0.97** (NOT onset-burst, would be > 2.0);
trace oscillates between ≈ 10 and ≈ 200 cells/step (synchronised
recurrent bursting / neuronal avalanche). 0 canonical-target
cells fire in the entire 20 ms.

**(c) θ effect size**:
At INTER_WEIGHT = 1.0, θ_mean = 0.05 mV, max = 0.86 mV,
frac > 1 mV = 0.000 over 4 epochs — i.e. **0.3 % of the 15 mV LIF
swing**, operationally invisible. At INTER_WEIGHT = 0.7, epochs
0-2 identical (≈ 0.03 mV), then in epoch 3 (cascade)
**θ_mean jumps 95× to 2.84 mV** with 99.9 % of cells > 1 mV —
Diehl-Cook is **reactive after the cascade, not preventive**.

### Iter-48 architecture, data-driven

The iter-47-decision-note's default fallback (k-WTA per step) is
**ruled out**: spike volume at 0.7 is oscillatory, not
onset-burst; per-step k-WTA shaves peaks but does not remove the
recurrent imbalance that drives the avalanche.

The data point unambiguously at **tighter iSTDP (Vogels 2011)**
for fast EI balance:
- `R2_INH_FRAC: 0.20 → 0.30` (more inhibition)
- `IStdpParams.tau_minus: 30 ms → 8 ms` (fast response)
- `IStdpParams.a_plus: 0.10 → 0.30` (stronger LTP on silent E
  targets)
- Optional: enable iSTDP during prediction (CLI flag for A/B)
- Keep INTER_WEIGHT = 1.0; Diehl-Cook stays as a tie-breaker
  but is not the primary stabiliser.

Acceptance criteria for iter-48 phase 1 to be fixed pre-run:
selectivity > 0.0 AND target_hit_mean > 5 AND max per-step active
< 50 after 4 epochs.

## Unreleased — Iteration 47a (forward-drive scaling + adaptive threshold)

Iter-46 ended with `clamp_hit_rate = 1.00` but
`correct_minus_incorrect_margin ∈ [-0.06, -0.03]`: the teacher chain
was alive but recurrent learning had no fair shot against the R1 → R2
forward drive (90–180 active R2 cells per cue vs. 30 canonical
target cells). Iter-47a tests the literature-grounded fix
(Brunel-style INTER_WEIGHT scaling + Diehl-Cook adaptive threshold)
through a sequential 4-epoch sweep with explicit, pre-fixed
acceptance criteria.

### Added (commit `99540d0`)

- `INTER_WEIGHT 2.0 → 1.0` (after sweep evidence; see notes/47a).
- `enable_intrinsic_plasticity(intrinsic())` on R2 with the
  Diehl-Cook parameter set (`alpha_spike = 0.05, tau = 2000 ms,
  target = 0, beta = 1, offset_max = 5`). Mechanism existed since
  iter-44; iter-47 wires it into the harness.
- `TrialOutcome.pred_target_hits` — counts canonical-target
  neurons that fired during the prediction phase.
- 5 new `RewardEpochMetrics` fields:
  `r2_active_pre_teacher_{mean,p10,p90}`,
  `target_hit_pre_teacher_mean`, `selectivity_index`.
- `render_markdown` emits a second per-arm table with these.
- `percentile_u32` helper.

### Verified — sweep, 16 + 16 pairs, vocab 32, seed 42, 4 epochs

| INTER_WEIGHT | r2_act mean | r2_act p10/p90 | tgt_hit | selectivity | margin |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.8 | 0 / 3 | 0.00 | -0.0005 | -0.01 |
| **1.0** | **139** | 88 / 165 | 2.59 | -0.0005 | -0.02 |
| 0.7 | 507 (cascade) | 8 / 1599 | 9.38 | -0.0047 | -0.02 |

Acceptance criteria (4-of-4): no sweep point reached ≥ 3/4. The
**0.7 bistability** (epochs 0-2 stable at ~10 cells, epoch 3
explodes to mean 507 / p90 1599) is the key second-order finding —
adaptive θ alone cannot stop a recurrent cascade once STDP-grown
weights cross threshold.

### Honest reading

The forward-drive-only fix (iter-47a-2) is **insufficient**, but
the diagnosis is dramatically sharper than iter-46's:

- The forward-drive vs. recurrent-weight balance is in the right
  order of magnitude at INTER_WEIGHT = 1.0.
- The Diehl-Cook adaptive-threshold mechanism *does* work at the
  per-cell level: `target_hit_mean` grew monotonically
  1.16 → 2.59 over 4 epochs at INTER_WEIGHT = 1.0, and
  `selectivity_index` rose from -0.022 toward 0.
- Hard sparsity control (47a-3 = k-WTA) is **necessary, not
  optional** — the bistability at INTER_WEIGHT = 0.7 makes that
  explicit. This was identified pre-experiment in the
  architecture-decision note as a fallback; Phase 1 of 47a-2
  produced direct evidence that it is the actual next step,
  not a downstream-iter optimisation.

The iter-47 sparsity metrics (`r2_active_pre_teacher_{mean,p10,p90}`
and `selectivity_index`) are already wired to A/B-test the k-WTA
addition (iter-48 entry).

## Unreleased — Iteration 46 (teacher-forcing + R1→R2 gate)

Iter-45 documented honestly that pure STDP and R-STDP both stayed
at random-baseline accuracy on the 16-pair association task —
because the cue's R2 representation was almost entirely the
random R1 → R2 forward projection, leaving recurrent learning no
room to bias it. Iter-46 attacks the bottleneck directly: during
training, the canonical target SDR is **clamped into R2** via a
new dedicated driver, plasticity is gated to keep evaluation
clean, and the per-trial diagnostics make the chain visible.

### Added (5 atomic commits)

- `TeacherForcingConfig` (`enabled`, `cue_ms`, `delay_ms`,
  `prediction_ms`, `teacher_ms`, `tail_ms`,
  `target_clamp_strength`, `plasticity_during_{prediction,teacher}`,
  `wta_k`, three reward levels, `homeostatic_normalization`,
  `debug_trials`, `r1r2_prediction_gate`). Default `off()` —
  every iter-45 path stays bit-identical.
- `RewardConfig.teacher: TeacherForcingConfig` plus
  `with_teacher(epochs)` constructor.
- `canonical_target_r2_sdr(word, e_pool, k, salt)` — deterministic
  hash → fixed K-of-pool R2-E indices. Stable per `(word, salt)`,
  no per-trial drift.
- `drive_with_r2_clamp(brain, cue, r2_target, r1_str, r2_str,
  dur_ms, r2_e)` — drives R1 forward *and* R2 directly, returns
  spike counts.
- Six-phase trial schedule: cue → delay → prediction (plasticity
  off) → teacher (cue lead-in then clamp) → reward (from
  prediction, never from teacher) → tail.
- Anti-causal STDP timing fix: `lead_in = clamp(teacher_ms / 4,
  4, 12)` ms of cue alone before the clamp — without this fix the
  clamp's instantaneous target spikes would lead the cue's
  delayed R2 spikes and STDP would learn the wrong direction.
- `--association-training-gate-r1r2 [value]` CLI flag (default
  `1.0`, flag-only sets `0.3`) to attenuate cue current during
  the prediction phase only — training-only knob, evaluation
  always runs at full strength.
- Optional epoch-end homeostatic L2 normalisation of R2 → R2
  weights, off by default (`--homeostasis`).
- Extended `RewardEpochMetrics` with `random_top3_baseline`,
  `mean_rank`, `mrr`, `target_clamp_hit_rate`,
  `prediction_top3_before_teacher`, `eligibility_nonzero_count`,
  `r2_recurrent_weight_{mean,max}`, `active_r2_units_per_cue`,
  `correct_minus_incorrect_margin`, `decoder_micros`.
- `--debug-trial` prints up to 3 example pair traces per epoch.
- 2 new unit tests
  (`canonical_target_r2_sdr_is_stable_and_in_pool`,
  `drive_with_r2_clamp_makes_target_neurons_fire`); existing
  smoke (`reward_benchmark_smoke`) still passes ~ 6 s.

### Verified — full sweep, 16 pairs + 16 noise, vocab 32, seed 42

| Arm | Description | Epoch 19 top-3 | margin | clamp | Wall |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Pure STDP, no teacher | 0.06 | n/a | n/a | 4 min |
| B | R-STDP, no teacher | 0.19 | n/a | n/a | 4 min |
| C | R-STDP + teacher (step 3) | 0.00 | -0.05 | **1.00** | 7 min |
| C′ | + step-4b lead-in (4 epochs) | 0.06 | -0.03 | **1.00** | 1 min |
| D | + R1 → R2 gate 0.3 + homeostasis (4 ep) | 0.00 | -0.04 | **1.00** | 1 min |

### Honest reading

The teacher-forcing infrastructure works exactly as designed.
**`target_clamp_hit_rate = 1.00`** across every teacher epoch —
the clamp activates every one of the 30 canonical target
neurons every time. The first run (arm C) produced a clean
diagnostic surprise: `margin = -0.04` (canonical-target cells
fire *less* than the rest under cue-only recall) — caused by
anti-causal STDP timing, fixed in step 4b.

After the timing fix, the margin stays around -0.03 to -0.05.
With the R1 → R2 prediction gate at 0.3 plus aggressive
homeostatic L2 normalisation, the *first* non-zero
`prediction_top3_before_teacher = 0.02` shows up at epoch 3 —
the right signal in the right direction, but two orders of
magnitude below the 20 % acceptance threshold and within
trial-to-trial variance.

**The bottleneck has moved.** Iter-45 said "R1 → R2 dominates
and we can't measure how much"; iter-46 says "R1 → R2 dominates
to a magnitude we can now read off the per-trial spike density
(90–180 active R2-E cells per cue when the canonical target is
30 cells)". The honest next-iter (47) bottleneck is **the
absolute scale of R1 → R2 vs R2 → R2**:

- Reduce `INTER_WEIGHT` from 2.0 to 0.5 so STDP-grown weights
  actually compete with forward drive.
- Add an explicit association-bridge region between R1 and R2.
- Make the R1 → R2 path itself learnable so teacher reward can
  shape the input projection, not only the recurrent loop.

The iter-46 harness is the platform on which any of those can
be A/B-tested against the same scoring contract — `notes/46`
has the full chain of measurements, the per-arm tables, and the
diagnostic stack.

## Unreleased — Iteration 45 (reward-aware pair-association harness)

### Added
- `crates/eval/src/reward_bench.rs` — `RewardPair`, `RewardCorpus`,
  `RewardConfig`, `RewardEpochMetrics`, `run_reward_benchmark`,
  `default_reward_corpus`, `render_markdown`. Pair-association
  task with deliberate distractors, staggered cue → target
  training, per-trial reward delivery, per-epoch top-1 / top-3
  readout.
- `crates/eval/examples/reward_benchmark.rs` — CLI runner that
  produces a side-by-side Markdown comparison of pure STDP vs
  R-STDP across N epochs.
- 1 new smoke test (`reward_benchmark_smoke`) — runs both arms on
  a 4-pair sub-corpus for 2 epochs and asserts every metric is
  finite and in [0, 1].
- `notes/45-reward-bench.md` — full architectural rationale,
  trial schedule, measured results, honest reading.

### Verified
The harness wires R-STDP cleanly: dopamine + eligibility tag move
weights, both noise-only-arms have a `mean_reward = -1.0` baseline,
and the smoke test passes in ~22 s. R-STDP shows a small
noise-suppression advantage over pure STDP (mean noise-top-3 0.10
vs 0.16 over 6 epochs at reps = 4) but neither configuration
converges to above-chance pair-association accuracy in the
training time the current architecture allows. The likely next
experiment — teacher-forcing the target SDR directly into R2 —
is documented in `notes/45` as a concrete follow-up.

## Unreleased — Iteration 44.1 (decoder confidence floor)

### Added
- `EngramDictionary::decode_top_above(active, k, min_score)` —
  identical to `decode_top` but **omits** engrams whose containment
  ratio is below `min_score` instead of filling the top-k slot with
  the next-best garbage. `decode_top` is now `decode_top_above(_,
  k, 0.0)` so existing callers see no change.
- `ScaleBrain::query_with_threshold` /
  `ScaleBrain::evaluate_with_threshold` — pass-through wrappers so
  the scale benchmark can take a confidence floor.
- `--decode-threshold` flag on the `scale_benchmark` example.
- Two new unit tests in `crates/encoders/src/decode.rs`
  (`decode_top_above_filters_low_confidence_matches`,
  `decode_top_unchanged_by_threshold_refactor`) + the regression
  guard that `decode_top` is bit-identical to the new method at
  threshold `0.0`.

### Verified — same 32-sentence corpus, seed 42, `--iter44 off`

| `--decode-threshold` | FP / Q | Token reduction | Self-recall |
| ---: | ---: | ---: | ---: |
| `0.0` (pre-iter-44) | 4.50 | 38.9 % | 100 % |
| `0.10` | 4.50 | 38.9 % | 100 % |
| **`0.20`** | **0.62** | **79.7 %** | 100 % |
| `0.30` | 0.00 | 84.7 % | 100 % |

The headline:
- **FP / query: 4.50 → 0.62 (− 86 %)**
- **Token reduction: 38.9 % → 79.7 % (+ 2.0×)**
- self-recall stays at 100 %; decoder latency unchanged.

The "recall of co-occurring neighbours" that fell to zero was almost
entirely noise: the random-overlap floor for two KWTA-100 patterns
in an R2 of 8000 E neurons is 12.5 %, so anything below that is
statistically indistinguishable from chance. The threshold makes
the *real* engram-orthogonality problem visible — and that's the
gap iter-44's reward + structural mechanisms are designed to close.

## Unreleased — Iteration 44 (breakthrough plasticity stack)

### Added
- **Triplet STDP** (Pfister & Gerstner 2006). Slow `pre_trace2` /
  `post_trace2` lazy buffers on `Network`; new `StdpParams.a3_plus`,
  `a3_minus`, `tau_x`, `tau_y` fields. Default 0 → identical to
  pair-STDP for every pre-iter-44 configuration.
- **Reward-modulated STDP with eligibility traces** (`crates/snn-core/src/reward.rs`).
  Per-synapse eligibility tag, decay τ ≈ 1 s, gated by a global
  scalar `Network::neuromodulator`. `Brain::set_neuromodulator(...)`
  broadcasts a dopamine surrogate to every region.
- **Metaplasticity** with the BCM sliding LTP/LTD threshold
  (`crates/snn-core/src/metaplasticity.rs`). Per-post-neuron rate +
  θ traces; `MetaplasticityParams::modulator(rate, θ)` multiplies
  the STDP Δw on incoming edges.
- **Intrinsic plasticity / spike-frequency adaptation**
  (`crates/snn-core/src/intrinsic.rs`). Per-neuron adapt trace +
  `v_thresh_offset` slot; the LIF integration reads
  `v_threshold + offset` whenever the feature is on.
- **Heterosynaptic L1 / L2 normalisation**
  (`crates/snn-core/src/heterosynaptic.rs`). Periodic per-post
  excitatory-incoming weight-norm cap. Defaults: L2 with target
  1.5, applied every 200 steps.
- **Structural plasticity** (`crates/snn-core/src/structural.rs`).
  Pruning: E→E synapses below `prune_threshold` for `prune_age_steps`
  evaluations are removed from the adjacency buckets and marked
  dead. Sprouting: hot pre/post pairs with no current edge get a new
  one at `sprout_initial`. `Network::compact_synapses()` reclaims
  dead slots.
- **Offline replay / consolidation**
  (`crates/snn-core/src/replay.rs`, `Network::consolidate`,
  `Brain::consolidate`). Drives the top-k engram cells in pulses
  with full plasticity left on; alternates forward / reverse order
  on successive calls.
- `Brain::compact_synapses` — sums the per-region compaction count.
- `crates/snn-core/tests/iter44_breakthrough.rs` — 15 new tests, one
  positive + one regression-guard per mechanism plus a composite
  full-stack + a passive-network regression test.
- `notes/44-breakthrough-plasticity.md` — architectural rationale,
  composition into the existing pipeline, references, and limits.

### Changed
- `Network::step` now decays the new traces (when their feature is
  on), reads the BCM modulator on every STDP Δw, runs the periodic
  heterosynaptic and structural passes, and applies the
  reward-gated update at the end of the step. Off paths early-out
  on a single `bool` and stay byte-identical to the pre-iter-44
  hot loop.
- `Network` gained: `metaplasticity`, `intrinsic`, `heterosynaptic`,
  `structural`, `reward` as `Option<...>` configs;
  `pre_trace2`/`post_trace2`/`eligibility`/`rate_trace`/`theta_trace`/
  `adapt_trace`/`v_thresh_offset`/`prune_counters`/`dead_synapses`/
  `replay_flip` as transient lazy buffers.
- `lib.rs` re-exports `MetaplasticityParams`, `IntrinsicParams`,
  `HeterosynapticParams`, `NormKind`, `StructuralParams`,
  `PruneCounter`, `RewardParams`, `ReplayParams`.

### Verified
All 113 pre-existing tests still pass; the regression guard
`classical_passive_network_unchanged_by_iter44` asserts byte-identity
of the off-by-default hot loop. 15 new unit tests in
`crates/snn-core/tests/iter44_breakthrough.rs`.

### Benchmark
Measured against iter-43 on the deterministic 32-sentence corpus
(seed 42, `cargo run --release -p eval --example scale_benchmark
-- --iter44 {off | stability | tuned | full}`):

| Config | Train sec | Recall | FP / query | Latency |
| --- | ---: | ---: | ---: | ---: |
| `off` (iter-43 baseline) | 175 | 4.4 % | 4.50 | 16.1 ms |
| `stability` | 236 | 4.4 % | 4.50 | 16.9 ms |
| `tuned` | **27** | 2.7 % | 4.69 | **14.0 ms** |
| `full` | 355 | 1.6 % | 4.81 | 44.0 ms |

The iter-44 stack does **not** improve recall on this short-corpus
benchmark — see `notes/44` for the full reading. Heterosynaptic and
BCM scale weights uniformly per post-neuron and so don't change the
fingerprint kWTA pattern; reward-modulated STDP / replay / BCM-θ all
need longer training windows or a reward signal that the current
eval harness does not emit. The mechanisms are present, unit-tested
and ready for the *next* benchmark — multi-epoch corpora, streaming
input with consolidation gaps, and reward-aware retrieval — none of
which the iter-25 evaluation harness exercises.

### Caveats
- Snapshot schema gains six `Option<...>` fields and several
  `#[serde(skip)]` lazy buffers. Old snapshots load fine
  (`#[serde(default)]` everywhere); newly-saved snapshots store the
  new params if their feature was on at save time.
- Reward learning currently treats `excitatory_only = true` as the
  default; striatal D1 vs D2 sign asymmetries are out of scope.
- The structural sprouting walk is deterministic and bounded by
  `max_new_per_step`; a randomised reservoir may be needed at
  larger hot-set sizes than R2 = 10 000 currently produces.

## Unreleased — Iteration 25 (topology scaling: R2 → 10 000)

### Changed
- `R2_N`: 2 000 → **10 000** (5× more orthogonal space).
- `R2_P_CONNECT`: 0.10 → **0.03** (sparser recurrent; keeps
  synapse count manageable at 3 M instead of 10 M).
- `KWTA_K`: 220 → **100** (1 % sparsity instead of 11 %; baseline
  random-overlap probability drops by an order of magnitude).
- `CONTEXT_KWTA_K`: 60 → 30 (proportional).
- `FAN_OUT` (R1→R2): 10 → 30 (preserves forward drive density).
- `IStdpParams.a_plus`: 0.05 → 0.10 ; `a_minus`: 0.55 → 1.10 ;
  `w_max`: 5 → 8. More aggressive LTD on co-active E-targets so
  inhibitory plasticity can build separating walls in the
  larger pool.
- Same constant set propagated to `viz::state`, `eval::token_efficiency`,
  `eval::scale_bench`.

### Verified
All 113 existing tests still pass at the new topology without
threshold adjustments. `wiki_benchmark` (5-paragraph corpus)
preserves its ≥ 70 % token-reduction guarantee on the larger
brain. Re-run of the 100-sentence scale benchmark
(notes/43): metrics will be appended once the run completes.

### Caveats
- Single-region snapshot file size grows from ~30 MB to
  ~120-150 MB at this topology. Snapshot compression is a
  follow-up if it becomes a deploy issue.
- Brain-step wall-time at R2=10 000 is ~5× the iter-24 baseline.
  Pipeline profile and load-test numbers (notes/40, /41) are
  iter-24 baselines and will need re-running on the new
  topology.

## Iteration 24 — validation at scale, honest limits

### Added
- `crates/eval/src/scale_corpus.rs` — deterministic
  template-driven corpus generator across 8 knowledge domains.
  Reproducible from a `(seed, n_sentences)` pair, no external
  dataset; ground-truth co-occurrence map is recorded by
  construction so precision/recall/FP/FN can be measured against
  a real reference.
- `crates/eval/src/scale_bench.rs` — `ScaleBrain` train-once /
  query-many harness. Per-query metrics: token reduction,
  decoder latency, has-self, false positives (cross-domain
  bleed), false negatives (missed co-occurrences). Aggregated to
  a `ScaleSummary` and rendered as Markdown via
  `ScaleReport::render_markdown()`.
- `crates/eval/examples/scale_benchmark.rs` — CLI runner with
  `--sentences`, `--queries`, `--decode-k`, `--seed` flags.
- `crates/eval/tests/scale_bench_smoke.rs` — small-scale
  regression test (16 sentences, ~8 s) wired into CI.

### Verified (notes/42)
| n_sentences | precision | recall | mean reduction | FP / 6 | mean decode |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1.000 | 0.022 | 34.9 % | 4.75 | 383 µs |
| 100 | 1.000 | 0.021 | 40.6 % | 4.70 | 603 µs |

Honest publication-grade story (notes/42): the headline
"96.7 % token reduction" was on a 5-paragraph corpus and is not
reproducible at scale. On a 100-sentence corpus with 286 unique
vocabulary words Javis hits 100 % self-recall and ~40 % token
reduction, but associative-recall drops to ~2 % and
cross-domain bleed dominates the decoded output. Engram
capacity at the current R2 size (2000 neurons, KWTA_K=220) is
the next architectural wall.

## Iteration 23 — AoS → SoA + WS fire-and-forget, 1.4× pipeline

### Changed (`snn-core`)
- **AoS → SoA refactor.** `LifNeuron` now holds *only* `params` and
  `kind` — 32 B per neuron, 2 fitting in one cache line. The
  per-neuron transient state (`v`, `refractory_until`,
  `last_spike`, `activity_trace`) moved to parallel `Vec<f32>`
  buffers on `Network`, indexed lock-step with `neurons`.
- `LifNeuron::step` removed. The LIF math is now inlined in
  `Network::step` and `Network::step_immutable`, operating on the
  parallel slices directly — single straight-line loop, no per-
  neuron struct loads.
- `Network::add_neuron`, `reset_state`, `ensure_transient_state`
  updated to manage the new parallel Vecs.
- `Network::apply_synaptic_scaling` reads `self.activity_trace[post]`
  instead of `self.neurons[post].activity_trace`.

### Changed (`viz`)
- `tx.send(Event::Step{…}).await` → `tx.try_send(...)` in
  `run_with_cue_streaming_immutable`. Step events are
  visualisation breadcrumbs; if the WS consumer falls behind we
  drop the event into a new `javis_ws_step_dropped_total` counter
  rather than awaiting backpressure into the simulation loop.

### Tests
- `lif_basic.rs` rewritten to drive a one-neuron `Network` instead
  of calling the (removed) `LifNeuron::step`.
- `homeostasis.rs`, `injection.rs`, `immutable_step_equivalence.rs`
  updated to read transient state from the new parallel Vecs.
- All four spike-bit-identity equivalence tests still pass without
  modification — SoA path produces bit-identical spikes to the
  pre-refactor path.

### Verified (notes/41)
Criterion, `Network::step_immutable`, p < 0.05:

| size | iter 21 | iter 23 | improvement | total since iter 20 |
| ---: | ---: | ---: | ---: | ---: |
| 100  | 307 ns | 240 ns | -22.5 % | 2.27× |
| 500  | 1.66 µs | 1.17 µs | -26.7 % | 2.40× |
| 1000 | 3.73 µs | 2.69 µs | -26.8 % | 2.17× |
| 2000 | 9.32 µs | 7.35 µs | -18.6 % | 1.88× |

Pipeline (200 sequential recalls): total 8.05 ms → 5.77 ms (-28 %).
Load test: throughput 358 → 432 ops/s (+21 %), p99 -14 % to -19 %
across concurrency 1/10/50/100.

## Iteration 22 — pipeline profile: 77 % brain, NOT Amdahl-bound

### Added
- Six phase timers in `AppState::run_recall` covering
  `lock_overhead`, `encode`, `snn_compute`, `decode`,
  `rag_search`, `response_build`. Plus two sub-phase timers
  inside `run_with_cue_streaming_immutable` (`brain_compute`,
  `ws_stream`) so the dominant `snn_compute` phase can be split
  further.
- Prometheus histograms `javis_recall_phase_seconds{phase}` and
  `javis_recall_subphase_seconds{phase}`, sharing the existing
  duration-bucket layout from `viz::metrics::init`.
- Structured `tracing::info!` line per recall with all six phase
  durations as fields.
- `scripts/pipeline_profile.py` — drives N recalls, reads the
  per-phase histograms before+after, prints the breakdown sorted
  by mean. Defaults: 100 sequential recalls.

### Verified (notes/40)
Across 200 sequential recalls against the docker stack:

| phase | mean ms | share |
| --- | ---: | ---: |
| `snn_compute` | 7.91 | 97.9 % |
| `decode` | 0.13 | 1.6 % |
| `response_build` | 0.02 | 0.2 % |
| `lock_overhead` | 0.009 | 0.1 % |
| `encode` | 0.004 | 0.0 % |
| `rag_search` | 0.002 | 0.0 % |

`snn_compute` breakdown: `brain_compute` 6.18 ms (77 % of total
recall), `ws_stream` 0.53 ms (6.5 %).

Conclusion: brain-compute is still the dominant cost. Amdahl
hasn't taken over yet — a 2× brain-step speedup still buys 1.65×
on the full pipeline.

## Iteration 21 — profile-driven LIF speedup, 1.5×

### Added
- `crates/snn-core/examples/profile_step_immutable.rs` — hand-
  instrumented phase-by-phase profiler for `Network::step_immutable`.
  Three `Instant::now()` brackets around decay / LIF integration /
  delivery; runs 5000 steps and reports mean/p50/p99 plus share of
  total step time. Used in lieu of perf, which is unavailable in
  the dev sandbox.
- `network_step_immutable` benchmark in
  `crates/snn-core/benches/network_step.rs` covering sizes
  100/500/1000/2000.
- `total_input: Vec<f32>` scratch buffer on `NetworkState` for
  pre-summed channel input.

### Changed (`snn-core`)
- `Network::step_immutable` rewritten for autovectoriser-friendly
  inner loop: NMDA / GABA presence checks hoisted outside the
  per-neuron loop, channels pre-summed into `state.total_input`
  via four specialised straight-line loops, no per-iteration
  `Option::get` accesses. Spike-bit-identity to the mutating path
  preserved.

### Verified
- Profile (notes/39): LIF integration is 92 % of step time (decay
  4 %, deliver 4 %), so optimising the LIF loop is the right
  target.
- Criterion bench across sizes 100/500/1000/2000:

  | size | before | after | speedup |
  | ---: | ---: | ---: | ---: |
  | 100  | 545 ns | 307 ns | **1.78×** |
  | 500  | 2.81 µs | 1.66 µs | 1.69× |
  | 1000 | 5.85 µs | 3.73 µs | 1.57× |
  | 2000 | 13.79 µs | 9.32 µs | 1.48× |

  All p < 0.05.

## Iteration 20 — read-only recall: 2.5× throughput

### Added (`snn-core`)
- `NetworkState` and `Network::step_immutable(&self, &mut state,
  external)` — read-only step path that does the same LIF / synaptic
  delivery math as `Network::step` but never mutates synapses,
  ignores plasticity unconditionally, and writes every transient
  buffer to the caller-provided `state`.
- `BrainState` and `Brain::step_immutable(&self, &mut state,
  externals)` — same pattern at the multi-region orchestration
  layer; per-region `NetworkState`s and a per-recall `PendingQueue`
  live in the state argument.
- `Network::fresh_state()` and `Brain::fresh_state()` constructors.
- Four equivalence tests in
  `crates/snn-core/tests/immutable_step_equivalence.rs` proving
  spike-bit-identity with `Network::step` / `Brain::step` when
  plasticity is off, and that the read-only path leaves the brain's
  weights/clock untouched.

### Changed (`viz`)
- `AppState.inner` switched from `Mutex<Inner>` to `RwLock<Inner>`.
  Train / reset / snapshot-load take the write lock; recall, stats,
  snapshot-save take the read lock. Multiple concurrent recalls now
  proceed in parallel.
- `run_recall` builds a per-call `BrainState` and runs through
  `Brain::step_immutable` against the shared brain.
- The `disable_stdp_all() / disable_istdp_all() /
  disable_homeostasis_all()` voodoo at the start of recall is gone
  — the read-only step ignores plasticity unconditionally.

### Verified (notes/38)
Re-running `scripts/load_test.py --levels 1,10,50,100` against the
docker stack:

| concurrency | throughput | p99 client | server-mean |
| ---: | ---: | ---: | ---: |
| 1 | 112 ops/s | 11 ms | 7 ms |
| 10 | 357 ops/s | 48 ms | 9 ms |
| 50 | 359 ops/s | 244 ms | 9 ms |
| 100 | 358 ops/s | 564 ms | 9 ms |

Throughput 2.5× the Mutex-bottlenecked baseline; server-side latency
constant ~9 ms across all concurrency levels (was 7→68→346→685 ms).
The remaining client-side queueing is tokio runtime, not Brain.

## Iteration 19 — snapshot schema versioning

### Added
- Snapshot schema bumped to v2; new mandatory `metadata` block
  records `created_at_unix` and `javis_version` for ops triage.
- Migration framework: a `MIGRATIONS: &[(u32, MigrationFn)]` table
  walks the chain on load, so a snapshot at version `N` is parsed
  as `N`, transformed into `N+1`, … into the current version
  before the canonical struct is deserialised. Adding v3 later
  needs only one new entry in the table.
- `migrate_v1_to_v2` injects the synthesised metadata
  (`created_at_unix: 0`, `javis_version: "migrated-from-v1"`) so
  pre-v2 snapshots load on a v2 build.
- Refusing future versions with a clear error message — we cannot
  downgrade safely.
- Four new tests in `crates/viz/tests/snapshot_migration.rs`:
  current-version round-trip, v1-loads-through-migration,
  future-version-rejected, missing-version-field-rejected.

### Changed
- `load_from_file` now logs the schema migration explicitly when
  it happens (`from_version`, `to_version` fields), so operators
  see in their logs that an old snapshot was upgraded.

## Iteration 18 — concurrency cap

### Added
- `Semaphore`-based cap on simultaneous WebSocket sessions, default
  32. Configurable via `JAVIS_MAX_CONCURRENT_SESSIONS`.
- When the cap is reached, the upgrade handler responds with
  `503 Service Unavailable` + `Retry-After: 1` instead of letting
  the request queue indefinitely on the inner brain Mutex.
- New counter `javis_ws_rejected_total{action,reason}` tracks
  rejections; a single `reason="concurrency_cap"` label today,
  extensible for future rejection paths.
- `AppState::with_session_cap(cap)` constructor for tests that need
  to exercise the rejection path.
- Two new integration tests in `crates/viz/tests/concurrency_cap.rs`
  speaking raw HTTP/1.1 to observe the 503 response (the WS client
  hides non-101 statuses as handshake errors).

## Iteration 17 — load test

### Added
- `scripts/load_test.py` — drives N concurrent WebSocket recall
  sessions for a fixed duration, then reports throughput, client-
  and server-side latency percentiles, and cross-checks them
  against the `javis_recall_duration_seconds_count` Prometheus
  histogram. Default sweep: concurrency 1, 10, 50, 100 × 15 s.

### Verified
- Server sustains ~141 recalls/sec single-tenant (Mutex-serialised
  recall path).
- Latency scales linearly with concurrency: p99 11 ms at c=1,
  84 ms at c=10, 397 ms at c=50, 771 ms at c=100. No errors, no
  drops across 8 277 recalls.
- Memory footprint: 27 MiB idle → 35 MiB peak under sustained
  load → 34 MiB after cool-down. No leak.
- Documented bottleneck (`Arc<Mutex<Inner>>`) and three potential
  scaling paths in `notes/35-load-test.md`.

## Iteration 16 — end-to-end sanity

### Fixed
- Grafana dashboard panels referenced datasource `uid: "Prometheus"`
  but the auto-provisioning let Grafana hash a fresh UID, so every
  panel showed "Datasource not found" in the UI. Pinned
  `uid: prometheus` in the datasource provisioning and updated all
  five dashboard panels to match.

### Added
- `scripts/sanity_check.py` — reproducible end-to-end smoke test
  that drives `train` / `recall` / `ask` / parallel-recall flows
  over the WebSocket interface, then asserts on `/ready` deltas
  and `/metrics` counter values. Exits 0 on full pass; 1 on the
  first failed expectation. Targets `localhost:7777` by default,
  override via `JAVIS_HOST`.

### Verified live (notes/34)
- Train → recall → ask → 5 parallel recalls completed cleanly
  against `docker compose up`.
- Snapshot persistence in the live flow: train a new sentence,
  `docker compose restart javis-viz`, then recall both bootstrap
  and live-trained words still works (both ≥ 86 % token reduction).
- Lifetime token saving across the test run: 92.6 %.

## Iteration 15 — container & deploy

### Added
- Multi-stage `Dockerfile` (builder on `rust:1.86-bookworm`, runtime
  on `debian:bookworm-slim`). Final image runs as non-root user
  `javis` (uid 1000) with `tini` as PID 1 and a `curl /health`
  HEALTHCHECK. Layer-cache trick stubs the workspace so `cargo
  fetch` only re-runs on manifest changes.
- Persistent brain volume: `javis-data:/app/data` mount plus
  `--snapshot /app/data/brain.snapshot.json` arg in compose. Brain
  state survives `docker compose restart` (verified locally:
  29.5 MB snapshot, save → load round-trip preserves sentences/words).
- Optional `--secret id=hostca` build-time CA bundle for sandbox /
  TLS-intercepting-proxy environments. Declared in
  `docker-compose.yml` so a single `docker compose up --build`
  works without manual flag-fiddling.

### Fixed
- Dockerfile stub-source step now creates placeholder files for
  `[[bench]]` targets too — without them `cargo fetch` refuses to
  parse the manifest. Discovered when the iter-15 image build was
  first exercised end-to-end.
- `.dockerignore` keeps the build context lean (no `target/`,
  `.git/`, `notes/`, etc.).
- `docker-compose.yml` brings up three services: javis-viz,
  Prometheus 3.0 (scrapes `/metrics` every 15 s), Grafana 11 with
  anonymous-admin access for local demo.
- `deploy/prometheus.yml` — single scrape job for the
  `javis-viz:7777/metrics` endpoint.
- `deploy/grafana/provisioning/` auto-wires Prometheus as the
  default datasource and registers a dashboard provider.
- `deploy/grafana/dashboards/javis-overview.json` — 5-panel
  dashboard: brain sentence/word gauges, lifetime token-saving %,
  WS-sessions-per-action rate, and p95 latency timeseries for
  train/recall/ask.

### Changed
- `viz/src/main.rs` resolves the static-asset directory and bind
  address at runtime via `JAVIS_STATIC_DIR` / `JAVIS_BIND_ADDR` env
  vars. Defaults stay at the source-tree `static/` and `127.0.0.1:7777`
  so `cargo run` keeps working unchanged; the Docker image sets both
  to container-friendly values (`/app/static`, `0.0.0.0:7777`).

## Iteration 14 — performance benchmarks

### Added
- Three Criterion benchmark files: `crates/snn-core/benches/
  network_step.rs`, `crates/snn-core/benches/brain_step.rs`,
  `crates/encoders/benches/encode_decode.rs`. Seven separate bench
  functions covering passive vs. STDP-enabled `Network::step`,
  multi-region `Brain::step` (heap-backed inter-region delivery),
  and the encoder/decoder per-request hot paths.
- `criterion = "0.8"` as a dev-dependency on `snn-core` and
  `encoders`. `default-features = false` keeps the dep tree slim.
- New `benches` CI job runs `cargo bench --workspace --no-run` to
  catch API drift in the bench code without trying to extract real
  perf numbers from a shared GitHub runner.
- Baseline numbers documented in `notes/31-criterion-benchmarks.md`
  (e.g. `network_step_passive/1000` ≈ 3.2 µs,
  `decode_strict/vocab_1000` ≈ 253 µs on local x86_64).

## Iteration 13 — supply-chain hygiene (parts A + B + C + D)

### Added (part D — rustdoc warnings as errors)
- New `docs` CI job runs `cargo doc --workspace --no-deps
  --all-features` with `RUSTDOCFLAGS="-D warnings"`, so broken
  intra-doc links, invalid codeblock attributes, and malformed HTML
  in doc comments fail the build.
- Verified locally: existing 29 iterations of doc comments pass
  with both `-D warnings` and `-D rustdoc::all` — no fixes needed.

### Added (part C — Dependabot)
- `.github/dependabot.yml` tracks `cargo` (root + every crate) and
  `github-actions` (every workflow file) on a weekly schedule.
- Grouped updates: minor/patch bumps batched into one PR per
  ecosystem; the tracing stack and the tokio/tower/axum/hyper stack
  each have their own group so co-versioned crates stay in sync.
- Major bumps stay solo so they get individual review.
- PR commit prefixes `deps:` (cargo) / `ci:` (actions); open-PR
  caps at 5 and 3 respectively.

### Added (part B — MSRV)
- `[workspace.package].rust-version = "1.86"` — explicit MSRV
  contract. Every member `[package]` declares
  `rust-version.workspace = true` so the inheritance is literal.
- New `msrv` CI job in `.github/workflows/ci.yml` runs
  `cargo build --locked` and the full test suite against
  `dtolnay/rust-toolchain@1.86` on every push, so accidental use of
  1.87+-only features fails CI.

### Changed (part B)
- `snn-core/src/network.rs` — replaced `u64::is_multiple_of`
  (stabilised in 1.87) with the equivalent `% == 0`. Discovered
  during MSRV verification.

### Added (part A — cargo-deny)
- `deny.toml` — repository-root `cargo-deny` configuration. Four
  checks: advisories (RustSec), licenses (allow-list), bans
  (wildcard / duplicate detection), sources (only crates.io).
- License allow-list explicitly enumerates the ten permissive
  licenses our dep tree actually contains. `unused-allowed-license =
  "deny"` keeps the list tight.
- `.github/workflows/ci.yml` — new `deny` job runs `cargo-deny
  check` on every push via `EmbarkStudios/cargo-deny-action@v2`.

### Changed (part A)
- `[workspace.package]` declares `publish = false` and every member
  `[package]` adds `publish.workspace = true`. Required so
  `allow-wildcard-paths = true` applies — without it, intra-
  workspace path deps would be flagged as wildcards.

## Iteration 12 — operational hardening (parts A + B + C)

### Added (part C — Prometheus metrics)
- `GET /metrics` — Prometheus exposition endpoint. Returns the global
  recorder snapshot in `text/plain; version=0.0.4` format.
- `viz::metrics::init()` — idempotent, installs a `metrics-exporter-
  prometheus` recorder once per process, configured with histogram
  buckets covering our 5 ms – 30 s operation range.
- Counters: `javis_ws_sessions_total{action}`,
  `javis_recall_tokens_rag_total`, `javis_recall_tokens_javis_total`
  (the difference is total token saving over server lifetime).
- Histograms: `javis_train_duration_seconds`,
  `javis_recall_duration_seconds`, `javis_ask_duration_seconds{real}`,
  `javis_snapshot_duration_seconds{op}`.
- Gauges: `javis_brain_sentences`, `javis_brain_words`.
- Three new tests in `crates/viz/tests/metrics_endpoint.rs`.

### Added (part B — health/readiness probes)
- `GET /health` — liveness probe. Returns `200 ok` as long as the
  HTTP runtime can answer. Cheap enough for sub-second probe intervals.
- `GET /ready` — readiness probe. Returns `200` plus a JSON body
  with `status`, `sentences`, `words`, and `llm` mode (real / mock).
  Both probes are registered on `router` and `router_no_static`.
- Three new tests in `crates/viz/tests/health.rs` driving the router
  via `tower::ServiceExt::oneshot` (no real TCP needed).

### Added (part A — structured logging)
- `tracing` + `tracing-subscriber` for structured logging. Subscriber
  in `viz::main` honours `RUST_LOG` for level/target filtering and
  `JAVIS_LOG_FORMAT=json` to switch from human-readable to JSON output
  (for log aggregators like Loki / ELK).
- Per-WebSocket-session spans with monotonic `session` id and `action`
  field, so concurrent client logs can be disentangled.
- Structured fields on every state-mutating operation:
  `train completed` / `recall completed` / `ask completed` /
  `snapshot saved|loaded` / `brain reset` all carry timing
  (`elapsed_ms`) and outcome counters.

### Changed
- `println!`/`eprintln!` in the production code paths (binary +
  library + `llm` crate fallback) replaced by `tracing` macros at the
  appropriate level. Examples and tests keep their plain stdout output.

## Iteration 11 — production polish

### Added
- GitHub Actions CI workflow (`.github/workflows/ci.yml`) that runs
  `cargo fmt --check`, `cargo clippy -D warnings`, the full release
  test suite, and the doc-test suite on every push and pull request.
- Three doc-tests on the public API (`snn-core` quick-start, two
  `encoders` quick-start examples). They run as part of the standard
  `cargo test --doc` workflow, so the documentation cannot drift
  out of sync with the code.
- `crates/eval/examples/hello_javis.rs` — minimal end-to-end demo
  that trains on the wiki corpus, queries every topic, prints
  the RAG-vs-Javis token comparison. No external dependencies.

## Iteration 10 — heap queue, AMPA/NMDA/GABA channels, zero lints

### Added
- `PendingQueue` — `BinaryHeap`-backed min-heap on arrival time
  with a sequence-tiebreak for FIFO determinism. Replaces the
  `Vec<PendingEvent>` field on `Brain`.
- `SynapseKind { Ampa, Nmda, Gaba }` plus per-network
  `tau_nmda_ms` / `tau_gaba_ms`. Lazy NMDA/GABA buffers — AMPA-only
  networks pay no extra cost.
- `Network::set_synaptic_taus(ampa, nmda, gaba)` setter with
  positive-finite validation.

### Changed
- All `let mut x = Foo::default(); x.field = ...` test-code patterns
  rewritten as struct-init with `..Foo::default()`.
- `len() > 0` → `!is_empty()`, idiomatic slice iteration.

### Result
- 91/91 tests passing, **zero clippy warnings** workspace-wide.
- See `notes/22-heap-channels-lints.md`.

## Iteration 9 — architecture hardening

### Added
- Bounds-checked `Network::connect` and `Brain::connect` with clear
  panic messages naming the bad value.
- Per-network `tau_syn_ms` configurable via `Network::set_tau_syn_ms`.

### Changed
- Removed dead `Synapse.tau_syn` field (was set, never read; the
  decay loop used a hardcoded 5.0).

### Result
- 79/79 tests passing including 16 hardening tests.
- See `notes/21-architektur-haertung.md`.

## Iteration 8 — bio-inspired optimisations

### Added
- BTSP-style soft bounds for STDP (`StdpParams.soft_bounds: bool`).
- `FingerprintMode::Contextual` — engrams captured during training
  co-activity, not by isolated re-stimulation post-training.
- `EngramDictionary::decode_top(active, k)`.

### Result
- 63/63 tests passing including 5 associative-recall tests and 3 BTSP
  tests. See `notes/19-zwei-decode-modi.md` and
  `notes/20-bio-optimierungen.md`.

## Earlier iterations (0–7)

| Iteration | Topic | Note |
| ---: | --- | --- |
| 0–1 | snn-core baseline (LIF, STDP) | `notes/00`–`01` |
| 2 | Assembly formation + throughput | `notes/02` |
| 3 | E/I balance + sparse adjacency | `notes/03` |
| 4 | Multi-region AER | `notes/04` |
| 5 | Encoder stub | `notes/05` |
| 6 | Pattern completion | `notes/06` |
| 7 | Homeostatic scaling | `notes/07`, `notes/08` |

For pre-Iteration-8 details, see the corresponding research notes.

## Format

Loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Categories used per release: `Added`, `Changed`, `Removed`, `Result`.
