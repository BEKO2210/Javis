# Iter 65 — Perforant Path Robustness (Axis C value=0.3 at 8 seeds)

**Status: ENTRY (pre-registration, no measurements yet).**
**Hypothesis, locked seed set, locked acceptance matrix, and
supplementary measurement plan are committed in this file
*before* any run is executed. They may not be relaxed
post-hoc.**

> **Core sentence.** Axis C value=0.3 is the only identified
> mechanism; iter-65 tests whether it survives 8 seeds.

## Why iter-65

iter-64 ran the three pre-registered diagnostic axes
(`dg_to_r2_weight`, `r2_p_connect`, `direct_r1r2_weight_scale`)
isolated, each at 4 seeds × 16 ep smoke + (where indicated)
4 seeds × 32 ep full. Verdict per the locked acceptance
matrix:

| Axis | Verdict | Mechanism contribution |
| :--- | :-: | :--- |
| A (`dg_to_r2_weight`) | β/β/α-unstable/β | None — narrow window at iter-46 default; weak/strong both kill the signal |
| B (`r2_p_connect`) | β/α-unstable/β | None — narrow window at iter-46 default; sparse/dense both lock plasticity |
| C (`direct_r1r2_weight_scale`) | α/β/α-persistent/β | **value=0.3 — perforant-path sweet-spot, persistent across smoke and full** |

Per the iter-64 ENTRY's locked iter-65 fork ("exactly one
axis (α): deepen at 8 seeds × 32 ep at the most promising
value, with paired-t power computation. CA3/CA1 still
deferred"), iter-65 is the **robustness check** of axis C
value=0.3 at 8 seeds.

Axis C value=0.3 was measured at 4 seeds in iter-64:

```text
seed=42 smoke +0.0430  full +0.0215   ✓ persistent positive
seed=7  smoke +0.0371  full +0.0215   ✓ persistent positive (wakes up)
seed=13 smoke +0.0234  full +0.0508   ✓ persistent positive (doubles at full)
seed=99 smoke −0.0273  full −0.0283   ✗ deterministic negative outlier

aggregate full: μ_trained=0.0405, μ_untrained=0.0242,
                Δ̄=+0.0164, σ_Δ=0.0328, n_pos=3/4, t(3)=+0.996
classification: α
```

iter-65 doubles the seed count to surface whether seed=99 is
a 1-in-4 anomaly (≈ 25 % implied failure rate) or a 1-in-8
≈ 12 % rate. The acceptance matrix below distinguishes
sample artefact (Reject) from robust mechanism (Confirm) from
real-but-below-threshold (Partial).

## Pre-registered hypothesis

> On the iter-63-baseline brain configuration *with* axis C
> value=0.3 (perforant-path scale = 0.3 alongside DG mossy-
> fibre at full weight 1.0), 32 epochs of full plasticity
> produce a measurable cue → target learning signal on
> `target_top3_overlap` such that
>
> Δ̄ > 0 AND n_pos ≥ 6/8 AND paired t(7) > 0
>
> with the magnitude either at-or-above the iter-63-locked
> threshold (Δ̄ ≥ 0.0621 on 8/8 — Confirm) or robustly
> directional but below threshold (Partial), and not
> collapsing across seeds (Reject).

This hypothesis is **strictly weaker** than the iter-63
trained-main-run hypothesis (which required all 4 seeds to
clear 0.0621). iter-65 is the *robustness* check; iter-63's
threshold is preserved as a documented reference but
iter-65's primary acceptance is direction stability across
seeds, not threshold magnitude.

## Locked seed set (no relaxation)

iter-65 uses **exactly these 8 seeds, in this order**:

```text
seeds = 42, 7, 13, 99, 1, 2, 3, 4
```

The first four are the iter-53 / iter-63 / iter-64 locked
set. The last four are sequential u64 values chosen for
*maximally low* RNG-pollution from prior runs (seed=1, 2, 3,
4 are arithmetically simple, with no special meaning that
would make them a hidden-cherry-pick).

The seed list must be the same in trained and untrained runs
(paired-seed invariant; enforced by `run_axis_sweep`'s
`UntrainedCacheKey` infrastructure and the
`render_axis_sweep` paired-t calculation).

## Locked configuration (no parameter change)

```text
axis              = direct_r1r2_weight_scale
value             = 0.3                               (single value, not a sweep)
seeds             = 42, 7, 13, 99, 1, 2, 3, 4
phase             = full (32 epochs)
vocab             = 64
DG bridge         = on (dg_size=4000, dg_k=80, fanout=30, weight=1.0)
recall_mode_eval  = on (--plasticity-off-during-eval)
decorrelated_init = NOT on
                    (axis-sweep bench does not pass --decorrelated-init
                     by default; this matches iter-64 axis C smoke + full
                     measurements, so the iter-65 result is comparable
                     to the iter-64 axis C row exactly)
target_clamp      = 250 (default)
teacher_ms        = 40 (default)
teacher_forcing   = NOT on by default
                    (matches iter-64; cross-axis comparability)
```

**Important sanity:** the iter-64 axis C measurements at
value=0.3 used the *axis-sweep CLI default config* —
`--decorrelated-init` and `--teacher-forcing` were NOT
passed. iter-65 reproduces that exactly so the 8-seed sample
is paired with the iter-64 4-seed sample at the same
configuration. *If* a future iter-66 wants to test under
`--decorrelated-init + --teacher-forcing` (the iter-63
trained-main-run config), that's a separate iteration with
its own pre-registration.

## Run command (locked)

```sh
cargo run --release -p eval --example reward_benchmark -- \
  --axis-sweep direct-r1r2-weight-scale \
  --values 0.3 \
  --seeds 42,7,13,99,1,2,3,4 \
  --axis-sweep-phase full \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval
```

Wallclock estimate: ~4 h on local hardware (8 trained + 8
untrained runs at 32 ep; cache hit on baseline value=0.0
not applicable — this is value=0.3, not the baseline).

## Locked acceptance matrix

| Outcome | Per-seed pattern | Aggregate | iter-66 entry |
| --- | --- | --- | --- |
| **(A) Confirm** | Δ ≥ 0.0621 on **8/8** seeds | paired t(7) > 1.895 (one-sided p < 0.05) | iter-66 = downstream architecture (CA3/CA1 split or analogue) on a verified mechanism |
| **(B) Partial** | Δ̄ > 0 on **≥ 6/8** seeds | paired t(7) > 0 (direction matters; p-value not required) | iter-66 = parameter co-tuning around value=0.3 (e.g. dg_to_r2_weight × direct scale 2 × 2 grid at value=0.3 vicinity, OR scan 0.2 / 0.3 / 0.4 / 0.5 to find the optimum) |
| **(C) Reject** | Δ̄ ≤ 0 OR n_pos ≤ 4/8 | paired t(7) ≤ 0 OR p ≥ 0.5 | iter-66 = structural binding question — re-open the target SDR encoding, the prediction_top3_before_teacher metric, or the CA3/CA1 pivot as a last resort |

Edge cases collapse to (B) by default — same hygiene as the
iter-64 matrix collapsed unclassified to (β).

## Pre-registered measurement plan

Primary: `target_top3_overlap` per seed, paired with cached
untrained baseline at the same configuration. Renderer
emits the standard axis-sweep table with
`(μ_untrained, μ_trained, Δ̄, σ_Δ, n_pos, n_pass(0.0621),
t(df=7), classification)`.

Reported additionally in the iter-65 results commit:

- per-seed `untrained` and `trained` values
- per-seed `Δ`
- `n_pos / 8`
- `n_above_threshold / 8` (= `n_pass(0.0621)`)
- `mean ± SE` where `SE = σ_Δ / √8`
- paired `t(7)` value

Supplementary (separate run, same 8 seeds, same value=0.3,
issued after the main axis-sweep completes):

- **same-cue mean** (jaccard-bench, recall-mode invariant
  reading): does the trained engram remain stable under
  read-only recall, or does eval-time plasticity drift it?
  iter-62 locked same-cue = 1.000 on 4/4 seeds at recall-
  mode; iter-65 confirms across 8 seeds at value=0.3.
- **cross-cue separation**: jaccard cross-cue mean. iter-60
  locked the geometric floor at ≈ 0.026 with DG bridge;
  iter-65 sanity-checks that adding the perforant path at
  value=0.3 does not break separation.
- **eval-drift L2**: pre-eval vs post-eval L2 norms of R2
  recurrent weights. The recall-mode invariant asserts
  bit-identity (panic on violation) — already enforced by
  `run_target_overlap_one_seed` in untrained mode and by
  `run_jaccard_arm` in recall_mode_active mode for the
  trained arm. iter-65 reports whether the assert held on
  all 8 seeds (Pass) or fired (which would block the run
  before reaching the renderer).

The supplementary jaccard-bench command:

```sh
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-bench \
  --seeds 42,7,13,99,1,2,3,4 \
  --epochs 32 \
  --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval \
  --direct-r1r2-weight-scale 0.3
```

(Wallclock ≈ 30–40 min additional.)

## Methodological commitments (locked, not relaxable)

1. **No new architecture.** No CA3/CA1 split, no perforant-
   path bypass at SDR layer, no target-side restructuring.
2. **No new metric.** `target_top3_overlap` (mean
   `top3_accuracy` across epochs) only.
3. **No threshold shift.** iter-63's 0.0621 threshold is
   the documented reference; iter-65's primary acceptance is
   direction stability (n_pos / t-sign), the 0.0621 figure
   is in the table for reference only.
4. **No DG / R2 / perforant parameter change** beyond the
   locked `direct_r1r2_weight_scale = 0.3`. All other
   parameters at iter-46 defaults.
5. **No parallel axes.** Only axis C value=0.3, single
   point, robustness across seeds.
6. **No goalpost shift on seed=99.** If seed=99 is one of
   the 8 negatives, that contributes to the (C) Reject
   classification. Edge cases collapse to (B), not (A).
7. **Pre-flight verification mandatory:** before the main
   run, `grep -c "build_benchmark_brain\|run_axis_sweep"
   crates/eval/src/reward_bench.rs` must return ≥ 30
   (silent-wiring-gap protection inherited from iter-63
   incident).
8. **Explicit rebuild before run.** `cargo build --release`
   first, then `cargo run --release` — no implicit rebuild.

## What this commit is NOT

- Not a measurement. No data.
- Not iter-66 spec. iter-66 is conditional on iter-65's
  classification (A / B / C) and will be specified in its
  own pre-registration after the iter-65 verdict.
- Not a re-litigation of iter-64's verdict. Axis A and B
  are out as candidates; iter-65 does not re-test them.

## Verdict — (C) Reject

The 4-seed (3/4 = 75 %) positive result from iter-64 axis C
value=0.3 was a sample-frequency artefact of a true ~50 %
success-rate distribution. With 8 seeds the per-axis pattern
collapses to chance level on the locked acceptance matrix.

### Per-seed table (8 seeds × 32 epochs at value=0.3)

| seed | untrained | trained | Δ | direction | note |
| ---: | ---: | ---: | ---: | :-: | :--- |
| 42 | 0.0205 | 0.0420 | +0.0215 | + | iter-64 reproduced bit-identical |
| 7  | 0.0000 | 0.0215 | +0.0215 | + | iter-64 reproduced bit-identical |
| 13 | 0.0332 | 0.0840 | +0.0508 | + | iter-64 reproduced bit-identical |
| 99 | 0.0430 | 0.0146 | **−0.0283** | − | deterministic outlier (iter-64 bit-identical) |
| **1** | 0.0391 | 0.0371 | **−0.0020** | − | NEW seed: marginal-negative |
| **2** | 0.0068 | 0.0176 | +0.0107 | + | NEW seed: positive |
| **3** | 0.0625 | 0.0586 | **−0.0039** | − | NEW seed: marginal-negative |
| **4** | 0.0566 | 0.0410 | **−0.0156** | − | NEW seed: clearly negative |

### Renderer aggregate

```text
| value | μ_untrained | μ_trained | Δ̄        | σ_Δ    | n_pos | n_pass(0.0621) | t(df=7) | classification |
|   0.3 | 0.0327      | 0.0396    | +0.0068  | 0.0248 | 4/8   | 0/8            | +0.779  | β              |
```

**Aggregate stats** (per the iter-65 ENTRY measurement plan):

- `Δ̄ = +0.0068`
- `σ_Δ = 0.0248`
- `SE = σ_Δ / √8 = 0.00877`
- `n_pos = 4/8` (chance)
- `n_above_threshold(0.0621) = 0/8`
- `t(7) = +0.779`
- two-sided 95 % CI on Δ̄: `±0.0207` — straddles zero

### Locked acceptance matrix applied

| Outcome | Per-seed pattern | Aggregate | Match? |
| --- | --- | --- | :-: |
| **(A) Confirm** | Δ ≥ 0.0621 on 8/8 AND t(7) > 1.895 | n/a | ❌ (n_above_threshold = 0/8) |
| **(B) Partial** | Δ̄ > 0 AND **n_pos ≥ 6/8** AND t(7) > 0 | direction matters | ❌ (n_pos = 4/8 < 6/8) |
| **(C) Reject** | Δ̄ ≤ 0 OR **n_pos ≤ 4/8** | n/a | ✓ (n_pos = 4/8 ≤ 4/8) |

**Verdict: Branch (C) Reject — locked.** Pre-registration
discipline preserved: the iter-65 ENTRY committed n_pos ≥ 6/8
as the (B) Partial floor and n_pos ≤ 4/8 as the (C) Reject
trigger. n_pos = 4/8 satisfies (C) by the *boundary* rule.
There is no goalpost-shift available — the matrix was
committed before any iter-65 trained-arm data was peeked at.

### Honest reading

Three findings stand out, all consistent with the
iter-66-deep-research literature interpretation
(`notes/66-deep-research-cue-target-binding.md`):

1. **Determinism is not the problem.** Seeds 42, 7, 13, 99
   reproduced their iter-64 axis C value=0.3 full-phase Δ
   values bit-for-bit (cache pre-seed + RNG-determinism).
   The architecture is doing what the architecture does;
   what changes between iter-64 and iter-65 is *which* 4
   seeds are in the sample, and the answer is sample-
   frequency-dependent.

2. **The 4 new seeds split 1:3.** seed=2 is the only new
   positive (+0.0107). Seeds 1, 3, 4 are all in the
   marginal-negative-to-clearly-negative range. Combined
   with the deterministic seed=99 outlier, the 8-seed pool
   sits at exactly 4 positive / 4 negative — chance-level on
   a binary direction test.

3. **σ_Δ = 0.0248 ≈ σ_untrained_iter63 = 0.0213.** The
   trained-vs-untrained spread is at the same magnitude as
   the untrained-arm seed-to-seed variance. The
   plasticity-driven Δ̄ (= +0.0068) is approximately one-third
   of one σ_untrained — well inside the noise floor that
   iter-63's calibration locked.

### What this verdict means structurally

Iter-65 falsifies the hypothesis that *the current Javis
architecture* (R1/DG/R2 with the iter-46 plasticity stack
plus the axis C value=0.3 perforant-path re-introduction)
can write a robust cue → target signal that the iter-44/45
decoder can read. It does NOT falsify "the Javis architecture
can never solve binding"; it does NOT falsify "the
plasticity stack is broken"; it does NOT falsify "DG
separation works" (that remains iter-60-locked) or "recall
mode works" (iter-62-locked).

What it falsifies is the architectural assumption that the
heteroassociative cue → target binding is a property of R2
(or the R2 + perforant configuration) under STDP-stack
training. Per the deep-research literature scan in
`notes/66-deep-research-cue-target-binding.md`, this is
exactly the architectural mistake O'Reilly & McClelland
(1994) explicitly warned against — making one structure
carry both autoassociative completion *and* heteroassociative
binding. The iter-65 falsification is the empirical
signature of that mistake on Javis's specific configuration.

## iter-66 entry — locked by (C) Reject

Per the iter-65 ENTRY locked fork:

> (C) Reject → iter-66 = structural question — re-open the
> target SDR encoding, the prediction_top3_before_teacher
> metric, or the CA3/CA1 pivot as a last resort.

The deep-research literature scan (28 peer-reviewed sources,
`notes/66-deep-research-cue-target-binding.md`) collapses
the (C) fork's three sub-options to a single recommendation:

> **iter-66 = CA3/CA1 split.** New CA1-equivalent layer
> (Mechanism M1) with target-presence-gated three-factor
> R-STDP on a R2 → C1 projection. Primary metric:
> `c1_target_top3_overlap` (mean across epochs of top3
> decoder accuracy on C1 fingerprints, not R2). Same 8 seeds
> for cross-iter comparability.

The deep research's "Hard Recommendation" section makes this
the unique "Do this next" item, with explicit literature
support from Marr (1971), Treves & Rolls (1994), O'Reilly &
McClelland (1994), Norman & O'Reilly (2003), and Schapiro
et al. (2017).

The seed=99 outlier question is *not* answered by iter-65
(it remains deterministically negative across both iter-64
4-seed and iter-65 8-seed runs at the same configuration).
It will be re-asked under iter-66's new C1 readout: if M1's
binding mechanism resolves the cue → target mapping
robustly, the seed=99 specific failure mode at value=0.3 is
either resolved (because the binding no longer depends on
the perforant-injection-into-CA3 confound) or surfaced as a
seed-specific R2 attractor pathology that iter-67+ would
need to address.

iter-66 ENTRY pre-registration to follow in
`notes/66-cu-target-binding-implementation.md` (separate
file from the deep-research scan).

## Supplementary measurement plan — pending

The iter-65 ENTRY locked three additional measurements on
the same 8 seeds at value=0.3:

- same-cue mean (recall stability)
- cross-cue separation (sanity check)
- eval-drift L2 (recall-mode invariant assert)

Run command:

```sh
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-bench \
  --seeds 42,7,13,99,1,2,3,4 \
  --epochs 32 \
  --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval \
  --direct-r1r2-weight-scale 0.3
```

Currently running in background; partial data so far at
seeds 42, 7:

| seed | arm | same-cue | cross-cue | eval-drift L2 |
| ---: | :--- | ---: | ---: | :-: |
| 42 | untrained | 1.000 ± 0.000 | 0.030 ± 0.087 | n/a (no plasticity) |
| 42 | trained (recall-mode) | 1.000 ± 0.000 | 0.025 ± 0.079 | bit-identical ✓ |
| 7  | untrained | 1.000 ± 0.000 | 0.029 ± 0.089 | n/a |

These confirm what iter-60 / iter-62 already established:
DG separation + recall-mode stability are intact. The
supplementary table will be appended to this note as a
follow-up commit when the run completes.

The supplementary measurements do *not* change the (C)
Reject verdict on the primary `target_top3_overlap` metric;
they are sanity-checks on the architecture's other
properties.
