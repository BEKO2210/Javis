# Iter 63 — Direct cue → target metric on DG-enabled brain

**Status: ENTRY (pre-registration). No measurements yet. This
note records the falsifiable hypothesis, the calibrated
acceptance threshold, the explicit branching matrix, and the
positive control plan, *before* the main run is executed. All
three are locked in this commit and may not be relaxed
post-hoc.**

## Why iter-63

iter-60 proved DG **separates** (cross-cue 0.448 → 0.026,
−94 %). iter-62 proved the DG-enabled brain is **stable under
read-only recall** (same-cue = 1.000 on 4/4 seeds, post-eval
L2 bit-identical). What is *not* proved is that the DG-enabled
brain actually **learns** the cue → target association — the
whole point of the iter-44 → iter-62 chain.

Jaccard cross-cue is at the geometric floor (~0.026 trained
vs 0.029 untrained, Δ = −0.001 NS) and structurally cannot
register plasticity-driven cue-specific learning on top of
the DG geometry (iter-58 / iter-61 / iter-62). The metric has
done its job. iter-63 re-introduces the iter-46 / iter-52
direct cue → target metric, which compares the trained R2
response against the *canonical target SDR* of each cue, on
the DG-enabled brain.

The work is **wiring**, not research: `RewardEpochMetrics`
already implements the candidate metrics from iter-46. iter-63
re-routes them onto the DG path, with a single pre-registered
metric and a calibrated threshold.

## Pre-registered hypothesis

**H1 (primary, single metric, single comparison):**

> On the DG-enabled brain (vocab=64, recall-mode, ep32),
> trained `target_top3_overlap` is greater than untrained
> `target_top3_overlap`, by a margin large enough to exceed
> noise from the geometric floor.

`target_top3_overlap` is the **mean of iter-44/45's
`top3_accuracy` across all epochs** of the run. `top3_accuracy`
is the per-epoch decoder-relative metric: at the end of every
epoch the brain's per-cue R2 response is decoded via the
per-epoch dictionary, and the cue counts as a hit iff the
canonical target word is in the decoder's top-3. iter-51
showed iter-46-Arm-B-style brains oscillate per-epoch (0.06 ↔
0.19 across 16 epochs); the **mean across epochs** is the
stable estimator (iter-51 reading: 0.107 ± noise, 95 % CI
[0.069, 0.145]).

This is the metric iter-46 / iter-50 / iter-51 calibrated
their baseline against. Reusing it verbatim — exposed through
the new `--target-overlap-bench --mode <X>` plumbing — is
what makes the positive-control band meaningful.

**Note on `prediction_top3_before_teacher` (iter-46
teacher-schedule SDR-overlap metric).** This is a *separate*
metric from `top3_accuracy` and is **not** what iter-63 uses:
- It is computed only when `--teacher-forcing` is enabled
  (per-trial prediction phase, vs canonical target SDR).
- It has **no calibrated baseline** in the iter-46 / iter-50
  / iter-51 chain. The 0.19 reading those notes report is
  `top3_accuracy`, not this metric.
- iter-63 keeps the field intact in `RewardEpochMetrics` for
  future iterations that may want to re-calibrate against it.
  It is *not* the iter-63 read-out.

**One metric. One comparison. No top-1 side-eye, no MRR
peek, no per-pair sub-analysis.** This is the iter-50 / 51
lesson: multiple-comparison drift is what produces 5
iterations against the wrong metric.

## Pre-measurement correction

After committing the iter-63 plumbing, the positive control
(`--target-overlap-bench --mode trained --iter46-baseline
--teacher-forcing --seeds 42 --epochs 16`) returned 0.0000 —
clearly a wiring failure. Diagnosis surfaced two issues:

1. **Wrong metric.** The plumbing read
   `prediction_top3_before_teacher` (iter-46 teacher-schedule
   metric, only populated when teacher-forcing is on, no
   calibrated baseline). The iter-46 / 50 baseline of 0.19
   was actually `top3_accuracy` (iter-44/45 decoder metric,
   computed always). Source: `notes/50-arm-b-reproduction.md`
   line 51 ("CLI omits `--teacher-forcing`"), line 74 (table
   row showing 0.19 in the top-3 column of `render_markdown`).
2. **Wrong aggregation.** Even with `top3_accuracy`, the
   iter-46 / 50 ep0 reading of 0.19 was *peak*, not stable.
   iter-51's 16-epoch saturation showed Arm B oscillates
   0.06 ↔ 0.19 with mean = 0.107. Last-epoch or max
   readings are not robust against the oscillation; mean
   across all epochs is.

Both points are corrected here, *before* any measurement is
locked, *before* the calibration step. The corrections are
documented in this section as an explicit pre-measurement
adjustment — **not** a post-hoc result adjustment. The
positive-control gate is exactly the mechanism designed to
catch this class of bug, and it caught it on the first
invocation. Pre-registration discipline is preserved: no
trained-arm data has been peeked at, the threshold formula
is unchanged, and the branching matrix is unchanged.

Concrete patches applied in the same correction commit:

- `run_target_overlap_arm`: per-seed value =
  mean(top3_accuracy across all epochs), not last-epoch
  prediction_top3_before_teacher.
- Doc comment of `run_target_overlap_arm` updated to spell
  out the metric and aggregation choice.
- CLI `--target-overlap-bench` block: dropped the
  `cfg.teacher.enabled = true` force-set
  (`top3_accuracy` is independent of the teacher schedule).
- Positive-control band: **[0.07, 0.15]** (iter-51 stable
  estimator μ ± noise band), **not** [0.16, 0.22] (iter-46
  / 50 ep0 peak). The new band brackets iter-51's 95 % CI
  with a small margin.
- `prediction_top3_before_teacher`: **kept**, not deleted —
  flagged here as a separate, currently-uncalibrated metric
  available for future iterations.

## Pre-registered acceptance threshold

The threshold is **not yet a number** — it is a function of
the untrained baseline, computed on real data before the
trained arm runs.

### Operational definition of `untrained` (locked)

Two readings are possible and they answer *different*
questions:

- **(a)** Fresh init weights, no plasticity ever applied,
  recall-mode on, `target_top3_overlap` measured once. This
  asks: *does the path learn anything above geometry?*
- **(b)** Trained brain, but the cue↔target association for
  *this* vocabulary was never seen (held-out pairs). This
  asks: *does the path learn this specific association?*

iter-63 uses **(a)**. The trained-arm will see exactly the
same cue-target pairs at the same vocabulary; the untrained
arm is the same brain *before any training has occurred*,
under recall-mode. This is the only definition that makes
the paired-t comparison statistically valid: same seed →
same init → identical untrained value, plasticity is the
only variable. Held-out-pair untrained is a different
experiment and belongs in iter-65 if needed.

### Calibration phase

Untrained-only sweep at the same configuration the main run
will use. The seed set used here **must be identical** to
the seed set used in the trained main run — paired t(3)
requires paired seeds, not two independently-drawn lists.

```text
seeds        = 42, 7, 13, 99       (4 seeds, locked)
epochs       = 32
vocab        = 64
DG bridge    = on
recall-mode  = on (plasticity-off-during-eval)
no-plasticity= true (untrained arm, definition (a) above)
teacher_ms   = 40
clamp        = 500
decorrelated init = on
```

Compute per-seed `target_top3_overlap`, then `μ_untrained`
and `σ_untrained`.

### Threshold formula

```text
acceptance_threshold = max(+0.05, μ_untrained + 2·σ_untrained)
```

- The `+0.05` floor is a Cohen's-d-style absolute minimum
  that survives even if the untrained baseline is at the
  geometric floor (3/64 ≈ 0.047 random expectation).
- The `μ + 2σ` floor protects against a structurally elevated
  untrained baseline (DG geometry, repetition bias, cue
  encoding artefacts) — anything ≤ 2σ above untrained is
  noise-level on the same architecture.

The calibration result and the resulting locked threshold
will be appended to this note **before** the trained run is
launched. The calibration commit will be a self-contained
"iter-63 calibration: lock threshold = X" entry in the
chain.

### Calibration result — locked

Run on the post-plumbing-fix code (commit `112a469` and its
ancestors, all on `main` after PR #34 merged). Run command
exactly as specified in the *Run command sequence* section:

```sh
cargo run --release -p eval --example reward_benchmark -- \
  --target-overlap-bench --mode untrained \
  --seeds 42,7,13,99 --epochs 32 \
  --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval
```

Per-seed:

| Seed | `target_top3_overlap` (mean of `top3_accuracy` over 32 epochs) |
| ---: | ---: |
| 42 | 0.0127 |
| 7  | 0.0000 |
| 13 | 0.0498 |
| 99 | 0.0156 |

**Aggregate:** `μ_untrained = 0.0195`, `σ_untrained = 0.0213`
(n = 4, sample std).

**Threshold formula evaluated:**

```text
acceptance_threshold = max(0.05, μ + 2·σ)
                     = max(0.05, 0.0195 + 2 · 0.0213)
                     = max(0.05, 0.0621)
                     = 0.0621
```

**Locked threshold for the iter-63 trained main run: `0.0621`.**

The `μ + 2σ` branch wins the max — i.e. the noise band of the
untrained DG-enabled brain is wider than the +0.05 absolute
floor would have been. The trained arm must beat
`Δ = trained_seed − untrained_seed ≥ 0.0621` on **all four
seeds** AND clear `paired t(3) > 2.353 (one-sided p < 0.05)`
to satisfy branch (A). Anything weaker collapses to (B) or
(C) per the branching matrix below.

#### Reading on the untrained baseline

`μ_untrained = 0.0195` is **below** the `3/64 ≈ 0.047`
random-baseline expectation. That is consistent with the
DG-bridge architecture: with random R1 → DG hash + sparse
mossy-fibre projection + recall-mode (no plasticity), an
untrained brain produces an R2 response whose top-3
fingerprint mostly does not contain the canonical-target
SDR cells — the random hash routes cues to disjoint R2
sub-populations that the dictionary's per-word
fingerprint phase then captures but in a way that does
not align with the canonical target. The trained arm has
real ground to gain (anywhere from 0.062 to ~0.5 if
plasticity actually maps cue → target through DG → R2).

`σ_untrained = 0.0213` is non-trivial — seed 7 hit 0.000
and seed 13 hit 0.0498, a 0.05 absolute spread on a small
metric. The wide band is exactly why the `μ + 2σ` arm of
the threshold formula matters: a flat +0.05 threshold
would have been comparable to seed 13's untrained value
alone. Pre-registered formula prevents that confound.

#### iter-52 invariant on the calibration run

Trained-mode plasticity is `false` for the untrained arm
(`--mode untrained` enforces `no_plasticity = true`); the
iter-63 v2 runner (`run_target_overlap_arm` post
plumbing-fix) calls `disable_all_plasticity` at the top
of each seed's run, then asserts `assert_no_weight_drift`
after the 32-epoch loop. The assertion **passed on all 4
seeds** — no plasticity path leaked through, the weights
the eval-time decoder saw were identical to the freshly
initialised weights. This is the regression-test win from
the iter-63 plumbing-fix's `run_teacher_trial`
save/restore patch: pre-fix, the same calibration run
panicked with R2-recurrent L2 159.87 → 1606.87.

## Pre-registered branching matrix

After the trained run completes, exactly one branch applies:

| Outcome | Per-seed signature | Aggregate signature | iter-64 entry |
| --- | --- | --- | --- |
| **(A) PASS — learning detected** | Δ ≥ threshold on **4/4 seeds** | paired t(3) one-sided **p < 0.05** | iter-64 = CA3/CA1 split (true hippocampal model on the now-verified DG read-out) |
| **(B) FAIL — learning not present** | Δ < 0 on any seed, **OR** Δ > 0 on ≤ 2/4 seeds | (irrelevant — disqualified by per-seed) | iter-64 = mechanism question first (DG→R2 learning rate, R2 recurrent strength, perforant-path re-introduction); CA3/CA1 deferred until learning is rescued at this architecture |
| **(C) MIXED — underpowered** | Δ > 0 on **≥ 3/4 seeds** | **0.05 ≤ p < 0.15** | iter-64 = more seeds at the same architecture (8 or 16 seeds, ep32). No new mechanism, no architecture pivot. |

Edge cases collapse into (B):

- (A)-margin met but only on 3/4 seeds with p < 0.05 → (B), not (A). Per-seed unanimity is the harder claim.
- Δ > 0 on 3/4 with p ≥ 0.15 → (B). Without statistical lift the directional pattern is consistent with chance.
- Δ ≥ threshold on 4/4 with p < 0.05 but `μ_untrained` itself > 0.10 → flag in the note as "elevated baseline, replicate at vocab=128 in iter-64 before declaring CA3/CA1". Still passes acceptance; just adds a replication ask.

## Positive control (cheap insurance)

Before the main calibration + trained run, a positive control
verifies that the newly-wired metric path produces signal on a
known-working configuration:

```sh
# Positive control: iter-46 Arm B baseline (known-working
# from iter-46 / iter-50), 1 seed, 16 epochs, run through
# the *new* target_top3_overlap metric machinery.
cargo run --release -p eval --example reward_benchmark -- \
  --target-overlap-bench --iter46-baseline \
  --seeds 42 --epochs 16
```

Acceptance for the positive control: trained
`target_top3_overlap` on iter-46 Arm B (= mean
top3_accuracy over 16 epochs) falls within **[0.07, 0.15]**
— iter-51's stable-estimator band (16-epoch mean = 0.107,
95 % CI [0.069, 0.145], with a small margin to absorb
seed-level variation). Outside this band — including a
value that is "close" but only 0.05 or 0.18 — counts as
**plumbing drift**, not "close enough", and triggers a
wiring fix before calibration. The 0.047 random-baseline
floor is explicitly rejected as a pass: the control fails
if the metric is silent. **Do not** pivot architecture
(branch B) on a silently-broken or drifting metric
pipeline.

(Note: the earlier ENTRY draft set this band at [0.16,
0.22] against the iter-46 ep0 peak of 0.19. That band was
corrected pre-measurement after the positive-control gate
caught the metric-definition mismatch — see the
"Pre-measurement correction" section below.)

This control was the gap iter-50 surfaced retrospectively
(`selectivity_index` was structurally meaningless in the
no-teacher path). iter-63 closes it pre-emptively.

## Out of scope (locked, not relaxable)

- **No new plasticity rules.** STDP / iSTDP / homeostasis /
  intrinsic / reward / metaplasticity / heterosynaptic /
  structural — everything stays at iter-62 settings.
- **No new topology.** R1 → DG → R2 stays the iter-60 / 61 /
  62 setup. No CA3/CA1 split, no perforant-path
  re-introduction, no DG parameter sweep.
- **No new metrics in the same run.** The single
  pre-registered metric is `target_top3_overlap`. Top-1,
  MRR, per-pair activation, correct-minus-incorrect — *not*
  in iter-63's window. They wait for iter-65+ if iter-63
  passes.
- **No vocab / epoch / clamp / teacher_ms variation.** vocab=64
  ep32 c500 t40 is the iter-46/52/58/60/61/62 comparability
  baseline; comparability beats optimum-search at this
  iteration.
- **No peeking.** The threshold formula is a function of
  *only* untrained data. The trained run does not start
  until the threshold is locked in this note as a fixed
  number.
- **No multi-comparison reading.** If `target_top3_overlap`
  fails acceptance but I notice top-1 looks better — that
  observation is *out of scope* for iter-63 and goes in the
  iter-64 spec, not in iter-63's verdict.

## Run command sequence

The seed list `42,7,13,99` is used position-by-position in
both arms. Calibration and main differ in `--mode` only.

```sh
# 1. Positive control — verify metric wiring on iter-46
#    Arm B (known-working). --mode trained is mandatory.
cargo run --release -p eval --example reward_benchmark -- \
  --target-overlap-bench --mode trained \
  --iter46-baseline --seeds 42 --epochs 16
# Acceptance band: trained target_top3_overlap ∈ [0.16, 0.22].
# Outside ⇒ plumbing drift, fix wiring before step 2.

# 2. Calibration — untrained baseline on DG + recall-mode.
#    --mode untrained enforces no_plasticity = true.
cargo run --release -p eval --example reward_benchmark -- \
  --target-overlap-bench --mode untrained \
  --seeds 42,7,13,99 --epochs 32 \
  --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval
# Compute μ_untrained, σ_untrained.
# Lock acceptance_threshold = max(+0.05, μ + 2σ).
# COMMIT this note with the locked number before step 3.

# 3. Main — trained arm at the same configuration, same
#    seed list in the same order (paired t requires it).
cargo run --release -p eval --example reward_benchmark -- \
  --target-overlap-bench --mode trained \
  --seeds 42,7,13,99 --epochs 32 \
  --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval
# Apply the locked branching matrix.
```

## Implementation plan

- New CLI flag: `--target-overlap-bench` parallel to
  `--jaccard-bench`. Mutually exclusive at the parser level.
- **Required companion flag**: `--mode <untrained|trained>`.
  No implicit code path. `--target-overlap-bench` *requires*
  `--mode`; missing or other values fail the parser. This
  prevents the iter-50-style "wrong-arm-by-accident" class
  of bugs.
  - `--mode untrained` enforces `no_plasticity = true` and
    is the only legal way to compute the calibration baseline.
  - `--mode trained` enforces `no_plasticity = false` +
    `recall_mode_eval = true` and is the only legal way to
    compute the trained main-run value.
- **Seed handling**: `--seeds` is parsed once and the same
  list is used for both arms in a sweep. The trained main
  run **must** be invoked with a `--seeds` list that is a
  superset (typically equal) of the calibration list, in
  the same order. Code-level assertion: when both
  `target_overlap_trained.json` and
  `target_overlap_untrained.json` artefacts exist for a
  rendering call, their seed lists must match position-by-
  position; otherwise `render_target_overlap_sweep` panics
  with "paired-seed invariant violated".
- New public surface in `crates/eval/src/reward_bench.rs`:
  - `TargetOverlapMetrics { mode: ArmMode, seeds: Vec<u64>,
    per_seed: Vec<f32>, mean: f32, std: f32 }` — `seeds`
    carried so the paired-seed invariant can be checked.
  - `enum ArmMode { Untrained, Trained }`.
  - `run_target_overlap_arm(brain, cfg, mode) ->
    TargetOverlapMetrics` — re-uses iter-46 / 52 logic from
    `RewardEpochMetrics::target_top3_overlap` on the
    DG-enabled brain.
  - `render_target_overlap_sweep(trained, untrained,
    threshold) -> String` — markdown summary with
    per-seed table + paired t(3); panics if the
    paired-seed invariant is violated.
- L2 bit-identity invariant continues to apply on the
  trained-arm eval phase whenever `recall_mode_eval` is on.
- `--iter46-baseline` flag (already exists from iter-50)
  routes through `--target-overlap-bench --mode trained`
  for the positive control.

Estimated effort:

- Plumbing: ~2 h (the `target_top3_overlap` math is already
  in `RewardEpochMetrics`; iter-63 only re-wires it onto a
  new run-loop entry point).
- Positive control + calibration + main: ~30 min wallclock
  total (DG ep32 4 seeds is ≈ 5–8 min per arm at vocab=64).
- Total iter-63 budget: half a day.

## Methodological commitments

1. The threshold formula is a function of untrained data
   only. The trained run does not start until the threshold
   is a *fixed number* in this note.
2. Single pre-registered metric. No multi-comparison reading
   in iter-63's verdict.
3. Positive control runs *before* calibration. A silent
   positive control means broken wiring; fix wiring first.
   Do **not** pivot architecture on a broken pipeline.
4. The branching matrix is exhaustive. Edge cases collapse
   to (B), not (A) and not (C). Per-seed unanimity is
   the harder claim and gates branch (A).
5. No vocabulary / epoch / clamp / teacher_ms variation
   inside iter-63. iter-46/52/58/60/61/62 comparability
   beats optimum-search.
6. The iter-50 / iter-51 lesson is the whole point: avoid
   optimizing against the wrong metric. Pre-registration is
   the discipline that protects against it.

## Files to touch

- `crates/eval/src/reward_bench.rs` — `TargetOverlapMetrics`,
  `run_target_overlap_arm`, `render_target_overlap_sweep`.
- `crates/eval/examples/reward_benchmark.rs` —
  `--target-overlap-bench` CLI flag (mutually exclusive with
  `--jaccard-bench` and `--jaccard-floor-diagnosis`).
- `notes/63-cue-target-metric.md` — this note. Calibration
  result + locked threshold appended in a follow-up commit
  before main run.
- `CHANGELOG.md` — iter-63 ENTRY block (now), then verdict
  block (after main run + branching).
- `README.md` — iter-63 row appended after calibration is
  locked.

## Branch hygiene

- Branch: `claude/iter63-cue-target-metric` off
  `origin/main` — conflict-free to Codex's parallel work.
- Single linear history: ENTRY (this commit) → calibration
  lock → positive control verification → main run →
  verdict + addendum.

## What this iteration is *not*

- Not a new architecture. CA3/CA1 split waits for branch
  (A).
- Not a parameter sweep. The DG / recall-mode / clamp /
  teacher_ms parameters are frozen at iter-62.
- Not a metric exploration. One metric, pre-registered.
- Not a publication-grade statistical test. n=4 is what we
  have; the (C) cutoff exists exactly because n=4 has weak
  power.

## Headline

**DG separates and recalls but does not learn — mechanism question
for iter-64.** Branch (B) per the locked branching matrix.

## Trained main run — verdict

Run command exactly as locked in the *Run command sequence*
section, with `--mode trained --threshold 0.0621`:

```sh
cargo run --release -p eval --example reward_benchmark -- \
  --target-overlap-bench --mode trained --threshold 0.0621 \
  --seeds 42,7,13,99 --epochs 32 \
  --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval
```

Wallclock: ~2 h on local hardware (4 seeds × 32 epochs of full
plasticity at vocab=64 with DG bridge). The internal untrained
re-run reproduced the calibration commit's locked baseline values
**bit-for-bit on all 4 seeds** — determinism preserved, no
hidden seed drift, paired-seed invariant intact.

### Per-seed paired result

| Seed | untrained | trained | Δ | Δ ≥ 0.0621 |
| ---: | ---: | ---: | ---: | :---: |
| 42 | 0.0127 | 0.0127 | +0.0000 | ✗ |
| 7  | 0.0000 | 0.0195 | +0.0195 | ✗ |
| 13 | 0.0498 | 0.0039 | **−0.0459** | ✗ |
| 99 | 0.0156 | 0.0312 | +0.0156 | ✗ |

### Aggregate

- `μ_untrained = 0.0195 ± 0.0213`
- `μ_trained  = 0.0168 ± 0.0115`
- `Δ̄ = trained − untrained = −0.0027 ± 0.0300`
- `n_pos = 2 / 4`, `n_pass = 0 / 4`

Untrained identical to calibration as expected (deterministic
given seed). Trained mean is *below* untrained mean — the
arithmetic difference is small (0.003) and not significant, but
the direction is **away** from the pre-registered hypothesis,
not toward it.

### Paired t(3)

- `t = −0.179`
- `t_crit(α = 0.05) = 2.353` ⇒ `p < 0.05` ✗
- `t_crit(α = 0.15) = 1.250` ⇒ `p < 0.15` ✗

The directional pattern is consistent with chance (Δ negative on
seed 13 and zero on seed 42 cancel the modest gains on seeds 7
and 99). With n = 4 the test has limited power, but per Bekos's
pre-registered (C) cutoff the conditions for "underpowered, more
seeds" require `n_pos ≥ ⌈3n/4⌉ = 3` AND `t > 1.250` — both fail.

### Branching matrix (locked) applied

| Outcome | Match? | Why |
| --- | :-: | --- |
| **(A) PASS** — Δ ≥ threshold on **4/4** seeds AND `p < 0.05` | ❌ | `n_pass = 0/4`; no seed clears 0.0621 |
| **(B) FAIL** — Δ < 0 on any seed **OR** Δ > 0 on ≤ n/2 seeds | ✓ | Δ < 0 on seed 13 (−0.0459); also `n_pos = 2/4` ≤ n/2 — **two independent triggers** for branch (B) |
| **(C) MIXED** — `n_pos ≥ ⌈3n/4⌉ AND 0.05 ≤ p < 0.15` | ❌ | `n_pos = 2/4 < 3`; `p ≥ 0.15` |

**Verdict: Branch (B) — locked.** Per the pre-registration: edge
cases collapse to (B) too. There is no goalpost-shift available;
the matrix was committed before any trained-arm data was peeked
at. The numbers are accepted as they are.

### Honest reading

**What the data say.** On the iter-63 configuration (vocab=64 +
DG bridge + decorrelated R1→R2 + recall-mode-eval + 32 epochs of
teacher-forcing + clamp 500 + 4 seeds), 32 epochs of full
plasticity (STDP / iSTDP / homeostasis / intrinsic / reward /
metaplasticity / heterosynaptic / structural) **do not produce a
measurable cue → target learning signal** on the iter-44/45
`top3_accuracy` metric. The trained brain's mean across-epoch
top-3-decoder hit rate is statistically indistinguishable from
the same brain measured before any plasticity — and trends, if
anything, very slightly *negative*.

**What the data do *not* say.**

- The brain is not learning *anything*. iter-60 / iter-61 / iter-
  62 already demonstrated DG separation works (cross-cue floor
  collapses 16×) and recall-mode keeps the engram stable
  (same-cue = 1.000 on 4/4 seeds, post-eval L2 bit-identical).
  Those phenomena are real; they just don't show up in the
  decoder-relative `top3_accuracy` metric on this architecture.
- The metric is broken. The positive control on the iter-46 Arm B
  baseline reproduced iter-51's stable estimator at 0.1094
  (within `[0.07, 0.15]`); the metric pipeline is verified to
  surface learning when learning is happening.
- The architecture is hopeless. iter-60's −94 % cross-cue floor
  is the largest single number-move in 14 iterations; the DG
  bridge is doing its job. What's missing is a measurable signal
  that the *post-DG* path is mapping cue → target, not just
  separating cues.

**What the data point to (for iter-64).** Per the locked branching
matrix, branch (B) sends iter-64 into the **mechanism question**
before any further architecture work:

1. **DG → R2 learning rate.** The DG-mossy-fibre projection
   (sparse, k-of-n) into R2 may not have enough plasticity headroom
   under the iter-46 STDP `a_plus` (0.020) at the DG → R2
   weight scale used in iter-60+ (`dg_to_r2_weight = 1.0`,
   `dg_to_r2_fanout = 30`). A sweep of `dg_to_r2_weight` and the
   `a_plus` on the DG → R2 synapses (currently treated identically
   to R1 → R2 by the per-region plasticity) would show whether
   plasticity is structurally able to write the cue → target
   mapping at all.
2. **R2 recurrent strength.** The R2-E recurrent network builds
   the engram. If recurrent weights are too sparse / weak, the
   recurrent attractor doesn't carry the cue → target association
   between trials and the per-epoch dictionary fingerprint phase
   captures something disconnected from training. Sweep
   `R2_P_CONNECT` and the initial recurrent weight band.
3. **Perforant-path re-introduction.** iter-60 set
   `direct_r1r2_weight_scale = 0.0` (DG sole cue-routing path).
   The hippocampus has *both* the perforant path and the
   mossy-fibre path; with no direct R1 → R2 input, the trained R2
   has no consistent "raw cue" handle that plasticity can exploit
   independently of DG's hashed code. A sweep of
   `direct_r1r2_weight_scale ∈ {0.0, 0.1, 0.3, 1.0}` would test
   whether re-introducing a weak perforant path lets cue → target
   learning surface.

CA3/CA1 split is **deferred** until at least one of these three
mechanism questions produces a measurable learning signal. Adding
biological detail on top of an unverified read-out is the
iter-50/51 mistake the discipline exists to prevent.

### Methodological self-audit

Two issues surfaced *during* this iteration, both caught by the
pre-registered gates and both cleaned up before locking:

1. **Plumbing bug v1 (silent wiring gap + iter-52-invariant
   gap).** `run_target_overlap_arm` v1 went through
   `run_reward_benchmark` which ignores `decorrelated_init` and
   `dg.enabled`, and the existing 5-rule plasticity gate left
   metaplasticity / heterosynaptic / structural / BCM unattended.
   v2 introduces `build_benchmark_brain` and
   `disable_all_plasticity` shared helpers and a save/restore
   patch in `run_teacher_trial`. Verified bit-identical to
   pre-refactor on `run_jaccard_arm` via three snapshot tests.
2. **Pre-measurement metric correction.** The positive control
   caught the `prediction_top3_before_teacher` vs `top3_accuracy`
   mismatch on its first invocation; the band was recalibrated
   from `[0.16, 0.22]` (iter-46 ep0 peak) to `[0.07, 0.15]`
   (iter-51 stable estimator). Both corrections were applied
   before any trained-arm data was peeked at.

The trained main run's results above are on v2 code with the
correct band. No further corrections are applied post-data —
the verdict is what the matrix says it is.


