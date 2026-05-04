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

`target_top3_overlap` = mean over the vocab of |R2_top3(cue)
∩ canonical_target_SDR(cue)| / 3, where canonical_target_SDR
is the iter-46 target encoding for the cue's pair partner.

**One metric. One comparison. No top-1 side-eye, no MRR
peek, no per-pair sub-analysis.** This is the iter-50 / 51
lesson: multiple-comparison drift is what produces 5
iterations against the wrong metric.

## Pre-registered acceptance threshold

The threshold is **not yet a number** — it is a function of
the untrained baseline, computed on real data before the
trained arm runs.

### Calibration phase

Untrained-only sweep at the same configuration the main run
will use:

```text
seeds        = 42, 7, 13, 99       (4 seeds)
epochs       = 32
vocab        = 64
DG bridge    = on
recall-mode  = on (plasticity-off-during-eval)
no-plasticity= true (untrained arm)
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
`target_top3_overlap` on iter-46 Arm B reproduces the
iter-46 / iter-50 reading (top-3 = 0.19 ± noise, **not** at
the random-baseline 0.047). If the positive control shows
no signal, the wiring is broken — fix the plumbing before
running calibration. **Do not** pivot architecture (branch
B) on a silently-broken metric pipeline.

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

```sh
# 1. Positive control — verify metric wiring on iter-46
#    Arm B (known-working).
cargo run --release -p eval --example reward_benchmark -- \
  --target-overlap-bench --iter46-baseline \
  --seeds 42 --epochs 16
# Acceptance: trained target_top3_overlap ≈ 0.19 (iter-46 / 50
# reading). If silent, fix plumbing first.

# 2. Calibration — untrained baseline on DG + recall-mode.
cargo run --release -p eval --example reward_benchmark -- \
  --target-overlap-bench --no-plasticity \
  --seeds 42,7,13,99 --epochs 32 \
  --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval
# Compute μ_untrained, σ_untrained.
# Lock acceptance_threshold = max(+0.05, μ + 2σ).
# COMMIT this note with the locked number before step 3.

# 3. Main — trained arm at the same configuration.
cargo run --release -p eval --example reward_benchmark -- \
  --target-overlap-bench \
  --seeds 42,7,13,99 --epochs 32 \
  --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval
# Apply the locked branching matrix.
```

## Implementation plan

- New CLI flag: `--target-overlap-bench` parallel to
  `--jaccard-bench`. Mutually exclusive at the parser level.
- New public surface in `crates/eval/src/reward_bench.rs`:
  - `TargetOverlapMetrics { per_seed: Vec<f32>, mean: f32,
    std: f32 }`
  - `run_target_overlap_arm(brain, cfg) ->
    TargetOverlapMetrics` — re-uses iter-46 / 52 logic from
    `RewardEpochMetrics::target_top3_overlap` on the
    DG-enabled brain.
  - `render_target_overlap_sweep(trained, untrained,
    threshold) -> String` — markdown summary with
    per-seed table + paired t(3).
- L2 bit-identity invariant continues to apply on the
  trained-arm eval phase whenever `recall_mode_eval` is on.
- `--iter46-baseline` flag (already exists from iter-50)
  routes through `--target-overlap-bench` for the positive
  control.

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

## Headline (placeholder)

> *to be filled after the main run; one of:*
> - **Direct cue → target learning verified on DG-enabled
>   brain** (branch A)
> - **DG separates and recalls but does not learn — mechanism
>   question for iter-64** (branch B)
> - **Directional learning signal under-powered at n=4 — more
>   seeds before iter-64** (branch C)
