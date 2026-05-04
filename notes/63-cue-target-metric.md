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

`target_top3_overlap` is operationally identical to iter-46's
existing `prediction_top3_before_teacher` metric: the fraction
of real-pair prediction-phase trials whose top-k decoded R2
response contained **any** neuron from the cue's
canonical-target SDR (`canonical_target_r2_sdr` in
`reward_bench.rs`). Re-using the iter-46 metric verbatim — and
exposing it through new `--target-overlap-bench --mode <X>`
plumbing — is what makes the positive-control band [0.16,
0.22] meaningful: that band is calibrated to iter-46 / iter-50's
reading of `prediction_top3_before_teacher ≈ 0.19`. A
different metric (granular overlap fraction, target rank,
MRR) would require a re-calibrated band and is out of
scope for iter-63.

**One metric. One comparison. No top-1 side-eye, no MRR
peek, no per-pair sub-analysis.** This is the iter-50 / 51
lesson: multiple-comparison drift is what produces 5
iterations against the wrong metric.

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
`target_top3_overlap` on iter-46 Arm B falls within
**[0.16, 0.22]** (iter-46 / iter-50 reading 0.19 ± 0.03,
where ±0.03 absorbs known seed and binning noise from
iter-46). Outside this band — including a value that is
"close" but only 0.13 or 0.25 — counts as **plumbing
drift**, not "close enough", and triggers a wiring fix
before calibration. The 0.047 random-baseline floor is
explicitly rejected as a pass: the control fails if the
metric is silent. **Do not** pivot architecture (branch B)
on a silently-broken or drifting metric pipeline.

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

## Headline (placeholder)

> *to be filled after the main run; one of:*
> - **Direct cue → target learning verified on DG-enabled
>   brain** (branch A)
> - **DG separates and recalls but does not learn — mechanism
>   question for iter-64** (branch B)
> - **Directional learning signal under-powered at n=4 — more
>   seeds before iter-64** (branch C)
