# Iter 66 — CA1-equivalent heteroassociative readout (Mechanism M1)

**Status: ENTRY (pre-registration). No measurements, no
implementation yet. The hypothesis, locked seed set, locked
acceptance matrix, locked metric, and locked CLI surface are
committed in this file *before* any code is written. They may
not be relaxed post-hoc.**

> **Core sentence.** Iter-65 falsified the assumption that R2
> alone can bind cue → target under the iter-46 plasticity
> stack (n_pos = 4/8 = chance). Iter-66 implements the
> deep-research-recommended Mechanism M1: a CA1-equivalent C1
> layer with target-presence-gated three-factor R-STDP on a
> R2 → C1 projection, and asks whether the binding signal
> appears at C1 when it could not appear at R2.

## Why iter-66

Iter-65 closed Branch (C) Reject on the perforant-path
robustness check (`notes/65-perforant-path-robustness.md`).
The deep-research literature scan
(`notes/66-deep-research-cue-target-binding.md`, 28 sources)
converged on a single architectural recommendation:

> **CA3/CA1 split.** Add a CA1-equivalent C1 layer with
> target-presence-gated three-factor R-STDP on a R2 → C1
> projection. Primary metric: `c1_target_top3_overlap`
> (decoder reads C1 fingerprints, not R2).

Justification chain (from notes/66 §4):

- Marr (1971), Treves & Rolls (1994): canonical hippocampal
  memory has *two* learned matrices (EC→CA3, CA3→CA1); Javis
  has only the CA3 analogue.
- O'Reilly & McClelland (1994): separation + completion +
  binding in one structure is structurally precluded.
- Schapiro et al. (2017): the iter-64 axis-C value=0.3
  perforant-path α was a confound — the direct EC→CA1 path
  is real but lands in CA1, not CA3.
- Cassenaer & Laurent (2007/2012), Frémaux & Gerstner
  (2016), Brea et al. (2013), Bellec et al. (2020):
  recurrent-attractor SNNs *can* bind cue → target, but
  only under three-factor / target-conditioned learning
  rules — never under pure two-factor STDP. Javis already
  has R-STDP in the stack; the question is whether it is
  gated by a target-presence signal at the right synapses.

iter-66 implements the minimum-scope intervention consistent
with this literature: one new layer, one new projection,
one new modulator, one new metric. No new plasticity
rules (re-uses R-STDP). No CA3/CA1-deeper-anatomy
refinements (those are iter-67+ if iter-66 succeeds).

## Pre-registered hypothesis

> On the iter-63-baseline brain configuration *with* a new
> C1 layer (1000 LIF neurons, k=20 sparsity) connected from
> R2 by a plastic R2 → C1 projection trained under
> target-presence-gated three-factor R-STDP, 32 epochs of
> training produce a measurable cue → target learning
> signal on the new metric `c1_target_top3_overlap` such
> that
>
> Δ̄ > 0 AND n_pos ≥ 7/8 AND paired t(7) > 2.5
>
> with magnitude `Δ̄ > 0.05` (substantially above the
> iter-65 R2 noise floor of σ_untrained ≈ 0.0213).

This hypothesis is *strictly stricter* than iter-65's. iter-65
locked n_pos ≥ 6/8 for (B) Partial; iter-66 raises that to
n_pos ≥ 7/8 for full success because the deep-research
recommendation makes a strong architectural claim — if M1
were correct and iter-66 could not clear 7/8, the literature
would predict the architecture itself is still mis-specified
(suggesting M5 BTSP or a different binding rule entirely).

## Locked seed set (no relaxation)

iter-66 uses **exactly the same 8 seeds as iter-65, in the
same order**:

```text
seeds = 42, 7, 13, 99, 1, 2, 3, 4
```

This is mandatory for cross-iter comparability. Every
seed × value × phase tuple in iter-66 has a directly-paired
iter-65 baseline on the R2 metric, so the iter-66 verdict
can be expressed as a *delta of deltas* between the two
read-outs at the same configuration: `(c1_Δ̄ − r2_Δ̄)` with
both arms paired by seed.

In particular, seed=99 is included to test whether its
deterministic-negative-on-R2 pattern (iter-64 axis C
value=0.3 smoke −0.0273, full −0.0283; iter-65 −0.0283)
also fails at the C1 read-out, or whether C1 resolves the
seed-specific R2 attractor pathology by side-stepping it.

## Locked configuration

```text
Layer config (NEW):
  C1.size              = 1000 LIF neurons
  C1.sparsity_k        = 20 (matches R1/target SDR k)
  C1.inh_frac          = 0.0 (excitatory only — readout layer)

Projection R2 → C1 (NEW):
  topology             = sparse random, fan_out = 30 per R2 cell
  initial weight       = uniform(0, w_max / 2), w_max = 1.0
  plasticity rule      = target-presence-gated three-factor R-STDP
                         (rest of stack unchanged at R2)

Modulator M_target (NEW):
  Encoding phase: M_target(t) = +1 when canonical-target SDR
                  is the supervisory pattern for the current
                  cue × target trial.
  Eval phase:     M_target(t) = 0.

Teacher forcing on C1 during encoding:
  Strength             = c1_teacher_strength (CLI flag, default 1.0)
  Pattern              = canonical-target SDR for the current
                         trial's target word
  Window               = encoding-phase only (NOT during eval)

Recall mode (iter-62 invariant extension):
  Pre-eval, every plasticity rule on the C1 region disabled
  (disable_all_plasticity covers it via the same code path).
  Pre-eval and post-eval L2 norms on the C1 region must be
  bit-identical (assert_no_weight_drift extended to the
  C1 region).

All R1 / DG / R2 parameters: iter-46 defaults.
  R2_P_CONNECT         = 0.05
  dg_to_r2_weight      = 1.0
  direct_r1r2_weight_scale = 0.0   (iter-64/65 falsified
                                     anything else; staying
                                     at zero per the deep-research
                                     "killed paths" list)

Schedule:
  vocab                = 64
  epochs               = 32 (full phase by default)
  teacher_forcing      = on (six-phase schedule for cue/target trials)
  recall_mode_eval     = on (--plasticity-off-during-eval)
  decorrelated_init    = on (matches iter-63 trained main run)
  target_clamp_strength = 500
  teacher_ms           = 40
```

## Locked primary metric

`c1_target_top3_overlap` — exactly analogous to the iter-44/45
`top3_accuracy` but computed on **C1** fingerprints rather
than R2 fingerprints. Per epoch:

1. Build a per-epoch decoder dictionary from C1 fingerprints
   of every vocab word (run each word's cue, capture C1
   activity, kWTA top-K, store as that word's fingerprint).
2. For every real (cue, target) pair in the corpus, present
   the cue alone (no target SDR forcing), capture C1's
   decoded top-3, score 1 if the canonical target word is
   in the top-3 else 0.
3. `c1_target_top3_overlap_epoch = top3_hits / n_real_pairs`.

`c1_target_top3_overlap` = mean over all 32 epochs of the
per-epoch values. iter-51 / iter-63 / iter-64 lessons apply:
mean across epochs is the stable estimator, last-epoch is
not.

For cross-iter comparability, every iter-66 run also reports
the legacy `target_top3_overlap` on R2 (unchanged metric, on
the same brain) so iter-66 vs iter-65 can be compared
row-for-row at the same configuration.

## Locked acceptance matrix

| Outcome | Per-seed pattern | Aggregate | iter-67 entry |
| --- | --- | --- | --- |
| **(A) Confirm — strong** | Δ̄ ≥ 0.05 on **8/8** seeds AND paired `t(7) > 2.5` AND `c1_target_top3_overlap` mean > 0.30 | (consequence of per-seed) | iter-67 = downstream architecture refinement on a verified mechanism (theta-phase gating, BTSP plasticity rule, replay/consolidation) |
| **(B) Robust — directional** | Δ̄ > 0 AND `n_pos ≥ 7/8` AND `t(7) > 1.895` | n_above_threshold ≥ 6/8 | iter-67 = parameter sweep around the M1 design (c1_teacher_strength, R2→C1 fan-out, C1 sparsity); 8 → 16 seeds for power |
| **(C) Partial — needs different rule** | Δ̄ > 0 AND `5/8 ≤ n_pos < 7/8` | t(7) > 0 | iter-67 = swap target-presence-gated R-STDP for BTSP-style one-shot (Mechanism M5); same C1 layer, different rule |
| **(D) Reject — architecture insufficient** | Δ̄ ≤ 0 OR `n_pos ≤ 4/8` | (chance level on direction) | iter-67 = Mechanism M2 (Willshaw binary heteroassociative store) as upper-bound baseline; if M2 also fails, the DG code is the bottleneck and iter-68 = re-encode target SDRs |

Edge cases (`n_pos = 5/8` or `6/8` with weak t-stat) collapse
to (C) — the iter-66 ENTRY does *not* permit re-classifying
edge cases as (B) post-hoc.

## Locked CLI surface

```sh
# Smoke (1 seed, 8 epochs) — pipeline integrity check.
cargo run --release -p eval --example reward_benchmark -- \
  --c1-readout \
  --c1-teacher-strength 1.0 \
  --seeds 42 \
  --epochs 8 \
  --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval \
  --decorrelated-init

# Main run (8 seeds, 32 epochs) — primary measurement.
cargo run --release -p eval --example reward_benchmark -- \
  --c1-readout \
  --c1-teacher-strength 1.0 \
  --seeds 42,7,13,99,1,2,3,4 \
  --epochs 32 \
  --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval \
  --decorrelated-init
```

Mutually exclusive with `--axis-sweep`, `--jaccard-bench`,
`--target-overlap-bench`, etc. (existing iter-64 mutual-
exclusion guard at the top of `main()` extends to
`--c1-readout`).

## Locked methodological commitments

1. **No new plasticity rule.** Re-use the existing R-STDP
   from the iter-46 stack; the only addition is the
   target-presence modulator (M_target) gating.
2. **No CA3/CA1-deeper-anatomy.** No theta-phase gating, no
   Schaffer-collateral subdivision, no subiculum, no EC
   back-projection. M1 is the minimum-scope CA1 analogue;
   refinements are iter-67+ contingent on iter-66 verdict.
3. **No metric expansion.** Only `c1_target_top3_overlap`
   for the verdict; legacy R2 `target_top3_overlap` is
   reported for cross-iter comparability but does NOT
   factor into the (A/B/C/D) classification.
4. **No seed-set change.** Same 8 seeds as iter-65. Adding
   seeds is iter-67+ if (B) verdict.
5. **No epoch-count change.** 32 epochs (full phase) only.
   The two-phase smoke/full distinction from iter-64 is
   preserved as a debug option but the verdict applies the
   32-ep run.
6. **No goalpost shift on seed=99.** seed=99's
   deterministic-negative behaviour at R2 is one of the
   open questions iter-66 is designed to answer. It may
   stay negative on C1 (suggesting a deeper R2-attractor
   pathology that propagates) or resolve (suggesting C1
   binding side-steps the R2 pathway issue). Either outcome
   is informative; neither triggers a matrix re-write.
7. **iter-52 invariant extends to C1.** Recall-mode pre/post
   L2 on the C1 region must be bit-identical when the
   `disable_all_plasticity` helper is invoked. Implementation
   must extend `assert_no_weight_drift` to cover C1.
8. **Pre-flight verification mandatory:** before the main
   run, `grep -c "C1Region\|c1_target_top3_overlap" crates/`
   must return ≥ 5 (the four touched-files-baseline plus
   one re-export, conservative lower bound).
9. **Explicit cargo build before each cargo run.**

## Files to touch (planned, no code yet)

- `crates/snn-core/`: lightweight C1 region descriptor
  (re-uses existing `LifNeuron`/`Region`/`Network` types).
  No new plasticity rule needed — re-use the existing
  R-STDP and gate it at the projection level via a new
  `TargetPresenceGated` flag on the projection metadata.
- `crates/eval/src/reward_bench.rs`:
  - `BrainBuild` extended with optional `c1: Option<C1Build>`
    field (returned by `build_benchmark_brain` when the new
    `cfg.teacher.c1_readout = true` is set).
  - `run_target_overlap_one_seed` extended with a `c1_arm`
    return value (per-epoch top3 on C1 if C1 is enabled,
    else `None`).
  - New helper `run_c1_overlap_arm` analogous to
    `run_target_overlap_arm` but reading the C1 metric.
  - `assert_no_weight_drift` extended to cover the C1 region
    when present.
- `crates/eval/examples/reward_benchmark.rs`:
  - New CLI flag `--c1-readout` and `--c1-teacher-strength`.
  - Mutual-exclusion guard updated to include `--c1-readout`
    in the bench-mode set.
- `crates/eval/src/lib.rs`: re-exports for any new public
  surface (`C1Config`, `run_c1_overlap_arm`,
  `c1_target_top3_overlap` metric struct).
- `notes/66-ca1-heteroassoc-readout.md` (this file): updated
  with smoke + main-run results in a follow-up commit.

## Implementation sequence (locked)

1. **Step 1 — TeacherForcingConfig + new C1 types.** Add
   `c1_readout: bool` and `c1_teacher_strength: f32` to
   `TeacherForcingConfig`. Define minimal `C1Build` struct.
2. **Step 2 — C1 region construction + R2→C1 projection
   wiring** in `build_benchmark_brain`. Mechanical only;
   no plasticity yet.
3. **Step 3 — Target-presence modulator + R-STDP gating
   on R2→C1.** Extend the per-trial schedule in
   `run_target_overlap_one_seed` to assert M_target during
   encoding, drop it during eval.
4. **Step 4 — `c1_target_top3_overlap` metric** (per-epoch
   eval phase: build C1 dictionary, present cues, decode
   top-3, score against canonical target).
5. **Step 5 — `disable_all_plasticity` + `snapshot_weights`
   extended to cover C1 region** (iter-52 invariant for
   C1).
6. **Step 6 — CLI block** (`--c1-readout`,
   `--c1-teacher-strength`); mutual-exclusion guard.
7. **Step 7 — Smoke** (1 seed, 8 epochs at value=0.0
   perforant path, C1 enabled). Pipeline integrity:
   compiles, runs, doesn't panic, asserts hold,
   `c1_target_top3_overlap` is in [0.0, 1.0].
8. **Step 8 — Main run** (8 seeds, 32 epochs). Apply locked
   acceptance matrix.

Steps 1–6 are committed together as the implementation;
step 7 is a smoke commit; step 8 is the verdict commit.

## What this commit is NOT

- Not a measurement. No data.
- Not iter-67 spec. iter-67 is conditional on iter-66's
  classification (A / B / C / D) and will be specified in
  its own pre-registration after the iter-66 verdict.
- Not a re-litigation of iter-65. Axis-C value=0.3 stays
  Reject; iter-66 does not re-test the perforant-path
  hypothesis.

## Headline (placeholder)

> *to be filled after the run; one of:*
> - **(A) Confirm — C1 binding strong; iter-67 = anatomy
>   refinement.**
> - **(B) Robust — C1 binding directional; iter-67 = M1
>   parameter sweep + 16 seeds.**
> - **(C) Partial — needs different rule; iter-67 = swap
>   in BTSP (M5).**
> - **(D) Reject — architecture insufficient; iter-67 =
>   Willshaw baseline (M2) to test if DG code itself is the
>   bottleneck.**
