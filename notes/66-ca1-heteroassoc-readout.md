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

---

## Step 7 — Smoke result (1 seed × 8 epochs, pipeline integrity)

**Date:** 2026-05-06.
**Commits:** `e0dea5a` (steps 1–6), `246d863` (step 6 cont. CLI
routing), this commit (step 7 smoke result).

**CLI invocation:** locked from the ENTRY — verbatim.

```sh
cargo run --release -p eval --example reward_benchmark -- \
  --c1-readout --c1-teacher-strength 1.0 \
  --seeds 42 --epochs 8 \
  --teacher-forcing --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval \
  --decorrelated-init
```

**Result table:**

| Seed | target_top3_overlap (R2) | c1_target_top3_overlap (C1) | C1 − R2 |
| ---: | ---: | ---: | ---: |
| 42 | 0.0273 | 0.0000 | −0.0273 |

**Pipeline integrity (ENTRY's smoke success criteria):**

- Compiles: ✓
- Runs end-to-end (no panic): ✓
- Asserts hold (`assert_no_weight_drift` not exercised because
  trained arm: no untrained-mode pre/post L2 invariant required;
  decorrelated_init `assert_decorrelated_disjoint` ✓): ✓
- `c1_target_top3_overlap` is in `[0.0, 1.0]`: ✓ (= 0.0000)

**All four ENTRY smoke criteria PASS.** The smoke is therefore
a **GO** to step 8 (main run, 8 seeds × 32 epochs).

### Smoke observations (informative, not gating)

1. **R2 readout at 8 ep = 0.0273.** Consistent with iter-65 R2
   training trajectory (iter-65 trained mean over 32 ep was
   0.0396; at 8 ep the partial-trajectory reading would land
   around 0.025–0.035). No regression vs iter-65's R2 path.
2. **C1 readout at 8 ep = 0.0000.** Three candidate explanations,
   to be discriminated by the step-8 32-ep main run:
   - **(C1.a)** *Insufficient training.* The R2 metric at 8 ep
     is itself only 0.0273; R2-E cells don't fire the canonical-
     target pattern reliably during cue-only eval, so the
     R2-E → C1 path doesn't see a discriminative pre-spike
     pattern to learn. At 32 ep the R2 readout climbs to
     0.0396; if C1 readout climbs in lock-step, M1 is on the
     right trajectory.
   - **(C1.b)** *C1 cells silent during eval.* If R2-E spikes
     are too sparse / R2-E → C1 weights too low to push C1
     cells over LIF threshold, the eval-phase kWTA returns
     empty, no concept is learned in the C1 dictionary, and
     the metric drops to 0.0 mechanically. Diagnostic at
     step-8: per-epoch C1 spike count log.
   - **(C1.c)** *Non-discriminative C1 fingerprints.* If every
     cue produces the same C1 fingerprint (e.g. a single
     dominant attractor), the dictionary learns identical
     fingerprints for every word and decoding is ambiguous /
     degenerate to chance. iter-53 confirmed deterministic
     dynamics under recall-mode (mean Jaccard = 1.0
     within-cue), so this would manifest as cross-cue Jaccard
     near 1.0 across the dictionary entries — testable at
     step-8 with a one-shot Jaccard diagnostic if needed.
3. **Architectural side-effect: R2-R2 R-STDP fires during the
   M_target window.** The implementation gates the modulator
   globally on R2's `Network`, so the M_target = +1 pulse
   during the teacher Phase 4 also drives R-STDP updates on
   R2-R2 synapses (the existing recurrent attractor). This is
   a deliberate consequence of the minimum-scope design — to
   isolate per-synapse M_target gating would require adding a
   per-synapse plasticity flag to snn-core, which iter-66
   ENTRY explicitly declines ("No new plasticity rule"). The
   side-effect tightens R2's recurrent representation
   alongside building C1; net direction on the R2 readout is
   ambiguous (could help or hurt). Step-8 will report
   per-seed R2 deltas vs iter-65 to surface any drift.

### Step-8 main run — green-lit (provisionally; revoked at step 7.5)

Per the ENTRY: "step 7 is a smoke commit; step 8 is the
verdict commit." The step-7 smoke meets all pre-registered
pipeline-integrity criteria. Proceeding to step-8 (8 seeds ×
32 epochs) with the locked acceptance matrix unchanged.

> **NOTE (added in step 7.5 commit):** The smoke's pipeline
> integrity criteria pass, but a follow-up diagnostic round
> requested by Bekos before burning the 8-seed × 32-ep main
> run (locked acceptance gate: "C1 activity > 0 in relevant
> phases AND R2→C1 weights change measurably AND target gate
> triggered AND no bit-genau zero dynamics over the C1 path")
> identified that the architecture as locked DOES NOT produce
> a discriminative C1 readout. **Step 8 is REVOKED at the
> step 7.5 verdict.** See "Step 7.5 — diagnostic verdict"
> below.

---

## Step 7.5 — diagnostic verdict (REVOKES step-8 green-light)

**Date:** 2026-05-06.
**Commits:** `3d1ec0f` (instrumentation), `ead6e32` (C1 spike
tracking fix in teacher phase), this commit (verdict).
**Diagnostic logs:** `/tmp/iter66/diag-seed42-ep8-v2.log`,
`/tmp/iter66/sweep-iwm.log`, `/tmp/iter66/sweep-fanout.log`.

### Diagnostic invocation

```sh
# Locked iter-66 ENTRY config + --c1-diagnostic, baseline FO=30, IWM=0.5
cargo run --release -p eval --example reward_benchmark -- \
  --c1-readout --c1-diagnostic --c1-teacher-strength 1.0 \
  --seeds 42 --epochs 8 --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval \
  --decorrelated-init

# Plus init_w_max sweep ∈ {0.5, 1.0, 2.0} × 4 epochs
# Plus from_r2_fanout sweep ∈ {30, 100, 200, 400} × 4 epochs
```

Bekos's pre-locked discrimination targets:
- **(A)** insufficient training: C1 active, weights change, but 8 ep too short
- **(B)** silent C1: kWTA empty during eval ⇒ readout mechanically zero
- **(C)** non-discriminative: C1 fires, but fingerprints don't separate cues

### Wiring sanity check (✓)

Pre-train R2-E → C1 stats (locked FO=30, IWM=0.5):

```text
r2_r2_synapses    = 199 492
r2c1_synapses     = 42 000     (= 1400 R2-E × 30 fan-out ✓)
r2c1_l2           = 59.0621    (matches uniform(0, 0.5) closed form ✓)
r2c1_mean_w       = 0.2496     (≈ 0.25 expected ✓)
```

Wiring end-to-end correct.

### Observation 1 — teacher phase works

Across 3 epochs at locked config:

```text
teacher: trials=256 c1_active_frac=1.000 c1_spikes_mean=572→362
         clamp_eff=1.000
```

Every trial fires every canonical-target C1 cell (clamp_eff = 1.0,
i.e. 20/20 target cells fired under the 500 nA clamp). C1 activity
during teacher = 100%. **R-STDP gating is correctly triggered**
(c1_active_frac = 1.000 ≡ M_target = +1 fires on every trial).

### Observation 2 — R2→C1 plasticity is alive

```text
r2c1: l2=59.06 → 43.65 → 44.69 → 43.92  (oscillating equilibrium)
      nz_upd=42 000     (every synapse changed)
      max|Δw|=0.7995    (single-synapse swing of nearly the full
                         w_max range; some weight went 0 ↔ 0.8)
      sum|Δw|=10 424    (≈ N×0.25; total magnitude of weight
                         movement matches the init range)
```

Weights are NOT bit-frozen. R-STDP + STDP + homeostasis operate on
every R2-E → C1 synapse every epoch.

### Observation 3 — eval phase: C1 is silent

```text
eval: kwta_empty=32/32 target_in_dict=0/32 spikes_mean=0.00
      top3_c1=0.0000 mrr_c1=0.0000 raw_overlap=0.000
      dict_concepts=0
```

**C1 cells fire ZERO spikes during cue-only eval at the locked
config**, across all 32 vocab cues. The dictionary builder fails
to learn any concept (kWTA returns empty for every word), so the
decoder has nothing to match against — top3_c1 = 0.000 is
mechanically forced.

### init_w_max sweep — IWM axis is dead

| init_w_max | r2c1_l2 (post-ep0) | kwta_empty (eval) | top3_c1 |
| ---: | ---: | ---: | ---: |
| 0.5 (locked) | 43.65 | 32/32 | 0.0000 |
| 1.0 | 44.22 | 32/32 | 0.0000 |
| 2.0 | 44.52 | 32/32 | 0.0000 |

Post-epoch L2 converges to ≈ 44 *regardless of init*. Mechanism:
`homeostasis()` has `a_target=2.0, scale_only_down=true` —
heavy teacher-phase firing of canonical C1 target cells triggers
homeostatic down-scaling on their incoming R2-E synapses, AND
intrinsic plasticity (`alpha_spike=0.05, offset_max=5.0`) raises
those cells' thresholds. The network drives weights to a fixed
equilibrium that the IWM sweep cannot break.

**Verdict on IWM axis:** init_w_max alone cannot fix C1 silence.
Architectural constraint, not parameter-tuning.

### from_r2_fanout sweep — structural axis surfaces life

| FO | r2c1 syn | kwta_empty/32 (final) | spikes_mean | dict_concepts | raw_overlap | top3_c1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 (locked) | 42 000 | 32 | 0.00 | 0 | 0.000 | 0.000 |
| 100 | 140 000 | 31 | 0.06 | 1 | 0.000 | 0.000 |
| 200 | 280 000 | 31 | 0.22 | 1 | 0.005 | 0.000 |
| 400 | 560 000 | 28–29 | 0.34–0.78 | 3–4 | 0.000–0.003 | 0.000 |

Fan-out crosses the silence threshold somewhere between FO=30 and
FO=100 — at FO=400, ~4 of 32 cues produce nonzero C1 firing during
eval. **Structural axis confirmed as the right knob.**

But — even at FO=400 with C1 firing on multiple cues:

```text
target_in_dict = 0/32  raw_overlap = 0.000–0.003  top3_c1 = 0.000
```

**The C1 cells that DO fire during eval fire the WRONG pattern.**
The canonical-target C1 SDR (the supervised goal pattern) is never
overlapped (raw_overlap ≈ 0); the target word never enters the
decoded ranking (target_in_dict = 0). top3_c1 stays bit-frozen at
0.000 across all 4 epochs.

### Diagnosis: (B) ∧ (C) — both branches simultaneously fail

At locked config (FO=30): **(B) silent C1.**
At FO=100–400: silent partially breaks, but **(C) non-discriminative
fingerprints** takes over — C1 fires, but on patterns uncorrelated
with the canonical target.

**Root-cause walk-through.** During teacher Phase 4:

- Canonical R2 target SDR (~30 cells) is clamped via 500 nA →
  fires heavily.
- Canonical C1 target SDR (~20 cells) is clamped via 500 nA →
  fires heavily.
- M_target = +1 is set for the teacher window.
- R-STDP eligibility tag accumulates only on synapses whose pre
  AND post both fire → **only the ~30×20×(30/1000) ≈ 18 R2-target
  → C1-target synapses per trial get LTP**. The other 41 982
  synapses get pre-only or post-only events (no eligibility ⇒ no
  R-STDP update; STDP-LTD via post-trace if post fires before
  pre, but post-trace is small for non-target cells that don't
  fire ⇒ ≈ no STDP either).
- Homeostasis sees the 20 C1 target cells firing at saturation
  → scales their incoming weights DOWN to keep activity at
  `a_target=2.0`. This *opposes* the R-STDP-driven LTP on the
  18 active synapses.

Net result on weight matrix: a small positive R-STDP-driven LTP
on R2-target → C1-target synapses, *cancelled* by homeostatic
down-scaling on those same synapses. The R2-target → C1-target
pathway is NOT meaningfully strengthened.

During eval (cue-only, no clamps):

- R2's natural cue-driven response is whatever R2's recurrent
  attractor produces — and iter-65 already showed this is
  **basically chance** vs the canonical target SDR
  (top3_r2 = 0.0312 ≈ 1/32 ≈ chance for vocab=32 at-best, here
  vocab=64 with corpus-vocab=64 means ≈ 2-3% top-3 hits per cue).
- Even the small R-STDP-strengthened R2-target → C1-target
  pathway never *fires* during eval, because R2 doesn't produce
  the canonical target SDR.
- C1 fires (when fan-out is large enough) on whatever R2 actually
  produces, which has no learned mapping to canonical C1 target.
  Hence raw_overlap ≈ 0 and top3_c1 ≈ 0.

**The architectural mismatch:** the iter-66 ENTRY trains R-STDP on
(canonical R2 target) → (canonical C1 target) under double-clamp,
but the canonical R2 target SDR doesn't emerge from R2 during
eval. So the readout learning signal lives at a pattern that
never gets re-activated by the cue-only eval phase.

### Bekos's step-7.5 acceptance gate (locked) — outcome

| Gate | Required | Observed | Pass? |
| --- | --- | --- | :---: |
| C1 activity > 0 in relevant phases | yes (teacher AND eval) | teacher: ✓ (100%); eval: ✗ at locked FO, partial at FO≥100 | ✗ |
| R2→C1 weights change measurably | yes | nz_upd=42 000 every epoch | ✓ |
| target gate triggered | yes | c1_active_frac=1.000, M_target=+1 fires | ✓ |
| no bit-genau zero dynamics over C1 path | yes | top3_c1=0.0000 across 4 ep at FO=400 (no rising curve) | ✗ |

**Bekos's "Wenn C1 aktiv + Gewichte ändern sich, aber
c1_target_top3_overlap bleibt 0" branch:** "Full Run nur dann
starten, wenn 32-epoch single-seed eine steigende Kurve zeigt."
Across 4 epochs at FO=400 the curve is *flat at 0*, no upward
trajectory. **Step 8 (8-seed × 32-ep main run) is NOT green-lit.**

### Recommendation — what to fix before iter-66 verdict run

The diagnostic localised the failure to a clean architectural
constraint: R-STDP under simultaneous double-clamp + active
homeostasis cannot produce a learning signal that survives the
cue-only eval phase. Three minimal-scope fixes for an
iter-66.5 follow-up:

1. **Bias R-STDP toward eval-phase R2 patterns.** Instead of
   double-clamp during teacher, drive R2 with the cue *only*
   (no R2-target clamp), let R2's natural response emerge,
   then apply a delayed C1 clamp + M_target pulse. R-STDP
   learns (R2 cue-driven pattern) → (canonical C1 target) —
   exactly the mapping needed at eval. Does NOT add new
   plasticity rules; just reorders the teacher schedule.
   Implementation: invert Phase 4's R2-clamp; gate C1-clamp
   on a 10–20 ms delay vs cue onset.
2. **Disable homeostasis on R2's `Network` for iter-66 runs.**
   `homeostasis().scale_only_down = true` + the
   `a_target = 2.0` Hz target are tuned for R2's recurrent
   attractor; they actively cancel R-STDP signal on
   feedforward R2-E → C1 synapses. The iter-46 stack assumed
   one shared region; iter-66's logical sub-region split
   needs per-pathway control. Workaround without snn-core
   changes: omit `enable_homeostasis(homeostasis())` in
   `run_target_overlap_one_seed` when c1.enabled. Does NOT
   add new plasticity rules; just disables an existing one.
3. **Switch to BTSP one-shot rule (Mechanism M5 from
   notes/66-deep-research-cue-target-binding.md).** A single
   supervised trial creates a stable R-STDP-equivalent
   eligibility window without iterative competition with
   homeostasis. Adds new plasticity rule to snn-core; biggest
   scope of the three but the literature-recommended
   alternative if (1)/(2) don't hold up.

(1) is cheapest and most likely to work; (2) is a one-line code
change orthogonal to (1); (3) is the iter-67 fallback if (1)+(2)
don't break the C1 silence.

### Iter-66 status: PAUSED at step 7.5

The implementation (steps 1–7) is correct and committed.
The architecture as locked in the ENTRY does not produce a
learning signal at C1. **Step 8 (8-seed × 32-ep main run) is
NOT executed.** Branch `claude/iter66-ca1-heteroassoc-readout`
holds the diagnostic verdict; iter-66.5 (or iter-67) will be
pre-registered separately to address fix (1) and/or (2) above.
The locked acceptance matrix and seed set carry over verbatim
to that follow-up.
