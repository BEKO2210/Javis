# Gate-B — γ.1.1 8-seed pre-registration (ENTRY, locked)

**Status:** ENTRY (pre-registration).  No measurements, no
implementation changes.  The hypothesis, locked seed set,
locked acceptance matrix, locked metrics, and analysis plan
are committed in this file *before* any seed beyond seed 3
has been run at Gate-B scale.  They may not be relaxed
post-hoc.

## Why Gate-B

iter-67-γ.1.1 cleared Gate-A:

- Candidate seed=42: PASS_LOWER_BOUND (lower-bound proof,
  31/32 epochs, last-8 mean ≥ 0.0508).
- 4-seed Gate-A confirmation: 3/4 PASS (seeds 1, 2, 3 PASS;
  seed 0 FAIL with last-8 mean = 0.0156).

Mean last-8 across 4 seeds: 0.069 ± 0.037.  Without seed 0:
0.087 ± 0.012.  This is the FIRST iter-66+ configuration to
produce a multi-seed-confirmed non-zero
`c1_target_top3_overlap` aggregate.

Gate-B is the locked iter-66 ENTRY's 8-seed × 32-epoch full
verdict matrix (`notes/66 §"Locked acceptance matrix"`).  At
the current 3/4 pass rate, the 4-seed result is consistent
with both **(B) Robust** (n_pos ≥ 7/8) and **(C) Partial**
(5/8 ≤ n_pos < 7/8).  Gate-B disambiguates.

## Seed-0 diagnostic (informs Gate-B priors, NOT a verdict)

A diagnostic reading of the existing 4 seed-runs revealed
that the mechanism failure on seed 0 is in the eval-phase
fingerprint discrimination, NOT in the BTSP training:

| Metric | seed 0 (FAIL) | seeds 1/2/3 (PASS) |
| --- | ---: | ---: |
| BTSP plateau events / 32-ep mean | 226 310 | 226 511 – 226 593 |
| C1 spikes (teacher) / 32-ep mean | 6 329 | 6 327 – 6 345 |
| Clamp efficacy | 1.0 | 1.0 |
| `tgt_w` / `non_w` 32-ep | 0.7993 / 0.7993 | 0.7978–0.7998 / 0.7980–0.7998 |
| **raw_overlap (kWTA ∩ canon)** | **0.020** | 0.018–0.019 |
| **top3_c1 first-half / second-half** | 0.045 / 0.027 | 0.057–0.074 / 0.068–0.090 |
| **Trajectory direction** | **DEGRADING** | improving |

Training-side metrics are identical across all 4 seeds
(within 0.3 %).  γ.1.1's BTSP+E/I-split mechanism is
deterministic-uniform on the training side regardless of
seed.  raw_overlap is also nearly identical: the kWTA hits
the canonical target SDR equally often.  But on seed 0, the
per-word C1 fingerprints in the eval-phase dictionary fail
to discriminate cleanly — different cues' kWTA sets overlap
enough that the decoder ranks the wrong word higher.  AND
the situation degrades with training (second-half mean is
WORSE than first-half on seed 0; first-half mean was already
sub-threshold).

Implication for the Gate-B prior:
- Pass rate at 8 seeds is *not* a sample-size question alone.
  It depends on what fraction of seeds produce R2/DG wiring
  + cue-engram patterns that γ.1.1 can discriminate at the
  fingerprint level.
- Cannot extrapolate from 4 seeds to 8 with confidence.
- Need the full 8-seed run to know.

## Pre-registered hypothesis

> On the iter-67-γ.1.1 frozen configuration (commit `ab9ae16`,
> file `reports/gate_a_gamma_1_1_config.json`), 8 seeds × 32
> epochs each produce a `c1_target_top3_overlap` distribution
> consistent with the iter-66 locked acceptance matrix.  The
> verdict class will be one of (A) / (B) / (C) / (D)
> per the locked criteria below.  No threshold relaxation,
> no seed cherry-picking, no post-hoc reclassification.

## Locked seed set (no relaxation)

iter-67 Gate-B uses 8 seeds = {Gate-A 4-seed already-run} ∪
{4 new seeds}:

```text
already-run (Gate-A): 0, 1, 2, 3
new (Gate-B addition): 4, 5, 6, 7
final 8-seed set:    0, 1, 2, 3, 4, 5, 6, 7
```

This is mandatory for cross-iter comparability.  Each seed
× 32-epoch run is deterministic-reproducible, so the 4
already-completed seeds carry over directly; only seeds
4 / 5 / 6 / 7 require new compute.

The candidate seed=42 result (PASS_LOWER_BOUND from a
partial 31/32-epoch run) is *referenced* but NOT counted
in the 8-seed verdict — the seed=42 candidate was for
mechanism validation; the locked 8-seed set above is the
formal Gate-B input.

## Locked acceptance matrix (verbatim from iter-66 ENTRY)

| Class | Per-seed pattern | Aggregate | iter-67 next step on Gate-B verdict |
| --- | --- | --- | --- |
| **(A) Confirm — strong** | last-8 mean ≥ 0.05 on **8/8** seeds AND paired `t(7) > 2.5` | mean(last-8) > 0.07 | iter-67 LOCKED.  Move to iter-68 anatomy refinement (theta-phase gating, BTSP plasticity rule details, replay/consolidation). |
| **(B) Robust — directional** | last-8 mean ≥ 0.05 AND `n_pos ≥ 7/8` AND `t(7) > 1.895` | mean(last-8) ≥ 0.05 | iter-67 LOCKED at γ.1.1 with caveat.  Optional 16-seed extension to tighten the pass-rate estimate; otherwise lock and move on. |
| **(C) Partial — needs different rule** | last-8 mean ≥ 0.05 on `5/8 ≤ n_pos < 7/8` seeds | t(7) > 0 | iter-67-γ.4 (per-post target-gating with non-target depression) at the same C1 layer.  Same locked corpus + recall-mode discipline; tests whether weight-magnitude separation lifts the failing seeds without breaking the passing seeds. |
| **(D) Reject — architecture insufficient** | last-8 mean ≥ 0.05 on `n_pos ≤ 4/8` seeds OR mean(last-8) ≤ 0.0 | (chance) | iter-67 architecture insufficient for the binding task.  Escalate to Bekos for direction (Mechanism M2 Willshaw baseline OR pivot to non-CA1 readout).  No further parameter sweeps without explicit Go. |

Edge cases:

- `n_pos = 4/8` with mean(last-8) > 0 collapses to (D) per
  the iter-66 ENTRY's "Edge cases (`n_pos = 5/8` or `6/8`
  with weak t-stat) collapse to (C)" rule.
- `n_pos = 7/8` with `t(7) ≤ 1.895` collapses to (C).
- Sample-size note: paired t(n−1) here uses the 8 last-8
  means against the 4-seed-prior-mean of 0.069 as the null
  baseline, NOT against zero.  This is a deviation from
  iter-65 / iter-66 paired-against-untrained-arm; iter-67-γ.1.1
  has no untrained baseline because the C1 readout depends
  on training (without training, C1 doesn't fire at recall
  → kwta_empty would be 32/32 → no fingerprints → no decoding
  possible).  The t-stat is therefore a robustness indicator,
  NOT a trained-vs-untrained comparison.

## Locked methodological commitments

1. **No hyperparameter tuning** between Gate-A and Gate-B.
2. **No threshold changes** to last-8 mean ≥ 0.05.
3. **No cherry-picking**: failed seeds reported alongside
   passing seeds; the 4 already-completed seeds (0, 1, 2, 3)
   are included verbatim in the 8-seed verdict.
4. **No mechanism redesign before Gate-B verdict**: γ.4 / γ.3
   / etc. fallbacks are pre-registered but NOT explored
   pre-Gate-B.
5. **iter-65 / iter-66 / iter-66.5 numerics preserved when
   c1.btsp = false**: snapshot tests must stay 11/11 PASS.
6. **Same evaluator** (`scripts/evaluate_gate_a.py`, exits
   0/1/2 for PASS / FAIL / INCONCLUSIVE) for every seed.
   Cross-platform: PowerShell / UTF-16 LE auto-decoded by
   `scripts/clean_powershell_log.py` (verified: identical
   verdicts on the cross-platform Linux ↔ Windows seed-0
   replicate).
7. **Determinism contract**: the BTSP rule / C1 layer / DG
   bridge / R-STDP plumbing all run deterministically per
   `--seeds N`.  Any cross-platform drift on weight-magnitude
   counters (`tgt_w`, `plateau_events`) below 1 % is
   acceptable; the Gate-B verdict depends only on integer
   kWTA hits which are bit-identical across platforms.

## Locked CLI surface (for the 4 new seeds)

Runner script for Bekos's Windows PC:

```powershell
mkdir reports\runs -ErrorAction SilentlyContinue
foreach ($SEED in 4, 5, 6, 7) {
  Write-Host "=== START seed=$SEED $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ') ==="
  cargo run --release -p eval --example reward_benchmark -- `
    --c1-readout --c1-diagnostic --c1-eval-aligned-rstdp `
    --c1-btsp --c1-btsp-target-gated --c1-btsp-no-r2-isolation `
    --c1-btsp-window-ms 200 --c1-btsp-strength 0.4 `
    --c1-btsp-teacher-recurrent-e-scale 1.0 `
    --c1-btsp-teacher-recurrent-i-scale 0.3 `
    --c1-teacher-strength 1.0 `
    --seeds $SEED --epochs 32 `
    --teacher-forcing --target-clamp-strength 500 --teacher-ms 40 `
    --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval `
    --decorrelated-init `
    > "reports\runs\gate_a_gamma_1_1_seed$SEED.log" 2>&1
  Write-Host "=== DONE  seed=$SEED $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ') ==="
}
```

(Sequential ≈ 8–10 h on a single core; parallel 4 PowerShell
windows ≈ 2.5–3 h on a 4-core CPU with ~8 GB RAM available.)

After completion:

```powershell
git add reports\runs\gate_a_gamma_1_1_seed4.log
git add reports\runs\gate_a_gamma_1_1_seed5.log
git add reports\runs\gate_a_gamma_1_1_seed6.log
git add reports\runs\gate_a_gamma_1_1_seed7.log
git commit -m "exp: Gate-B seeds 4-7 complete on Windows PC"
git push origin main
```

(Note: file naming intentionally keeps `gate_a_gamma_1_1_seed{N}.log`
to match the 4-seed Gate-A naming and the existing evaluator
pipeline.  The Gate-B step is identified by the seed range and
the verdict in `gate_b_gamma_1_1_8seed_summary.md`, not by the
log filename.)

## Locked analysis plan (Step 7 deliverables, post-run)

1. Run `scripts/evaluate_gate_a.py` on all 8 logs (seeds 0–7),
   produce `reports/runs/gate_a_gamma_1_1_seed{0..7}.verdict.json`.
2. Aggregate the 8 last-8 means into a single dataframe.
   Apply the locked acceptance matrix above; emit the Gate-B
   class (A / B / C / D).
3. Compute paired t(7) against the 4-seed prior mean (0.069);
   report n_pos at threshold 0.05.
4. Cross-platform replicate seeds 0 & 1 from the existing
   container logs (already in the repository) → confirm
   verdict matches the Windows-PC runs.
5. Diagnostic: for seeds 4–7, replicate the seed-0
   diagnostic above (training-vs-eval split, raw_overlap,
   first-half-vs-second-half top3_c1).  Check whether any
   newly-run seed shows the seed-0 degrading-trajectory
   pattern; this informs the prior on which seeds will pass
   in any future 16-seed extension.
6. Write `reports/gate_b_gamma_1_1_8seed_summary.md` with:
   - 8-seed verdict table
   - Aggregate stats (mean ± std of last-8 means; pass count;
     paired t-stat; class label)
   - Per-class iter-67 next step (verbatim from the matrix above)
   - Honest limitations (cross-platform footnote, w_ratio ≈ 1.0
     observation, single-platform compute caveat)
7. Lock the verdict in a git commit "exp: Gate-B γ.1.1
   8-seed verdict — class (A/B/C/D)" and push.  No more
   compute on iter-67-γ.1.1 after that without a fresh ENTRY.

## What this commit is NOT

- Not a measurement.  No new data.
- Not a Gate-B verdict.  Only the pre-registration.
- Not iter-67-γ.4 spec.  γ.4 is conditional on Gate-B
  verdict (C); spec lands in its own ENTRY only after
  Gate-B (C) is observed.
- Not a 16-seed extension.  16-seed is conditional on
  Gate-B verdict (B); spec lands in its own ENTRY if needed.

## Headline (placeholder)

> *to be filled after the 8-seed run; one of:*
> - **(A) Confirm — strong; iter-67 LOCKED, move to iter-68
>   anatomy refinement.**
> - **(B) Robust — directional; iter-67 LOCKED with caveat.
>   Optional 16-seed tightening.**
> - **(C) Partial — iter-67-γ.4 (per-post target-gating).**
> - **(D) Reject — architecture insufficient; escalate.**
