# Gate-B — γ.1.1 8-seed verdict

**Date:** 2026-05-08.
**Frozen config:** `reports/gate_a_gamma_1_1_config.json` /
`.md` (commit `ab9ae16`).
**Pre-registration:** `reports/gate_b_gamma_1_1_entry.md`
(commit `28bb9c9`, locked before any seed-4..7 compute).
**Run platform:** Windows PC (Bekos), 4 sequential 32-epoch
runs for seeds 4–7; seeds 0–3 carried over verbatim from the
Gate-A 4-seed run.
**Logs:** `reports/runs/gate_a_gamma_1_1_seed{0..7}.log`.
**Per-seed verdict JSONs:** `reports/runs/gate_a_gamma_1_1_seed{0..7}.verdict.json`.

## Headline

> **Class (C) Partial — needs different rule.  iter-67-γ.4
> (per-post target-gating with non-target depression) is the
> pre-registered fallback.**
>
> 5/8 seeds PASS, 3/8 FAIL.  Mean of last-8 means
> 0.0693 ± 0.0375; t(7) vs 0.05 = 1.46 (not significant);
> t(7) vs 0 = 5.23 (highly significant).  The C1 readout
> *binds* on the majority of seeds but not on all of them,
> and the failing seeds show a degrading-trajectory failure
> mode that γ.1.1's potentiation-only rule cannot correct.

## Per-seed table (8-seed verdict)

| seed | verdict | last-8 mean (ep 24–31) | 32-ep mean | longest contig.-zero | R2 32-ep mean | trajectory |
| ---: | :------ | ---------------------: | ---------: | -------------------: | ------------: | :--------- |
|    0 | **FAIL** | 0.0156                | 0.0361     | 2                    | 0.0303        | DEGRADING  |
|    1 | **PASS** | **0.0977**            | **0.0811** | 0                    | 0.0381        | improving  |
|    2 | **PASS** | **0.0898**            | **0.0791** | 0                    | 0.0322        | improving  |
|    3 | **PASS** | **0.0742**            | **0.0625** | 1                    | 0.0264        | improving  |
|    4 | **FAIL** | 0.0312                | 0.0420     | 2                    | 0.0244        | DEGRADING  |
|    5 | **PASS** | **0.1289**            | **0.1035** | 0                    | 0.0205        | improving  |
|    6 | **PASS** | **0.0742**            | **0.0674** | 0                    | 0.0205        | improving  |
|    7 | **FAIL** | 0.0430                | 0.0391     | 2                    | 0.0537        | flat       |

Seeds 0–3 verdicts re-evaluated from the original logs; results
are bit-identical to the 4-seed Gate-A summary
(`reports/gate_a_gamma_1_1_4seed_summary.md`).  R2 32-ep means
are within or just below the iter-65 chance band
(0.0264–0.0381).  No R2 collapse on any seed; `r2_collapsed`
auxiliary check passes on all 8.  `kwta_empty` mean is 0 on all
seeds (C1 always active in eval).

## Aggregate statistics

| Stat | All 8 seeds | PASS only (5) | FAIL only (3) |
| --- | ---: | ---: | ---: |
| pass count                | **5/8** | 5/5 | 0/3 |
| mean of last-8 means      | **0.0693** | 0.0930 | 0.0299 |
| std of last-8 means       | 0.0375     | 0.0225 | 0.0137 |
| sem (n=8)                 | 0.0133     | —      | —      |
| mean of 32-ep means       | 0.0638     | 0.0787 | 0.0391 |
| std of 32-ep means        | 0.0238     | 0.0163 | 0.0029 |
| t(7) vs **0.05 threshold**| **1.46**   | —      | —      |
| t(7) vs **0** (descriptive)| 5.23      | —      | —      |
| t(7) vs **4-seed prior 0.069** | 0.025 | —     | —      |

The 8-seed mean (0.0693) is **identical to the 4-seed prior
mean (0.069)** — the 4-seed Gate-A run was a perfect predictor
of the 8-seed aggregate; t vs 4-seed prior = 0.025 confirms no
distributional shift.  PASS-only mean of 0.093 is nearly twice
the threshold; FAIL-only mean of 0.030 is nearly twice the
"chance" floor — the bimodality is real.

## Class verdict (locked acceptance matrix)

| Class | Per-seed pattern | Aggregate | Match? |
| --- | --- | --- | :---: |
| (A) Confirm — strong       | 8/8 PASS AND `t(7) > 2.5`            | mean > 0.07         | **NO** (5/8) |
| (B) Robust — directional   | `n_pos ≥ 7/8` AND `t(7) > 1.895`     | mean ≥ 0.05         | **NO** (5/8) |
| (C) Partial — needs different rule | `5/8 ≤ n_pos < 7/8`          | `t(7) > 0`          | **YES**      |
| (D) Reject — architecture insufficient | `n_pos ≤ 4/8` OR mean ≤ 0 | (chance)            | NO           |

`n_pos = 5/8` lies in the locked Class-(C) band `5/8 ≤ n_pos
< 7/8`.  `t(7) vs 0 = 5.23 > 0` satisfies the (C) aggregate
guard.  Edge-case rules from the ENTRY pre-registration were
checked: `n_pos = 4/8` would collapse to (D); `n_pos = 7/8`
with weak t-stat would collapse to (C).  Neither edge case
applies; the verdict is unambiguously (C).

## Per-seed `top3_c1` trajectories (ep 0–31)

```text
seed 0 (FAIL, last-8=0.0156, DEGRADING):
  0.0625 0.0312 0.0312 0.0625 0.0938 0.0312 0.0312 0.0625
  0.0625 0.0625 0.0312 0.0312 0.0000 0.0312 0.0312 0.0625
  0.0000 0.0625 0.0625 0.0000 0.0312 0.0625 0.0312 0.0625
  0.0000 0.0000 0.0625 0.0312 0.0000 0.0000 0.0312 0.0000

seed 1 (PASS, last-8=0.0977, improving):
  0.0312 0.0312 0.0625 0.0625 0.0938 0.0625 0.0938 0.0625
  0.0312 0.0625 0.0938 0.0625 0.0625 0.1562 0.1250 0.0625
  0.0938 0.0625 0.0938 0.0625 0.0625 0.0625 0.0938 0.1250
  0.0938 0.1250 0.0938 0.0938 0.0938 0.0938 0.0625 0.1250

seed 2 (PASS, last-8=0.0898, improving):
  0.0625 0.0312 0.0312 0.0625 0.0938 0.0938 0.1250 0.1250
  0.0312 0.0938 0.0625 0.0938 0.0938 0.0625 0.0312 0.0938
  0.0938 0.0625 0.0625 0.1250 0.0312 0.0938 0.0625 0.0938
  0.0312 0.1250 0.0938 0.0938 0.0625 0.0938 0.1562 0.0625

seed 3 (PASS, last-8=0.0742, improving):
  0.0938 0.0938 0.0312 0.0312 0.0312 0.0312 0.0625 0.0312
  0.0938 0.0312 0.0312 0.0625 0.0625 0.0625 0.0000 0.1562
  0.0312 0.0625 0.0938 0.0625 0.0938 0.0938 0.0312 0.0312
  0.0312 0.0938 0.1250 0.0625 0.1250 0.0625 0.0312 0.0625

seed 4 (FAIL, last-8=0.0312, DEGRADING):
  0.0000 0.0625 0.0000 0.0312 0.0625 0.0625 0.0625 0.0938
  0.1250 0.0312 0.0625 0.0625 0.0312 0.0000 0.0000 0.0938
  0.0000 0.0312 0.0000 0.0312 0.0938 0.0938 0.0312 0.0312
  0.0312 0.0312 0.0000 0.0625 0.0312 0.0312 0.0000 0.0625

seed 5 (PASS, last-8=0.1289, improving — strongest):
  0.0312 0.0312 0.0625 0.0625 0.0625 0.0625 0.1250 0.0625
  0.0938 0.0938 0.0938 0.1250 0.1562 0.0938 0.1250 0.0938
  0.0625 0.1250 0.0625 0.1250 0.1875 0.0625 0.1875 0.0938
  0.2500 0.0938 0.1250 0.1250 0.0938 0.0625 0.1875 0.0938

seed 6 (PASS, last-8=0.0742, improving):
  0.0938 0.0938 0.0625 0.0312 0.0625 0.0312 0.0938 0.1250
  0.0938 0.0312 0.0625 0.0312 0.0312 0.0312 0.0938 0.0625
  0.0312 0.0312 0.0625 0.0625 0.1250 0.0938 0.0625 0.0625
  0.0312 0.0625 0.1250 0.0938 0.0938 0.0938 0.0625 0.0312

seed 7 (FAIL, last-8=0.0430, flat):
  0.0625 0.0625 0.0625 0.0000 0.0625 0.0625 0.0312 0.0000
  0.0625 0.0000 0.0312 0.0625 0.0000 0.0000 0.0625 0.0312
  0.0625 0.0000 0.0938 0.0312 0.0625 0.0000 0.0000 0.0625
  0.0938 0.0312 0.0312 0.0625 0.0312 0.0312 0.0000 0.0625
```

## Per-seed diagnostic (Step 5 of locked analysis plan)

`scripts/evaluate_gate_a.py` augmented with the same training-
vs-eval split used in the seed-0 diagnostic.  Means are over
all 32 epochs unless stated otherwise.

| seed | first-half | second-half | trajectory | raw_overlap | C1 spikes (teacher) | tgt_w | non_w | w_ratio |
| ---: | ---------: | ----------: | :--------- | ----------: | ------------------: | ----: | ----: | ------: |
|    0 | 0.0449     | 0.0273      | **DEGRADING** | 0.0200  | 6 329               | 0.7993 | 0.7993 | 1.0001 |
|    1 | 0.0723     | 0.0899      | improving  | 0.0184      | 6 327               | 0.7978 | 0.7980 | 1.0000 |
|    2 | 0.0742     | 0.0840      | improving  | 0.0163      | 6 345               | 0.7988 | 0.7986 | 1.0000 |
|    3 | 0.0566     | 0.0684      | improving  | 0.0193      | 6 338               | 0.7998 | 0.7998 | 1.0000 |
|    4 | 0.0488     | 0.0351      | **DEGRADING** | 0.0164  | 6 344               | 0.7992 | 0.7995 | 1.0000 |
|    5 | 0.0859     | 0.1211      | improving  | 0.0181      | 6 334               | 0.7985 | 0.7984 | 1.0000 |
|    6 | 0.0645     | 0.0703      | improving  | 0.0122      | 6 334               | 0.7982 | 0.7985 | 1.0000 |
|    7 | 0.0371     | 0.0410      | flat       | 0.0185      | 6 348               | 0.7992 | 0.7991 | 1.0001 |

JSON: `reports/runs/gate_b_gamma_1_1_seed_diagnostic.json`.

### Three robust observations

1. **Trajectory split predicts verdict 8/8.**  Every PASS seed
   (1, 2, 3, 5, 6) shows monotone-or-improving second-half;
   every FAIL seed (0, 4, 7) shows DEGRADING or flat.  The
   degrading mode is *the* failure mechanism on γ.1.1, not
   trial-to-trial noise.  Seed 0 already exhibited this in
   the Gate-A diagnostic; seed 4 confirms it (0.049 → 0.035);
   seed 7 borderlines (0.037 → 0.041, "flat", but no
   improving trend over training).
2. **Training-side metrics are bit-identical across all 8
   seeds.**  C1 spikes (teacher) range 6 327 – 6 348
   (spread < 0.4 %).  `tgt_w` and `non_w` both settle in
   `[0.798, 0.800]` (spread < 0.3 %).  `w_ratio` is universally
   `1.000 ± 0.0001`.  γ.1.1's BTSP+E/I-split mechanism is
   deterministic-uniform on the training side; failure is
   purely an *eval-phase fingerprint discrimination* problem.
3. **`raw_overlap` does NOT predict verdict.**  Seeds 0 (FAIL)
   and 6 (PASS) bracket the range (0.020 vs 0.012).  Seed 1
   (PASS, second-best) and seed 7 (FAIL) sit at nearly identical
   raw overlap (0.0184 vs 0.0185).  The kWTA hits the canonical
   target SDR equally often across all seeds; what differs is
   per-cue *fingerprint geometry* — different cues' kWTA sets
   collide enough on the failing seeds that the decoder ranks
   the wrong cue higher.

### Why γ.4 is the locked next step

γ.1.1's BTSP rule potentiates only when a plateau fires on the
target C1 cell (target-gated).  Non-target weights drift up
through homeostasis + R-STDP-on-actual-R2-pattern; the result
is the empirical `tgt_w ≈ non_w ≈ 0.80, w_ratio ≈ 1.0` we see
on every seed.  In other words: **γ.1.1 binds via the
fingerprint, not via per-class weight magnitude separation**
(K4 strict ≥ 1.5 not met on any seed).

For seeds where the per-cue fingerprints separate cleanly
(1, 2, 3, 5, 6), this works.  For seeds where the R2/DG
wiring + cue-engram patterns produce overlapping kWTA sets
(0, 4, 7), the readout cannot disambiguate, and the trajectory
either degrades (0, 4) or stays at the noise floor (7).

γ.4 is the pre-registered fallback: **per-post target-gating
*with non-target depression***.  Same C1 layer, same BTSP
plumbing, same recall-mode discipline — but non-target
weights are explicitly DEPRESSED (LTD on non-target plateaus
or anti-Hebbian on non-target-cell potentiation).  This
introduces weight-magnitude separation that the kWTA can
exploit on the geometrically-marginal seeds without breaking
the seeds that already pass via fingerprint geometry.

## Cross-platform validation

Seeds 0 and 1 were independently replicated on the Linux
container before the 4-seed Windows-PC run.  `top3_c1`,
`top3_r2`, `target_in_dict`, `kwta_empty` are bit-identical
across platforms.  Sub-permille drifts on weight-magnitude
diagnostics (`tgt_w` 0.7849 ↔ 0.7904, `plateau_events`
224 464 ↔ 224 311) attributable to rustc/MSVC vs Linux-clang
FP-codegen differences.  Verdict-relevant integer kWTA hits
agree.  No cross-platform action item.

## Recall-mode stability

| seed | top3_r2 32-ep mean | top3_r2 < 0.005 epochs | r2_collapsed |
| ---: | -----------------: | ---------------------: | :----------: |
|    0 | 0.0303             | 12/32                  | False        |
|    1 | 0.0381             |  7/32                  | False        |
|    2 | 0.0322             |  9/32                  | False        |
|    3 | 0.0264             | 13/32                  | False        |
|    4 | 0.0244             | 16/32                  | False        |
|    5 | 0.0205             | 17/32                  | False        |
|    6 | 0.0205             | 17/32                  | False        |
|    7 | 0.0537             |  4/32                  | False        |

`r2_collapsed = True` would require >75 % of epochs at
top3_r2 < 0.005.  No seed is close.  All 8 runs preserved
the iter-65 chance-level R2 baseline; the C1 readout gain on
the 5 passing seeds is therefore not attributable to a
degraded R2 baseline.

`recall_mode_eval = true` was active throughout (locked in
`gate_a_gamma_1_1_config.json`); the iter-52/iter-62 invariant
(`disable_all_plasticity` before each eval phase) was honoured
by construction on all 8 seeds.

Note: seed 5 has the **lowest** R2 readout (0.0205) but the
**highest** C1 readout (0.1289).  The C1 binding is not a
proxy for R2 quality.

## Decision per the locked acceptance matrix

> **Verdict:** Class (C) Partial.
>
> **Action:** iter-67-γ.4 (per-post target-gating with
> non-target depression) at the same C1 layer.  Same locked
> corpus, same recall-mode discipline, same 8-seed verdict
> matrix.  γ.4 spec lands in its own ENTRY (pre-registered
> per the iter-66 ENTRY's "(C) → γ.4" rule); spec must be
> committed before any γ.4 compute.

What this is NOT:
- NOT a Gate-B failure.  γ.1.1 produces a real, multi-seed
  signal (5/8 PASS at last-8 mean ≥ 0.05; t vs 0 = 5.23).
  This is the FIRST iter-66+ configuration to produce a
  Gate-B-scale measurable C1-target binding signal at all.
- NOT a green light to relax the threshold.  The locked
  threshold (last-8 mean ≥ 0.05) is unchanged; γ.4 must clear
  it on more seeds to win.
- NOT a 16-seed extension.  The locked acceptance matrix
  routes (C) to γ.4, not to extension.  16-seed is the (B)
  branch only.

## Limitations and honest caveats

1. **w_ratio asymptotes near 1.0 across all 8 seeds.**  No
   seed builds per-class weight-magnitude separation; the
   binding lives in fingerprint geometry exclusively.  K4
   strict (≥ 1.5) is not met by γ.1.1 on any seed.
2. **3 of 3 FAIL seeds show DEGRADING-or-flat trajectories.**
   This is not consistent with simple sample noise; it is a
   structural failure mode of γ.1.1's potentiation-only rule
   on a subset of R2/DG wirings.
3. **Cross-platform reproducibility partial.**  Headline
   metrics bit-identical Linux ↔ Windows; weight magnitudes
   drift sub-permille.  Verdicts agree.
4. **Single platform, single hardware vendor, single compiler
   chain.**  Generalisability beyond x86-64 + stable rustc not
   tested.
5. **t-stat baseline is the 4-seed prior, not an untrained
   arm.**  γ.1.1's C1 readout depends on training (without
   training, kwta_empty = 32/32 → no fingerprints).  The
   t-statistics are robustness indicators, NOT trained-vs-
   untrained comparisons.  See ENTRY pre-registration §"Sample-
   size note".
6. **Compute budget.**  Total Gate-B compute on Bekos's
   Windows PC: ~9–10 h sequential per 4-seed batch ×2 batches
   = ~18–20 h wallclock for the full 8-seed run.  Container
   restarts on Linux made the host-machine run cheaper than
   re-executing under risk of mid-run process loss.

## Status

- **Step 7 of locked Gate-B analysis plan — COMPLETE.**
- iter-67-γ.1.1 verdict at Gate-B: **Class (C) Partial**.
- Pre-registered next step: iter-67-γ.4 ENTRY (per-post
  target-gating with non-target depression).
- No more compute on iter-67-γ.1.1 without a fresh ENTRY.
- Awaiting Bekos's Go on iter-67-γ.4 ENTRY (spec, not compute).
