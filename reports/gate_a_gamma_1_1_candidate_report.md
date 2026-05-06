# Gate-A — γ.1.1 candidate report (lower-bound PASS)

**Date:** 2026-05-06.
**Run:** iter-67-γ.1.1-extended, seed=42, target ep=0..31.
**Status:** **PASS by lower-bound proof** despite missing
ep 31 (process died between ep 30 and ep 31 logging, likely
container restart at ~18:35 UTC).
**Log:** `notes/67-step-7-iter67-gamma11-extended-ep0-30.log`
(31/32 epochs captured).

## Executive summary

iter-67-γ.1.1's seed=42 × 32-epoch candidate run completed
31 of 32 epochs.  The locked Gate-A criterion is
`last-8 mean ≥ 0.05` over epochs 24–31.  The observed sum
over ep 24–30 (7 of the 8 last-window epochs) is 0.40625;
the Gate-A required sum-over-8 is 0.400.  **The missing
epoch 31 cannot reduce the last-8 sum below 0.400 because
`top3_c1 ≥ 0` by construction.**  Therefore the final
last-8 mean is mathematically guaranteed to be at least
0.4062 / 8 = 0.0508 ≥ 0.05.  **PASS.**

## Per-epoch top3_c1 trajectory (ep 0–30)

```text
ep  | top3_c1 | mrr_c1
 0  | 0.0312  | 0.0432
 1  | 0.0312  | 0.0446
 2  | 0.0312  | 0.0285
 3  | 0.0625  | 0.0659
 4  | 0.0625  | 0.0549
 5  | 0.0938  | 0.0618
 6  | 0.0312  | 0.0389
 7  | 0.0312  | 0.0246
 8  | 0.0000  | 0.0344    ← isolated zero
 9  | 0.0625  | 0.0592
10  | 0.1250  | 0.0896    ← peak
11  | 0.0312  | 0.0570
12  | 0.0625  | 0.0553
13  | 0.1250  | 0.1067    ← peak (mrr also peak)
14  | 0.0938  | 0.0780
15  | 0.0625  | 0.0420
16  | 0.0312  | 0.0588
17  | 0.0312  | 0.0552
18  | 0.0000  | 0.0339    ← isolated zero
19  | 0.0625  | 0.0677
20  | 0.0000  | 0.0219    ← isolated zero
21  | 0.0312  | 0.0498
22  | 0.0938  | 0.0618
23  | 0.0312  | 0.0308
24  | 0.0938  | 0.1191    ← Gate-A window starts (last-8)
25  | 0.0312  | 0.0636
26  | 0.0312  | 0.0690
27  | 0.0312  | 0.0543
28  | 0.0625  | 0.0486
29  | 0.0938  | 0.0710
30  | 0.0625  | 0.0700
31  | (missing — process died before logging)
```

## Aggregate statistics

| Window | Span | Sum | Mean |
| --- | --- | ---: | ---: |
| All 31 epochs | ep 0–30 | 1.6246 | **0.0524** |
| First-8 | ep 0–7  | 0.3748 | 0.0469 |
| Middle-8 | ep 8–15 | 0.5625 | 0.0703 |
| ep 16–23 | (8 ep) | 0.2811 | 0.0351 |
| ep 24–30 (last-7 of last-8 window) | (7 ep) | **0.4062** | 0.0580 |

## Gate-A lower-bound proof

The locked Gate-A criterion is:

> last-8-ep mean `c1_target_top3_overlap ≥ 0.05` over epochs
> 24–31, plus auxiliary checks (no sustained collapse,
> C1 active, recall-mode stable).

Observed:

```text
ep24..ep30 top3_c1 = [0.0938, 0.0312, 0.0312, 0.0312,
                     0.0625, 0.0938, 0.0625]
sum(ep24..ep30) = 0.40625
```

Required:

```text
last-8 mean ≥ 0.05
⇒ sum(ep24..ep31) ≥ 0.05 × 8 = 0.400
```

Worst-case bound on missing epoch:

```text
top3_c1 ∈ [0, 1] by metric construction
ep31 ≥ 0
⇒ sum(ep24..ep31) ≥ sum(ep24..ep30) + 0 = 0.40625
⇒ last-8 mean ≥ 0.40625 / 8 = 0.05078 ≥ 0.05  ✓
```

**The missing epoch 31 cannot change the PASS verdict.**  Even
if ep 31 were exactly 0.0000 (the lowest possible value), the
last-8 mean would still equal 0.05078, which exceeds the
Gate-A threshold of 0.05.

## Auxiliary Gate-A checks

| Check | Required | Observed | Pass? |
| --- | --- | --- | :---: |
| `last-8 mean ≥ 0.05` | yes | 0.05078 (lower bound) | **✓** |
| no sustained collapse | longest contiguous-zero ≤ 2 ep | longest = 1 ep (isolated zeros at ep 8, 18, 20) | **✓** |
| C1 active throughout | `kwta_empty < 16/32` majority | `kwta_empty = 0/32` every epoch | **✓** |
| recall-mode stable | `top3_r2` not collapsed | top3_r2 oscillates 0.0 / 0.0312 / 0.0625 / 0.0938 (chance band) | **✓** |
| no L2 drift in eval | `recall_mode_eval` invariant | `recall_mode_eval = true` (enforced); not directly logged in the diagnostic line but the iter-52/iter-62 invariant is preserved by the `disable_all_plasticity` call before each eval phase, and any drift would manifest as drifting `tgt_w` / `non_w` values across epochs — observed `tgt_w` stable at 0.7985, `non_w` stable at 0.7974 | **✓** |

All five auxiliary criteria pass.

## Final Gate-A verdict

> **(A) PASS — γ.1.1 qualifies for 4-seed Gate-A confirmation.**

The verdict is established by lower-bound proof on a partial
run (31/32 epochs).  Per Bekos's locked instruction this
candidate is NOT to be rerun, patched, or deleted.

## Honest limitations

1. **Missing epoch 31.** The process died between ep 30 and
   the ep 31 logging line, likely a container restart.  The
   final aggregate `c1_target_top3_overlap` value normally
   emitted by the renderer is therefore unavailable.  The
   31-epoch-mean of 0.0524 is reported as a conservative
   point estimate; the true 32-epoch aggregate could be
   anywhere in `[0.0508, 0.0820]` depending on ep 31's
   value.
2. **Single seed.** The Gate-A candidate is seed=42 only.
   The 4-seed confirmation (seeds 0, 1, 2, 3) follows.
3. **High per-epoch variance.** `top3_c1` ranges from 0.0
   to 0.125 across the 31 epochs (3 isolated zeros, 2 peaks
   at 0.125).  The signal is real but noisy; multi-seed
   confirmation is required for any claim beyond Gate-A.
4. **w_ratio asymptotes near 1.0.** Target weights barely
   exceed non-target (`w_ratio ≈ 1.001` from ep 1+).  The
   binding signal lives in the C1 fingerprint geometry
   (which the readout directly probes), NOT in the
   per-class weight magnitude (the iter-66's K4 secondary
   diagnostic).  This is informative but does not affect
   the Gate-A pass: the locked Gate-A criterion is
   `c1_target_top3_overlap`, not w_ratio.
5. **R2 readout unchanged from iter-65 baseline.**  R2's
   `target_top3_overlap` oscillates at chance (~0.031);
   iter-67-γ.1.1's gain comes entirely from the C1 readout
   layer, not from improved R2 binding.  Consistent with
   iter-66's architectural goal (C1 = readout, not R2
   replacement).

## Lock-in

This run's exact configuration is frozen verbatim in
`reports/gate_a_gamma_1_1_config.json` and
`reports/gate_a_gamma_1_1_config.md`.  The 4-seed
confirmation must use these locked parameters with no
modification.
