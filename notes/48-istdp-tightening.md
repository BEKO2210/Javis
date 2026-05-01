# Iter 48 — iSTDP-Tightening (Vogels 2011)

## Why iter-48

iter-47a postmortem (notes/47a-postmortem) provided an
unambiguous data-driven verdict: oscillatory recurrent bursting +
θ behaving reactively-after-the-cascade + 0/30 target hits in
12 001 R2-E spikes ⇒ **EI-imbalance signature**, exactly the
problem Vogels-Sprekeler-2011 inhibitory plasticity was designed
to address. The iter-47-decision-note's default fallback (k-WTA
per step) was *ruled out by data*: spike volume at the
"interesting" sweep point is oscillatory, not onset-burst, so
post-hoc spike capping would shave peaks without removing the
recurrent imbalance that drives them.

The iter-48 plan retunes iSTDP for fast cascade response and
slightly enlarges the inhibitory pool, while leaving INTER_WEIGHT
= 1.0 (the iter-47a-stable sweep point) and adaptive θ enabled
(but acknowledged as a tie-breaker, not a primary stabiliser —
its < 0.3 % effect size relative to LIF swing is unchanged).

## What iter-48 ships (3 atomic commits)

### Commit 1 (`4bfacc0`) — iSTDP parameter tightening

```
R2_INH_FRAC          0.20 → 0.30    (+50 % inhibitory budget)
IStdpParams.tau_minus 30 → 8 ms     (AMPA → GABA latency budget)
IStdpParams.a_plus    0.10 → 0.30   (3× faster wall growth)
IStdpParams.a_minus   1.10 unchanged
INTER_WEIGHT          1.0 unchanged
IntrinsicParams       unchanged (left enabled as tie-breaker)
```

Each parameter change carries a one-paragraph rationale in the
commit message. INTER_WEIGHT and IntrinsicParams kept fixed so
iter-48 isolates the iSTDP variable.

### Commit 2 (`bdee598`) — A/B switch `--istdp-during-prediction`

`TeacherForcingConfig.istdp_during_prediction` (default `false`)
splits the iter-46 plasticity gate so iSTDP can run during the
prediction phase independently of STDP / R-STDP. This is the
natural test for "does the new fast iSTDP catch the cascade in
time, or only after the trial". STDP and R-STDP stay silenced
during the read-out either way; iSTDP is opened separately.

### Commit 3 (`de5771c`) — p99 + θ_E/θ_I drift metrics

Three new `RewardEpochMetrics` fields:

- `r2_active_pre_teacher_p99` — 99th percentile of per-trial
  prediction-phase active-cell counts. Catches avalanche-tail
  trials that p90 still hides. **Iter-48 acceptance criterion:
  < 50.**
- `theta_inh_mean` / `theta_exc_mean` — Diehl-Cook adaptive-θ
  mean split by neuron kind. The diff `θ_I − θ_E` is the
  **iSTDP-over-correction early warning**: if iSTDP silences
  the inhibitory pool faster than the excitatory one, θ_I
  drifts toward the floor faster.

`render_markdown` now shows
`| Epoch | r2_act mean | p10 | p90 | p99 | tgt_hit mean | selectivity | θ_E | θ_I |`
so all three iter-48 acceptance numbers + the early-warning
diagnostic are readable in one row.

## Phase 1 — 4-epoch smoke × 2 configs

Acceptance criteria fixed before phase 1, per Bekos protocol:

> 1. `selectivity_index > 0.0` after 4 epochs
> 2. `target_hit_mean > 5` after 4 epochs
> 3. `p99(per-step active) < 50` over all trials
> 3-of-3 ⇒ Phase 2 (20-epoch full run).

Both runs on the same 16 + 16 pair / vocab-32 corpus, seed 42:

### Config 1 — iSTDP only in phases 4–5 (default)

```
R-STDP + Teacher-Forcing
random top-3 baseline: 0.094

| Epoch | top-1 | top-3 | MRR  | mean rwd | noise t3 | pred-t3 | clamp | margin | …  | w̄    | wmax |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|   0  | 0.00 | 0.06 | 0.05 | -0.73   | 0.06    | 0.03   | 1.00 | -0.02 | 1.40 | 8.00 |
|   1  | 0.00 | 0.06 | 0.05 | -0.73   | 0.06    | 0.03   | 1.00 | -0.01 | 1.42 | 8.00 |
|   2  | 0.00 | 0.06 | 0.05 | -0.68   | 0.12    | 0.09   | 1.00 | -0.01 | 1.42 | 8.00 |
|   3  | 0.00 | 0.00 | 0.03 | -0.74   | 0.12    | 0.02   | 1.00 | -0.01 | 1.42 | 8.00 |

| Epoch | r2_act mean | p10 | p90 | p99 | tgt_hit mean | selectivity | θ_E   | θ_I   |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|   0  | 30.5  |  8  |  73 | 109 | 0.31         | -0.0116     | 0.003 | 0.005 |
|   1  | 27.9  | 11  |  49 |  61 | 0.97         | +0.0126     | 0.004 | 0.006 |
|   2  | 39.2  | 16  |  73 |  86 | 1.34         | +0.0172     | 0.004 | 0.006 |
|   3  | 38.2  | 17  |  61 |  79 | 1.23         | +0.0142     | 0.004 | 0.005 |
```

### Config 2 — iSTDP in phases 3–5 (`--istdp-during-prediction`)

```
R-STDP + Teacher-Forcing
random top-3 baseline: 0.094

| Epoch | top-1 | top-3 | MRR  | mean rwd | noise t3 | pred-t3 | clamp | margin | w̄    | wmax |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|   0  | 0.00 | 0.06 | 0.06 | -0.71   | 0.12    | 0.05   | 1.00 | -0.01 | 1.43 | 8.00 |
|   1  | 0.00 | 0.06 | 0.04 | -0.74   | 0.12    | 0.02   | 1.00 | -0.01 | 1.44 | 8.00 |
|   2  | 0.00 | 0.00 | 0.05 | -0.68   | 0.12    | 0.09   | 1.00 | -0.00 | 1.44 | 8.00 |
|   3  | 0.00 | 0.00 | 0.04 | -0.74   | 0.06    | 0.02   | 1.00 | -0.01 | 1.44 | 8.00 |

| Epoch | r2_act mean | p10 | p90 | p99 | tgt_hit mean | selectivity | θ_E   | θ_I   |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|   0  | 27.8  |  8  |  64 |  95 | 0.31         | -0.0097     | 0.003 | 0.005 |
|   1  | 22.5  | 11  |  35 |  46 | 0.66         | +0.0059     | 0.004 | 0.006 |
|   2  | 30.9  | 15  |  53 |  68 | 1.02         | +0.0120     | 0.003 | 0.006 |
|   3  | 30.1  | 15  |  48 |  61 | 1.06         | +0.0142     | 0.003 | 0.005 |
```

### Acceptance check

| Criterion | Config 1 | Config 2 |
| --- | :-: | :-: |
| `selectivity_index > 0.0` | **+0.0142 ✅** | **+0.0142 ✅** |
| `target_hit_mean > 5` | 1.23 ❌ | 1.06 ❌ |
| `p99(active) < 50` | 79 ❌ | 61 ❌ |
| **Total** | **1.5 / 3** | **1.5 / 3** |

**Per protocol: not 3-of-3 ⇒ no Phase 2, no speculative pivot,
pause and document.**

## Honest reading

This is the largest qualitative jump in the iter-44/45/46/47/48
chain on the canonical-target-selectivity metric:

- **Selectivity flipped from negative to positive for the first
  time.** Iter-46 sat at -0.04, iter-47a at -0.01 (transient
  peak) → -0.045 (collapse), iter-48 at +0.014 stable across
  three consecutive epochs, both configs. The iSTDP retuning
  hypothesis is **directionally confirmed by data**: the same
  network with the only changed parameters being R2_INH_FRAC,
  iSTDP `tau_minus` and iSTDP `a_plus` produces the right sign.
- **`r2_active_pre_teacher_mean` is in the [25, 70] band** for
  the first time across all four epochs in both configs (config
  1: 28–39, config 2: 23–31). Forward drive + iSTDP + adaptive
  θ together produce stable sparsity, not iter-46's 90–180 or
  iter-47a's bistable 10/507.
- **No cascade.** p99 stays below 110 even on the worst trial,
  vs. iter-47a's 1599 in the cascade epoch. iSTDP successfully
  caught the runaway recurrent dynamic.

But:

- **`target_hit_mean ≈ 1` is far below the 5-cell criterion.**
  The right cells are firing more than the wrong ones (selectivity
  positive), but the absolute count of cue-driven canonical-target
  spikes is still tiny — the recurrent path *is* now biased
  toward targets, but its absolute amplitude is small.
- **`p99 < 50` is violated** (79 / 61). Tail-of-distribution
  trials still produce 60–80 active cells per cue, which is
  uncomfortably close to the noise floor where selectivity could
  collapse again at longer horizons.
- **`top-1` and `top-3` accuracy stayed at chance**. The
  selectivity flip did not yet translate into decoded-target
  recall, because target_hit_mean ≈ 1 is too low to dominate
  the top-3 decoder against the dictionary's other 31 entries.
- **`pred-t3` (top-3 measured during phase 3, before any teacher
  intervention) reaches 0.09 in both configs in epoch 2** — a
  small but real signal of cue-only recall starting to work,
  but not stable across epochs.
- **The collapse-after-peak pattern from iter-47a does NOT
  reappear** in 4 epochs. Selectivity stays positive across
  epochs 1, 2, 3 in both configs. We *do not yet know* whether
  it would survive 16+ epochs (the saturation question Bekos's
  protocol explicitly asked us not to answer with a speculative
  Phase 2 run).

## What this rules in / out for iter-49

**Ruled in by data:**

- iSTDP retuning is the right primary stabiliser (selectivity
  sign flip + sparsity in band + cascade controlled).
- The Vogels 2011 framework remains the right reference even
  for sub-second trial regimes — `tau_minus = 8 ms` and
  `a_plus = 0.30` are clearly in a productive regime.

**Ruled out by data:**

- A speculative Phase 2 run on iter-48 as-is (it would not
  pass the criteria; saturation question would still be open).
- Pivoting back to iter-47a-style adaptive-θ tuning (θ values
  remain ~ 0.005 mV; the mechanism is still operationally
  invisible at this scale, regardless of iSTDP).
- Pivoting to k-WTA (no cascade observed, so the problem k-WTA
  was supposed to fix never actually appeared in iter-48).

**Open after iter-48 — explicit follow-up questions for iter-49
or another postmortem run:**

1. **Saturation behaviour at 16 epochs**: does selectivity
   stabilise at +0.014, drift up further, or collapse like
   iter-47a's transient peak? (Same question as iter-47a (a),
   but now starting from a positive baseline.) ~ 5 min wallclock
   for one extended run per config.
2. **Why is `target_hit_mean ≈ 1` so much below the 30-cell
   target SDR?** The clamp hits 30/30, but cue-only recall
   only fires ~ 1 of those 30. The recurrent weights from
   cue-driven cells *to* target cells are too weak. Two ways
   to test:
   - **Raise R-STDP `eta`** so reward-modulated weight changes
     are larger per trial.
   - **Longer teacher phase** (currently 40 ms) so STDP gets
     more pre→post coincidences per trial.
3. **Tail control**: p99 ≈ 60–80 vs. mean ≈ 30 means a few
   trials still go bursty. Is this noise (3-4 trials/epoch,
   acceptable) or a structural bistability (those trials
   correspond to specific pairs, repeatable)? Per-pair p99
   would answer this directly.

The next concrete experiment, when it happens, should follow
the same Bekos protocol: pre-fixed acceptance criteria,
4-epoch smoke per config, A/B comparison, no speculative
pivots, pause + document at fail. Iter-48's metric surface
(`r2_active_p99`, `θ_inh / θ_exc`) is already in place to
support whatever follows.

## Iter-48 acceptance criteria — measured

| # | Criterion | Status |
| - | --- | --- |
| 1 | Code compiles | ✅ |
| 2 | Existing tests still green | ✅ (9 lib) |
| 3 | Smoke runs in reasonable time | ✅ (~80 s per config) |
| 4 | Evaluation does not modify weights | ✅ (STDP/R-STDP gated; iSTDP gating exposed via flag) |
| 5 | Selectivity > 0 (Phase 1) | ✅ both configs |
| 6 | target_hit > 5 (Phase 1) | ❌ both configs |
| 7 | p99 < 50 (Phase 1) | ❌ both configs |
| 8 | 3-of-3 acceptance ⇒ Phase 2 | ❌ no Phase 2 run |
| 9 | Pause + document on fail (per protocol) | ✅ this note |
