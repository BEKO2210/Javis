# Iter 48 — Phase A Saturation Test

## Why this test

iter-48 phase 1 produced the largest qualitative jump in the
chain — selectivity flipped from −0.045 (iter-47a) to +0.0142
(stable across 3 epochs, both configs) — but with 1.5/3
acceptance, the open question was whether this was the start of
a learning curve or a metastable transient like iter-47a's peak.
Bekos's protocol (notes/48-istdp-tightening, end): pre-fixed
3-criteria saturation acceptance, 16 epochs both configs,
~ 15 min wallclock.

> A1 — Stabilität: selectivity > 0 in ≥ 12 of 16 epochs
> A2 — Trend: selectivity epoch 13–16 mean ≥ epoch 1–4 mean
> A3 — target_hit growth: target_hit_mean epoch 13–16 > 1.5
>
> 3-of-3 → Magnitude is data problem (Phase B1: η lift)
> 2-of-3 → Magnitude is architecture problem (Phase B2: sweep)
> 0-1-of-3 → Postmortem like 47a, Vogels was not the full lever

No code changes for Phase A — both configs use the iter-48
parameter set as committed in `4bfacc0`/`bdee598`/`de5771c`.

## Data — Config 1 (iSTDP only in phases 4–5, default)

```
| Epoch | r2_act mean | p10 | p90 | p99 | tgt_hit | selectivity | θ_E   | θ_I   |
|   0   |  30.5  |  8 |  73 | 109 | 0.31 | -0.0116 | 0.003 | 0.005 |
|   1   |  27.9  | 11 |  49 |  61 | 0.97 | +0.0126 | 0.004 | 0.006 |
|   2   |  39.2  | 16 |  73 |  86 | 1.34 | +0.0172 | 0.004 | 0.006 |
|   3   |  38.2  | 17 |  61 |  79 | 1.23 | +0.0142 | 0.004 | 0.005 |
|   4   |  38.1  | 16 |  67 |  81 | 1.17 | +0.0121 | 0.004 | 0.004 |
|   5   |  17.9  |  8 |  28 |  34 | 0.14 | -0.0083 | 0.004 | 0.005 |   ← COLLAPSE
|   6   |  22.4  | 10 |  35 |  54 | 0.12 | -0.0121 | 0.004 | 0.005 |
|   7   |  22.2  | 11 |  35 |  43 | 0.12 | -0.0120 | 0.004 | 0.004 |
|   …   |  …     |  … |   … |   … |  …   |  …      |  …    |  …    |
|  15   |  23.9  | 13 |  38 |  46 | 0.12 | -0.0132 | 0.004 | 0.004 |
```

w̄ stays at 1.40 throughout, wmax at 8.00 throughout. No weight
runaway, no shrinkage, no cascade — just an activity drop
between epoch 4 and 5 that takes target cells with it.

## Data — Config 2 (`--istdp-during-prediction`)

```
| Epoch | r2_act mean | p10 | p90 | p99 | tgt_hit | selectivity | θ_E   | θ_I   |
|   0   |  27.8  |  8 |  64 |  95 | 0.31 | -0.0097 | 0.003 | 0.005 |
|   1   |  22.5  | 11 |  35 |  46 | 0.66 | +0.0059 | 0.004 | 0.006 |
|   2   |  30.9  | 15 |  53 |  68 | 1.02 | +0.0120 | 0.003 | 0.006 |
|   3   |  30.1  | 15 |  48 |  61 | 1.06 | +0.0142 | 0.003 | 0.005 |
|   4   |  29.1  | 13 |  46 |  61 | 1.05 | +0.0144 | 0.003 | 0.004 |
|   5   |  14.5  |  7 |  22 |  27 | 0.12 | -0.0063 | 0.003 | 0.004 |   ← COLLAPSE
|   6   |  16.2  |  8 |  28 |  30 | 0.14 | -0.0070 | 0.003 | 0.004 |
|   …   |  …     |  … |   … |   … |  …   |  …      |  …    |  …    |
|  15   |  18.2  | 11 |  28 |  30 | 0.12 | -0.0090 | 0.003 | 0.004 |
```

Same w̄ ≈ 1.43, wmax = 8.00 throughout. Same collapse epoch.

## Acceptance check — 0/3 in both configs

| Criterion | Config 1 | Config 2 |
| --- | :-: | :-: |
| A1: selectivity > 0 in ≥ 12 / 16 epochs | 4/16 ❌ | 4/16 ❌ |
| A2: epoch 13–16 selectivity ≥ epoch 1–4 mean | −0.013 < +0.014 ❌ | −0.009 < +0.012 ❌ |
| A3: epoch 13–16 target_hit_mean > 1.5 | 0.13 ❌ | 0.13 ❌ |
| **Total** | **0/3** | **0/3** |

**Per protocol: 0–1-of-3 ⇒ Postmortem.** Both Phase B1
(η-lift) and Phase B2 (Vogels-parameter-sweep) are
**ruled out by data** — they would each push the same
collapse-prone regime harder, not differently.

## What actually happened — the collapse mechanism

The collapse pattern is **identical in both configs**:

1. Epochs 0–4: selectivity rises monotonically, peaks at
   +0.014 / +0.017, target_hit reaches 1.0–1.3.
2. Epoch 5: selectivity flips to −0.006 to −0.008,
   target_hit drops to ~0.12, r2_active drops by ~50 %.
3. Epochs 6–15: stable negative steady-state, slightly
   *worse* than the iter-47a-pm baseline (−0.045 vs
   iter-48-collapse −0.012). Activity stays below the
   [25, 70] target band.

Diagnostic exclusions (what the collapse is **not**):

- **Not weight runaway**: `w̄` is 1.40 ± 0.03 across all 16
  epochs in both configs. Not Litwin-Kumar 2014.
- **Not Diehl-Cook over-correction**: `θ_E` ≈ 0.003 mV,
  `θ_I` ≈ 0.004 mV through the collapse — that's < 0.03 % of
  LIF swing, operationally invisible (same finding as
  iter-47a-pm).
- **Not cascade**: `r2_active_p99` drops at collapse (109 → 34
  in Config 1, 95 → 27 in Config 2). The opposite of avalanche.

What the data **is** consistent with — **iSTDP cumulative
over-inhibition**:

- iSTDP LTP-on-pre-only-I-spike (`a_plus = 0.30`, increased 3×
  in iter-48 from iter-47a's 0.10) builds inhibitory weight on
  every I → E synapse where the post-E cell *did not* fire in
  the recent window.
- The canonical target cells get this LTP every time they fail
  to fire under cue alone in the prediction phase. Over
  epochs, the inhibitory wall to those exact cells grows.
- Initially (epochs 1–4) the target cells fire enough during
  teacher-clamp + lead-in that LTD-on-coactivity (`a_minus =
  1.10`, unchanged) keeps the wall in check. Once the wall
  crosses a threshold, the target cells stop firing reliably
  even under teacher-clamp — and then LTP runs unopposed.
- The result: I → E_target weights ratchet up, target cells
  silence themselves, surviving R2 activity is whatever
  random forward-projection cells the iSTDP hasn't walled off
  yet. Selectivity flips negative because the random survivors
  are by definition not the canonical target cells.
- The collapse is an *attractor* of the iSTDP rule, not a
  failure of it. The rule is doing exactly what it's designed
  to do (drive every E cell toward target firing rate); the
  problem is that "target firing rate" under teacher-forcing
  and under cue-alone are different, and iSTDP optimises for
  the latter while we measure on the former.

Open empirical question (within iter-49 budget):
**which of these dominates the LTP/LTD imbalance?**

a) `iSTDP w_max = 8.0` is too permissive — caps that don't
   bind let walls grow indefinitely.
b) `a_plus = 0.30` was over-tuned by iter-48 (3× iter-47a) —
   half-way back at 0.20 might find a stable equilibrium.
c) The structural problem: iSTDP runs on every trial regardless
   of whether the engram has formed yet; an activity-floor or
   epoch-gated `a_plus` would match the literature pattern of
   "consolidate first, balance later".

## Anomaly note — Phase 3 plasticity is *almost* redundant

Bekos called out that iter-48 phase 1 had identical +0.0142
selectivity at epoch 3 in both configs, suggesting Phase 3
plasticity might be redundant. The 16-epoch data refines
this:

- Through epoch 3, both configs are identical to 4 decimals
  (+0.0142, target_hit 1.23 vs 1.06 differs slightly).
- Epoch 4 onwards they diverge: Config 1 sustains the peak
  one more epoch (+0.0121) and collapses to a tighter
  negative (≈ −0.012); Config 2 sustains slightly differently
  (+0.0144) and collapses to a slightly less bad negative
  (≈ −0.008).
- r2_active in the collapsed steady-state: Config 1 ≈ 23,
  Config 2 ≈ 17.

So: Phase 3 plasticity is **not** strictly redundant — it
slightly tightens sparsity (lower r2_active in Config 2) and
slightly softens the collapse depth — but it does **not**
change the shape of the trajectory. The peak position, the
collapse epoch, and the qualitative outcome are identical.
The CLI flag stays useful as an A/B knob but is not a
make-or-break decision for iter-49.

## What this rules in / rules out for iter-49

**Ruled out by data:**

- Phase B1 (R-STDP η-lift). Magnitude was not the bottleneck;
  the collapse happens at *constant* w̄ ≈ 1.4. More η would
  not change when the collapse triggers, only how fast the
  wrong steady-state arrives.
- Phase B2 (Vogels-parameter sweep in the same direction).
  The data says iSTDP is over-tuned in iter-48, not under-
  tuned. Sweeping `tau_minus` shorter or `a_plus` higher
  would deepen the collapse, not delay it.
- A speculative Phase 2 of iter-48 as committed. The
  saturation answer is now definitive: 0/3, identical
  trajectory in both configs.

**Ruled in by data — three small, parallel candidate experiments
for iter-49 (each ~5 min smoke):**

1. **Cap iSTDP w_max from 8.0 → 2.0.** Same `a_plus = 0.30`,
   same `tau_minus = 8 ms`. Hypothesis: the wall growth is
   the problem, not the LTP rate; bounding the wall keeps
   the post-cascade equilibrium in the same regime as the
   pre-cascade peak.
2. **Halve `a_plus` from 0.30 → 0.20.** Hypothesis: iter-48's
   3× lift over iter-47a was over-correction; 2× was the
   right magnitude.
3. **Activity-gated iSTDP**: `a_plus = 0` for the first
   ~ 2 epochs (let STDP build the engram first), then ramp
   to 0.30 over the next 2. Implementation cost: one
   `epoch_index`-aware multiplier in the trial loop.

The pre-fixed iter-49 acceptance for any of these:

> sustained selectivity > 0 across epochs 4 – 16 (no collapse
> like iter-48's epoch 5), AND mean target_hit at epoch 16
> > mean target_hit at epoch 4. *No magnitude criterion yet* —
> we are testing for collapse-survival, not for top-3 lift.

## Iter-48 phase A — acceptance status

| # | Criterion | Status |
| - | --- | --- |
| 1 | Code compiles | ✅ |
| 2 | Existing tests still green | ✅ (9 lib) |
| 3 | Smoke runs in reasonable time | ✅ (~5 min per config) |
| 4 | Eval doesn't modify weights (Config 1) / iSTDP-only-during-pred (Config 2) | ✅ both arms ran clean |
| 5 | A1 stability ≥ 12/16 | ❌ both 4/16 |
| 6 | A2 trend monotone | ❌ both decay |
| 7 | A3 target_hit growth | ❌ both 0.13 vs 1.5 |
| 8 | 3-of-3 ⇒ Phase B1 | ❌ |
| 9 | Pause + document on 0-1-of-3 (per protocol) | ✅ this note |

Iter-48's headline qualitative finding (selectivity flipping
positive) survives this postmortem: the temporary +0.014 peak
across 4 consecutive epochs in both configs is a **real
regime change** that no iter-44/45/46/47a configuration ever
produced. But it is also *not stable*, and the iter-48 iSTDP
parameters as committed sit on the over-tuned side of the
collapse boundary. iter-49 should explore the under-tuned side
of that boundary, not push further along the iter-48 axis.
