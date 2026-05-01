# Iter 47a — Postmortem (saturation, cascade pattern, θ effect size)

## Why the postmortem

`notes/47a` ended on a partial result: iter-47a-2's 4-epoch
acceptance sweep found INTER_WEIGHT = 1.0 to be the best sweep
point, but no point passed the 4-of-4 acceptance criteria. The
honest jump to iter-48 (k-WTA) was tempting but speculative.
Bekos's protocol asked for 25 minutes of further diagnostics
*before* writing more code, to choose the iter-48 architecture
from data instead of from intuition. Three questions:

- **(a)** Does iter-47a-2 (INTER_WEIGHT = 1.0 + adaptive θ)
  *saturate* the selectivity at zero (mechanism right but
  exhausted) or *pass through* zero (mechanism right, just slow)?
- **(b)** At INTER_WEIGHT = 0.7, is the cascade an **onset-burst**
  (k-WTA fix) or a **sustained / oscillatory drive** (iSTDP fix)?
- **(c)** Is the Diehl-Cook adaptive-θ effect size large enough,
  in mV, to actually move the needle in this LIF regime — or is
  the mechanism present-but-irrelevant?

All three are now answered by data. The architectural conclusion
**reverses the iter-47-decision-note's default ordering**: the next
step is *not* k-WTA but iSTDP retuning.

## Implementation: the diagnostic surface

Added in this commit:

- `drive_with_r2_clamp_traced(...)` — a parallel of
  `drive_with_r2_clamp` that records, per simulation step, both
  the total R2-E spike count and the canonical-target hit count.
  Used only on the postmortem path; the hot training loop is
  untouched.
- `intrinsic_stats(brain, r2_e_set)` — returns
  `(mean, std, min, max, frac > 1 mV)` of `Network.v_thresh_offset`
  over R2-E. The cleanest single-call read of "did the Diehl-Cook
  rule actually move the threshold by anything that matters
  vs. the 15 mV LIF swing".
- `run_postmortem_diagnostic(corpus, cfg, train_epochs)` — trains
  for `train_epochs` epochs, prints per-epoch `θ_{mean, std, min,
  max, frac > 1 mV}` plus weight stats, then runs ONE additional
  read-only trial that captures the prediction-phase per-step
  trace. Returns the final θ stats.
- CLI flags: `--debug-cascade` switches to the postmortem path,
  `--postmortem-train N` controls epoch count (default 4).

All workspace tests stay green; clippy `-D warnings` clean.

## Question (a) — does iter-47a-2 saturate or pass through?

`reward_benchmark --epochs 16 --reps 4 --teacher-forcing --only reward`
at INTER_WEIGHT = 1.0:

| Epoch | r2_act mean | tgt_hit mean | selectivity |
| ---: | ---: | ---: | ---: |
| 0 | 96.1 | 1.16 | -0.022 |
| 1 | 124.6 | 1.97 | -0.013 |
| 2 | 139.1 | 2.58 | -0.001 |
| **3** | **139.1** | **2.59** | **-0.0005** ← peak |
| 4 | 126.3 | 2.22 | -0.005 ← reversal |
| 5 | 68.8 | 0.11 | -0.040 |
| 6–15 | ≈ 75 | < 0.20 | ≈ -0.046 |

**Verdict:** the selectivity does **not** asymptote to 0 and does
**not** pass through. It approaches zero in epoch 3 (close enough
that one might mistake it for converging) and then **collapses
back** — not just below the iter-46 baseline (-0.04) but slightly
worse (≈ -0.046). `target_hit_mean` follows the same curve:
1.16 → 2.59 (gain) → 0.08 (collapse). The mechanism is **partially
right** (right direction in epochs 0-3) and then **structurally
unstable** (catastrophic interference: the very cells that started
to encode targets get suppressed by something).

This is a stronger pivot signal than "didn't pass acceptance":
it says iter-47a-2 has a negative-feedback loop that fights the
goal once the goal is approached. We need a stabiliser that
*doesn't punish the learning cells*.

## Question (b) — onset-burst or sustained / oscillatory?

`reward_benchmark --reps 4 --teacher-forcing --debug-cascade
--postmortem-train 4` at two INTER_WEIGHT values.

### At INTER_WEIGHT = 1.0 (the stable sweep point)

```
prediction phase per-step trace (200 steps × 0.10 ms = 20 ms):
  step | active | target
     0 |     0 |     0
   ... |   ... |   ...
   100 |     0 |     0
   ... |   ... |   ...
   180 |     3 |     0
   199 |     4 |     0

sums: total_active=140 total_target=0 max_active/step=4
early-vs-late dominance ratio (sum first 25% / sum last 25%): 0.00
```

Activity is **very sparse per step** (~ 0.7 cells/step, max 4) and
**all in the late half** of the prediction window. This is the
"mostly-quiet" regime. Total spike count = 140 over 200 steps,
matching the per-trial r2_active_mean ≈ 96–139 we already
recorded (140 spikes ≈ each unique-active cell firing once or
twice).

### At INTER_WEIGHT = 0.7 (the bistable / cascade regime)

```
prediction phase per-step trace:
  step | active | target
     0 |    10 |     0
    10 |    54 |     0
    20 |    37 |     0
    30 |     8 |     0
    40 |    60 |     0
    50 |    51 |     0
    60 |    11 |     0
    70 |    97 |     0
    80 |    15 |     0
    90 |    16 |     0
   100 |   189 |     0
   110 |    10 |     0
   120 |    59 |     0
   130 |    30 |     0
   140 |    23 |     0
   150 |   180 |     0
   160 |    11 |     0
   170 |    55 |     0
   180 |    45 |     0
   190 |    21 |     0
   199 |   153 |     0

sums: total_active=12001 total_target=0 max_active/step=219 max_target/step=0
early-vs-late dominance ratio: 0.97
```

**12,001 spikes in 20 ms, 60 cells/step on average, max 219 in a
single step**, and crucially:

- early-vs-late ratio = **0.97** ⇒ NOT an onset-burst (would be
  > 2.0). The cascade is *not* "everything fires in the first ms,
  then exhaustion".
- The trace is highly volatile: counts oscillate between ≈ 10 and
  ≈ 200 per step (`10, 54, 37, 8, 60, 51, 11, 97, 15, 16, 189,
  10, 59, 30, 23, 180, …`). This is the classic **synchronised
  recurrent bursting** signature — a neuronal-avalanche pattern,
  not a step-function explosion.
- **0 canonical-target cells fire in the entire 20 ms**, despite
  12,001 R2-E spikes. The target population is *actively
  silenced* (see (c)) while the rest of R2-E avalanches.

**Verdict:** the cascade is NOT an onset-burst. k-WTA per step
would shave the peaks (limit each step to top-k) but **does not
remove the average drive** that produces the avalanche in the
first place — and adding k-WTA AND adaptive θ AND iSTDP is
mechanism-soup that the data does not justify.

The pattern `oscillatory bursting + zero target hits + θ-jump
post-cascade` is a **canonical EI-imbalance signature** —
the very picture Vogels et al. (2011) used to motivate inhibitory
plasticity (iSTDP). The fix the literature points at is *not*
post-hoc spike capping but proactive inhibition that follows
excitation.

## Question (c) — is adaptive θ effect-size large enough to matter?

Per-epoch `intrinsic_stats(brain, r2_e_set)` → `(mean, std, min,
max, frac > 1 mV)` of `v_thresh_offset` over R2-E.

### At INTER_WEIGHT = 1.0 (the "stable" point)

| Epoch | θ mean | θ std | θ max | frac > 1 mV | w̄ | wmax |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.051 | 0.145 | 0.809 | 0.000 | 0.78 | 8.00 |
| 1 | 0.049 | 0.145 | 0.902 | 0.000 | 0.92 | 8.00 |
| 2 | 0.049 | 0.143 | 0.858 | 0.000 | 0.96 | 8.00 |
| 3 | 0.059 | 0.159 | 0.858 | 0.000 | 0.96 | 8.00 |

**Mean θ = 0.05 mV, max θ ≈ 0.9 mV, frac above 1 mV = 0.000.**
The full LIF dynamic range from `v_rest = -70` to
`v_threshold = -55` is **15 mV**. A 0.05 mV mean offset is **0.3%
of dynamic range** — operationally invisible. Even the maximum
0.86 mV is only 5.7%, and only one neuron sits there.

The Diehl-Cook mechanism is firing (we see non-zero θ values),
but the effect size is so small relative to the LIF threshold
window that **it cannot meaningfully shape per-cell selectivity
in this regime**. The (a) collapse therefore is not Diehl-Cook
suppressing the learning cells — it is some other process. By
elimination of the candidate mechanisms (homeostasis is gentle,
adaptive θ is too small to matter), the most likely actor is
**iSTDP misalignment**: the inhibitory pool is firing on the
same cells that just learned to be informative, suppressing them
in the next trial.

### At INTER_WEIGHT = 0.7 (the cascade regime)

| Epoch | θ mean | θ std | θ max | frac > 1 mV | w̄ | wmax |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.028 | 0.106 | 0.667 | 0.000 | 0.46 | 8.00 |
| 1 | 0.031 | 0.108 | 0.716 | 0.000 | 0.59 | 8.00 |
| 2 | 0.035 | 0.111 | 0.716 | 0.000 | 0.67 | 8.00 |
| **3** | **2.841** | 0.088 | **3.039** | **0.999** | 0.55 | 8.00 |

**The θ behaves catastrophically reactively at 0.7.** Epochs 0-2
look identical to the 1.0 case (θ_mean ≈ 0.03 mV, frac > 1 mV
= 0). Then in epoch 3 — the cascade epoch — θ_mean jumps to
**2.84 mV (95× the prior value)** and **99.9 % of all R2-E cells
have θ > 1 mV**. That is the avalanche: every cell that
participated raised its own threshold, mostly to the offset
ceiling (offset_max = 5; we hit 3.04). And the consequence we
already saw: in the postmortem prediction trial, **0 canonical
target cells fire** because their θ is now elevated like
everyone else's.

**Verdict:** Diehl-Cook's effect size is *too small to prevent the
cascade*, then *too large after the cascade* (catches up by
elevating everything together). Tuning `alpha_spike` or
`tau_adapt` will not fix this — the mechanism is per-cell-rate-
limited, the bottleneck is per-network-balance.

## What this rules in / rules out for iter-48

**Ruled out by (b)+(c) data:**

- **k-WTA per step (the iter-47-decision-note's default
  fallback).** Spike volume at 0.7 is oscillatory, not
  onset-burst. Per-step k-WTA would clip the peaks but not
  remove the recurrent imbalance that drives the avalanche.
  The cascade would re-form on the next trial.
- **Tightening Diehl-Cook (`α_spike ↑` or `offset_max ↑`).**
  Effect size is already too small to prevent the cascade;
  scaling it up just makes the post-cascade flattening worse
  (every learning cell gets pinned at offset_max).

**Ruled in by (a)+(b)+(c) data:**

- **Tighter iSTDP (Vogels et al. 2011, fast EI balance).**
  Oscillatory bursting + θ behaving reactively + zero target
  firing in the cascade are exactly the symptoms iSTDP was
  invented to fix. The harness already has iSTDP enabled with
  parameters tuned for slow homeostasis (`a_minus = 1.10`,
  `tau_minus = 30 ms`) — iter-48 should retune for *fast*
  cascade suppression: lower `tau_minus` (≤ 10 ms), higher
  `a_plus`, and possibly more inhibitory cells
  (`R2_INH_FRAC = 0.30` instead of 0.20).
- **Inhibitory plasticity *during the prediction phase*.**
  Iter-46 disables STDP/iSTDP during prediction to keep the
  read-out clean; with proper iSTDP that response time is
  the main lever, this needs at least a feature flag.

**Open after this postmortem (but no longer blockers):**

- Whether bounded R-STDP on R1 → R2 (47b) helps once iSTDP
  stabilises the recurrent dynamics. We can A/B-test with the
  existing harness once iter-48 lands.
- Whether the canonical-target SDR size (30 cells) needs to
  match the natural inhibitory budget. 30 / 1600 R2-E = 1.9 %
  is in the HTM SP / hippocampal DG ballpark; should be fine
  if iSTDP works.

## Iter-48 entry, evidence-driven

```
1. R2_INH_FRAC: 0.20 → 0.30                  (more inhibition)
2. IStdpParams.tau_minus: 30 ms → 8 ms       (fast response)
3. IStdpParams.a_plus: 0.10 → 0.30           (stronger LTP on
                                              silent E targets)
4. Optional: enable iSTDP during prediction  (CLI flag for A/B)
5. Keep INTER_WEIGHT = 1.0
6. Keep IntrinsicParams off (effect size too small to matter)
   — OR keep them as a per-cell tie-breaker but acknowledge
     the heavy lifting is done by iSTDP.
```

Acceptance criteria for iter-48 (proposed, fix before phase 1):

- After 4 epochs: `selectivity_index > 0.0` AND `target_hit_mean > 5`
  AND no cascade (max per-step active < 50). 3-of-3 ⇒ phase 2.
- After 16 epochs: monotone selectivity *or* monotone target_hit
  growth (no collapse like iter-47a-2 produced).

The iter-47 sparsity diagnostics + the new postmortem
instrumentation are already wired to A/B-test all of these
without further code. iter-48 phase 0 should be roughly the same
size as iter-47a-2 phase 0 (~30 lines of parameter changes plus
a CLI flag), not a refactor.
