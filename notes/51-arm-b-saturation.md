# Iter 51 — Arm B Saturation (Schritt 1 vor jeder Parameterstudie)

## Why iter-51 Schritt 1

iter-50 reproduced iter-46's reported `top-3 = 0.19` for R-STDP
Arm B in **epoch 0** of a 4-epoch smoke. That was strong enough
to invalidate the iter-47/48/49 architectural pivot, but it left
open the most basic question Bekos's protocol kept asking and
that we kept answering with parameter sweeps:

> Is `0.19` a stable operating point, the start of a learning
> curve, or a metastable transient that decays — like iter-47a's
> selectivity peak did?

iter-46 itself reported `0.19` after running 4 epochs and
declared it as "stable across consecutive epochs". That was a
4-data-point claim. The real question is what happens at epoch
5, 8, 12, 16 — exactly the saturation question we asked of
iter-47a and iter-48 selectivity.

This iter is a **single 16-epoch run with `--iter46-baseline`,
no `--teacher-forcing`**, no code change. The most diagnostically
valuable run in the chain because it reproduces the only
"working" baseline against the real time-axis.

## Data — full 16 epochs

### Arm A (pure STDP, control)

```
Epoch top-1 top-3  MRR   r2_act  tgt_hit  selectivity
  0   0.00  0.00  0.09    68.6     1.19   -0.0096
  1   0.00  0.06  0.07    51.1     0.88   -0.0075
  2   0.00  0.06  0.06    47.8     0.88   -0.0050
  3   0.00  0.00  0.05    46.2     0.75   -0.0082
  4   0.00  0.06  0.05    45.0     0.75   -0.0067
  5–15 stabilises around r2_act 44, tgt_hit 0.75, top-3 ≤ 0.06
```

### Arm B (R-STDP, the one we care about)

```
Epoch top-1 top-3  MRR   mean rwd  r2_act  tgt_hit  selectivity
  0   0.00  0.19  0.14   -0.59     173.4    3.00    -0.0085
  1   0.00  0.12  0.09   -0.58     143.4    2.38    -0.0107
  2   0.00  0.06  0.08   -0.53     143.8    2.38    -0.0109
  3   0.00  0.06  0.11   -0.53     144.1    2.44    -0.0090
  4   0.00  0.12  0.11   -0.55     141.9    2.50    -0.0054
  5   0.00  0.06  0.07   -0.55     141.6    2.50    -0.0053
  6   0.00  0.12  0.10   -0.55     145.5    2.50    -0.0077
  7   0.00  0.12  0.10   -0.59     145.8    2.50    -0.0079
  8   0.00  0.19  0.13   -0.61     149.6    2.50    -0.0104
  9   0.00  0.19  0.12   -0.64     149.1    2.44    -0.0122
 10   0.00  0.06  0.11   -0.61     149.0    2.56    -0.0079
 11   0.00  0.06  0.08   -0.59     147.3    2.75    -0.0004
 12   0.00  0.12  0.12   -0.64     151.2    2.75    -0.0029
 13   0.00  0.06  0.10   -0.62     147.8    2.50    -0.0092
 14   0.00  0.06  0.09   -0.64     147.0    2.31    -0.0151
 15   0.00  0.12  0.13   -0.67     152.6    2.75    -0.0038
```

`top-1` is **0.00 in every single epoch** — the network never
puts the right target as its single best guess. `top-3` oscillates
between 0.06 and 0.19 with no monotone trajectory.

## The hard read — `0.19` is noise, not a learning signal

**top-3 mean over 16 epochs: 0.107**
**Random baseline: 3/32 = 0.094**
**Difference: 0.013**

Statistical test: with 16 pairs evaluated per epoch and
`p ≈ 0.107`, the per-epoch Bernoulli standard deviation is
`sqrt(p(1−p)/n) = sqrt(0.107 × 0.893 / 16) ≈ 0.077`.

The 95 % confidence interval for the per-epoch mean is
`0.107 ± 1.96 × (0.077 / sqrt(16))` ≈ `0.107 ± 0.038` =
**[0.069, 0.145]**.

**Random 0.094 sits inside this CI.** The Arm B mean is **not
statistically distinguishable from random**.

## What the secondary metrics confirm

`mean reward ≈ −0.60` across all 16 epochs. With the per-trial
reward schedule `+1 / 0 / −0.5 / −1` and a roughly random
distribution of "predicted target was in top-3, in noise top-3,
or absent", the *expected* mean reward under random performance
is approximately:
- 16 real pairs: with `p_top3 = 0.094` (random), each trial's
  expected reward is roughly `0.094 × 1.0 + 0.906 × (−0.5)` ≈ `−0.36`.
- 16 noise pairs: always `−1.0`.
- combined mean: `(−0.36 − 1.0) / 2 = −0.68`.

Observed `−0.60` is slightly *less negative* than random by
about 0.08. That gap is real but very small — consistent with
"the brain is doing something, but only barely above chance".

`r2_active` stays at ≈ 145 across epochs 1–15 — no growth, no
decay. STDP weight `w̄` sits at 1.13, `wmax` at 8.00 (iSTDP cap
hit), all stable. There is no learning *trajectory* in any of
the recorded metrics; everything plateaus at epoch 1 and stays.

`tgt_hit_mean` rises slowly from 2.38 to 2.75 over 16 epochs —
a 16 % increase — but starts and ends both *below* the random
expectation of `r2_active × 30 / 1400 ≈ 145 × 0.0214 = 3.10`.
Even the slow rise stays sub-random.

## Outcome classification — beyond Bekos's three-option schema

Bekos asked: stable, rising, or falling? The data answers
**none of these in a meaningful sense**:

- Not rising (no monotone trend, mean stable at 0.107 across
  epochs 1–15).
- Not falling (no metastable peak-then-collapse like
  iter-47a/48; the oscillation is noise around the same level
  throughout).
- Not stable in the "operating point" sense either — top-3
  jumps between 0.06 and 0.19 from epoch to epoch with no
  predictable structure.

**It is stable as a noisy distribution that is statistically
indistinguishable from random.** The "operating point" is
"chance ± noise".

## What this means for the entire iter-44…50 chain

This is the most uncomfortable result in the chain. Reading it
honestly:

1. **iter-46 reported `top-3 = 0.19` after a 4-epoch run and
   we — and the original iter-46 commit message — interpreted
   that as a working baseline.** It was the highest of four
   noise samples from a distribution whose mean is at chance.
2. **Every subsequent iteration (47/48/49) measured against a
   selectivity metric we now know is structurally meaningless
   in the no-teacher path.** When teacher-forcing was added,
   we measured a *different* (also chance-level) noise
   distribution and judged it relative to a baseline that was
   itself chance.
3. **The whole iter-44…50 chain has, by 95 % CI, never
   demonstrated learning above chance** on this benchmark.
   Mean reward gaps of `0.08 vs −0.68` and `tgt_hit slow rise
   from 2.38 to 2.75` are real but tiny — at most "the system
   is doing 5–10 % better than a randomly-initialised brain".

This is not the same as "we made no progress". The methodology
is now sound, the diagnostic surface is rich, the regression
guards exist. But the *learning claim* that motivated everything
since iter-44 is **not supported by the current data**.

## Why Schritt 2 + Schritt 3 are now the wrong next step

Bekos's iter-51 plan was:
- Schritt 2: decoder-relative metric
- Schritt 3: 3-point parameter sweep on Arm B

Both presupposed that Arm B's `0.19` was a real signal and that
the question was "ceiling vs starting point". The data answers
that presupposition: **neither**. Sweeping `reps_per_pair`,
`R-STDP eta`, `w_max` against a chance-level baseline would
either:

- find no improvement (most likely; if the system isn't learning
  at all, parameters within the same family won't help), or
- find a "winner" that looks better at one seed but is also
  noise (the Arm B `0.19` epoch-0 trap, repeated).

Either way the methodology yields no learning gradient.

The decoder-relative metric (Schritt 2) is still useful — but
its first job is now different: **distinguish "the brain learnt
something small" from "the brain is at chance and we're reading
fluctuations"**. A trial-to-trial consistency metric on the
decoded top-k would show whether the same cue produces the
same predictions across reps; under chance it doesn't.

## Concrete iter-52 entry — statistical validation, not parameter sweep

The bottleneck has moved from "which mechanism" to "is there
any learning at all". Three small experiments to find out, none
require new architecture:

1. **Multi-seed Arm B baseline** (~ 30 min wallclock). Run
   `--iter46-baseline` 16 epochs at seeds 41, 42, 43, 44, 45.
   Check whether the per-seed `top-3 mean` distribution is
   centred at `0.094` (= no learning) or higher. With 5 seeds ×
   16 epochs = 80 epoch-samples, the standard error of the
   mean drops to `0.077 / sqrt(80) ≈ 0.009`. Then
   `top-3 mean > 0.11` would be 2 σ above chance — a real
   signal.
2. **Untrained-brain control** (~ 5 min). Run the eval pipeline
   with a brain whose plasticity has never fired
   (`disable_stdp_all + disable_istdp_all` from `cfg.use_reward
   = false` with reps = 0). What is the top-3 from pure
   forward-projection + random recurrent weights, no learning
   at all? If that is also ≈ 0.107, the entire chain has been
   measuring the forward-projection baseline. If untrained is
   ≈ 0.05 and trained is ≈ 0.107, learning is real but small.
3. **Trial-to-trial consistency** (one new metric, ~ 30 min
   code). For each cue, present it three times; measure
   Jaccard similarity of the decoded top-3 sets across the
   three presentations. Under chance, Jaccard ≈ random (low);
   under learning, the same engram should produce the same
   prediction set repeatedly (high).

These three together resolve the "is there learning" question
empirically, in roughly the same wallclock as iter-49's
parameter sweeps. **They should run before any architecture or
parameter change**, because every sweep against a
chance-level metric is a coin-flip with extra steps.

## The methodological lesson

iter-50's lesson was "save the simplest working configuration
as a regression guard". iter-51's lesson is the next layer
down:

> **A regression guard is only a guard if its baseline is
> statistically distinguishable from the null.**

We never tested whether iter-46 Arm B was above chance. We took
its 4-data-point `0.19` and treated it as the gold standard
that every subsequent architecture had to clear. Five iterations
of methodology measured against an unverified baseline.

This isn't a failure of the methodology — Bekos's protocol
forced exactly this 16-epoch reproduction that revealed the
statistical situation. But it is a sharp lesson in what the
"baseline" in "compare against baseline" requires before it
counts as evidence: **a confidence interval that excludes the
null hypothesis**.

iter-52 should fix that gap before any new mechanism work.
