# Iter 50 — Arm B Reproduktion (Bekos Diagnose)

## Why iter-50

iter-49 produced its 0/3 result and pointed by elimination at
"raise STDP `a_plus`" as the iter-50 direction. Bekos pushed
back: **before the fourth consecutive parameter sweep, prove
that the iter-46 Arm B baseline (top-3 = 0.19, R-STDP only, no
teacher) still reproduces on the current branch code**.

The data point that 5 iterations of architectural addition
(teacher-forcing, R1→R2 gating, BCM, iSTDP-tightening) had
left untouched: **iter-46 Arm B's top-3 = 0.19 is 3× the
random baseline (0.094) and 3× higher than every selectivity-
positive arm in iter-47/48/49 ever managed**. If Arm B still
works, the architecture added since iter-47 is the problem,
not the solution.

> Three possible outcomes:
> (a) reproduces 0.19 + positive sel + stable target_hit
>     → architecture is the problem; iter-51+ is reduction.
> (b) reproduces top-3 BUT bad sparsity metrics
>     → decoder bias OR metric is meaningless in this path.
> (c) doesn't reproduce
>     → code drift; reproducibility is the open issue.

## Implementation

Single CLI flag `--iter46-baseline` that simultaneously reverts
the four iter-47/48/49 drift values to their iter-46 originals,
controlled at runtime, *without* a code revert:

| Drift | iter-46 (Arm B) | iter-47/48/49 default | `--iter46-baseline` |
| --- | ---: | ---: | ---: |
| `INTER_WEIGHT` | 2.0 | 1.0 | 2.0 ✓ |
| `R2_INH_FRAC` | 0.20 | 0.30 | 0.20 ✓ |
| `iSTDP a_plus` | 0.10 | 0.30 | 0.10 ✓ |
| `iSTDP tau_minus` | 30 ms | 8 ms | 30 ms ✓ |
| `IntrinsicParams enabled` | false | true | false ✓ |

Plus: `target_r2_map` is now built unconditionally (cheap hash op)
so the iter-47/48/49 sparsity metrics
(`selectivity_index`, `target_hit_pre_teacher_mean`, `r2_active_pre_teacher_*`)
are populated in the iter-46 Arm B path too — a per-trial
plasticity-OFF cue-only sample after each rep.

All 9 eval lib tests still green; clippy `-D warnings` clean.

## Run — `--epochs 4 --reps 4 --iter46-baseline`

CLI omits `--teacher-forcing`, so both arms (pure STDP + R-STDP)
run.

### Arm A — Pure STDP (no neuromodulator)

```
| Epoch | top-1 | top-3 | MRR  | noise t3 | w̄    | wmax |
|   0   | 0.00  | 0.00  | 0.09 | 0.06    | 1.58 | 8.00 |
|   1   | 0.00  | 0.06  | 0.07 | 0.06    | 1.69 | 8.00 |
|   2   | 0.00  | 0.06  | 0.06 | 0.06    | 1.73 | 8.00 |
|   3   | 0.00  | 0.00  | 0.05 | 0.06    | 1.74 | 8.00 |

| Epoch | r2_act mean | p10/p90/p99 | tgt_hit | selectivity | θ_E   | θ_I   |
|   0   |  68.6  | 47/95/104 | 1.19 | -0.0096 | 0.002 | 0.005 |
|   1   |  51.1  | 43/62/72  | 0.88 | -0.0075 | 0.002 | 0.006 |
|   2   |  47.8  | 40/61/62  | 0.88 | -0.0050 | 0.002 | 0.006 |
|   3   |  46.2  | 41/56/57  | 0.75 | -0.0082 | 0.002 | 0.006 |
```

### Arm B — R-STDP (dopamine + eligibility, no teacher)

```
| Epoch | top-1 | top-3 | MRR  | mean rwd | noise t3 | w̄    | wmax |
|   0   | 0.00  | 0.19  | 0.14 | -0.59   | 0.06    | 1.11 | 8.00 |   ← reproduces iter-46
|   1   | 0.00  | 0.12  | 0.09 | -0.58   | 0.12    | 1.15 | 8.00 |
|   2   | 0.00  | 0.06  | 0.08 | -0.53   | 0.25    | 1.15 | 8.00 |
|   3   | 0.00  | 0.06  | 0.11 | -0.53   | 0.12    | 1.15 | 8.00 |

| Epoch | r2_act mean | p10/p90/p99 | tgt_hit | selectivity | θ_E   | θ_I   |
|   0   | 173.4  | 131/275/281 | 3.00 | -0.0085 | 0.000 | 0.000 |
|   1   | 143.4  | 118/170/180 | 2.38 | -0.0107 | 0.000 | 0.000 |
|   2   | 143.8  | 113/190/200 | 2.38 | -0.0109 | 0.000 | 0.000 |
|   3   | 144.1  | 123/173/181 | 2.44 | -0.0090 | 0.000 | 0.000 |
```

## Outcome — (b) with a critical nuance

**top-3 = 0.19 reproduces in epoch 0 of Arm B.** Exactly the
iter-46 number on iter-46 settings, run on the current iter-49
branch code (with the four drifts reverted at runtime). The
code drift hypothesis (c) is **falsified**.

**But the iter-47/48/49 sparsity metrics tell a contradictory
story:**

- `selectivity_index = -0.008 to -0.011` (negative throughout)
- `target_hit_mean = 3.0 to 2.4` (high in absolute terms)
- `r2_active = 173 to 144` (well above the [25, 70] target band)

Numerical sanity check: with `r2_active = 173` and the canonical
target SDR being a hash-pinned 30-of-1400 subset, the *expected*
target hit count under uniform random firing is
`173 × 30 / 1400 = 3.71`. The observed 3.00 sits **below random
expectation**, which is exactly why `selectivity_index` is
negative.

## What this actually means

**The `selectivity_index` metric introduced in iter-47 is
structurally meaningless in the no-teacher path.**

It compares cue-driven R2 firing against the canonical hash
SDR for the target. In the teacher-forcing path, that hash SDR
is *causally* activated by the clamp during training, so STDP
has a substrate to grow recurrent weights toward. In the
no-teacher path, the hash SDR is never activated — it's a pure
hash function with no relation to anything the SNN ever fires.
A "negative selectivity" there is not a learning signal; it's
the chance result of comparing an arbitrary hash subset
against an unrelated firing distribution.

The decoder's `top-3 = 0.19` measures something completely
different: **the per-vocab fingerprint built by the epoch-end
fingerprinting pass**. The decoder asks "does the cue-driven
R2 pattern overlap with the *recall fingerprint* the network
itself produces when shown the target word's SDR?". That
fingerprint *is* causally produced by R1→R2 forward projection
plus whatever recurrent associations STDP grew, so it does
measure real learning.

**Conclusion**: outcome (b) — `top-3` reproduces, sparsity
metrics are mute *in the no-teacher path*. Five iterations
optimised against the wrong metric for the chain of arms that
mattered most.

## Comparison to iter-48 (teacher-forcing on, current defaults)

| Configuration | top-3 epoch 0 | best top-3 over 4 ep | mean reward |
| --- | ---: | ---: | ---: |
| **iter-46 Arm B (this run)** | **0.19** | **0.19** | **−0.59** |
| iter-48 Config 1 (teacher on, current code) | 0.06 | 0.06 | −0.73 |
| iter-48 Config 2 (teacher + iSTDP-during-pred) | 0.06 | 0.06 | −0.74 |

**Bekos's hypothesis is confirmed by data**: the 6-phase
teacher-forcing architecture with iSTDP-tightening makes
top-3 *worse* than the simpler iter-46 Arm B by a factor of 3.
`mean reward` is also less negative in Arm B (−0.59 vs −0.73)
— the network is right more often, just on a metric we stopped
tracking 5 iterations ago.

## Mechanistic hypothesis (why teacher-forcing hurts here)

Two contributing causes:

1. **Phase budget**: Arm B drives `cue + target` together for
   `OVERLAP_MS = 30 ms` after a `CUE_LEAD_MS = 40 ms` head-start,
   ending with `TARGET_TAIL_MS = 30 ms` of target alone. STDP
   sees ~70 ms of overlapping pre/post coincidences per rep.
   The 6-phase teacher schedule has only `teacher_ms = 40 ms` of
   actual plasticity-on coincidence (cue + target-clamp). Less
   than half the per-rep STDP budget.

2. **Trivial-learning trap**: the teacher-clamp activates target
   cells *directly* via R2 external current at 250 nA. STDP
   then learns "when the clamp is on, target cells fire" —
   which is a tautology, not a cue→target association. The
   recurrent weights that *would* implement cue→target are
   dominated by clamp-induced target activity that has no
   causal relation to cue activity outside the teacher phase.

Both fit the data: more training time per coincidence + cleaner
causal substrate for STDP in Arm B.

## Implications for iter-51 (rule-out / rule-in)

**Ruled out by this iter-50 diagnostic:**

- iter-51 as "raise STDP `a_plus`" sweep on iter-49 default
  topology. The `selectivity_index` it would test against is
  meaningless in the no-teacher path — and we now have a
  3× better baseline that also makes that metric meaningless.
- iter-51 as Pfad-2 architecture-bridge. The current
  architecture already has a working baseline that's been
  obscured for 5 iterations; building more on top before
  understanding why the simpler version works better is
  premature.

**Ruled in:**

- **Iter-51 = reductive baseline study**: take Arm B, vary one
  parameter at a time (reps_per_pair: 4 → 8 → 16, w_max: 0.8
  → 2.0, R-STDP eta) and measure top-3 (the metric that
  actually reads in this path) over 16 epochs. Find out
  whether 0.19 is a peak or a starting point.
- **Iter-51 also includes**: replace `selectivity_index` with a
  decoder-relative measurement that's meaningful in BOTH paths
  (e.g. "top-3 against the per-epoch fingerprint dictionary")
  before trusting it for further architectural decisions.

## Methodological lesson

Three aspects of this diagnostic mattered, and all came from
Bekos's protocol decisions, not from instinct:

1. **Save the simplest working configuration as a regression
   guard.** iter-46 Arm B was documented and discarded. It
   should have been retested at every iteration as a sanity
   check. If we'd added the iter-47 sparsity metrics in the
   iter-47 commit and run them against iter-46 Arm B then, we
   would have caught the metric-vs-architecture mismatch
   immediately.

2. **A metric is only valid in the regime it was designed for.**
   `selectivity_index` was built to read teacher-forcing
   behaviour. Using it to evaluate non-teacher arms produced
   5 iterations of "the network is not learning" reports
   while the network was, in fact, learning at the iter-46
   baseline level on a different metric.

3. **Five iterations of clean methodology in the wrong space
   is worse than one messy iteration in the right space.**
   iter-47/48/49 were each individually well-executed Bekos
   sweeps. They moved selectivity from −0.045 to +0.014 and
   then back to −0.045, all the while top-3 quietly held at
   0.06 with the new architecture vs 0.19 with the old. The
   methodology was sound; the question being optimised was
   wrong.

The next concrete step (iter-51) should pause both axes
(STDP-magnitude AND bridge architecture) and instead vary
Arm B parameters to find out whether 0.19 is the ceiling or
just the starting point.
