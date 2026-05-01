# Iter 52 — Untrained-Brain Control (`--no-plasticity`)

## Why iter-52

iter-51 found that Arm B's `top-3` mean over 16 epochs sits at
**0.107** with a 95 % CI `[0.069, 0.145]` that *includes* the
naive random baseline `3/32 = 0.094`. The honest read at iter-51
was "statistically not distinguishable from chance" — but that
test compared trained Arm B against an analytical random model,
not against an empirical untrained-brain control. The two are
not the same when the decoder is biased.

iter-52's question is one shift to the right: **is the iter-51
top-3 = 0.107 actually higher than what the same decoder
returns on a brain whose plasticity is gated off entirely?**

If trained ≈ untrained, plasticity is inert on this benchmark.
If trained ≫ untrained, plasticity is doing something the
analytical random model couldn't capture. If trained ≪ untrained,
plasticity is making things actively worse.

## Implementation — `--no-plasticity` flag (single commit)

`TeacherForcingConfig.no_plasticity` + CLI flag (alias
`--frozen-weights`) gates every plasticity enable in
`run_reward_benchmark`:

| Path | Behaviour under `--no-plasticity` |
| --- | --- |
| `enable_stdp` | skipped at run start AND in epoch loop AND in mid-trial cycles |
| `enable_istdp` | skipped at run start, in epoch loop, and in mid-trial cycles |
| `enable_homeostasis` | skipped |
| `enable_intrinsic_plasticity` | skipped (Diehl-Cook off) |
| `enable_reward_learning` | skipped (R-STDP eligibility tag never allocated) |
| Forward LIF dynamics | UNCHANGED — spikes propagate, only weight updates are silenced |
| Decoder, fingerprinting, all metrics | UNCHANGED |

Three plasticity-enable sites had to be gated, not just one:

1. Setup (`run_reward_benchmark` start) — obvious
2. Per-epoch `enable_istdp` — Iter-49 ActivityGated infrastructure
3. Mid-trial `disable_stdp` / `enable_stdp(stdp_params)` cycles
   inside the Arm-B-diagnostic-sample block AND inside the
   epoch readout

**The first run with only (1) gated produced a 9× weight blowup
that the L2-norm sanity assertion caught immediately**: pre L2
136.10 → post L2 1242.82 (R2→R2). Sites (2) and (3) were
re-enabling plasticity mid-run despite the setup-time gate.
After all three were closed: pre 136.10 == post 136.10
**bit-for-bit across all 4 seeds**. The assertion is now
permanent in the run loop and panics on any drift.

## Run — 4 seeds × 16 epochs, --iter46-baseline --no-plasticity

```
=== seed 41 ===  initial L2 = 136.58 → post L2 = 136.58 ✓ bit-identical
top-3: 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.06, 0.06,
       0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06   mean = 0.0413

=== seed 42 ===  initial L2 = 136.10 → post L2 = 136.10 ✓ bit-identical
top-3: 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.00, 0.00,
       0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00   mean = 0.0225

=== seed 43 ===  initial L2 = 136.36 → post L2 = 136.36 ✓ bit-identical
top-3: 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.06, 0.06,
       0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06   mean = 0.0375

=== seed 44 ===  initial L2 = 136.57 → post L2 = 136.57 ✓ bit-identical
top-3: 0.00, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06,
       0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06   mean = 0.0563
```

Aggregated across 4 seeds × 16 epochs = **64 epoch-samples**:

```
Untrained top-3 mean = 0.039
Untrained top-3 SE   = sqrt(p(1−p)/64) = sqrt(0.039 × 0.961 / 64) ≈ 0.024
95 % CI              = [−0.008, 0.086]
```

## Comparison to iter-51 trained (same arm, same flags except no_plasticity = false)

| | Trained (iter-51) | Untrained (iter-52) |
| --- | ---: | ---: |
| top-3 mean | 0.107 | **0.039** |
| 95 % CI | [0.069, 0.145] | [−0.008, 0.086] |
| Δ vs random 0.094 | +0.013 (≈ 0.2 σ) | **−0.055 (≈ 2.3 σ)** |
| Δ vs each other | — | **0.068, ≈ 2.2 σ** |

Two new statistical readings:

1. **Trained vs untrained: Δ = 0.068, ~2.2 σ — statistically
   significant.** Plasticity is doing something measurable.
2. **Untrained vs random: untrained is significantly *below*
   random** (95 % CI excludes 0.094 from above). The decoder
   has a bias *against* the correct top-3 set on a fresh
   brain, and training (partially) corrects it.

## What this means — Mess-Frage, per Bekos's matrix

The pre-fixed iter-53 branching matrix, applied verbatim:

| Untrained top-3 | Branch | This data |
| --- | --- | :-: |
| ≈ 0.107 (CI overlap with trained) | Architecture inert | ❌ |
| ≤ 0.085 (no overlap, lower) | **Measurement question** | **✓** (0.039 ≪ 0.085) |
| ≥ 0.13 (no overlap, higher) | Sign question | ❌ |

iter-53 entry is **the measurement question**: trained is doing
something, but the top-3 metric saturates against a low
empirical ceiling because the decoder has a strong "default"
prior. The metric is not a good readout of the learning that
*does* exist.

## What's actually happening with the decoder bias

`r2_active` in untrained runs is **180** (vs trained Arm B's
145). The forward-projection-only brain fires more cells per
cue, more uniformly, because no plasticity has shaped any
preference. The kWTA selects the top 60 of 180, and the
fingerprint dictionary built each epoch fingerprints all 32
vocab words against this same forward-dominated pattern.

Result: every word's stored engram fingerprint is highly
similar (all driven by the same forward-projection kernel),
so `decode_top` gets a near-uniform overlap profile across
the dictionary and breaks ties alphabetically. The "default
winner" is never the cue's correct target — and `top-1 = 0.00
in every single epoch of every untrained seed` confirms it.

Training breaks this symmetry by growing recurrent weights
that bias the cue's R2 response away from the uniform forward
pattern. The bias is small (`r2_active` drops 180 → 145, the
fingerprint distribution sharpens slightly), but enough to
push top-3 from 0.039 to 0.107 — a 2.7× lift over untrained,
landing right at random. Above-random would require either
stronger training or a less-saturating decoder.

## What iter-53 should NOT do

- ❌ Architecture work on the brain (the brain is doing
  something; the bottleneck is reading it).
- ❌ More plasticity-parameter sweeps (we now know they
  produce a real but small lift; no parameter inside the
  current setup is going to multiply 0.068 by 10×).
- ❌ Multi-seed Arm B power analysis as originally planned in
  iter-51's Schritt 3 (we already have 4 seeds; the trained-
  vs-untrained Δ is the relevant signal, not a tighter CI on
  trained alone).

## What iter-53 SHOULD do — decoder-relative readout

The single right next step is now clear:

> **Replace `top-3 against the per-epoch fingerprint dictionary`
> with a metric that does NOT degrade when the dictionary's
> entries are highly correlated.**

Two candidates that both work in trained AND untrained paths:

1. **`top-3 lift over untrained-control`**: same decoder, same
   trial, but two passes — once on the trained brain, once on
   a frozen copy of the *initial* brain. The metric is the
   per-trial difference. This is exactly the trained −
   untrained = 0.068 we just measured, expressed per trial
   instead of per run.
2. **Trial-to-trial Jaccard consistency**: present each cue
   3 times, compute Jaccard of the decoded top-3 sets.
   Untrained: high (decoder always returns its bias set).
   Trained-but-noise: low (no consistent engram).
   Trained-and-learning: high AND with the *correct* target
   in the consistent set.

Both are ~ 30 minutes of code. Option (1) is cheaper because
it reuses the existing decoder; option (2) is the more
informative metric long-term.

## Acceptance status

| # | Criterion | Status |
| - | --- | --- |
| 1 | Code compiles | ✅ |
| 2 | Lib tests still green | ✅ (9 lib) |
| 3 | L2 norms bit-identical pre / post in every seed | ✅ (all 4 seeds, all 5 plasticity types verified gated) |
| 4 | Multi-seed run produces 64 epoch-samples | ✅ |
| 5 | Bekos branching matrix applied verbatim | ✅ → Mess-Frage |
| 6 | Single commit, single CLI flag, no architecture refactor | ✅ |

## Methodological notes

**The L2 sanity assertion saved this iteration.** Without it,
the first run would have produced "untrained top-3 = 0.21"
(weights blowing up 9× look like learning to the decoder),
and we would have concluded the opposite — that plasticity is
inert because untrained beats trained. The two leak sites in
the diagnostic and epoch-readout cycles were textbook silent
state-restoration bugs; the assertion caught them in 2 seconds
of run wallclock.

**The "Mess-Frage" outcome is the most useful possible result
for iter-53 planning.** It says: there is signal, the metric
just doesn't read it well. That is a much more tractable
problem than "there is no signal" or "the signal is in the
wrong direction".

iter-50 lesson: save the simplest configuration as a regression
guard.
iter-51 lesson: a regression guard is only a guard if its
baseline excludes the null.
**iter-52 lesson: an analytical null hypothesis (random) is
not the same as an empirical untrained control. Use the
control whenever the decoder can have a bias.**
