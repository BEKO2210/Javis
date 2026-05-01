# Iter 46 — Teacher-forcing pair-association harness

## Why iter-45 did not converge

`notes/45` documented honestly: pure STDP and R-STDP both stayed at
random-baseline accuracy on the 16-pair association task. The
diagnosis pointed at the architecture: R1 → R2 forward weights
(FAN_OUT = 12, weight = 2.0) drive R2 directly on every cue
presentation, so R2's response to a cue is dominated by the random
forward projection — not by recurrent associations. STDP / R-STDP
*do* shape recurrent weights, but in the available training time
the change is too small to show up against the noise floor.

Iter-46 attacks the architectural bottleneck directly: during
training, the canonical target SDR is **clamped into R2 itself**,
not just driven through R1. STDP now sees pre→post coincidences
between the cue's R2 cells and the *teacher-forced* target cells,
with full external current.

**Architectural rule** (the scientific reason this is *not*
cheating): teacher-forcing only fires during the training phase.
Evaluation is always cue-only on R1, plasticity off — the exact
same readout pure STDP gets. The teacher is a training *signal*,
never an evaluation shortcut.

## What iter-46 ships

### Configuration (step 1, commit `76b2c0b`)

`TeacherForcingConfig` controls every knob (`enabled`, `cue_ms`,
`delay_ms`, `prediction_ms`, `teacher_ms`, `tail_ms`,
`target_clamp_strength`, `plasticity_during_{prediction,teacher}`,
`wta_k`, three reward levels, `homeostatic_normalization`,
`debug_trials`, `r1r2_prediction_gate`). CLI flags:
`--teacher-forcing`, `--teacher-ms`, `--prediction-ms`,
`--delay-ms`, `--tail-ms`, `--cue-ms`, `--wta-k`,
`--target-clamp-strength`, `--reward-positive`,
`--reward-negative`, `--noise-reward`, `--homeostasis`,
`--debug-trial`, `--association-training-gate-r1r2`.

### Primitives (step 2, commit `37f47b6`)

- `canonical_target_r2_sdr(word, e_pool, k, salt) → Vec<u32>` —
  deterministic hash mapping. Stable across epochs and trials.
- `drive_with_r2_clamp(brain, cue, r2_target, r1_strength,
  r2_strength, dur_ms, r2_e) → spike counts`.
- Two unit tests:
  `canonical_target_r2_sdr_is_stable_and_in_pool` and
  `drive_with_r2_clamp_makes_target_neurons_fire` — the latter
  asserts ≥ 50 % of clamped neurons fire at strength 500 nA.

### Six-phase trial schedule (step 3, commit `82dc313`)

```
1. cue        (cue_ms)         drive cue on R1
2. delay      (delay_ms)       no input
3. prediction (prediction_ms)  cue on R1 × r1r2_prediction_gate,
                               plasticity OFF, capture top-k
4. teacher    (teacher_ms)     cue lead-in, then cue + R2 clamp,
                               plasticity ON
5. reward                      modulator from prediction (NOT
                               teacher), brief consolidation
6. tail       (tail_ms)        traces decay
```

Plasticity gating around phase 3 toggles `Network::disable_stdp`
/ `disable_istdp` and restores them after. Reward computation
reads the prediction phase top-k; noise pairs always emit
`noise_reward`.

### Metrics + diagnostics (step 4, commit `1b67ee8`)

`RewardEpochMetrics` carries `random_top3_baseline`, `mean_rank`,
`mrr`, `target_clamp_hit_rate`, `prediction_top3_before_teacher`,
`eligibility_nonzero_count`, `r2_recurrent_weight_{mean,max}`,
`active_r2_units_per_cue`, `correct_minus_incorrect_margin`,
`decoder_micros`. `--debug-trial` prints per-trial diagnostics.

### Anti-causal STDP fix (step 4b, commit `88d467d`)

The first run with the teacher schedule produced a clean
diagnostic surprise: `clamp = 1.00, margin = -0.04`. Why
*negative* margin? Because the R1 → R2 inter-region edge has a
2 ms delay, plus LIF integration adds a few more, so when the
clamp arrives at t = 0 of the teacher phase, the cue's R2 cells
have not yet fired. Target spikes lead, cue spikes follow →
STDP grows target → cue weights, the wrong direction.

Fix: drive cue alone for `lead_in = clamp(teacher_ms / 4, 4, 12)`
ms before the clamp turns on. Cue R2 cells get a head-start so
the clamped target cells *follow*. The fix is necessary — but, as
the next section documents, not sufficient on its own.

### R1 → R2 prediction gate (step 4c, commit `1ebad1f`)

`r1r2_prediction_gate < 1.0` attenuates the cue current during
the prediction phase only. Lets recurrent learning, whatever has
grown so far, express itself against the otherwise-dominant
forward drive. Default `1.0` (no gating); `--association-
training-gate-r1r2` defaults to `0.3`. Training-only knob —
evaluation always runs at full strength.

## Acceptance criteria — measured

| # | Criterion | Status |
| - | --- | --- |
| 1 | Code compiles | ✅ |
| 2 | Existing tests still green | ✅ (9 lib tests + iter-45 smoke) |
| 3 | Smoke test runs in reasonable time | ✅ (~6 s) |
| 4 | Evaluation does not modify weights | ✅ (plasticity disabled around eval) |
| 5 | Teacher phase not counted as recall | ✅ (reward reads prediction phase) |
| 6 | Clamp activates target R2 SDR | ✅ (`target_clamp_hit_rate = 1.00` every epoch) |
| 7 | `correct_minus_incorrect_margin` rises over epochs | ❌ **stays at -0.03..-0.06** |
| 8 | R-STDP + Teacher visibly better than R-STDP alone | ❌ **same flat curve** |
| 9 | WTA + negative reward suppresses noise better | ⚠️ noise top-3 drops to 0 in arm C, but only because every prediction is wrong (no class collapses to "noise" specifically) |
| 10 | top-3 ≥ 9.4 % stable | ❌ **at chance, not above** |
| 11 | top-3 ≥ 20 % over multiple epochs | ❌ **never reached** |
| 12 | If accuracy doesn't rise, document the chain | ✅ (this section) |

## Measured — full sweep, 16 pairs + 16 noise, vocab 32, decode_k 3, seed 42

(Random top-3 baseline `3 / 32 = 0.094`. Wall-time on a release
build with R2 = 2 000.)

### Arm A: pure STDP, no teacher (20 epochs, ~ 4 min)

| metric | epoch 0 | epoch 10 | epoch 19 |
| --- | ---: | ---: | ---: |
| top-3 | 0.19 | 0.12 | 0.06 |
| mean reward | 0.00 | 0.00 | 0.00 |
| noise t3 | 0.00 | 0.12 | 0.19 |
| margin (n/a, no teacher) | — | — | — |

Oscillates at chance, no learning, no convergence — same as iter-45.

### Arm B: R-STDP, no teacher (20 epochs, ~ 4 min)

| metric | epoch 0 | epoch 10 | epoch 19 |
| --- | ---: | ---: | ---: |
| top-3 | 0.19 | 0.06 | 0.19 |
| mean reward | -0.59 | -0.64 | -0.64 |
| noise t3 | 0.06 | 0.12 | 0.00 |

Mean reward stays around -0.6 across all 20 epochs. R-STDP is
running, eligibility tags are being modulated, but the recurrent
path it shapes never improves the prediction enough to flip the
sign of the average reward.

### Arm C: R-STDP + Teacher-Forcing (20 epochs, ~ 7 min, *step-3 timing*)

| metric | epoch 0 | epoch 10 | epoch 19 |
| --- | ---: | ---: | ---: |
| top-3 | 0.06 | 0.06 | 0.00 |
| pred-t3 | 0.00 | 0.00 | 0.00 |
| **clamp** | **1.00** | **1.00** | **1.00** |
| **margin** | **-0.04** | **-0.03** | **-0.05** |
| mean reward | -0.75 | -0.75 | -0.75 |

Clamp works perfectly. Margin is *negative* — the canonical-target
neurons fire **less** than the rest under cue-only recall. The
diagnosis was anti-causal STDP timing (target spikes lead cue
spikes during teacher phase). Step 4b fixed the timing.

### Arm C′: same as C with step-4b lead-in (4 epochs smoke)

| metric | epoch 0 | epoch 1 | epoch 2 | epoch 3 |
| --- | ---: | ---: | ---: | ---: |
| margin | -0.04 | -0.04 | -0.03 | -0.03 |
| pred-t3 | 0.00 | 0.00 | 0.00 | 0.00 |

Lead-in fix is correct in principle but does not flip the margin
sign on its own — the random R1 → R2 forward path keeps R2's
spike density too high (90–180 active cells per cue) for any
recurrent bias to register.

### Arm D: + R1 → R2 prediction gate at 0.3 + homeostasis (4 epochs smoke)

| metric | epoch 0 | epoch 1 | epoch 2 | epoch 3 |
| --- | ---: | ---: | ---: | ---: |
| margin | -0.05 | -0.04 | -0.03 | -0.04 |
| pred-t3 | 0.00 | 0.00 | 0.00 | **0.02** |
| w̄ | 0.03 | 0.02 | 0.02 | 0.02 |
| wmax | 0.21 | 0.20 | 0.20 | 0.20 |

Homeostatic L2 cap is too aggressive (`wmax` falls from 8.00 to
0.20 in one epoch), but the *right* signal appears: epoch 3 is
the first time `pred_top3_before_teacher` is non-zero. The signal
exists in principle; it just hasn't been amplified beyond the
noise floor in 4 epochs.

## Honest summary

**What the iter-46 commits demonstrably deliver:**

1. A clean, tested teacher-forcing infrastructure: canonical
   target SDRs (deterministic), `drive_with_r2_clamp` (verified
   to activate ≥ 50 % of clamped cells), six-phase trial
   schedule, plasticity gating, R1 → R2 prediction gate,
   homeostatic L2 cap, R-STDP eligibility tag, full per-trial
   diagnostics — every piece independently exercised.
2. **`target_clamp_hit_rate = 1.00`** every epoch in every
   teacher arm. The clamp itself is unambiguously working: every
   one of the 30 canonical target neurons fires every time it is
   clamped.
3. The anti-causal STDP timing trap diagnosed and fixed (step
   4b) — without this fix STDP grows target → cue weights, the
   wrong direction. The negative margin in arm C made this
   visible; without the metrics added in step 4 the trap would
   have stayed hidden.
4. The R1 → R2 dominance bottleneck (notes/45's diagnosis) is
   now *measurably real*: even with a perfect clamp, perfect
   lead-in timing, R-STDP eligibility on, and the gate at 0.3,
   four epochs are not enough to flip the cue-driven R2 spike
   count toward the canonical target. The bottleneck has moved
   from "we can't measure it" to "we can measure it; here is
   the number".

**What did not happen:**

- top-3 did not reach 20 %. It did not even cleanly exceed the
  9.4 % chance floor across the runs we measured.
- `correct_minus_incorrect_margin` did not turn positive.
- R-STDP + Teacher-Forcing did not visibly outperform R-STDP
  without teacher-forcing on the test set.

This is *not* a "the teacher works" headline result. It is a
"the teacher infrastructure is in place, the clamp activates the
right cells, and recurrent recall is still being suppressed by
forward drive at every test point we have measured" result.

## Where the next bottleneck is

The diagnosis chain after iter-46 is now sharper:

1. ✅ Clamp fires the right cells (`clamp = 1.00`).
2. ✅ Eligibility traces accumulate during teacher phase
   (smoke run with `--debug-trial` shows non-zero spike counts
   and reward delivery every trial).
3. ✅ Reward sign matches intent (positive on prediction-correct
   real pairs, -1 on noise — the schedule reaches every branch).
4. ❌ R2 → R2 weights *do* shift, but their absolute scale stays
   small relative to the R1 → R2 forward drive (~ 0.97 mean,
   8.00 max recurrent vs. ~ 2.0 forward fan-out × 12).
5. ❌ Cue-only prediction is therefore still dominated by the
   random forward projection, and the canonical-target subset of
   R2 fires *no more* than the rest.

The honest next-iter (47) bottleneck is **the absolute scale of
R1 → R2 vs R2 → R2**, *not* the timing or the reward signal.
Three concrete experiments would each move the needle:

- Reduce `INTER_WEIGHT` (R1 → R2) from 2.0 to 0.5 so STDP-grown
  recurrent weights of 0.5–0.8 actually compete.
- Add an explicit *association bridge region* between R1 and R2
  (a small dedicated cue → target hub).
- Replace the random R1 → R2 wiring with a *learnable* feed-forward
  layer that the teacher signal also reaches — i.e. teach the
  feed-forward path, not just the recurrent loop.

Iter-46's harness can A/B-test any of these against the same
scoring contract on the same corpus. That is what the
infrastructure was built for.
