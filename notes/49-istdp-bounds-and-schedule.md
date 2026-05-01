# Iter 49 — iSTDP Bounds & Schedule Sweep

## Why iter-49

iter-48 + Phase A (notes/48-saturation) found that the iter-48 iSTDP
parameters sit on the **over-tuned side** of a collapse boundary:
selectivity flips positive (peak +0.014 to +0.017 epoch 2-4), then
**hard-collapses at epoch 5** to a stable negative steady-state. The
mechanism is iSTDP cumulative over-inhibition — distinct from any
prior failure mode (NOT cascade, NOT weight runaway, NOT Diehl-Cook
over-correction). Bekos's protocol: 3-point parallel sweep on the
*under-tuned* side, three orthogonal axes against the same attractor.

> Sweep A (WmaxCap)        — symptom: cap inhibitory weight 8.0 → 2.0
> Sweep B (APlusHalf)      — dynamic: a_plus 0.30 → 0.20 (half-way to iter-47a)
> Sweep C (ActivityGated)  — temporal: a_plus = 0 first 2 epochs, ramp 0→0.30 over 2 epochs

All three under Config 2 base (`--istdp-during-prediction`). All
under the iter-49 acceptance:

> sustained selectivity > 0 across epochs 4–16
> AND mean target_hit at epoch 16 > mean target_hit at epoch 4
> NO magnitude criterion — testing for collapse-survival only.

Bekos's pre-experiment hypothesis (~ 60 % confidence): WmaxCap as
the cleanest survivor, because the postmortem identified "walls
that can't be torn down even during teacher-clamp" — an
asymmetry between buildup and teardown that a hard cap addresses
directly.

## Implementation (commit `5ba5c25`)

- `Iter49Mode` enum (`None | WmaxCap | APlusHalf | ActivityGated`),
  re-exported.
- `TeacherForcingConfig.iter49_mode` + `gated_warmup_epochs` (2)
  + `gated_ramp_epochs` (2). Default `None` reproduces iter-48
  exactly.
- `istdp_iter49(cfg, epoch)` — mode + epoch aware iSTDP builder.
- `run_reward_benchmark` calls `enable_istdp(...)` at the start
  of each epoch with the current value (no-op for None / WmaxCap
  / APlusHalf, ramped for ActivityGated).
- CLI: `--iter49-mode {none|wmax-cap|a-plus-half|activity-gated}`.

All 9 eval lib tests still green; clippy `-D warnings` clean.

## Sweep results (16 epochs each, 16 + 16 pairs, vocab 32, seed 42)

### Sweep A — WmaxCap (`w_max = 2.0`, all else iter-48)

```
| Epoch | r2_act | tgt_hit | selectivity |
|   0   |  62.8  |  0.48   | -0.0294 |
|   1   |  99.3  |  1.66   | -0.0161 |
|   2   | 118.8  |  2.31   | -0.0080 (peak — but still negative)
|   3   | 118.8  |  2.20   | -0.0117 |
|   4   | 109.9  |  2.08   | -0.0094 |
|   5   |  54.5  |  0.12   | -0.0355 (collapse)
|   6-15|  58-62 |  0.12   | -0.037..-0.040 (worse than iter-48 baseline)
```

Acceptance: **0/3.** Selectivity NEVER crosses zero — even at peak
(epoch 2) it sits at −0.008. r2_active blows up to 100-119 (over
the iter-47a [25, 70] band) because reduced inhibitory cap weakens
suppression → more cells fire → tgt_hit absolute is *higher* than
iter-48 (2.31 vs 1.34) but selectivity is *worse* because non-target
firing scales proportionally. **Cap addresses symptom but not
cause; Bekos hypothesis falsified by data.**

### Sweep B — APlusHalf (`a_plus = 0.20`, all else iter-48)

```
| Epoch | r2_act | tgt_hit | selectivity |
|   0   |  37.0  |  0.44   | -0.0121 |
|   1   |  34.3  |  1.02   | +0.0096 ✓ |
|   2   |  47.0  |  1.55   | +0.0184 ✓ ← highest peak in entire iter chain
|   3   |  46.9  |  1.53   | +0.0179 ✓ |
|   4   |  44.0  |  1.42   | +0.0163 ✓ |
|   5   |  20.3  |  0.12   | -0.0106 (collapse — same epoch as iter-48)
|   6-15|  24-27 |  0.12   | -0.013..-0.015 |
```

Acceptance: **0/3.** Same trajectory shape as iter-48; **higher
peak** (+0.0184 vs iter-48's +0.0172, with target_hit_mean 1.55 vs
1.34) but **identical collapse epoch (5)**. Halving `a_plus`
delays nothing — it just produces a slightly higher peak and a
slightly less-bad steady-state. The wall buildup in epochs 0-4 is
dominated by `a_minus = 1.10` (LTD on coactivity) responding to
target firing under teacher-clamp; once that signal stops at the
collapse, the residual walls are essentially the same as iter-48's.

### Sweep C — ActivityGated (`a_plus = 0` epochs 0-1, ramp 0→0.30 epochs 2-3, full 4+)

```
| Epoch | r2_act | tgt_hit | selectivity |
|   0   |  100.4 |  1.14   | -0.0344  (a_plus=0, no inhibition forming)
|   1   | 1299.7 | 27.78   | -0.0023  (warmup blew up — every cell fires)
|   2   | 1400.0 | 30.00   | +0.0000  (ALL R2-E cells fire every trial)
|   3-15| ≈1396  | ≈29.7   | +0.000..+0.003 (hyperactivity lock)
```

Acceptance — **technically 3/3, but artifact**:
- A1 sel > 0 in 14/16 ✓ (epochs 2-15 are all ≥ 0)
- A2 trend ep 13-16 ≥ ep 1-4 ✓ (both near zero, +0.0026 vs +0.0006)
- A3 target_hit ep 16 > 1.5 ✓ (29.6 ≫ 1.5)

**This is a Pyrrhic acceptance.** With r2_active = 1400 = the
entire R2-E pool, ALL canonical-target cells fire (target_hit =
30/30) but so does everything else (1370 non-targets). selectivity
of +0.003 is statistical noise around the maximum-entropy state
where every cell fires every trial. The Bekos magnitude-free
acceptance criterion does not detect "trivially saturated" —
caught here by the `r2_active = 1400` red flag.

The mechanism Bekos warned about ("Ramp-Übergang als neuer
Kollapszeitpunkt") materialises differently: the warmup epochs
(no iSTDP LTP) let STDP build E→E weights unopposed, so by the
time the iSTDP ramp kicks in, the E activity is already at the
ceiling. iSTDP `a_minus = 1.10` then dampens recurrent E→E
slightly each trial (w̄ = 0.41 vs iter-48's 1.40), but not enough
to prevent the activity lock — every cell receives enough input
from its now-very-strong forward + recurrent path to fire every
trial.

## Acceptance summary

| Sweep | A1 sel > 0 in ≥ 12/16 | A2 trend monotone | A3 tgt_hit ep16 > 1.5 | Total | Reading |
| --- | :-: | :-: | :-: | :-: | --- |
| iter-48 baseline | 4/16 ❌ | ❌ | 0.13 ❌ | 0/3 | over-tuned (collapse to hypo) |
| **A — WmaxCap** | **0/16 ❌** | ❌ | 0.13 ❌ | **0/3** | symptom relaxed, cause untouched |
| **B — APlusHalf** | 4/16 ❌ | ❌ | 0.13 ❌ | **0/3** | same shape, higher peak, same collapse |
| **C — ActivityGated** | 14/16 ✓ | ✓ | 29.6 ✓ | 3/3 (artifact) | **hyperactivity lock — different failure mode** |

Honest read: **0 out of 3 sweeps produce a positive learning
regime** by any meaningful definition. C technically passes the
acceptance but with `r2_active = 1400` (max possible value) and
selectivity ≈ 0.003 — pure noise around uniform activity.

## Synthesis — what 3 distinct failure modes mean

The three orthogonal sweeps each produce a distinct failure mode:

- **Sweep A** weakens iSTDP suppression structurally → more
  unspecific R2 activity, no positive selectivity.
- **Sweep B** slows iSTDP buildup → identical collapse trajectory,
  same time-to-failure.
- **Sweep C** delays iSTDP application → STDP runs unopposed in
  warmup → E→E weights saturate → activity lock that iSTDP
  cannot recover from.

This is **diagnostic information that no single experiment could
have produced**. The pattern says: **iSTDP is not the primary
lever**. Every axis of iSTDP tuning (Cap / Rate / Schedule)
produces a *different* failure, but **none produce a positive
learning regime**. The system cannot be parametrised within the
iSTDP space to learn the cue → target association.

## Hypothesis for iter-50 — STDP magnitude, not iSTDP balance

The data points to a different bottleneck entirely:

> **STDP excitatory plasticity is too weak to form selective
> engrams BEFORE iSTDP suppresses or saturates the network.**

Evidence:
1. With iSTDP ON (iter-48): peak target_hit = 1.34 at epoch 2,
   collapses to 0.12 at epoch 5. STDP grew weights for 2 epochs,
   then iSTDP undid them.
2. With iSTDP weakened (Sweep A wmax-cap): peak target_hit =
   2.31 at epoch 2, also collapses. Even with iSTDP weakened by
   4×, STDP still loses the race.
3. With iSTDP delayed (Sweep C activity-gated): STDP runs
   unopposed → activity locks at maximum → no selectivity.
   STDP without inhibition produces *uniform* growth, not
   selective.
4. STDP `a_plus = 0.020`, `w_max = 0.8` (from iter-46's `stdp()`).
   iSTDP `a_plus = 0.30`, `w_max = 8.0`. The STDP-vs-iSTDP rate
   asymmetry is **15×** in `a_plus` and **10×** in `w_max`.

iter-50 entry hypothesis: **raise STDP `a_plus` and/or `w_max`**
so excitatory plasticity can outpace iSTDP. Either:

1. **STDP `a_plus`: 0.020 → 0.060 (3×)** — match the iSTDP rate
   so E→E weights grow as fast as I→E walls.
2. **STDP `w_max`: 0.8 → 2.0** — let E→E weights reach a
   magnitude where they can sustain post-spikes against iSTDP
   suppression.
3. **Both 1+2** as the most aggressive variant.

Pre-fixed iter-50 acceptance candidate: same as iter-49
(sustained selectivity > 0 epochs 4-16, target_hit ep 16 >
ep 4) PLUS a saturation guard (`r2_active < 200`) to detect
the Sweep C-style hyperactivity artifact.

## Methodological note — magnitude-free acceptance criteria need a saturation guard

The iter-49 acceptance was deliberately magnitude-free per Bekos
("testing collapse-survival, not top-3 lift"). Sweep C exploited
this by hitting the upper limit (every cell fires every trial)
which trivially satisfies "selectivity > 0" (technically) and
"target_hit growing" (trivially). For iter-50, add a saturation
guard:

> r2_active_pre_teacher_mean < 0.5 × |R2_E| at epoch 16

(i.e. < 700 cells; `|R2_E|` = 1400). This catches the sweep-C
artifact without re-introducing magnitude pressure.

## What this iter delivered

1. **Three orthogonal failure modes** — distinct, repeatable,
   diagnostically useful. None produce positive learning.
2. **Bekos's WmaxCap hypothesis falsified** — 60 % confidence
   prior, 0/3 + worst-case "never positive" outcome.
3. **iter-50 axis identified by elimination** — STDP
   excitatory rate vs iSTDP inhibitory rate ratio (15× too
   asymmetric for E→E selectivity to grow against iSTDP walls).
4. **Saturation guard** added to the acceptance template for
   iter-50.
5. **iter-49 infrastructure** (Iter49Mode + epoch-aware iSTDP)
   stays in the repo as an A/B platform for future iSTDP
   experiments — the diagnostic value of "which axis fails how"
   doesn't disappear because no axis succeeded.
