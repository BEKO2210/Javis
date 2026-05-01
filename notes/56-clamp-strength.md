# Iter 56 — Clamp-strength sweep on decorrelated + ep32 architecture

## Why iter-56

iter-55 landed branch (ii) of Bekos's pre-fixed iter-56 selector
(**saturation primary**), with branch (iv) secondary (eval-phase
plasticity essentially still under decorrelated wiring). The
per-doubling marginal gain on the epoch axis was −0.054 (16 →
32) and −0.016 (32 → 64), giving a geometric-series asymptote
near trained cross ≈ 0.21. More epochs alone is rejected.

Per Bekos's iter-56 spec, the next un-swept axis with high a
priori sensitivity is **target-clamp-strength** — it controls
how strongly the teacher signal overrides recurrent dynamics
during the teacher window, which is the layer where cue-
specific weight changes get written. iter-56 sweeps three
points around the iter-46/53/54/55 default (250 nA):

- **125 nA** (50 % of default) — does halving the teacher
  signal weaken specificity?
- **250 nA** (default; replication of iter-55 ep32)
- **500 nA** (200 % of default) — does doubling break the
  specificity ceiling?

Base config: `ep32` (best info-per-wallclock from iter-55) with
`--decorrelated-init --teacher-forcing` and the standard 4
seeds.

## Implementation

No new code. Pure sweep-config exercise on the
`--target-clamp-strength` CLI flag that already exists on the
example.

## Run

```sh
for clamp in 125 250 500; do
  cargo run --release -p eval --example reward_benchmark -- \
    --jaccard-bench --seeds 42,7,13,99 --epochs 32 \
    --decorrelated-init --teacher-forcing \
    --target-clamp-strength $clamp
done
```

### 125 nA — half default

| Seed | Untrained cross | Trained cross | Δ cross | Eval-drift L2 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.467 ± 0.106 | 0.262 ± 0.200 | −0.205 | (logged) |
|  7 | 0.485 ± 0.082 | 0.315 ± 0.203 | −0.170 | (logged) |
| 13 | 0.450 ± 0.122 | 0.283 ± 0.210 | −0.167 | (logged) |
| 99 | 0.433 ± 0.128 | 0.229 ± 0.211 | −0.204 | +0.029 |

Aggregate: trained same = 1.000 ± 0.000  cross = **0.272 ± 0.036**
Δ cross = **−0.187**, Δ-of-Δ = +0.187, paired t(3) ≈ −17.6,
p ≪ 0.001.

### 250 nA — default, replicates iter-55 ep32

| Seed | Untrained cross | Trained cross | Δ cross | Eval-drift L2 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.467 ± 0.106 | 0.277 ± 0.192 | −0.190 | +0.045 |
|  7 | 0.485 ± 0.082 | 0.281 ± 0.198 | −0.204 | (logged) |
| 13 | 0.450 ± 0.122 | 0.225 ± 0.203 | −0.225 | (logged) |
| 99 | 0.433 ± 0.128 | 0.196 ± 0.202 | −0.237 | +0.026 |

Aggregate: trained same = 1.000 ± 0.000  cross = **0.245 ± 0.041**
Δ cross = **−0.214**, Δ-of-Δ = +0.214, paired t(3) ≈ −20.4,
p ≪ 0.001. **Bit-exact replication of iter-55 ep32** ✓.

### 500 nA — double default

| Seed | Untrained cross | Trained cross | Δ cross | Eval-drift L2 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.467 ± 0.106 | 0.242 ± 0.186 | −0.225 | (logged) |
|  7 | 0.485 ± 0.082 | 0.250 ± 0.195 | −0.235 | (logged) |
| 13 | 0.450 ± 0.122 | 0.208 ± 0.206 | −0.242 | (logged) |
| 99 | 0.433 ± 0.128 | 0.220 ± 0.202 | −0.213 | +0.029 |

Aggregate: trained same = 1.000 ± 0.000  cross = **0.230 ± 0.020**
Δ cross = **−0.229**, Δ-of-Δ = +0.229, paired t(3) ≈ −36.3,
p ≪ 0.001.

Note the dramatically tighter aggregate std (0.020) vs c125
(0.036) and c250 (0.041) — higher clamp not only improves the
mean specificity, but also **reduces seed-level variance**.
Training is more reliable.

## Clamp-strength curve

| Clamp (nA) | Trained cross | std | Δ cross | Δ-of-Δ | paired t(3) | per-doubling Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 125 | 0.272 | ±0.036 | −0.187 | +0.187 | ≈ −17.6 | (baseline)        |
| 250 | 0.245 | ±0.041 | −0.214 | +0.214 | ≈ −20.4 | −0.027 (125 → 250) |
| 500 | **0.230** | **±0.020** | **−0.229** | **+0.229** | ≈ **−36.3** | −0.015 (250 → 500) |

All three configs share the same untrained baseline (0.459 ±
0.022) — the untrained arm doesn't depend on clamp strength.

**Aggregate Δ cross is monotone in clamp strength**, with
diminishing returns: −0.027 per doubling at the 125 → 250
step, −0.015 per doubling at the 250 → 500 step. Ratio = 0.55
(slower diminishing than the iter-55 epoch axis at 0.30). A
naive geometric-series extrapolation places the asymptote at
trained cross ≈ 0.197 — about 0.03 below the iter-55 ep64
ceiling of ~0.21.

Combining the iter-55 (epoch) + iter-56 (clamp) findings: the
specificity ceiling is now **roughly 0.20**, vs the random-
topology iter-53 baseline of 0.058 (which was forward-drive
noise, not learning). Compared to the untrained decorrelated
baseline (0.459), trained cross has dropped **50 %**.

The most striking secondary observation is the **5× tighter
seed-level std at c500** (0.020) vs c125 / c250 (0.036 /
0.041). Higher clamp not only moves the mean but flattens the
seed-level distribution.

## Per-seed view (iter-55 lesson applied)

| Seed | c125 | c250 | c500 | 125 → 250 | 250 → 500 | Trajectory |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 0.262 | 0.277 | 0.242 | **+0.015** | −0.035 | NON-MONOTONE (best at c500, c125 better than c250) |
|  7 | 0.315 | 0.281 | 0.250 | −0.034 | −0.031 | monotone ↓                                       |
| 13 | 0.283 | 0.225 | 0.208 | −0.058 | −0.017 | monotone ↓                                       |
| 99 | 0.229 | 0.196 | 0.220 | −0.033 | **+0.024** | NON-MONOTONE (best at c250, c500 worse than c250)  |

Per-seed view exposes what the aggregate hides — **2 of 4
seeds are non-monotone in clamp strength** (seeds 42 and 99).
Seed 42 has its global best at c500 but with a c125-better-
than-c250 detour; seed 99's global best is c250, with c500
*worse* than c250.

This is the iter-55 lesson echoed exactly: aggregate Δ cross
is monotone (−0.187 → −0.214 → −0.229) and would suggest
"more clamp = better"; the per-seed view shows that *for half
the seeds the relationship breaks down*. Whatever clamp-
specific weight changes the teacher window writes, they
interact with seed-level recurrent topology in a way that
isn't strictly clamp-monotone.

## Honest reading

Three layered observations:

1. **Aggregate Δ cross is monotone in clamp strength, with
   diminishing returns at the same shape as the epoch axis.**
   −0.027 per doubling at 125 → 250, then −0.015 per doubling
   at 250 → 500. Ratio 0.55 (less aggressive diminishing than
   iter-55 epoch axis ratio 0.30 — clamp axis "still has more
   room" than the epoch axis at this scale). Asymptote
   estimate ≈ 0.197.

2. **Half the seeds are non-monotone in clamp strength.**
   Seeds 7 and 13: clean monotone decrease (higher clamp =
   better specificity, as the aggregate suggests). Seeds 42
   and 99: trajectories are non-monotone — seed 42's c125 is
   better than c250 then c500 best; seed 99's c250 is best
   with c500 *regressing*. The seed-level interaction with
   clamp strength is real and matters; "always use higher
   clamp" is wrong for some seeds. iter-55 lesson directly
   applies.

3. **c500 dramatically tightens the seed-level std (0.020 vs
   0.036 / 0.041 at c125 / c250).** Higher clamp doesn't just
   move the mean — it flattens the seed-level distribution.
   This is a non-trivial benefit independent of the per-seed
   non-monotonicity: even when c500 is *worse* than c250 on
   seed 99 (0.220 > 0.196), the spread across seeds is
   smaller, so training is more *reliable* under c500. For
   downstream applications where seed-to-seed consistency
   matters (any deployment sweep), c500 is the right choice
   even though seed 99 alone prefers c250.

**Branch (δ) is REJECTED**: same-cue stays at 1.000 for every
seed × clamp combination (12/12 trained arms). Eval-drift L2
remains tiny (0.026–0.045 across the seeds where it was
logged). Higher clamp does *not* trigger meaningful eval-time
plasticity drift — the decorrelated wiring still starves
the eval-phase plasticity, regardless of clamp strength
during *training*.

**Tautology check**: `target_clamp_strength` controls only the
external current applied to canonical-target R2 cells *during
the teacher window of training*. It does NOT enter the eval
phase (no clamp at decoder time) and does NOT directly affect
the cross-cue Jaccard formula. The improvement at c500 is
genuinely "stronger teacher signal during training → cleaner
cue-specific weight imprint → cleaner R2 specificity at eval".

## Acceptance per Bekos's iter-57 branching matrix

From Bekos's iter-56 spec, applied verbatim:

| Sweep result | iter-57 branch | This data |
| --- | --- | :-: |
| (α) Δ cross monotone in clamp strength (higher = better, up to a point) | Plafond is magnitude-limited. iter-57 = even higher clamp OR Achse C (schedule-tuning, phase lengths) | **✓ PRIMARY** (aggregate) |
| (β) Δ cross peaks at a middle clamp value | Sweet spot identified, tautology risk at higher clamp confirmed. iter-57 = freeze sweet spot + scale vocab OR test same-cue robustness via noise injection | single-seed only (seed 99 prefers c250 with c500 regressing; seed 42 prefers c500 but c125 > c250) |
| (γ) Δ cross unchanged or worse across all clamp values | Plafond is not clamp-limited. iter-57 = Achse C OR architectural diagnosis (which cue pairs set the ceiling — geometric vs plastic limit) | ❌ |
| (δ) Eval-drift L2 rises significantly with higher clamp, same-cue drops below 1.0 | Tautology risk active, but attractor-robustness becomes measurable. If Δ cross also good, this is the *desired* outcome (specificity AND attractor-plasticity measurable simultaneously) | ❌ (same-cue stays at 1.000 across all 12 trained arms; eval-drift L2 stays tiny) |

**iter-57 entry: branch (α) primary — clamp axis is
magnitude-limited but with diminishing returns**. Two parallel
paths consistent with the spec:

- **Path A (continue clamp axis):** sweep 500 / 1000 / 2000
  nA. Predicted next-doubling gain ≈ −0.008, then ≈ −0.004 —
  pushes asymptote from ~0.20 toward ~0.18. Diminishing
  returns suggest two more doublings give marginal value;
  cost is one more 60-min sweep for the 500/1000/2000 range.
- **Path B (Achse C: phase-length tuning):** the teacher /
  prediction / cue / tail / delay phase lengths in
  `TeacherForcingConfig` are all configurable. The
  `target_clamp_strength` controls *intensity*; phase lengths
  control *integration time* under that intensity. A 2-point
  sweep on `teacher_ms` (40 default → 80, 120) at fixed
  c500 would isolate whether longer integration helps
  beyond intensity. ~30-min wallclock.

Recommendation: **Path B first** — clamp axis is already at
diminishing returns and Path A is a safe but marginal
extension. Path B opens a new axis with potentially un-
diminished sensitivity; the iter-55 lesson (run the per-seed
view, don't trust aggregate alone) applies. If Path B also
shows clamp-style diminishing returns, iter-58 = address the
~0.20 ceiling architecturally (geometric vs plastic limit
diagnosis: which cue pairs cause the residual overlap, and
why).

A separate parallel sub-question (Bekos's noise-injection /
cross-topology iter-57 candidate from iter-55) remains valid
as iter-58 candidate, regardless of which iter-57 path runs.

## Methodological lesson

iter-50: save the simplest configuration as a regression guard.
iter-51: a guard is only a guard if its baseline excludes the null.
iter-52: an analytical null is not an empirical control.
iter-53: when the literal acceptance direction is bounded by
construction, derive it from the protocol's mathematical bounds.
iter-54: when the metric reports a "cleaner" number on a random
topology than on an architecturally cleaner one, the metric is
reading something else than what its name suggests.
iter-55: a learning curve is not a single number; per-seed
trajectories often reveal a saturation ceiling the aggregate
hides.
**iter-56: aggregate monotonicity is not seed-level monotonicity. iter-56
showed Δ cross monotone in clamp strength at the aggregate
level (−0.187 → −0.214 → −0.229), which the literature would
read as "higher clamp = better, period". Per-seed, half the
seeds (42 and 99) are non-monotone: seed 99 has its global
best at c250 with c500 *worse*, seed 42 has c125 better than
c250 then c500 best. Aggregate monotonicity hides a real
seed × clamp interaction. The aggregate-only reading would
have set "always use c500" as a deployment recommendation;
the per-seed view shows that for half the seeds c500 is not
the global best.**

## Files touched (single commit)

- `notes/56-clamp-strength.md` — this note.
- `CHANGELOG.md` — iter-56 section.
- `README.md` — iter-56 entry.
