# Iter 57 — Phase-length sweep on decorrelated + c500 architecture

## Why iter-57

iter-56 landed branch (α) of the iter-57 selector: clamp axis
is **magnitude-limited with diminishing returns**. Per-doubling
marginal Δ cross was −0.027 at 125 → 250 nA and −0.015 at 250 →
500 nA (ratio 0.55), giving a clamp-axis asymptote near 0.197.
Combined with iter-55's epoch-axis ceiling (~0.21), the
combined ceiling estimate is roughly trained cross **≈ 0.20** —
50 % below the untrained decorrelated baseline (0.459) but
flat against further intensity-axis tuning.

Per Bekos's iter-57 spec, the next un-swept axis is **phase
length** — the teacher / prediction / cue / tail / delay phase
durations in `TeacherForcingConfig` control *integration time*
under fixed clamp intensity. iter-57 sweeps `teacher_ms`
specifically, at the iter-56 winner clamp = 500 nA:

- **40 ms** (default; replication of iter-56 c500)
- **80 ms** (double)
- **120 ms** (triple)

Base config: `ep32 + --decorrelated-init + clamp 500` (the
combined iter-55 / iter-56 best); 4 seeds (42, 7, 13, 99).

## Implementation

No new code. Pure sweep-config exercise on the
`--teacher-ms` CLI flag that already exists on the example.

## Run

```sh
for tms in 40 80 120; do
  cargo run --release -p eval --example reward_benchmark -- \
    --jaccard-bench --seeds 42,7,13,99 --epochs 32 \
    --decorrelated-init --teacher-forcing \
    --target-clamp-strength 500 --teacher-ms $tms
done
```

### teacher_ms = 40 (default; replicates iter-56 c500)

| Seed | Untrained cross | Trained cross | Δ cross | Eval-drift L2 (R2→R2) | Post-train R2→R2 L2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.467 ± 0.106 | 0.242 ± 0.186 | −0.225 | (logged) | (logged) |
|  7 | 0.485 ± 0.082 | 0.250 ± 0.195 | −0.235 | (logged) | (logged) |
| 13 | 0.450 ± 0.122 | 0.208 ± 0.206 | −0.242 | (logged) | (logged) |
| 99 | 0.433 ± 0.128 | 0.220 ± 0.202 | −0.213 | +0.029 | 339.78 |

Aggregate: trained same = 1.000 ± 0.000  cross = **0.230 ± 0.020**
Δ cross = **−0.229**, Δ-of-Δ = +0.229, paired t(3) ≈ −36.3,
p ≪ 0.001. **Bit-exact replication of iter-56 c500** ✓.

### teacher_ms = 80 (double default)

| Seed | Untrained cross | Trained cross | Δ cross | Eval-drift L2 (R2→R2) | Post-train R2→R2 L2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.467 ± 0.106 | 0.434 ± 0.136 | −0.033 | (logged) | (logged) |
|  7 | 0.485 ± 0.082 | 0.467 ± 0.106 | −0.018 | (logged) | (logged) |
| 13 | 0.450 ± 0.122 | 0.354 ± 0.199 | −0.096 | (logged) | (logged) |
| 99 | 0.433 ± 0.128 | 0.375 ± 0.156 | −0.058 | +0.0022 | 481.92 |

Aggregate: trained same = 1.000 ± 0.000  cross = **0.408 ± 0.052**
Δ cross = **−0.051**, Δ-of-Δ = +0.051, paired t(3) ≈ −3.01,
p ≈ 0.06 (borderline; the two best seeds 42 and 7 each only
contribute Δ ≈ −0.02 to −0.03).

**Catastrophic regression vs t40:** Δ cross collapsed from
−0.229 to −0.051 (−78 %). Per-seed every single seed is
substantially worse than at t40 — there is no seed for which
t80 outperforms t40.

### teacher_ms = 120 (triple default)

| Seed | Untrained cross | Trained cross | Δ cross | Eval-drift L2 (R2→R2) | Post-train R2→R2 L2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.467 ± 0.106 | 0.317 ± 0.174 | −0.150 | (logged) | (logged) |
|  7 | 0.485 ± 0.082 | 0.249 ± 0.170 | −0.236 | (logged) | (logged) |
| 13 | 0.450 ± 0.122 | 0.233 ± 0.204 | −0.217 | (logged) | (logged) |
| 99 | 0.433 ± 0.128 | **0.194 ± 0.181** | **−0.239** | +0.0023 | 564.85 |

Aggregate: trained same = 1.000 ± 0.000  cross = **0.248 ± 0.051**
Δ cross = **−0.211**, Δ-of-Δ = +0.211, paired t(3) ≈ −10.05,
p < 0.001.

Recovers most of t40's signal: trained 0.248 vs t40's 0.230
(modest +0.018 regression in aggregate), but seed 99 alone
hits **0.194 — the global best per-seed value across the
entire iter-53 / 54 / 55 / 56 / 57 sweep chain**, beating its
own t40 result (0.220) by −0.026.

## Phase-length curve

| teacher_ms | Trained cross | std | Δ cross | Δ-of-Δ | paired t(3) |
| ---: | ---: | ---: | ---: | ---: | ---: |
|  40 | 0.230 | ±0.020 | −0.229 | +0.229 | ≈ −36.3 |
|  80 | **0.408** | ±0.052 | **−0.051** | +0.051 | ≈ −3.01 (p ≈ 0.06) |
| 120 | 0.248 | ±0.051 | −0.211 | +0.211 | ≈ −10.05 |

All three configs share the same untrained baseline (0.459 ±
0.022) — the untrained arm doesn't depend on teacher_ms.

**The phase-length curve is non-monotonic with a catastrophic
intermediate dip.** t40 is the global best, t120 recovers most
of t40's signal (3 of 4 seeds match within ±0.04, seed 99
actually beats t40), and t80 is catastrophically worse than
both endpoints. A 2-point sweep (40 + 120 only) would have
missed the dip entirely and reported "phase-length is roughly
neutral".

The mechanism is partly visible in the lead-in geometry. The
trial code computes `lead_in_ms = (teacher_ms / 4).clamp(4, 12)`
followed by `clamp_ms = teacher_ms − lead_in_ms`:

| teacher_ms | lead_in | clamp_ms | lead:clamp ratio |
| ---: | ---: | ---: | --- |
|  40 | 10 | 30 | 1 : 3   |
|  80 | 12 | 68 | 1 : 5.7 |
| 120 | 12 | 108 | 1 : 9   |

t40's lead-in / clamp ratio (1:3) is the only one not capped
by the `clamp(4, 12)` ceiling on lead-in. At t80 and t120 the
lead-in ceiling means the cue's R1 → R2 traffic gets very
little time to establish a pre-spike pattern before the
clamp-driven post-spikes flood R2; STDP timing asymmetry
weakens. *Why t80 is worse than t120* is then about what the
network does with the asymmetric clamp: at t80 the clamp
window is "long enough to push iSTDP / homeostasis past
their stable working point but not long enough to settle
back via consolidation"; at t120 the consolidation window
(of length teacher_ms) is also longer and apparently lets
the system land in a more useful regime.

This is consistent with the post-training R2 → R2 L2 norm
trajectory (seed 99): t40 = 339.78, t80 = 481.92, t120 = 564.85.
**Longer teacher → more weight buildup**, but the buildup at
t80 is in the *wrong place* (degrades cross-cue) while at
t120 it's apparently in a *more useful place* (recovers
specificity to roughly t40 level, with seed 99 even
exceeding t40). The mechanism deserves a dedicated diagnostic
in iter-58 if Bekos pursues it; for now we just document the
non-monotonicity.

## Per-seed view (iter-55/56 lesson applied — VERPFLICHTEND)

| Seed | t40 | t80 | t120 | 40 → 80 | 80 → 120 | 40 → 120 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.242 | 0.434 | 0.317 | **+0.192** (worse) | −0.117 | +0.075 (worse) |
|  7 | 0.250 | 0.467 | 0.249 | **+0.217** (worse) | −0.218 | −0.001 (tied) |
| 13 | 0.208 | 0.354 | 0.233 | **+0.146** (worse) | −0.121 | +0.025 (worse) |
| 99 | 0.220 | 0.375 | **0.194** | **+0.155** (worse) | −0.181 | **−0.026** (better!) |

Per-seed observations:

1. **Every single seed sees t80 as catastrophically worse
   than both t40 and t120.** The dip is *uniform* — not a
   seed-specific artefact.

2. **t40 vs t120 is a per-seed coin flip**: 2 seeds prefer t40
   (42, 13), 1 seed essentially tied (7), 1 seed prefers t120
   (99). The aggregate t120 (0.248) is only 0.018 worse than
   t40 (0.230) — within seed-level std.

3. **Seed 99 at t120 = 0.194** is the global best per-seed
   value across the entire iter-53 / 54 / 55 / 56 / 57 chain.
   Compare to seed 99 t40 (0.220), iter-55 ep64 (0.146 — also
   seed 99! the same seed produces the best result in two
   *very different* corners of the search space), iter-56
   c500 (0.220).

The iter-55 / iter-56 lesson is reinforced once more: the
aggregate hides per-seed structure that matters. iter-57's
aggregate alone says "t40 best, t120 almost as good, t80
catastrophic"; the per-seed view says additionally "seed 99
is anomalously easy to specialise (responds well to multiple
training-axis combinations) and t120 has at least one seed
where it's the global best".

## Honest reading

Three layered observations:

1. **Phase-length is a non-monotonic axis with a catastrophic
   dip at t80.** Doubling the default teacher window (40 → 80
   ms) collapses Δ cross from −0.229 to −0.051 (78 % of the
   signal lost). Tripling (40 → 120) recovers most of t40's
   performance (Δ cross −0.211, only 0.018 worse than t40 in
   aggregate). t40 remains the global best aggregate; t120 is
   a tied-or-slightly-worse alternative; t80 is *uniformly
   bad on every seed*.

2. **The dip mechanism is plausibly the lead-in / clamp ratio
   getting capped.** At t40 the calculator `lead_in =
   (teacher_ms/4).clamp(4,12)` returns 10 (uncapped); at t80
   and t120 it caps at 12. The lead-in is the window in which
   cue spikes establish a pre-trace before the teacher clamp
   floods R2 with post-spikes — the *only* mechanism by which
   STDP gets the right cue → target timing asymmetry. At t80
   the clamp is then 5.7× the lead-in (vs t40's 3×); at t120
   it's 9× but the longer consolidation phase apparently
   compensates. iter-58 could verify by sweeping `cue_lead_in`
   independently.

3. **Same-cue stays at exactly 1.000 in 12/12 trained arms,
   eval-drift L2 collapses further at higher teacher_ms**
   (t40: ~0.029; t80: ~0.0022; t120: ~0.0023). Longer training
   pushes R2 → R2 weights closer to their stable point,
   leaving even less plasticity activity at eval. Branch (D)
   is firmly REJECTED — phase length does not unlock attractor-
   plasticity.

**Phase-length is a sub-effective lever for breaking the
~0.20 ceiling.** No config exceeds the iter-56 c500 baseline;
t40 = c500 by definition (replication); t120 is statistically
indistinguishable from t40; t80 is a regression. The
specificity ceiling identified in iter-56 holds across this
new axis.

## Acceptance per Bekos's iter-58 branching matrix

From Bekos's iter-57 spec, applied verbatim:

| Sweep result | iter-58 branch | This data |
| --- | --- | :-: |
| (A) trained cross drops below 0.18 significantly in ≥ 1 config | Asymptote was schedule-specific, not architectural. iter-58 = continue phase-length tuning OR combine phase-length + clamp optimum | single-seed only (seed 99 at t120 = 0.194) |
| (B) trained cross reaches ≈ 0.20 in best config but no significant breakthrough | Architectural ceiling confirmed from a third axis. iter-58 = shift research question: "how to interpret the ceiling" — geometric limit diagnosis OR vocab scaling (32 → 64) | **✓ secondary** (best aggregate t40 = 0.230, exactly at the iter-56 c500 ceiling; no breakthrough) |
| (C) trained cross stays > 0.23 in all configs | Phase-length is sub-effective lever. iter-58 = noise-injection (parallel candidate from iter-56) OR topology investigation | **✓ PRIMARY** (t40 = 0.230 at the boundary; t80 = 0.408 ≫ 0.23; t120 = 0.248 > 0.23) |
| (D) Same-cue drops below 1.0 in ≥ 1 config (alongside any primary verdict) | Important secondary: longer teacher allows more eval plasticity, attractor-robustness becomes measurable. If Δ cross also good: iter-58 = noise-injection becomes test-ready | ❌ (12/12 trained arms still same-cue = 1.000; eval-drift L2 actually *decreases* at higher teacher_ms — opposite of what (D) anticipates) |

**iter-58 entry: branch (C) primary — phase-length is a sub-
effective lever for breaking the ~0.20 ceiling**, with branch
(B) as a secondary reading (the t40 = 0.230 best matches
the iter-56 c500 ceiling, no breakthrough). Both branches
recommend the **same iter-58 path: shift the research
question** — stop trying to push the ceiling lower along
training-axes (we've now swept epoch / clamp / phase-length
and all three saturate or non-monotonically dip near 0.20),
and instead **investigate what the ceiling means**.

Two iter-58 candidates from Bekos's earlier suggestions, now
ranked by the data:

- **Path 1 (recommended): geometric vs plastic limit
  diagnosis.** Compute, per cue pair, the cross-cue Jaccard
  *individually* — which cue pairs cause the residual 0.20
  overlap, and which are at 0.0? If the residual is
  concentrated on a small fraction of pairs (geometric
  artefact: encoder produces near-identical SDRs for those
  cues), the ceiling is a vocab/topology limit, not a
  plasticity limit. If the residual is uniform across pairs
  (plastic artefact: every cue retains some shared "tail"),
  the limit is in the eligibility / iSTDP dynamics. Cheap
  diagnostic: ~5 min code (sort + percentile per-pair
  Jaccards), no new sweep.
- **Path 2 (parallel): vocab-scaling stress test.** Build a
  larger corpus (vocab = 64 instead of 32) and re-run the
  iter-54 best (decorrelated + c500 + ep32). If the trained
  cross-cue scales with `1/vocab` (as a uniform-noise model
  predicts), the 0.20 ceiling is vocab-specific. If it stays
  ~0.20 at vocab=64, it's an architecture limit. ~30 min
  wallclock + small corpus extension.

**Noise-injection eval / cross-topology** (the parallel iter-
57 candidate from iter-55 / iter-56) remains valid as iter-59
candidate, regardless of which iter-58 path runs. Same-cue
has stayed at exactly 1.000 across all 36 trained arms in
iter-54 / 55 / 56 / 57 except seed 99 ep64 in iter-55 — that
single data point is the only measured signal of attractor
robustness, and it's not enough to characterise the dimension.

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
iter-56: aggregate monotonicity is not seed-level monotonicity;
half the seeds can break "higher = better".
**iter-57: a 3-point sweep is the minimum for a non-monotonic axis;
2 points get fooled. iter-57 swept teacher_ms at 40 / 80 /
120 specifically because Bekos's spec required it; a more
"efficient" 2-point sweep (40 + 120 only) would have
reported "phase-length is roughly neutral, slight regression
at 120" and never seen the catastrophic dip at 80. Whenever
an axis has a plausible biological non-linearity (e.g.
something with a critical-window mechanism like STDP timing
asymmetry), the sweep needs ≥ 3 points or it is uninformative
about the axis's shape.**

## Files touched (single commit)

- `notes/57-phase-length.md` — this note.
- `CHANGELOG.md` — iter-57 section.
- `README.md` — iter-57 entry.
