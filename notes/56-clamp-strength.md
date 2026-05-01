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

<!-- @SWEEP_OUTPUT_125@ -->

<!-- @SWEEP_OUTPUT_250@ -->

<!-- @SWEEP_OUTPUT_500@ -->

## Clamp-strength curve

<!-- @CLAMP_CURVE@ -->

## Per-seed view (iter-55 lesson applied)

<!-- @PER_SEED@ -->

## Honest reading

<!-- @HONEST_READING@ -->

## Acceptance per Bekos's iter-57 branching matrix

From Bekos's iter-56 spec, applied verbatim:

| Sweep result | iter-57 branch | This data |
| --- | --- | :-: |
| (α) Δ cross monotone in clamp strength (higher = better, up to a point) | Plafond is magnitude-limited. iter-57 = even higher clamp OR Achse C (schedule-tuning, phase lengths) | <!-- @A1@ --> |
| (β) Δ cross peaks at a middle clamp value | Sweet spot identified, tautology risk at higher clamp confirmed. iter-57 = freeze sweet spot + scale vocab OR test same-cue robustness via noise injection | <!-- @A2@ --> |
| (γ) Δ cross unchanged or worse across all clamp values | Plafond is not clamp-limited. iter-57 = Achse C OR architectural diagnosis (which cue pairs set the ceiling — geometric vs plastic limit) | <!-- @A3@ --> |
| (δ) Eval-drift L2 rises significantly with higher clamp, same-cue drops below 1.0 | Tautology risk active, but attractor-robustness becomes measurable. If Δ cross also good, this is the *desired* outcome (specificity AND attractor-plasticity measurable simultaneously) | <!-- @A4@ --> |

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
**iter-56: <!-- @LESSON@ -->.**

## Files touched (single commit)

- `notes/56-clamp-strength.md` — this note.
- `CHANGELOG.md` — iter-56 section.
- `README.md` — iter-56 entry.
