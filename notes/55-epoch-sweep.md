# Iter 55 — Epoch sweep on decorrelated + plasticity architecture

## Why iter-55

iter-54 landed branch M1 of Bekos's pre-fixed iter-55 selector:
**Δ-of-Δ = +0.160, ACCEPTANCE PASSED** with the hard-decorrelated
R1 → R2 init. Trained cross-cue dropped from 0.459 to 0.299
(35 % below untrained, p ≪ 0.001 paired across 4 seeds), trained
same-cue stayed at 1.000 (no attractor erosion), eval-phase L2
drift collapsed from iter-53's +25–29 down to +0.02–0.25.

Per Bekos's iter-55 spec, M1 mandates **keep decorrelation +
plasticity combined, sweep training schedule** to characterise
the learning curve. Three configs: 16 / 32 / 64 epochs, all
other parameters identical to iter-54 (seeds 42, 7, 13, 99 +
`--decorrelated-init --teacher-forcing`).

The 16-epoch run is a **replication** of iter-54 — values
should reproduce 0.299 / 0.459 within seed-level noise. Any
deviation would surface a non-determinism bug we missed.

## Implementation

No new code. Pure sweep-config exercise on the iter-54
infrastructure. The `--epochs N` CLI flag already accepted
arbitrary values; iter-55 just runs three.

## Run — three configs × 4 seeds

```sh
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-bench --seeds 42,7,13,99 --epochs 16 \
  --decorrelated-init --teacher-forcing

cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-bench --seeds 42,7,13,99 --epochs 32 \
  --decorrelated-init --teacher-forcing

cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-bench --seeds 42,7,13,99 --epochs 64 \
  --decorrelated-init --teacher-forcing
```

<!-- @SWEEP_OUTPUT_16@ -->

<!-- @SWEEP_OUTPUT_32@ -->

<!-- @SWEEP_OUTPUT_64@ -->

## Learning curve — Δ cross vs epochs

<!-- @LEARNING_CURVE@ -->

## Honest reading

<!-- @HONEST_READING@ -->

## Acceptance per Bekos's iter-56 branching matrix

From Bekos's iter-55 spec, applied verbatim:

| Sweep result | iter-56 branch | This data |
| --- | --- | :-: |
| (i) Δ cross sinkt monoton mit Epochen, same-cue sinkt unter 1.0 → Lernen schreitet voran und Attraktor-Plastizität ist messbar | iter-56 = noch mehr Epochen ODER Cross-Validation auf größerem Vokabular ODER Attraktor-Robustheits-Test (Noise-Injection) | <!-- @B1@ --> |
| (ii) Δ cross saturiert (z. B. −0.16 / −0.18 / −0.18) → Plastizität hat ein Plafond unter diesem Schedule | iter-56 = Achse B (Clamp-Strength-Sweep) | <!-- @B2@ --> |
| (iii) Δ cross verschlechtert sich bei längerem Training → Catastrophic Interference oder Over-Training | iter-56 = Schedule-Inspektion oder Consolidation-Mechanismus | <!-- @B3@ --> |
| (iv) Same-cue bleibt 1.000 auch bei 64 ep → Plastizität ist während Eval still, weil decorrelated init keine Pre-Post-Material liefert | iter-56-Pfad: Attraktor-Robustheit über Noise-Injection beim Eval, oder explizite Cross-Topology-Tests | <!-- @B4@ --> |

## Methodological lesson

iter-50: save the simplest configuration as a regression guard.
iter-51: a guard is only a guard if its baseline excludes the null.
iter-52: an analytical null is not an empirical control.
iter-53: when the literal acceptance direction is bounded by
construction, derive it from the protocol's mathematical bounds.
iter-54: when the metric reports a "cleaner" number on a random
topology than on an architecturally cleaner one, the metric is
reading something else than what its name suggests.
**iter-55: <!-- @LESSON@ -->.**

## Files touched (single commit)

- `notes/55-epoch-sweep.md` — this note.
- `CHANGELOG.md` — iter-55 section.
- `README.md` — iter-55 entry.
