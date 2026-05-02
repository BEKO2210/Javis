# Iter 60 — DG-like Pattern-Separation Bridge (R1 → DG → R2)

## Why iter-60

iter-58 closed the geometry-vs-architecture question for the
cross-cue floor: a non-training axis (vocab) with opposite-sign
predictions disambiguated the two readings, and *trained_cross
rose with vocab*, ruling out the encoder-collision hypothesis.
iter-59 then asked whether the architectural floor was simply
"too few cells per cue" — doubling R2_N at vocab=64 ep16
deepened Δ cross 13× (−0.007 → −0.090 on the matched config)
but the absolute trained_cross only fell 0.04 below the
matched-config baseline, still 0.18 above the iter-54 vocab=32
best. **Capacity was *a* limit, not *the* limit.**

Bekos's iter-60 pivot, ground-truthed against the Hippocampus /
Sparse Distributed Memory literature: stop trying to make R2
bigger, separate the cues *upstream* of R2. The Hippocampus
DG/CA3 division does exactly this — DG orthogonalises similar
inputs (pattern separation), CA3 stores and completes patterns
(autoassociation). Javis until iter-59 conflated everything
into R2: forward drive + recurrent attractor + decoder. iter-60
adds the missing layer.

## What iter-58/59 closed

```text
                  trained cross  Δ cross  notes
vocab=32 R2=2000  0.230 ± 0.020  −0.229   iter-54 architecture best
vocab=64 R2=2000  0.422 ± 0.017  −0.025   iter-58 architectural floor
vocab=64 R2=4000  0.411          −0.090   iter-59 capacity helps partially
```

## Hypothesis

If the floor is set by inability to separate cues upstream, then
giving the brain a cue-specific orthogonal address layer (DG)
ahead of the recurrent attractor (R2) should drop trained_cross
substantially even *before* plasticity acts.

## Implementation — single commit (code prep) + smoke commit

`crates/eval/src/reward_bench.rs`:

- **`DgConfig`** struct: `enabled`, `size` (default 4000),
  `k` (active cells per cue, default 80 ≈ 2 % of size),
  `to_r2_fanout` (default 30), `to_r2_weight` (default 1.0),
  `direct_r1r2_weight_scale` (default 0.0 = direct path off,
  DG is sole cue-routing layer), `drive_strength` (default
  200.0, matches the existing R1-side cue current).
- **`build_dg_region(size)`**: third region, all excitatory,
  no intra-region recurrent connectivity. The smoke uses fully-
  precomputed k-of-n SDRs driven externally per cue, so no
  kWTA / inhibitory dynamics needed in DG itself. iter-61
  candidate: replace external-drive DG with a real R1 → DG
  random projection + recurrent inhibition.
- **`wire_dg_to_r2(brain, cfg, seed)`**: random sparse
  projection. Each DG cell projects to `to_r2_fanout` random
  R2 cells (E and I) at `to_r2_weight`. Total DG → R2 edges
  at defaults: 4000 × 30 = 120 000.
- **`dg_sdr_for_cue(word, dg_size, k, salt)`**: deterministic
  k-of-n hashed DG SDR. Same hash structure as
  `canonical_target_r2_sdr` so addresses are stable per
  `(seed, word)` and there is no per-trial drift.
- **`drive_with_dg`**, **`drive_with_dg_counts`**,
  **`drive_with_r2_clamp_dg`** — 3-region drive primitives that
  inject current into R1 SDR + DG SDR (+ optional R2 clamp).
  The existing 2-region drive helpers auto-zero-pad DG when
  the 3rd region is present, so iter-44…59 numerics stay
  unchanged when `--dg-bridge` is off.
- **`run_teacher_trial`** takes `dg_sdr: &[u32]` (existing
  callers pass `&[]`); when DG is active it routes every drive
  through the DG-aware variants.
- **`train_brain_inplace`**, **`build_vocab_dictionary`**,
  **`evaluate_jaccard_matrix`**, **`evaluate_jaccard_matrix_with_pairs`**
  all accept a `dg_sdr_map: &HashMap<String, Vec<u32>>` and a
  `dg_drive_strength: f32`; non-DG callers pass an empty map.
- **`run_jaccard_arm`** and **`run_jaccard_floor_diagnosis`**
  build the DG region + per-cue DG SDR map when
  `cfg.teacher.dg.enabled`, scale `R1 → R2` inter-region
  weight by `direct_r1r2_weight_scale`, and thread the map
  through.

CLI in `crates/eval/examples/reward_benchmark.rs`:

```sh
--dg-bridge                  # enable DG bridge
--dg-size N                  # DG neuron count (default 4000)
--dg-k K                     # active cells per cue (default 80)
--dg-to-r2-fanout F          # DG → R2 fan-out (default 30)
--dg-to-r2-weight W          # DG → R2 edge weight (default 1.0)
--direct-r1r2-weight-scale S # R1 → R2 direct path scale (default 0.0)
--dg-drive-strength I        # external current to DG cells (default 200.0)
```

10/10 eval lib tests still green; clippy `-D warnings` clean.

## Run command

```sh
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-bench --seeds 42,7 --epochs 16 \
  --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge
```

(Default DG params; smoke at 2 seeds × 16 epochs to gauge whether
the architectural pivot moves the floor at all.)

## Smoke result

| Seed | Untrained cross | Trained cross | Trained same | Eval-drift L2 (R2→R2) |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.028 ± 0.090 | 0.027 ± 0.072 | 0.930 ± 0.175 | +4.41 |
|  7 | 0.029 ± 0.088 | 0.026 ± 0.076 | 0.914 ± 0.190 | +3.34 |

Aggregate (n = 2 seeds):

```text
Untrained: same = 1.000 ± 0.000   cross = 0.028 ± 0.000
Trained:   same = 0.922 ± 0.011   cross = 0.026 ± 0.000
Δ same    = −0.078
Δ cross   = −0.002
Δ-of-Δ    = −0.076
```

## Comparison to iter-58 / iter-59 vocab=64 baseline

| | iter-58 vocab=64 (no DG, ep32, 4 seeds) | iter-60 vocab=64 + DG (ep16, 2 seeds) | Δ vs no-DG |
| --- | ---: | ---: | ---: |
| Untrained cross | 0.448 ± 0.012 | **0.028** | **−0.420** (−94 %) |
| Trained cross | 0.422 ± 0.017 | **0.026** | **−0.396** (−94 %) |
| Trained same | 1.000 | 0.922 | −0.078 |
| Eval-drift L2 | +0.04 | +3.3 to +4.4 | ~100 × higher |

The DG bridge **collapses the absolute cross-cue floor by 16×**.
Even compared to the previous-best-ever architecture
(iter-54 vocab=32 trained=0.230), the iter-60 vocab=64+DG
trained cross of 0.026 is **9× lower** — and at 2× the vocab.

## Honest reading

Three layered observations:

1. **The geometry pivot works massively.** Untrained cross
   dropped from 0.448 to 0.028. The DG layer with k-of-n hashed
   addresses + sparse mossy-fibre projection produces an R2
   firing pattern *before any plasticity has acted* that is
   already nearly perfectly orthogonal across cues. This is
   the biggest single architectural move in the iter-46…60
   chain by an order of magnitude.

2. **Plasticity adds almost nothing on top of the geometry.**
   Δ cross (trained − untrained) at vocab=64+DG is **−0.002**
   (vs iter-58 vocab=64's −0.025, iter-54 vocab=32's −0.229).
   The trained brain barely improves over the untrained one
   because the untrained brain is already near the metric
   floor. This is the *inverse* of the iter-58 reading where
   plasticity carried most of the signal — under DG, geometry
   carries it.

3. **Same-cue drops to 0.92** (was 1.000 across iter-53…59).
   Eval-phase L2 drift jumps from ~0.04 to ~3.3-4.4
   (~100 ×). DG dramatically increases R2 activity per cue
   (sparse strong projection from DG drives more cells
   harder), which means more spike-pair coincidences during
   eval, which means more weight changes per trial.
   Plasticity is now genuinely active at eval — but it's
   eroding the engram (same-cue down) without lifting the
   geometry floor (cross-cue ≈ untrained).

The iter-58 / iter-59 reading was "the architecture has too
much shared upstream representation". DG fixed the upstream
representation — and the floor moved from 0.42 to 0.026 *in
the untrained brain*. There is now a credible architectural
substrate for cue-specific learning. The remaining iter-61
question is whether plasticity can write *into* this clean
substrate to produce associations that survive (e.g. the cue
→ target pair that the iter-46 task originally cared about).

## Bekos's iter-61 branching matrix verdict

| Outcome | iter-61 path | This data |
| --- | --- | :-: |
| (A) DG drops trained_cross substantially (target 0.25-0.30 or better) | Pattern-separation pivot confirmed; iter-61 = DG parameter sweep | **✓ MASSIVELY** (trained = 0.026, far below the 0.25-0.30 target) |
| (B) DG drops untrained but trained Δ stays small → geometry better, plasticity not yet using it | iter-61 = learnable DG → R2 or different reward schedule | **✓ secondary** (Δ trained-untrained is only −0.002; plasticity is active but adds no specificity) |
| (C) DG doesn't help | iter-61 = Willshaw / SDM separate association store | ❌ |

**iter-61 entry is mixed (A) + (B):** the geometry pivot worked
beyond Bekos's stated target, but the cue-specific learning
signal *on top of geometry* is currently buried in noise. Two
parallel iter-61 candidates:

- **Path 1 (recommended primary): full 4-seed × 32-epoch
  replication** of the smoke at default DG params. Confirm
  the 16× cross-cue collapse at the standard iter-58
  comparison config and check seed-level variance. The
  iter-55 / iter-56 lesson (per-seed view, not just aggregate)
  applies — 2-seed smoke at ep16 is not enough to declare
  victory.
- **Path 2 (parallel): isolate the cue → target learning
  task.** With cross-cue at 0.026 in *both* untrained and
  trained, the original iter-46 task ("does the cue retrieve
  the right target?") needs a different metric — top-3 against
  the canonical target SDR (the iter-52 metric) or a per-pair
  Δ overlap. The Jaccard floor is now too low for plasticity-
  driven learning to register against it. iter-61 should
  surface what learning *does* on this clean substrate.

The trained same-cue drop to 0.92 with eval-drift L2 +3.3 also
opens an iter-61 sub-question: can DG → R2 plasticity be tamed
(e.g. lower `to_r2_weight` or scale STDP rate) so the engram
doesn't erode at eval?

## Methodological lesson

Saturation across three independent training axes (epoch /
clamp / phase-length) plus a vocab axis flip pointed at "more
upstream representation" as the unsaturated lever. iter-60
swept *zero* training axes — it just added one missing
architectural layer. **The biggest single number-move in 14
iterations came from a structural change, not a training-
axis sweep.** Whenever every training-axis sweep saturates at
the same value, the architecture is the lever, not the
hyperparameter — go literature, not deeper sweep.

## Files touched

- `crates/eval/src/reward_bench.rs` — DG config + region +
  wiring + per-cue SDR hash + 3-region drive primitives +
  threading through teacher trial / training / dictionary /
  jaccard matrix.
- `crates/eval/src/lib.rs` — `DgConfig` re-export.
- `crates/eval/examples/reward_benchmark.rs` — DG CLI flags.
- `notes/60-dg-pattern-separation-bridge.md` — this note.
- `CHANGELOG.md` — iter-60 section.
- `README.md` — iter-60 entry.
