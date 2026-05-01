# Iter 54 — Hard-decorrelated R1 → R2 init + Jaccard re-eval

## Why iter-54

iter-53's 4-seed × 16-epoch sweep produced **Δ-of-Δ = −0.121,
acceptance FAILED**: plasticity drifted weights (eval-phase L2
+25 to +29 on every seed) and produced *some* attractor
structure (trained same-cue = 0.879 vs untrained 1.000), but
the trained brain showed **zero cross-cue specificity gain**
(trained 0.058 ≈ untrained 0.058 ± 0.003). The honest reading:
training redistributes weight mass and creates per-cue basins,
but those basins are not aligned with cue identity at the
decoder layer — forward-drive bias (the iter-52 finding) is
unbroken by 16 epochs of teacher-forcing.

Bekos's iter-54 spec attacks the bottleneck at the architecture
layer: replace the random-FAN_OUT R1 → R2 wiring with a
**hard-decorrelated** projection where each vocab word's R1
SDR cells project *only* into a disjoint block of R2-E cells.
The mechanical invariant (`assert_decorrelated_disjoint`) is the
iter-54 equivalent of iter-52's L2 bit-identity check.

## Implementation — single commit

New code in `crates/eval/src/reward_bench.rs`:

- `wire_forward_decorrelated(brain, encoder, vocab, seed,
  inter_weight) -> Vec<Vec<usize>>` — partition the R2-E pool
  into `vocab.len()` disjoint blocks; for each R1 cell that
  appears in *exactly one* cue SDR, fan out `FAN_OUT` times
  (with replacement) into that cue's block. R1 cells that
  appear in multiple SDRs are **dropped from connectivity
  entirely** — the only way to preserve pairwise R2-reach
  disjointness given a non-disjoint encoder.
- `assert_decorrelated_disjoint(brain, encoder, vocab)` —
  end-to-end invariant: for every cue pair `(i, j), i < j`,
  the set of R2 cells reachable from cue *i*'s R1 SDR via any
  R1 → R2 edge must be disjoint from cue *j*'s. Iterates
  `brain.outgoing[0][...]` + `brain.inter_edges` (no shortcut
  to the block allocation), so it would catch a wiring bug
  the topology code accidentally introduced.
- `TeacherForcingConfig.decorrelated_init: bool` (default
  `false` keeps iter-46/53 random topology).
- `--decorrelated-init` CLI flag in
  `crates/eval/examples/reward_benchmark.rs`.
- Unit test `decorrelated_init_is_pairwise_disjoint` —
  builds the wiring against the real default corpus + encoder,
  asserts both block-level disjointness AND end-to-end
  reachability disjointness.

Block math at iter-46/53 defaults: vocab = 32, R2-E ≈ 1400
(R2_N = 2000, R2_INH_FRAC = 0.30) ⇒ block_size = 43 cells per
cue. The encoder produces ~17 unique R1 cells per word on
average (out of ENC_K = 20, with ~3 shared cells dropped),
giving ~17 × 12 = 204 connections per cue — sparser than the
random 12 000-edge baseline but cleanly cue-specific.

## Run — 4 seeds × 16 epochs × {trained, untrained}, --decorrelated-init

```sh
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-bench --seeds 42,7,13,99 --epochs 16 \
  --teacher-forcing --decorrelated-init
```

<!-- @SWEEP_OUTPUT@ -->

## Honest reading

<!-- @HONEST_READING@ -->

## Acceptance status

<!-- @ACCEPTANCE@ -->

## What iter-54 should NOT be read to claim

- ❌ "Decorrelated init solves cue-specificity." It re-routes
  the question. Cross-cue at the decoder is no longer
  forward-drive-uniform; whether the *Jaccard* signal then
  improves depends on the recurrent dynamics + plasticity
  combination — measured by the sweep, not assumed.
- ❌ "FAN_OUT = 12 with replacement on a 43-cell block is the
  right density." The block size was determined mechanically
  by `R2-E / vocab`, not optimised. iter-55 may need to
  revisit it.
- ❌ "iter-54 is a substitute for plasticity." Decorrelation
  is a *structural prior* on which cells-route-to-which; it
  doesn't replace plasticity, it gives plasticity a more
  cue-specific surface to deepen.

## Methodological lesson

iter-50: save the simplest configuration as a regression guard.
iter-51: a guard is only a guard if its baseline excludes the null.
iter-52: an analytical null is not an empirical control.
iter-53: when the literal acceptance direction is bounded by
construction, derive it from the protocol's mathematical bounds.
**iter-54: when the metric reports a "cleaner" number on a
random topology than on an architecturally cleaner one, the
metric is reading something else than what its name suggests.
Re-derive what the metric measures under the new topology
*before* claiming a result.**

The disjointness invariant (`assert_decorrelated_disjoint`,
called every run) is the iter-54 equivalent of iter-52's L2
bit-identity check. Both catch a class of "the protocol leaked"
bug that the visible config would have hidden.

## iter-55 branching matrix (Bekos's pre-fixed version)

From Bekos's iter-54 spec, applied verbatim against the sweep
result:

| Sweep result | iter-55 branch | This data |
| --- | --- | :-: |
| Δ-of-Δ > 0 (specificity gain ≫ erosion) | Plasticity + decorrelation combined — keep both, sweep training schedule / epochs / target_clamp_strength to maximise the gain | <!-- @M1@ --> |
| Δ-of-Δ ≤ 0 *despite* significant cross-cue Δ | Engrams are fragile — iter-55 = consolidation: replay / sleep schedule / structural compaction / heterosynaptic stabilisation | <!-- @M2@ --> |
| Trained cross-cue ≈ untrained cross-cue (no significant Δ) | Decorrelation reicht nicht — problem deeper, iter-55 = ernsthafte Topologie/Bridge-Frage (R2 → R2 sparsity? per-block recurrent structure? bridge layer?) | <!-- @M3@ --> |

## Files touched (single commit)

- `crates/eval/src/reward_bench.rs` —
  `wire_forward_decorrelated` + `assert_decorrelated_disjoint`
  + `decorrelated_init` field + branch in `run_jaccard_arm` +
  `decorrelated_init_is_pairwise_disjoint` unit test.
- `crates/eval/examples/reward_benchmark.rs` —
  `--decorrelated-init` CLI flag + plumbing.
- `notes/54-decorrelated-init.md` — this note.
- `CHANGELOG.md` — iter-54 section.
- `README.md` — iter-54 entry.
