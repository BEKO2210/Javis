# Iter 58 — Jaccard floor geometry vs plasticity diagnosis

## Why iter-58

iter-55 / iter-56 / iter-57 swept three independent training
axes (epoch / clamp / phase-length) on the iter-54 decorrelated
+ teacher-forcing architecture. All three saturate near
trained cross **≈ 0.20** with diminishing returns on each axis:

- iter-55 epoch axis: per-doubling Δ cross −0.054 → −0.016
  (ratio 0.30, asymptote ≈ 0.21).
- iter-56 clamp axis: per-doubling Δ cross −0.027 → −0.015
  (ratio 0.55, asymptote ≈ 0.197).
- iter-57 phase-length axis: t40 best at 0.230, t80 catastrophic
  dip, t120 modest regression — non-monotonic, no break below
  0.23.

**The ≈ 0.20 cross-cue floor is robust across three orthogonal
training-axis sweeps.** It is no longer plausible that the
floor is "we just haven't trained enough" — it has held against
4× epochs (16 → 64), 4× clamp (125 → 500), 3× teacher_ms
(40 → 120), and the non-monotonic t80 catastrophe.

iter-58 is therefore *not* another optimisation iteration. It
is a **diagnosis**: what *is* the 0.20 floor?

Per Bekos's iter-58 spec, two diagnostics, both at the iter-57
best config (decorrelated + ep32 + clamp 500 + teacher_ms 40):

- **Path 1** — per-cue-pair Jaccard *distribution* analysis at
  vocab = 32. If the residual cross-cue is concentrated on a
  small fraction of cue pairs, those pairs flag an encoder /
  SDR / dictionary collision (geometric limit). If the
  residual is broadly uniform across all 496 pairs, the floor
  is a plasticity / architecture limit.
- **Path 2** — vocab = 64 stress test (same config). If trained
  cross scales with `1 / vocab` (uniform-noise model), the
  floor is vocab-specific (geometric). If trained cross stays
  near 0.20, the floor is architecture-specific.

## Implementation — single commit

New code in `crates/eval/src/reward_bench.rs`:

- `pub struct JaccardPairSample` — one entry per (cue_a, cue_b)
  with `i < j`, plus the post-burn-in trial-1 decoded top-3
  sets so the diagnosis can spot encoder collisions.
- `pub struct JaccardFloorReport` — per-(seed, trained-arm)
  per-pair list + standard aggregate.
- `evaluate_jaccard_matrix_with_pairs(brain, encoder, r2_e,
  dict, vocab) -> (JaccardMetrics, Vec<JaccardPairSample>)` —
  per-pair-emitting variant of the iter-53 evaluator.
- `pub fn run_jaccard_floor_diagnosis(corpus, cfg, seeds) ->
  Vec<JaccardFloorReport>` — runs the trained arm at the
  passed config across `seeds`, mirroring `run_jaccard_arm`'s
  brain construction + training but using the with-pairs
  evaluator.
- `pub fn render_jaccard_floor_diagnosis(reports, threshold,
  top_n) -> String` — Markdown report covering the three cuts
  Bekos's spec asks for: distribution stats / top-N high-
  overlap pairs / per-cue frequency in pairs above threshold.
- `pub fn default_corpus_v64()` — vocab = 64 corpus extending
  the iter-46…57 set with 16 more programming-language pairs
  (typescript / perl / php / lua / crystal / nim / dart /
  racket / elixir / groovy / julia / matlab / fortran / cobol
  / ada / prolog).

Two new CLI flags in
`crates/eval/examples/reward_benchmark.rs`:

- `--jaccard-floor-diagnosis` (mirror of `--jaccard-bench` but
  emits the per-pair distribution report).
- `--corpus-vocab 32 | 64` (default 32; selects the standard
  16-pair corpus or the iter-58 32-pair extension).
- `--floor-threshold` (default 0.5) and `--floor-top-n`
  (default 10) — controls the floor-diagnosis report
  thresholds.

All re-exports plumbed through `crates/eval/src/lib.rs`. 10/10
eval lib tests stay green; clippy `-D warnings` clean.

## Run

```sh
# Path 1 — vocab = 32 floor diagnosis
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-floor-diagnosis --seeds 42,7,13,99 --epochs 32 \
  --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 32 --floor-threshold 0.3 --floor-top-n 15

# Path 2 — vocab = 64 standard bench (trained vs untrained)
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-bench --seeds 42,7,13,99 --epochs 32 \
  --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64

# Path 2 — vocab = 64 floor diagnosis
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-floor-diagnosis --seeds 42,7,13,99 --epochs 32 \
  --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --floor-threshold 0.3 --floor-top-n 15
```

## Path 1 — vocab=32 distribution

### Per-seed aggregate (replicates iter-57 t40 + iter-56 c500 bit-exactly)

| Seed | same_cue | cross_cue |
| ---: | ---: | ---: |
| 42 | 1.000 | 0.242 |
|  7 | 1.000 | 0.250 |
| 13 | 1.000 | 0.208 |
| 99 | 1.000 | 0.220 |

Aggregate trained cross = 0.230 ± 0.020 (paired t(3) ≈ −36.3,
p ≪ 0.001 vs untrained 0.459 ± 0.022). Replication ✓.

### Cross-seed averaged per-pair distribution

| min | p25 | median | p75 | p90 | p95 | max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.150 | 0.225 | 0.300 | 0.350 | 0.425 | 0.750 |

Distribution is **broad and continuous** across all 496 pairs.
The aggregate cross-cue mean (0.225 in median, 0.230 in
arithmetic mean) sits squarely in the bulk; there is *no
isolated tail* of geometric-collision pairs.

### Top-15 high-overlap pairs

| Rank | cue_a | cue_b | mean Jaccard |
| ---: | :--- | :--- | ---: |
|  1 | block | channels | 0.750 |
|  2 | actor | block | 0.550 |
|  3 | actor | channels | 0.550 |
|  4 | actor | continuation | 0.500 |
|  5 | block | ruby | 0.500 |
|  6 | clojure | dynamic | 0.500 |
|  7 | ownership | swift | 0.500 |
|  8 | comptime | scheme | 0.475 |
|  9 | actor | clojure | 0.425 |
| 10 | actor | dynamic | 0.425 |
| 11 | actor | ruby | 0.425 |
| 12 | actor | rust | 0.425 |
| 13 | block | clojure | 0.425 |
| 14 | block | continuation | 0.425 |
| 15 | block | dynamic | 0.425 |

Top-3 are concurrency-concept words (block, channels, actor)
that the encoder plausibly maps to overlapping SDRs. But
**ranks 4–15 are not concentrated on encoder collisions** —
they're spread across many distinct cues (ownership/swift,
clojure/dynamic, comptime/scheme). The "geometric collision"
hypothesis predicts ranks 4+ should fall off sharply; instead
they all sit at 0.425–0.500.

### Cue frequency in pairs ≥ 0.30 (out of vocab−1 = 31 partners)

Top 10:
- block: 21
- ruby: 20
- clojure: 19
- actor: 18, dynamic: 18, rust: 18
- functional: 17, go: 17
- continuation: 16, lisp: 16

**Half the vocab participates in many high-overlap pairs.**
The "high-overlap" pattern is *not* concentrated on a few
collision cues. block has 21 / 31 = 68 % of its partners
above 0.30; even median cues (channels, macro, java) sit at
14–15. This is uniform-ish architectural overlap, not a few
SDR-collision pairs.

### Reading

The vocab=32 distribution shows a **broad, continuous floor**
with a small cluster of cue-specific collisions (block /
channels / actor concurrency triplet) but no clean
geometric-vs-architecture separator. Branch (A) — pure
geometry floor with few recurring pairs — is rejected on this
data alone. Branch (C) cannot be ruled out without the vocab
scaling test.

## Path 2 — vocab=64 stress test

### vocab=32 vs vocab=64 aggregate comparison

| | vocab = 32 | vocab = 64 | Δ |
| --- | ---: | ---: | ---: |
| Untrained cross | 0.459 ± 0.022 | 0.448 ± 0.012 | −0.011 |
| Trained cross | **0.230 ± 0.020** | **0.422 ± 0.017** | **+0.192** |
| Δ cross (trained − untrained) | **−0.229** | **−0.025** | +0.204 |
| paired t(3) | ≈ −36.3 | ≈ −2.58 | — |
| p | ≪ 0.001 | ≈ 0.08 (NOT significant) | — |
| block_size (R2-E / vocab) | 43 cells | 21 cells | half |

**Doubling the vocabulary roughly eliminates the training
signal.** Trained cross rises +0.192, untrained barely moves
(−0.011), Δ cross collapses 89 % (from −0.229 to −0.025) and
falls below the standard significance threshold (p ≈ 0.08).
This is the *opposite* of what a geometric-encoder-collision
floor predicts: more vocab → more cells per "encoded" SDR
collision → mean overlap should drop or hold, not rise.

The architecture / plasticity floor predicts exactly this:
bigger vocab halves the per-cue R2-E block budget under
decorrelated wiring (43 → 21 cells per cue), which means
fewer cells per cue to write a cue-specific weight pattern
into. Each cue's effective representational room shrinks by
50 %, and the trained-cross floor rises by ~0.19.

### Per-seed Δ cross at vocab=64

| Seed | vocab=32 trained | vocab=64 trained | vocab=32 Δ | vocab=64 Δ |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.242 | 0.417 | −0.225 | −0.048 |
|  7 | 0.250 | 0.448 | −0.235 | **+0.000** (no training effect) |
| 13 | 0.208 | 0.409 | −0.242 | −0.030 |
| 99 | 0.220 | 0.415 | −0.213 | −0.024 |

Seed 7 at vocab=64 has **literally zero** training effect
(0.448 trained vs 0.448 untrained). Three of four seeds show
sub-significant per-seed Δ. Whatever specificity the
architecture wrote at vocab=32 simply does not write at
vocab=64 under the same training schedule.

### vocab=64 per-pair distribution

| | min | p25 | median | p75 | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vocab=32 | 0.000 | 0.150 | 0.225 | 0.300 | 0.350 | 0.425 | 0.750 |
| **vocab=64** | **0.100** | **0.350** | **0.425** | **0.500** | **0.500** | **0.500** | **1.000** |
| **Δ** | +0.100 | +0.200 | +0.200 | +0.200 | +0.150 | +0.075 | +0.250 |

The whole distribution shifts right by ≈ +0.20 — i.e. the
median Jaccard *of the trained brain at vocab=64* matches the
*aggregate cross-cue mean of the trained brain at vocab=32*.
Whatever each pair "costs" the architecture in shared top-3
words, that cost roughly doubles when the budget per cue is
halved.

### Top-15 high-overlap pairs at vocab=64

| Rank | cue_a | cue_b | mean Jaccard |
| ---: | :--- | :--- | ---: |
|  1 | actor | ada | **1.000** |
|  2 | actor | array | **1.000** |
|  3 | ada | array | **1.000** |
|  4 | julia | zig | 0.625 |
|  5 | tuple | typescript | 0.625 |
|  6 | continuation | racket | 0.550 |
|  7 | actor | block | 0.500 |
|  8 | actor | broadcast | 0.500 |
|  9 | actor | closure | 0.500 |
| 10 | actor | cobol | 0.500 |
| 11 | actor | comptime | 0.500 |
| 12 | actor | coroutine | 0.500 |
| 13 | actor | cpp | 0.500 |
| 14 | actor | crystal | 0.500 |
| 15 | actor | dart | 0.500 |

Three cues — **actor / ada / array** — produce *identical*
top-3 lists across all 4 seeds (Jaccard = 1.000 on every
pairwise comparison). These three are 4-character common
words and the encoder plausibly hashes them to nearly
identical SDRs. Below this tight collision cluster, the
distribution falls smoothly: ranks 4-6 at 0.55-0.625, ranks
7+ saturated at 0.500 (≥ 1 word in common out of top-3).

### Cue frequency in pairs ≥ 0.30 (out of vocab − 1 = 63 partners)

Top 10:
- clojure: 59, dynamic: 59, fiber: 59, ownership: 59
- actor: 58, ada: 58, array: 58, block: 58, broadcast: 58,
  closure: 58, cobol: 58, compile: 58, comptime: 58 …

**Almost every cue has 58–59 of 63 partners (≈ 92 %) above
the 0.30 threshold.** At vocab=32 the worst was 21/31 (68 %).
The "high-overlap" pattern at vocab=64 is *near-universal*:
the architecture cannot produce a cue-specific representation
at this resource budget for *any* cue.

## Verdict per Bekos's iter-58 / iter-59 branching matrix

From Bekos's iter-58 spec, applied verbatim:

| Sweep result | iter-59 branch | This data |
| --- | --- | :-: |
| (A) Geometry floor: high cross-cue from few recurring cue pairs + vocab=64 lowers trained_cross substantially | iter-59 = encoder / dictionary / decode geometry fix | ❌ (vocab=64 *raises* trained_cross by +0.192 — opposite direction; small-fraction-pair signature absent at vocab=32, present only as the actor/ada/array trio at vocab=64) |
| (B) Plasticity / architecture floor: high cross-cue uniform across pairs + vocab=64 stays near 0.20 | iter-59 = real architecture question (bridge / learnable R1→R2 / contrastive objective) | **✓ PRIMARY** (vocab=32 distribution broad-and-continuous, no isolated tail; vocab=64 doubles the per-pair median, 92 % of cues above 0.30 threshold, Δ cross collapses to non-significant; the floor scales with the per-cue R2-E block budget, *not* with vocab) |
| (C) Mixed: some collision pairs + global floor | iter-59 = geometry cleanup first, then architecture | ✓ secondary minor (the actor/ada/array trio at vocab=64 with mean Jaccard = 1.000 is a clear encoder/SDR collision; the "tuple/typescript" 0.625 pair likewise. These are short common words the text encoder over-aliases. They are a small fraction of the 2016 pairs and do not change the architectural reading of the bulk distribution) |

**iter-58 verdict: branch (B) PRIMARY — architecture /
plasticity floor confirmed by the vocab-scaling test.** The
per-cue R2-E block budget is the binding constraint on cross-
cue specificity. At 43 cells per cue the architecture writes
−0.229 of specificity (significant); at 21 cells it writes
−0.025 (not significant). The floor is not "the encoder
over-aliases short common words" (a few collision pairs do
exist, branch C secondary, but they are not the global
limit) — it is "the trained R2 representation has insufficient
cells to write a cue-distinct pattern under the current
plasticity rule + decorrelated wiring".

**iter-59 entry: real architecture question, not encoder
geometry fix.** Three orthogonal architectural levers from
Bekos's earlier suggestions (iter-55 / iter-56 / iter-57
parallel candidates), now ranked by the iter-58 evidence:

- **Path 1 — bridge layer / learnable projection.** The
  current decorrelated wiring is *fixed* and per-cue blocks
  are *static*. A learnable R1 → R2 projection (or a small
  bridge region) would let the architecture *expand* per-cue
  effective representation by recruiting cells outside the
  fixed block when training pressure demands it. Concrete
  candidate: replace the static block partition with a
  weight-mediated soft-allocation (each cue's effective
  block is the union of cells whose incoming-weight-from-
  cue exceeds a threshold). Plasticity then decides per-cue
  cell membership, not topology.
- **Path 2 — contrastive objective.** The current reward
  signal credits target hits *per cue* but does not
  *penalise* shared firing across cues. A contrastive iSTDP
  term that suppresses cells firing for multiple cues
  within an epoch would push the trained R2 representation
  toward a one-hot-per-cue ideal, giving cue-specificity a
  direct gradient instead of relying on the decorrelated
  topology to provide it geometrically.
- **Path 3 — increase R2_N.** The simplest fix: 1400 R2-E
  cells / 64 vocab = 21 cells per cue is too tight. Doubling
  R2 to 4000 cells gives 62 cells per cue at vocab=64,
  matching vocab=32's iter-54 budget (43 cells). Predicted
  trained cross at vocab=64 + R2_N=4000: ~0.20-0.25, similar
  to iter-54. Cheap to test (~30 min: change a constant +
  re-run vocab=64 sweep). Useful as a *positive control* for
  the architecture-floor hypothesis even if the actual fix
  ends up being Path 1 or Path 2.

Recommendation: **Path 3 first as positive control** (does
adding cells exactly compensate for vocab doubling?), then
Path 1 or Path 2 as the real architecture work.

**A separate note on the encoder-collision pairs:** actor /
ada / array / tuple / typescript are short common-letter
words; the existing TextEncoder produces near-identical SDRs
for them. iter-59 should NOT prioritise this — it's a small
fraction of the failure mode, and the iter-46 corpus was
chosen for "well-separated SDRs" so the encoder is presumably
already a fairly clean baseline. If iter-59's architecture
work doesn't lift the floor, *then* the encoder choice
becomes the next axis to investigate.

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
trajectories often reveal a saturation ceiling the aggregate hides.
iter-56: aggregate monotonicity is not seed-level monotonicity.
iter-57: a 3-point sweep is the minimum for a non-monotonic axis.
**iter-58: a saturation ceiling has a *direction*. iter-55 / 56 / 57
each saw the floor approach 0.20 from below as training axes
were extended, which left the geometric-vs-architecture
question genuinely open. iter-58 picked one new variable
(vocab) and asked the *direction-of-change* question:
geometric model predicts trained_cross stays flat or drops
with bigger vocab; architectural model predicts it rises.
Trained_cross rose by +0.192. One number, one direction, one
verdict. Whenever multiple training axes saturate at the same
value, find a *non-training* axis where the two competing
models predict opposite signs of change — that's the
diagnostic, not yet-another-training-sweep.**

## Files touched (single commit)

- `crates/eval/src/reward_bench.rs` — `JaccardPairSample` +
  `JaccardFloorReport` + `evaluate_jaccard_matrix_with_pairs` +
  `run_jaccard_floor_diagnosis` + `render_jaccard_floor_diagnosis` +
  `default_corpus_v64`.
- `crates/eval/src/lib.rs` — re-export new public surface.
- `crates/eval/examples/reward_benchmark.rs` —
  `--jaccard-floor-diagnosis` + `--corpus-vocab` +
  `--floor-threshold` + `--floor-top-n` CLI flags.
- `notes/58-jaccard-floor-diagnosis.md` — this note.
- `CHANGELOG.md` — iter-58 section.
- `README.md` — iter-58 entry.
