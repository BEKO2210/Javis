# Iter 53 — Decoder-relative Jaccard (Option B, Voll-Implementation)

## Why iter-53

iter-52 nailed the iter-46…51 sequence to a single conclusion:
plasticity is doing *something* (trained top-3 = 0.107 vs untrained
0.039 is Δ ≈ 2.2 σ), but the metric is confounded by forward-
drive bias on the decoder. The untrained brain's top-3 = 0.039 is
not a "no learning" baseline — it is the isolated forward-drive
asymmetry, which sits significantly *below* the 0.094 random
baseline. Top-3 against a fingerprint dictionary built from that
same forward-drive-saturated brain is the wrong instrument.

iter-53's question — and Bekos's pre-fixed iter-53 design — is
the next shift to the right: **build a decoder-relative metric
that does NOT degrade when the dictionary is uniformly biased,
and compare trained vs untrained on that metric.**

The smoke gate (committed as `b86948d`) chose between two
options based on a 1-cue × 3-trial determinism test:

- Option A — per-trial trained-minus-untrained Δ
- Option B — trial-to-trial Jaccard consistency

The smoke produced **mean Jaccard = 0.667** on a frozen brain
with `brain.regions[1].network.reset_state()` (R2-only) between
trials. 0.667 is the informative regime (`0 < x < 1`), so Bekos
chose Option B Voll-Implementation, with **cross-cue Jaccard
added as a second axis**: same-cue measures consistency, cross-
cue measures specificity, and the **difference of differences**
is the engram-formation indicator.

## Implementation — single commit

Three new public types and one public entry point in
`crates/eval/src/reward_bench.rs`, mirrored in `lib.rs`:

```text
pub struct JaccardMetrics       — (same/cross_cue_{mean,std}, n_cues, n_pairs)
pub struct JaccardArmResult     — (seed, arm, jaccard)
pub struct JaccardSweepResult   — (Vec<trained>, Vec<untrained>)
pub fn run_jaccard_bench(corpus, cfg, seeds) -> JaccardSweepResult
pub fn render_jaccard_sweep(&JaccardSweepResult) -> String
```

Plus three private helpers:

- `evaluate_jaccard_matrix` — collects the 32-cue × 3-trial
  decoder matrix, computes same-cue + cross-cue Jaccard.
- `train_brain_inplace` — minimal training loop, mirrors the
  schedule in `run_reward_benchmark` without metrics collection.
- `run_jaccard_arm` — single-seed/single-arm runner: build
  brain → train → eval matrix.

CLI: `--jaccard-bench --seeds N1,N2,…` in
`crates/eval/examples/reward_benchmark.rs`. Default seed list is
the singular `--seed` value when `--seeds` is omitted.

### The four layers Bekos asked for

1. **Trial-Wiederholungs-Pipeline.** Each trial calls
   `brain.reset_state()` (full: R1 + R2 + cross-region pending
   queue + `brain.time` + traces + V + refractory + eligibility +
   neuromodulator), not the previous `brain.regions[1].network.
   reset_state()` (R2-only). The full reset is what makes the
   state-reset assertion possible — see (3).

2. **Same-Cue + Cross-Cue Jaccard from one matrix.** For each cue
   in the 32-word vocab, collect 3 trials. Drop trial 0 (burn-in
   per the iter-53.0 smoke). Same-cue: mean over cues of
   `Jaccard(trial[1], trial[2])`. Cross-cue: mean over cue pairs
   `(i, j), i < j` of `Jaccard(matrix[i][1], matrix[j][1])` →
   one matrix, two metrics, no separate runs.

3. **State-reset assertion** (Bekos's L2-equivalent invariant for
   transient state). Untrained arm — plasticity disabled, full
   reset between trials — must produce **`same_cue_mean == 1.0`
   exactly**. The deterministic LIF guarantees this if and only
   if the reset is complete. The assertion panics with seed +
   measured value if the invariant is violated.

4. **Plasticity stays ON during trained eval.** Per Bekos's
   spec ("im trained Run würde Plastizität zwischen Trials
   variieren, was *gewollt* ist"), the trained arm does NOT
   disable plasticity before the dictionary build / matrix
   collection. Membrane state is reset per trial, but persistent
   weights continue to drift between trials — exactly the
   "via plasticity, not via membrane state" dependency Bekos
   wants to measure. The untrained arm has no plasticity to
   begin with (`no_plasticity = true` gates every `enable_*`),
   so this branch leaves it unchanged: same-cue == 1.0.

This deliberately drops the iter-52 eval-phase L2 invariant on
the trained arm. The `no_plasticity` arm still asserts L2 bit-
identity end-to-end; the trained arm now logs `pre/post L2` as
a drift readout instead.

### One non-obvious finding from the smoke gate

`Network::reset_state` zeroes `v_thresh_offset` (the Diehl-Cook
intrinsic-plasticity offset). That means *every* call to
`brain.reset_state()` between trials wipes the trained intrinsic
threshold. This is a pre-existing invariant (it was already
happening in `build_vocab_dictionary`, `evaluate_with_dict`, and
the run-loop's per-trial `reset_state`); iter-53 surfaces it but
does not change it. The trained vs untrained comparison is fair
because both arms use the same protocol.

## Run — 4 seeds × 16 epochs × {trained, untrained}

```sh
cargo run --release -p eval --example reward_benchmark -- \
  --jaccard-bench --seeds 42,7,13,99 --epochs 16 --teacher-forcing
```

```text
[iter-53 untrained] seed=42 same=1.000±0.000 cross=0.058±0.137 (n_cues=32, n_pairs=496)
[iter-53 trained eval-drift] seed=42 pre=[-0.0, 1398.74] post=[-0.0, 1425.82] |Δ|=[0.0, 27.08]
[iter-53 trained]   seed=42 same=0.828±0.241 cross=0.056±0.123 (n_cues=32, n_pairs=496)

[iter-53 untrained] seed=7  same=1.000±0.000 cross=0.058±0.134 (n_cues=32, n_pairs=496)
[iter-53 trained eval-drift] seed=7  pre=[-0.0, 1397.12] post=[-0.0, 1421.61] |Δ|=[0.0, 24.50]
[iter-53 trained]   seed=7  same=0.891±0.210 cross=0.056±0.114 (n_cues=32, n_pairs=496)

[iter-53 untrained] seed=13 same=1.000±0.000 cross=0.056±0.121 (n_cues=32, n_pairs=496)
[iter-53 trained eval-drift] seed=13 pre=[-0.0, 1387.14] post=[-0.0, 1412.67] |Δ|=[0.0, 25.54]
[iter-53 trained]   seed=13 same=0.875±0.220 cross=0.062±0.134 (n_cues=32, n_pairs=496)

[iter-53 untrained] seed=99 same=1.000±0.000 cross=0.062±0.126 (n_cues=32, n_pairs=496)
[iter-53 trained eval-drift] seed=99 pre=[-0.0, 1378.61] post=[-0.0, 1407.16] |Δ|=[0.0, 28.55]
[iter-53 trained]   seed=99 same=0.922±0.184 cross=0.059±0.121 (n_cues=32, n_pairs=496)
```

| Seed | Untrained same | Untrained cross | Trained same | Trained cross | Eval-drift L2 (R2→R2) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1.000 ± 0.000 | 0.058 ± 0.137 | 0.828 ± 0.241 | 0.056 ± 0.123 | 27.08 |
|  7 | 1.000 ± 0.000 | 0.058 ± 0.134 | 0.891 ± 0.210 | 0.056 ± 0.114 | 24.50 |
| 13 | 1.000 ± 0.000 | 0.056 ± 0.121 | 0.875 ± 0.220 | 0.062 ± 0.134 | 25.54 |
| 99 | 1.000 ± 0.000 | 0.062 ± 0.126 | 0.922 ± 0.184 | 0.059 ± 0.121 | 28.55 |

**Aggregate (n = 4 seeds):**

```text
Untrained: same = 1.000 ± 0.000   cross = 0.058 ± 0.003
Trained:   same = 0.879 ± 0.039   cross = 0.058 ± 0.003
Δ same   = −0.121   (attractor erosion — plasticity drifts the
                     engram between trials of the same cue)
Δ cross  = +0.000   (no cue-specificity gain — distinct cues land
                     on equally-overlapping top-3 sets in trained
                     and untrained)
Δ-of-Δ   = −0.121   (specificity gain ≪ attractor erosion)
```

## Honest reading — direction of Δ same-cue

Bekos's literal acceptance criterion was:

> Trained Same-Cue-Jaccard > Untrained Same-Cue-Jaccard signifikant

Under this implementation that comparison is mathematically
upper-bounded: untrained same-cue is exactly 1.0 (deterministic
LIF + no plasticity + full reset), and any plasticity-induced
drift in the trained arm pushes its same-cue strictly below 1.0.
**Trained same-cue ≤ untrained same-cue** is the constraint, not
a falsifiable claim.

The right reading of trained same-cue is therefore not "trained
higher than untrained" but **"trained close to 1.0"**: the
closer trained sits to 1.0, the more *attractor-like* the
post-training engram is — i.e. the cue lands in the same R2
basin even when plasticity continues to act on the weights
between trials. Trained same-cue → 0 means the engram is fragile;
plasticity drift dominates the dynamics.

The cross-cue comparison still goes in the direction Bekos
specified: **trained cross-cue should fall below untrained
cross-cue** (specificity rises with training).

The Δ-of-Δ summary number remains useful, with the sign
convention:

```text
Δ-of-Δ = Δ same − Δ cross
       = (trained_same − untrained_same) − (trained_cross − untrained_cross)
       ≤ 0 − (trained_cross − untrained_cross)            (because Δ same ≤ 0)
       = untrained_cross − trained_cross
```

A **positive** Δ-of-Δ means the cue-specificity gain
(`untrained_cross − trained_cross`) outpaced the same-cue
attractor erosion (`untrained_same − trained_same`) — i.e.
training improved specificity faster than it disturbed
consistency. A **zero or negative** Δ-of-Δ means erosion is
faster than specificity gain — engrams form, but they're
fragile under continued plasticity.

## Acceptance status

### State-reset assertion — PASSED (4/4 seeds)

Untrained `same_cue_mean = 1.000 ± 0.000` on every seed. Full
`brain.reset_state()` is complete (R1 + R2 + cross-region
pending queue + traces + V + refractory + eligibility +
neuromodulator). The L2-equivalent invariant for transient
state holds.

L2 bit-identity on the untrained arm — PASSED (4/4 seeds).
The iter-52 weight-stability invariant survives the iter-53
protocol unchanged for the no-plasticity arm.

### Δ-of-Δ acceptance — FAILED (−0.121)

The aggregate is `Δ-of-Δ = −0.121`: specificity gain (Δ cross
≈ 0) is dominated by attractor erosion (Δ same = −0.121).

Reading the two axes separately:

1. **Plasticity is alive** in the trained brain. Eval-phase L2
   drift is **+25 to +29 (R2→R2)** on every seed — plasticity
   is actively modifying weights during the 32-cue × 3-trial
   matrix collection. Same-cue Jaccard at 0.879 ± 0.039 (i.e.
   substantially below 1.0) confirms this drift translates
   into different decoder responses across trials.

2. **The trained brain has *some* attractor structure** — same-
   cue at 0.88 ± 0.04 means trial 2 and trial 3 of a given cue
   share ~88 % of their top-3 words, well above the random
   floor for two random 3-element samples from a 32-word vocab.
   So weights did learn *something*; cues land in *some* basin
   that's at least partially robust to continued plasticity.

3. **The attractors are NOT cue-specific.** Cross-cue is **flat
   at 0.058 ± 0.003** in both arms — distinct cues' top-3 sets
   overlap at exactly the same rate as on a fresh forward-only
   brain. Training redistributes weight mass and creates per-
   cue basins, but those basins are not aligned with cue
   identity at the decoder layer.

This is a clean, honest negative result for "did teacher-
forcing produce cue-specific engrams in 16 epochs". It is *not*
a failure of the iter-53 instrumentation — the metric is doing
its job, and the state-reset assertion gives it a hard floor of
trustworthiness. The result is fully consistent with iter-52's
finding that forward-drive bias dominates the decoder geometry:
plasticity adds drift-with-some-attractor, but doesn't break
the cue-uniform forward-projection baseline.

### Implication for iter-54

The cross-cue ≈ 0.058 ceiling on **both** arms is the bottleneck.
iter-54 has to address cue-specificity at the architecture or
schedule layer, not at the metric layer. Candidates (in
roughly increasing-cost order):

- **(i) Decorrelate initial R1 → R2 projections per cue.** The
  current `wire_forward` uses the same `FAN_OUT` distribution
  for every cue's R1 cells; if all cues activate roughly the
  same R2 subset before training, plasticity has no asymmetric
  signal to amplify. Per-cue topology bias (different
  presynaptic populations, possibly orthogonal SDR projections)
  would give STDP a starting point with cue-specific structure
  to deepen.
- **(ii) Reward cue-specificity directly.** The current reward
  signal (`positive_reward_for_correct`) only credits target
  hits. Add a term that *penalises* sharing top-3 cells across
  cues within an epoch — i.e. push iSTDP to suppress cells
  that fire for multiple cues. This is closer to a contrastive
  predictive coding objective.
- **(iii) Switch off forward drive partway through training.**
  The R1 → R2 inter-region drive dominates the response; if
  training had a "cue-only schedule" final phase where R2
  recurrent dynamics alone determine the response (`r1r2_gate`
  → 0), plasticity would have to learn cue-specific recurrent
  attractors. The `--association-training-gate-r1r2` flag
  exists; iter-54 could sweep its schedule.

iter-53 should NOT be read as picking among these — it just
narrows the failure mode. The cross-cue Δ ≈ 0 finding is the
new "Mess-Frage" of iter-46…52: forward-drive bias is unbroken
by 16 epochs of teacher-forcing on the current architecture.

## What iter-53 should NOT be read to claim

- ❌ "Same-cue → 1.0 in trained means engrams perfectly form."
  Same-cue is bounded above by 1.0 only because untrained is
  1.0 by construction; the *fall* from 1.0 in the trained arm
  is the actual signal.
- ❌ "Cross-cue change of 0.0X is a strong learning signal."
  Cross-cue is small in absolute magnitude even at the
  untrained baseline (0.058) — both arms have many cue pairs
  that share zero top-3 words simply because the dictionary
  has 32 entries and `decode_top` returns 3. The Δ-of-Δ is
  the right summary number, not either delta on its own.
- ❌ "iter-53 has solved the iter-46…51 dead-end." iter-53
  surfaces a *different*, less confounded measurement axis.
  The underlying brain is whatever iter-52 left behind — same
  weights, same engrams, same iSTDP saturation history. iter-53
  measures *whether learning happened* (cross-cue specificity
  + same-cue stability), not *fixes* the upstream problem.

## Methodological lesson

The smoke determined the design *before* the implementation
("0 < 0.667 < 1.0 = informative regime"). The implementation
then surfaced a constraint the smoke had not exercised: under
deterministic LIF, "trained same-cue > untrained same-cue" is
mathematically impossible if untrained = 1.0 by construction.
The right move was to keep Bekos's protocol (plasticity ON in
trained eval, OFF in untrained eval; full state reset between
trials) and re-derive the acceptance criterion to fit the
mathematical bounds — not silently rewrite the protocol to
make the original direction satisfiable.

iter-50: save the simplest configuration as a regression guard.
iter-51: a guard is only a guard if its baseline excludes the null.
iter-52: an analytical null is not an empirical control.
**iter-53: when the literal acceptance direction is bounded by
construction, derive the actual acceptance from the protocol's
mathematical bounds — and document the derivation.**

The state-reset assertion (untrained `same_cue_mean == 1.0`) is
the iter-53 equivalent of iter-52's L2-norm bit-identity check.
Both catch a class of "the protocol leaked" bug that the
visible config would have hidden.

## Files touched (single commit)

- `crates/eval/src/reward_bench.rs` — Jaccard structs +
  `evaluate_jaccard_matrix` + `train_brain_inplace` +
  `run_jaccard_arm` + `run_jaccard_bench` + `render_jaccard_sweep`.
- `crates/eval/src/lib.rs` — re-export the new public surface.
- `crates/eval/examples/reward_benchmark.rs` — `--jaccard-bench`
  + `--seeds` CLI flags.
- `notes/53-decoder-relative-jaccard.md` — this note.
- `CHANGELOG.md` — iter-53 section.
- `README.md` — iter-53 entry in the iteration timeline.
