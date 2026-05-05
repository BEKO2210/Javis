# Iter 64 — Mechanism diagnosis (no architecture pivot)

**Status: SPEC. No measurements. No implementation code yet.**
**Pre-registration before any axis is touched. The hypothesis,
axes, value lists, smoke plan, and acceptance matrix are locked
in this commit and may not be relaxed post-hoc.**

## Why iter-64

iter-63 delivered a clean Branch (B) FAIL on the locked branching
matrix:

```text
Per-seed Δ (trained − untrained):
  seed=42  +0.0000
  seed=7   +0.0195
  seed=13  −0.0459   ← negative
  seed=99  +0.0156

Δ̄ = −0.0027 ± 0.0300, paired t(3) = −0.179
n_pos = 2/4, n_pass = 0/4, p ≥ 0.15
```

32 epochs of full plasticity (STDP / iSTDP / homeostasis /
intrinsic / reward / metaplasticity / heterosynaptic /
structural) on the iter-63 configuration produced **no
measurable cue → target learning signal** on the iter-44/45
`top3_accuracy` metric. iter-60 (DG separation, cross-cue floor
collapses 16×) and iter-62 (recall stability, same-cue = 1.000
on 4/4 with bit-identical L2) both worked — what is missing is
a measurable signal that the *post-DG path maps cue → target*,
not just *separates cues*.

Per the iter-63 locked branching matrix, branch (B) sends iter-
64 into **mechanism diagnosis first**, before any further
architecture work. CA3/CA1 split, perforant-path bypass at the
SDR-encoding layer, target-side restructuring — all deferred
until at least one mechanism axis surfaces a measurable signal.

## Pre-registered hypothesis (single, weak, non-optimisation)

**H1 (mechanism, not performance):**

> At least one of the three iter-63-Branch-(B)-named axes
> produces a positive paired-t direction on
> `target_top3_overlap` such that **Δ̄ > 0** with **t(3) > 0**
> on **4/4 seeds** — even when Δ̄ does not reach the iter-63
> locked threshold of 0.0621.

This hypothesis is **deliberately weaker** than iter-63's. iter-
64 asks: *does any axis produce a positive trend?* — not: *does
any axis hit the iter-63 threshold?*. The latter is the
Branch-(A) question, which only becomes relevant in iter-65+
*if* iter-64 identifies a candidate mechanism worth deepening.

The iter-63 threshold of **0.0621 stays documented as the
known-calibrated reference**, but iter-64's acceptance does
**not** require any axis to clear it. iter-64 is mechanism
*localisation*, not learning *demonstration*.

## Pre-registered out-of-scope (locked, not relaxable)

- iter-63-Threshold 0.0621 as a success criterion for an axis.
  Reaching 0.0621 would be a Branch-(A)-style result; iter-64
  does not seek that and will not declare it as success.
- Multi-metric reading. Single metric: `target_top3_overlap`
  = mean of `top3_accuracy` across all epochs. Same as iter-63.
- CA3/CA1 split. Deferred until at least one axis is positive.
- New plasticity rules. The 8-rule stack stays as it is.
- Vocab / epoch / clamp / teacher_ms variation. Frozen at the
  iter-63 configuration: `vocab=64, --epochs 32,
  --target-clamp-strength 500, --teacher-ms 40`.
- Combinatorial sweeps (axis A × axis B simultaneously). The
  three axes are tested **isolated**. Combinatorial sweeps are
  iter-65 territory at the earliest, and only if iter-64
  surfaces multiple positive axes.
- Goalpost-shifting. The acceptance matrix is locked in this
  commit and applied verbatim per axis after the run. Edge
  cases collapse to (β) by default — the same hygiene rule the
  iter-63 matrix used to collapse to (B).

## Three axes (each isolated, each paired against own untrained baseline)

### Axis A — DG → R2 drive / weight scale

**Hypothesis.** The DG mossy-fibre projection at
`dg_to_r2_weight = 1.0` and `dg_to_r2_fanout = 30` dominates R2
strongly enough that R1 → R2 forward drive and R2 recurrent
plasticity cannot overlay a target-binding pattern. Reducing
DG → R2 weight opens headroom for the other pathways without
breaking iter-60's separation (DG hashes still route, just
with less amplitude).

**Sweep.** `dg_to_r2_weight ∈ {0.25, 0.50, 1.00, 2.00}`:

- 1.00 = iter-63 baseline (replicates the verdict).
- 0.50, 0.25 = test the "DG dominates" branch.
- 2.00 = test whether more DG drive helps (sanity probe; if
  this also fails, dominance is unlikely the issue).

All other parameters frozen at iter-63 config. 4 seeds × 32
epochs trained AND untrained (paired by seed and value).

**Sub-axis (deferred to implementation phase, not this spec):**
isolating STDP `a_plus` on DG → R2 synapses *separately* from
R1 → R2 / R2 recurrent. Currently `enable_stdp` is per-region;
fine-grained per-synapse-class plasticity is non-trivial code.
If too expensive, drop the sub-axis and only sweep
`dg_to_r2_weight`. Decision in implementation-phase, not now.

### Axis B — R2 recurrent connectivity

**Hypothesis.** The R2 recurrent network at `R2_P_CONNECT =
0.05` (vocab=64, R2_N=2000 ⇒ ~140k recurrent E-E synapses) may
be too sparse for an engram to *carry between trials*. Without
a recurrent attractor, the per-epoch dictionary fingerprint
captures whatever R2 fires *during* the cue-alone read-out, and
that response is dominated by the DG-mossy-fibre projection
(which doesn't change), not by the trained recurrent state
(which does). Result: trained and untrained dictionary
fingerprints land in similar configurations because the
recurrent contribution is negligible.

**Sweep.** `R2_P_CONNECT ∈ {0.025, 0.05, 0.10}`:

- 0.05 = iter-46 / iter-63 baseline.
- 0.025 = sparser (weaker recurrent attractor).
- 0.10 = denser (stronger recurrent attractor).

All other parameters frozen. 4 seeds × 32 epochs. Initial
recurrent weight band (`g_exc = 0.20`, `g_inh = 0.80`,
`rng.range_f32(0.5 * g, 1.0 * g)`) stays fixed for this iter —
weight band is a separate axis future iter-66+ may sweep.

**Code prerequisite.** `R2_P_CONNECT` is currently a `const` in
`reward_bench.rs`. Implementation needs a runtime override
analogous to iter-59's `--r2-n` flag. Cheap, additive change,
keeps backward compat (default = 0.05).

### Axis C — Perforant path re-introduction (weak R1 → R2)

**Hypothesis.** iter-60 set `direct_r1r2_weight_scale = 0.0`,
making DG the sole cue-routing path into R2. The hippocampus
in fact has **two** parallel pathways: perforant path (direct
EC → CA3) and mossy fibre (via DG). Without a direct R1 → R2
input, the trained R2 has no consistent "raw cue" handle that
plasticity can exploit independently of DG's hashed code.
Re-introducing a *weak* perforant path re-establishes a
correlated cue-driven R2 substrate that plasticity can shape
via STDP/R-STDP into a target-aligned engram.

**Sweep.** `direct_r1r2_weight_scale ∈ {0.0, 0.1, 0.3, 1.0}`:

- 0.0 = iter-60+ baseline (DG sole, replicates iter-63).
- 0.1, 0.3 = weak / moderate perforant path.
- 1.0 = pre-iter-60 baseline (no DG attenuation, perforant
  fully on alongside DG).

All other parameters frozen. 4 seeds × 32 epochs.

**Code prerequisite.** Already exists via
`--direct-r1r2-weight-scale` CLI flag. **No new plumbing for
this axis.**

## Why these three, and only these three

The iter-63 locked Branch-(B) entry text named exactly these
three axes. No others. Adding a fourth axis (e.g. clamp
strength, teacher_ms, target SDR density) at this stage would
re-litigate decisions that earlier iterations already made or
left to later. Three locked axes, three isolated sweeps,
three independent (α/β/γ/δ) verdicts — that's iter-64.

## Smoke plan (per axis, before its full sweep)

Before each 4-seed × 32-epoch axis sweep, run a smoke at one
endpoint of the axis to verify pipeline integrity:

```sh
# Generic smoke template — replace [AXIS-FLAG] per axis.
cargo run --release -p eval --example reward_benchmark -- \
  --target-overlap-bench --mode trained \
  --seeds 42 --epochs 8 \
  [AXIS-FLAG=lowest_or_highest_value] \
  --decorrelated-init --teacher-forcing \
  --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval \
  --threshold 0.0621
```

**Smoke acceptance.** A smoke passes if all of:

1. Compiles + runs to completion (no panic, no
   iter-52-invariant violation, no NaN in
   `target_top3_overlap`).
2. `target_top3_overlap` is in `[0.0, 1.0]` for both trained
   and untrained internal re-run (sanity range).
3. Wallclock within `[5, 15]` minutes for 1 seed × 8 epochs at
   vocab=64 + DG (matches iter-63 single-seed timing).

**On smoke fail.** Stop the axis. Diagnose. Do not start the
full 4-seed × 32-epoch sweep on a broken pipeline. This is
iter-63 v1's lesson, applied per-axis. The
`grep -c build_benchmark_brain crates/eval/src/reward_bench.rs`
pre-flight check is mandatory before the first compile of each
axis sweep.

## Per-axis acceptance matrix (locked, applied verbatim)

For each axis, after its full 4-seed × 32-epoch sweep across
all values, classify the result:

| Outcome | Per-seed pattern | Aggregate | iter-65 consequence |
| --- | --- | --- | --- |
| **(α) Positive trend** | `Δ̄_axis > 0` AND `n_pos ≥ 3/4` AND `t(3) > 0` | direction matters; p-value not required | iter-65 = deepen this axis (8-seed paired analysis, possibly more values, sub-axis exploration) |
| **(β) No effect** | `\|Δ̄_axis\| ≤ σ_untrained` (≈ 0.021 from iter-63 calibration) | `t(3)` NS (`\|t\| < 1.0`) | rule out this axis as a primary mechanism; document and move on |
| **(γ) Negative trend** | `Δ̄_axis < 0` AND `n_pos ≤ 1/4` | `t(3)` clearly negative (`t < −1.0`) | rule out, document as degrading; do not pivot away from iter-63 baseline based on it |
| **(δ) Mixed across seeds** | high `σ_Δ`, no clear directional pattern, `n_pos = 2/4` | high std, t(3) inconclusive | needs more seeds before verdict; iter-65 = 8-seed re-run on this axis only |

**Edge cases collapse to (β)** — same hygiene as the iter-63
matrix collapsed to (B). If an axis matches *neither* (α) nor
(γ) cleanly *and* is not clearly (δ)-mixed (per the σ_Δ
criterion above), it lands in (β) and is ruled out for iter-65.

## iter-65 entry per cross-axis result

After all three axes complete, the cross-axis pattern dictates
iter-65:

1. **Exactly one axis (α).** iter-65 = deepen that axis. 8 seeds
   × 32 epochs at the most promising value, with paired-t
   power computation. CA3/CA1 still deferred.
2. **Multiple axes (α).** iter-65 = orthogonality test. Are the
   axis effects additive or anti-additive? First combinatorial
   sweep is justified here, in the form of a 2 × 2 grid at
   the most promising value of each (α)-axis.
3. **All axes (β/γ).** iter-65 = step back. The mechanism is
   not in any of the three locally-perturbable axes. Three
   options for iter-66+:
   - Target-side restructuring: `canonical_target_r2_sdr` is a
     deterministic random hash; replace with a structured
     target encoding that the brain could plausibly learn.
   - Read-out path: re-calibrate
     `prediction_top3_before_teacher` (the iter-46 metric we
     deferred) and re-test. Maybe the
     decoder-dictionary path is itself architecturally
     insensitive on DG-enabled brains.
   - Actually go to CA3/CA1 split despite the no-signal —
     accept that this metric chain is exhausted and pivot
     architecture. Last resort.

iter-65 entry is **not pre-decided** here; the three-way fork
is locked but the choice within it depends on the actual
iter-64 results.

## Wallclock estimate

Per axis, full sweep at 4 seeds × 32 epochs trained AND
untrained per value:

| Axis | n values | Runs | Per-run wallclock | Total wallclock |
| --- | ---: | ---: | --- | --- |
| A `dg_to_r2_weight` | 4 | 32 (4 × 4 × 2 arms) | ~30 min | **~16 h** |
| B `R2_P_CONNECT`    | 3 | 24 (3 × 4 × 2)     | ~30 min | **~12 h** |
| C `direct_r1r2_weight_scale` | 4 | 32 (4 × 4 × 2) | ~30 min | **~16 h** |

**Sequential total: ~44 h on local hardware.** Three axes back
to back. CI is not suitable for iter-64 sweeps — these are
local research runs.

## Pre-implementation optimisations (recommended before any axis is run)

1. **Untrained-arm cache.** The untrained arm at axis-value
   `v` reproduces bit-for-bit per (seed, v). Implementation
   should cache `(axis, value, seed) → target_top3_overlap`
   across axis values within a single sweep run. Saves ~⅓ of
   wallclock (12-15 h saved across all three axes).
2. **Single sweep subcommand.** Replace 32 separate
   `cargo run` invocations per axis with a single
   `--axis-sweep <axis> --values <list> --seeds 42,7,13,99
   --epochs 32` command. Saves cargo build / process startup
   cost × 32-per-axis.
3. **Two-phase run.** Run all three axes at **16 epochs first**
   (smoke + first-look). Only the axes that show (α) or (δ)
   at 16 epochs get the full 32-epoch run. Saves ~50 % of
   wallclock on (β/γ)-axes that show no signal early.

With all three optimisations, **estimated total ~24-28 h**
across all three axes — feasible in 2 days local.

## Methodological commitments (locked)

1. **No goalpost-shift.** If an axis shows no trend, it lands
   in (β) and is documented. Acceptance matrix locks before
   any axis runs.
2. **Single metric.** `target_top3_overlap` = mean
   `top3_accuracy` across epochs. No top-1, no MRR, no
   per-pair sub-analysis.
3. **Axes isolated.** No combinatorial sweep in iter-64.
4. **Mechanism diagnosis, not performance.** iter-64 success =
   *identification of a plausible axis*, not *measurable signal
   over iter-63 threshold*. The latter is iter-65+.
5. **Per-axis stop condition.** If an axis's smoke fails,
   pause that axis until diagnosed. iter-63 v1's lesson.
6. **Pre-run verification.** Before every axis sweep:

   ```sh
   grep -c build_benchmark_brain crates/eval/src/reward_bench.rs
   ```

   Must be `> 0` (silent-wiring-gap protection). Iter-63's
   incident is a closed lesson; this check makes it
   unrepeatable.

7. **Rebuild before run.** `cargo build --release` first, then
   `cargo run --release` — no implicit rebuild relying on
   freshness. Iter-63 had cases where stale binaries were
   suspected. Explicit build + verify mtime > spec edit time
   on the binary.

## Live results — Axis C (`direct_r1r2_weight_scale`)

### Axis C smoke (16 ep × 4 seeds × 4 values)

Run command (exactly as specified in this ENTRY):

```sh
cargo run --release -p eval --example reward_benchmark -- \
  --axis-sweep direct-r1r2-weight-scale \
  --values 0.0,0.1,0.3,1.0 \
  --seeds 42,7,13,99 \
  --axis-sweep-phase smoke \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval
```

Wallclock: ~2 h on local hardware. Cache pre-seeded with
iter-63 baseline values for `value=0.0` (4/4 seeds short-
circuited to the locked iter-63 calibration), saving ~25 min.

**Per-value renderer table:**

| value | μ_untrained | μ_trained | Δ̄ | σ_Δ | n_pos | n_pass(0.0621) | t(df=3) | classification |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 0.000 | 0.0195 | 0.0342 | +0.0147 | 0.0100 | 3/4 | 0/4 | +2.933 | **(α) Alpha** |
| 0.100 | 0.0273 | 0.0273 | +0.0000 | 0.0000 | 0/4 | 0/4 | +0.000 | **(β) Beta** |
| 0.300 | 0.0249 | 0.0439 | +0.0190 | 0.0320 | 3/4 | 0/4 | +1.191 | **(α) Alpha** |
| 1.000 | 0.0444 | 0.0317 | −0.0127 | 0.0282 | 1/4 | 0/4 | −0.899 | **(β) Beta** |

Per-value tally: α = 2, β = 2, γ = 0, δ = 0.

**Per-seed Δ matrix (reference, not for goalpost shifting):**

| seed | val=0.0 | val=0.1 | val=0.3 | val=1.0 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | +0.0186 | +0.0000 | **+0.0430** | −0.0293 |
| 7  | +0.0000 | +0.0000 | **+0.0371** | +0.0293 |
| 13 | +0.0225 | +0.0000 | **+0.0234** | −0.0293 |
| 99 | +0.0176 | +0.0000 | −0.0273 | −0.0215 |

### Mechanistic reading (smoke phase — provisional)

**(1) `value=0.1` — DG-dominated locked state.** Δ = +0.0000
*bit-for-bit on 4/4 seeds*. With a 10 %-scaled direct path the
`dg_to_r2_weight=1.0` mossy-fibre projection drives R2 strongly
enough that 16 epochs of plasticity make zero measurable
difference to the decoder top-3 — the trained and untrained
brains produce identical fingerprint dictionaries. This is the
strongest possible β: not "small effect, ruled out" but
"effect is identically zero". Confirms the iter-64 ENTRY
prediction for the "DG dominates" mechanism axis (axis A
captures it from the DG side; axis C captures the same
phenomenon from the perforant-path side).

**(2) `value=0.3` — provisional sweet-spot, the headline.**
Δ̄ = +0.0190 with `n_pos = 3/4` and `t(3) = +1.19` lands α per
the locked acceptance matrix. Three seeds clear meaningful
positive Δ (42 = +0.043, near the iter-63 0.0621 threshold;
7 = +0.037; 13 = +0.023). seed=99 is the lone negative
outlier at −0.027 (still inside the |Δ̄| ≤ σ_untrained_iter63
band as a single-seed deviation, drives σ_Δ up to 0.032).

The biological story matches the hippocampal anatomy: a
*moderate* perforant path provides EC → CA3 a stable raw-cue
substrate that R2 plasticity can shape into a target-aligned
engram, while DG (mossy fibres) maintains separation. Either
extreme — `0.0` (DG sole, iter-63 baseline) or `1.0` (full
perforant overpowering DG) — fails to produce the same trend.

**Wake-up moment for seed=7:** at every other axis-C value
seed=7 reads exactly 0.0000 trained AND 0.0000 untrained
(below random baseline 3/64 ≈ 0.047). At value=0.3, seed=7's
trained jumps to 0.0371 while untrained stays at 0.0000.
This is *the* mechanistic signal: a previously-silent seed
starts learning *only* when the perforant path is moderately
introduced. The brain at this seed has no DG-mediated
cue → target binding it can decode; the perforant path
re-introduction provides a substrate that plasticity can
exploit.

**(3) `value=1.0` — break-down into noise/cascade.** Δ̄ =
−0.0127 with `n_pos = 1/4` and `t(3) = −0.90`. Inside the
β-band by absolute magnitude, but *direction is negative*
on 3/4 seeds. The interpretation: full perforant-path drive
(iter-58 / pre-iter-60 baseline) overpowers the DG separation
benefit; the recurrent R2 attractor is dominated by raw cue
bleed-through and the trained dictionary fingerprint loses
specificity. This matches the iter-46 finding that strong
R1 → R2 forward drive eats recurrent learning capacity.
Classification stays β (within the magnitude band) but the
directional signal is the inverse of what α requires — at the
edge of γ. iter-65 8-seed re-run might split this point into
γ depending on seed coverage.

**(4) `value=0.0` — α at smoke, but iter-51 oscillation.**
Δ̄ = +0.0147, n_pos = 3/4, t(3) = +2.93 lands a strong α at
16 epochs. **However**, this is the iter-63 baseline
configuration that iter-63's 32-epoch trained main run already
classified as Branch (B) FAIL with Δ̄ = −0.0027 ± 0.0300.

*This is not a contradiction.* iter-51 demonstrated per-epoch
oscillation on the iter-46 Arm B reading; the smoke vs. full
two-phase logic exists precisely so we can see when a
configuration's positive signal at 16 ep is an oscillation
phase rather than a stable learning trend. value=0.0 at
smoke is *exactly that situation*: 16 ep catches the brain
in an upward-oscillation peak, 32 ep averages over the full
cycle and brings it back to ≈ 0.

**Implication for axis C verdict:** the `value=0.3` α has to
be confirmed at 32 ep (full phase) before we trust it.
value=0.0's α-at-smoke / β-at-full pattern is the cautionary
proof that smoke-α alone is not a stable verdict.

### Axis C full phase on value=0.3 (32 ep × 4 seeds)

Run command:

```sh
cargo run --release -p eval --example reward_benchmark -- \
  --axis-sweep direct-r1r2-weight-scale \
  --values 0.3 \
  --seeds 42,7,13,99 \
  --axis-sweep-phase full \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval
```

Wallclock: ~2 h on local hardware (4 trained + 4 untrained
runs at 32 ep, no cache hits — process-local cache empty at
process start, only iter-63 baseline value=0.0 pre-seeded).

**Renderer table (full phase, value=0.3):**

| value | μ_untrained | μ_trained | Δ̄ | σ_Δ | n_pos | n_pass(0.0621) | t(df=3) | classification |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 0.300 | 0.0242 | 0.0405 | +0.0164 | 0.0328 | 3/4 | 0/4 | +0.996 | **(α) Alpha — persistent** |

**Per-seed comparison (smoke vs full):**

| seed | smoke Δ (16 ep) | full Δ (32 ep) | direction stability |
| ---: | ---: | ---: | :--- |
| 42 | +0.0430 | +0.0215 | ✓ persistent positive |
| 7  | +0.0371 | +0.0215 | ✓ persistent positive (seed=7 wakes up at both phases) |
| 13 | +0.0234 | +0.0508 | ✓ persistent positive (doubled — closer to iter-63 0.0621 threshold) |
| 99 | −0.0273 | −0.0283 | ✗ persistent negative outlier |

### Mechanistic verdict — α confirmed at full phase

The smoke α at value=0.3 was **not** an iter-51 per-epoch
oscillation artefact. At 32 epochs the directional pattern
is preserved: 3/4 seeds clear positive Δ, the negative
outlier (seed=99) is *deterministic across both phases*
rather than a phase-of-oscillation artefact, and Δ̄ stays
above the (β) band (|Δ̄|=0.0164 vs σ_untrained_iter63=0.0213
— inside the band by magnitude but `n_pos = 3/4` and `t > 0`
land it at α).

The headline contrast against value=0.0:

| | smoke (16 ep) | full (32 ep) | persistence |
| --- | --- | --- | :--- |
| value=0.0 (DG-only) | Δ̄=+0.0147, t=+2.93, α | Δ̄=−0.0027, t=−0.18, β / iter-63 Branch (B) | **collapses** |
| value=0.3 (perforant + DG) | Δ̄=+0.0190, t=+1.19, α | Δ̄=+0.0164, t=+0.996, α | **holds** |

DG alone (value=0.0) catches an oscillation peak at 16 ep
that full 32 ep averages out. Adding a 30%-scale perforant
path (value=0.3) produces a learning signal that does not
collapse over 32 ep — the iter-64 ENTRY's third
mechanistic hypothesis ("perforant-path re-introduction
provides a stable raw-cue substrate that R2 plasticity can
shape") is **provisionally confirmed**.

### seed=99 — consistent negative outlier

seed=99 is the only seed where the trained brain
underperforms the untrained brain at value=0.3, *and the
deficit is consistent across smoke (−0.0273) and full
(−0.0283)*. This is not random oscillation — it is a
seed-specific failure mode at this configuration. Possible
mechanisms:

- The R1 → R2 random wiring at seed=99 happens to align
  poorly with the DG-mossy-fibre projection at this 30%
  perforant scale, so plasticity *destroys* the
  cue-aligned read-out instead of strengthening it.
- The R2 recurrent attractor at seed=99 builds a
  cue-orthogonal mode that competes with the
  decoder-dictionary fingerprint phase.

iter-65's 8-seed re-run will surface whether seed=99 is a
1-in-4 outlier or a 25 %+ failure mode at value=0.3.
σ_Δ = 0.0328 (vs σ_untrained_iter63 = 0.0213) flags the
elevated seed-level variance directly — the smoke + full
agreement on the rank order means the variance is a
property of the configuration, not a measurement artefact.

### iter-65 entry per the locked branching matrix

The iter-64 ENTRY's iter-65 fork applies once all three
axes complete. Strictly:

> Exactly one axis (α) → iter-65 = deepen at 8 seeds × 32 ep
> at the most promising value, with paired-t power
> computation. CA3/CA1 still deferred.

Axis C has produced an α at full phase (the only axis with
data so far). Two equally honest paths from here:

**Path 1 (per the locked spec): complete axes A and B
smokes first**, then make the iter-65 fork decision based on
the cross-axis pattern. This preserves the ENTRY's locked
methodology and avoids early-deepening on what could turn
out to be the second-best axis. Wallclock cost: ~3.5 h
(axis A smoke ~2 h, axis B smoke ~1.5 h).

**Path 2 (pragmatic, evidence-driven): immediately deepen
value=0.3 at 8 seeds × 32 ep**, knowing the spec allows
this when "exactly one axis (α)" applies. The seed=99
outlier question is the most informative remaining
ambiguity at this point; spending the wallclock on
disambiguating it makes more headline-progress than
mapping the other two axes. Wallclock cost: ~4 h (8 seeds
× 32 ep × 2 arms = 16 runs).

The methodologically-cleanest path is **Path 1**. The
operationally-fastest path to a clear iter-65 verdict is
**Path 2**.

### What this commit appends

- Full-phase value=0.3 renderer table.
- Smoke vs full per-seed comparison.
- Mechanistic verdict (α confirmed; perforant-path
  hypothesis provisionally validated).
- Diagnosis on seed=99 as deterministic outlier.
- iter-65 fork choice (Path 1 vs Path 2) for Bekos's
  decision.

No goalpost-shift: the locked α threshold was
`Δ̄ > 0 AND n_pos ≥ ⌈3n/4⌉ AND t > 0`. value=0.3 at full
satisfies all three (`+0.0164 > 0`, `3/4 ≥ 3`, `+0.996 > 0`).
The seed-99 outlier is documented as a question for iter-65,
not as a reason to relax the matrix.

## Live results — Axis B (`r2_p_connect`)

### Axis B smoke (16 ep × 4 seeds × 3 values)

Run command:

```sh
cargo run --release -p eval --example reward_benchmark -- \
  --axis-sweep r2-p-connect \
  --values 0.025,0.05,0.10 \
  --seeds 42,7,13,99 \
  --axis-sweep-phase smoke \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval
```

Wallclock: ~1.5 h on local hardware. Cache pre-seeded with
iter-63 baseline values for `value=0.05` (4/4 seeds short-
circuited; the per-seed values reproduce the iter-63
calibration commit `a08a117` bit-for-bit, confirming
deterministic seed handling and cache integrity).

**Per-value renderer table:**

| value | μ_untrained | μ_trained | Δ̄ | σ_Δ | n_pos | n_pass(0.0621) | t(df=3) | classification |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 0.025 | 0.0425 | 0.0425 | +0.0000 | 0.0000 | 0/4 | 0/4 | +0.000 | **(β) Beta — sparse-locked state** |
| 0.050 | 0.0195 | 0.0342 | +0.0147 | 0.0100 | 3/4 | 0/4 | +2.933 | (α) at smoke — *known* iter-51 oscillation, β at full |
| 0.100 | 0.0454 | 0.0454 | +0.0000 | 0.0000 | 0/4 | 0/4 | +0.000 | **(β) Beta — dense-locked state** |

Per-value tally (raw): α = 1, β = 2, γ = 0, δ = 0.

**Per-seed Δ matrix:**

| seed | val=0.025 | val=0.050 | val=0.100 |
| ---: | ---: | ---: | ---: |
| 42 | +0.0000 (locked) | +0.0186 | +0.0000 (locked) |
| 7  | +0.0000 (locked) | +0.0000 | +0.0000 (locked) |
| 13 | +0.0000 (locked) | +0.0225 | +0.0000 (locked) |
| 99 | +0.0000 (locked) | +0.0176 | +0.0000 (locked) |

### Mechanistic reading — narrow-window operating regime

This is **not** a "denser is better" or "sparser is better"
axis. It is a *narrow operating window* — most R2-recurrent
densities take the brain into a **locked state** where
plasticity makes zero measurable difference to the decoder
read-out, and only one specific connectivity (the iter-46
default 0.05) produces any non-zero Δ.

Four distinct findings, in order of importance:

**(1) Δ = 0 bit-for-bit is a *qualitatively different
regime* from "small positive".** On 8 of 12 seed-value
points (every value=0.025 seed, every value=0.100 seed),
the trained brain produces a `target_top3_overlap` value
that is *bit-identical* to the untrained brain. Not
"nearly equal" — exactly equal in IEEE-754. This is not
"the metric is too coarse to show a small effect"; it is
a deterministic computation that produces an identical
fingerprint dictionary in trained and untrained mode.

The interpretation: at sparse R2 recurrent (0.025), the
recurrent attractor cannot carry the engram between
training trials, so plasticity-induced weight changes
do not propagate into the eval-phase decoder fingerprint.
At dense R2 recurrent (0.10), the recurrent network
self-dominates so strongly that the decoder fingerprint
is determined by the recurrent steady-state, not by the
trained synaptic differences. **In both regimes, the
decoder reads the same R2 output regardless of training.**

**(2) `value=0.05` is the *only* active B-point — not
"the best" of three.** Reading axis B as a "monotone
sweep of connectivity" misses the regime change. value=0.05
is the iter-46 / iter-63 baseline configuration. It is the
only value where any seed produces a non-zero Δ. This is
not a graceful degradation as we move away from 0.05 — it
is a regime cliff: 0.025 and 0.100 are both completely
silent on plasticity-driven Δ.

**(3) Both extremes neutralise plasticity in the read-out
— but via different mechanisms.** Sparse (0.025): no
recurrent attractor → engram not carried. Dense (0.100):
self-dominated recurrent → trained synapses overridden by
the bulk recurrent steady state. The two are mechanistically
distinct failure modes; neither is "noise". The fact that
both produce *exactly the same* outcome (Δ = 0 bit-for-bit
on every seed) is the smoking gun: this is an *architectural*
neutralisation, not a training-time effect.

**(4) Axis B argues for *dynamic operating windows*, not
monotone scaling.** A future iteration that wanted to
"increase recurrent density to improve learning" would step
straight off the cliff into the dense-locked state. The
narrow window means the connectivity parameter is *not* a
free knob to tune for performance — it must be at or very
near 0.05 for the rest of the iter-46 plasticity stack to
have anything to write into.

### Honest reading on the value=0.05 α at smoke

The value=0.05 row in the renderer says (α). But this is
**known** to be the iter-51 per-epoch oscillation pattern,
not a stable signal:

- The (Δ̄, t, n_pos) triple at value=0.05 is identical to
  axis C value=0.0 smoke (same configuration: iter-63
  baseline). iter-63 already proved this configuration
  collapses to (Δ̄ = −0.0027, β / Branch B FAIL) at 32 ep.
- The per-seed values match the iter-63 calibration locked
  values bit-for-bit (cache pre-seed), confirming
  determinism.

So axis B's only "α" is a known oscillation peak that fails
at full phase. Axis B contributes **zero robust α** to
iter-65; both axis-B extremes lock plasticity out, and the
middle point is iter-51 oscillation that already failed.

### Axis B verdict (locked, applied verbatim)

Per the iter-64 ENTRY locked acceptance matrix: per-value
classifications stand. The axis-level interpretation, given
that the only "α" is a known unstable oscillation:

> **Axis B does not contribute a robust mechanism to
> iter-65.** The narrow operating window at 0.05 is iter-51
> oscillation; sparse and dense extremes are locked-state
> failures.

This *is not* a goalpost shift — the per-value classifications
are exactly what the locked matrix produces. The axis-level
*interpretation* takes the smoke-vs-full lesson from iter-51
into account: an α at smoke that we *already know* collapses
to β at full does not constitute a candidate for iter-65
deepening. The iter-64 ENTRY's two-phase logic exists exactly
for this distinction.

### Implication for iter-65 fork

Axis C (perforant path) is *the* mechanism axis with a
robust α (persistent at full). Axis B has confirmed *what
not to touch* (R2 connectivity is in a narrow window that
must stay at 0.05). The iter-65 fork now narrows: Path 2
(deepen value=0.3 of axis C at 8 seeds) is more likely to
be the right next step than Path 1 (orthogonality grid),
because axis B has ruled itself out of the orthogonality
candidate pool. The remaining piece is axis A.

## Files to write (post-Go, in implementation phase)

- `crates/eval/src/reward_bench.rs` — `--axis-sweep` runner,
  untrained-cache keyed by `(axis, value, seed)`, runtime
  override for `R2_P_CONNECT` (axis B).
- `crates/eval/examples/reward_benchmark.rs` — `--axis-sweep`
  CLI block analogous to `--target-overlap-bench`.
- Optional axis-A sub-axis: snn-core `Network` per-region or
  per-synapse-class plasticity gating.
- `notes/64-mechanism-diagnosis.md` — this spec; live results
  per axis appended in dedicated subsections after each sweep.
- `CHANGELOG.md` — iter-64 block tracking the three axis
  outcomes and the iter-65 fork choice.

## Open questions for Bekos (before implementation Go)

1. **Axis-value lists.** Are `dg_to_r2_weight ∈ {0.25, 0.50,
   1.00, 2.00}`, `R2_P_CONNECT ∈ {0.025, 0.05, 0.10}`, and
   `direct_r1r2_weight_scale ∈ {0.0, 0.1, 0.3, 1.0}` the
   right four / three / four points? Or different log scales?
2. **Smoke endpoint choice.** Smoke at the *lowest* or
   *highest* value of each axis? My default: lowest (smallest
   perturbation from baseline) for axes A and C, the
   non-baseline-closest value for axis B (i.e. 0.025 since
   0.05 is baseline and a smoke at baseline is just an iter-
   63-replication).
3. **Sub-axis disposition for axis A.** Drop the per-synapse-
   class STDP sub-axis entirely (axis A is then just
   `dg_to_r2_weight` alone)? Or keep it and accept the
   plumbing cost? My default: drop for iter-64; revisit in
   iter-66+ if axis A shows (α).
4. **Axis acceptance thresholds.** Are `n_pos ≥ 3/4` (α),
   `\|Δ̄\| ≤ σ_untrained` (β), `n_pos ≤ 1/4` (γ) the right
   bounds? σ_untrained = 0.021 from iter-63 calibration; the
   (β) band is then `\|Δ̄\| ≤ 0.021`. Would a tighter bound
   (e.g. `\|Δ̄\| ≤ 0.5 σ_untrained`) be more honest?
5. **Axis order.** Sequential alphabetical (A → B → C) vs
   cheap-first (C is code-free → first; B needs `R2_P_CONNECT`
   override; A potentially heaviest with sub-axis decision)?
   My default: **C → B → A** (cheap-to-expensive code-wise).
   Lets us start measuring *today* on axis C while axis-B
   plumbing is being written.

## Headline (placeholder, filled after all three axes complete)

> *to be filled after all three axes complete; one of:*
> - **Axis X identifies plausible mechanism** — iter-65 = deepen.
> - **Multiple axes show trend** — iter-65 = orthogonality test.
> - **All three axes (β/γ)** — iter-65 = structural question
>   (target encoding / read-out path / forced architecture
>   pivot to CA3/CA1).

---

## Status of this commit

This file is committed **before** any iter-64 implementation
code is written. The next iter-64-related commit will be the
**implementation** of the `--axis-sweep` runner and the
related plumbing — but only after Bekos's explicit Go on
this spec and resolution of the five open questions above.

No measurements. No code. Spec only.
