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
