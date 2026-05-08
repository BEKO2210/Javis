# Gate-B — γ.4 8-seed verdict

**Date:** 2026-05-08.
**Frozen config:** `reports/gate_b_gamma_4_entry.md` + the locked
γ.1.1 base + `--c1-btsp-non-target-depression-strength 0.2` (= 0.5 ×
`--c1-btsp-strength`).
**Pre-registration:** `reports/gate_b_gamma_4_entry.md` (commit
`475f379`, locked before any γ.4 compute).
**Run platform:** Windows PC (Bekos), 8 sequential 32-epoch runs,
2026-05-08 08:34 → 21:30 UTC (~13 h wallclock).
**Logs:** `reports/runs/gate_b_gamma_4_seed{0..7}.log`.
**Per-seed verdict JSONs:** `reports/runs/gate_b_gamma_4_seed{0..7}.verdict.json`.

## Headline

> **Class (D) Reject — γ.4 mechanism wrong.**
>
> 0/8 seeds PASS. **All 8 seeds collapse to `top3_c1 = 0.0000` for
> the entire 32-epoch run** (longest contiguous-zero = 32/32 on
> every seed). Mean(last-8) = 0.0000 ± 0.0000. The locked
> `n_pos ≤ 4/8 OR mean(last-8) ≤ 0.0` (D)-condition is met *both*
> ways. Per the pre-registered acceptance matrix: **γ.4 mechanism
> insufficient — escalate to Bekos for direction (γ.5 / Mechanism M2
> Willshaw baseline / pivot to non-CA1 readout). No further
> parameter sweeps without explicit Go.**

## Per-seed table (8-seed verdict)

| seed | γ.1.1 last-8 | γ.4 last-8 | Δ | γ.4 32-ep mean | longest contig.-zero | R2 32-ep mean |
| ---: | -----------: | ---------: | --: | ---: | ---: | ---: |
|    0 | 0.0156       | **0.0000** | **−0.0156** | 0.0000 | 32 | 0.0303 |
|    1 | 0.0977       | **0.0000** | **−0.0977** | 0.0000 | 32 | 0.0381 |
|    2 | 0.0898       | **0.0000** | **−0.0898** | 0.0000 | 32 | 0.0322 |
|    3 | 0.0742       | **0.0000** | **−0.0742** | 0.0000 | 32 | 0.0264 |
|    4 | 0.0312       | **0.0000** | **−0.0312** | 0.0000 | 32 | 0.0244 |
|    5 | 0.1289       | **0.0000** | **−0.1289** | 0.0000 | 32 | 0.0205 |
|    6 | 0.0742       | **0.0000** | **−0.0742** | 0.0000 | 32 | 0.0205 |
|    7 | 0.0430       | **0.0000** | **−0.0430** | 0.0000 | 32 | 0.0537 |

R2 readouts (`top3_r2`) are bit-identical to γ.1.1 — γ.4 only
modifies BTSP on R2-E → C1, not R2-R2. The R2 path passes through
the iter-65 chance band cleanly (0.0205 – 0.0537) on every seed.

## Aggregate statistics

| Stat | γ.4 (8 seeds) | γ.1.1 (8 seeds, prior) |
| --- | ---: | ---: |
| pass count | **0/8** | 5/8 |
| mean of last-8 means | **0.0000** | 0.0693 |
| std of last-8 means | 0.0000 | 0.0375 |
| Δ vs γ.1.1 mean | **−0.0693 (−100 %)** | — |

`t(7)` against thresholds is degenerate — the γ.4 distribution is
identically zero (every seed = 0.0000, std = 0.0). The verdict
collapses to direct comparison: γ.4 is exactly *"γ.1.1 minus γ.1.1's
signal"*.

## Class verdict (locked acceptance matrix)

| Class | Per-seed pattern | Aggregate | Match? |
| --- | --- | --- | :---: |
| (A) Confirm — strong       | 8/8 PASS AND `t(7) > 2.5`            | mean > 0.07         | NO (0/8) |
| (B) Robust — directional   | `n_pos ≥ 7/8` AND `t(7) > 1.895`     | mean ≥ 0.05         | NO (0/8) |
| (C) Partial — needs different rule | `5/8 ≤ n_pos < 7/8`          | `t(7) > 0`          | NO (0/8) |
| **(D) Reject — γ.4 mechanism wrong** | `n_pos ≤ 4/8` **OR** mean(last-8) ≤ 0 | (chance) | **YES — both conditions met** |

`n_pos = 0/8` (≤ 4/8) AND `mean(last-8) = 0.0` (≤ 0.0). The Gate-B
γ.4 ENTRY's tightening clause also applies: *"γ.4 must beat γ.1.1's
8-seed mean (0.0693) to count as Class (A) or (B); equal-to-γ.1.1
collapses to (C)"* — γ.4 is **not equal to**, but **catastrophically
below** γ.1.1, which routes unambiguously to (D).

## Pre-registered hypothesis check (H1–H4)

| Hyp. | Statement | Required | Observed | Verdict |
| --- | --- | --- | --- | :---: |
| **H1 (lift)** | seeds 0, 4, 7 (γ.1.1 FAIL) cross last-8 mean ≥ 0.05 under γ.4 | 1+/3 | **0/3** | **FAIL** |
| **H2 (preserve)** | seeds 1, 2, 3, 5, 6 (γ.1.1 PASS) stay ≥ 0.05 under γ.4; no catastrophic loss > 50 % | 5/5 above threshold AND no >50% regression | **0/5** above threshold; all 5 dropped 100 %  | **CATASTROPHIC FAIL** |
| **H3 (separation)** | `w_ratio = tgt_w / non_w > 1.05` on ≥ 5/8 seeds | 5+/8 | **1/8** (seed 6, w_ratio = 1.31, but absolute weights 0.0002 / 0.0001 — no real separation, just ratio noise on near-zero magnitudes) | **FAIL** |
| **H4 (trajectory)** | DEGRADING per-cue trajectory eliminated on ≥ 2/3 of previously-failing seeds (0, 4, 7) | 2+/3 | **vacuous** — no trajectory exists when readout is identically 0 across all 32 epochs | **N/A** (rule trivially "eliminates" DEGRADING by eliminating signal entirely) |

All four hypotheses falsified. **The non-regression guard (H2) is
the most damning**: γ.4 didn't just fail to lift the failing seeds,
it *destroyed* the binding signal on every seed that γ.1.1 had
already won.

## Per-seed γ.4 diagnostic

| seed | tgt_w | non_w | w_ratio | C1 spikes (teacher) | kwta_empty | dict_concepts | raw_overlap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|    0 | 0.0007 | 0.0017 | 0.399 | 1 205 | 32/32 | 0.0 | 0.0000 |
|    1 | 0.0006 | 0.0017 | 0.335 | 1 107 | 32/32 | 0.0 | 0.0000 |
|    2 | 0.0004 | 0.0015 | 0.280 | 1 126 | 32/32 | 0.0 | 0.0000 |
|    3 | 0.0002 | 0.0007 | 0.316 | 1 100 | 32/32 | 0.0 | 0.0000 |
|    4 | 0.0001 | 0.0005 | 0.120 | 1 146 | 32/32 | 0.0 | 0.0000 |
|    5 | 0.0002 | 0.0005 | 0.354 | 1 078 | 32/32 | 0.0 | 0.0000 |
|    6 | 0.0002 | 0.0001 | 1.310 | 1 144 | 32/32 | 0.0 | 0.0000 |
|    7 | 0.0009 | 0.0019 | 0.465 | 1 118 | 32/32 | 0.0 | 0.0000 |

Compare to γ.1.1 (`tgt_w ≈ non_w ≈ 0.80`, C1 teacher spikes ≈ 6 327
– 6 348, kwta_empty = 0/32, dict_concepts = 64). γ.4 collapses
*every* dimension of the binding signal:

1. **Weights collapsed to ~zero on every seed.** R2-E → C1 mean
   weights are 0.0001 – 0.0019 (target *and* non-target). The
   initial weight band was `[0, 0.5]` with mean ≈ 0.25 — the LTD
   branch drove every synapse to `w_min = 0.0` already in epoch 0
   (`tgt_w` = 0.0007 at ep 0 was virtually identical to ep 31 →
   collapse complete on the first epoch).
2. **w_ratio is < 1 on 7/8 seeds.** Target weights are *lower* than
   non-target weights — the rule is producing the *opposite* of the
   intended separation on most seeds. Seed 6's w_ratio = 1.31
   sounds promising but is meaningless: both numerator and
   denominator are ~0.0001, near-zero noise.
3. **C1 teacher firing dropped 5×** (γ.1.1: 6 327 mean → γ.4: 1 117
   mean). With R2-E → C1 weights at ~0, the only thing driving C1
   firing during teacher is the direct 500 nA target clamp on the
   30 canonical-target cells (~14 spikes / cell × 30 cells × 256
   trials / 32 epochs = ~3 300 expected) plus residual recurrent
   noise. The observed 1 100 / epoch is consistent with clamp-only
   drive, no R2-E contribution.
4. **kwta_empty = 32/32 on every epoch on every seed.** At
   recall-time (no clamp), R2-E → C1 weights are too low to drive
   any C1 cells to fire. Empty kWTA → empty fingerprint dictionary
   → no decoder readout possible. dict_concepts = 0 confirms.
5. **R2 readout unchanged** (top3_r2 = 0.020 – 0.054, identical to
   γ.1.1's range). γ.4 was correctly scoped to R2-E → C1.

## Mechanism diagnosis: why γ.4 destroys the signal

The pre-registration's design assumption was that *target* C1 cells
(30 of them) plateau under the 500 nA clamp, while *non-target* C1
cells (970 of them) plateau only rarely. Under that assumption,
LTD on non-target plateaus would gently pressure non-target weights
down without reaching `w_min`.

The actual γ.1.1 conditions (which γ.4 inherits verbatim by design
— same locked seed set, same E/I split, same R2-isolation OFF)
violate that assumption:

- `--c1-btsp-no-r2-isolation` keeps cue + DG drive at full strength
  during teacher Phase 4. R2-E fires its full cue + recurrent + DG
  response throughout the clamp window.
- `--c1-btsp-teacher-recurrent-i-scale 0.3` reduces R2-R2 inhibitory
  recurrent to 30 %. With reduced inhibition, R2-E firing density
  is high.
- This R2-E activity drives a *substantial fraction* of C1's 1000
  cells into firing — and a meaningful subset reach plateau
  threshold (5 spikes / 30 ms).

Empirical evidence: `plateau_events ≈ 30 000 – 80 000 per epoch` on
γ.4 (logged in every per-epoch diag line). The 30 canonical-target
cells alone, even saturating at ~14 spikes each over 256 trials,
account for at most ~10 000 plateau events / epoch
(30 cells × 256 trials × 1 plateau event each + a few re-arms).
The remaining 20 000 – 70 000 plateaus per epoch fire on
**non-target cells**.

Consequence: every plateau on a non-target cell triggers LTD on its
~30 incoming synapses with `Δw = -0.2 × tag`. Across 256 trials/epoch
× ~50 000 non-target plateaus/epoch × ~30 incoming synapses each
≈ 384 million LTD applications per epoch (with tag consumption
collapsing this to a much smaller number of effective unique
synapse-events, but still vastly outnumbering the LTP events).

The asymmetry is structural, not numerical: **there are ~30 target
cells and ~hundreds of non-target cells with R2-E drive sufficient
to plateau under γ.1.1's E/I-split.** Any non-zero
`non_target_depression_strength` makes the LTD population dwarf the
LTP population, driving every weight to `w_min`. Lowering the
strength would slow the collapse but not change its direction.

The rule design has a *multiplicity* problem, not a *strength*
problem. **γ.5 needs a different mechanism, not a γ.4 sweep.**

## Recall-mode stability

| seed | top3_r2 32-ep mean | top3_r2 < 0.005 epochs | r2_collapsed |
| ---: | -----------------: | ---------------------: | :----------: |
|    0 | 0.0303             | 12/32                  | False        |
|    1 | 0.0381             |  7/32                  | False        |
|    2 | 0.0322             |  9/32                  | False        |
|    3 | 0.0264             | 13/32                  | False        |
|    4 | 0.0244             | 16/32                  | False        |
|    5 | 0.0205             | 17/32                  | False        |
|    6 | 0.0205             | 17/32                  | False        |
|    7 | 0.0537             |  4/32                  | False        |

Bit-identical to the γ.1.1 8-seed numbers (γ.4 doesn't touch R2-R2
plasticity). No R2 collapse on any seed. The C1 destruction is
isolated to R2-E → C1 weights.

`recall_mode_eval = true` was active throughout (locked in
`gate_a_gamma_1_1_config.json` + γ.4 base). The iter-52 / iter-62
invariant (`disable_all_plasticity` before each eval phase) was
honoured by construction on all 8 seeds.

## Decision per the locked acceptance matrix

> **Verdict:** Class (D) Reject — γ.4 mechanism wrong.
>
> **Action (verbatim from `reports/gate_b_gamma_4_entry.md` (D)
> row):** *"γ.4 mechanism insufficient. Escalate to Bekos for
> direction (γ.5 vs Mechanism M2 Willshaw baseline vs pivot to
> non-CA1 readout). No further parameter sweeps without explicit
> Go."*

What this is NOT:
- **NOT a sweep candidate.** The pre-registration explicitly
  forbids parameter sweeps after a (D) verdict. Lowering
  `--c1-btsp-non-target-depression-strength` would not change the
  qualitative collapse (the rule's multiplicity asymmetry would
  just take more epochs to drive weights to zero).
- **NOT a γ.4-with-different-config attempt.** The collapse is
  apparent in epoch 0 already; it is not a long-term homeostatic
  artefact that a tighter teacher window could fix.
- **NOT a license to weaken or relax γ.1.1's Class (C) verdict.**
  γ.1.1's 5/8-PASS, mean(last-8) = 0.0693 result stands. γ.4 has
  failed to improve on it; it has not retroactively changed it.

## Three pre-registered next steps (Bekos's call required)

Per the locked Gate-B γ.4 ENTRY's Class (D) row, the path forward
is *not* a γ.4 retune but one of three architectural alternatives.
None of these may be launched without an explicit Bekos Go +
fresh ENTRY pre-registration:

### Option 1 — γ.5: heterosynaptic competition

Keep the C1 readout layer, keep the BTSP eligibility tag, but
change the rule:
> When LTP fires on a target cell, simultaneously LTD all non-target
> post-cells receiving the same pre-pattern (R2-E pre cells with
> non-zero tag).

This swaps the multiplicity asymmetry: instead of ~hundreds of
non-target depressors per target potentiator (γ.4), heterosynaptic
LTD couples 1:1 with each LTP event — exactly one depression per
potentiation, anchored to the *same* presynaptic pattern. Bittner
2017 / Magee & Grienberger 2020 explicitly describe this as the
canonical pairing for plateau-induced field formation in CA1.

Implementation cost: ~50 LOC in `crates/snn-core/src/network.rs`
(reverse-adjacency pre → post lookup at LTP-event time, scan
tagged synapses on non-target post-cells, apply LTD). Same
diagnostic plumbing.

### Option 2 — Mechanism M2: Willshaw binary heteroassociative store

The iter-66 deep-research note's M2 fallback. Replace the BTSP
plateau-eligibility rule on R2-E → C1 with a Willshaw-style binary
heteroassociative matrix: R2-E → C1 weights are 0/1, set to 1 on
co-activity of a target trial, never reset. Decoded by C1
threshold + kWTA. Information-theoretic upper bound on capacity
known (Willshaw 1969); serves as a *baseline* the BTSP rule must
beat.

Implementation cost: ~150 LOC for the Willshaw matrix module +
decoder; can co-exist with BTSP as an A/B path.

### Option 3 — Pivot to non-CA1 readout

Abandon the CA1-equivalent C1 layer entirely. Per O'Reilly &
Rudy (2001), heteroassociative binding might require a different
substrate — direct DG → C1 with a contrastive Hebbian rule, or
the modern-Hopfield head deferred from iter-66 (Mechanism M3).
This is the deepest pivot; would invalidate the iter-66 / 66.5 /
67 chain at the architecture level (the C1 readout itself, not
just the binding rule).

## Limitations and honest caveats

1. **The diagnostic counter `btsp_depression_events` was not logged
   per-epoch.** I added the counter to `Network::btsp_depression_events`
   in commit `475f379`, but the eval harness's `[iter-66 diag]` line
   was not extended to print it. Implementation-readiness gap, not
   a verdict-affecting issue: the LTD path *clearly* fired (weights
   dropped from 0.25 init to 0.001 already in epoch 0), but the
   exact event count is not directly visible from the logs. Future
   γ.5 ENTRY should add this line item.
2. **Single platform.** All 8 seeds ran on Bekos's Windows PC. No
   Linux cross-platform replicate (the verdict is so unambiguous
   — every seed identically zero — that cross-platform replication
   would not change the class).
3. **Sub-permille drift caveat unchanged.** R2 metrics bit-identical
   to γ.1.1 across platforms by inspection.
4. **Compute budget.** Total γ.4 wallclock: ~13 h sequential on
   the Windows PC (08:34 → 21:30 UTC, 8 seeds × ~95 min each).
   No process kills.

## Status

- **iter-67-γ.4 verdict at Gate-B: Class (D) Reject.**
- iter-67-γ.4 mechanism is qualitatively wrong, not numerically
  miscalibrated. Parameter sweeps are forbidden by the locked
  acceptance matrix.
- **No more compute on iter-67-γ.4 without a fresh ENTRY.**
- **Pre-registered next steps require Bekos's explicit Go.** Three
  options on the table: γ.5 heterosynaptic competition (lowest
  pivot, ~50 LOC), Mechanism M2 Willshaw baseline (capacity
  baseline, ~150 LOC), or full architecture pivot to non-CA1
  readout (deepest revision).
- iter-67's two-iteration chain (γ.1.1 Class C, γ.4 Class D) leaves
  the project at the same open question Marr 1971 / O'Reilly &
  McClelland 1994 / Schapiro 2017 already named: the binding
  problem on top of pattern separation is *not* solved by simple
  Hebbian / pair-STDP / target-gated BTSP / target-gated-with-LTD
  rules. The next mechanism either tightens the LTP / LTD pairing
  geometry (γ.5) or steps outside the gradient family entirely
  (M2 / M3).
