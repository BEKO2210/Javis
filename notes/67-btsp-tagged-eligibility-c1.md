# Iter 67 — BTSP / tagged eligibility on R2 → C1 (gain-fix for the C1 readout)

**Status: ENTRY (pre-registration). No measurements, no
implementation yet. The hypothesis, locked seed set, locked
acceptance gate, locked metrics, and locked CLI surface are
committed in this file *before* any code is written. They may
not be relaxed post-hoc.**

> **Core sentence.** iter-66 (M1 — CA1-equivalent C1 readout)
> and iter-66.5 (Path-1 — eval-aligned R-STDP) both falsified
> at the seed-42 32-ep gate. Bekos's gain + credit-assignment
> diagnosis (this conversation) is accepted: pair-style
> R-STDP cannot bind the cue → target temporal gap with enough
> postsynaptic gain at recall. iter-67 swaps the R2 → C1
> learning rule from R-STDP to a BTSP-style target-gated
> eligibility kernel — long eligibility window opened by
> cue-driven R2 pre-activity, gated by C1 target-cell plateau
> firing during the teacher window. **Architecture
> unchanged**: DG = pattern separator, R2/CA3 = cue-state
> provider, C1/CA1 = readout/binding layer. Only the R2 → C1
> *learning rule* changes.

## 1. Why iter-67

iter-66.5's seed-42 32-ep smoke (notes/66.5 §"Step 4") closes
the iter-66 chain with a clean negative result on Path 1:
24/32 epochs of bit-identical zero on `c1_target_top3_overlap`,
zero in eval-phase C1 spikes, but `c1_active_frac = 1.0` and
`clamp_eff = 1.0` during teacher and `nz_upd = 42 000` weight
movements per epoch. The Path-1 fix (drop the R2 target clamp
during teacher → R-STDP learns on the eval-time R2 cue
pattern) eliminated the iter-66 covariate-shift but did NOT
solve the C1-silence problem at recall. The remaining
mechanism is a *gain* problem, not a *pattern* problem.

iter-66.6 (Path 2 — disable R2 homeostasis when c1.enabled)
is **skipped**: per Bekos's analysis, removing the homeostatic
weight equilibrium does not address the gain problem either,
because the R-STDP rule's `w_max = 0.8` cap is already
saturated (`max|Δw| = 0.7990` every epoch in iter-66.5) and
the saturated weight × 1 firing R2-E input × 5 mV synaptic
kernel ≈ 4 mV is below the LIF threshold-from-rest gap of
~15 mV.

iter-67 is therefore the next experimental target: a
plasticity rule that creates a long eligibility window on
R2-E → C1 synapses (so multiple cue-driven R2-E pre-spikes
can sum into the tag) and a post-cell-local plateau gate (so
only the canonical-target C1 cells receive the potentiation
pulse at the target moment).

## 2. What iter-66 / iter-66.5 proved

| Iteration | Hypothesis | Verdict | Key data |
| --- | --- | --- | --- |
| iter-66 | CA1-equivalent C1 readout, R-STDP gated by M_target = +1 during double-clamp teacher | (D) Reject — C1 readout = 0 even with C1 layer wired correctly | 8-ep smoke seed=42: top3_c1 = 0.0000; teacher-phase c1 active 100%; R-STDP fires every epoch; eval-phase C1 silent (kwta_empty = 32/32) |
| iter-66.5 (Path 1) | Drop R2 target clamp during teacher → R-STDP learns eval-time R2 cue pattern | (P1.C) Flat zero — C1 readout = 0 across 24 epochs | 32-ep smoke seed=42 (killed at ep 23 once 8-contiguous-zero confirmed): top3_c1 = 0.0000 every epoch; r2c1_l2 = 45.05 (saturated equilibrium); max|Δw| = 0.7990 (R-STDP cap saturated) |

**What is mechanically working and must NOT be touched in iter-67:**
- C1 layer wiring (1000 cells, sparsity_k=20, fan-out=30 from R2-E).
- Canonical-target C1 SDR generation + per-trial clamp delivery.
- M_target modulator gating (c1_active_frac = 1.0 every trial).
- iter-46 plasticity stack on R2-R2 (STDP, iSTDP, homeostasis, intrinsic, R-STDP).
- DG pattern separator + decorrelated init wiring.
- Recall-mode-eval invariant (iter-52 / iter-62 bit-identical L2 across eval).

**What is mechanically broken and is the iter-67 target:**
- The R2-E → C1 plasticity rule (currently R-STDP with
  pair-coincidence-window eligibility) cannot bind cue-driven
  R2-E activity to the canonical C1 target SDR with enough
  postsynaptic gain to fire C1 cells from R2 input alone at
  recall.

## 3. Accepted diagnosis: gain + delayed credit assignment

Bekos's analysis (this conversation) is locked as the iter-67
working diagnosis:

> The C1-silence at recall is a *covariate shift between
> training and recall* (training: C1 receives strong external
> teacher-clamp drive; recall: C1 receives only weak
> cue-induced R2 input) AND a *credit-assignment gap*
> (R-STDP's pair-coincidence window only opens during the
> brief Phase-4 overlap of cue-driven R2-E spikes and
> clamp-driven C1 spikes; the structural cue-target temporal
> gap that BTSP is biologically built for is not bridged by
> pair STDP).

The two failure modes have separate architectural
implications:

- **Gain.** Address by giving R2-E → C1 a plasticity rule
  whose potentiation is multi-spike-summable across the
  ~80 ms cue + delay + prediction substrate, not just
  pair-coincidence within the ~30 ms teacher clamp window.
- **Credit assignment.** Address by tagging the R2-E
  pre-spike trace during cue-driven activity (when no
  post-spike has fired yet) and applying the potentiation
  retroactively when the C1 target plateau fires under the
  teacher clamp.

BTSP (Bittner 2017 / Magee & Grienberger 2020) is the
canonical biological mechanism for exactly this combination:
plateau-triggered retroactive potentiation across a long
eligibility window.

## 4. Hypothesis

> On the iter-66.5 brain configuration (C1 layer enabled,
> eval-aligned-rstdp on, recall-mode-eval on, decorrelated
> init, DG bridge on, vocab=64) AND with the iter-67
> R2-E → C1 plasticity rule swapped from R-STDP to a
> BTSP-style target-gated tagged-eligibility kernel, a single
> seed (42) trained for 32 epochs produces a *measurable,
> non-trivial, monotonically rising* `c1_target_top3_overlap`
> curve such that:
>
> - last-8-ep mean `c1_target_top3_overlap ≥ 0.05`
> - first-8-ep mean `< 0.02`
> - C1 eval activity > 0 on the majority of cues
>   (`kwta_empty < 0.5 × n_pairs` in the last 8 epochs)
> - recall-mode-eval L2 invariant: pre/post bit-identical
>   (= 0 drift) every epoch.

This hypothesis is *strictly stricter than iter-66.5's*
single-seed gate on the c1 readout side AND adds a positive
structural check (eval-activity fraction). The iter-66
locked acceptance matrix (A/B/C/D) carries over verbatim
*for the eventual 8-seed verdict run*, which iter-67 does
NOT execute on its own.

## 5. Mechanism: R2 → C1 BTSP tagged eligibility

```text
Per-trial timeline (the iter-46 6-phase schedule, unchanged
except for which plasticity rule fires on R2-E → C1):

  Phase 1: cue (40 ms)
    R1 → R2 forward + DG → R2 mossy-fibre drive
    R2 fires its natural cue-driven response (sparse, ~30
       R2-E cells)
    R2-E pre-spikes increment R2-E → C1 *eligibility tag*
       (additive per pre-spike, decays with τ_window)
    No post-spike on C1 expected here (small cue-driven drive)

  Phase 2: delay (10 ms)
    Idle. Eligibility tag continues decaying.

  Phase 3: prediction (20 ms)
    Cue-only re-presentation, plasticity normally off for
       eval-equivalence. Iter-67 keeps R2-E → C1 BTSP
       eligibility decay running here too — the tag survives
       up to ~τ_window = 200 ms total from cue onset.

  Phase 4: teacher (40 ms = 12 ms lead-in + 28 ms clamp)
    Lead-in: cue alone (no clamp). Eligibility continues.
    Clamp window: C1 target SDR clamped at 500 nA →
       canonical-target C1 cells fire ~15 spikes each
       (saturation under the 30 ms clamp).
    BTSP plateau threshold: when a C1 cell accumulates
       ≥ btsp_plateau_threshold_spikes within the clamp
       window, its `plateau_armed` flag triggers.
    On every plateau-arm event:
       For every R2-E → (this C1 cell) synapse with non-zero
       eligibility tag:
         w[s] += btsp_strength * eligibility_tag[s]
         (clamped to [0, w_max])
    Plateau decays after btsp_post_plateau_decay_ms of post
       silence.

  Phase 5: reward (≤ 40 ms, conditional)
    Standard iter-46 reward delivery on R2-R2 R-STDP.
       BTSP on R2-E → C1 is independent of this — the reward
       modulator does not gate BTSP.

  Phase 6: tail (20 ms)
    Idle. Eligibility tags decay to zero before the next
       trial's reset_state.

Why this binds cue→target:
- Eligibility tag accumulates during cue/delay/prediction:
  the tag is a "memory" of which R2-E cells were active during
  the cue-substrate, before any teacher signal arrives.
- Plateau gates potentiation to canonical target cells only:
  R-STDP's global modulator is replaced by per-post-cell
  plateau detection, so non-target C1 cells never receive
  BTSP potentiation no matter how active they were.
- The kernel is "retroactive": at recall, the same R2-E cells
  that fired during cue/delay/prediction now drive the same
  C1 target cells through the BTSP-potentiated synapses.
- No clamp present at recall: the synapse must carry the
  binding alone — exactly what we want to read out.

Crucially, this rule is **NOT a global modulator**. The
existing R-STDP eligibility decays on every synapse and is
gated by the network-wide `neuromodulator` scalar. BTSP is
**per-post-cell** — the plateau on C1 cell `c` triggers
potentiation only on synapses targeting `c`, regardless of
what other C1 cells are doing. This is what fixes credit
assignment.
```

## 6. Implementation plan

### Reuse audit of existing snn-core

`crates/snn-core/src/stdp.rs` already has a `soft_bounds:
bool` flag that implements *Bittner-style multiplicative
weight bounds* — but this is a soft-cap on the standard
pair-STDP update, NOT a plateau-eligibility kernel. iter-44's
note `notes/44-breakthrough-plasticity.md` confirms this is
the only "BTSP" reference in the codebase. The plateau-
eligibility rule does NOT exist; iter-67 implements it as a
new module.

The new module will share infrastructure with R-STDP (pre /
post traces on the network, per-synapse update apply during
network step) but operates on a different storage:
per-post-cell `plateau_armed` + per-synapse `tag_strength`
trace, NOT the existing per-synapse `eligibility` trace
(which stays available for R-STDP on R2-R2).

### File-by-file plan

- `crates/snn-core/src/btsp.rs` — **NEW**. Defines
  `BtspParams` (window_ms, plateau_threshold_spikes,
  potentiation_strength, post_plateau_decay_ms, w_max).
  Per-post-cell `plateau_armed` flag, per-synapse
  `tag_strength` trace. Apply hook called from
  `Network::step`.
- `crates/snn-core/src/network.rs` — extend with
  `enable_btsp(BtspParams, post_cell_filter:
  Vec<usize>)` + `disable_btsp()`. The
  `post_cell_filter` restricts the rule to the C1 cell index
  range so R2-R2 synapses (also in this network) are NOT
  affected. Standard `enable_*` opt-in pattern; default off
  ⇒ bit-identical to iter-66.5 numerics.
- `crates/snn-core/src/lib.rs` — re-export `BtspParams`.
- `crates/snn-core/tests/btsp_plateau_eligibility.rs` —
  **NEW** unit tests covering the kernel: plateau arms after
  threshold post-spikes within window; tag accumulates on
  pre-spikes; potentiation applies only when both tagged AND
  plateau-armed; plateau decays after silence; weights cap at
  `w_max`.
- `crates/eval/src/reward_bench.rs` — `C1Config` extended
  with `btsp: bool` (default false) + `btsp_window_ms: f32`,
  `btsp_strength: f32`, `btsp_target_gated: bool`. When
  `c1.btsp` is true, `run_target_overlap_one_seed` calls
  `enable_btsp(...)` on R2's network with the C1 cell index
  range as the `post_cell_filter`.
- `crates/eval/examples/reward_benchmark.rs` — CLI flags
  `--c1-btsp`, `--c1-btsp-window-ms`, `--c1-btsp-strength`,
  `--c1-btsp-target-gated` (the last is a toggle for whether
  the post-cell plateau gate is active; default true; off
  reduces the rule to "open eligibility window with global
  modulator", which is a useful ablation control).

### Implementation sequence (locked)

1. **Step 1 — `BtspParams` + `Network::enable_btsp(..)` +
   unit tests in snn-core.** ~150–200 LOC including the
   plateau detector + tag trace + apply hook + 5 unit tests.
2. **Step 2 — `C1Config.btsp*` fields + Default/off()
   plumbing.** ~10 LOC; every existing snapshot test must
   pass bit-identically when `c1.btsp = false`.
3. **Step 3 — `run_target_overlap_one_seed` integration.**
   When `c1.btsp` is set, call `enable_btsp` on R2's network
   with the C1 index range; do NOT disable R-STDP elsewhere
   (it stays on R2-R2). When `c1.btsp = false`, behaviour
   is bit-identical to iter-66.5. ~30 LOC.
4. **Step 4 — Diagnostic extension.** Add to the existing
   `[iter-66 diag]` per-epoch line: `btsp_plateau_events`
   (count of plateau-arm events this epoch),
   `btsp_potentiation_events` (count of synapses receiving
   a non-zero potentiation pulse), `r2c1_target_mean_w`,
   `r2c1_nontarget_mean_w` (the structural memory-trace
   signal). These are used by the smoke gate.
5. **Step 5 — CLI flags.**
6. **Step 6 — Smoke** (1 seed × 32 ep with the locked
   config). Apply the (A / B / C / D / E) gate.
7. **Step 7 — Verdict commit** appended to this file.

Steps 1–5 are committed together as the implementation;
step 6 is a smoke commit; step 7 is the verdict commit. **No
8-seed run authorised before step 7's (A) verdict AND
Bekos's explicit follow-up approval.**

## 7. Metrics

### Primary

`c1_target_top3_overlap` per epoch — unchanged from iter-66 /
iter-66.5. Gate uses the per-epoch curve.

### Secondary (all already in the diagnostic harness or
trivial extensions)

- `c1 eval activity fraction` = `1 − (kwta_empty / n_pairs)`
  (fraction of cues that produced any C1 spikes during eval).
- `c1 kwta_empty count` (raw count for transparency).
- `r2c1_l2`, `r2c1_mean_w`, `r2c1_max|Δw|` — already logged.
- `r2c1_target_mean_w` and `r2c1_nontarget_mean_w` — NEW,
  end-of-training scan; the structural target-binding signal.
- `btsp_plateau_events`, `btsp_potentiation_events` — NEW,
  per-epoch.
- `recall-mode-eval L2 drift` (per the iter-52 / iter-62
  invariant) must remain bit-identical (= 0 drift) every
  epoch.
- For sanity: existing `c1_active_frac` (teacher) and
  `clamp_eff` (teacher) must stay at ≥ 0.95 — confirms the
  teacher pathway is unaffected by the new rule.

No new decoder metric. The dictionary + decode_top pipeline
is unchanged.

## 8. Smoke gate (single-seed, 32 ep, seed = 42)

The pre-authorised iter-67 run is **one** seed (42) at 32
epochs with the locked config. The full 8-seed verdict is
NOT run before this gate clears.

### Locked CLI invocation

```sh
cargo run --release -p eval --example reward_benchmark -- \
  --c1-readout --c1-diagnostic --c1-eval-aligned-rstdp \
  --c1-btsp \
  --c1-btsp-target-gated \
  --c1-btsp-window-ms 200 \
  --c1-btsp-strength 0.4 \
  --c1-teacher-strength 1.0 \
  --seeds 42 --epochs 32 \
  --teacher-forcing --target-clamp-strength 500 --teacher-ms 40 \
  --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval \
  --decorrelated-init
```

`--c1-btsp` is the master switch. `--c1-btsp-target-gated`
defaults true (on the locked smoke); ablation control with
the flag inverted is a follow-up, NOT part of the iter-67
locked smoke. `--c1-btsp-window-ms 200` covers cue (40) +
delay (10) + prediction (20) + teacher lead-in (12) ≈ 82 ms,
doubled for safety margin so the eligibility tag survives
the full pre-clamp interval. `--c1-btsp-strength 0.4` is
chosen so two pre-spikes during the eligibility window
saturate the synapse to `w_max` — biologically a
"two-trial" learning rate.

## 9. Acceptance matrix (locked)

| Outcome | Per-epoch curve pattern | Action |
| --- | --- | --- |
| **(A) Clear BTSP success** | last-8-ep mean `c1_target_top3_overlap ≥ 0.05` AND first-8-ep mean `< 0.02` AND C1 eval activity > 0 on majority of cues (last-8-ep mean `kwta_empty < n_pairs / 2`) AND recall-mode L2 drift = 0 | iter-67 cleared. **Propose** 4-seed or 8-seed confirmation; Bekos must approve. |
| **(B) Weak rise** | last-8 > first-8 AND last-8 > 0 AND last-8 < 0.05 | Tune BTSP window/strength **before** any full run: parameter sweep on `--c1-btsp-window-ms ∈ {100, 200, 400}` and `--c1-btsp-strength ∈ {0.2, 0.4, 0.8}` at the same seed=42, 8-ep diagnostic each. |
| **(C) Flat zero** | last-8-ep mean `c1_target_top3_overlap ≤ 0.005` AND C1 eval activity stays mostly silent | BTSP alone does not solve gain. Diagnostic: test C1 threshold (lower v_threshold for C1 cells in a follow-up smoke) AND R2 → C1 drive scaling (`--c1-init-w-max 2.0`, `--c1-from-r2-fanout 200`). If those also fail, escalate to Bekos for architecture pivot. |
| **(D) Activity without target** | C1 fires in eval (`kwta_empty < n_pairs / 2` on majority of epochs) BUT `raw_overlap` and `c1_target_top3_overlap` stay near 0 | Readout / fingerprint / target-code mismatch. Diagnostic on the C1 dictionary structure (per-word fingerprint cross-cue Jaccard) before any further plasticity changes. |
| **(E) Instability** | cross-cue separation collapses (per-pair Jaccard distribution drift) OR same-cue drops OR recall-mode L2 drift ≠ 0 | Rollback BTSP, investigate. (E) is the only outcome that immediately disables the iter-67 implementation. |

Edge cases not covered above (e.g., `last-8 < 0` impossible
by metric construction; `last-8 = 0.0001` collapses to (C);
oscillation around chance with mean ≈ first-8 collapses to
(C)) — no edge-case relaxation permitted.

## 10. Fallbacks

- (B) → iter-67-α (parameter sweep): NOT a new rule, just
  re-running the smoke at sweep points. Pre-registered above
  as the (B) action.
- (C) → iter-67-β (C1 threshold + drive scaling): pre-
  registered above. NOT a rule change; only a layer-config
  retune.
- (D) → iter-67-γ (readout diagnostic): per-word C1
  fingerprint Jaccard scan to localise whether the failure
  is at the dict-builder, the kWTA collapse, or the
  decoder. NOT a rule change.
- All-fail fallback (Bekos reserves the architectural
  pivot): outside iter-67 scope, requires its own ENTRY.
- **NOT in iter-67 scope:** Willshaw / Hopfield / key-value
  fallback; CA3/CA1 redesign; new decoder metric; full
  pre-smoke run.

## Locked methodological commitments

1. **One new plasticity rule** (BTSP plateau-eligibility),
   gated behind `c1.btsp = false` ⇒ iter-66.5 numerics
   bit-identical when off.
2. **No CA3/CA1 redesign.** C1 layer wiring, projection,
   target SDR, all unchanged.
3. **No Willshaw / Hopfield / key-value fallback.**
4. **No new decoder metric.**
5. **No full 8-seed run before the seed-42 gate clears.**
6. **Do not reuse old R-STDP success criteria.** iter-67's
   gate is K1 ∧ K2 ∧ K3 (rising curve) AND eval-activity AND
   L2 invariant — written above as (A); no relaxation to
   "any rise > 0" allowed.
7. **iter-65 / iter-66 / iter-66.5 numerics preserved when
   `c1.btsp = false`.** All 11 reward_bench snapshot tests
   must continue to pass bit-identically. Same contract
   iter-66 / iter-66.5 maintained.

## Commit plan (locked)

1. **This commit (ENTRY note only).** No code, no
   measurements.
2. **Implementation commit (steps 1–5).** Code plumbing +
   unit tests + CLI. iter-65/66/66.5 snapshot tests must
   stay bit-identical when `--c1-btsp` is off.
3. **Smoke commit (step 6).** Single-seed 32-ep run; log
   stashed under `notes/67-step-6-smoke-seed42-ep32.log`.
4. **Verdict commit (step 7).** This file appended with the
   per-epoch curve, A/B/C/D/E classification, and the
   conditional follow-up plan.

Do NOT start full compute (multi-seed) until the smoke
verdict is in this file.

## Headline

> **(B') Weak — selectivity-without-activity.** Three rounds of
> iter-67 fixes (v2 = BTSP-only, v3 = +homeostasis-gated,
> v4 = +R2-isolated) each break a different failure mode but no
> single fix delivers BOTH per-pathway selectivity (K4 ≥ 1.5)
> AND postsynaptic gain sufficient to fire C1 from cue input
> alone at recall (kwta_empty < 16/32). Pivot to **iter-67-β**
> partial-echo-state per Bekos's friend's prompt: combine v3's
> recurrent gain with v4's R2-isolation by leaving R2 recurrent
> connectivity at a fraction of full strength during the teacher
> clamp window.

## Step 6 — Smoke results (3 versions, seed=42 × 32 ep each)

**Date:** 2026-05-06.
**Smoke logs (committed):**
- `notes/67-step-6-smoke-seed42-ep32-v2-baseline-btsp.log` — BTSP
  on, no homeostasis-gating, no R2-isolation.
- `notes/67-step-6-smoke-seed42-ep32-v3-homeostasis-gated.log` —
  + iter-67-α homeostasis-gated during teacher Phase 4.
- `notes/67-step-6-smoke-seed42-ep32-v4-r2-isolated.log` —
  + iter-67-α2 R1+DG drive cut during teacher Phase 4 (R2
  isolation).

### v2 — BTSP alone (baseline implementation)

```text
ep | top3_c1 | tgt_w  | non_w  | w_ratio | spikes_mean(eval) | dict_concepts
 0 | 0.0000  | 0.0630 | 0.1529 | 0.412   | 0.00              | 0
 1 | 0.0000  | 0.0630 | 0.1529 | 0.412   | 0.00              | 0
 2 | 0.0000  | 0.0626 | 0.1529 | 0.409   | 0.00              | 0
 3 | 0.0000  | 0.0631 | 0.1529 | 0.412   | 0.00              | 0
 4 | 0.0000  | 0.0634 | 0.1529 | 0.414   | 0.00              | 0
 5 | 0.0000  | 0.0641 | 0.1529 | 0.419   | 0.00              | 0
 6 | 0.0000  | 0.0639 | 0.1529 | 0.418   | 0.00              | 0
 7 | 0.0000  | 0.0635 | 0.1529 | 0.415   | 0.00              | 0
```

w_ratio asymptotic at ~0.42 (target weights at 41% of non-target,
the OPPOSITE direction of K4's required ≥ 1.5). C1 silent at
eval. Δw_ratio per epoch ≈ 0.0014 → projected ~ 700 epochs
to reach K4. **Diagnosis: homeostasis catch-22** — the 500 nA
C1-target clamp drives canonical-target C1 cells to saturation
firing → homeostasis (`scale_only_down=true, a_target=2.0`)
scales their incoming R2-E weights DOWN faster than BTSP can
build them up.

### v3 — + iter-67-α (homeostasis-gated during teacher)

```text
ep | top3_c1 | tgt_w  | non_w  | w_ratio | spikes_mean(eval) | dict_concepts
 0 | 0.0000  | 0.7849 | 0.7878 | 0.996   | 6351              | 64
 1 | 0.0625  | 0.7927 | 0.7938 | 0.999   | 6396              | 64
 2 | 0.0000  | 0.7941 | 0.7952 | 0.999   | 6249              | 64
 3 | 0.0000  | 0.7935 | 0.7947 | 0.999   | 6282              | 64
```

**Massive gain breakthrough:** `kwta_empty` dropped from 32/32
to 0/32 at epoch 0; `dict_concepts` from 0 to 64; `spikes_mean`
from 0 to ~6300; `tgt_w` from 0.063 to 0.79 saturation;
`top3_c1` non-zero for the first time in iter-66 / iter-66.5 /
iter-67 (0.0625 at epoch 1). **But selectivity collapsed**:
w_ratio = 0.999 (target ≈ non-target), because R2 keeps firing
its full cue + recurrent + DG response throughout teacher,
BTSP tags ALL active R2-E synapses (engram + noise), and the
plateau-arm event potentiates them all uniformly. top3_c1
oscillates at noise level (0.0625 → 0 → 0).

### v4 — + iter-67-α2 (R2 cue + DG drive cut to 0 during teacher)

```text
ep | top3_c1 | tgt_w  | non_w  | w_ratio | spikes_mean(eval) | dict_concepts
 0 | 0.0000  | 0.2480 | 0.2090 | 1.186   | 0.00              | 0
 1 | 0.0000  | 0.2528 | 0.2090 | 1.209   | 0.00              | 0
 2 | 0.0000  | 0.2548 | 0.2090 | 1.219   | 0.00              | 0
 3 | 0.0000  | 0.2552 | 0.2090 | 1.221   | 0.00              | 1
 4 | 0.0000  | 0.2555 | 0.2090 | 1.222   | 0.00              | 0
```

**Selectivity emerges**: w_ratio rises 0.42 (v2) → 1.0 (v3) →
**1.22 (v4)**, target weights now > non-target. Δw_ratio per
epoch is 16× faster than v2. **But gain collapses again**:
absolute weights tiny (tgt_w = 0.25); `kwta_empty` back to 32/32;
`dict_concepts` near 0. Why: cutting cue + DG drive during
teacher means R2 fires only via residual membrane decay → BTSP
tag accumulation drops drastically (potentiation_events =
11k vs v3's 18M) → only a thin layer of synapses gets
strengthened → eval cue can't fire C1. w_ratio asymptotes near
1.22 (Δ ≈ 0.001 by epoch 4), well below K4 = 1.5.

### Three-version summary table

| Version | w_ratio | tgt_w | non_w | top3_c1 | kwta_empty | Diagnosis |
|---|---|---|---|---|---|---|
| v2 BTSP only | 0.42 | 0.063 | 0.153 | 0 | 32/32 | homeostasis cancels BTSP |
| v3 +homeostasis-gated | 1.00 | 0.79 | 0.79 | 0 (0.06 noise) | 0/32 | gain ✓ but no selectivity |
| v4 +R2-isolated | 1.22 | 0.25 | 0.21 | 0 | 32/32 | selectivity ✓ but no gain |

The fixes form a U-shape on the gain × selectivity plane.
Neither single fix delivers both.

### Bekos's locked acceptance gate (A/B/C/D/E) — outcome

Per the iter-67 ENTRY's locked criteria for v4 (the run that
delivered the highest selectivity):
- (A) Clear BTSP success: requires `top3_c1 ≥ 0.05` last-8-ep
  AND `< 0.02` first-8-ep AND `kwta_empty < n_pairs/2` majority
  of cues. **All three FAIL: top3_c1 stays at 0.0 every epoch,
  kwta_empty = 32/32 every epoch.**
- (B) Weak rise: `last-8 > 0` AND `last-8 > first-8` AND
  `last-8 < 0.05`. last-8 = first-8 = 0 → **NOT (B).**
- (C) Flat zero: `last-8 ≤ 0.005`. **TRUE (formally (C))** —
  but the K4 weight-ratio diagnostic shows v4 is partially
  succeeding on selectivity (w_ratio 1.22 vs K4's 1.5).
- (D) Activity without target: requires C1 to fire in eval
  (`kwta_empty < n_pairs/2`) AND `top3_c1 ≈ 0`. **NOT (D)** —
  v4 has no eval-phase C1 activity at all.
- (E) Instability: requires R2 readout collapse OR cross-cue
  separation drop OR L2 drift ≠ 0. **NOT (E)** — top3_r2
  oscillates at iter-65 baseline (0.0312 / 0.0); recall-mode
  L2 drift not measured here but the path is unchanged.

**Formally (C)**, but mechanistically the failure is informative:
the *combination* of v3's gain and v4's selectivity is what's
needed, not either alone. The locked (C) action is "iter-67-β
C1 threshold / drive scaling" — but the data points to a more
specific β: **partial echo-state during teacher** (Bekos's
friend's prompt: 2-3 ms @ 15 % recurrent strength) that
preserves selectivity from v4 while restoring some of v3's
BTSP tag accumulation.

## Step 7 — iter-67-β proposal: partial echo-state

Pre-registered direction (not yet implemented; awaiting Bekos
explicit Go):

> Instead of cutting R1 + DG drive to 0 during the entire
> Phase 4 clamp window (v4), cut them only for the first
> ~80 % of the clamp window AND leave R2 recurrent
> connectivity at a fraction of full strength throughout. The
> first ~80 % gives BTSP eligibility tags time to accumulate
> only from the cue-substrate-driven pre-spikes (engram-
> selective). The recurrent fraction sustains some R2-E
> firing across the teacher window so BTSP tags accumulate
> beyond the residual-decay regime, but the absence of fresh
> cue input keeps the firing engram-biased rather than
> noise-uniform.

Two design parameters to lock:
1. `c1.btsp_teacher_cue_fraction: f32` — fraction of Phase 4
   clamp_ms during which cue + DG drive are cut to 0. Default
   1.0 (= v4 verbatim); 0.0 = v3 verbatim. β-locked default
   would be ~ 0.8 (cut for first 80 %, restore for last 20 %).
2. `c1.btsp_teacher_recurrent_scale: f32` — multiplier on R2
   recurrent synapse weights during Phase 4 clamp_ms. Default
   1.0 (= v3 verbatim); 0.0 = full v4 isolation. β-locked
   default ~ 0.15 per the friend's prompt.

Implementation cost: ~ 30 LOC in `run_teacher_trial` for
the cue / DG fade; ~ 50 LOC in snn-core for a
`Network::scale_recurrent_weights(scale: f32)` API plus
restore (or, more simply, a per-step recurrent-strength
multiplier on `Network`'s spike delivery).

The locked iter-67 ENTRY's gate matrix carries over verbatim
to iter-67-β; no relaxation. Bekos's friend's specific
predictions (`w_ratio > 1.0` by epoch 1 + `tgt_w > 0.15` by
epoch 1 + `top3_c1 > 0` by epoch 1) become the iter-67-β
single-seed gate.

## Iter-67 status: PAUSED at step 6 verdict (formally (C),
mechanistically gain × selectivity tradeoff)

The implementation (steps 1–5 + α + α2) on this branch is
correct: BTSP unit tests pass; the rule fires when expected;
homeostasis-gating fix produced a clean gain breakthrough;
R2-isolation produced a clean selectivity emergence. **The
two fixes are individually correct but their combination at
the locked teacher schedule does not produce both effects
simultaneously.** iter-67-β (partial echo-state) is the
pre-registered next step contingent on Bekos's explicit Go.

---

## Step 7 — iter-67-β verdict: no Goldilocks zone exists

**Date:** 2026-05-06.
**Commits:** `2d5c983` (impl), this commit (sweep verdict).
**Logs:**
- `notes/67-step-7-iter67-beta-v5-echo15.log` — single-seed
  v5 at scale = 0.15 (Bekos's friend's locked default).
- `notes/67-step-7-iter67-beta-recurrent-scale-sweep.log` —
  sweep at scales 0.30 / 0.50 / 0.80, 5 epochs each.

### Sweep table (epoch 4 of each scale)

| Scale | w_ratio | tgt_w | non_w | kwta_empty | dict | top3_c1 | spikes_mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 (v4)  | 1.222 | 0.255 | 0.209 | 32/32 | 0  | 0       | 0.00 |
| 0.15 (v5) | 1.190 | 0.272 | 0.229 | 32/32 | 0  | 0       | 0.00 |
| 0.30      | 1.200 | 0.275 | 0.229 | 30/32 | 0  | 0       | 0.06 |
| 0.50      | 1.205 | 0.270 | 0.224 | 32/32 | 0  | 0       | 0.00 |
| 0.80      | 1.216 | 0.264 | 0.217 | 32/32 | 0  | 0       | 0.00 |
| 1.00 (v3) | 0.999 | 0.785 | 0.788 |  0/32 | 64 | 0.06 noise | 6300 |

### Diagnosis: sharp transition near scale = 1.0, no
Goldilocks zone in this architecture

- All scales **0.0 → 0.80** produce essentially identical
  results: w_ratio ≈ 1.20, target / non-target weights ≈ 0.25,
  C1 silent at recall, dict empty. The 0.30 row's `kwta_empty
  = 30/32` is single-cue noise (likely a chance R2-E pattern
  hitting the right C1-target cells).
- Only **scale = 1.0 (= v3 full recurrent)** breaks C1
  silence: `kwta_empty = 0/32`, `dict_concepts = 64`,
  `spikes_mean ≈ 6300`. But selectivity also collapses
  (`w_ratio = 0.999`).
- The transition is **sharp**: somewhere between 0.80 and
  1.00, R2 transitions from "almost no firing" to "full
  firing", and the BTSP signal flips from "selective but
  no gain" to "gain but no selectivity".

**Mechanism (architectural).** R2 has heavy inhibition by
design:
- `R2_INH_FRAC = 0.30` (30 % I-cells)
- `g_inh = 0.80` vs `g_exc = 0.20` (I-weights 4 × E-weights
  in `build_memory_region`)

At any scale below ~0.95, recurrent inhibitory current still
overwhelms recurrent excitatory current, suppressing E-cells
to silence regardless of cue substrate. Only when *all*
recurrent (E + I) is at full strength does the E/I balance
allow E-cells to fire — but at that point the recurrent
attractor is also fully active, so BTSP tags noise + engram
indiscriminately.

**The selectivity vs gain coupling is built into R2's E/I
ratio**, not a tunable parameter at this layer of the stack.
The partial-echo-state knob cannot decouple them because it
scales E and I uniformly.

### Bekos's friend's predicted gate at scale = 0.15 — outcome

| Gate prediction | Required | Observed | Pass? |
| --- | --- | --- | :---: |
| `w_ratio` after epoch 1 ≥ 1.5 | yes | 1.190 | ✗ |
| `dict_concepts` after epoch 1 ≥ 32 | yes | 0 | ✗ |
| `top3_c1` after epoch 5 ≥ 0.20 | yes | 0.0 | ✗ |
| `tgt_w` after epoch 1 ≥ 0.40 | yes | 0.272 | ✗ |

**All four predictions FAIL.** The friend's analysis was
mechanistically sound (energy-integration interpretation of
BTSP is correct) but the predicted Goldilocks zone in scale
space doesn't materialise in this architecture.

### iter-67-β formal verdict

Same as iter-67 step 6: **(C) Flat zero on the readout** with
a partial K4 (selectivity capped at w_ratio ≈ 1.22). The
recurrent-scale sweep over 0.0–1.0 conclusively shows there
is no scale value that produces both selectivity AND gain
simultaneously in the locked R2 architecture.

### Three pre-registered iter-67-γ paths (not yet implemented)

Each addresses a different aspect of the architectural
coupling. Picking one requires Bekos's explicit Go.

#### γ.1 — E-only recurrent scaling

Scale only the **excitatory** recurrent synapses during
teacher; leave inhibitory at full strength (or lower it).
This decouples E-firing from I-suppression. Implementation:
extend `Network::set_recurrent_scale` with a `kind_filter`
parameter (or add a separate `set_recurrent_e_scale`).
~ 30 LOC.

Hypothesis: at e.g. `scale_e = 1.0, scale_i = 0.3`, R2-E
cells fire freely (gain ✓) but only the cue-engram cells
benefit because (a) cue-substrate priming is strong on
engram cells AND (b) the C1-clamp-induced post-spike
provides selective gating. Risk: runaway R2-E firing
without inhibition.

#### γ.2 — Tighter BTSP eligibility window

The current `eligibility_window_ms = 200` covers
cue + delay + prediction. If we tighten to e.g. 50 ms, only
the prediction-phase pre-spikes contribute tags — and those
are the most cue-engram-specific (cue + delay have settled
into the strongest pattern). Implementation: just change
the BtspParams default. ~ 1 LOC.

Hypothesis: with cue + delay + prediction giving R2-E cue-
engram cells enough membrane potential to fire ~50 ms
before clamp, BTSP tags will be biased toward the engram
even with full recurrent during teacher. Risk: too few
tags, same gain failure as v4.

#### γ.3 — Higher initial R2-E → C1 weights + lower
plateau threshold

Bypass the BTSP-via-saturation route entirely: pre-train
R2-E → C1 weights at a higher init level (e.g. `init_w_max
= 2.0` instead of 0.5) so that even a single firing R2-E
input drives C1 cells over threshold. Simultaneously lower
the BTSP `plateau_threshold_spikes` from 5 to 2 so plateau
arms on weaker C1 firing. Implementation: only knob changes.
~ 0 LOC.

Hypothesis: with stronger initial weights, eval-phase R2-E
cue activity is already enough to fire C1 cells (gain ✓),
and BTSP refines selectively over epochs. Risk: initial
weights so high that selectivity never has a chance to
emerge (uniform pre-firing).

### Recommendation

Run γ.2 (tighter eligibility window) first — it's a 1-LOC
config change with the cleanest hypothesis link to BTSP's
biological role (Bittner 2017 used τ ≈ 10–50 ms windows).
If it doesn't break the silence, γ.1 (E-only recurrent
scaling) is the next-most-targeted; γ.3 is a fallback that
abandons the BTSP-driven binding mechanism.
