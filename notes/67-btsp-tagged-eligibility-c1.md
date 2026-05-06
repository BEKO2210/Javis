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

## Headline (placeholder)

> *to be filled after the seed-42 32-ep smoke; one of:*
> - **(A) Clear BTSP success — propose multi-seed
>   confirmation.**
> - **(B) Weak rise — iter-67-α parameter sweep.**
> - **(C) Flat zero — iter-67-β C1 threshold/drive scaling
>   diagnostic.**
> - **(D) Activity without target — iter-67-γ readout
>   diagnostic.**
> - **(E) Instability — rollback and investigate.**
