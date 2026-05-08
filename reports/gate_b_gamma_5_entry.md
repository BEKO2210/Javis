# iter-67-γ.5 — heterosynaptic competition (ENTRY, locked)

**Status:** ENTRY (pre-registration). No measurements, no code, no
implementation.  The hypothesis, mechanism design, locked seed set,
locked acceptance matrix, and analysis plan are committed in this
file *before* any γ.5 code is written.  They may not be relaxed
post-hoc.

**Entry conditioned on:** Gate-B γ.4 verdict Class (D) Reject
(`reports/gate_b_gamma_4_8seed_summary.md`, commit `ebb9ea4`).
γ.5 is the (D)-branch's locked next step per the Gate-B γ.4 ENTRY's
acceptance matrix:
> "(C) Partial — needs different rule | iter-67-γ.5 (heterosynaptic
> competition: when LTP fires on target, simultaneously LTD all
> non-target post-cells receiving the same pre-pattern). Same locked
> corpus + recall-mode discipline."

(Although γ.4 landed in (D), Bekos's locked direction is to take the
(C)-row's γ.5 spec as the next mechanism rather than escalate to
M2 Willshaw or non-CA1 pivot. This is consistent with the locked
acceptance matrix's spirit: γ.5 is the *minimum* mechanism change
that addresses γ.4's diagnosed multiplicity-asymmetry without
discarding the C1 readout architecture.)

## Why γ.5

| Iteration | Verdict | Headline |
| --- | :---: | --- |
| iter-67-γ.1.1 | Class (C) Partial | 5/8 PASS at last-8 ≥ 0.05; mean = 0.0693. C1 readout *lives* but binds via fingerprint geometry only (`tgt_w ≈ non_w ≈ 0.80`, `w_ratio ≈ 1.000` universal). 3 FAIL seeds (0, 4, 7) all DEGRADING-or-flat trajectory. |
| iter-67-γ.4 | Class (D) Reject | 0/8 PASS; mean = 0.0000 on every seed across all 32 epochs. Non-target-plateau-driven LTD destroyed C1 gain entirely (`tgt_w` collapsed to ~0.001 by epoch 0). Mechanism diagnosis: ~30 target plateau events vs hundreds of non-target plateau events per trial under γ.1.1's E/I-split → LTD population dwarfs LTP population → all weights → `w_min = 0`. |

γ.5's core design intent: **fix the multiplicity asymmetry by
coupling LTD 1:1 to target LTP events** — not to non-target plateau
events. Each successful target potentiation simultaneously depresses
the *same pre-pattern* on non-target post-cells. The trigger count
is then bounded by the target population (~30 cells), not by the
much larger non-target population.

This is the canonical Bittner 2017 / Magee & Grienberger 2020
heterosynaptic competition pairing: plateau-induced LTP on the
target field, paired LTD on neighbouring non-target fields receiving
the same input.

## Pre-registered hypothesis

> On the iter-67-γ.1.1 frozen base configuration plus γ.5
> heterosynaptic competition with `non_target_depression_strength = 0.0`
> (γ.4 disabled) and `heterosynaptic_strength > 0.0`, a paired smoke
> on one γ.1.1 PASS seed (seed 5) and one γ.1.1 FAIL seed (seed 0)
> at 32 epochs produces:
> - **the FAIL-seed crosses last-8 mean ≥ 0.05** (target lift), AND
> - **the PASS-seed retains last-8 mean ≥ 75 % of γ.1.1 baseline**
>   (non-regression preserve), AND
> - **C1 readout stays alive** (`kwta_empty < 32/32`,
>   `dict_concepts > 0` on majority of epochs), AND
> - **per-class weight-magnitude separation emerges** (`w_ratio`
>   improves over γ.1.1's universal 1.000 toward ≥ 1.05).
>
> If all four hold, γ.5 qualifies for an 8-seed Gate-B confirmation
> on the locked seed set {0..7}.

## Mechanism (locked)

γ.5 modifies the BTSP plateau-arm branch on the C1 readout layer
**without touching γ.4 code paths or γ.1.1 numerics**:

| Plateau-arm event | γ.1.1 | γ.4 | **γ.5 (this iter)** |
| --- | --- | --- | --- |
| Target post-cell crosses threshold | LTP on target's tagged incoming | LTP on target | **LTP on target's tagged incoming (UNCHANGED from γ.1.1)** + heterosynaptic LTD: for each tagged pre-cell `r` consumed by this LTP, scan `outgoing[r]` and apply `Δw = −heterosynaptic_strength × tag_h` to every synapse `r → c_n` where `c_n ≠ target_post`, `btsp_target_post[c_n] = false`, and `btsp_post_mask[c_n] = true`. Tags on heterosynaptic synapses are NOT consumed (they decay naturally per the 200 ms eligibility window). |
| Non-target post-cell crosses threshold | LTP on non-target's tagged incoming (γ.1.1's unwanted cross-talk) | LTD on non-target's tagged incoming (γ.4's failed rule) | **No-op when γ.4-strength = 0 AND γ.5 is active.** γ.5 leaves non-target plateau events alone — depression is exclusively coupled to target LTP events. |
| No plateau-arm | no-op | no-op | no-op |

### Locked design constraints (from Bekos's prompt)

1. **Do NOT reuse γ.4 mass-LTD.** `non_target_depression_strength = 0.0` for the γ.5 smoke.
2. **No non-target plateau-driven LTD.** Non-target plateau events do nothing in γ.5.
3. **No global non-target kill.** LTD only on synapses with non-zero tag for the *exact* pre-cell whose LTP just fired.
4. **LTD is coupled to real target LTP events.** Trigger count = target plateau events, not non-target plateau events.
5. **LTD is bounded.** Strength capped by `heterosynaptic_strength`; per-event scope bounded by the pre-cell's `outgoing[r]` list (R2-E → C1 fanout = 30 by locked γ.1.1 default).
6. **γ.1.1 base unchanged.** Same C1 layer, same E/I split (1.0 / 0.3), same R2-isolation OFF, same 200 ms BTSP window, same 0.4 potentiation strength.
7. **Recall-mode active.** `--plasticity-off-during-eval` stays on (locked since iter-62).
8. **DG / R2 / C1 architecture unchanged.** No structural modifications.
9. **No Willshaw, no non-CA1 pivot in γ.5.** Those are reserved for the locked γ.5 (D) fallback.

### Why this design (vs alternatives)

1. **Why couple LTD to target LTP, not to non-target plateau?**
   This is the *direct* fix for γ.4's multiplicity-asymmetry. γ.4
   triggered LTD on every non-target plateau (~hundreds per trial);
   γ.5 triggers LTD on every target LTP (~30 per trial). 10× fewer
   LTD trigger events; 1:1 coupling to the LTP event that "earned"
   the potentiation.
2. **Why not consume the heterosynaptic-LTD tag?**  Two reasons.
   (a) The same R2-E pre-cell often has live tags on multiple C1
   post-cells simultaneously; consuming on heterosynaptic-LTD would
   prematurely deplete the eligibility field for subsequent target
   plateaus on different cells in the same trial. (b) The 200 ms
   window already provides natural decay; explicit consumption would
   stack on top of decay and produce an over-bounded LTD signal.
3. **Why limit to non-target post-cells (`btsp_target_post[c_n] = false`)?**
   The target post-cell `c_t` is the one that just LTP'd; depressing
   it via the heterosynaptic path would directly cancel the LTP. The
   skip-self check is mandatory.
4. **Why limit to BTSP-mask post-cells (`btsp_post_mask[c_n] = true`)?**
   `outgoing[r]` includes R2-E → R2-E synapses (recurrent) AND
   R2-E → C1 synapses. Heterosynaptic competition is scoped to the
   C1 readout pathway only; R2-R2 weights stay under R-STDP /
   homeostasis / iSTDP control as in γ.1.1.
5. **Why not also limit to top-K most-tagged neighbours?**
   Adds complexity (sort by tag value) for marginal benefit. The
   200 ms window + tag decay already produces a *natural* top-K
   effect: only neighbours that recently received pre-spikes have
   non-zero tags. Reserving top-K cap as a γ.5-strength sweep
   parameter for a follow-up ENTRY if the smoke surfaces it as
   needed.

## Locked CLI surface (γ.5 only)

One new CLI flag added to the γ.1.1 base (γ.4's flag stays at default 0.0):

```text
--c1-btsp-heterosynaptic-strength <float>   default 0.0
```

`0.0` reproduces γ.1.1 (off-path bit-identical: `gamma5_active`
gate is `strength > 0 && target_post non-empty`, mirrors γ.4's
gate pattern).

**γ.5 ENTRY locks `0.1` as the production value** = 0.25 × the
default `--c1-btsp-strength = 0.4`. Justification:
- γ.4 with `non_target_depression_strength = 0.2` collapsed
  weights to zero in epoch 0 because of the multiplicity factor
  (~hundreds of non-target plateaus driving hundreds of LTDs per
  trial each).
- γ.5 has at most ~30 target plateaus per trial × fanout 30 to
  non-target post-cells = ~900 LTD applications per trial — vs
  γ.4's ~3 000+ × per-trial LTD applications (rough estimate from
  the ~30 000 plateau events / epoch / 256 trials = ~120 per
  trial × 30 incoming = ~3 600).
- Per-event strength 0.1 (vs γ.4's 0.2) gives γ.5 roughly
  **(900 × 0.1) / (3 600 × 0.2) ≈ 12.5 % of γ.4's per-trial LTD
  pressure**. Should be in the right zone for graded separation
  rather than collapse.
- If the smoke shows γ.5 too weak (Class B "Partial"), strength
  micro-sweep at 0.05 / 0.10 / 0.15 / 0.20 lands in a separate
  γ.5-strength-sweep ENTRY.

## Locked γ.5 paired smoke configuration

```sh
cargo run --release -p eval --example reward_benchmark -- \
  --c1-readout \
  --c1-diagnostic \
  --c1-eval-aligned-rstdp \
  --c1-btsp \
  --c1-btsp-target-gated \
  --c1-btsp-no-r2-isolation \
  --c1-btsp-window-ms 200 \
  --c1-btsp-strength 0.4 \
  --c1-btsp-non-target-depression-strength 0.0 \
  --c1-btsp-heterosynaptic-strength 0.1 \
  --c1-btsp-teacher-recurrent-e-scale 1.0 \
  --c1-btsp-teacher-recurrent-i-scale 0.3 \
  --c1-teacher-strength 1.0 \
  --seeds <SEED> \
  --epochs 32 \
  --teacher-forcing \
  --target-clamp-strength 500 \
  --teacher-ms 40 \
  --corpus-vocab 64 \
  --dg-bridge \
  --plasticity-off-during-eval \
  --decorrelated-init
```

Two new knobs added to γ.1.1: `--c1-btsp-non-target-depression-strength 0.0` (γ.4 explicitly off) + `--c1-btsp-heterosynaptic-strength 0.1` (γ.5 active). Every other parameter locked at γ.1.1 baseline.

## Locked seed set (paired smoke)

iter-67-γ.5 paired smoke uses **2 seeds**:

```text
γ.1.1 PASS seed: 5  (γ.1.1 last-8 = 0.1289, strongest PASS — most conservative non-regression test)
γ.1.1 FAIL seed: 0  (γ.1.1 last-8 = 0.0156, canonical FAIL with rich prior diagnostic)
```

Choosing the strongest γ.1.1 PASS (seed 5) maximises the H2 (preserve)
test stringency: if seed 5 stays above threshold under γ.5, the rule
is preserving binding cleanly even on the cleanest fingerprint
geometry. Choosing seed 0 as FAIL maintains continuity with the
existing seed-0 diagnostic (Gate-A 4-seed summary + Gate-B 8-seed
diagnostic both already characterise it).

The 8-seed Gate-B set {0..7} is reserved for the conditional
γ.5-confirmation run (see "Conditional 8-seed continuation" below);
it is **not** launched without an explicit Bekos Go after the smoke
verdict.

## Locked acceptance matrix (verbatim from Bekos's prompt)

| Class | Per-seed pattern | Aggregate / criteria | iter-67 next step on γ.5 paired smoke verdict |
| --- | --- | --- | --- |
| **(A) Rescue without regression** | seed 0 last-8 ≥ 0.05 AND seed 5 last-8 ≥ 0.75 × γ.1.1 (= 0.0967) | C1 active (`kwta_empty < 32/32`, `dict_concepts > 0` on majority of epochs); `w_ratio` improves over γ.1.1's 1.000 (target ≥ 1.05); no recall / separation collapse | γ.5 qualifies for 8-seed Gate-B confirmation on locked set {0..7} (separate ENTRY pre-registers the 8-seed run with full A/B/C/D acceptance matrix) |
| **(B) Partial** | seed 0 last-8 > γ.1.1 baseline (0.0156) AND seed 5 last-8 < 0.0967 (regression < 25 %) OR `w_ratio` improves but `top3_c1` borderline | mixed | γ.5 strength micro-sweep ONLY after a fresh γ.5-strength-sweep ENTRY (sweep `--c1-btsp-heterosynaptic-strength` over `{0.05, 0.10, 0.15, 0.20}` at the same paired-smoke seeds × 32 ep); no 8-seed run before sweep verdict |
| **(C) Collapse** | seed 0 OR seed 5 `top3_c1 = 0` for majority of epochs OR `kwta_empty = 32/32` majority OR weights near zero | mean(last-8) < 0.005 on either seed | γ.5 falsified by mechanism (same failure mode as γ.4; the heterosynaptic coupling did not bound LTD enough). Locked next step: **M2 Willshaw baseline** ENTRY |
| **(D) No effect** | seed 0 last-8 ≈ γ.1.1 (≤ 0.020 lift) AND seed 5 last-8 ≈ γ.1.1 (within ±10 %) | `w_ratio` stays at 1.000 | γ.5 mechanism inactive at strength 0.1; either the implementation is wrong OR the heterosynaptic coupling produces no measurable selectivity at this strength. Locked next step: **M2 Willshaw baseline** ENTRY (do NOT raise strength without falling through to the (C) failure mode first) |
| **(E) Instability** | R2 readout collapse (top3_r2 < 0.005 majority of epochs on either seed), same-cue invariant violated, OR eval L2 drift ≠ 0 | safety-rail tripped | rollback the γ.5 implementation; do not commit; investigate before any further compute |

Edge cases:
- seed 0 lift but seed 5 catastrophic regression (>50 %): collapses
  to (B) Partial — the strength is too aggressive but the mechanism
  works qualitatively; sweep at lower strength.
- seed 5 preserves but seed 0 stays at γ.1.1 baseline: collapses to
  (D) No effect on the FAIL-seed-rescue dimension; the locked next
  step is M2 (γ.5 cannot rescue the geometrically-marginal seeds).
- Both seeds drop below γ.1.1 with weights collapsed to ~zero:
  unambiguous (C).

## Locked methodological commitments

1. **No code before this ENTRY commits.** Implementation only after
   Bekos's explicit Go on the ENTRY.
2. **No hyperparameter tuning** between γ.4 and γ.5 base. Only
   `--c1-btsp-heterosynaptic-strength 0.1` added; γ.1.1 base
   verbatim; γ.4 flag explicitly set to 0.0.
3. **No threshold changes** to last-8 mean ≥ 0.05.
4. **No goalpost shifts.** The smoke acceptance matrix above is
   immutable post-commit.
5. **No 8-seed run before paired-smoke (A) verdict.** Locked.
6. **iter-65 / iter-66 / iter-66.5 / γ.1.1 / γ.4-off-path numerics
   preserved when `c1.btsp_heterosynaptic_strength = 0.0`**:
   snapshot tests must stay at 11/11 PASS; BTSP plateau-eligibility
   tests must stay at 6/6 PASS. Off-path bit-identity verified
   *before* the smoke is launched.
7. **Same evaluator** (`scripts/evaluate_gate_a.py`) for every seed.
   Cross-platform: PowerShell / UTF-16 LE auto-decoded by
   `scripts/clean_powershell_log.py`.
8. **Determinism contract**: γ.5's heterosynaptic-LTD path is
   deterministic per `--seeds N`. Sub-permille drift on
   weight-magnitude counters across platforms is acceptable; verdict
   depends only on integer kWTA hits.
9. **Active-hold scope** (CLAUDE.md §6): when the paired smoke is
   running on Bekos's PC, no `crates/` touches, no Cargo.toml /
   .gitignore / CI changes, no README badge bumps.
10. **Per-cue trajectory split** (first-half / second-half) reported
    on every smoke seed alongside last-8 mean (per the Gate-B γ.1.1
    diagnostic discipline).

## Locked verbotene Änderungen

- ❌ Re-introduction of γ.4 mass-LTD on non-target plateaus (would
  re-create the multiplicity asymmetry).
- ❌ Heterosynaptic LTD with tag consumption (would deplete the
  eligibility field too fast).
- ❌ Heterosynaptic LTD on R2-R2 synapses (must be scoped to BTSP
  mask only).
- ❌ Heterosynaptic LTD on the target post-cell itself (must skip-self).
- ❌ Architecture-level change to C1 layer / DG / R2 wiring.
- ❌ Threshold relaxation under any circumstance.
- ❌ Cross-platform-only smoke (must be one platform; verdict
  depends only on integer kWTA hits which are bit-identical).
- ❌ 8-seed run before (A) verdict on the paired smoke.

## Locked CLI surface for the paired smoke (Windows PC)

PowerShell runner script for Bekos's machine:

```powershell
mkdir reports\runs -ErrorAction SilentlyContinue
foreach ($SEED in 0, 5) {
  Write-Host "=== START gamma5 smoke seed=$SEED $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ') ==="
  cargo run --release -p eval --example reward_benchmark -- `
    --c1-readout --c1-diagnostic --c1-eval-aligned-rstdp `
    --c1-btsp --c1-btsp-target-gated --c1-btsp-no-r2-isolation `
    --c1-btsp-window-ms 200 --c1-btsp-strength 0.4 `
    --c1-btsp-non-target-depression-strength 0.0 `
    --c1-btsp-heterosynaptic-strength 0.1 `
    --c1-btsp-teacher-recurrent-e-scale 1.0 `
    --c1-btsp-teacher-recurrent-i-scale 0.3 `
    --c1-teacher-strength 1.0 `
    --seeds $SEED --epochs 32 `
    --teacher-forcing --target-clamp-strength 500 --teacher-ms 40 `
    --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval `
    --decorrelated-init `
    > "reports\runs\gate_b_gamma_5_smoke_seed$SEED.log" 2>&1
  Write-Host "=== DONE  gamma5 smoke seed=$SEED $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ') ==="
}
```

Sequential ≈ 3–4 h on a single core (2 seeds × 32 epochs).

After completion:

```powershell
git add reports\runs\gate_b_gamma_5_smoke_seed0.log
git add reports\runs\gate_b_gamma_5_smoke_seed5.log
git commit -m "exp: Gate-B gamma.5 paired smoke complete on Windows PC"
git pull --rebase origin main
git push origin main
```

## Locked analysis plan (post-smoke deliverables)

1. Run `scripts/evaluate_gate_a.py` on both logs, produce
   `reports/runs/gate_b_gamma_5_smoke_seed{0,5}.verdict.json`.
2. Per-seed metric extraction (paralleling the Gate-B γ.1.1 + γ.4
   diagnostic): last-8 mean, 32-ep mean, longest contig.-zero,
   first-half / second-half top3_c1, raw_overlap, C1 spikes
   (teacher), `tgt_w` / `non_w` / `w_ratio`, kwta_empty,
   dict_concepts, plateau_events, R2 top3 sanity.
3. **NEW γ.5-specific diagnostic counters** (require the γ.5
   implementation to expose them in the per-epoch
   `[iter-66 diag]` line — locked as part of the implementation
   contract before the smoke runs):
   - `heterosynaptic_ltd_events` — number of non-target synapses
     receiving γ.5 LTD per epoch.
   - `heterosynaptic_ltd_per_ltp` — average ratio (LTD events /
     LTP events) per epoch — should converge near the R2-E → C1
     fanout (30) minus 1 (since target self is skipped).
4. Compare against γ.1.1 seed-5 and seed-0 baselines (already in
   `reports/runs/gate_a_gamma_1_1_seed{0,5}.verdict.json`).
5. Apply the locked acceptance matrix above; emit class
   (A / B / C / D / E).
6. Write `reports/gate_b_gamma_5_smoke_summary.md` with:
   - Per-seed verdict table (γ.5 alongside γ.1.1 baseline).
   - Per-cue top3_c1 trajectory for both seeds.
   - H1-H4 / acceptance-criterion check matrix.
   - Mechanism check (heterosynaptic_ltd_events count + per-LTP
     ratio).
   - Trajectory check (first-half vs second-half on the FAIL seed).
   - Honest limitations.
7. Lock the verdict in a git commit
   `exp: Gate-B gamma.5 paired smoke verdict — class (A/B/C/D/E)`
   and push.

## Implementation roadmap (deferred — code only after this ENTRY commit + Bekos's Go)

The implementation lands in **its own commit**, named
`iter-67-γ.5: heterosynaptic competition implementation`, BEFORE the
smoke is launched:

- `crates/snn-core/src/btsp.rs` — extend `BtspParams` with
  `heterosynaptic_strength: f32` (default `0.0`).
- `crates/snn-core/src/network.rs` — extend the BTSP plateau-arm
  branch's LTP path: after consuming a tag for an LTP event on a
  target post-cell, scan `outgoing[pre]` and apply LTD to non-target
  post-cells per the locked design above. Add
  `btsp_heterosynaptic_ltd_events: u64` diagnostic counter.
- `crates/eval/src/reward_bench.rs` — extend `C1Config` with
  `btsp_heterosynaptic_strength`. Wire through to `BtspParams` at
  `enable_btsp` site. Extend the `[iter-66 diag]` line to print
  `btsp_heterosynaptic_ltd_events`.
- `crates/eval/examples/reward_benchmark.rs` — add CLI flag
  `--c1-btsp-heterosynaptic-strength <float>`.
- `crates/snn-core/tests/btsp_plateau_eligibility.rs` — extend
  with γ.5-specific tests:
  1. `gamma5_off_path_is_bit_identical` — when
     `heterosynaptic_strength = 0.0`, every BTSP event is
     bit-identical to γ.1.1 (re-uses the existing
     `btsp_off_path_is_bit_identical` pattern).
  2. `gamma5_ltp_couples_to_heterosynaptic_ltd` — single target LTP
     event on a synthetic 3-post-cell network; verify exactly the
     non-target synapses with non-zero tag receive LTD, the target
     synapse stays at LTP value, and tags on heterosynaptic
     synapses are NOT consumed.
  3. `gamma5_skips_self_post_cell` — verify the target post-cell
     does not receive LTD on the same synapse it just LTP'd.
  4. `gamma5_skips_btsp_inactive_post_cells` — verify R2-R2
     synapses (post not in `btsp_post_mask`) are not affected.
- Verify off-path bit-identity: 6/6 BTSP tests + 11/11 eval tests
  PASS unchanged at `heterosynaptic_strength = 0.0`.
- CI gates green locally (fmt, clippy, doc) before the smoke runs.

The implementation commit MUST cite this ENTRY commit in its message
and reference the smoke-runner CLI exactly as locked above.

## Conditional 8-seed continuation (locked, NOT launched without (A))

**Only if the paired smoke verdict is Class (A) — Rescue without
regression** — the 8-seed Gate-B γ.5 confirmation runs at the
same locked seed set as Gate-B γ.1.1 / γ.4: `{0, 1, 2, 3, 4, 5, 6, 7}`.
That run lands in its own ENTRY (`reports/gate_b_gamma_5_entry.md`
becomes the *paired smoke* pre-registration; the 8-seed
confirmation gets a fresh `reports/gate_b_gamma_5_8seed_entry.md`
with the full A/B/C/D acceptance matrix + analysis plan).

Wallclock: ~13 h sequential on Bekos's Windows PC (matching γ.4's
empirical ~95 min/seed × 8 seeds), or ~3–4 h parallel 4 PowerShell
windows.

## What this commit is NOT

- Not a measurement.  No γ.5 data.
- Not a γ.5 verdict.  Only the paired-smoke pre-registration +
  conditional-8-seed-continuation policy.
- Not the implementation.  Code lands in its own commit AFTER this
  ENTRY ships AND Bekos confirms.
- Not a strength sweep.  `--c1-btsp-heterosynaptic-strength = 0.1`
  is locked. Sweep is a (B)-Partial fallback in a separate ENTRY.
- Not a γ.4 retune.  γ.4 is dead (Class (D) Reject); its flag
  remains in the binary at default `0.0` for off-path bit-identity
  verification.
- Not an architecture change.  No M2 Willshaw, no non-CA1 pivot.
- Not a goalpost shift on γ.1.1.  γ.1.1's Class (C) verdict stands;
  γ.5 must beat it on at least the FAIL-seed dimension to claim (A).

## Headline (placeholder)

> *to be filled after the paired smoke; one of:*
> - **(A) Rescue without regression — γ.5 qualifies for 8-seed
>   Gate-B confirmation.**
> - **(B) Partial — γ.5 strength micro-sweep ENTRY.**
> - **(C) Collapse — γ.5 falsified, M2 Willshaw baseline next.**
> - **(D) No effect — γ.5 inactive at strength 0.1, M2 Willshaw
>   baseline next.**
> - **(E) Instability — rollback, no commit, investigate.**
