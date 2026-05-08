# iter-67-γ.4 — per-post target-gating with non-target depression (ENTRY, locked)

**Status:** ENTRY (pre-registration). No measurements, no
verdict.  The hypothesis, mechanism design, locked seed set,
locked acceptance matrix, and analysis plan are committed in
this file *before* any γ.4 compute.  They may not be relaxed
post-hoc.

**Entry conditioned on:** Gate-B γ.1.1 verdict Class (C)
Partial (`reports/gate_b_gamma_1_1_8seed_summary.md`,
commit `6656888`).  γ.4 is the (C) branch's pre-registered
fallback per `reports/gate_b_gamma_1_1_entry.md`.

## Why γ.4

γ.1.1 produces a multi-seed-confirmed C1-target binding signal
(5/8 PASS, mean last-8 = 0.0693) but binds purely via *kWTA
fingerprint geometry*, not via per-class weight magnitude
separation:

- All 8 seeds settle at `tgt_w ≈ non_w ≈ 0.80` (w_max).
- `w_ratio = tgt_w / non_w ≈ 1.000 ± 0.0001` universally.
- K4 strict criterion (≥ 1.5) unmet on every seed.

The 3 FAIL seeds (0, 4, 7) all show DEGRADING-or-flat per-cue
top3_c1 trajectories.  Training-side metrics are bit-identical
across all 8 seeds (< 0.4 % spread).  The failure mode is
purely eval-phase fingerprint discrimination: different cues'
kWTA top-K sets collide enough on the failing seeds that the
decoder ranks the wrong cue higher.

γ.4's core hypothesis: **adding explicit non-target depression
to the BTSP rule introduces per-class weight-magnitude
separation that the kWTA can exploit on the geometrically-
marginal seeds, without breaking the seeds that already pass
via fingerprint geometry.**

## Mechanism (locked)

γ.4 modifies the BTSP plateau-arm branch in
`crates/snn-core/src/network.rs` at the C1 readout layer:

| Plateau-arm event | γ.1.1 behavior | γ.4 behavior |
| --- | --- | --- |
| Target post-cell crosses threshold | LTP: `Δw = +strength × tag` on every tagged incoming synapse | **same** (LTP, unchanged) |
| Non-target post-cell crosses threshold | LTP: `Δw = +strength × tag` (same as target — no distinction made) | **LTD: `Δw = −depression_strength × tag`** on every tagged incoming synapse |
| No plateau-arm (post-cell sub-threshold) | no-op | **same** (no-op) |

Implementation is gated on `BtspParams::non_target_depression_strength > 0.0`:

- `non_target_depression_strength = 0.0` ⇒ γ.4 path inactive.
  The plateau-arm hot loop is bit-identical to γ.1.1
  (verified by `btsp_off_path_is_bit_identical` and the 6/6
  `btsp_plateau_eligibility` tests).
- `non_target_depression_strength > 0.0` ⇒ the eval harness
  populates `Network::btsp_target_post` from `c1_target_sdr`
  before each teacher Phase 4 drive (and clears it after).
  The plateau-arm branch then reads `btsp_target_post[src]`
  to decide LTP vs LTD.

`Δw` is clamped to `[w_min, w_max] = [0.0, 0.8]` so weights
cannot go negative.  The eligibility tag is consumed
(`tag = 0`) on both LTP and LTD to preserve the one-shot
semantics: a non-target plateau cannot keep depressing the
same synapse indefinitely without fresh pre-spikes.

### Why this design (vs alternatives)

1. **Why only non-target post-cells, not non-target
   pre-cells?**  The kWTA decoder reads C1 post-cell spike
   patterns; per-post weight separation is what it can
   exploit.  Per-pre depression would muddle the engram
   pre-pattern.
2. **Why LTD on plateau-arm and not continuous LTD?**
   Continuous LTD on every spike would drive non-target
   weights to `w_min` immediately; we want graded separation,
   not annihilation.  Plateau-arm is the rate-limiter:
   non-target cells fire fewer plateaus than target cells
   (the target clamp guarantees target plateau-arms; non-
   targets only plateau under reduced inhibition at γ.1.1's
   E/I-split scales 1.0/0.3).
3. **Why consume the tag on LTD?**  Symmetry with LTP:
   one-shot per pre-spike-window.  Without consumption, a
   non-target cell that re-arms during the eligibility-window
   would re-depress the same synapses, racing toward `w_min`
   on every refresh.
4. **Why target_gated remains true?**  γ.4 is per-post
   credit assignment by definition.  The non-locality
   ablation (`target_gated = false`) is preserved in code
   for future control runs but not used in γ.4.

## Locked CLI surface

```text
--c1-btsp-non-target-depression-strength <float>   default 0.0
```

`0.0` reproduces γ.1.1 (off-path bit-identical).  γ.4 ENTRY
locks `0.2` as the production value (= 0.5 × the default
`--c1-btsp-strength = 0.4`):
- A single tagged pre-spike on a non-target plateau drops the
  synapse from 0.80 toward 0.60 (Δ = −0.2, before clamp).
- A doubly-tagged pre-spike drops it 0.4 → reaches `w_min`
  if it had been at 0.4.
- This is the same magnitude the LTP rule applies in the
  opposite direction at `--c1-btsp-strength = 0.4`, so γ.4
  is symmetric LTP/LTD around tag.

## Locked γ.4 configuration

Identical to the locked γ.1.1 config
(`reports/gate_a_gamma_1_1_config.json`) plus the new flag:

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
  --c1-btsp-non-target-depression-strength 0.2 \
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

Only ONE knob added to γ.1.1.  All other parameters locked.

## Locked seed set (no relaxation)

iter-67-γ.4 uses the **same 8 seeds** as Gate-B γ.1.1:

```text
locked seed set: 0, 1, 2, 3, 4, 5, 6, 7
```

Reusing the γ.1.1 seed set is mandatory for direct
seed-by-seed comparison: did γ.4 lift the failing seeds
(0, 4, 7) without breaking the passing seeds (1, 2, 3, 5, 6)?

## Pre-registered hypotheses (locked)

- **H1 (lift):** seeds 0, 4, 7 (γ.1.1 FAIL) cross the
  Gate-A threshold (last-8 mean ≥ 0.05) under γ.4.
- **H2 (preserve):** seeds 1, 2, 3, 5, 6 (γ.1.1 PASS) remain
  above the threshold under γ.4 (last-8 mean ≥ 0.05; no
  catastrophic loss > 50 % of γ.1.1's last-8 mean).
- **H3 (separation):** γ.4 produces measurable per-class
  weight-magnitude separation: `w_ratio = tgt_w / non_w >
  1.05` on at least 5/8 seeds (vs γ.1.1's 1.000 universal).
- **H4 (trajectory):** γ.4 eliminates the DEGRADING
  trajectory pattern on at least 2/3 of the previously-
  failing seeds (first-half ≥ second-half mean does not
  recur on those seeds).

H1 is the *primary* hypothesis (Gate-B-equivalent question).
H3 is the *mechanism* hypothesis (does the rule do what we
think it does?).  H2 is the *non-regression* guard.  H4 is
the *failure-mode-resolution* indicator.

## Locked acceptance matrix (Gate-B γ.4)

Same matrix as Gate-B γ.1.1, applied to the γ.4 8-seed run:

| Class | Per-seed pattern | Aggregate | iter-67 next step on Gate-B γ.4 verdict |
| --- | --- | --- | --- |
| **(A) Confirm — strong** | last-8 mean ≥ 0.05 on **8/8** seeds AND `t(7) > 2.5` | mean(last-8) > 0.07 | iter-67 LOCKED at γ.4.  Move to iter-68 anatomy refinement. |
| **(B) Robust — directional** | last-8 mean ≥ 0.05 AND `n_pos ≥ 7/8` AND `t(7) > 1.895` | mean(last-8) ≥ 0.05 | iter-67 LOCKED at γ.4 with caveat.  Optional 16-seed extension; otherwise lock and move on. |
| **(C) Partial — needs different rule** | last-8 mean ≥ 0.05 on `5/8 ≤ n_pos < 7/8` seeds | `t(7) > 0` | iter-67-γ.5 (heterosynaptic competition: when LTP fires on target, simultaneously LTD all non-target post-cells receiving the same pre-pattern).  Same locked corpus + recall-mode discipline. |
| **(D) Reject — γ.4 mechanism wrong** | last-8 mean ≥ 0.05 on `n_pos ≤ 4/8` OR mean(last-8) ≤ 0.0 | (chance) | γ.4 mechanism insufficient.  Escalate to Bekos for direction (γ.5 vs Mechanism M2 Willshaw baseline vs pivot to non-CA1 readout).  No further parameter sweeps without explicit Go. |

Edge cases (verbatim from Gate-B γ.1.1):
- `n_pos = 4/8` with mean(last-8) > 0 collapses to (D).
- `n_pos = 7/8` with `t(7) ≤ 1.895` collapses to (C).
- The t-stat baseline is the γ.1.1 8-seed mean (0.0693), NOT
  zero.  γ.4 must beat γ.1.1 to count as Class A or B; equal-
  to-γ.1.1 collapses to (C) (the rule didn't help).

## Locked methodological commitments

1. **No hyperparameter tuning** between Gate-B γ.1.1 and
   Gate-B γ.4.  Only `--c1-btsp-non-target-depression-strength
   0.2` added; every other knob is verbatim γ.1.1.
2. **No threshold changes** to last-8 mean ≥ 0.05.
3. **No cherry-picking**: failed seeds reported alongside
   passing seeds; same 8 seeds as γ.1.1 (no dropping seed 0
   even though it failed under γ.1.1).
4. **No mechanism redesign before Gate-B γ.4 verdict**: γ.5
   spec is conditional on (C) and lands in its own ENTRY only
   after (C) is observed.
5. **iter-65 / iter-66 / iter-66.5 numerics preserved when
   `c1.btsp = false` AND when `non_target_depression_strength
   = 0.0`**: snapshot tests must stay at 11/11 PASS
   (verified at commit time: 6/6 BTSP tests PASS, 11/11 eval
   tests PASS).
6. **Same evaluator** (`scripts/evaluate_gate_a.py`) for every
   seed.  Cross-platform: PowerShell / UTF-16 LE auto-decoded
   by `scripts/clean_powershell_log.py`.
7. **Determinism contract**: γ.4's BTSP rule plus per-step
   target-post mask are deterministic per `--seeds N`.  Sub-
   permille drift on weight-magnitude counters across
   platforms is acceptable; verdict depends only on integer
   kWTA hits.

## Implementation summary (already committed in this branch)

- `crates/snn-core/src/btsp.rs` —
  `BtspParams::non_target_depression_strength: f32` (default 0.0).
- `crates/snn-core/src/network.rs` —
  - `Network::btsp_target_post: Vec<bool>` (lazy-allocated).
  - `Network::set_btsp_target_post(&[usize])`.
  - `Network::clear_btsp_target_post()`.
  - Plateau-arm branch: when `non_target_depression_strength
    > 0` AND mask non-empty, branch LTP/LTD on
    `btsp_target_post[src]`.
  - `Network::btsp_depression_events: u64` diagnostic counter.
- `crates/eval/src/reward_bench.rs` —
  - `C1Config::btsp_non_target_depression_strength: f32`
    (default 0.0).
  - Wired through to `BtspParams` at `enable_btsp` site.
  - In `run_teacher_trial`: when
    `c1.btsp_non_target_depression_strength > 0`, call
    `set_btsp_target_post(c1_target_sdr_as_usize)` immediately
    before the teacher Phase 4 drive; call
    `clear_btsp_target_post()` immediately after.
- `crates/eval/examples/reward_benchmark.rs` — CLI flag
  `--c1-btsp-non-target-depression-strength <float>`
  (default 0.0).

Snapshot tests pinning iter-65 / iter-66 / iter-66.5 numerics
all PASS at commit time (11/11 eval tests, 6/6 BTSP tests).

## Locked CLI surface (for Bekos's Windows PC)

Sequential 8-seed runner script:

```powershell
mkdir reports\runs -ErrorAction SilentlyContinue
foreach ($SEED in 0, 1, 2, 3, 4, 5, 6, 7) {
  Write-Host "=== START gamma4 seed=$SEED $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ') ==="
  cargo run --release -p eval --example reward_benchmark -- `
    --c1-readout --c1-diagnostic --c1-eval-aligned-rstdp `
    --c1-btsp --c1-btsp-target-gated --c1-btsp-no-r2-isolation `
    --c1-btsp-window-ms 200 --c1-btsp-strength 0.4 `
    --c1-btsp-non-target-depression-strength 0.2 `
    --c1-btsp-teacher-recurrent-e-scale 1.0 `
    --c1-btsp-teacher-recurrent-i-scale 0.3 `
    --c1-teacher-strength 1.0 `
    --seeds $SEED --epochs 32 `
    --teacher-forcing --target-clamp-strength 500 --teacher-ms 40 `
    --corpus-vocab 64 --dg-bridge --plasticity-off-during-eval `
    --decorrelated-init `
    > "reports\runs\gate_b_gamma_4_seed$SEED.log" 2>&1
  Write-Host "=== DONE  gamma4 seed=$SEED $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ') ==="
}
```

(Sequential ≈ 18–20 h on a single core; parallel 4 PowerShell
windows ≈ 5–6 h on a 4-core CPU with ~8 GB RAM available.)

After completion:

```powershell
git add reports\runs\gate_b_gamma_4_seed0.log
git add reports\runs\gate_b_gamma_4_seed1.log
git add reports\runs\gate_b_gamma_4_seed2.log
git add reports\runs\gate_b_gamma_4_seed3.log
git add reports\runs\gate_b_gamma_4_seed4.log
git add reports\runs\gate_b_gamma_4_seed5.log
git add reports\runs\gate_b_gamma_4_seed6.log
git add reports\runs\gate_b_gamma_4_seed7.log
git commit -m "exp: Gate-B gamma.4 8-seed run complete on Windows PC"
git push origin main
```

## Locked analysis plan (Step 7 deliverables, post-run)

1. Run `scripts/evaluate_gate_a.py` on all 8 logs, produce
   `reports/runs/gate_b_gamma_4_seed{0..7}.verdict.json`.
2. Aggregate the 8 last-8 means.  Apply the locked acceptance
   matrix above; emit class (A / B / C / D).
3. Compute paired-t(7) against γ.1.1's 8-seed mean (0.0693)
   to test whether γ.4 lifted the distribution.
4. Per-seed comparison:
   `delta_seed[i] = γ.4_seed[i].last8 − γ.1.1_seed[i].last8`.
   - Failing-seed lift: how many of {0, 4, 7} crossed 0.05?
   - Passing-seed regression: any of {1, 2, 3, 5, 6} dropped
     below 0.05? (Class C / D auto-trigger.)
5. Mechanism check: per-seed `w_ratio` and `tgt_w` − `non_w`.
   Is the rule producing the magnitude separation it was
   designed to produce?  H3 acceptance.
6. Trajectory check: per-seed first-half vs second-half
   top3_c1.  H4 acceptance: did γ.4 eliminate the DEGRADING
   pattern on the previously-failing seeds?
7. Diagnostic counters: `btsp_depression_events` per seed.
   Confirms the LTD path actually fired (sanity check, not a
   verdict).
8. Write `reports/gate_b_gamma_4_8seed_summary.md` with:
   - 8-seed verdict table (γ.4 alongside γ.1.1 baseline).
   - Aggregate stats + paired-t.
   - Per-seed lift / regression matrix.
   - Mechanism check (H3 weight separation).
   - Trajectory check (H4 first-half/second-half).
   - Honest limitations.
9. Lock the verdict in a git commit "exp: Gate-B gamma.4
   8-seed verdict — class (A/B/C/D)".  No more compute on
   γ.4 after that without a fresh ENTRY.

## What this commit is NOT

- Not a measurement.  No new γ.4 data.
- Not a Gate-B γ.4 verdict.  Only the pre-registration +
  implementation.
- Not γ.5 spec.  γ.5 is conditional on Gate-B γ.4 (C)
  verdict; spec lands in its own ENTRY only after (C) is
  observed.
- Not a parameter sweep.  `--c1-btsp-non-target-depression-
  strength` is locked at 0.2; sweeps require a separate
  ENTRY.

## Headline (placeholder)

> *to be filled after the 8-seed γ.4 run; one of:*
> - **(A) Confirm — strong; iter-67 LOCKED at γ.4.**
> - **(B) Robust — directional; iter-67 LOCKED at γ.4 with
>   caveat.  Optional 16-seed tightening.**
> - **(C) Partial — iter-67-γ.5 (heterosynaptic
>   competition).**
> - **(D) Reject — γ.4 mechanism insufficient; escalate.**
