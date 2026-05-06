# γ.1.1 frozen Gate-A config

**Frozen at:** 2026-05-06T18:35:00Z (after the seed=42 candidate
run reached the lower-bound PASS verdict at ep 24–30).
**Git commit at lock:** `ab9ae16`
**Branch:** `claude/iter66-ca1-heteroassoc-readout`
**Machine-readable form:** `reports/gate_a_gamma_1_1_config.json`

## Locked CLI invocation

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

`<SEED>` is the only varying parameter for the 4-seed
confirmation: `0`, `1`, `2`, `3`.  All other parameters are
locked at the values above; no override is permitted in any
seed-confirmation run.

## What each flag does

| Flag | Purpose |
| --- | --- |
| `--c1-readout` | Enable the C1 readout layer (iter-66 architecture). |
| `--c1-diagnostic` | Emit per-epoch `[iter-66 diag]` line with the auxiliary metrics. |
| `--c1-eval-aligned-rstdp` | iter-66.5 fix: drop the canonical R2 target SDR from the teacher Phase 4 clamp so R-STDP trains on actual cue-driven R2 patterns. |
| `--c1-btsp` | Enable the iter-67 BTSP plateau-eligibility kernel on R2-E → C1 synapses. |
| `--c1-btsp-target-gated` | iter-67 default-on intent flag for per-post-cell credit assignment (only the C1 cell that crossed plateau receives retroactive potentiation). |
| `--c1-btsp-no-r2-isolation` | iter-67-γ.1.1 opt-out: keep cue + DG drive at full strength during teacher Phase 4 (DON'T cut to 0 as iter-67-α2 would).  Tests Bekos's locked γ.1 hypothesis where cue-engram E-cells fire under reduced inhibition. |
| `--c1-btsp-window-ms 200` | BTSP eligibility-tag decay constant.  Window-sweep `100/150/250` showed all converge to identical asymptote; 200 ms is the iter-67 ENTRY locked default. |
| `--c1-btsp-strength 0.4` | Per-tagged-pre-spike weight increment at the plateau-arm transition.  iter-67 ENTRY locked default. |
| `--c1-btsp-teacher-recurrent-e-scale 1.0` | iter-67-γ.1: keep R2-R2 excitatory recurrent at full strength during teacher (no scaling). |
| `--c1-btsp-teacher-recurrent-i-scale 0.3` | iter-67-γ.1: scale R2-R2 inhibitory recurrent to 30 % during teacher (the locked γ.1 prompt's E/I imbalance knob). |
| `--c1-teacher-strength 1.0` | M_target neuromodulator pulse strength during teacher Phase 4 (iter-66 ENTRY default). |
| `--epochs 32` | Run length matches iter-66.5 / iter-67 ENTRY.  Last-8 = ep 24–31. |
| `--teacher-forcing` | iter-46 six-phase trial schedule. |
| `--target-clamp-strength 500` | C1 target SDR clamp current in nA. |
| `--teacher-ms 40` | Teacher Phase 4 total length (12 ms lead-in + 28 ms clamp). |
| `--corpus-vocab 64` | iter-58/65/66 vocab size. |
| `--dg-bridge` | DG pattern-separation region (iter-60 architecture, retained in iter-66+). |
| `--plasticity-off-during-eval` | iter-62 recall-mode invariant: every plasticity rule disabled between training and the per-epoch eval phase, so the readout dictionary builds + per-pair eval are deterministic given the seed. |
| `--decorrelated-init` | iter-54 decorrelated R1 → R2 wiring (each cue's R1 SDR projects only into a disjoint block of R2-E cells). |

## Numerical locks (compiled-in)

These are NOT exposed as CLI flags but are part of the locked
γ.1.1 configuration via the source code at commit `ab9ae16`:

| Parameter | Value | Source |
| --- | --- | --- |
| BTSP `plateau_window_ms` | 30.0 | `crates/snn-core/src/btsp.rs` `BtspParams::default` |
| BTSP `plateau_threshold_spikes` | 5.0 | same |
| BTSP `post_plateau_decay_ms` | 50.0 | same |
| BTSP `w_max` | 0.8 | same |
| BTSP `w_min` | 0.0 | same |
| C1 size | 1000 cells | `iter-66 ENTRY` (`C1Config::default()`) |
| C1 sparsity_k | 20 | iter-66 ENTRY |
| C1 from_r2_fanout | 30 | iter-66 ENTRY |
| C1 init_w_max | 0.5 | iter-66 ENTRY |
| R2 size (R2_N) | 2000 | snn-core compile-time constant |
| R2 inhibitory fraction | 0.30 | iter-46 default |
| R2 recurrent connectivity | 0.05 | iter-46 default |
| Homeostasis scale_only_down | true | iter-67-α homeostasis-gating respects the iter-46 default but disables the rule entirely during teacher Phase 4 |
| Homeostasis a_target | 2.0 | iter-46 default |

## Lock rules (per Bekos's instruction)

1. **No hyperparameter tuning** across the 4-seed run.
2. **No threshold changes** to the Gate-A criterion mid-run.
3. **No cherry-picking** of seeds (don't drop a failed seed
   silently).
4. **Failed seeds must be reported** alongside passing seeds
   in the 4-seed summary.
5. **Same evaluator** (`scripts/evaluate_gate_a.py`) applied
   to every seed, including the partial-run lower-bound mode
   if a seed's process dies before completion.
