# CLAUDE.md — operational policies for the Javis repo

## Active hold (set 2026-05-08)

**Do NOT touch anything in `crates/` while the γ.4 8-seed compute is
running on Bekos's Windows PC** (~18–20 h sequential, started after
commit `475f379` was merged into main and pulled by Bekos).
"Anything" means: source files, Cargo.toml, deny.toml, gitignore,
README badges, CI workflows. Documentation in `notes/` and `reports/`
is fine if explicitly requested.

The hold is released once Bekos pushes `reports/runs/gate_b_gamma_4_seed{0..7}.log`
to origin/main and asks for the Gate-B γ.4 verdict.

## Performance-optimization policy (locked 2026-05-08)

Result of the GPU-suitability analysis (full reasoning preserved in
the conversation transcript at this commit):

1. **First step is always Option E: SIMD (`std::simd`) + Rayon.**
   Aufwand 3/10, expected 4–10× speedup, zero numeric-drift risk
   (deterministic stable reductions), bit-identity contract preserved.
   Apply this *only* if Gate-B γ.4 lands in Class (B) "Robust" with
   the optional 16-seed extension OR if a future iteration (γ.5+)
   genuinely needs the headroom.

2. **Custom CUDA kernels via `cudarc` come ONLY AFTER Option E has
   been measured + committed.** Aufwand 6/10, expected 100–500×
   speedup, but breaks the bit-identity contract on weight-counter
   reductions (kWTA integer hits stay stable). Requires its own
   ENTRY pre-registration covering the determinism-contract
   relaxation and a snapshot-test extension.

3. **Tensor-frameworks (Burn / Candle / tch-rs) are explicitly
   ruled out.** Javis is event-driven sparse spike propagation,
   not dense neural-network tensor math; the abstractions don't
   fit.

## Iteration discipline (carries over from iter-66 ENTRY)

- No hyperparameter tuning between gates.
- No threshold relaxation post-hoc.
- No cherry-picking of seeds — failed seeds reported alongside
  passing seeds.
- Same evaluator (`scripts/evaluate_gate_a.py`) for every run.
- Cross-platform PowerShell logs decoded by
  `scripts/clean_powershell_log.py` (UTF-16 LE auto-detect).
- Bit-identity contract: same `--seeds N` ⇒ bit-identical kWTA
  integer hits across Linux/Windows. Sub-permille drifts on
  weight-magnitude counters are documented and acceptable;
  verdicts depend only on integer hits.
