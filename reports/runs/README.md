# `reports/runs/` — per-seed Gate-A 4-seed confirmation logs

This directory holds the live log files written by the
4-seed Gate-A confirmation runner
(`/tmp/iter67/run_4seed.sh`).  One file per seed:

```
gate_a_gamma_1_1_seed0.log
gate_a_gamma_1_1_seed1.log
gate_a_gamma_1_1_seed2.log
gate_a_gamma_1_1_seed3.log
```

Each file is appended to during its run (~3.2 h CPU per
seed at the locked γ.1.1 config).  An intermediate commit
may capture a partial log; the final commit at the end of
the 4-seed sweep captures each file in its complete state.

The frozen γ.1.1 CLI invocation that produced these logs
is in `reports/gate_a_gamma_1_1_config.json`.  The locked
Gate-A criterion is in `reports/gate_a_gamma_1_1_config.md`.
The same Gate-A evaluator (`scripts/evaluate_gate_a.py`)
applies to every per-seed log; partial-run lower-bound
proofs are supported.

The 4-seed verdict is summarised at the end of the sweep in
`reports/gate_a_gamma_1_1_4seed_summary.md`.
