# CLAUDE.md — Javis project constitution

> **This file is the rulebook. Every Claude Code session in this
> repository reads it before any other input. Where a user prompt
> conflicts with this constitution, the constitution wins — say so
> explicitly and refuse to proceed until the conflict is resolved
> by the operator.**

---

## §1 Mission (north star, non-negotiable)

**Build a working AGI-grade associative-memory cortex for LLM agents.**
Javis is the cortex; the LLM is the mouth. The end-state is a
spiking neural network that *binds cue → target* robustly enough to
ship as the memory layer of a working agent.

Every iteration, every commit, every measurement exists to move the
project toward that end-state. **Javis must work in the end.** That is
the only success criterion.

### §1.1 Mission deviation check (BLOCKING)

Before executing a prompt, run this check silently:

1. Does this prompt move us toward (or measurably preserve)
   binding-task performance?
2. Does it respect the locked iteration discipline (§3)?
3. Does it respect data integrity (§4)?

If **any answer is no**, STOP. Output exactly:

> "This prompt deviates from the locked Javis mission (§<N>).
> Specifically: <one-sentence reason>. I will not proceed until
> Bekos resolves the conflict (relaxes the rule with a written
> ENTRY, or rephrases the prompt)."

Then wait. Do not "compromise" by partially executing.

---

## §2 Hard stops (refuse the prompt; do not negotiate)

| Trigger | Action |
| --- | --- |
| Prompt asks to modify locked γ.X config (`reports/gate_a_gamma_1_1_config.json`) | REFUSE — locks are post-hoc-immutable per Bekos's iter-66 ENTRY rule. |
| Prompt asks to delete or "clean up" a failed seed's log | REFUSE — failed seeds are part of the verdict (§3). |
| Prompt asks to relax the Gate-A `last-8 mean ≥ 0.05` threshold | REFUSE — threshold is locked at iter-66 ENTRY; Gate-B γ.1.1 verdict already used it. |
| Prompt asks to change a published verdict's class (A/B/C/D) | REFUSE — verdicts are written, signed, immutable. |
| Prompt asks to skip the evaluator (`scripts/evaluate_gate_a.py`) and judge a run by eye | REFUSE — same evaluator for every seed is locked policy. |
| Prompt asks to introduce a tensor framework (Burn / Candle / tch-rs) | REFUSE — wrong abstraction for event-driven sparse spike propagation (§5.3). |
| Prompt asks to bypass a hook (`--no-verify`, `--no-gpg-sign`) without an explicit Bekos override in the same prompt | REFUSE — hook outputs are diagnostic signal we need. |
| Prompt asks to force-push to `main` | REFUSE — never. |

---

## §3 Iteration discipline (locked since iter-66 ENTRY)

1. **Pre-registration before compute.** Each iteration lands its
   hypothesis, locked config, locked seed set, locked acceptance
   matrix, and post-run analysis plan as an ENTRY commit *before*
   any new compute is launched.
2. **No hyperparameter tuning between gates** of the same iteration.
3. **No threshold relaxation post-hoc.** Gate-A `last-8 mean ≥ 0.05`
   is the line; that line does not move.
4. **No seed cherry-picking.** Failed seeds (e.g. γ.1.1 seed 0)
   are reported alongside passing seeds. Pass-rate is what it is.
5. **Same evaluator for every seed** — `scripts/evaluate_gate_a.py`.
   Exit codes 0 / 1 / 2 = PASS / FAIL / INCONCLUSIVE.
6. **PowerShell ↔ Linux determinism.** Logs from Bekos's Windows PC
   are decoded by `scripts/clean_powershell_log.py` (UTF-16 LE BOM
   auto-detect, line-wrap rejoin).
7. **Lock rules in their own file.** Frozen γ.X configs live in
   `reports/gate_a_gamma_*_config.{json,md}` at a named git commit;
   that commit is the lock.

---

## §4 Data integrity (STOP-LEVEL invariants)

Anything that puts these at risk → halt and ask Bekos:

- **Bit-identity contract.** Same `--seeds N` ⇒ bit-identical kWTA
  integer hits across all platforms. Sub-permille drifts on
  weight-magnitude counters (`tgt_w`, `plateau_events`) are
  documented FP-codegen artefacts and OK; verdict-relevant integer
  hits are NOT.
- **Active-compute hold.** When a multi-hour seed run is in flight
  on Bekos's machine, do NOT touch `crates/`, `Cargo.toml`,
  `deny.toml`, `.gitignore`, README badges, or CI workflows.
  Documentation in `notes/` and `reports/` is fine on explicit
  request. The hold is released when Bekos pushes the run logs
  and asks for the verdict.
- **Off-path bit-identity.** New mechanisms (γ.4
  `non_target_depression_strength`, future γ.5+) MUST default to
  the off value that makes prior-iter numerics bit-identical.
  Verified by snapshot tests + the BTSP `btsp_off_path_is_bit_identical`
  test before commit.
- **Same corpus.** `--corpus-vocab 64` is locked since iter-58.
  Changes here invalidate every prior verdict.

---

## §5 Performance optimization order (locked 2026-05-08)

After the GPU-suitability analysis (full reasoning in the conversation
transcript at commit `3bac233`):

1. **§5.1 Option E first — `std::simd` + Rayon.** Effort 3/10,
   expected 4–10× speedup, zero numeric-drift risk (deterministic
   stable reductions), bit-identity preserved.
2. **§5.2 Custom CUDA via `cudarc` only after §5.1 ships.** Effort
   6/10, expected 100–500× speedup, BREAKS bit-identity on
   weight-counter reductions (kWTA integer hits stay stable).
   Requires its own ENTRY pre-registering the determinism-contract
   relaxation and the snapshot-test extension.
3. **§5.3 Tensor frameworks (Burn / Candle / tch-rs) are EXCLUDED.**
   Javis is event-driven sparse spike propagation, not dense
   tensor math; the abstractions actively hurt.

§5.1 is itself triggered only by Class (B) extension to 16 seeds OR
by a future iteration (γ.5+) where compute genuinely blocks progress.
Don't pre-optimize.

---

## §6 Active hold (mutable — update when state changes)

**Status:** γ.4 8-seed compute running on Bekos's Windows PC since
the merge of commit `475f379` into `main`. ETA ~18–20 h sequential.

**Hold scope:** §4 active-compute rule applies. No `crates/` touches,
no `Cargo.toml` / `deny.toml` / `.gitignore` / CI changes, no README
badge bumps.

**Release condition:** Bekos pushes `reports/runs/gate_b_gamma_4_seed{0..7}.log`
to `origin/main` and asks for the verdict.

**On release:** pull, run `scripts/evaluate_gate_a.py` over the 8
logs, compute aggregate stats, apply the locked acceptance matrix
in `reports/gate_b_gamma_4_entry.md`, write
`reports/gate_b_gamma_4_8seed_summary.md`, commit class verdict.

---

## §7 Build, test, evaluate

```sh
# Workspace build / lint / test (CI gates):
cargo fmt --all -- --check
cargo clippy --all-targets --workspace -- -D warnings
cargo test --workspace --release            # 147 binary + 3 doc-tests = 150
cargo doc --workspace --no-deps --all-features   # rustdoc -D warnings

# γ.X frozen-config measurement (one seed at a time):
# CLI surface locked in reports/gate_a_gamma_1_1_config.md (γ.1.1)
# and reports/gate_b_gamma_4_entry.md (γ.4).

# Evaluate a measurement log:
python3 scripts/evaluate_gate_a.py reports/runs/<log>.log --json-only
```

---

## §8 Anchors (deep specs Claude must consult before deciding)

- @reports/gate_a_gamma_1_1_config.md — frozen γ.1.1 CLI + numeric locks
- @reports/gate_b_gamma_1_1_entry.md — Gate-B γ.1.1 pre-registration + acceptance matrix
- @reports/gate_b_gamma_1_1_8seed_summary.md — Gate-B γ.1.1 verdict (Class C)
- @reports/gate_b_gamma_4_entry.md — γ.4 pre-registration + H1–H4 hypotheses
- @notes/67-btsp-tagged-eligibility-c1.md — iter-67 BTSP design + sweep history
- @CHANGELOG.md — full iter-00 → iter-67-γ.4 chain

---

## §9 Anti-patterns (catch yourself before doing these)

| Wrong | Right |
| --- | --- |
| "Let me also clean up X while I'm here" during an active hold | Active hold = no scope creep. Only the explicit task. |
| Adding a feature flag "just in case" for backwards-compat | Don't. The codebase is research; delete unused code. |
| Writing prose docstrings explaining what well-named code does | Only document the *why* — hidden constraints, surprises. |
| Running ad-hoc python to "quickly check" a verdict | Always go through `scripts/evaluate_gate_a.py`. |
| Patching a partial run silently to make it pass | Use the lower-bound proof or call it INCONCLUSIVE. |
| Inferring a fix from a single-seed observation | Multi-seed verdicts only. Single-seed = candidate, not verdict. |

---

## §10 Constitution amendment process

This file is amended by an explicit Bekos prompt of the form
*"add to / remove from / change the Javis constitution: <text>"*.
Drift-by-implication ("we kind of did X today, let's update CLAUDE.md
to match") is forbidden. The constitution leads behavior; behavior
does not retroactively rewrite the constitution.

When in doubt: STOP, quote the relevant section to Bekos, ask.

---

*References for the structure of this file (Anthropic Claude Code
official guidance, accessed 2026-05-08):*

- *https://code.claude.com/docs/en/best-practices*
- *https://code.claude.com/docs/en/memory.md*
- *https://www.humanlayer.dev/blog/writing-a-good-claude-md*
