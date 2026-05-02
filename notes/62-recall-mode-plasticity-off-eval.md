# Iter-62 — recall-mode eval (plasticity-off)

## Status
Partial implementation recovery.

- Added a recall-mode eval switch (`--plasticity-off-during-eval`, alias `--recall-mode-eval`).
- Training path remains unchanged.
- Eval path now disables STDP, iSTDP, homeostasis, metaplasticity, intrinsic plasticity, structural plasticity, and zeroes neuromodulator before dictionary/Jaccard evaluation.

## Recovery notes
When this handoff was resumed, the branch was clean (no uncommitted iter-62 code), so there was no partial patch from the prior agent to preserve.

## Pending
The full 4-seed A/B measurement run from the iter-62 spec is still in progress / pending completion in this environment.
