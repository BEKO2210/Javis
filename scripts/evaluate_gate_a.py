#!/usr/bin/env python3
"""Gate-A evaluator for iter-67-γ.1.1 (and any future config that
uses the same `[iter-66 diag]` log line schema).

Inputs
------
- One log file (path positional arg) produced by
  `cargo run --release -p eval --example reward_benchmark` with
  `--c1-diagnostic` set.

Outputs
-------
- One JSON object on stdout with the verdict + supporting stats.
- Exit code: 0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE.

Verdicts
--------
- PASS:
    * complete run (>= 32 epochs): last-8-ep mean
      `top3_c1` >= 0.05  AND  no sustained collapse  AND
      C1 active  AND  recall-mode stable.
    * partial run (< 32 epochs): the lower-bound proof
      establishes that even with the missing epochs at 0.0,
      the resulting last-8 mean would still meet >= 0.05.
- FAIL:
    * complete run with last-8 mean < 0.05  OR
    * sustained collapse (longest contiguous-zero streak > 2)
      OR  C1 silent on > 16/32 cues majority of epochs  OR
      recall-mode unstable.
- INCONCLUSIVE:
    * partial run where the lower-bound proof does NOT
      establish PASS (i.e. observed sum + 0 * remaining < 0.400).
      Returned ONLY when the partial run does not have enough
      epochs in the last-8 window to mathematically guarantee
      a result.

The evaluator is deliberately strict: any case where the
mathematical lower bound is below 0.05 returns INCONCLUSIVE,
not FAIL.  A FAIL verdict requires either a complete run or
unambiguous evidence that no plausible completion can change
the outcome.

Usage
-----
    python3 scripts/evaluate_gate_a.py <log_path>
    python3 scripts/evaluate_gate_a.py --json-only <log_path>
    python3 scripts/evaluate_gate_a.py --total-epochs 32 <log_path>

The script does not import any third-party libraries.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import List, Optional


# Regex for the per-epoch diagnostic line emitted when
# `--c1-diagnostic` is set.  We pull out epoch index, top3_c1,
# kwta_empty (eval), top3_r2, and a few auxiliary fields.
_DIAG_RE = re.compile(
    r"\[iter-66 diag\] seed=(?P<seed>\d+) epoch=(?P<epoch>\d+)/(?P<total>\d+) "
    r"teacher: trials=\d+ c1_active_frac=(?P<c1_active_frac>[0-9.]+) "
    r"c1_spikes_mean=[0-9.]+ clamp_eff=(?P<clamp_eff>[0-9.]+) "
    r"\| eval: kwta_empty=(?P<kwta_empty>\d+)/(?P<n_pairs>\d+) "
    r"target_in_dict=\d+/\d+ spikes_mean=[0-9.]+ "
    r"top3_r2=(?P<top3_r2>[0-9.]+) top3_c1=(?P<top3_c1>[0-9.]+) "
    r"mrr_c1=(?P<mrr_c1>[0-9.]+) raw_overlap=(?P<raw_overlap>[0-9.]+) "
    r"dict_concepts=(?P<dict>\d+) "
    r"\| r2c1: l2=(?P<l2>[0-9.]+) nz_upd=\d+ max\|.{1,3}w\|=(?P<max_dw>[0-9.]+) "
    r"sum\|.{1,3}w\|=(?P<sum_dw>[0-9.]+) tgt_w=(?P<tgt_w>[0-9.]+) "
    r"non_w=(?P<non_w>[0-9.]+) w_ratio=(?P<w_ratio>[0-9.]+)"
)

LAST_K = 8
PASS_MIN_MEAN = 0.05
COLLAPSE_MAX_RUN = 2


def _parse_log(path: str) -> tuple[Optional[int], List[dict]]:
    """Parse the diagnostic log.  Returns (declared_total_epochs,
    list-of-per-epoch-dicts).  Empty list when no diagnostic lines
    found."""
    rows: List[dict] = []
    total: Optional[int] = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _DIAG_RE.search(line)
            if not m:
                continue
            row = m.groupdict()
            for k in ("seed", "epoch", "total", "kwta_empty", "n_pairs", "dict"):
                row[k] = int(row[k])
            for k in (
                "c1_active_frac",
                "clamp_eff",
                "top3_r2",
                "top3_c1",
                "mrr_c1",
                "raw_overlap",
                "l2",
                "max_dw",
                "sum_dw",
                "tgt_w",
                "non_w",
                "w_ratio",
            ):
                row[k] = float(row[k])
            if total is None:
                total = row["total"]
            rows.append(row)
    return total, rows


def _longest_zero_run(values: List[float]) -> int:
    longest = 0
    current = 0
    for v in values:
        if v <= 1e-9:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _kwta_silent_majority(rows: List[dict], n_pairs_default: int) -> bool:
    n = 0
    silent = 0
    for r in rows:
        n += 1
        denom = r.get("n_pairs", n_pairs_default) or n_pairs_default or 1
        if r["kwta_empty"] >= denom // 2:
            silent += 1
    return n > 0 and silent > n / 2


def evaluate(
    log_path: str,
    total_epochs_override: Optional[int] = None,
) -> dict:
    declared_total, rows = _parse_log(log_path)
    total = total_epochs_override or declared_total or 0

    if not rows:
        return {
            "verdict": "INCONCLUSIVE",
            "reason": "no [iter-66 diag] lines parsed",
            "log_path": log_path,
            "epochs_observed": 0,
            "epochs_declared": total,
        }

    seed = rows[0]["seed"]
    observed_epochs = max(r["epoch"] for r in rows) + 1
    top3_c1 = [r["top3_c1"] for r in sorted(rows, key=lambda r: r["epoch"])]
    top3_r2 = [r["top3_r2"] for r in sorted(rows, key=lambda r: r["epoch"])]
    n_pairs = rows[0].get("n_pairs", 32) or 32

    # --- Compute last-8 stats (or lower bound if partial). ---
    last8_start = total - LAST_K
    last8_end = total - 1
    in_window = [
        r["top3_c1"]
        for r in sorted(rows, key=lambda r: r["epoch"])
        if last8_start <= r["epoch"] <= last8_end
    ]
    observed_last_window_count = len(in_window)
    missing_in_window = LAST_K - observed_last_window_count
    observed_sum = sum(in_window)
    # Lower bound: missing epochs contribute >= 0
    lower_bound_sum = observed_sum
    lower_bound_mean = lower_bound_sum / LAST_K if LAST_K > 0 else 0.0
    # Upper bound: missing epochs contribute <= 1.0
    upper_bound_sum = observed_sum + missing_in_window * 1.0
    upper_bound_mean = upper_bound_sum / LAST_K if LAST_K > 0 else 0.0

    full_run = observed_epochs >= total and missing_in_window == 0
    last8_mean_observed = sum(in_window) / max(1, observed_last_window_count)

    # --- Auxiliary checks. ---
    longest_zero = _longest_zero_run(top3_c1)
    kwta_silent_majority = _kwta_silent_majority(rows, n_pairs)
    r2_collapsed = (
        sum(1 for v in top3_r2 if v < 0.005) > observed_epochs * 0.75
    )

    # Determine verdict. Order: FAIL > PASS > INCONCLUSIVE.
    failures: List[str] = []
    if longest_zero > COLLAPSE_MAX_RUN:
        failures.append(
            f"sustained collapse: longest contiguous-zero streak = {longest_zero} epochs (limit {COLLAPSE_MAX_RUN})"
        )
    if kwta_silent_majority:
        failures.append("C1 silent on majority of epochs (kwta_empty >= n/2)")
    if r2_collapsed:
        failures.append("R2 readout collapsed (top3_r2 < 0.005 on > 75 % of observed epochs)")

    if full_run:
        last8_mean = last8_mean_observed
        if failures:
            verdict = "FAIL"
            reason = "complete run, but: " + "; ".join(failures)
        elif last8_mean >= PASS_MIN_MEAN:
            verdict = "PASS"
            reason = (
                f"complete 32-ep run; last-8 mean = {last8_mean:.4f} >= {PASS_MIN_MEAN}; "
                "auxiliary checks pass"
            )
        else:
            verdict = "FAIL"
            reason = (
                f"complete run; last-8 mean = {last8_mean:.4f} < {PASS_MIN_MEAN}"
            )
    else:
        # Partial run.  Apply lower-bound proof.
        if failures:
            verdict = "FAIL"
            reason = "partial run with: " + "; ".join(failures)
        elif lower_bound_mean >= PASS_MIN_MEAN:
            verdict = "PASS_LOWER_BOUND"
            reason = (
                f"partial run ({observed_epochs}/{total} epochs); "
                f"observed last-8 sum = {observed_sum:.4f} >= "
                f"{PASS_MIN_MEAN * LAST_K:.4f} required → "
                f"last-8 mean lower bound = {lower_bound_mean:.4f} >= "
                f"{PASS_MIN_MEAN}"
            )
        elif upper_bound_mean < PASS_MIN_MEAN:
            verdict = "FAIL"
            reason = (
                f"partial run; upper bound on last-8 mean = "
                f"{upper_bound_mean:.4f} < {PASS_MIN_MEAN} (cannot pass "
                "even if remaining epochs are 1.0)"
            )
        else:
            verdict = "INCONCLUSIVE"
            reason = (
                f"partial run; last-8 lower bound = "
                f"{lower_bound_mean:.4f} < {PASS_MIN_MEAN} <= "
                f"upper bound {upper_bound_mean:.4f} (remaining "
                f"{missing_in_window} epoch(s) determine outcome)"
            )

    return {
        "verdict": verdict,
        "reason": reason,
        "log_path": log_path,
        "seed": seed,
        "epochs_observed": observed_epochs,
        "epochs_declared": total,
        "full_run": full_run,
        "last8_window": [last8_start, last8_end],
        "last8_observed_count": observed_last_window_count,
        "last8_observed_sum": round(observed_sum, 6),
        "last8_required_sum_for_pass": round(PASS_MIN_MEAN * LAST_K, 6),
        "last8_mean_observed": round(last8_mean_observed, 6),
        "last8_lower_bound_mean": round(lower_bound_mean, 6),
        "last8_upper_bound_mean": round(upper_bound_mean, 6),
        "longest_zero_run": longest_zero,
        "kwta_silent_majority": kwta_silent_majority,
        "r2_collapsed": r2_collapsed,
        "all_top3_c1": [round(v, 4) for v in top3_c1],
        "all_top3_r2": [round(v, 4) for v in top3_r2],
        "aggregate_top3_c1_observed_mean": round(
            sum(top3_c1) / max(1, len(top3_c1)), 6
        ),
        "auxiliary_failures": failures,
    }


def _exit_code_for(verdict: str) -> int:
    if verdict == "PASS" or verdict == "PASS_LOWER_BOUND":
        return 0
    if verdict == "FAIL":
        return 1
    return 2


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_path", help="path to the [iter-66 diag] log file")
    parser.add_argument(
        "--total-epochs",
        type=int,
        default=None,
        help="override the declared total-epoch count from the log",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="emit only the JSON object (no human-readable preamble)",
    )
    args = parser.parse_args(argv)

    result = evaluate(args.log_path, args.total_epochs)
    if not args.json_only:
        print(f"Gate-A evaluation: {result['verdict']}")
        print(f"  reason: {result['reason']}")
        print(f"  log: {args.log_path}")
        print(f"  epochs observed: {result['epochs_observed']}/{result['epochs_declared']}")
        if result.get("last8_lower_bound_mean") is not None:
            print(
                f"  last-8 mean (observed/lower-bound/upper-bound) = "
                f"{result['last8_mean_observed']:.4f} / "
                f"{result['last8_lower_bound_mean']:.4f} / "
                f"{result['last8_upper_bound_mean']:.4f}"
            )
        print(f"  longest-zero run: {result['longest_zero_run']}")
    print(json.dumps(result, indent=2))
    return _exit_code_for(result["verdict"])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
