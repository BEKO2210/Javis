#!/usr/bin/env python3
"""Per-phase profile of the WebSocket `recall` pipeline.

Drives the configured number of recalls against the running stack,
then scrapes `/metrics` and computes the mean wall-time per phase
from the `javis_recall_phase_seconds` histograms (one label-bucket
per phase). Reports a sorted breakdown with absolute mean time and
percentage share of the total recall.

Usage
-----
    python3 scripts/pipeline_profile.py
    python3 scripts/pipeline_profile.py --n 200 --concurrency 1

Default 100 sequential recalls — sequential keeps the histograms
clean from concurrency-induced queueing.

Dependencies: `websockets` from PyPI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import urllib.parse
import urllib.request

import websockets

HOST = os.environ.get("JAVIS_HOST", "localhost:7777")
HTTP_BASE = f"http://{HOST}"
WS_BASE = f"ws://{HOST}"


def http_get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read().decode("utf-8")


def histogram_pair(prom_text: str, name: str, labels: dict) -> tuple[float, float]:
    """Return (count, sum) for a Prometheus histogram labelled with
    the exact `labels` dict. Both numbers match the `_count` and
    `_sum` series the recorder emits."""
    label_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
    cnt_re = rf"^{re.escape(name)}_count{re.escape(label_str)}\s+([\d.eE+-]+)$"
    sum_re = rf"^{re.escape(name)}_sum{re.escape(label_str)}\s+([\d.eE+-]+)$"
    cnt = re.search(cnt_re, prom_text, re.MULTILINE)
    sm = re.search(sum_re, prom_text, re.MULTILINE)
    return (
        float(cnt.group(1)) if cnt else 0.0,
        float(sm.group(1)) if sm else 0.0,
    )


def histogram_pair_unlabelled(prom_text: str, name: str) -> tuple[float, float]:
    cnt = re.search(
        rf"^{re.escape(name)}_count\s+([\d.eE+-]+)$", prom_text, re.MULTILINE
    )
    sm = re.search(rf"^{re.escape(name)}_sum\s+([\d.eE+-]+)$", prom_text, re.MULTILINE)
    return (
        float(cnt.group(1)) if cnt else 0.0,
        float(sm.group(1)) if sm else 0.0,
    )


async def one_recall(query: str) -> None:
    qs = urllib.parse.urlencode({"action": "recall", "query": query})
    async with websockets.connect(f"{WS_BASE}/ws?{qs}", max_size=2_000_000) as ws:
        async for raw in ws:
            ev = json.loads(raw)
            if ev.get("type") == "done":
                break


async def run_workload(n: int, concurrency: int, queries: list[str]) -> None:
    if concurrency == 1:
        for i in range(n):
            await one_recall(queries[i % len(queries)])
        return

    sem = asyncio.Semaphore(concurrency)

    async def bounded(i: int) -> None:
        async with sem:
            await one_recall(queries[i % len(queries)])

    await asyncio.gather(*[bounded(i) for i in range(n)])


PHASES = [
    "lock_overhead",
    "encode",
    "snn_compute",
    "decode",
    "rag_search",
    "response_build",
]

SUBPHASES = [
    "brain_compute",
    "ws_stream",
]


def diff(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (b[0] - a[0], b[1] - a[1])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--concurrency", type=int, default=1)
    args = p.parse_args()

    queries = ["rust", "python", "cpp", "language", "memory", "safety"]

    print(f"Target {HTTP_BASE} — {args.n} recalls, concurrency {args.concurrency}\n")

    # Snapshot the per-phase histograms BEFORE the workload, so we
    # only count the increments produced by this run.
    metrics0 = http_get(f"{HTTP_BASE}/metrics")
    before = {ph: histogram_pair(metrics0, "javis_recall_phase_seconds", {"phase": ph}) for ph in PHASES}
    total_before = histogram_pair_unlabelled(metrics0, "javis_recall_duration_seconds")

    asyncio.run(run_workload(args.n, args.concurrency, queries))

    metrics1 = http_get(f"{HTTP_BASE}/metrics")
    after = {ph: histogram_pair(metrics1, "javis_recall_phase_seconds", {"phase": ph}) for ph in PHASES}
    total_after = histogram_pair_unlabelled(metrics1, "javis_recall_duration_seconds")

    print(f"{'phase':<16}  {'count':>6}  {'mean ms':>9}  {'share':>6}")
    print("-" * 44)

    rows = []
    sum_means = 0.0
    for ph in PHASES:
        d_count, d_sum = diff(before[ph], after[ph])
        if d_count <= 0:
            continue
        mean_s = d_sum / d_count
        rows.append((ph, d_count, mean_s))
        sum_means += mean_s

    if sum_means == 0:
        print("(no per-phase samples seen — was the recorder initialised?)")
        return 1

    # Sort by mean descending so the bottleneck pops up first.
    rows.sort(key=lambda r: r[2], reverse=True)
    for ph, n_obs, mean_s in rows:
        ms = mean_s * 1000.0
        share = 100.0 * mean_s / sum_means
        print(f"{ph:<16}  {int(n_obs):>6}  {ms:>9.3f}  {share:>5.1f}%")

    print("-" * 44)
    total_count, total_sum = diff(total_before, total_after)
    if total_count > 0:
        total_mean_ms = (total_sum / total_count) * 1000.0
        unaccounted_ms = total_mean_ms - sum_means * 1000.0
        print(
            f"{'phase sum':<16}  {'':>6}  {sum_means * 1000.0:>9.3f}  {100.0:>5.1f}%"
        )
        print(
            f"{'recall total':<16}  {int(total_count):>6}  "
            f"{total_mean_ms:>9.3f}  (= per-phase + {unaccounted_ms:.3f} ms gap)"
        )

    # Sub-phase breakdown of `snn_compute`. Distinguishes the raw
    # `Brain::step_immutable` cost from the interleaved WS Step-event
    # streaming.
    sub_before = {ph: histogram_pair(metrics0, "javis_recall_subphase_seconds", {"phase": ph}) for ph in SUBPHASES}
    sub_after = {ph: histogram_pair(metrics1, "javis_recall_subphase_seconds", {"phase": ph}) for ph in SUBPHASES}
    sub_rows = []
    sub_sum_means = 0.0
    for ph in SUBPHASES:
        d_count, d_sum = diff(sub_before[ph], sub_after[ph])
        if d_count <= 0:
            continue
        m = d_sum / d_count
        sub_rows.append((ph, d_count, m))
        sub_sum_means += m
    if sub_rows:
        print()
        print(f"snn_compute sub-phases:")
        print(f"{'  phase':<16}  {'count':>6}  {'mean ms':>9}  {'share':>6}")
        print("  " + "-" * 42)
        sub_rows.sort(key=lambda r: r[2], reverse=True)
        for ph, n_obs, mean_s in sub_rows:
            ms = mean_s * 1000.0
            share = 100.0 * mean_s / sub_sum_means if sub_sum_means > 0 else 0.0
            print(f"  {ph:<14}  {int(n_obs):>6}  {ms:>9.3f}  {share:>5.1f}%")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.URLError as e:
        print(f"FATAL: cannot reach {HTTP_BASE} ({e})", file=sys.stderr)
        sys.exit(2)
