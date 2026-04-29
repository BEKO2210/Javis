#!/usr/bin/env python3
"""Load test for the WebSocket recall path.

Spawns N concurrent workers, each repeatedly opens a fresh `recall`
WebSocket session, drains it, closes, repeats. Runs each concurrency
level for `--duration` seconds and reports throughput + latency
percentiles.

Usage
-----
    python3 scripts/load_test.py
    python3 scripts/load_test.py --levels 1,10,50,100 --duration 20

The test writes the live-aggregated `javis_recall_duration_seconds`
histogram counts before+after each level so server-side and
client-side measurements can be cross-checked.

Dependencies: `websockets` from PyPI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
import urllib.parse
import urllib.request

import websockets

HOST = os.environ.get("JAVIS_HOST", "localhost:7777")
HTTP_BASE = f"http://{HOST}"
WS_BASE = f"ws://{HOST}"

QUERIES = ["rust", "python", "cpp", "language", "memory", "safety"]


def http_get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read().decode("utf-8")


def metric(prom_text: str, name: str, labels: dict | None = None) -> float | None:
    label_str = ""
    if labels:
        label_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
    pattern = rf"^{re.escape(name)}{re.escape(label_str)}\s+([\d.eE+-]+)$"
    m = re.search(pattern, prom_text, re.MULTILINE)
    return float(m.group(1)) if m else None


async def one_recall(query: str) -> float:
    """One full recall round-trip; returns wall-time in seconds."""
    qs = urllib.parse.urlencode({"action": "recall", "query": query})
    t0 = time.monotonic()
    async with websockets.connect(f"{WS_BASE}/ws?{qs}", max_size=2_000_000) as ws:
        async for raw in ws:
            ev = json.loads(raw)
            if ev.get("type") == "done":
                break
    return time.monotonic() - t0


async def worker(stop_at: float, results: list[float]) -> None:
    """Run recalls until the deadline; record latencies."""
    i = 0
    while time.monotonic() < stop_at:
        q = QUERIES[i % len(QUERIES)]
        i += 1
        try:
            dt = await one_recall(q)
            results.append(dt)
        except Exception as e:  # noqa: BLE001
            results.append(float("inf"))
            print(f"  worker error: {e}", file=sys.stderr)


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    sorted_xs = sorted(xs)
    k = (len(sorted_xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(sorted_xs) - 1)
    if lo == hi:
        return sorted_xs[lo]
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (k - lo)


async def run_level(concurrency: int, duration: float) -> dict:
    print(
        f"\n=== concurrency={concurrency:>3}, duration={duration:.0f}s ==="
    )
    metrics_before = http_get(f"{HTTP_BASE}/metrics")
    server_recalls_before = metric(
        metrics_before, "javis_recall_duration_seconds_count"
    ) or 0.0
    server_sum_before = metric(
        metrics_before, "javis_recall_duration_seconds_sum"
    ) or 0.0

    results: list[float] = []
    stop_at = time.monotonic() + duration
    t0 = time.monotonic()
    await asyncio.gather(*[worker(stop_at, results) for _ in range(concurrency)])
    wall = time.monotonic() - t0

    metrics_after = http_get(f"{HTTP_BASE}/metrics")
    server_recalls_after = metric(
        metrics_after, "javis_recall_duration_seconds_count"
    ) or 0.0
    server_sum_after = metric(
        metrics_after, "javis_recall_duration_seconds_sum"
    ) or 0.0

    server_n = server_recalls_after - server_recalls_before
    server_total_s = server_sum_after - server_sum_before
    server_mean_ms = (server_total_s / server_n * 1000) if server_n > 0 else float("nan")

    n = len(results)
    finite = [x * 1000 for x in results if x != float("inf")]
    p50 = percentile(finite, 0.50)
    p95 = percentile(finite, 0.95)
    p99 = percentile(finite, 0.99)
    mean = statistics.fmean(finite) if finite else float("nan")
    throughput = n / wall

    print(
        f"  ops:        {n} client-side, {server_n:.0f} server-side  "
        f"({throughput:.1f} ops/s)"
    )
    print(
        f"  client-ms:  mean={mean:.1f}  p50={p50:.1f}  p95={p95:.1f}  p99={p99:.1f}"
    )
    print(f"  server-ms:  mean={server_mean_ms:.1f}")
    print(f"  errors:     {sum(1 for x in results if x == float('inf'))}")

    return {
        "concurrency": concurrency,
        "wall_s": wall,
        "ops_total": n,
        "ops_per_s": throughput,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "server_mean_ms": server_mean_ms,
    }


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--levels", default="1,10,50,100")
    p.add_argument("--duration", type=float, default=15.0)
    args = p.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    print(f"Target {HTTP_BASE}, concurrency levels {levels}, {args.duration}s each\n")

    rows: list[dict] = []
    for c in levels:
        rows.append(await run_level(c, args.duration))
        # Brief cool-down so each level starts from a quiet state.
        await asyncio.sleep(2.0)

    print("\n=== SUMMARY ===")
    print(f"{'conc':>5}  {'ops':>6}  {'ops/s':>7}  "
          f"{'p50':>6}  {'p95':>6}  {'p99':>6}  {'srv mean':>9}")
    for r in rows:
        print(
            f"{r['concurrency']:>5}  {r['ops_total']:>6}  "
            f"{r['ops_per_s']:>7.1f}  "
            f"{r['p50_ms']:>6.1f}  {r['p95_ms']:>6.1f}  {r['p99_ms']:>6.1f}  "
            f"{r['server_mean_ms']:>9.1f}"
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except urllib.error.URLError as e:
        print(f"FATAL: cannot reach {HTTP_BASE} ({e})", file=sys.stderr)
        sys.exit(2)
