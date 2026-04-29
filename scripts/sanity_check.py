#!/usr/bin/env python3
"""End-to-end sanity check for a running Javis stack.

Drives a sequence of `train`, `recall`, and `ask` operations over the
WebSocket interface, then scrapes `/ready` and `/metrics` to confirm
the brain actually moved. Designed to be run against the local
`docker compose up` stack on `localhost:7777`, but works against any
javis-viz instance that exposes the standard endpoints.

Usage
-----
    python3 scripts/sanity_check.py            # default localhost:7777
    JAVIS_HOST=other-host:7777 \\
        python3 scripts/sanity_check.py        # custom target

Exits 0 if every check passes; 1 on the first failure with a clear
message.

Dependencies
------------
- `websockets` (PyPI). Install with `pip install websockets`.
"""
import asyncio
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import websockets

HOST = os.environ.get("JAVIS_HOST", "localhost:7777")
HTTP_BASE = f"http://{HOST}"
WS_BASE = f"ws://{HOST}"


def get(url: str, parse_json: bool = False):
    with urllib.request.urlopen(url, timeout=5) as r:
        body = r.read().decode("utf-8")
    return json.loads(body) if parse_json else body


def metric(prom_text: str, name: str, labels: dict | None = None) -> float | None:
    """Return the numeric value of a single Prometheus metric line."""
    label_str = ""
    if labels:
        label_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
    pattern = rf"^{re.escape(name)}{re.escape(label_str)}\s+([\d.eE+-]+)$"
    m = re.search(pattern, prom_text, re.MULTILINE)
    return float(m.group(1)) if m else None


async def drive_action(action: str, **params) -> dict:
    """Open one WS connection for `action`, drain until `done`."""
    qs = urllib.parse.urlencode({"action": action, **params})
    counts: dict[str, int] = {}
    last_decoded = None
    last_asked = None
    async with websockets.connect(f"{WS_BASE}/ws?{qs}", max_size=2_000_000) as ws:
        async for raw in ws:
            ev = json.loads(raw)
            t = ev.get("type", "?")
            counts[t] = counts.get(t, 0) + 1
            if t == "decoded":
                last_decoded = ev
            elif t == "asked":
                last_asked = ev
            elif t == "done":
                break
    return {"counts": counts, "decoded": last_decoded, "asked": last_asked}


async def main() -> int:
    print(f"=== Javis end-to-end sanity, target={HOST}, {time.strftime('%H:%M:%S')} ===\n")

    ready = get(f"{HTTP_BASE}/ready", parse_json=True)
    print(
        f"INITIAL  /ready: sentences={ready['sentences']} "
        f"words={ready['words']} llm={ready['llm']}"
    )

    print("\n--- TRAIN one new sentence ---")
    new_sentence = (
        "Lava is liquid molten rock from a volcano, "
        "cooling into igneous rock as it loses heat."
    )
    train = await drive_action("train", text=new_sentence)
    tc = train["counts"]
    print(
        f"  events: init={tc.get('init', 0)} phase={tc.get('phase', 0)} "
        f"step={tc.get('step', 0)}"
    )
    if tc.get("init", 0) != 1:
        print("  FAIL: expected exactly one 'init' event")
        return 1
    if tc.get("step", 0) < 50:
        print("  FAIL: too few step events for a 150 ms training run")
        return 1

    print("\n--- RECALL trained word ---")
    recall = await drive_action("recall", query="lava")
    decoded = recall["decoded"]
    if decoded is None:
        print("  FAIL: no decoded event for 'lava'")
        return 1
    cands = [c["word"] for c in decoded["candidates"]]
    print(
        f"  candidates={cands[:6]}  "
        f"reduction={decoded['reduction_pct']:.1f}%"
    )
    if "lava" not in cands:
        print("  FAIL: 'lava' missing from candidates")
        return 1
    if decoded["reduction_pct"] < 70:
        print(f"  FAIL: token reduction {decoded['reduction_pct']:.1f}% under 70%")
        return 1

    print("\n--- RECALL bootstrap word ---")
    recall = await drive_action("recall", query="rust")
    cands = [c["word"] for c in recall["decoded"]["candidates"]]
    print(f"  candidates={cands[:6]}")
    if "rust" not in cands:
        print("  FAIL: bootstrap word 'rust' missing from candidates")
        return 1

    print("\n--- ASK in mock mode ---")
    ask = await drive_action(
        "ask", query="rust", rag=new_sentence, javis="rust molten igneous"
    )
    if ask["asked"] is None:
        print("  FAIL: no 'asked' event")
        return 1
    rag, javis = ask["asked"]["rag"], ask["asked"]["javis"]
    print(
        f"  rag:   real={rag['real']} "
        f"input={rag['input_tokens']} output={rag['output_tokens']}"
    )
    print(
        f"  javis: real={javis['real']} "
        f"input={javis['input_tokens']} output={javis['output_tokens']}"
    )
    if rag["real"] is True or javis["real"] is True:
        print("  WARN: real LLM call detected — sanity script expects mock mode")
    if rag["input_tokens"] <= javis["input_tokens"]:
        print("  FAIL: RAG should consume more input tokens than Javis")
        return 1

    print("\n--- 5 parallel recalls ---")
    t0 = time.monotonic()
    parallel = await asyncio.gather(
        *[drive_action("recall", query="rust") for _ in range(5)]
    )
    parallel_ms = (time.monotonic() - t0) * 1000
    print(
        f"  5 sessions completed in {parallel_ms:.0f} ms "
        f"(serialised through one tokio::Mutex)"
    )
    if any(r["decoded"] is None for r in parallel):
        print("  FAIL: at least one parallel recall got no decoded event")
        return 1

    print("\n--- FINAL state + metrics ---")
    after = get(f"{HTTP_BASE}/ready", parse_json=True)
    text = get(f"{HTTP_BASE}/metrics")
    print(f"  /ready: sentences={after['sentences']} words={after['words']}")
    if after["sentences"] != ready["sentences"] + 1:
        print(
            f"  FAIL: sentence count did not increment "
            f"({ready['sentences']} → {after['sentences']})"
        )
        return 1

    def m(name: str, labels: dict | None = None) -> float:
        v = metric(text, name, labels)
        return v if v is not None else 0.0

    train_n = m("javis_ws_sessions_total", {"action": "train"})
    recall_n = m("javis_ws_sessions_total", {"action": "recall"})
    ask_n = m("javis_ws_sessions_total", {"action": "ask"})
    print(
        f"  ws_sessions: train={train_n:.0f} recall={recall_n:.0f} ask={ask_n:.0f}"
    )
    if train_n < 1 or ask_n < 1 or recall_n < 7:
        print("  FAIL: counter values lower than expected")
        return 1

    rag_t = m("javis_recall_tokens_rag_total")
    javis_t = m("javis_recall_tokens_javis_total")
    if rag_t > 0:
        saving = 100 * (1 - javis_t / rag_t)
        print(
            f"  lifetime tokens: rag={rag_t:.0f} javis={javis_t:.0f} "
            f"saving={saving:.1f}%"
        )
        if saving < 70:
            print("  FAIL: lifetime token saving < 70%")
            return 1

    print("\nALL SANITY CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except urllib.error.URLError as e:
        print(f"FATAL: cannot reach {HTTP_BASE} ({e})", file=sys.stderr)
        sys.exit(2)
