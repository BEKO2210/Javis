<div align="center">

<img src="assets/logo.svg" alt="Javis" width="640">

<br/>

**An associative SNN memory co-processor for LLM agents, built in Rust.**

A spiking neural network that stores knowledge as emergent cell assemblies and
retrieves it through pattern completion. Sits between your retrieval layer and
your LLM, returning a few decoded concepts instead of full document chunks.

[![Rust edition 2021](https://img.shields.io/badge/rust-edition%202021-CE422B?logo=rust&logoColor=white)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-3a86ff)](#license)
[![CI](https://img.shields.io/github/actions/workflow/status/BEKO2210/Javis/ci.yml?branch=main&label=ci&logo=github)](.github/workflows/ci.yml)
[![Tests 130/130](https://img.shields.io/badge/tests-130%2F130%20passing-3fb950)](#tests)
[![Clippy clean](https://img.shields.io/badge/clippy-0%20warnings-3fb950)](#tests)
[![MSRV 1.86](https://img.shields.io/badge/MSRV-1.86-CE422B?logo=rust&logoColor=white)](#tests)
[![Self-recall 100%25](https://img.shields.io/badge/self--recall-100%25-3fb950)](#performance-profile)
[![Token reduction 35-80%25](https://img.shields.io/badge/token%20reduction-35--80%25-ffd166)](#performance-profile)
[![Observability](https://img.shields.io/badge/observability-tracing%20%C2%B7%20Prometheus-7aa2ff)](#production-readiness)
[![Container](https://img.shields.io/badge/container-Docker%20%2B%20Compose-2496ed?logo=docker&logoColor=white)](#run-with-docker)
[![Bio inspired](https://img.shields.io/badge/bio--inspired-LIF%20%C2%B7%20STDP%20%C2%B7%20iSTDP%20%C2%B7%20BTSP-62d6ff)](#plasticity)
[![Iter 44](https://img.shields.io/badge/iter--44-Triplet%20%C2%B7%20R--STDP%20%C2%B7%20BCM%20%C2%B7%20Replay-ff66c4)](#plasticity)

</div>

---

## Why Javis

Modern LLM pipelines spend most of their token budget on **retrieval context**.
Naive RAG ships an entire chunk to the model on every query — even when only
a single fact inside that chunk matters.

Javis flips the architecture: knowledge is stored as **emergent cell assemblies**
in a spiking neural network. A query is a partial cue; pattern completion
inside the network reactivates the relevant assembly; only the few decoded
concepts go to the LLM.

```
Naive RAG:   "Rust is a systems programming language focused on memory
              safety and ownership; the borrow checker prevents data races
              at compile time."                                       63 tokens
Javis:       "rust"                                                    2 tokens
```

That gap is the whole pitch. Whether it holds at scale — and where it stops
holding — is reported below in plain numbers, not slogans.

---

## Performance profile

Measured on a deterministic 100-sentence / 286-vocabulary benchmark
(`cargo run --release -p eval --example scale_benchmark -- --sentences 100`).
Reproducible from a single `--seed`; no external dataset, no network.

### What survives

| Property | Value |
| --- | ---: |
| **Self-recall** (query concept always retrievable) | **100 %** |
| **Token reduction** vs naïve-RAG baseline | **35 – 45 %** |
| **Decoder latency** at vocab ≤ 300 | sub-millisecond |
| **Self-recall test suite** | 113 / 113 passing |

The first row is the architectural claim that Javis stands behind: train a
concept once, recall it deterministically. The second row is the headline
number — modest but real, on a non-toy corpus. The third row makes Javis
practical as a co-processor in front of an LLM.

### Known limits (iter ≤24 baseline)

These are the failure modes a senior reviewer would find on day one. Better
to publish them than have someone tweet them.

| Limit | Measured value | Mechanism |
| --- | ---: | --- |
| **Associative recall** | ≈ 2 % | Of every word that genuinely co-occurs with the query in the corpus, only ~ 2 % is decoded. Javis returns the query plus 5 noise words, not the expected 5–10 related concepts. |
| **Cross-domain bleed** | 4.7 / 6 decoded words | At N > 50 distinct concepts the R2 layer (2 000 neurons, K=220, 11 % sparsity) saturates; iSTDP can no longer build separating walls between engrams, so unrelated domains leak into each other. |
| **Engram capacity** | ≈ 50 concepts | Geometric upper bound from R2_size / KWTA_K = 2 000 / 220 ≈ 9 fully-orthogonal engrams; with overlap-tolerance about 50 work cleanly before interference dominates. |

### What changes in iter 25 (this branch)

R2 was scaled from 2 000 → 10 000 neurons, recurrent connectivity sparsened
from p=0.10 → p=0.03, KWTA from 220 → 100 (1 % sparsity), iSTDP retuned for
aggressive LTD on co-active E-targets. The 113 existing tests still pass at
the new topology. Updated cross-bleed and recall numbers are in
[`notes/43-topology-scaling.md`](notes/43-topology-scaling.md) once the
benchmark run completes.

### What changes in iter 47a (this branch)

The iter-46 negative-margin diagnosis identified the R1 → R2 forward
drive as the dominant factor in the cue's R2 response. Iter-47a tests
the literature-grounded fix (Brunel scaling + Diehl-Cook adaptive
threshold) through a sequential 4-epoch sweep with pre-fixed
acceptance criteria. Result, on the same 16 + 16 pair / vocab-32
corpus, seed 42:

| INTER_WEIGHT | r2_act mean | tgt_hit | selectivity | margin |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.8 | 0.00 | -0.0005 | -0.01 |
| 1.0 | 139 | 2.59 | -0.0005 | -0.02 |
| 0.7 | 507 (cascade) | 9.38 | -0.0047 | -0.02 |

iter-47a-2 alone does **not** flip the margin sign. But the
diagnosis is sharper than iter-46's: at INTER_WEIGHT = 1.0,
`target_hit_mean` grew monotonically over 4 epochs (1.16 → 2.59)
and `selectivity_index` rose from -0.022 to -0.0005 — the right
direction. The 0.7 bistability (recurrent cascade in epoch 3) is
the key second-order finding: hard sparsity control (k-WTA, iter-48
entry) is necessary, not optional. The iter-47 metrics
(`r2_active_pre_teacher_{mean,p10,p90}`, `selectivity_index`) are
wired to A/B-test it cleanly. See
[`notes/47a`](notes/47a-forward-drive-and-adaptive-threshold.md).

### What changes in iter 46 (this branch)

The pair-association harness from iter-45 grows a *teacher-forcing*
training arm: a deterministic per-word canonical R2-E SDR
(`canonical_target_r2_sdr`) and a `drive_with_r2_clamp` primitive
that injects target spikes directly into R2 — bypassing the
random R1 → R2 forward path. Plus a six-phase trial schedule
(cue → delay → prediction → teacher → reward → tail) with
plasticity gating around the prediction window so evaluation
never contaminates training, an anti-causal STDP timing fix
(cue lead-in before the clamp), and a `--association-training-
gate-r1r2` flag to attenuate forward drive during the prediction
phase only.

Honest result on the same 16-pair + 16-noise corpus, seed 42:
`target_clamp_hit_rate = 1.00` across every teacher epoch (the
clamp itself works perfectly), but `correct_minus_incorrect_margin`
stays in `[-0.06, -0.03]` — the canonical-target cells fire
*less* than the rest under cue-only recall, even with the
timing fix and the R1 → R2 gate. The first non-zero
`prediction_top3_before_teacher = 0.02` appears at epoch 3 with
homeostasis on, but does not stabilise above the 9.4 % chance
floor in any run. The bottleneck has moved from iter-45's "we
can't measure it" to iter-46's "we can measure it; here is the
number". See [`notes/46`](notes/46-teacher-forcing.md) for the
full chain of measurements and the next-iter (47) directions
(reduce `INTER_WEIGHT`, add an association-bridge region, or
make R1 → R2 itself learnable).

### What changes in iter 45 (this branch)

A *reward-aware pair-association benchmark*
(`cargo run --release -p eval --example reward_benchmark`) that
finally lets the iter-44 R-STDP / dopamine machinery be exercised:
16 (cue, target) pairs with 16 distractor pairs, staggered
cue → target training, per-trial reward delivery, per-epoch
top-1 / top-3 readout. Pure STDP is run as the baseline arm.

The honest reading: **neither arm reaches above-chance accuracy in
the available training time**. R-STDP shows a small advantage on
noise suppression (mean noise-top-3 `0.10` vs pure STDP `0.16`)
but the architecture's R1 → R2 forward path dominates the cue's
R2 representation, leaving STDP too little room to grow strong
recurrent associations. The infrastructure is in place; the next
experiment (teacher-forcing the target SDR into R2 during
training) is documented in
[`notes/45`](notes/45-reward-bench.md).

### What changes in iter 44.1 (this branch)

A *decoder confidence floor* via `--decode-threshold` (default `0.0`
= pre-iter-44 behaviour, recommended `0.2` for the 32-sentence
corpus). The original `decode_top` always returned `k` results even
when the highest scoring engram sat right at the random-overlap
baseline (`KWTA_K / R2_E = 12.5 %`). The floor omits low-confidence
matches instead of filling the slot with garbage.

Measured on the same 32-sentence corpus, seed 42, `--iter44 off`:

| `--decode-threshold` | FP / Q | Token reduction | Self-recall |
| ---: | ---: | ---: | ---: |
| `0.0` (pre-iter-44) | 4.50 | 38.9 % | 100 % |
| **`0.20`** | **0.62** | **79.7 %** | 100 % |
| `0.30` | 0.00 | 84.7 % | 100 % |

That is **FP − 86 %** and **token reduction × 2.0** with no plasticity
change at all; the SNN's engrams were already orthogonal, the decoder
just refused to admit it.

### What changes in iter 44 (this branch)

Seven new biology-grade plasticity mechanisms join the existing
LIF / STDP / iSTDP / homeostasis / BTSP stack, all opt-in and
default-off so every pre-iter-44 test stays bit-identical.

**Honest benchmark result**: on the deterministic 32-sentence corpus
(seed 42), the new mechanisms *do not* improve recall over the
iter-43 baseline out of the box — `off` 4.4 %, `stability` 4.4 %,
`tuned` 2.7 %, `full` 1.6 %. Heterosynaptic / BCM scale weights
uniformly per post and don't change the kWTA fingerprint;
reward-modulated STDP and replay both need longer training windows
or a reward signal the current eval harness does not provide. The
stack is infrastructure for the *next* benchmark — multi-epoch
streaming corpora and reward-aware retrieval — see
[`notes/44`](notes/44-breakthrough-plasticity.md) for the full
reading. The mechanisms themselves:

1. **Triplet STDP** (Pfister-Gerstner 2006) — frequency-dependent LTP.
2. **Reward-modulated STDP with eligibility traces** — three-factor
   learning, gated by `Brain::set_neuromodulator(...)` (the
   dopamine surrogate). Closes the temporal-credit-assignment loop
   that pure pair-STDP cannot solve.
3. **BCM metaplasticity** — sliding LTP/LTD threshold per post-neuron;
   stops the runaway-LTP failure mode under sustained drive.
4. **Intrinsic plasticity** — adaptive per-neuron threshold; every
   cell drifts towards its target rate, no dead or saturated neurons.
5. **Heterosynaptic L2 normalisation** — the direct fix for the R2
   saturation problem in `notes/43`. Hard-bounds each post-neuron's
   incoming-weight budget.
6. **Structural plasticity** — sprout new edges between repeatedly
   co-active E cells, prune persistently-dormant ones. Engram
   capacity stops being a hard topology constant.
7. **Offline replay / consolidation** — `Brain::consolidate(...)`
   drives the top-k engram cells in pulses with full plasticity on,
   the way slow-wave-sleep replay deepens hippocampal engrams.

Switch the whole stack on in the live viz with
`JAVIS_ITER44=1 cargo run -p viz --release`.

The full architectural rationale, composition into the existing
pipeline, and 15 new tests are documented in
[`notes/44-breakthrough-plasticity.md`](notes/44-breakthrough-plasticity.md).

### Reproducibility

```sh
# Train + evaluate on 100 sentences. ~5 min wall on R2=10 000.
cargo run --release -p eval --example scale_benchmark \
    -- --sentences 100 --queries 30 --decode-k 6 --seed 42

# Smaller smoke run for CI / quick checks (~30 s):
cargo run --release -p eval --example scale_benchmark -- --sentences 32
```

The benchmark prints a Markdown summary table; redirect stdout to capture it
verbatim into a release note.

---

## Architecture

<div align="center">
  <img src="assets/architecture.svg" alt="Javis architecture" width="100%">
</div>

The full pipeline runs end-to-end: text in, encoded into a Sparse Distributed
Representation, injected into R1 (input cortex), routed via address-event
spikes into R2 (memory cortex) where STDP, iSTDP and homeostasis shape an
engram, then read out by kWTA and an engram dictionary back into a list of
text concepts.

Every box on the diagram corresponds to a real Rust module:

| Stage | Module |
| --- | --- |
| `Text → SDR` | [`crates/encoders`](crates/encoders) |
| `R1 / R2 / AER` | [`crates/snn-core`](crates/snn-core) |
| `Plasticity` | [`crates/snn-core`](crates/snn-core) (`stdp`, `istdp`, `homeostasis`) |
| `Decode` | [`crates/encoders/src/decode.rs`](crates/encoders/src/decode.rs) |
| `Eval / RAG` | [`crates/eval`](crates/eval) |
| `LLM (Anthropic)` | [`crates/llm`](crates/llm) |
| `Live UI` | [`crates/viz`](crates/viz) |

---

## Quick start

```sh
# build everything
cargo build --release

# run the full test suite (98/98 should pass)
cargo test --release

# minimal 30-line demo printing RAG vs Javis token saving
cargo run --release -p eval --example hello_javis

# fire up the live 3D brain in a browser
cargo run -p viz --release --bin javis-viz
# → http://127.0.0.1:7777
```

Optional persistent brain:

```sh
cargo run -p viz --release -- --snapshot brain.json
# trains the bootstrap corpus, persists on Ctrl-C, reloads on next start
```

Optional real Claude API calls (otherwise the LLM adapter runs in mock mode):

```sh
ANTHROPIC_API_KEY=sk-ant-... cargo run -p viz --release
# the "send both to Claude" button now fires real calls
```

### Run with Docker

A multi-stage `Dockerfile` plus a `docker-compose.yml` brings up the
full observability stack — Javis, Prometheus, and Grafana — in one
command:

```sh
docker compose up --build
```

| URL | What |
| --- | --- |
| http://localhost:7777 | Javis 3D brain (WebSocket + frontend) |
| http://localhost:7777/metrics | Prometheus exposition |
| http://localhost:9090 | Prometheus UI (already scraping Javis) |
| http://localhost:3000 | Grafana, Javis dashboard pre-provisioned |

The brain state lives on a named volume (`javis-data:/app/data`),
so `docker compose restart` saves a `brain.snapshot.json` on
shutdown and reloads it on startup — no retraining needed.

The Grafana instance runs anonymous-admin and the Prometheus
datasource is auto-wired — meant for local-dev only, see
`docker-compose.yml` for the relevant `GF_AUTH_*` flags before
exposing it anywhere.

---

## Live 3D brain

Open `http://127.0.0.1:7777` and you get a Three.js / `3d-force-graph` view of
the live brain:

- Two anatomical lobes — R1 input cortex (blue) and R2 memory cortex (yellow)
  with embedded inhibitory cells (pink)
- Spike pulses light each neuron as it fires, fading back over ~220 ms
- A side panel streams phase, live spike rates, decoded concepts, the token
  saving headline and the actual RAG-vs-Javis payloads
- Two text inputs let you live-train sentences and live-query the brain
- A "send both to Claude" button fires both payloads to the Anthropic API in
  parallel and shows the answers + real input/output token counts

---

## Plasticity

Javis composes twelve biologically-motivated plasticity mechanisms, each opt-in:

| Mechanism | Purpose | Reference |
| --- | --- | --- |
| **LIF dynamics** | leaky integrate-and-fire neurons with refractory period | classical |
| **Pair STDP (E)** | Hebbian potentiation between excitatory neurons | Bi & Poo 1998 |
| **iSTDP** | heterosynaptic plasticity at I→E, gives engram selectivity | Vogels et al. 2011 |
| **Asymmetric homeostasis** | scale-only-down multiplicative renormalisation | Turrigiano 2008 |
| **BTSP soft bounds** | `Δw = a · trace · (w_max − w)` instead of hard clamp | Bittner 2017 / Milstein 2024 |
| **Contextual engrams** | fingerprints captured during co-activity, not post-hoc | Tonegawa engram-cell line |
| **Triplet STDP** *(iter-44)* | frequency-dependent LTP via slow `r2` / `o2` traces | Pfister & Gerstner 2006 |
| **Reward-modulated STDP** *(iter-44)* | three-factor learning, dopamine-gated eligibility tag | Frémaux & Gerstner 2016; Izhikevich 2007 |
| **Metaplasticity (BCM)** *(iter-44)* | sliding LTP/LTD threshold per post-neuron | BCM 1982; Cooper & Bear 2012 |
| **Intrinsic plasticity (SFA)** *(iter-44)* | adaptive per-neuron threshold | Desai 1999; Chrol-Cannon 2014 |
| **Heterosynaptic L1/L2 norm** *(iter-44)* | per-post incoming-weight budget | Royer & Paré 2003; Field 2020 |
| **Structural plasticity** *(iter-44)* | sprout + prune to grow/shrink topology | Yang 2009; Holtmaat & Svoboda 2009 |
| **Offline replay / consolidation** *(iter-44)* | drives top-k engram cells with plasticity on | Buzsáki 2015; Wilson & McNaughton 1994 |

The math behind each lives in `crates/snn-core/src/{stdp,istdp,homeostasis,
metaplasticity,intrinsic,heterosynaptic,structural,reward,replay}.rs`,
the trade-offs are documented in [`notes/`](notes), and the full iter-44
rationale is in [`notes/44-breakthrough-plasticity.md`](notes/44-breakthrough-plasticity.md).

---

## Token efficiency — the small-corpus picture

Two integration tests measure Javis against a naïve RAG baseline on small,
hand-curated corpora. The numbers here are favourable to Javis (each query
returns a single decoded concept, full RAG returns the whole paragraph) and
are the *floor* of the architecture's reach, not its ceiling:

| Corpus | Mean RAG | Mean Javis | Mean reduction |
| --- | ---: | ---: | ---: |
| 3 paragraphs about programming languages | 27 tok | 2.3 tok | 91.3 % |
| 5 Wikipedia-shaped paragraphs (geology, transport, biology, …) | 60 tok | 2.0 tok | 96.6 % |

These are the *ideal-conditions* numbers. For the benchmark that includes
every realistic failure mode — cross-bleed, missed co-occurrences, decoder
saturation — read [Performance profile](#performance-profile) above.

```sh
cargo test -p eval --release token_efficiency  -- --nocapture
cargo test -p eval --release wiki_benchmark    -- --nocapture
```

---

## Production readiness

What separates Javis from a typical research demo:

**Observability** (notes 24–26)

| Endpoint | Purpose |
| --- | --- |
| `tracing` + `RUST_LOG` | structured logs, JSON mode via `JAVIS_LOG_FORMAT=json`, per-WebSocket-session spans |
| `GET /health` | liveness — always 200 |
| `GET /ready` | readiness — JSON with `sentences`, `words`, `llm` mode |
| `GET /metrics` | Prometheus exposition: counters, histograms (5 ms – 30 s buckets), gauges |

**Supply-chain** (notes 27–30)

| Tool | Where | Catches |
| --- | --- | --- |
| `cargo-deny` | CI `deny` job | RustSec advisories, license drift, banned/duplicate crates, unknown sources |
| Pinned MSRV (1.86) | CI `msrv` job | accidental use of newer-rustc-only features |
| Dependabot | weekly | grouped `cargo` and `github-actions` updates |
| `cargo doc -D warnings` | CI `docs` job | broken intra-doc links, invalid codeblock attrs |

**Container** (notes 32–33)

| | |
| --- | --- |
| Multi-stage `Dockerfile` | `rust:1.86-bookworm` builder → `debian:bookworm-slim` runtime, ~150 MB final |
| Non-root user | `javis` (uid 1000) with `tini` as PID 1 |
| HEALTHCHECK | `curl /health`, 15 s interval |
| Snapshot volume | `javis-data:/app/data` survives restarts |
| Optional CA secret | for sandbox / corporate-proxy environments |

**Performance baselines** (note 31, local x86_64 Linux)

| Path | Time |
| --- | ---: |
| `Network::step` (1 000 neurons, sparse, passive) | 3.2 µs |
| `Network::step` (1 000 neurons, sparse, +STDP) | 3.4 µs |
| `Network::step_immutable` (1 000 neurons, recall path, post-SoA) | **2.7 µs** |
| `Brain::step` (two regions × 1 000) | 7.7 µs |
| `encode_sentence` (18 words) | 21 µs |
| `decode_strict` (vocab 1 000) | 253 µs |

**End-to-end load profile** (note 41, against `docker compose` stack)

| Concurrent WS clients | Throughput | p50 / p99 latency | Server-mean |
| ---: | ---: | ---: | ---: |
| 1 | 138 ops/s | 7.2 / 8.9 ms | 5.8 ms |
| 10 | 430 ops/s | 22.5 / 41 ms | 7.5 ms |
| 50 | 436 ops/s | 116 / 197 ms | 7.6 ms |
| 100 | 432 ops/s | 229 / 486 ms | 7.6 ms |

Recall runs against an `Arc<RwLock<Inner>>` with a per-call
`BrainState`, so multiple recalls proceed in parallel. After the
SoA refactor (note 41), server-side latency is ~7.6 ms across all
concurrency levels — Brain step is now ~4.5 ms / recall, ws-stream
0.31 ms, decode 0.13 ms.

CI runs eight jobs on every push: `fmt`, `clippy -D warnings`,
`test`, `doc-tests`, `deny`, `msrv`, `docs`, `benches` (compile-only).

---

## Project structure

```
javis/
├── crates/
│   ├── snn-core/   ─ LIF neurons, STDP, iSTDP, homeostasis, BTSP, AER routing
│   ├── encoders/   ─ Text → SDR (DefaultHasher, k-of-n) + EngramDictionary
│   ├── eval/       ─ Token-efficiency benchmarks vs. naive RAG
│   ├── llm/        ─ Anthropic API adapter (real + deterministic mock)
│   └── viz/        ─ Axum + WebSocket server, 3D-force-graph frontend
├── notes/          ─ 43 research notes — every decision documented
├── scripts/        ─ End-to-end sanity check + load test (Python)
├── deploy/         ─ Prometheus + Grafana provisioning for docker-compose
└── assets/         ─ Logo and architecture diagram (programmatic SVG)
```

---

## Tests

```sh
cargo test --release
```

| Suite | Tests | Validates |
| --- | ---: | --- |
| `snn-core` | 54 | LIF dynamics, STDP & iSTDP, homeostasis, BTSP soft bounds, E/I balance, multi-region routing, snapshot serde, assembly formation, bounds-checked APIs, heap pending queue, AMPA/NMDA/GABA channels, read-only step equivalence |
| `snn-core` iter-44 | 15 | triplet STDP, reward-modulated STDP / eligibility, BCM metaplasticity, intrinsic plasticity, heterosynaptic L2, structural sprout/prune, offline replay/consolidation, full-stack composite, passive-network regression guard |
| `encoders` | 24 | SDR union/overlap, hash determinism, top-k decode, threshold-floor decode (iter 44.1), injection, full pattern completion |
| `eval` | 13 | RAG-vs-Javis token efficiency, Wikipedia scaling, intra-topic recall, contextual mode, scale-bench smoke |
| `llm` | 3 | Anthropic adapter mock contract, token heuristic |
| `viz` | 16 | WebSocket smoke, train+recall, ask both, snapshot round-trip, `/health` + `/ready`, `/metrics`, concurrency cap, snapshot schema migration (v1→v2) |
| Doc-tests | 3 | Public quick-start examples in `snn-core` and `encoders` |
| **Total** | **130** | with **zero clippy warnings** workspace-wide |

---

## Documentation

Every iteration is logged in [`notes/`](notes). Each note explains
**what changed, why, and what was measured**:

| Note | Topic |
| --- | --- |
| 00 | Architecture sketch |
| 01 | snn-core baseline |
| 02 | Assembly formation + throughput budget |
| 03 | E/I balance + sparse adjacency |
| 04 | Multi-region AER |
| 05 | Encoder stub |
| 06 | Pattern completion |
| 07 | Homeostatic scaling |
| 08 | Pattern completion with homeostasis |
| 09 | Decoder |
| 10 | Multi-concept coexistence |
| 11 | iSTDP — intrinsic selectivity |
| 12 | Token-efficiency benchmark |
| 13 | Live visualisation iter 1 (raster) |
| 14 | Live visualisation iter 2 (3D brain) |
| 15 | Live visualisation iter 3 (persistent training) |
| 16 | Live visualisation iter 4 (Claude API) |
| 17 | Persistence (snapshots) |
| 18 | Wikipedia scaling |
| 19 | Two decode modes |
| 20 | Bio-inspired optimisations: contextual engrams + BTSP |
| 21 | Architecture hardening: dead code, bounds checks, lints |
| 22 | Min-heap pending queue, AMPA/NMDA/GABA channels, zero lints |
| 23 | Production polish: CI, doc-tests, examples, CHANGELOG |
| 24 | Structured logging via `tracing` (RUST_LOG, JSON mode, session spans) |
| 25 | `/health` (liveness) + `/ready` (readiness with brain stats) |
| 26 | Prometheus metrics: `/metrics` endpoint, counters/histograms/gauges |
| 27 | Supply-chain hygiene: `cargo-deny` (advisories + licenses + bans + sources) |
| 28 | MSRV pinned to Rust 1.86, dedicated CI job |
| 29 | Dependabot (cargo + github-actions, grouped weekly updates) |
| 30 | `cargo doc -D warnings` as CI gate |
| 31 | Criterion benchmarks for `Network::step`, `Brain::step`, encode/decode |
| 32 | Container & deploy: Dockerfile + docker-compose with Prometheus + Grafana |
| 33 | Docker stack verified end-to-end + snapshot volume |
| 34 | End-to-end sanity script + Grafana datasource UID fix |
| 35 | Load test: ~141 recalls/sec sustained, Mutex-serialised, no leak |
| 36 | Concurrency cap: Semaphore + 503/Retry-After, `JAVIS_MAX_CONCURRENT_SESSIONS` |
| 37 | Snapshot schema versioning: v2 with metadata, migration chain, v1 backward-compat |
| 38 | Read-only recall: `Brain::step_immutable` + `RwLock`, 2.5× throughput |
| 39 | Profile-driven LIF rewrite: pre-summed channel buffer, 1.5× faster step |
| 40 | Pipeline profile: brain compute is 77 % of recall — not Amdahl-bound yet |
| 41 | AoS → SoA refactor + WS fire-and-forget: 1.40× pipeline, 2× LIF total |
| 42 | Validation-at-scale: honest 100-sentence benchmark, FP/FN/recall metrics |
| 43 | Topology scaling: R2 2 000→10 000, sparser connectivity, retuned iSTDP |
| 44 | **Breakthrough plasticity stack**: triplet-STDP, R-STDP, BCM metaplasticity, intrinsic plasticity, heterosynaptic norm, structural plasticity, offline replay |
| 44.1 | Decoder confidence floor (`--decode-threshold`): FP −86%, token reduction +2× |
| 45 | **Reward-aware pair-association harness**: dopamine + eligibility tag exercised end-to-end, honest "no convergence yet" finding documented |
| 46 | **Teacher-forcing harness**: R2 target-clamp + 6-phase schedule + R1→R2 gate + anti-causal STDP fix; `clamp = 1.00`, but R1→R2 forward dominance survives — honest diagnosis of next bottleneck |
| 47a | **Forward-drive scaling + adaptive θ**: INTER_WEIGHT sweep + Diehl-Cook intrinsic plasticity + 5 sparsity metrics; INTER_WEIGHT 1.0 + adaptive θ produces *first* monotone learning signal (target_hit 1.16 → 2.59), but bistability at 0.7 proves k-WTA is necessary for iter-48 |
| 47a-pm | **Postmortem diagnostics**: 16-epoch saturation test (selectivity *collapses* in epochs 5–15) + per-step cascade trace (oscillatory bursting, NOT onset-burst, early/late ratio 0.97) + θ effect-size measurement (0.05 mV mean, < 0.3 % of 15 mV LIF swing). Reverses iter-48 plan: k-WTA out, **fast iSTDP (Vogels 2011) in** |
| 48 | **iSTDP-tightening**: `R2_INH_FRAC 0.20→0.30`, `tau_minus 30→8 ms`, `a_plus 0.10→0.30` + new p99 / θ_E / θ_I metrics + `--istdp-during-prediction` A/B flag. Phase 1 smoke (4 ep × 2 configs): **selectivity flipped from −0.045 → +0.0142 stable** for the first time in the chain, `r2_act_mean` in [25, 70] band, no cascade. Acceptance 1.5/3 (selectivity ✅, target_hit/p99 ❌) — paused per protocol, no Phase 2 |
| 48-sat | **Phase A saturation postmortem**: 16 epochs × both configs, identical trajectory: positive-selectivity *peak* through epochs 1–4 (no iter-44/45/46/47a config ever achieved this), **hard collapse epoch 5**, stable negative −0.008 to −0.012 thereafter. Acceptance 0/3 in both configs ⇒ Postmortem per protocol. Mechanism: **iSTDP cumulative over-inhibition** — distinct from cascade / runaway / θ-overcorrection (w̄ stable, r2_act DROPS at collapse). iter-49 should explore the *under-tuned* side of the boundary (cap w_max, halve a_plus, or activity-gate iSTDP) |
| 49 | **iSTDP bounds & schedule sweep**: 3 orthogonal axes (WmaxCap, APlusHalf, ActivityGated), 3 distinct failure modes — A never positive (Bekos's 60 %-prior hypothesis falsified), B same collapse epoch with higher peak (+0.0184), C **hyperactivity lock** (`r2_active = 1400` = entire pool fires every trial). 0/3 produce positive learning. **iSTDP is not the primary lever** — the 15× STDP-vs-iSTDP rate asymmetry is the actual bottleneck. iter-50 hypothesis (by elimination): raise STDP `a_plus` 0.020 → 0.060 + `w_max` 0.8 → 2.0 |
| 50 | **Arm B reproduction (Bekos diagnostic)**: `--iter46-baseline` flag reverts INTER_WEIGHT/R2_INH_FRAC/iSTDP/intrinsic at runtime. Result: **iter-46 Arm B's top-3 = 0.19 reproduces** on current branch code — **3× iter-48's 0.06 with full teacher architecture**. The `selectivity_index` metric (iter-47-49) is structurally meaningless in the no-teacher path (compares arbitrary hash SDR vs unrelated firing); 5 iterations were optimised against the wrong metric. Iter-51 = reductive Arm B parameter study, NOT a-plus sweep, NOT bridge |
| 51 | **Arm B 16-epoch saturation (the harder read)**: top-3 oscillates 0.06↔0.19 across all 16 epochs, **mean = 0.107** vs random baseline 0.094, 95 % CI `[0.069, 0.145]` includes random — **statistically not distinguishable from chance**. The 0.19 hits in epochs 0/8/9 are noise peaks of a chance-level distribution. Top-1 = 0.00 every epoch. Mean reward 5–10 % less negative than random. **The whole iter-44…50 chain optimised against a baseline never verified to be above chance.** iter-52 = statistical validation (multi-seed + untrained control + trial-to-trial Jaccard), NOT mechanism work |
| 52 | **Untrained-brain control (`--no-plasticity`)**: gate every `enable_*` plasticity call + L2-norm bit-identity assertion. Sanity assertion immediately caught a **9× weight blowup** in the first run from two un-gated mid-trial enable/disable cycles. After all three gate sites closed: 4 seeds × 16 epochs all bit-identical. Untrained top-3 = **0.039** (95 % CI [−0.008, 0.086]) — significantly *below* random 0.094. Trained vs untrained Δ = **0.068, ≈ 2.2 σ**: plasticity IS doing something. The iter-51 "indistinguishable from chance" reading was too conservative — wrong null. Branch: **Mess-Frage** (decoder bias on fresh brain saturates the metric). iter-53 = decoder-relative readout, NOT new mechanism, NOT parameter sweep |
| 53 | **Decoder-relative Jaccard (Option B Voll)**: 32-cue × 3-trial matrix with full `brain.reset_state()` between trials (R1 + R2 + cross-region queue + traces, not just R2). Same-cue Jaccard = consistency, cross-cue Jaccard = specificity, Δ-of-Δ = engram formation indicator. Untrained arm (no plasticity, full reset) hits `same_cue_mean = 1.0` exactly (state-reset assertion); trained arm keeps plasticity ON during eval per Bekos's spec, so trial 2 depends on trial 1 *via plasticity, not via membrane state*. Public surface: `JaccardMetrics` / `run_jaccard_bench` / `render_jaccard_sweep` + `--jaccard-bench --seeds N1,N2,…` CLI. Direction caveat: trained same-cue ≤ untrained = 1.0 by construction — read as "how close to 1.0 the trained arm stays under continued plasticity" (engram attractor strength). 4 seeds × 16 epochs sweep: trained same = 0.879 ± 0.039 (engram has moderate attractor strength + plasticity drift, eval-phase L2 +25 to +29); cross-cue **flat at 0.058 ± 0.003 in both arms** (no cue-specificity gain); **Δ-of-Δ = −0.121, FAILED**. iter-54 must address cue-specificity at the architecture or schedule layer (decorrelated initial projections / reward cue-specificity / cue-only schedule via `--association-training-gate-r1r2`), not at the metric layer |
| 62 | **Recall-mode (plasticity-off-during-eval)**, single-bit intervention. New `--plasticity-off-during-eval` flag (alias `--recall-mode-eval`) disables every plasticity rule (STDP / iSTDP / homeostasis / intrinsic / reward / metaplasticity / heterosynaptic / structural) between training and the jaccard-matrix eval phase; pre/post L2 norms must match bit-for-bit (asserted). Sweep at vocab=64 + DG + recall-mode, 4 seeds × 32 epochs. Per-seed: trained_same went from iter-61's heterogeneous (0.961 / **0.875** / **0.898** / 0.930, 2 of 4 below 0.90) to **1.000 on 4/4 seeds**. Eval-drift L2 went from 0.9–4.6 (with seed 13's sign-flip) to **0 bit-identical on 4/4 seeds**. Cross-cue stayed at the floor (trained 0.028 ± 0.001 vs iter-61's 0.026 ± 0.001 — recall-mode adds ~0.002 to the absolute floor). Δ cross trained-untrained = −0.001 NS — Jaccard is at the geometric floor and no longer measures learning. Per Bekos's iter-63 selector: **branch (A) primary** + **branch (D) secondary**. iter-63 entry: re-wire the iter-46 / 52 direct cue→target metric (canonical-target top-k, target rank, MRR, correct-minus-incorrect) on the DG-enabled brain. Methodological lesson: when a sweep isolates a clean mechanism, the right intervention often has fewer parameters than the wrong one — iter-62 was a single boolean. Headline: **Recall-mode restores same-cue stability; DG separation is now stable under read-only recall** |
| 61 | **DG-bridge full replication** at 4 seeds × 32 epochs (no new code, just the iter-55/56 lesson applied to iter-60). Cross-cue floor robustly reproduces: per-seed 0.025–0.033, aggregate trained 0.026 ± 0.001, untrained 0.029 ± 0.002 — bit-close to the iter-60 smoke (0.026 / 0.028). Cross-seed averaged per-pair distribution: **median 0.000**, p95 0.10, max 0.275 (vs iter-58's max 1.000). **Δ cross trained-untrained = −0.003, paired t(3) ≈ −2.07, p ≈ 0.13 NS** — Jaccard floor saturated, plasticity adds no measurable cue-specific improvement. Per-seed trained_same: 0.961, **0.875**, **0.898**, 0.930 — **2 of 4 seeds drop below the 0.90 stability threshold**; aggregate (0.916 ± 0.037) hides this with 3× wider std than the smoke. Eval-drift L2: +4.6, +1.6, **−0.9** (sign flip = net depression on seed 13), +3.7. Per Bekos's iter-62 selector: **branch (B) PRIMARY** — DG separates but stability erodes. iter-62 entry: Path 1 plasticity-off-during-eval (recall-mode, branch B prescription) + sub-question direct cue→target metric (top-3 against canonical target — Jaccard floor too low to register plasticity-driven learning). Headline: **DG robustly solves cross-cue separation, but Jaccard no longer measures learning, and same-cue erodes on half the seeds under continued plasticity** |
| 60 | **DG pattern-separation bridge** (architecture pivot, not capacity). New 3rd region (DG) with k-of-n hashed per-cue SDRs + sparse mossy-fibre-style DG → R2 projection; direct R1 → R2 path scaled to 0.0 (DG sole cue-routing). New CLI: `--dg-bridge --dg-size N --dg-k K --dg-to-r2-fanout F --dg-to-r2-weight W --direct-r1r2-weight-scale S --dg-drive-strength I`. Smoke at vocab=64 c500 ep16 seeds 42,7 with default DG (size=4000, k=80): **untrained cross 0.448 → 0.028 (−94 %)**, **trained cross 0.422 → 0.026 (−94 %)**, **9× lower than iter-54 vocab=32 best of 0.230 at 2× the vocab**. Δ cross trained-untrained collapses to −0.002 — geometry carries the signal, not plasticity. Same-cue drops 1.000 → 0.922 (engram erosion); eval-drift L2 jumps ~100× (+0.04 → +3.3-4.4) — plasticity is now active at eval but eroding rather than building specificity. Per Bekos's iter-61 selector: **branch (A) MASSIVELY confirmed** (trained 0.026 ≪ 0.25-0.30 target), **branch (B) secondary** (Δ trained-untrained tiny). iter-61 entry: full 4-seed × 32-epoch DG replication + isolate cue→target learning task with a different metric (Jaccard floor is now too low for plasticity to register). Methodological lesson: when every training-axis sweep saturates at the same value, the architecture is the lever, not the hyperparameter — biggest single number-move in 14 iterations came from a structural change, not a training sweep |
| 59 | **R2 capacity scaling for the vocab=64 floor** (no new metric, runtime R2_N override + capacity sweep CLI). Pre-flight ruled out R2_N=16000+ on `R2_N²` recurrent-synapse cost (~120 h+ for ep32 sweep). Sweep at vocab=64 c500 teacher_ms=40 with ep32 (R2=2000 from iter-58, 4 seeds), ep16 (R2=2000 fairness baseline 2 seeds, R2=4000 primary 1 seed), ep4 (R2=8000 smoke 1 seed). At ep16 fixed: Δ cross **deepens 13× from −0.007 (R2=2000) to −0.090 (R2=4000)** but absolute trained_cross only drops 0.04 (0.449 → 0.411), still 0.18 above the vocab=32 best of 0.230. R2=8000 ep4 too undertrained to read. Untrained baseline rises with R2_N (0.448 → 0.501) — KWTA_K=60 over growing R2-E selects an increasingly cue-independent attractor. **Verdict: branch (B) Mixed limit — capacity is *a* limit, not *the* limit.** iter-60 entry: architecture pivot, not "more neurons" — Bekos flagged the Hippocampus / SDM literature (DG/CA3 separation), iter-60 will smoke a DG-like Pattern-Separation bridge. Methodological lesson: the number that matters wasn't trained_cross — it was that Δ deepened 13× while absolute moved 0.04. Plasticity got more room; the read-out floor lives elsewhere |
| 58 | **Jaccard floor diagnosis — geometry vs plasticity** (no new sweep on training axes; one new diagnostic axis: vocab scaling). Path 1 vocab=32 distribution: smooth and continuous (min 0.000 / median 0.225 / max 0.750), no isolated geometric-collision tail; half the vocab participates in many high-overlap pairs. Path 2 vocab=64: trained cross **rises** to 0.422 ± 0.017 (Δ cross collapses from −0.229 to −0.025, paired t(3) ≈ −2.58 p ≈ 0.08, **NOT significant**); per-pair median shifts +0.20; cue frequency in pairs ≥ 0.30 reaches 92 % of cues; actor/ada/array trio at Jaccard = 1.000 (geometric collision on short 4-letter words). **Direction-of-change verdict: branch (B) PRIMARY architecture floor** — bigger vocab halves block_size (43 → 21 cells/cue), trained_cross *rises* (geometric predicts flat/drop). Branch (C) minor secondary (handful of encoder collisions). New public surface: `JaccardPairSample` / `JaccardFloorReport` / `run_jaccard_floor_diagnosis` / `render_jaccard_floor_diagnosis` / `default_corpus_v64` + CLI flags `--jaccard-floor-diagnosis` / `--corpus-vocab 32\|64` / `--floor-threshold` / `--floor-top-n`. iter-59 entry: real architecture question. Path 3 first as positive control (double R2_N to 4000 → block_size returns to ~62 at vocab=64; ~30 min wallclock). Then Path 1 (learnable / weight-mediated R1→R2) or Path 2 (contrastive iSTDP). Methodological lesson: a saturation ceiling has a direction — find the non-training axis where competing models predict opposite signs of change |
| 57 | **Phase-length sweep on decorrelated + c500 + ep32** (no new code, three configs × 4 seeds, teacher_ms 40 / 80 / 120): trained cross **non-monotonic** — t40 = 0.230 (replicates iter-56 c500), **t80 = 0.408 catastrophic** (Δ cross collapses from −0.229 to −0.051, 78 % of signal lost; uniformly bad across all 4 seeds), t120 = 0.248 (recovers most of t40; seed 99 t120 = 0.194 is the global best per-seed value across iter-53…57). Mechanism: lead-in formula `(teacher_ms/4).clamp(4,12)` caps at teacher_ms ≥ 48; t80's 1:5.7 lead:clamp ratio pushes iSTDP/homeostasis past stable without enough consolidation to recover, t120's longer consolidation phase re-settles the system. Same-cue stays 1.000 in 12/12 trained arms; eval-drift L2 *decreases* at higher teacher_ms (branch (D) rejected). Per Bekos's iter-58 selector: **branch (C) PRIMARY** (phase-length is sub-effective lever), branch (B) secondary (best aggregate t40 = 0.230 = iter-56 c500 ceiling, no breakthrough). All three training axes (epoch/clamp/phase-length) now saturate near trained cross ~0.20. iter-58 entry = **shift the research question** — geometric vs plastic limit diagnosis (Path 1, ~5 min code) or vocab-scaling stress test (Path 2, ~30 min). Methodological lesson: a 3-point sweep is the minimum for a non-monotonic axis; 2 points would have missed the t80 dip entirely |
| 56 | **Clamp-strength sweep on decorrelated + ep32** (no new code, three configs × 4 seeds): trained cross 0.272 (c125) → 0.245 (c250) → 0.230 (c500); per-doubling Δ −0.027 then −0.015 (ratio 0.55) ⇒ asymptote ~0.20 combined with iter-55 epoch axis. c250 bit-exactly replicates iter-55 ep32. **c500 has 5× tighter seed std (0.020) vs c125/c250 (0.036/0.041)** — higher clamp flattens the seed distribution. **Half the seeds are non-monotone in clamp**: seed 99's c250 is best with c500 regressing, seed 42's c125 is better than c250. Same-cue stays at 1.000 in 12/12 trained arms (branch δ rejected). Per Bekos's iter-57 selector: **branch (α) magnitude-limited primary**. iter-57 path = Achse C (phase-length tuning, sweep `teacher_ms` 40 → 80, 120 at fixed c500). Methodological lesson: aggregate monotonicity is not seed-level monotonicity — half the seeds break the "higher = better" reading the aggregate alone would set as deployment guidance |
| 55 | **Epoch sweep on decorrelated + plasticity** (no new code, three configs × 4 seeds): trained cross 0.299 (ep16) → 0.245 (ep32) → 0.229 (ep64); per-doubling Δ −0.054 then −0.016 (ratio 0.30) ⇒ asymptote estimate ~0.21. ep16 bit-exactly replicates iter-54. Per-seed: 42 + 99 keep improving every doubling, 7 plateaus at ep32, **13 regresses at ep64** (0.225 → 0.245), **99 alone breaks same-cue at ep64** (0.984 — first time below 1.0 anywhere in iter-53/54/55). Per Bekos's pre-fixed iter-56 branching matrix: **branch (ii) Saturation primary**, branch (iv) eval-plasticity-still secondary. iter-56 entry = Achse B Clamp-Strength-Sweep; noise-injection / cross-topology = parallel iter-57. Methodological lesson: aggregate Δ-of-Δ rose monotonically (+0.160 → +0.214 → +0.226) which would suggest "keep training", but the per-seed view exposed the ceiling the aggregate hid |
| 54 | **Hard-decorrelated R1 → R2 init**: `wire_forward_decorrelated` partitions R2-E into vocab-sized disjoint blocks; each R1 cell appearing in *exactly one* cue SDR fans out `FAN_OUT` times into its owner-cue's block, shared cells dropped. Mechanical invariant `assert_decorrelated_disjoint` enforces pairwise-disjoint R2 reach end-to-end — iter-52-style L2-equivalent topology check. `TeacherForcingConfig.decorrelated_init` + `--decorrelated-init` CLI flag (default OFF preserves iter-46/53 random topology). Block math: vocab = 32, R2-E = 1400 ⇒ 43 cells per cue; ~17 unique R1 cells × 12 = 204 connections per cue (vs 12 000 random). 4 seeds × 16 epochs sweep: trained cross = **0.299 ± 0.038** vs untrained 0.459 ± 0.022 → Δ cross = **−0.160, all 4 seeds same direction, paired t(3) ≈ −16, p ≪ 0.001**. Trained same = 1.000 (no attractor erosion; eval-phase L2 drift collapses from iter-53's +25-29 to +0.02-0.25 because the sparser cue-specific drive starves eval-time plasticity). **Δ-of-Δ = +0.160, ACCEPTANCE PASSED**. iter-55 entry per Bekos's branching matrix = M1: keep decorrelation + plasticity combined, sweep training schedule / epochs / target_clamp_strength to amplify the gain |

---

## References

The plasticity rules and architectural choices come from current SNN
literature. Key papers:

- A. C. Vogels et al. — [Inhibitory Plasticity Balances Excitation and Inhibition](https://www.science.org/doi/10.1126/science.1211095) · _Science_ 2011
- A. D. Milstein et al. — [Rapid memory encoding in a recurrent network model with BTSP](https://pmc.ncbi.nlm.nih.gov/articles/PMC10484462/) · _PLOS Comp Bio_ 2023
- L. Bittner et al. — [Behavioral Time Scale Synaptic Plasticity (Nature Comms 2024)](https://www.nature.com/articles/s41467-024-55563-6)
- Caligiore et al. — [Selective inhibition in CA3](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013267) · _PLOS Comp Bio_ 2024
- L. Hu et al. — [Dynamic and selective engrams emerge with memory consolidation](https://www.nature.com/articles/s41593-023-01551-w) · _Nature Neurosci._ 2024

---

## License

MIT — see [`LICENSE`](LICENSE).
