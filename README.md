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
