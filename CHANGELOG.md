# Changelog

All notable changes to Javis. The version line follows the iteration
note that introduced the change — every iteration has a corresponding
`notes/NN-*.md` with the full reasoning, measurements, and references.

## Unreleased — Iteration 16 (end-to-end sanity)

### Fixed
- Grafana dashboard panels referenced datasource `uid: "Prometheus"`
  but the auto-provisioning let Grafana hash a fresh UID, so every
  panel showed "Datasource not found" in the UI. Pinned
  `uid: prometheus` in the datasource provisioning and updated all
  five dashboard panels to match.

### Added
- `scripts/sanity_check.py` — reproducible end-to-end smoke test
  that drives `train` / `recall` / `ask` / parallel-recall flows
  over the WebSocket interface, then asserts on `/ready` deltas
  and `/metrics` counter values. Exits 0 on full pass; 1 on the
  first failed expectation. Targets `localhost:7777` by default,
  override via `JAVIS_HOST`.

### Verified live (notes/34)
- Train → recall → ask → 5 parallel recalls completed cleanly
  against `docker compose up`.
- Snapshot persistence in the live flow: train a new sentence,
  `docker compose restart javis-viz`, then recall both bootstrap
  and live-trained words still works (both ≥ 86 % token reduction).
- Lifetime token saving across the test run: 92.6 %.

## Iteration 15 — container & deploy

### Added
- Multi-stage `Dockerfile` (builder on `rust:1.86-bookworm`, runtime
  on `debian:bookworm-slim`). Final image runs as non-root user
  `javis` (uid 1000) with `tini` as PID 1 and a `curl /health`
  HEALTHCHECK. Layer-cache trick stubs the workspace so `cargo
  fetch` only re-runs on manifest changes.
- Persistent brain volume: `javis-data:/app/data` mount plus
  `--snapshot /app/data/brain.snapshot.json` arg in compose. Brain
  state survives `docker compose restart` (verified locally:
  29.5 MB snapshot, save → load round-trip preserves sentences/words).
- Optional `--secret id=hostca` build-time CA bundle for sandbox /
  TLS-intercepting-proxy environments. Declared in
  `docker-compose.yml` so a single `docker compose up --build`
  works without manual flag-fiddling.

### Fixed
- Dockerfile stub-source step now creates placeholder files for
  `[[bench]]` targets too — without them `cargo fetch` refuses to
  parse the manifest. Discovered when the iter-15 image build was
  first exercised end-to-end.
- `.dockerignore` keeps the build context lean (no `target/`,
  `.git/`, `notes/`, etc.).
- `docker-compose.yml` brings up three services: javis-viz,
  Prometheus 3.0 (scrapes `/metrics` every 15 s), Grafana 11 with
  anonymous-admin access for local demo.
- `deploy/prometheus.yml` — single scrape job for the
  `javis-viz:7777/metrics` endpoint.
- `deploy/grafana/provisioning/` auto-wires Prometheus as the
  default datasource and registers a dashboard provider.
- `deploy/grafana/dashboards/javis-overview.json` — 5-panel
  dashboard: brain sentence/word gauges, lifetime token-saving %,
  WS-sessions-per-action rate, and p95 latency timeseries for
  train/recall/ask.

### Changed
- `viz/src/main.rs` resolves the static-asset directory and bind
  address at runtime via `JAVIS_STATIC_DIR` / `JAVIS_BIND_ADDR` env
  vars. Defaults stay at the source-tree `static/` and `127.0.0.1:7777`
  so `cargo run` keeps working unchanged; the Docker image sets both
  to container-friendly values (`/app/static`, `0.0.0.0:7777`).

## Iteration 14 — performance benchmarks

### Added
- Three Criterion benchmark files: `crates/snn-core/benches/
  network_step.rs`, `crates/snn-core/benches/brain_step.rs`,
  `crates/encoders/benches/encode_decode.rs`. Seven separate bench
  functions covering passive vs. STDP-enabled `Network::step`,
  multi-region `Brain::step` (heap-backed inter-region delivery),
  and the encoder/decoder per-request hot paths.
- `criterion = "0.8"` as a dev-dependency on `snn-core` and
  `encoders`. `default-features = false` keeps the dep tree slim.
- New `benches` CI job runs `cargo bench --workspace --no-run` to
  catch API drift in the bench code without trying to extract real
  perf numbers from a shared GitHub runner.
- Baseline numbers documented in `notes/31-criterion-benchmarks.md`
  (e.g. `network_step_passive/1000` ≈ 3.2 µs,
  `decode_strict/vocab_1000` ≈ 253 µs on local x86_64).

## Iteration 13 — supply-chain hygiene (parts A + B + C + D)

### Added (part D — rustdoc warnings as errors)
- New `docs` CI job runs `cargo doc --workspace --no-deps
  --all-features` with `RUSTDOCFLAGS="-D warnings"`, so broken
  intra-doc links, invalid codeblock attributes, and malformed HTML
  in doc comments fail the build.
- Verified locally: existing 29 iterations of doc comments pass
  with both `-D warnings` and `-D rustdoc::all` — no fixes needed.

### Added (part C — Dependabot)
- `.github/dependabot.yml` tracks `cargo` (root + every crate) and
  `github-actions` (every workflow file) on a weekly schedule.
- Grouped updates: minor/patch bumps batched into one PR per
  ecosystem; the tracing stack and the tokio/tower/axum/hyper stack
  each have their own group so co-versioned crates stay in sync.
- Major bumps stay solo so they get individual review.
- PR commit prefixes `deps:` (cargo) / `ci:` (actions); open-PR
  caps at 5 and 3 respectively.

### Added (part B — MSRV)
- `[workspace.package].rust-version = "1.86"` — explicit MSRV
  contract. Every member `[package]` declares
  `rust-version.workspace = true` so the inheritance is literal.
- New `msrv` CI job in `.github/workflows/ci.yml` runs
  `cargo build --locked` and the full test suite against
  `dtolnay/rust-toolchain@1.86` on every push, so accidental use of
  1.87+-only features fails CI.

### Changed (part B)
- `snn-core/src/network.rs` — replaced `u64::is_multiple_of`
  (stabilised in 1.87) with the equivalent `% == 0`. Discovered
  during MSRV verification.

### Added (part A — cargo-deny)
- `deny.toml` — repository-root `cargo-deny` configuration. Four
  checks: advisories (RustSec), licenses (allow-list), bans
  (wildcard / duplicate detection), sources (only crates.io).
- License allow-list explicitly enumerates the ten permissive
  licenses our dep tree actually contains. `unused-allowed-license =
  "deny"` keeps the list tight.
- `.github/workflows/ci.yml` — new `deny` job runs `cargo-deny
  check` on every push via `EmbarkStudios/cargo-deny-action@v2`.

### Changed (part A)
- `[workspace.package]` declares `publish = false` and every member
  `[package]` adds `publish.workspace = true`. Required so
  `allow-wildcard-paths = true` applies — without it, intra-
  workspace path deps would be flagged as wildcards.

## Iteration 12 — operational hardening (parts A + B + C)

### Added (part C — Prometheus metrics)
- `GET /metrics` — Prometheus exposition endpoint. Returns the global
  recorder snapshot in `text/plain; version=0.0.4` format.
- `viz::metrics::init()` — idempotent, installs a `metrics-exporter-
  prometheus` recorder once per process, configured with histogram
  buckets covering our 5 ms – 30 s operation range.
- Counters: `javis_ws_sessions_total{action}`,
  `javis_recall_tokens_rag_total`, `javis_recall_tokens_javis_total`
  (the difference is total token saving over server lifetime).
- Histograms: `javis_train_duration_seconds`,
  `javis_recall_duration_seconds`, `javis_ask_duration_seconds{real}`,
  `javis_snapshot_duration_seconds{op}`.
- Gauges: `javis_brain_sentences`, `javis_brain_words`.
- Three new tests in `crates/viz/tests/metrics_endpoint.rs`.

### Added (part B — health/readiness probes)
- `GET /health` — liveness probe. Returns `200 ok` as long as the
  HTTP runtime can answer. Cheap enough for sub-second probe intervals.
- `GET /ready` — readiness probe. Returns `200` plus a JSON body
  with `status`, `sentences`, `words`, and `llm` mode (real / mock).
  Both probes are registered on `router` and `router_no_static`.
- Three new tests in `crates/viz/tests/health.rs` driving the router
  via `tower::ServiceExt::oneshot` (no real TCP needed).

### Added (part A — structured logging)
- `tracing` + `tracing-subscriber` for structured logging. Subscriber
  in `viz::main` honours `RUST_LOG` for level/target filtering and
  `JAVIS_LOG_FORMAT=json` to switch from human-readable to JSON output
  (for log aggregators like Loki / ELK).
- Per-WebSocket-session spans with monotonic `session` id and `action`
  field, so concurrent client logs can be disentangled.
- Structured fields on every state-mutating operation:
  `train completed` / `recall completed` / `ask completed` /
  `snapshot saved|loaded` / `brain reset` all carry timing
  (`elapsed_ms`) and outcome counters.

### Changed
- `println!`/`eprintln!` in the production code paths (binary +
  library + `llm` crate fallback) replaced by `tracing` macros at the
  appropriate level. Examples and tests keep their plain stdout output.

## Iteration 11 — production polish

### Added
- GitHub Actions CI workflow (`.github/workflows/ci.yml`) that runs
  `cargo fmt --check`, `cargo clippy -D warnings`, the full release
  test suite, and the doc-test suite on every push and pull request.
- Three doc-tests on the public API (`snn-core` quick-start, two
  `encoders` quick-start examples). They run as part of the standard
  `cargo test --doc` workflow, so the documentation cannot drift
  out of sync with the code.
- `crates/eval/examples/hello_javis.rs` — minimal end-to-end demo
  that trains on the wiki corpus, queries every topic, prints
  the RAG-vs-Javis token comparison. No external dependencies.

## Iteration 10 — heap queue, AMPA/NMDA/GABA channels, zero lints

### Added
- `PendingQueue` — `BinaryHeap`-backed min-heap on arrival time
  with a sequence-tiebreak for FIFO determinism. Replaces the
  `Vec<PendingEvent>` field on `Brain`.
- `SynapseKind { Ampa, Nmda, Gaba }` plus per-network
  `tau_nmda_ms` / `tau_gaba_ms`. Lazy NMDA/GABA buffers — AMPA-only
  networks pay no extra cost.
- `Network::set_synaptic_taus(ampa, nmda, gaba)` setter with
  positive-finite validation.

### Changed
- All `let mut x = Foo::default(); x.field = ...` test-code patterns
  rewritten as struct-init with `..Foo::default()`.
- `len() > 0` → `!is_empty()`, idiomatic slice iteration.

### Result
- 91/91 tests passing, **zero clippy warnings** workspace-wide.
- See `notes/22-heap-channels-lints.md`.

## Iteration 9 — architecture hardening

### Added
- Bounds-checked `Network::connect` and `Brain::connect` with clear
  panic messages naming the bad value.
- Per-network `tau_syn_ms` configurable via `Network::set_tau_syn_ms`.

### Changed
- Removed dead `Synapse.tau_syn` field (was set, never read; the
  decay loop used a hardcoded 5.0).

### Result
- 79/79 tests passing including 16 hardening tests.
- See `notes/21-architektur-haertung.md`.

## Iteration 8 — bio-inspired optimisations

### Added
- BTSP-style soft bounds for STDP (`StdpParams.soft_bounds: bool`).
- `FingerprintMode::Contextual` — engrams captured during training
  co-activity, not by isolated re-stimulation post-training.
- `EngramDictionary::decode_top(active, k)`.

### Result
- 63/63 tests passing including 5 associative-recall tests and 3 BTSP
  tests. See `notes/19-zwei-decode-modi.md` and
  `notes/20-bio-optimierungen.md`.

## Earlier iterations (0–7)

| Iteration | Topic | Note |
| ---: | --- | --- |
| 0–1 | snn-core baseline (LIF, STDP) | `notes/00`–`01` |
| 2 | Assembly formation + throughput | `notes/02` |
| 3 | E/I balance + sparse adjacency | `notes/03` |
| 4 | Multi-region AER | `notes/04` |
| 5 | Encoder stub | `notes/05` |
| 6 | Pattern completion | `notes/06` |
| 7 | Homeostatic scaling | `notes/07`, `notes/08` |

For pre-Iteration-8 details, see the corresponding research notes.

## Format

Loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Categories used per release: `Added`, `Changed`, `Removed`, `Result`.
