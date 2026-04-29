# Changelog

All notable changes to Javis. The version line follows the iteration
note that introduced the change — every iteration has a corresponding
`notes/NN-*.md` with the full reasoning, measurements, and references.

## Unreleased — Iteration 24 (validation at scale, honest limits)

### Added
- `crates/eval/src/scale_corpus.rs` — deterministic
  template-driven corpus generator across 8 knowledge domains.
  Reproducible from a `(seed, n_sentences)` pair, no external
  dataset; ground-truth co-occurrence map is recorded by
  construction so precision/recall/FP/FN can be measured against
  a real reference.
- `crates/eval/src/scale_bench.rs` — `ScaleBrain` train-once /
  query-many harness. Per-query metrics: token reduction,
  decoder latency, has-self, false positives (cross-domain
  bleed), false negatives (missed co-occurrences). Aggregated to
  a `ScaleSummary` and rendered as Markdown via
  `ScaleReport::render_markdown()`.
- `crates/eval/examples/scale_benchmark.rs` — CLI runner with
  `--sentences`, `--queries`, `--decode-k`, `--seed` flags.
- `crates/eval/tests/scale_bench_smoke.rs` — small-scale
  regression test (16 sentences, ~8 s) wired into CI.

### Verified (notes/42)
| n_sentences | precision | recall | mean reduction | FP / 6 | mean decode |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1.000 | 0.022 | 34.9 % | 4.75 | 383 µs |
| 100 | 1.000 | 0.021 | 40.6 % | 4.70 | 603 µs |

Honest publication-grade story (notes/42): the headline
"96.7 % token reduction" was on a 5-paragraph corpus and is not
reproducible at scale. On a 100-sentence corpus with 286 unique
vocabulary words Javis hits 100 % self-recall and ~40 % token
reduction, but associative-recall drops to ~2 % and
cross-domain bleed dominates the decoded output. Engram
capacity at the current R2 size (2000 neurons, KWTA_K=220) is
the next architectural wall.

## Iteration 23 — AoS → SoA + WS fire-and-forget, 1.4× pipeline

### Changed (`snn-core`)
- **AoS → SoA refactor.** `LifNeuron` now holds *only* `params` and
  `kind` — 32 B per neuron, 2 fitting in one cache line. The
  per-neuron transient state (`v`, `refractory_until`,
  `last_spike`, `activity_trace`) moved to parallel `Vec<f32>`
  buffers on `Network`, indexed lock-step with `neurons`.
- `LifNeuron::step` removed. The LIF math is now inlined in
  `Network::step` and `Network::step_immutable`, operating on the
  parallel slices directly — single straight-line loop, no per-
  neuron struct loads.
- `Network::add_neuron`, `reset_state`, `ensure_transient_state`
  updated to manage the new parallel Vecs.
- `Network::apply_synaptic_scaling` reads `self.activity_trace[post]`
  instead of `self.neurons[post].activity_trace`.

### Changed (`viz`)
- `tx.send(Event::Step{…}).await` → `tx.try_send(...)` in
  `run_with_cue_streaming_immutable`. Step events are
  visualisation breadcrumbs; if the WS consumer falls behind we
  drop the event into a new `javis_ws_step_dropped_total` counter
  rather than awaiting backpressure into the simulation loop.

### Tests
- `lif_basic.rs` rewritten to drive a one-neuron `Network` instead
  of calling the (removed) `LifNeuron::step`.
- `homeostasis.rs`, `injection.rs`, `immutable_step_equivalence.rs`
  updated to read transient state from the new parallel Vecs.
- All four spike-bit-identity equivalence tests still pass without
  modification — SoA path produces bit-identical spikes to the
  pre-refactor path.

### Verified (notes/41)
Criterion, `Network::step_immutable`, p < 0.05:

| size | iter 21 | iter 23 | improvement | total since iter 20 |
| ---: | ---: | ---: | ---: | ---: |
| 100  | 307 ns | 240 ns | -22.5 % | 2.27× |
| 500  | 1.66 µs | 1.17 µs | -26.7 % | 2.40× |
| 1000 | 3.73 µs | 2.69 µs | -26.8 % | 2.17× |
| 2000 | 9.32 µs | 7.35 µs | -18.6 % | 1.88× |

Pipeline (200 sequential recalls): total 8.05 ms → 5.77 ms (-28 %).
Load test: throughput 358 → 432 ops/s (+21 %), p99 -14 % to -19 %
across concurrency 1/10/50/100.

## Iteration 22 — pipeline profile: 77 % brain, NOT Amdahl-bound

### Added
- Six phase timers in `AppState::run_recall` covering
  `lock_overhead`, `encode`, `snn_compute`, `decode`,
  `rag_search`, `response_build`. Plus two sub-phase timers
  inside `run_with_cue_streaming_immutable` (`brain_compute`,
  `ws_stream`) so the dominant `snn_compute` phase can be split
  further.
- Prometheus histograms `javis_recall_phase_seconds{phase}` and
  `javis_recall_subphase_seconds{phase}`, sharing the existing
  duration-bucket layout from `viz::metrics::init`.
- Structured `tracing::info!` line per recall with all six phase
  durations as fields.
- `scripts/pipeline_profile.py` — drives N recalls, reads the
  per-phase histograms before+after, prints the breakdown sorted
  by mean. Defaults: 100 sequential recalls.

### Verified (notes/40)
Across 200 sequential recalls against the docker stack:

| phase | mean ms | share |
| --- | ---: | ---: |
| `snn_compute` | 7.91 | 97.9 % |
| `decode` | 0.13 | 1.6 % |
| `response_build` | 0.02 | 0.2 % |
| `lock_overhead` | 0.009 | 0.1 % |
| `encode` | 0.004 | 0.0 % |
| `rag_search` | 0.002 | 0.0 % |

`snn_compute` breakdown: `brain_compute` 6.18 ms (77 % of total
recall), `ws_stream` 0.53 ms (6.5 %).

Conclusion: brain-compute is still the dominant cost. Amdahl
hasn't taken over yet — a 2× brain-step speedup still buys 1.65×
on the full pipeline.

## Iteration 21 — profile-driven LIF speedup, 1.5×

### Added
- `crates/snn-core/examples/profile_step_immutable.rs` — hand-
  instrumented phase-by-phase profiler for `Network::step_immutable`.
  Three `Instant::now()` brackets around decay / LIF integration /
  delivery; runs 5000 steps and reports mean/p50/p99 plus share of
  total step time. Used in lieu of perf, which is unavailable in
  the dev sandbox.
- `network_step_immutable` benchmark in
  `crates/snn-core/benches/network_step.rs` covering sizes
  100/500/1000/2000.
- `total_input: Vec<f32>` scratch buffer on `NetworkState` for
  pre-summed channel input.

### Changed (`snn-core`)
- `Network::step_immutable` rewritten for autovectoriser-friendly
  inner loop: NMDA / GABA presence checks hoisted outside the
  per-neuron loop, channels pre-summed into `state.total_input`
  via four specialised straight-line loops, no per-iteration
  `Option::get` accesses. Spike-bit-identity to the mutating path
  preserved.

### Verified
- Profile (notes/39): LIF integration is 92 % of step time (decay
  4 %, deliver 4 %), so optimising the LIF loop is the right
  target.
- Criterion bench across sizes 100/500/1000/2000:

  | size | before | after | speedup |
  | ---: | ---: | ---: | ---: |
  | 100  | 545 ns | 307 ns | **1.78×** |
  | 500  | 2.81 µs | 1.66 µs | 1.69× |
  | 1000 | 5.85 µs | 3.73 µs | 1.57× |
  | 2000 | 13.79 µs | 9.32 µs | 1.48× |

  All p < 0.05.

## Iteration 20 — read-only recall: 2.5× throughput

### Added (`snn-core`)
- `NetworkState` and `Network::step_immutable(&self, &mut state,
  external)` — read-only step path that does the same LIF / synaptic
  delivery math as `Network::step` but never mutates synapses,
  ignores plasticity unconditionally, and writes every transient
  buffer to the caller-provided `state`.
- `BrainState` and `Brain::step_immutable(&self, &mut state,
  externals)` — same pattern at the multi-region orchestration
  layer; per-region `NetworkState`s and a per-recall `PendingQueue`
  live in the state argument.
- `Network::fresh_state()` and `Brain::fresh_state()` constructors.
- Four equivalence tests in
  `crates/snn-core/tests/immutable_step_equivalence.rs` proving
  spike-bit-identity with `Network::step` / `Brain::step` when
  plasticity is off, and that the read-only path leaves the brain's
  weights/clock untouched.

### Changed (`viz`)
- `AppState.inner` switched from `Mutex<Inner>` to `RwLock<Inner>`.
  Train / reset / snapshot-load take the write lock; recall, stats,
  snapshot-save take the read lock. Multiple concurrent recalls now
  proceed in parallel.
- `run_recall` builds a per-call `BrainState` and runs through
  `Brain::step_immutable` against the shared brain.
- The `disable_stdp_all() / disable_istdp_all() /
  disable_homeostasis_all()` voodoo at the start of recall is gone
  — the read-only step ignores plasticity unconditionally.

### Verified (notes/38)
Re-running `scripts/load_test.py --levels 1,10,50,100` against the
docker stack:

| concurrency | throughput | p99 client | server-mean |
| ---: | ---: | ---: | ---: |
| 1 | 112 ops/s | 11 ms | 7 ms |
| 10 | 357 ops/s | 48 ms | 9 ms |
| 50 | 359 ops/s | 244 ms | 9 ms |
| 100 | 358 ops/s | 564 ms | 9 ms |

Throughput 2.5× the Mutex-bottlenecked baseline; server-side latency
constant ~9 ms across all concurrency levels (was 7→68→346→685 ms).
The remaining client-side queueing is tokio runtime, not Brain.

## Iteration 19 — snapshot schema versioning

### Added
- Snapshot schema bumped to v2; new mandatory `metadata` block
  records `created_at_unix` and `javis_version` for ops triage.
- Migration framework: a `MIGRATIONS: &[(u32, MigrationFn)]` table
  walks the chain on load, so a snapshot at version `N` is parsed
  as `N`, transformed into `N+1`, … into the current version
  before the canonical struct is deserialised. Adding v3 later
  needs only one new entry in the table.
- `migrate_v1_to_v2` injects the synthesised metadata
  (`created_at_unix: 0`, `javis_version: "migrated-from-v1"`) so
  pre-v2 snapshots load on a v2 build.
- Refusing future versions with a clear error message — we cannot
  downgrade safely.
- Four new tests in `crates/viz/tests/snapshot_migration.rs`:
  current-version round-trip, v1-loads-through-migration,
  future-version-rejected, missing-version-field-rejected.

### Changed
- `load_from_file` now logs the schema migration explicitly when
  it happens (`from_version`, `to_version` fields), so operators
  see in their logs that an old snapshot was upgraded.

## Iteration 18 — concurrency cap

### Added
- `Semaphore`-based cap on simultaneous WebSocket sessions, default
  32. Configurable via `JAVIS_MAX_CONCURRENT_SESSIONS`.
- When the cap is reached, the upgrade handler responds with
  `503 Service Unavailable` + `Retry-After: 1` instead of letting
  the request queue indefinitely on the inner brain Mutex.
- New counter `javis_ws_rejected_total{action,reason}` tracks
  rejections; a single `reason="concurrency_cap"` label today,
  extensible for future rejection paths.
- `AppState::with_session_cap(cap)` constructor for tests that need
  to exercise the rejection path.
- Two new integration tests in `crates/viz/tests/concurrency_cap.rs`
  speaking raw HTTP/1.1 to observe the 503 response (the WS client
  hides non-101 statuses as handshake errors).

## Iteration 17 — load test

### Added
- `scripts/load_test.py` — drives N concurrent WebSocket recall
  sessions for a fixed duration, then reports throughput, client-
  and server-side latency percentiles, and cross-checks them
  against the `javis_recall_duration_seconds_count` Prometheus
  histogram. Default sweep: concurrency 1, 10, 50, 100 × 15 s.

### Verified
- Server sustains ~141 recalls/sec single-tenant (Mutex-serialised
  recall path).
- Latency scales linearly with concurrency: p99 11 ms at c=1,
  84 ms at c=10, 397 ms at c=50, 771 ms at c=100. No errors, no
  drops across 8 277 recalls.
- Memory footprint: 27 MiB idle → 35 MiB peak under sustained
  load → 34 MiB after cool-down. No leak.
- Documented bottleneck (`Arc<Mutex<Inner>>`) and three potential
  scaling paths in `notes/35-load-test.md`.

## Iteration 16 — end-to-end sanity

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
