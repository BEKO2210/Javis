# Changelog

All notable changes to Javis. The version line follows the iteration
note that introduced the change — every iteration has a corresponding
`notes/NN-*.md` with the full reasoning, measurements, and references.

## Unreleased — Iteration 12 (operational hardening, parts A + B)

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
