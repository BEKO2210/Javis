# 24 — Strukturiertes Logging via `tracing`

**Stand:** Erster Schritt von Iteration 12 (Operational Hardening). Davor
hat der Server `println!`/`eprintln!` ins stdout gespuckt — keine
Levels, keine strukturierten Felder, keine Korrelation zwischen
WebSocket-Sessions. Damit war das Ding für Produktiv-Logging blind.

Jetzt: alle Logs gehen über `tracing`, jede WebSocket-Session läuft in
einem eigenen Span, jede Operation (train/recall/ask/snapshot) emittiert
strukturierte Felder mit Timing.

## Was eingezogen wurde

Workspace-`Cargo.toml`:

```toml
[workspace.dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "fmt", "json"] }
```

`viz` und `llm` ziehen `tracing` als reguläre Dep. Nur `viz` zieht
`tracing-subscriber` — das ist Sache des Binary, Libraries dürfen
keinen Subscriber installieren.

## Subscriber-Init

`viz/src/main.rs::init_tracing()`:

- Default: `info`-Level human-readable mit Spans an stderr.
- `RUST_LOG=viz=debug` → feingranulares Tuning pro Modul.
- `JAVIS_LOG_FORMAT=json` → strukturiertes JSON für ELK / Loki / etc.

Verifiziert mit beiden Modi:

```sh
$ RUST_LOG=info JAVIS_LOG_FORMAT=json target/release/javis-viz | head
{"timestamp":"…","level":"INFO","fields":{"message":"train completed",
 "sentence_len":134,"new_words":15,"total_sentences":1,"total_words":15,
 "elapsed_ms":719.75},"target":"viz::state"}
```

```sh
$ RUST_LOG=info target/release/javis-viz | head
2026-04-29T01:17:49Z  INFO bootstrapping brain on default corpus
2026-04-29T01:17:49Z  INFO train completed sentence_len=134 new_words=15 …
```

## WebSocket-Session-Spans

Jede `run_session` läuft in einem `info_span!("ws_session", session=…,
action=…)` mit einer monoton steigenden Session-ID aus einem
`AtomicU64`. Zwei Logs pro Session:

```
INFO ws_session{session=42 action="recall"}: session started
INFO ws_session{session=42 action="recall"}: recall completed
       query=rust candidates=5 rag_tokens=24 javis_tokens=2
       reduction_pct=91.7 elapsed_ms=180.3
INFO ws_session{session=42 action="recall"}: session ended events_sent=137
```

Damit korrelieren bei parallelen Clients alle Logs derselben Session
über die `session=N`-Felder, auch wenn die Spike-Streams ineinander
gewebt im stdout landen.

## Strukturierte Felder pro Operation

| Operation | Felder |
| --- | --- |
| `train completed` | `sentence_len`, `new_words`, `total_sentences`, `total_words`, `elapsed_ms` |
| `recall completed` | `query`, `candidates`, `rag_tokens`, `javis_tokens`, `reduction_pct`, `elapsed_ms` |
| `ask completed` | `real`, `rag_input_tokens`, `rag_output_tokens`, `javis_input_tokens`, `javis_output_tokens`, `elapsed_ms` |
| `snapshot saved` / `snapshot loaded` | `path`, `bytes`, `sentences`, `words`, `elapsed_ms` |
| `brain reset` | `dropped_sentences`, `dropped_words` |

Alles ohne Logs aus den heißen Pfaden — kein Log pro Brain-Step, kein
Log pro Spike. Nur Operation-Boundaries und Fehlerfälle. Das ist
billig: ein `tracing::info!` ohne aktiven Subscriber ist quasi gratis,
und der Default-Subscriber ist ein synchroner Writer auf stderr.

## Was nicht migriert wurde

- **`println!` in den `examples/`** bleiben — Beispiele sind keine
  Daemons, sie geben absichtlich CLI-Output für Menschen.
- **`println!` in `crates/snn-core/examples/throughput.rs`** dito.
- **`tests/`** benutzen weiter `eprintln!` für Debug-Output bei
  Failures. Tests sind auch keine Server.

`grep -rn 'println\|eprintln' crates/ --include='*.rs' | grep -v
'/tests/\|/examples/'` liefert jetzt **0 Treffer**.

## Test-Status

- 92/92 Tests grün (keine Test-Logik geändert, nur Logging-Pfade)
- `cargo clippy --all-targets -- -D warnings` → 0
- `cargo fmt --all -- --check` → clean

## Was als nächstes kommt (B + C aus Iter-12-Plan)

- **B** — `GET /health` (Liveness), `GET /ready` (mit Brain-Stats)
- **C** — `GET /metrics` mit Counter / Histogram / Gauge via `metrics`
  + `metrics-exporter-prometheus`
