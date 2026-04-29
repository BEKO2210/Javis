# 26 — Prometheus-Metriken

**Stand:** Teil C von Iteration 12 (Operational Hardening). Damit ist
der Plan aus Notiz 24 abgeschlossen. Der Server hat jetzt alle drei
Säulen (Logs, Probes, Metrics), die ein Prom-/Grafana-Stack braucht,
um aus dem Demo-Server einen monitorbaren Service zu machen.

## Stack

| Crate | Version | Zweck |
| --- | --- | --- |
| `metrics` | 0.24 | Facade-Macros (`counter!`, `histogram!`, `gauge!`) |
| `metrics-exporter-prometheus` | 0.18 | Recorder + Prometheus-Exposition-Format |

`metrics` ist die `tracing`-Analogie für Metriken: ohne installierten
Recorder sind die Macros No-Ops. Der Recorder ist process-global, wird
in `viz::main` einmal über `viz::metrics::init()` gesetzt, gibt einen
`PrometheusHandle` zurück, der in einem `OnceLock` lebt.

## Endpoint

`GET /metrics` → `200 text/plain; version=0.0.4`, Body ist die rohe
Prometheus-Exposition. Idempotent — leere Antwort wenn der Recorder
nie initialisiert wurde (Prometheus interpretiert das als „keine
Metriken yet", nicht als Fehler).

```
$ curl localhost:7777/metrics

# TYPE javis_brain_words gauge
javis_brain_words 43

# TYPE javis_brain_sentences gauge
javis_brain_sentences 3

# TYPE javis_train_duration_seconds histogram
javis_train_duration_seconds_bucket{le="0.005"} 0
javis_train_duration_seconds_bucket{le="0.5"} 0
javis_train_duration_seconds_bucket{le="1"} 3
javis_train_duration_seconds_bucket{le="2.5"} 3
javis_train_duration_seconds_bucket{le="+Inf"} 3
javis_train_duration_seconds_sum 2.209587657
javis_train_duration_seconds_count 3
```

## Metriken-Inventar

| Name | Typ | Labels | Bedeutung |
| --- | --- | --- | --- |
| `javis_ws_sessions_total` | counter | `action` | WebSocket-Sessions pro Action |
| `javis_train_duration_seconds` | histogram | — | Wall-time einer `train`-Operation |
| `javis_recall_duration_seconds` | histogram | — | Wall-time einer `recall`-Operation |
| `javis_ask_duration_seconds` | histogram | `real` | Wall-time eines `ask`, getrennt nach real/mock LLM |
| `javis_snapshot_duration_seconds` | histogram | `op` | save vs load, Wall-time |
| `javis_brain_sentences` | gauge | — | Aktuelle Anzahl trainierter Sätze |
| `javis_brain_words` | gauge | — | Aktuelle Wörterbuch-Größe |
| `javis_recall_tokens_rag_total` | counter | — | Σ Token, die RAG-Path verbraucht hätte |
| `javis_recall_tokens_javis_total` | counter | — | Σ Token, die Javis tatsächlich verbraucht |

`javis_recall_tokens_rag_total - javis_recall_tokens_javis_total` ist
über die Lebenszeit des Servers exakt die Token-Ersparnis. Das macht
sich gut auf einem Grafana-Panel.

## Histogram-Buckets

Default ist Summary; das gibt aber keine Bucket-Counts und macht
Latency-SLOs unmöglich. Wir setzen explizit:

```rust
const DURATION_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
];
```

Eine Bucket-Vector pro Operation wäre feiner aber nicht nötig — alle
unsere Operationen (recall ~30 ms, train ~700 ms, ask ~1-3 s) liegen
in diesen Buckets. `Matcher::Suffix("duration_seconds")` greift damit
für *jedes* Histogram im Code, ohne dass man jedes neu registrieren
muss.

## Init-Flow

```rust
// viz/src/main.rs
init_tracing();      // Subscriber für Logs
viz::metrics::init(); // Recorder für Metrics
// ...router/listener wie gehabt...
```

`init()` ist idempotent (`OnceLock::get_or_init`), gibt `bool` zurück
ob das aktuelle Call die Installation gemacht hat. Tests, die das
Recorder-Setup brauchen, rufen `init()` einfach am Anfang auf — der
zweite und dritte Test im selben Binary-Prozess kriegt den schon
installierten Recorder zurück.

## Tests

`crates/viz/tests/metrics_endpoint.rs`, drei Tests:

- `metrics_endpoint_returns_prometheus_text` — installiert den
  Recorder, läuft 1 train, scrape'd /metrics, prüft Content-Type +
  Anwesenheit von `javis_train_duration_seconds`, `javis_brain_*`.
- `metrics_endpoint_works_before_any_operation` — nackter State, keine
  Operation, /metrics gibt trotzdem 200 (Prometheus-Konvention).
- `ws_session_counter_increments_on_each_session` — echter TCP-Server,
  zwei WebSocket-Recall-Requests, dann /metrics scrape; verifiziert
  dass `javis_ws_sessions_total{action="recall"} ≥ 2`.

Alle drei nutzen `tower::ServiceExt::oneshot` für die HTTP-Probes; nur
der dritte Test braucht einen echten Listener für die WS-Sessions.

## Hot-path-Cost

`metrics::counter!`/`histogram!`/`gauge!` Macros expandieren zu Lookups
in einem `dashmap`-backed Index. Im hot path ist das sub-µs — weit
unter den ms-Skalen unserer Operationen. Wir messen nicht *innerhalb*
von Brain-Steps, sondern nur an Operation-Boundaries (genau wie beim
`tracing::info!`-Pendant), also kein Risiko für die Performance.

## Was nicht passiert ist

- **Kein OpenTelemetry / OTLP.** `metrics` ist Prometheus-spezifisch.
  Für OTLP müsste man `opentelemetry-otlp` ziehen — größerer Stack,
  größere Dep-Graph. Wenn der Use-Case nach OTel ruft, bauen wir das
  später.
- **Keine eingebauten /metrics Auth.** Wie /health und /ready liegt
  /metrics offen. Reverse-Proxy / Cluster-internes Network ist die
  Schicht, die das absichert.
- **Keine cardinality-Sprengstoffe.** Alle Labels (`action`, `real`,
  `op`) haben endliche, kleine Kardinalitäten. Insbesondere
  *kein* Label mit dem `query`-String oder der `session_id` —
  das würde explodieren.

## Test-Status

- 98/98 Tests grün (95 + 3 metrics)
- 0 Clippy-Warnings
- `cargo fmt --all -- --check` clean
- Live verifiziert: `curl localhost:7777/metrics` gibt korrekte
  Prometheus-Exposition mit Histogram-Buckets, Gauges, Counter
