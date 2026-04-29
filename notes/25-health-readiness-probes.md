# 25 — `/health` und `/ready` Probes

**Stand:** Teil B von Iteration 12 (Operational Hardening). Ohne
Health-Endpoints kann ein Container-Orchestrator nicht entscheiden, ob
ein Pod läuft oder neu gestartet werden muss; ohne Readiness-Endpoint
kann er nicht entscheiden, ob er Traffic auf den Pod routen darf. Beide
Probes sind jetzt da, beide rein über die existierende Axum-Router-
Wiring, ohne externe Abhängigkeiten.

## Endpoints

| Path | Status | Body | Zweck |
| --- | --- | --- | --- |
| `GET /health` | 200 immer | Plain `ok` | Liveness — Prozess antwortet |
| `GET /ready` | 200 immer | JSON | Readiness — Prozess kann Traffic verarbeiten |

```sh
$ curl localhost:7777/health
ok

$ curl localhost:7777/ready
{"status":"ready","sentences":3,"words":43,"llm":"mock"}
```

### Warum beide nicht-fehlschlagen?

`bootstrap_default_corpus` läuft *vor* `axum::serve`, also wenn der
Server überhaupt einen Request annimmt, ist er ready. Eine `reset`-
Operation hinterlässt einen leeren, aber funktionalen Zustand — recall
auf einem leeren Brain liefert keine Treffer, ist aber kein Server-
Fehler. Eine Probe darf in dem Moment nicht flappen.

Wenn später echte Failure-Modi dazukommen (z. B. Snapshot-Pfad nicht
beschreibbar, Persistenz-Worker tot), kann `/ready` Status `503` mit
einer Fehlerbegründung im Body zurückgeben. Im aktuellen Setup gibt es
keinen solchen Failure-Modus.

## Routing

In beiden Routern (`router` für die Binary, `router_no_static` für
Tests) sind die Probes vor `with_state` registriert, damit sie auch
durch das `tower-http`-Static-Fallback nicht abgefangen werden:

```rust
Router::new()
    .route("/ws", get(ws_handler))
    .route("/health", get(health_handler))
    .route("/ready", get(ready_handler))
    .with_state(state)
    .fallback_service(tower_http::services::ServeDir::new(static_dir))
```

## Tests via `tower::ServiceExt::oneshot`

Statt einen echten TCP-Listener für jeden Test zu spawnen (wie die
WebSocket-Smoke-Tests es machen), benutzen die HTTP-Probe-Tests
`tower::ServiceExt::oneshot` direkt auf dem `Router`:

```rust
let app = viz::server::router_no_static(state);
let resp = app.oneshot(Request::builder().uri("/ready").body(Body::empty())?).await?;
assert_eq!(resp.status(), StatusCode::OK);
```

`Router` ist selbst ein `tower::Service`, also kann man Requests einfach
durchreichen. Das umgeht TCP, ist deterministisch schnell, und braucht
keinen freien Port. Drei Tests:

- `health_returns_200_immediately` — body ist exakt `b"ok"`
- `ready_reports_brain_stats_as_json` — nach 1 Train: sentences=1,
  words>0, llm=mock, content-type startet mit `application/json`
- `ready_works_on_empty_brain` — frischer State: sentences=0, words=0,
  trotzdem 200

Dev-Deps neu: `tower = { version = "0.5", features = ["util"] }` und
`http-body-util = "0.1"`. Beide sind zero-cost im Release-Binary, weil
nur in `[dev-dependencies]`.

## Was die Probes nicht sind

- **Kein Auth.** Beide Endpoints sind öffentlich. In einer
  produktiven Deployment wären sie hinter einem internen Network-Pfad
  oder einem Reverse-Proxy mit IP-Whitelist. Das ist Sache der
  Deployment-Pipeline, nicht des Server-Codes.
- **Kein Deep-Healthcheck.** `/ready` macht keinen Brain-Step, keine
  LLM-Roundtrip-Probe und keinen Snapshot-Save-Test. Solche Checks
  würden den Probe-Pfad zu teuer machen für die typischen 5-Sekunden-
  Intervalle. Wenn solche Tests gewünscht sind, gehören sie in einen
  separaten `/diag`-Endpoint mit Authentifizierung.
- **Kein Per-Request-Counter.** Das kommt mit Teil C (Prometheus
  metrics).

## Test-Status

- 95/95 Tests grün (92 + 3 Health-Tests)
- 0 Clippy-Warnings
- `cargo fmt --all -- --check` clean
