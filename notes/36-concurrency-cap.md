# 36 — Concurrency-Cap

**Stand:** Iteration 18. Notiz 35 hat gezeigt, dass der Server bei
≥ 50 parallelen Recall-Sessions p99 in den Hunderten von Millisekunden
(50: 397 ms; 100: 771 ms) liefert — alle aufgrund der Mutex-
Serialisierung über `AppState.inner`. Ein Refactor zu read-only
Recall (Iter 20) löst das langfristig; bis dahin braucht der Server
einen Schutzmechanismus, damit eine Lastspitze ihn nicht in einen
unbrauchbaren Zustand schickt.

Die Lösung ist eine konfigurierbare obere Grenze für simultan offene
WebSocket-Sessions. Sobald die Grenze erreicht ist, wird ein neuer
Upgrade mit `503 Service Unavailable + Retry-After: 1` abgelehnt.
Server-Latency staut nicht mehr unbegrenzt; ein gut-erzogener Client
backed off.

## Mechanik

`AppState` hat ein neues Feld:

```rust
pub struct AppState {
    inner: Arc<Mutex<Inner>>,
    llm: Arc<LlmClient>,
    sessions: Arc<Semaphore>,   // ← neu
}
```

`Semaphore::new(cap)` mit `cap = JAVIS_MAX_CONCURRENT_SESSIONS`
(default 32, gewählt nach den Load-Test-Daten von Notiz 35: bei 32
parallelen sind p99 Latenzen noch tolerabel ~250 ms).

Im `ws_handler`:

```rust
let permit = match state.sessions().try_acquire_owned() {
    Ok(p) => p,
    Err(_) => {
        metrics::counter!("javis_ws_rejected_total",
            "action" => params.action.as_str(),
            "reason" => "concurrency_cap").increment(1);
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::RETRY_AFTER, "1")],
            "javis: concurrency cap reached, retry shortly\n",
        ).into_response();
    }
};
ws.on_upgrade(move |s| run_session(s, state, params, permit))
```

Der `OwnedSemaphorePermit` wandert in den `run_session`-Frame und
wird beim natürlichen `drop` am Funktionsende freigegeben — egal ob
durch normalen Exit, Client-Disconnect, oder Panic.

## Konfiguration

```sh
# Default 32, adjust if you have more or fewer cores / memory:
JAVIS_MAX_CONCURRENT_SESSIONS=128 javis-viz
```

`resolve_session_cap()` warnt mit `tracing::warn!` und benutzt den
Default, wenn die Var unparseable ist (z. B. `"abc"` oder `"0"`) —
ein Server soll wegen einer Tippfehler-Env-Var nicht den Start
verweigern.

## Neue Metriken

```
# TYPE javis_ws_rejected_total counter
javis_ws_rejected_total{action="recall",reason="concurrency_cap"} 12
javis_ws_rejected_total{action="train",reason="concurrency_cap"} 1
```

Labels:
- `action` — train / recall / reset / ask
- `reason` — `concurrency_cap` heute; future-proof für andere
  Rejection-Gründe (z. B. malformed query)

Zwei Sachen, die man dadurch in Grafana sehen will:

1. `rate(javis_ws_rejected_total[1m])` — wenn das > 0 ist,
   ist der Cap erreicht; entweder hochsetzen oder schneller
   skalieren.
2. `javis_ws_rejected_total / (javis_ws_rejected_total + javis_ws_sessions_total)` —
   Rejection-Rate. Operator-Alarm wenn das > 1 % über 5 min.

## Tests

`crates/viz/tests/concurrency_cap.rs`:

- `upgrade_rejected_at_cap_zero` — Cap = 0 (degenerierter Fall),
  jeder Upgrade wird abgelehnt; verifiziert 503 + `Retry-After: 1`.
- `released_permit_unblocks_next_session` — Cap = 1, blockiert mit
  einem laufenden Train (~700 ms wall), zweite Anfrage in dem
  Fenster wird abgelehnt, nach Train-Ende geht eine dritte Anfrage
  durch. Verifiziert dass der Permit *wirklich* released wird.
  Plus Metric-Counter-Inkrement.

Beide Tests sprechen HTTP/1.1 direkt über `tokio::net::TcpStream`
statt `tokio-tungstenite`, weil die WS-Client-Crate eine 503 als
Handshake-Fehler verbergen würde — Status-Code und Header wären weg.

## Was Iter 18 nicht macht

- **Kein adaptiver Cap.** Der Wert ist statisch. Ein Auto-Scaler-Mode
  („wenn p95 > X, senke Cap um Y") wäre ein eigener Schritt mit
  eigener Komplexität.
- **Keine Queue-with-fairness.** `Semaphore::try_acquire_owned()`
  ist non-blocking — wenn voll, sofort 503. Eine bounded Queue mit
  FIFO-fair waiting würde Latenz-Spikes glätten, aber Backpressure
  beim Client *verschleiern* statt ehrlich zu kommunizieren. Gegen
  current Production-Wisdom: ehrliches 503 > heimliches Hängen.
- **Keine Auswirkung auf den eigentlichen Mutex-Engpass.** Iter 20
  fixed das.

## Status

- 100/100 Tests grün (98 + 2 neue Concurrency-Tests)
- 0 Clippy-Warnings
- Default-Cap 32, Override über `JAVIS_MAX_CONCURRENT_SESSIONS`
- Neue Metric `javis_ws_rejected_total{action,reason}`
- Production-Verdikt aktualisiert: System bleibt unter Last
  *responsive*, lehnt aber ehrlich ab statt zu ersticken
