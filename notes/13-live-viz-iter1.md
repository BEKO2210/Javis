# 13 — Live-Visualisierung, Iteration 1

**Stand:** Javis hat einen Live-Server. Browser öffnet eine Page, sieht
das Brain trainieren und antworten — alles streamt per WebSocket aus
der echten Pipeline.

## Was Iteration 1 liefert

Ein neuer Crate `crates/viz`:

- **`pipeline.rs`** — gleiche Pipeline wie `eval::token_efficiency`,
  aber jeder Schritt pusht ein `Event` in einen `mpsc::Sender`. Pro
  simulierte Millisekunde geht ein gebatchtes `Step`-Event raus
  (Spike-Indizes für R1 und R2 separat). Phasen-Wechsel
  (training / cooldown / fingerprint / recall) und das finale
  `Decoded`-Event mit RAG-vs-Javis-Vergleich gehen über denselben
  Kanal.
- **`server.rs`** — Axum-Router, ein `/ws?query=<word>`-Endpoint.
  Verbindung öffnet eine Session: Brain hochfahren, Korpus trainieren,
  Query recallen, Decode senden, Verbindung schließen.
- **`main.rs`** — Binary `javis-viz`, serviert die statischen Files
  und den WebSocket auf `127.0.0.1:7777`.
- **`static/index.html` + `main.js`** — Spike-Raster auf Canvas
  (sweepender Cursor, Farbcodes für R1, R2-E, R2-I), Side-Panel mit
  Live-Spike-Raten, Phase-Anzeige, decoded Konzepten und
  Token-Differenz.

Der Smoke-Test (`tests/smoke.rs`) bringt den Server in-process hoch,
zieht das WebSocket leer, prüft dass alle vier Event-Typen kommen
(`init`, `step`, `decoded`, `done`) und der Reduktions-Wert ≥ 70 %
über die Leitung wirklich ankommt.

## Wie man es lokal startet

```sh
cargo run -p viz --release --bin javis-viz
# dann http://127.0.0.1:7777 im Browser öffnen
```

Default-Query ist `rust`. Im Header-Eingabefeld jede beliebige Query
schreiben → die Session bootet komplett neu und animiert das Brain.

## Wire-Format (knapp)

```jsonc
{ "type": "init", "r1_size": 1000, "r2_size": 2000,
  "r2_excitatory": 1600, "r2_inhibitory": 400 }

{ "type": "phase", "name": "training", "detail": "paragraph 1/3" }

{ "type": "step", "t_ms": 12.0, "r1": [ … ], "r2": [ … ] }

{ "type": "decoded", "query": "rust",
  "candidates": [ {"word":"rust","score":1.0} ],
  "rag_tokens": 28, "javis_tokens": 2, "reduction_pct": 92.9,
  "rag_payload": "…", "javis_payload": "rust" }

{ "type": "done" }
```

Klein genug, um pro Sekunde tausende dieser Frames zu schicken,
ohne den Socket zu sättigen.

## Was Iteration 1 absichtlich noch *nicht* liefert

- **Kein 3D-Brain** — der Canvas zeigt einen Spike-Raster (Standardform
  in der Neurowissenschaft). Three.js / `3d-force-graph` kommt in
  Iteration 2 und ersetzt nur die Visualisierung; alle Streams,
  Events, Stats und das Side-Panel bleiben gleich.
- **Kein Persistenz / Multi-Query** — jede Verbindung trainiert das
  Brain neu. Iteration 3 wird Brain-Snapshots cachen, sodass
  aufeinanderfolgende Queries die schon gelernten Engramme nutzen
  und sich Pattern Completion in Echtzeit „sammelt".
- **Kein echter LLM-Call** — das Decoded-Panel zeigt, *was* an einen
  LLM gehen würde. Iteration 4 hängt einen Claude-API-Adapter dran
  und beantwortet die Query mit beiden Payloads parallel, sodass das
  Video „RAG sagt X, Javis sagt Y, Javis braucht 95 % weniger Tokens"
  zeigen kann.

## Größere Linie

Die ursprüngliche Anfrage am Anfang dieses Branches war:

> Was wäre der nächste Schritt … etwas das viral geht weil es etwas
> komplett neues und super krass ist wie etwas aus der Zukunft

Iteration 1 macht den Sprung von „cargo test sagt es funktioniert" zu
„im Browser kann man dem System beim Denken zusehen". Die Visualisierung
ist im Iteration-1-Stand bereits aussagekräftig (Spike-Raster ist *die*
Sprache der Spiking-Neuroscience), und Iteration 2–4 baut darauf auf,
ohne irgendeine der nun bestehenden Schichten anzufassen.

## Status

- 41/41 Tests grün workspace-weit (40 alte + Smoke-Test)
- Server-Start: `cargo run -p viz --release --bin javis-viz`
- Branch unverändert: `claude/3d-graph-visualization-dpx8s`

## Nächste Schritte (in Reihenfolge)

1. **Iteration 2**: Three.js / `3d-force-graph` ersetzt den Raster.
   Knoten als 3D-Cluster, Position grob anatomisch (R1 vorne als
   „Sensory Cortex", R2 hinten als „Memory Cortex"), Spikes als
   kurze Lichtpulse, STDP-gestärkte Synapsen werden zu sichtbaren
   Bahnen.
2. **Iteration 3**: Persistente Brain-Session mit Training-UI;
   Korpus erweitern auf Live-Eingabe, Engramme behalten zwischen
   Queries, „Lerne diesen Satz" / „Frag das Brain"-Trennung.
3. **Iteration 4**: `crates/llm` mit Claude-API-Adapter, „Send to LLM"
   schickt beide Payloads parallel, zeigt beide Antworten und die
   Token-Bill nebeneinander.
