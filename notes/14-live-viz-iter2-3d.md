# 14 — Live-Visualisierung, Iteration 2: 3D-Brain

**Stand:** Spike-Raster ersetzt durch ein echtes 3D-Brain im Browser.
Zwei anatomische Lobes, organisch verteilte Neuronen, Spikes leuchten
weiß auf und fadeen über 220 ms zurück zur Basisfarbe.

## Was sich geändert hat

Pure Frontend-Iteration. **Keine einzige Zeile Rust** wurde angefasst,
das Wire-Format aus Iteration 1 ist bit-identisch geblieben.

### `static/index.html`

- 2D-Canvas raus, `<div id="brain3d">` rein
- Three.js + `3d-force-graph` als CDN-Skripts (vasturiano/3d-force-graph,
  die Library aus dem Original-Vorschlag ganz am Anfang dieses Branches)
- Style-Updates: glühender Token-Saving-Header, Glow-Akzente in den
  Pills und Legenden-Swatches
- „drag · scroll · right-click pan"-Hint unten rechts

### `static/main.js`

- **`buildGraph`** nimmt das Init-Event und macht 3000+ Knoten daraus,
  positioniert sie auf zwei Sphären:
  - **R1 (Sensory Cortex)**: linke Sphäre, 1000 Knoten, blau (`#62d6ff`)
  - **R2 excitatory**: rechte größere Sphäre, ~1600 Knoten, gelb
    (`#ffd166`)
  - **R2 inhibitory**: kleinere innere Schale in R2, ~400 Knoten,
    pink (`#ff5c8a`) — anatomisch sitzen Interneuronen in
    excitatorischem Gewebe
- **Fibonacci-Sphäre-Verteilung** für organische, gleichmäßige
  Punktwolke auf jeder Lobe (statt Grid-Look)
- **`fx/fy/fz` als feste Positionen** und Force-Strength = 0, damit die
  Lobes ihre Form behalten und nicht wegdriften
- **`nodeThreeObject`** baut für jeden Knoten ein eigenes
  `MeshBasicMaterial`. Per `nodeById`-Map halten wir Referenzen, damit
  Spike-Updates O(1) sind
- **Spike-Animation** läuft pro Frame:
  ```js
  const t = 1 - age / SPIKE_DECAY_MS;          // 220 ms decay
  mat.color.copy(base).lerp(SPIKE_WHITE, t);
  mat.opacity = 0.85 + 0.15 * t;
  ```
  — Knoten leuchtet heiß-weiß auf, fadet zurück zur Region-Farbe.
- WebSocket-Logik unverändert übernommen aus Iteration 1.

## Wie es jetzt aussieht (im Kopf)

Zwei „Hirnlappen" schweben nebeneinander auf schwarzem Grund:

- Beim Training-Phase-Cue: ein paar hundert blaue R1-Punkte feuern
  rhythmisch, Wellen gelber R2-E-Spikes folgen mit ~2 ms Verzögerung,
  dazwischen punktuelle pinke I-Spikes — sichtbare AER-Latenz
- Bei Recall: nur ein kleines blaues Cluster (das Query-Wort) leuchtet
  in R1, daraus wandert ein gelb-weißes Glow-Pattern in R2
- Während Phase-Wechsel klingt alles ab und das Side-Panel zeigt die
  neue Phase

## Was Iteration 2 nicht hat

- **Synapsen werden nicht visualisiert.** Eine 100k-Edges-Wolke
  erstickt alles andere. Iteration 2.5 oder 3 könnte die
  STDP-stärksten Verbindungen auswählen und sichtbar machen — z.B.
  „Top 200 Synapsen über `w_max * 0.7`".
- **Kein Spike-Travel-Animation auf Edges.** `linkDirectionalParticles`
  von 3d-force-graph kann das, aber es braucht Edges, siehe oben.
- **Server unverändert** — also kein Live-Training auf User-Input,
  jede Verbindung trainiert das Brain einmal vom Korpus durch.

## Wie man's anschaut

```sh
cargo run -p viz --release --bin javis-viz
# http://127.0.0.1:7777
```

Page lädt → 3D-Brain renders sofort → erste Session läuft mit
Default-Query `rust` und animiert die ganze Pipeline live.
Cue-Eingabe oben rechts → andere Query, Brain wird neu gebaut.

## 41/41 Tests grün

Smoke-Test bleibt grün, denn das Backend ist unangetastet. Der Frontend-
Sprint hat null Backend-Risiko.

## Iteration 3 (geplant)

- Persistente Brain-Sessions: Brain wird einmal beim Server-Start
  trainiert, alle weiteren Queries laufen schnell darauf
- Live-Training-UI: zweites Eingabefeld „Lerne diesen Satz" — schickt
  Trainings-Cue ohne komplette Re-Init
- Top-N strongest synapses als 3D-Linien über STDP-Wachstum
- „Pause / Step / Slow"-Knöpfe zum Anschauen einzelner Recall-Phasen

## Iteration 4 (geplant)

- `crates/llm` mit Claude-API-Adapter
- „Send to LLM"-Knopf: schickt RAG-Payload und Javis-Payload parallel,
  zeigt beide Antworten + reale Token-Bill
- Optional: BPE-Token-Counter via `tiktoken`-Rust-Port statt der
  1.3-Heuristik
