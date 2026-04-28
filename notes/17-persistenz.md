# 17 — Persistenz: Snapshots überleben Server-Neustarts

**Stand:** Javis ist nicht mehr stateless. Ein trainiertes Brain kann
als JSON-Snapshot auf Platte geschrieben und beim Server-Start wieder
geladen werden — das gelernte Wissen überlebt Prozess-Neustarts.

## Warum jetzt

Nach Iteration 4 lief alles, aber jeder Server-Restart zerstörte das
gelernte Wissen. Das war die fundamentale Lücke zwischen „Demo" und
„Werkzeug": ohne Persistenz kann niemand ein Brain langsam aufbauen
oder ein gelerntes Modell teilen.

## Was sich geändert hat

### `serde`-Derives durch den ganzen Stack

`snn-core` und `encoders` bekamen `serde` als Workspace-Dependency
(vorher hatte snn-core null Deps, encoders nur `snn-core`). Alle
relevanten Datentypen sind jetzt `Serialize + Deserialize`:

| Crate     | Typen                                                                     |
| --------- | ------------------------------------------------------------------------- |
| `snn-core`| `LifNeuron`, `LifParams`, `NeuronKind`, `Synapse`, `StdpParams`,          |
|           | `IStdpParams`, `HomeostasisParams`, `Network`, `Region`, `Brain`,         |
|           | `InterEdge`, `PendingEvent`                                               |
| `encoders`| `Sdr`, `TextEncoder`, `EngramDictionary`                                  |

Plus `Clone` auf `Network`, `Region`, `Brain` (alle Substruktur-Typen
waren schon `Clone` oder `Copy`), damit der Snapshot ohne
serde-round-trip kopiert werden kann.

### Transientes vs. Persistiertes

Was im Brain *Topologie + gelerntes Gewicht* ist, geht in die Datei.
Was *Simulations-Zustand für den nächsten Step* ist, wird mit
`#[serde(skip, default)]` ausgeschlossen und nach dem Laden über
`Brain::ensure_transient_state()` neu auf 0 initialisiert:

| Persistiert                              | Transient (skip)                |
| ---------------------------------------- | ------------------------------- |
| `neurons` (Topologie + Plastizitäts-      | `i_syn`, `pre_trace`,           |
| Defaults), `synapses` (Gewichte!),        | `post_trace`, `activity_trace`, |
| `outgoing`/`incoming`, `inter_edges`,     | `v`, `refractory_until`,        |
| `outgoing` Adjazenz, plasticity params    | `last_spike`, `time`,           |
| (`stdp`, `istdp`, `homeostasis`), `dt`    | `step_counter`, `pending`,      |
|                                           | `events_delivered`              |

Der Effekt: ein 2 000-Neuronen / ~100k-Synapsen-Snapshot ist ~2–3 MB
JSON statt ~6 MB, und das Brain ist beim Laden sofort runnable.

### Neue API auf `AppState`

```rust
pub async fn save_to_file(&self, path: impl AsRef<Path>) -> std::io::Result<()>;
pub async fn load_from_file(&self, path: impl AsRef<Path>) -> std::io::Result<()>;
```

Format ist JSON mit einem `Snapshot { version, brain, dict, encoder,
known_words, trained_sentences }`-Wrapper. `version: u32` (aktuell `1`)
schlägt beim Laden eines fremden Schemas hart Alarm — kein stilles
Lesen veralteter Daten.

### CLI-Flag `--snapshot`

```sh
cargo run -p viz --release --bin javis-viz -- --snapshot brain.json
```

- Existiert die Datei: laden, Bootstrap-Korpus überspringen.
- Existiert sie nicht: Default-Korpus trainieren als ob nie ein Flag
  gegeben wäre.
- Beim graceful shutdown (Ctrl-C / SIGTERM): aktueller State zurück
  in dieselbe Datei.

Kein CLI-Crate als Dependency — nur ein paar Zeilen
`std::env::args()`-Parsing.

### Graceful Shutdown im Server

`axum::serve(...).with_graceful_shutdown(shutdown_signal())` — der
Server reagiert jetzt auf SIGINT und SIGTERM, fährt sauber runter und
schreibt nur danach den Snapshot. Damit ist die letzte Trainings-
Sitzung garantiert mit drauf.

## Tests

Drei neue Tests, alle grün:

`crates/snn-core/tests/serde_roundtrip.rs`:
- `network_roundtrips_through_json` — kleines Netz mit allen drei
  Plastizitätsformen aktiviert → JSON → zurück → topologie + Gewichte
  bit-identisch, transient buffers via `ensure_transient_state`
  korrekt re-initialisiert.
- `brain_roundtrips_with_inter_region_edges` — zwei Regionen + zwei
  Inter-Edges, danach `step()` läuft ohne Panic.

`crates/viz/tests/smoke.rs`:
- `snapshot_round_trip_preserves_recall` — der Vertrag des Features:
  trainiere einen Satz auf `state_a`, schreibe Snapshot, lade auf
  einem **frisch gebauten** `state_b`, recall via Server, das gelernte
  Wort ist immer noch in den Decoded-Kandidaten. Ohne JSON-Bug
  würde das *nicht* funktionieren.

Plus die 46 bestehenden Tests aus den vorherigen Iterationen.
**49/49 grün workspace-weit.**

## Wie sich das anfühlt im Live-Demo

```
$ cargo run -p viz --release -- --snapshot brain.json
javis-viz: no snapshot at brain.json; bootstrapping default corpus
javis-viz: ready (3 sentences, 27 concepts)
javis-viz listening on http://127.0.0.1:7777

# … Browser auf, paar Sätze trainieren, Ctrl-C …

javis-viz: shutdown signal received
javis-viz: snapshot written to brain.json

$ cargo run -p viz --release -- --snapshot brain.json
javis-viz: loaded snapshot from brain.json (8 sentences, 64 concepts)
javis-viz: ready (8 sentences, 64 concepts)
```

Der zweite Start lädt das Brain wo der erste aufgehört hat — Recall
funktioniert sofort auf dem voll trainierten Wissen.

## Was Iteration 5 nicht hat

- **Frontend-Save/Load-Buttons** — bewusst weggelassen. CLI-Flag und
  graceful shutdown decken den 95 %-Fall (lokale Sessions). Browser-
  basiertes Upload/Download wäre Iteration 6 wenn es nötig wird.
- **Inkrementelle Snapshots** — aktuell wird die ganze Datei
  überschrieben. Bei großen Brains könnte ein Append-only-Log
  cleverer sein.
- **Alternative Formate** (bincode, MessagePack) — JSON ist 2–3× so
  groß wie binäre Formate, aber human-readable und mit `jq`
  inspizierbar. Für die aktuelle Größe ist das die richtige Wahl.

## Status

- 49/49 Tests grün workspace-weit
- 17 Forschungs-Notizen unter `notes/`
- Branch `claude/3d-graph-visualization-dpx8s`, alle 17 Commits gepusht
- Live-Server-Run mit Snapshot:
  `cargo run -p viz --release -- --snapshot brain.json`
