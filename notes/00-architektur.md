# Javis — Architektur-Skizze

## Ziel

Eine **token-effiziente Speichermethode** für LLMs/Agents, die nur die für eine
Anfrage relevanten Inhalte ans LLM weitergibt — analog dazu, dass im
menschlichen Gehirn nur die für eine Aufgabe relevanten Regionen aktiv sind.

## Prinzip

1. Wissen wird als **Spike-Pattern** in einem Spiking Neural Network kodiert.
2. Konzepte = **Assemblies** (Gruppen gleichzeitig feuernder Neuronen).
3. Eine Anfrage wird in ein Eingangs-Spike-Pattern übersetzt.
4. Die Aktivität propagiert durch das Netz, nur eng verwandte Assemblies
   feuern stark genug, um „aktiviert" zu gelten.
5. Der Inhalt der aktivierten Assemblies wird ans LLM gegeben — sonst nichts.

## Bausteine

| Schicht       | Komponente                                                     |
| ------------- | -------------------------------------------------------------- |
| Kern          | LIF-Neuronen, Synapsen, STDP (`crates/snn-core`)               |
| Topologie     | Regionen, Routing, Spezialisierung — geplant                   |
| Encoder       | Text/Code → Sparse Distributed Representations — geplant       |
| Ingest        | Markdown, Chat-Logs, Code-Repos, API — geplant                 |
| API           | Query-Endpunkt, gibt aktivierte Inhalte zurück — geplant       |
| Visualisierung | Live-Anzeige der Netz-Aktivität in 3D — geplant               |

## Designprinzipien

- **Flache Datenstrukturen** (`Vec<f32>`), damit ein späterer GPU-Port
  (wgpu/candle) ohne Algorithmus-Änderung möglich ist.
- **CPU zuerst** — bis Benchmarks zeigen, dass wir CPU-bound sind.
- **Forschen vor Bauen** — kleine, isolierte Tests pro Hypothese.

## Offene Fragen

- Wie groß muss eine Assembly sein, um stabil zu sein? (~50 Neuronen ist
  Numenta/HTM-Schätzung)
- Welche Encoder-Strategie ist am token-effizientesten:
  Embedding → SDR vs. Hash → SDR vs. Trainierter Encoder?
- Wie messen wir Token-Ersparnis sauber? Vorschlag: gleicher Korpus,
  gleicher Test-Query-Set, Vergleich Token-Count Javis vs. naives RAG.
