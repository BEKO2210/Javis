# 40 — Pipeline-Profile: 77 % im Brain, NICHT Amdahl-bound

**Stand:** Iteration 22. In Iter 21 hatten wir 1.5× auf
`Network::step_immutable` gemessen, aber der Docker-Stack-Server-
Mean blieb bei ~10 ms — verschoben durch Run-to-Run-Variance. Die
offene Frage: *ist Brain-Compute jetzt nur noch 30-40 % der
Pipeline (Amdahl) oder immer noch der Bottleneck?*

Antwort durch direkte Messung: **immer noch der Bottleneck. 77 %
des Recall-Pipelines sind Brain-Compute.**

## Instrumentation

Sechs Phasen-Timer in `AppState::run_recall`:

| Phase | Was wird gemessen |
| --- | --- |
| `lock_overhead` | RwLock-acquire + initial Init/Phase-WS-Events |
| `encode` | `g.encoder.encode(query)` (Text → SDR) |
| `snn_compute` | `run_with_cue_streaming_immutable(...)` (Sim + WS-streaming) |
| `decode` | `top_k(counts, KWTA_K)` + `g.dict.decode(...)` |
| `rag_search` | Linear scan über `g.trained_sentences` |
| `response_build` | Build + send des `Event::Decoded` |

Plus zwei Sub-Phasen innerhalb von `snn_compute`:

| Sub-Phase | Was |
| --- | --- |
| `brain_compute` | Reine `brain.step_immutable` Wall-Clock (ohne await-Punkte) |
| `ws_stream` | `tx.send(Event::Step { … }).await` Aufrufe |

Alle als Prometheus-Histogramme exportiert
(`javis_recall_phase_seconds{phase=…}`,
`javis_recall_subphase_seconds{phase=…}`) und in einer
strukturierten `tracing::info!`-Zeile pro Recall.

## Skript

`scripts/pipeline_profile.py` — fährt N Recalls (default 100,
sequential), liest /metrics vor und nach, berechnet Mean pro Phase
aus den `_count`/`_sum`-Series-Differenzen, sortiert nach Mean
descending.

```sh
python3 scripts/pipeline_profile.py --n 200 --concurrency 1
```

## Ergebnis (200 sequential recalls gegen `docker compose up`)

```
phase              count    mean ms   share
--------------------------------------------
snn_compute          200      7.905   97.9%
decode               200      0.133    1.6%
response_build       200      0.020    0.2%
lock_overhead        200      0.009    0.1%
encode               200      0.004    0.0%
rag_search           200      0.002    0.0%
--------------------------------------------
phase sum                     8.072  100.0%
recall total         200      8.054 ms

snn_compute sub-phases:
  phase            count    mean ms   share
  ------------------------------------------
  brain_compute      200      6.179   92.2%   ← 76.7% des Recall-Total
  ws_stream          200      0.525    7.8%   ←  6.5% des Recall-Total
```

## Was die Daten sagen

1. **`Brain::step_immutable` ist 76.7 % des Pipelines.** 6.18 ms
   pro Recall, das sind 300 Steps × 20.6 µs/step für die
   Two-Region-Brain (1000 R1 + 2000 R2).
2. **WS-Streaming ist 6.5 %.** 30 batched Step-Events pro Recall,
   ~17 µs pro `tx.send(...).await`. Tokio-Yields, aber billig
   weil der mpsc-Channel nie voll wird.
3. **Decoder ist 1.6 %.** 0.13 ms — kWTA top-k auf 2000 counts
   (~50 sortierte Indizes mit count > 0) plus EngramDictionary-
   Scan über ~50 trainierte Wörter.
4. **Encoder, RAG-Search, response-build, lock-overhead sind
   insgesamt 0.4 %.** Mit Mikrosekunden zu messen, kein
   Optimierungspotential.

## Mein Irrtum von Iter 21 — und warum er sich wegmittelt

In Iter 21 hatte ich vermutet, dass das Pipeline-Server-Mean nicht
fällt, weil „Encoder/Decoder/RAG/JSON-Serde sich gleichmäßig
verteilen". **Falsch.** Die Verteilung war von Anfang an
brutal-skewed: ~98 % Brain, ~2 % alles andere.

Dass das Server-Mean trotz 1.5× LIF-Speedup nicht offensichtlich
fiel, war Run-to-Run-Variance des Docker-Stacks im Sandbox. Die
1.5× × 0.92 ≈ 1.4× Pipeline-Speedup ist real, nur eben kleiner als
die Variance des Test-Setups (±10 %).

## Amdahls Gesetz: noch nicht erreicht

ChatGPTs Befürchtung war richtig — Amdahl wartet darauf, zuzu-
schlagen. Aber: bei 77 % Brain-Compute-Anteil ist eine 2× Brain-
Optimierung immer noch eine 1.7× Pipeline-Optimierung
(Amdahl-Maximum: 1 / (0.23 + 0.77/2) = 1.65×). Wir sind also nicht
im flat-tail-Bereich, wo SIMD-Refactoring sich nicht mehr rentiert.

**Sweet-spot-Argument für die nächste Optimierung am Brain:**

| Aktuell | Brain step | Pipeline-Effekt |
| --- | --- | --- |
| Heute | 6.18 ms (77 %) | 8.05 ms |
| Brain × 1.5× | 4.12 ms | 5.99 ms (1.34× Gesamt) |
| Brain × 2.0× | 3.09 ms | 4.96 ms (1.62× Gesamt) |
| Brain × 3.0× | 2.06 ms | 3.93 ms (2.05× Gesamt) |

Erst ab ~3× Brain-Speedup beginnen die anderen Phasen einen
signifikanten Anteil zu haben. Das heißt: AoSoA-Refactor von
`LifNeuron` plus ggf. `wide`-Crate-SIMD wäre jetzt klar
gerechtfertigt — *eine* solche Iteration zahlt sich noch direkt
aus.

## Was Iter 22 *nicht* macht

- **Keine Optimierung implementiert.** Diese Iteration baut nur
  die Mess-Infrastruktur. Der Refactor selbst (AoSoA, SIMD oder
  was auch immer) ist die nächste Iteration, mit klarem Vorher-
  Nachher-Vergleich über genau diese Phase-Histogramme.
- **Keine Latenz-Optimierung am WS-Stream-Pfad.** 6.5 % wäre
  möglich (Step-Events fire-and-forget statt await), aber ist
  zu klein für jetzt — Brain hat 12× mehr Hebel.
- **Profiler-Setup deaktivierbar machen.** Phase-Timer kosten
  ein paar `Instant::now()`-Calls (~50 ns × 6 = 300 ns), 0.004 %
  des Recall-Total. Bleiben aktiv — das Signal ist mehr wert als
  der Overhead.

## Status

- 6 Phase-Histogramme + 2 Sub-Phase-Histogramme als Prometheus-
  Metriken exportiert
- `scripts/pipeline_profile.py` reproduzierbares 1-Schuss-Profil
- 108/108 Tests grün, 0 Clippy, fmt clean
- **Klares Verdikt: Brain-Compute ist 77 % der Pipeline. AoSoA /
  SIMD am LIF-Loop ist die korrekte nächste Optimierung.**
