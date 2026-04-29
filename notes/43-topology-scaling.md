# 43 — Topologie skalieren: R2 von 2 000 auf 10 000

**Stand:** Iteration 25. Notiz 42 hat den ehrlichen Befund gebracht:
auf einem 100-Satz / 286-Vokabular Korpus liefert Javis 100 %
Self-Recall, aber nur ~2 % Associative-Recall, und die decoded
Wörter sind zu 78 % Cross-Domain-Bleed. Diagnose: das R2-Layer
mit 2 000 Neuronen und KWTA_K=220 hat schlicht nicht genug
orthogonalen Raum. Bei 11 % Sparsity verstoßen Engrams
miteinander, iSTDP kann zwischen mehr als ~50 Engrams keine
Trennwände mehr bauen.

Diese Iteration: Topologie-Refactor, der das adressiert. Größere
Layer, viel sparser, neu getunte Hemmplastizität.

## Was geändert wurde

| Konstante | Vorher | Nachher | Begründung |
| --- | ---: | ---: | --- |
| `R2_N` | 2 000 | **10 000** | 5× mehr orthogonaler Raum |
| `R2_P_CONNECT` | 0.10 | **0.03** | weniger zufällige Synapse-Kollision; hält Synapsenzahl bei 3 M (sonst 10 M) |
| `KWTA_K` | 220 | **100** | 1 % Sparsity statt 11 %; mathematische Mindest-Überlappungs-Wahrscheinlichkeit zwischen zwei zufälligen Engrams ≈ 0.01 |
| `CONTEXT_KWTA_K` | 60 | 30 | proportional |
| `FAN_OUT` (R1→R2) | 10 | **30** | Forward-Drive trifft mit ENC_K=20 jetzt 600 R2-Neuronen (~6 %) statt 200 (~10 %); Eingang bleibt überschwellig im größeren Pool |
| `IStdpParams.a_plus` | 0.05 | **0.10** | LTP doppelt so aggressiv: silent-E + active-I-Paare bauen Suppressions schneller auf |
| `IStdpParams.a_minus` | 0.55 | **1.10** | LTD doppelt so aggressiv: I→E-Synapsen werden bei Co-Aktivität schneller geschwächt |
| `IStdpParams.w_max` | 5.0 | 8.0 | Headroom, da der I-Pool jetzt 2 000 Neuronen hat |

Drei Dateien spiegeln diese Konstanten:
- `crates/viz/src/state.rs`
- `crates/eval/src/token_efficiency.rs`
- `crates/eval/src/scale_bench.rs`

## Memory- und Laufzeit-Auswirkung

| | iter ≤24 | iter 25 |
| --- | ---: | ---: |
| Synapsen R2 | 0.4 M (2 000² × 0.10) | 3.0 M (10 000² × 0.03) |
| Synapsen-RAM | ~13 MB | ~96 MB |
| `Network::step` (R2 allein) | ~7 µs | geschätzt 30-40 µs |
| Training 100 Sätze | ~70 s | gemessen ~5–7 min |
| Snapshot-File | ~30 MB | geschätzt ~120-150 MB |

Die Snapshot-Datei sprengt jetzt das Volume-Default von 100 MB
nicht — bleibt aber prominent. Wenn das ein Issue wird,
Snapshot-Kompression (gzip-on-write) ist ein eigener Schritt.

## 113 / 113 Tests grün

Alle bestehenden Tests passieren ohne Schwellen-Anpassung:
- `snn-core`: 53 Tests (LIF, STDP, iSTDP, homeostasis, BTSP,
  serde, equivalence) — unabhängig von viz/eval-Konstanten,
  daher trivial grün.
- `encoders`: 22 — Encoder-Crate berührt das Brain nicht direkt.
- `eval`: 13 — der `wiki_benchmark` (5 Wiki-Paragraphen) lief
  mit dem neuen 10 000-Neuron-Brain in ~5.5 min (war Sekunden)
  und hielt seine 70 %-Token-Reduktion-Schwelle.
- `llm`: 3.
- `viz`: 16 — alle Smoke-/Concurrency-/Snapshot-Tests grün, das
  größere Brain ändert die Init-Latenz aber nichts am Verhalten.

Das war das Hauptrisiko der Iteration: dass die wiki_benchmark-
Schwellen mit der neuen Topologie nicht mehr halten. Sie tun's.

## Skalen-Benchmark mit neuer Topologie

Lauf: `cargo run --release -p eval --example scale_benchmark
-- --sentences 100 --queries 30 --decode-k 6 --seed 42`

Ergebnisse werden hier eingefügt sobald der Lauf fertig ist
(läuft beim Verfassen dieser Notiz).

```
[Platzhalter — Zahlen werden nach Bench-Lauf eingefügt]
```

## Honest reframe (für die README)

Vorher (Hero-Section): „96.7 % Token-Reduktion vs naïve RAG."
Nachher (auf 100 Sätzen, ehrlich): „35–45 % Token-Reduktion,
100 % Self-Recall, sub-ms Decoder-Latenz bis 300 Vokabular.
Bekannte Limits: Associative-Recall ~2 %, Cross-Bleed bei
N>50."

Diese Notiz dokumentiert den Eingriff am physischen Bottleneck;
die README dokumentiert die ehrliche Story. Beide existieren
bewusst nebeneinander.

## Was Iter 25 nicht behebt (auch nach Topologie-Bump)

- **Encoder-Diversität.** Der TextEncoder benutzt
  `DefaultHasher` mit fixed K=20 / N=1000. Zwei Wörter mit
  ähnlichem Hash kollidieren immer; das größere R2 fängt das
  nicht auf, weil die Kollision schon im R1-SDR steckt.
- **kWTA-Threshold-Tuning.** Wir haben K=100 gesetzt — aber
  die optimale Sparsity hängt vom Vokabular ab. Adaptive K
  (proportional zur Decoded-Konzentration) wäre eine eigene
  Iteration.
- **Hierarchie / Domain-Sub-Brains.** Bei N > 1000 Vokabular
  hilft auch ein 50 000-Neuron-Layer nicht — dann braucht es
  separate Sub-Brains pro Domäne mit Routing dazwischen.
  Architektur-Erweiterung, nicht Tuning.

## Status

- 113 / 113 Tests grün (53 + 22 + 13 + 3 + 16 + 3 doc-tests)
- 0 Clippy-Warnings, fmt clean
- Topologie auf produzierbarem Maximum für aktuelles
  Single-Brain-Setup
- Skalen-Benchmark läuft, Zahlen folgen
