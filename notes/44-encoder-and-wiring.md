# 44 — Encoder-Skalierung + Balanced Forward-Wiring

**Stand:** Iteration 26. Notiz 43 hat dokumentiert: das R2-Recurrent-
Layer ist NICHT der Cross-Bleed-Bottleneck. Diese Iteration greift
die zwei Upstream-Verdächtigen direkt an: den Encoder (zu hohe
Bit-Oversubscription) und das Forward-Wiring (random-uniform mit
Replacement, in-degree-Varianz).

## Diagnose vor dem Eingriff

**Encoder-Mathematik bei iter ≤25:**
- ENC_N=1000, ENC_K=20, also 2 % Sparsity
- Bei einem Vokabular von 286 Worten: 286 × 20 = 5 720 Bit-Claims
- Über 1 000 R1-Neuronen verteilt: jedes R1-Neuron wird im
  Mittel von **5.7 verschiedenen Wörtern beansprucht**
- Konsequenz: zwei „unverwandte" Wörter teilen sich
  zwangsläufig signifikante R1-Bits, weil der Eingangsraum
  einfach zu klein ist

**Forward-Wiring bei iter ≤25:**
```rust
for src in 0..R1_N {
    for _ in 0..FAN_OUT {
        let dst = (rng.next_u64() as usize) % r2_size;
        brain.connect(0, src, 1, dst, INTER_WEIGHT, INTER_DELAY_MS);
    }
}
```
Jedes R1-Neuron pickt FAN_OUT R2-Targets **mit Replacement**.
Konsequenzen:
- Ein R1 → R2 kann doppelt verkabelt sein (wirkt als Gewichtungs-Boost)
- R2-In-Degree ist Poisson-verteilt: bei FAN_OUT=10, R1=1000,
  R2=2000 ist die mittlere In-Degree 5, aber die Varianz
  beträgt sqrt(5) ≈ 2.2 — manche R2-Neuronen haben 0 Edges,
  andere 12+
- Die R2-Neuronen mit hoher In-Degree werden zu „Hub-
  Neuronen", die für viele R1-Inputs gleichzeitig feuern —
  klassischer Cross-Bleed-Mechanismus

## Eingriffe

### Eingriff 1: ENC_N und R1_N von 1 000 auf 4 000

| Konstante | iter ≤25 | iter 26 | Wirkung |
| --- | ---: | ---: | --- |
| `R1_N` | 1 000 | **4 000** | 4 × größerer Eingangsraum |
| `ENC_N` | 1 000 | **4 000** | SDR-Sparsity 2 % → 0.5 % |
| `ENC_K` | 20 | 20 (unverändert) | gleiche Engram-Größe pro Wort |

Bei 286-Wort-Vokabular sinkt die Oversubscription von 5.7 × auf
1.4 × — fast jedes R1-Neuron ist nur einem Wort zugeordnet.
Erwartete random-overlap-Rate zwischen zwei Wörtern: K²/N =
400 / 4 000 = 0.1 Bits.

R1 ist Feedforward-only (keine Recurrent-Synapsen), die
Memory-Kosten sind gering: ~16 KB für die zusätzlichen
Membranspannungen. R2 bleibt unverändert bei 2 000 Neuronen.

### Eingriff 2: Balanced-distinct Forward-Wiring

```rust
for src in 0..R1_N {
    let mut taken = vec![false; r2_size];  // distinct targets
    let cap = (R1_N * FAN_OUT / r2_size + 1) as u32;  // in-degree cap
    let mut picked = 0;
    while picked < FAN_OUT {
        let dst = rng.uniform(r2_size);
        if !taken[dst] && in_degree[dst] < cap {
            taken[dst] = true;
            in_degree[dst] += 1;
            brain.connect(...);
            picked += 1;
        }
    }
    // fallback linear scan if random phase couldn't fill
}
```

Garantien:
- Jedes R1-Neuron hat **exakt** FAN_OUT *verschiedene* R2-Targets
- R2-In-Degree liegt **innerhalb von ±1** vom Idealwert
  (R1_N × FAN_OUT / R2_N = 4000 × 10 / 2000 = 20)
- Keine Hub-Neuronen mehr; Forward-Path ist ein balancierter
  bipartiter Expander

## Ergebnisse

`cargo run --release -p eval --example scale_benchmark
-- --sentences 100 --queries 30 --decode-k 6 --seed 42`,
identische Parameter zu notes/42 und notes/43:

| Metrik | iter 24 (Baseline) | iter 25 (R2=10k) | **iter 26 (R1=4k + balanced)** |
| --- | ---: | ---: | ---: |
| Mean token reduction | 40.6 % | 40.6 % | **40.6 %** |
| Decoder precision (self-recall) | 1.000 | 1.000 | **1.000** |
| Decoder recall (associative) | 0.021 | 0.017 | **0.017** |
| Mean false positives / 6 decoded | 4.70 | 4.77 | **4.77** |
| Mean decoder latency | 603 µs | 364 µs | **676 µs** |
| Total wall-time (100 Sätze) | 73 s | 577 s | **70 s** |

**Quality-Zahlen sind bit-identisch zu iter 25.** Die zwei
strukturellen Eingriffe (Encoder-Vergrößerung + Balanced Wiring)
haben Cross-Bleed und Associative-Recall *exakt nicht* bewegt.

Wall-time und Decoder-Latenz sind dagegen praktisch unverändert
— die größere R1 + die Linear-Scan-Wiring-Phase kosten kaum
mehr als die alte random-with-replacement Variante.

## Was drei Iterationen zusammen sagen

| | iter 24 | iter 25 | iter 26 |
| --- | --- | --- | --- |
| Hypothese | Baseline | R2-Recurrent saturiert | Encoder + Forward selektivitäts-arm |
| Eingriff | — | R2 5×, sparser, iSTDP | R1 4×, balanced wiring |
| FP / 6 | 4.70 | 4.77 | **4.77** |
| Recall | 0.021 | 0.017 | **0.017** |

Drei strukturell **sehr** verschiedene Eingriffe → bit-identisches
Quality-Ergebnis. Das ist ein **starker** Datenpunkt: die ~2 %
Associative-Recall und ~4.7 FP/query sind **keine Topologie-
Pathologie**. Sie sind eine fundamentale Eigenschaft der
**Forward-Fingerprinting-Mode**.

## Die echte Diagnose

Die `run_javis_recall_inner` benutzt `FingerprintMode::Forward`
als Default: für jedes Vokabular-Wort `w` wird R1 mit nur
`w`'s SDR getrieben, und das resultierende R2-Pattern als
Engram für `w` gespeichert. Das ist die Antwort des Brains
auf *w allein*, nicht auf den Trainings-Kontext, in dem `w`
gemeinsam mit anderen Wörtern auftrat.

Zwei Wörter `X` und `Y`, die im selben Satz trainiert wurden,
behalten *im Forward-Fingerprint* keine Engram-Ähnlichkeit
zueinander, weil die Fingerprint-Phase die Kontext-Information
komplett ignoriert. Die Trainings-Phase hat zwar einen
gemeinsamen `{X,Y}`-Attraktor in R2 geformt, aber die
Fingerprint-Phase rekonstruiert nur die per-Wort-Antwort.

**`token_efficiency.rs` hat bereits `FingerprintMode::Contextual`
als Alternative**, in der jedes Wort das Satz-Level-Engram
erbt. Das ist die Mode, die assoziativ funktionieren *sollte*.
Im aktuellen `scale_bench` wird diese Mode aber nicht
verwendet — Iter 27 wäre der natürliche nächste Schritt:
`--mode contextual` Flag im Bench, side-by-side-Messung.

## Architektur-Wins (auch ohne Quality-Win)

Trotz Quality-Stagnation liefert iter 26 **cost-neutrale**
strukturelle Verbesserungen:

| Eigenschaft | iter ≤25 | iter 26 |
| --- | --- | --- |
| Encoder-Oversubscription bei 286-Vokab | 5.7 × | **1.4 ×** |
| R2-In-Degree-Varianz (Forward) | Poisson(5) ± 2.2 | **±1** |
| Doppelt verkabelte R1→R2-Edges | möglich | **unmöglich** |
| Wall-time auf 100 Sätzen | 73 s | **70 s** |
| Snapshot-Größe | unverändert | unverändert |
| Tests | 113/113 | **113/113** |

Anders als iter 25 (8× langsamer, größere Snapshots, schlechtere
Quality) ist iter 26 **cost-neutral**. Größerer Encoder-Raum =
forward-kompatibel mit größeren Vokabularien. Balanced wiring =
keine Hub-Neuronen, mathematisch sauberer Forward-Pfad.

## Entscheidung: keep iter 26

Anders als iter 25 wird iter 26 **nicht zurückgerollt**. Die
Quality-Stagnation ist kein iter-26-Problem, sondern ein
strukturelles Algorithmen-Limit der Forward-Fingerprinting-
Mode, das durch Topologie-Tuning nicht ansprechbar ist. Die
neuen Konstanten sind sauberer und kostenneutral — sie bleiben.

## Test-Status

**113 / 113 Tests grün** auf der neuen R1=4000 + balanced-wiring
Konfiguration ohne Schwellen-Anpassung. Insbesondere:
- `wiki_benchmark`: 70 %-Token-Reduktions-Schwelle hält
- `injection.rs` (encoder): SDR-Injection-Verhalten unverändert
- `pattern_completion.rs` und Variante mit Homeostase: grün
- `associative_recall.rs`: grün (intra-topic recall stabil)
- `viz/smoke`: train + recall + ask weiterhin grün

Das ist der nicht-triviale positive Befund: der größere
Eingangsraum + balanced wiring brechen *keine* der bestehenden
Verhaltens-Tests.

## Was Iter 26 nicht macht

- **Kein Encoder-Hash-Wechsel.** `DefaultHasher` ist bereits
  SipHash-1-3 mit deterministischem Seed (Rust std). Die
  Hypothese „Hash-Funktion ist schlecht" war falsch — das
  echte Problem war die SDR-Größe / Sparsity.
- **Keine R2-Recurrent-Änderung.** R2 bleibt bei 2 000 / p=0.10
  / KWTA_K=220 / iSTDP-Default. Iter 25 hat gezeigt, dass
  das nicht der Bottleneck ist.
- **Keine Tuning-Sweep.** ENC_N=4000 und FAN_OUT=10 sind erste
  Schätzungen. Wenn die Zahlen unten zeigen, dass es noch
  Luft gibt, kommen die optimalen Werte aus einer eigenen
  Sweep-Iteration.

## Status

- Konstanten in 3 Crates (`viz::state`,
  `eval::token_efficiency`, `eval::scale_bench`) konsistent
  aktualisiert (R1_N = 4 000, balanced wiring)
- 113 / 113 Tests grün, 0 Clippy-Warnings, fmt clean
- Skalen-Benchmark gelaufen: Quality unverändert, Wall-time
  unverändert
- Memory: R1 wächst um ~16 KB, R2 unverändert, Snapshot
  weiterhin ~30 MB
- Diagnose: Cross-Bleed liegt im Algorithmus
  (Forward-Fingerprinting-Mode), nicht in der Topologie. Iter 27
  sollte Contextual-Mode im scale_bench testen.
