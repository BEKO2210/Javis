# 43 — Topologie-Scaling: gut dokumentierter Negativ-Befund

**Stand:** Iteration 25. Notiz 42 hat den Cross-Bleed bei
N > 50 Konzepten als architektonische Wand identifiziert.
Naheliegende Hypothese: das R2-Layer mit 2 000 Neuronen und
KWTA_K=220 (11 % Sparsity) hat schlicht zu wenig orthogonalen
Raum; mehr Neuronen + sparser engrams sollten die Wand
verschieben. Diese Iteration hat den Eingriff durchgezogen, mit
identischem 100-Satz / 286-Vokabular Korpus + Seed nochmal
gemessen, und das Ergebnis ist eindeutig: **die Hypothese
stimmt nicht.**

Der Refactor wurde am Ende dieser Iteration **zurückgerollt**.
Die Notiz dokumentiert was probiert wurde, was rauskam, und was
das für die nächste Iteration bedeutet.

## Was probiert wurde

| Konstante | iter ≤24 | iter 25 (probiert) |
| --- | ---: | ---: |
| `R2_N` | 2 000 | 10 000 |
| `R2_P_CONNECT` | 0.10 | 0.03 |
| `KWTA_K` | 220 | 100 |
| `CONTEXT_KWTA_K` | 60 | 30 |
| `FAN_OUT` (R1→R2) | 10 | 30 |
| `IStdpParams.a_plus` | 0.05 | 0.10 |
| `IStdpParams.a_minus` | 0.55 | 1.10 |
| `IStdpParams.w_max` | 5.0 | 8.0 |

Drei Crates (`viz::state`, `eval::token_efficiency`,
`eval::scale_bench`) gespiegelt synchron auf die neuen Werte.

## Was rauskam

`cargo run --release -p eval --example scale_benchmark
-- --sentences 100 --queries 30 --decode-k 6 --seed 42`,
identische Parameter zu notes/42:

| Metrik | iter 24 (Baseline) | **iter 25 (R2=10 000)** | Δ |
| --- | ---: | ---: | --- |
| Mean token reduction | 40.6 % | **40.6 %** | unverändert |
| Decoder precision (self-recall) | 1.000 | **1.000** | unverändert |
| Decoder recall (associative) | 0.021 | **0.017** | -19 % (marginal schlechter) |
| Mean false positives / 6 decoded | 4.70 | **4.77** | +1.5 % (marginal schlechter) |
| Mean decoder latency | 603 µs | **364 µs** | -40 % (besser, K kleiner) |
| Total wall-time (100 Sätze) | 73 s | **577 s** | **~8 × langsamer** |
| Snapshot-Datei | ~30 MB | ~120 MB | ~4 × größer |

**Quality unverändert oder leicht schlechter. Decoder-Latenz
besser, war aber schon sub-ms — kosmetisch. Trainingszeit
8 × langsamer, Snapshot 4 × größer.**

Zusätzlich verifiziert: alle 113 bestehenden Tests passieren
auch mit dem 10 000-Neuron-Brain ohne Schwellen-Anpassung — die
Architektur ist robust gegenüber Topologie-Änderungen, die
qualitative Verbesserung blieb aber aus.

## Warum das Bigger-Brain den Cross-Bleed nicht fixt

Drei plausible Mechanismen, die durch das R2-Skalieren *nicht*
adressiert werden:

1. **Encoder-Kollisionen.** Der `TextEncoder` benutzt
   `DefaultHasher` mit K=20 / N=1000, um aus einem Wort eine
   sparse SDR zu erzeugen. Zwei unverwandte Wörter mit
   ähnlichem Hash bekommen überlappende SDRs. Das
   propagiert durch R1 → R2 als Eingangs-Korrelation.
   Größere R2 ändert daran nichts — die Kollision steckt
   schon im SDR-Eingang.

2. **Forward-Wiring-Selektivität.** R1 → R2 ist mit fixed
   FAN_OUT = 10 (oder 30 in iter 25) zufällig verkabelt, nicht
   selektivitäts-getrimmt. Beim Recall reaktiviert das
   Query-SDR über genau diese Forward-Weights die R2-
   Kandidaten — die R2-Recurrent-Dynamik (wo iSTDP wirkt)
   sieht nur einen *bereits gefärbten* Eingangs-Zustand. Eine
   größere Recurrent-Population mit aggressiverer iSTDP kann
   selektive Engram-Trennwände bauen, aber das hilft erst,
   nachdem der Forward-Pfad schon entschieden hat, *welche*
   Engrams überhaupt aktiv werden.

3. **kWTA-Geometrie ist scale-invariant für das hier
   relevante Phänomen.** Top-100 von 10 000 statt Top-220 von
   2 000 senkt die mathematische Random-Overlap-Wahrscheinlichkeit
   tatsächlich um Größenordnungen — aber zwei Engrams, die
   *systematisch* (nicht zufällig) überlappende Forward-
   Inputs haben, picken auch beide systematisch denselben
   Top-100. Random-overlap-Statistik ist nicht der Engpass.

## Was wir aus der Iteration mitnehmen

1. **Die Architektur ist topologie-robust.** Skalierung um 5×
   bricht keine bestehenden Tests, kein numerisches Problem,
   keine Stabilitätsverluste. Das ist ein positiver Befund
   — falls jemand später wirklich einen Anwendungsfall hat,
   der ein 10 000+-Neuron-Brain rechtfertigt, geht das ohne
   Refactor.

2. **Cross-Bleed ist *nicht* ein R2-Topologie-Problem.** Wir
   können das Topologie-Argument aus der Liste der Verdächtigen
   streichen.

3. **Der eigentliche Gegner ist weiter vorne in der Pipeline:
   Encoder + Forward-Weights.** Die nächste Iteration sollte
   dort ansetzen — z. B. SipHash statt DefaultHasher, größeres
   N, oder selektivitäts-getrimmtes Forward-Wiring.

## Revert

Constants in `viz::state`, `eval::token_efficiency` und
`eval::scale_bench` wurden auf die iter-24-Werte zurückgesetzt.
Tests bleiben 113/113 grün. README ist auf den ehrlichen
iter-24-Stand polished (Performance profile, Known limits,
Reproducibility). Die forward-looking-Behauptung „iter 25
attacks the bottleneck" ist aus dem README raus, weil sie sich
nicht bestätigt hat.

## Status

- 113 / 113 Tests grün auf iter-24-Konstanten (zurückgerollt)
- 0 Clippy-Warnings, fmt clean
- Negativ-Befund vollständig dokumentiert mit echten Zahlen
- Klarer Pfad für die nächste Iteration: Encoder + Forward-
  Wiring statt R2-Topologie
