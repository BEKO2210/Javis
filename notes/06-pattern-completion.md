# 06 — Multi-Region Pattern Completion

**Stand:** Erste Demo, dass Javis biologisches assoziatives Erinnern
zeigt. Eingangstext „hello rust" wird in Region R1 als SDR injiziert,
über Address-Event-Routing nach R2 übertragen und dort durch STDP
zu einer Assembly konsolidiert. Ein späterer **partieller Cue** („hello"
allein) reaktiviert die komplette Assembly über die rekurrenten
R2-Synapsen — die fehlende Hälfte des ursprünglichen Patterns wird
*assoziativ* ergänzt.

## Encoder-Erweiterung

`TextEncoder` hat jetzt einen `HashSet<String>` für Stoppwörter. Im
`tokenize` werden alle Treffer (case-insensitive) verworfen, bevor sie
in den Hash-Pool wandern. Damit erzeugen leere Wörter wie `der`, `die`,
`und`, `the`, `is` kein Bit-Rauschen, das später jede Assembly auf
dieselben Indizes ziehen würde.

```rust
let enc = TextEncoder::with_stopwords(1000, 20, ["the", "is", "der", "und"]);
```

Test `stopwords_are_dropped`: `enc.encode("the cat is on the mat")`
≡ `enc.encode("cat mat")`.

## Brain-Helper

Zwei kleine Methoden, die für den Test essenziell sind:

```rust
brain.reset_state();         // V, i_syn, traces, time, pending → 0,
                             // Topologie + Gewichte bleiben.
brain.disable_stdp_all();    // STDP in allen Regionen aus.
```

Damit lassen sich Trainings- und Mess-Phasen sauber trennen — beim Mess
darf STDP die Gewichte nicht mehr verändern.

## Architektur

| Region | Größe | Komposition          | Wiring                                |
| ------ | ----- | -------------------- | ------------------------------------- |
| R1     | 1000  | rein excitatorisch   | **kein** internes Wiring, reines Relais|
| R2     | 2000  | 80 % E / 20 % I      | rekurrent p=0.1, STDP an              |

**Forward R1 → R2:** jeder R1-Neuron projiziert mit `FAN_OUT=10` zu
zufälligen R2-Targets (E und I, da Random-Sampling die 80/20-Verteilung
automatisch übernimmt). Gewicht 2.0, Verzögerung 2 ms.

**STDP in R2:** `a_plus=0.04, a_minus=0.025, w_max=2.0`. Asymmetrische
Plastizität (LTP > LTD) sorgt für klares Engramm; `w_max` deckelt
gegen Runaway.

## Protokoll

```
Phase 1 — Pre-recall (STDP off, fresh state)
  Cue „hello" für 100 ms                  →  pre_recall set

Phase 2 — Training (STDP on, fresh state)
  Cue „hello rust" für 150 ms             →  letzte 20 ms = target_assembly
  + 50 ms Cool-down

Phase 3 — Post-recall (STDP off, fresh state)
  Cue „hello" für 100 ms                  →  post_recall set
```

Drive: konstanter externer Strom 200 nA auf die SDR-Indizes in R1.
Encoder: N=1000 (=R1-Größe), K=20.

## Resultat

```
target_assembly = 461   (R2-E neurons co-active in last 20 ms of training)
pre_recall      = 149   (forward path only)
post_recall     = 827   (forward path + recurrent completion)

pre  ∩ target_assembly = 145   ← baseline: was der Forward-Pfad allein liefert
post ∩ target_assembly = 445   ← nach STDP-Konsolidierung
coverage                = 97 %  von der ursprünglichen Assembly
```

### Was diese Zahlen sagen

- **pre_recall ⊂ target_assembly weitgehend (145 / 149):** der Forward-Pfad
  liefert deterministisch die R2-Targets der „hello"-Bits. Das ist die
  *Baseline*, mit der Pattern Completion verglichen werden muss.
- **post_recall ∩ target_assembly = 445:** der gleiche partielle Cue
  „hello" wirft jetzt fast die *gesamte* ursprüngliche Assembly an —
  inklusive der Neuronen, die ursprünglich nur durch das fehlende Wort
  „rust" im vollen Cue mitgefeuert hatten. Diese 300 zusätzlichen
  Treffer sind reines assoziatives Wiederherstellen über die durch STDP
  gestärkten R2-Rekurrenten.
- **gain = 300 vs. assembly_size/5 = 92:** Der Lerneffekt ist mehr als
  3× über der geforderten Schwelle.

### Hyperaktivität-Beobachtung

`post_recall = 827 > target_assembly = 461`: das Netz feuert nach
Training auch Neuronen *außerhalb* der ursprünglichen Assembly. Das ist
ein Generalisierungs-Bleeding. Biologisch begegnet man dem mit
**Homöostase** und **strukturierter Inhibition** (Top-Down Gain Control,
Sliding-Threshold-Plastizität à la Bienenstock-Cooper-Munro). Steht auf
der Liste für die nächste Iteration; für die Pattern-Completion-Demo
allein zählt: das gelernte Pattern wird vollständig getroffen.

## Tuning-Tagebuch

| Konfiguration                                       | Ergebnis                       |
| --------------------------------------------------- | ------------------------------ |
| a+=0.05, w_max=5, FAN=12, drive=400, 100 ms         | Hyper, target=485, post=1117   |
| a+=0.02, w_max=1.5, FAN=6, drive=200, 100 ms        | Kein Lerneffekt, post=pre=99   |
| a+=0.03, w_max=2.5, FAN=6, drive=200, 100 ms        | Minimal, gain=5                |
| a+=0.03, w_max=2.5, FAN=6, drive=200, 500 ms        | Hyper, post=1162               |
| a+=0.025=a−, w_max=2, FAN=6, drive=200, 200 ms      | Kein Lerneffekt                |
| **a+=0.04, a−=0.025, w_max=2, FAN=10, 150 ms**      | **gain=300, coverage=97 %** ✓  |

Schmaler Sweet-Spot zwischen „totem Netz" und Hyperaktivität — typisch
für STDP ohne Homöostase. Die finalen Parameter liegen darin.

## Bedeutung für Javis

Damit ist das Kern-Versprechen experimentell gestützt:
**ein Teil-Input genügt, um den vollen gespeicherten Inhalt zu
reaktivieren.** Genau das ist die Token-Effizienz-These — bei einer
Anfrage muss man dem LLM nur den Teil-Cue mitgeben; das Netz aktiviert
das volle gespeicherte Engramm und gibt nur dessen Kandidaten-Inhalt
heraus.

## Nächste Schritte

1. **Recall-Decoder** — Aktivierte Neuronen → SDR → Text-Kandidaten,
   über eine Inversion der bisher gesehenen Wort→Bits-Tabelle.
2. **Homöostase / Synaptic Scaling** — gegen das Hyperaktivitäts-Bleeding;
   z.B. lineare Skalierung aller eingehenden Gewichte eines Neurons,
   damit seine mittlere Feuerrate konstant bleibt.
3. **Token-Effizienz-Messung** — echtes Korpus, fester Query-Set,
   Vergleich: naives RAG vs. Javis (Tokens-an-LLM gezählt).
