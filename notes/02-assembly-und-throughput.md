# 02 — Assembly-Bildung und Throughput-Budget

**Stand:** Erster echter „Speicher-Beweis" und Hardware-Budget für CPU-Single-Thread.

## Assembly-Bildung — Recall via STDP

### Setup

- 100 LIF-Neuronen, zufällig rekurrent verbunden, p_connect = 0.1 → ~950 Synapsen
- Initialgewichte uniform in [0.05, 0.30]
- STDP an mit `a_plus=0.05`, `a_minus=0.025`, sonst Defaults
- Pattern A := Indizes 0..10
- Pattern B := Indizes 10..20
- Pattern C := Indizes 50..60 (Kontrolle, nie mit B gepaart)

### Protokoll (Test `pattern_b_recalls_after_training_via_pattern_a`)

1. **Pre-A → B Recall** (STDP aus): Poisson-Drive nur auf A, 50 ms, zähle B-Spikes.
2. **Pre-C → B Recall** (STDP aus): Poisson-Drive nur auf C, 50 ms, zähle B-Spikes.
3. **Training** (STDP an): 100 Trials je 100 ms.
   Pro Trial: A solo (10 ms), A+B Overlap (10 ms), B solo (10 ms), 70 ms Idle.
4. **Post-A → B Recall** (STDP aus): wie Schritt 1.
5. **Post-C → B Recall** (STDP aus): wie Schritt 2.

### Ergebnis

```
drive  pre_A=34  pre_C=34  post_A=40  post_C=34
B-out  pre_A= 0  pre_C= 0  post_A=26  post_C= 0
```

- Vor Training treibt weder A noch C die B-Neuronen.
- Nach Training feuern B-Neuronen 26-mal, **ausschließlich** wenn A getrieben wird.
- Kontroll-Pattern C bleibt unverbunden zu B → keine generische Hyperaktivität.

**Interpretation:** STDP hat selektiv die A→B-Synapsen verstärkt. Echtes
assoziatives Erinnern in einem Spiking-Netzwerk. Damit ist die Kern-These
für Javis (Wissen als Assemblies, nicht als Embeddings) experimentell gestützt.

## Throughput-Budget (release-Build, Single-Core)

Workload: 1 s simulierte Zeit, dt=0.1 ms, p_connect=0.1, STDP an,
Poisson-Hintergrund 50 Hz / 80 nA pro Neuron.

| N    | Synapsen | Wall   | Real-Time-Faktor | Spikes  | ⟨Rate⟩  | Bemerkung                |
| ---- | -------- | ------ | ---------------- | ------- | ------- | ------------------------ |
| 100  |     953  |   8 ms |  117 ×           |     67  |  0.7 Hz | Test-Workload            |
| 250  |   6 238  |  16 ms |   63 ×           |    158  |  0.6 Hz |                          |
| 500  |  25 052  |  37 ms |   27 ×           |    367  |  0.7 Hz |                          |
| 1000 |  99 701  | 125 ms |    8 ×           |    788  |  0.8 Hz |                          |
| 2000 | 399 408  |   ≫    |   < 0.01 ×       | 828 669 |  414 Hz | **Runaway** (siehe unten)|

### Erkenntnisse

- **Bis ~1000 Neuronen ist Echtzeit (RTF ≥ 1) auf einem CPU-Thread komfortabel
  drin** — mit 8× Headroom. Das ist unser sinnvolles Größen-Budget für eine
  einzelne Region.
- Der Step-Loop ist aktuell O(spikes × E). Mit einer Adjazenzliste pro
  Pre-Neuron (Outgoing-Buckets) wird das O(spikes × ⟨degree⟩) — Faktor ~10
  Speedup bei N=1000 plausibel.
- **N=2000 explodiert in Seizure-artige Aktivität** (414 Hz mittlere Feuerrate).
  Ursache: nur exzitatorische Synapsen + STDP-Verstärkung + große rekurrente
  Verbindungsdichte. Das ist genau das Problem, das echte Hirne mit
  **Inhibition** lösen. → nächster Forschungsschritt.

### Größen-Budget Empfehlung

- **Pro Region:** 500–1000 Neuronen, Konnektivität ~10 %.
- **Mehrere Regionen parallel** auf separaten Threads — solange Kommunikation
  zwischen Regionen sparsam ist (was biologisch ohnehin der Fall ist), skaliert
  das fast linear.
- Das gibt uns für ein Multi-Region-System mit z.B. 16 Regionen × 1000 Neuronen
  = 16 000 Neuronen Live-Simulation ohne GPU.

## Nächste Schritte (priorisiert)

1. **Inhibition einführen** (~20 % der Neuronen mit negativen Synapsen
   vom Dale-Prinzip-Typ). Damit sollte N=2000 stabil bleiben.
2. **Adjacency-Liste statt flacher Synapsen-Liste** — Performance-Hebel,
   ohne den Algorithmus zu ändern.
3. **Multi-Region-Architektur** — Regionen als Module, sparse Inter-Region-
   Verbindungen, jeder Region kann auf eigenen Thread.
4. **Encoder-Stub** — erster Versuch: Text → SDR (Sparse Distributed
   Representation), simple Hash-basiert.
