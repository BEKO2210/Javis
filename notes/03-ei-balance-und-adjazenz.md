# 03 — E/I-Balance, Adjazenzliste, neuer Throughput

**Stand:** Inhibition stabilisiert das Netz. Adjazenzliste skaliert die
Synapsen-Iteration auf O(degree) pro Spike. Beides zusammen löst das
Runaway-Problem aus Notiz 02 *und* macht 4× größere Netze möglich.

## Änderungen am `snn-core`

### Dale-Prinzip

- `LifNeuron` trägt jetzt einen `kind: NeuronKind` (`Excitatory` oder
  `Inhibitory`).
- Im `Network::step` bestimmt der Typ des feuernden Neurons das *Vorzeichen*
  des post-synaptischen Stroms — das Synapsen-Gewicht selbst bleibt eine
  nicht-negative Magnitude.
- Default in `LifNeuron::new` ist `Excitatory`, damit ältere Tests
  unverändert grün bleiben.

### Sparse Adjazenz

- Bei jeder Verbindung werden Edge-IDs in `outgoing[pre]` und
  `incoming[post]` eingetragen (`Vec<u32>` pro Neuron, kompakter als
  `usize`).
- `step()` iteriert beim Feuern nur die Outgoing-Edges (Lieferung + LTD)
  und Incoming-Edges (LTP) — statt jedes Mal über die ganze flache
  Synapsen-Liste.

### E/I-Wiring im Benchmark

- 80 % E, 20 % I.
- Synaptische Skalierung: `g_exc = 0.20`, `g_inh = 0.80`. Die
  hemmenden Neuronen sind *einzeln* stärker, weil sie weniger sind —
  das ergibt grob ausgeglichene Netto-Eingänge und ist das Standard-
  Rezept aus der Brunel-/Vogels-Abbott-Linie.

## Throughput nach Refactor (release-Build)

Workload identisch zu Notiz 02 (1 s simuliert, dt=0.1 ms, p=0.1, STDP an,
Poisson-Hintergrund 50 Hz / 80 nA pro Neuron).

| N    | Synapsen   | Wall    | RTF     | Σ Spikes | ⟨rate_E⟩ | ⟨rate_I⟩ | Status         |
| ---- | ---------- | ------- | ------- | -------- | -------- | -------- | -------------- |
| 100  |       953  |  12 ms  |   81 ×  |       66 |  0.7 Hz  |  0.7 Hz  | stabil         |
| 250  |     6 238  |  24 ms  |   41 ×  |      150 |  0.6 Hz  |  0.6 Hz  | stabil         |
| 500  |    25 052  |  47 ms  |   21 ×  |      312 |  0.6 Hz  |  0.6 Hz  | stabil         |
| 1000 |    99 701  |  96 ms  |   10 ×  |      570 |  0.6 Hz  |  0.5 Hz  | stabil         |
| 2000 |   399 408  | 191 ms  |  5.2 ×  |    1 228 |  0.6 Hz  |  0.6 Hz  | **stabil** ✓   |
| 4000 | 1 598 732  | 407 ms  |  2.5 ×  |    2 640 |  0.7 Hz  |  0.7 Hz  | **stabil** ✓   |

### Vergleich mit Notiz 02

| N    | RTF vorher           | RTF jetzt       |
| ---- | -------------------- | --------------- |
| 100  | 117 ×                |  81 ×           |
| 1000 |   8 ×                |  10 ×           |
| 2000 | **0.00 × (runaway)** | **5.2 ×** ✓     |
| 4000 | n/a                  | **2.5 ×** ✓     |

- Bei kleinem N kostet der Adjazenz-Overhead etwas (extra Indirektion und
  ein Vec pro Neuron). Akzeptabel — bei kleinen Netzen sind wir ohnehin
  nirgends nahe an einer Grenze.
- Ab N=1000 zieht die Adjazenzliste deutlich an, und ab N=2000 ist sie
  *zusammen mit* Inhibition der Unterschied zwischen „Echtzeit + Headroom"
  und „läuft Minuten in den Tod".

## Neuer Test (`ei_stability`)

Regression-Guard: 500-Neuronen-E/I-Netz unter 1 s Poisson-Hintergrund muss
unter 30 Hz mittlere Feuerrate bleiben *und* darf nicht still sein.

Aktuelles Messergebnis: `rate_E=0.62 Hz, rate_I=0.63 Hz` — gesunder
asynchron-irregulärer Zustand, sehr nah an dem, was Cortex in vivo zeigt.

## Neues Größen-Budget

- **Pro Region:** 2000–4000 Neuronen sind komfortabel. Mit einem CPU-Thread
  bleibt 2.5–5 × Echtzeit-Headroom.
- **Multi-Region:** mehrere Regionen lassen sich parallel auf separaten
  Threads simulieren, Inter-Region-Verbindungen über sparse Channels.
- 16 Regionen × 2000 Neuronen = **32 000 Neuronen Live-Simulation**
  innerhalb des CPU-Budgets, vorausgesetzt Inter-Region-Traffic bleibt
  sparsam (was er biologisch sowieso ist — long-range Konnektivität ist
  &lt;1 %).

## Nächste Schritte

1. **Multi-Region-Architektur** — `Region` als Wrapper um `Network`,
   `Brain` als Liste von Regionen mit Inter-Region-Edges. Jede Region kann
   später auf eigenen Thread.
2. **Encoder-Stub** — Text/Code → Sparse Distributed Representation,
   Hash-basiert für den ersten Wurf.
3. **Erste Sinn-Demo:** zwei Regionen, eine empfängt SDRs, eine zweite
   speichert via STDP — Recall einer Region aus der anderen.
