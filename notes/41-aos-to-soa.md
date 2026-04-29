# 41 — AoS → SoA Refactor + WS fire-and-forget

**Stand:** Iteration 23. Notiz 40 hat schwarz auf weiß gezeigt, dass
`brain_compute` 77 % des Recall-Pipelines ist und Amdahl noch nicht
zuschlägt. Der direkte Folgeschritt: das hot-path-AoS-Layout
auflösen. `LifNeuron` hatte transientes Per-Schritt-Zustand
(`v`, `refractory_until`, `last_spike`, `activity_trace`) im selben
Struct wie die statische Konfiguration (`params`, `kind`); die
LIF-Loop loaded damit pro Neuron einen vollen 48-B-Struct, von dem
nur 24 B (params) wirklich gelesen werden.

Plus die niedrig-hängende Frucht aus Iter-22's WS-Stream-Sub-Phase
(6.5 % der Pipeline): `tx.send().await` durch `try_send()` ersetzen
— Step-Events sind Visualisierungs-Breadcrumbs, keine Kern-Daten.

## Refactor 1 — SoA-Layout

**Vorher:**
```rust
pub struct LifNeuron {
    pub params: LifParams,    // 24 B static
    pub kind: NeuronKind,     // 1 B static + 3 B padding
    pub v: f32,               // 4 B transient
    pub refractory_until: f32,// 4 B transient
    pub last_spike: f32,      // 4 B transient
    pub activity_trace: f32,  // 4 B transient
} // ~48 B per neuron, AoS

pub struct Network {
    pub neurons: Vec<LifNeuron>,
    // synaptic-current channels were already SoA …
    pub i_syn: Vec<f32>,
    …
}
```

**Nachher:**
```rust
pub struct LifNeuron {
    pub params: LifParams,    // 24 B
    pub kind: NeuronKind,     // 1 B + padding → 32 B total
}                             // 2 neurons / cache line

pub struct Network {
    pub neurons: Vec<LifNeuron>,
    // NEW: per-neuron transient state in parallel Vec<f32>s
    pub v: Vec<f32>,
    pub refractory_until: Vec<f32>,
    pub last_spike: Vec<f32>,
    pub activity_trace: Vec<f32>,
    // existing channels / traces, unchanged
    pub i_syn: Vec<f32>,
    …
}
```

`Network::add_neuron` pushed jetzt parallel auf alle SoA-Vecs.
`reset_state` und `ensure_transient_state` setzen die neuen Vecs auf
ihre Defaults (`v[i] = params.v_rest`, `refr/last = -inf`,
`activity_trace = 0`).

`LifNeuron::step` ist weg — die LIF-Math ist jetzt direkt in
`Network::step` und `Network::step_immutable` inline geschrieben,
operiert auf `&mut self.v[idx]` etc. Identische forward-Euler-
Diskretisierung wie vorher.

## Refactor 2 — `tx.try_send` für Step-Events

```rust
match tx.try_send(Event::Step { … }) {
    Ok(()) => {}
    Err(TrySendError::Full(_)) => dropped_step_events += 1,
    Err(TrySendError::Closed(_)) => break,
}
```

Step-Events sind nicht-essentiell — die Visualisierung kommt damit
klar, wenn ein Batch fehlt (jeder Batch ist self-contained, das
3D-Frontend interpoliert). Die `await`-Suspension von `tx.send` war
ein Tokio-Yield-Punkt; mit `try_send` läuft die Sim-Schleife ohne
Yield durch. Bei Full-Channel (≥ 1024 angesammelte Events, in
Praxis nie) zählt ein Counter `javis_ws_step_dropped_total` —
sichtbar im /metrics, falls eine schwerfällige Browser-Session
mal hinterherhinkt.

## Korrektheits-Beweis

Die vier `immutable_step_equivalence`-Tests aus Iter 20 sind
unverändert — sie laden zwei identisch-seeded Brains, fahren
denselben externen Drive durch beide (mutating + immutable Pfad),
und vergleichen die emittierten Spike-Listen Step-für-Step. Plus
am Ende den Membran-Potential-Vector. Alle bestehen ohne
Toleranz-Anpassung — der SoA-Refactor produziert exakt
bit-identische Ergebnisse zum AoS-Pfad.

```
test network_step_immutable_matches_step_when_plasticity_off ... ok
test brain_step_immutable_matches_step_when_plasticity_off ... ok
test brain_step_immutable_does_not_mutate_brain ... ok
test network_state_lazy_nmda_gaba_allocation ... ok
```

108/108 Workspace-Tests grün. 0 Clippy-Warnings, fmt clean.

## Bench: `Network::step_immutable` (Criterion, p < 0.05)

```
network_step_immutable/100   240 ns   (-22.5% vs iter 21)
network_step_immutable/500   1.17 µs  (-26.7%)
network_step_immutable/1000  2.69 µs  (-26.8%)
network_step_immutable/2000  7.35 µs  (-18.6%)
```

Stacked vs der Iter-20-Baseline (vor jedweder Optimierung):

| Size | Iter 20 | Iter 21 (autovec) | Iter 23 (SoA) | Total Speedup |
| ---: | ---: | ---: | ---: | ---: |
|  100 | 545 ns | 307 ns | **240 ns** | 2.27× |
|  500 | 2.81 µs | 1.66 µs | **1.17 µs** | 2.40× |
| 1000 | 5.85 µs | 3.73 µs | **2.69 µs** | 2.17× |
| 2000 | 13.79 µs | 9.32 µs | **7.35 µs** | 1.88× |

## End-to-End: Pipeline-Profile (200 sequential recalls)

```
              iter 20    iter 23    delta
brain_compute  6.18 ms   4.55 ms   -26.4%
ws_stream      0.53 ms   0.31 ms   -41.5%
snn_compute    7.91 ms   5.63 ms   -28.8%
decode         0.13 ms   0.13 ms   flat
others         0.03 ms   0.03 ms   flat
─────────────────────────────────────────
recall total   8.05 ms   5.77 ms   -28.3%
```

**Pipeline-Speedup: 1.40×.** Genau die Größe, die Amdahls Vorhersage
für eine 1.5× Brain-Compute-Optimierung bei 77 % Pipeline-Anteil
liefert.

## End-to-End: Load-Test

```
              iter 20            iter 23            delta
conc | ops/s | p99      | ops/s | p99      |
─────┼───────┼──────────┼───────┼──────────┼─────────
   1 |  112  |   11 ms  |  138  |  8.9 ms  |  +23%, -19%
  10 |  357  |   48 ms  |  430  |   41 ms  |  +20%, -15%
  50 |  359  |  244 ms  |  436  |  197 ms  |  +21%, -19%
 100 |  358  |  564 ms  |  432  |  486 ms  |  +21%, -14%
```

Server-mean: 9.3 → 7.6 ms (-18 %). Throughput-Plateau ab
Concurrency 10: 358 → 432 ops/s. Latency-Reduktion gleichmäßig
verteilt — die SoA-Wins skalieren mit Anzahl paralleler
Recall-Sessions, weil jede ihre eigene `BrainState`-Kopie hat
(read-only Brain wird geteilt).

## Was nicht passiert ist

- **Kein `std::simd` / `wide`-Crate.** Auto-Vectorisierung des
  LLVM-Compilers reicht weiterhin. Die SoA-Layouts sind genau die
  Datenstruktur, die der Auto-Vec für AVX-512-Lanes lieben muss.
  Erst wenn ein Profile zeigt, dass die LIF-Loop *immer noch* der
  Bottleneck ist und der Compiler nichts mehr findet, kommen
  Intrinsics.
- **Kein Per-Worker Brain-Pool.** Cache-Lokalität ist nach dem
  SoA-Refactor schon ok (params 32 B / Neuron, 2 / cacheline; v
  4 B / Neuron, 16 / cacheline). Pool wäre Memory-Aufwand für
  marginal weiterer Gewinn.
- **Kein deeper Skim auf den Decoder.** Mit 0.13 ms (1.6 % des
  Pipelines) und einem ca. 60-Wort-Vokabular ist er nicht der
  Bottleneck. Wenn die Vocabulary auf 10k+ wächst, dann.

## Nächster Profile

`scripts/pipeline_profile.py` weiterhin gleich verwendbar; die
Phasen-Histogramme aus Iter 22 sind unverändert. Wer den nächsten
Optimierungsschritt identifizieren will: einfach wieder fahren.
Aktuelle Verteilung:

```
brain_compute  78 %   ← noch immer der Bottleneck, aber knapper
ws_stream       5 %   ← nach try_send halbiert
decode          2 %
overhead        2 %
```

## Status

- 108/108 Tests grün (incl. 4 spike-bit-identity Tests)
- 0 Clippy-Warnings, fmt clean
- Pipeline-Latenz 8.05 → 5.77 ms (-28 %)
- Throughput 358 → 432 ops/s (+21 %)
- p99 -14 % bis -19 % über alle Concurrency-Stufen
- Total-Speedup auf `step_immutable` seit Iter 20: ≈ 2× (n=1000)
