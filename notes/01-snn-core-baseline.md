# 01 — SNN-Core Baseline

**Stand:** Erstes Crate `snn-core` mit LIF-Neuron, statischer Synapse,
exponentiellem PSC und Pair-STDP.

## Was implementiert ist

- `LifNeuron` — Leaky Integrate-and-Fire mit Refraktärzeit.
  Forward-Euler-Integration bei `dt = 0.1 ms`.
  Default-Parameter biologisch plausibel (V_rest=−70 mV, V_th=−55 mV,
  τ_m=20 ms, R_m=10 MΩ, ref=2 ms).
- `Synapse` — gerichtete Verbindung mit Gewicht. Spike fügt `weight` zum
  postsynaptischen `i_syn` hinzu, das mit τ_syn=5 ms exponentiell zerfällt.
- `Network` — `Vec<LifNeuron>` + `Vec<Synapse>`, flacher Step-Loop.
- `StdpParams` + Trace-basierte Pair-STDP-Regel im `Network::step`.

## Tests (alle grün)

| Test                                            | Aussage                                              |
| ----------------------------------------------- | ---------------------------------------------------- |
| `single_neuron_fires_at_expected_rate`          | I=2 nA → ≈29 Spikes/s (analytisch erwartet)          |
| `subthreshold_input_does_not_fire`              | I=1 nA (< Rheobase 1.5 nA) → keine Spikes            |
| `isolated_neuron_without_input_stays_silent`    | Kein Input → kein Spike                              |
| `presynaptic_spike_drives_postsynaptic_spike`   | Starke Synapse koppelt Pre → Post                    |
| `pre_before_post_strengthens_weight`            | LTP funktioniert                                     |
| `post_before_pre_weakens_weight`                | LTD funktioniert                                     |

## Bekannte Vereinfachungen

- τ_syn ist im `Network::step` aktuell hart auf 5 ms gesetzt, statt aus
  jeder Synapse einzeln zu lesen. Reicht für Baseline; wird gelockert,
  sobald wir Synapsen-Diversität brauchen.
- Keine synaptische Verzögerung (delay = 0). Echte Axone haben 1-10 ms.
- Keine Neuromodulation (Dopamin/Acetylcholin). Nötig für „belohnungsgetriebenes"
  Lernen.
- Kein Sparse-Sampling — `step()` iteriert über alle Synapsen pro Spike.
  Bei vielen Neuronen O(N·E). Optimierung später.

## Nächste Schritte (Vorschlag)

1. **Mikro-Benchmark**: Wie viele Neuronen × Synapsen × ms/s schaffen wir
   single-thread? (gibt uns das Größen-Budget für Region-Größe)
2. **Erste „Region"**: zufällig verbundene Population, gemessene
   Spontan-Aktivität bei Rauschen.
3. **Assembly-Bildung**: zwei zeitlich korrelierte Inputs → emergente
   Co-Aktivierung über STDP.
