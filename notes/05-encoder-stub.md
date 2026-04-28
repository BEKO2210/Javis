# 05 — Encoder-Stub: Text → SDR → Spikes

**Stand:** Symbolische Eingabe (Text) wird deterministisch in Sparse
Distributed Representations (SDRs) übersetzt und kann in eine Region
des SNN injiziert werden.

## Architektur

```
text  ──tokenize──▶  words  ──hash──▶  per-word SDR (k bits)
                                            │
                                            ▼
                                          union
                                            │
                                            ▼
                                       text SDR  ──inject_sdr──▶  Network
```

## Crate-Layout (`crates/encoders`)

| Modul        | Aufgabe                                                       |
| ------------ | ------------------------------------------------------------- |
| `sdr`        | `Sdr { n, indices }` — sortiert, unique. `union`, `overlap`.  |
| `text`       | `TextEncoder { n, k }` — tokenize, `encode_word`, `encode`.   |
| `inject`     | `inject_sdr(net, &indices, drive_na)`                         |

Standard-Library only. Hashing über `std::collections::hash_map::DefaultHasher`,
deterministisch innerhalb eines Builds.

## Wie der Hash arbeitet

Pro Wort und pro „Salt"-Iteration:

```rust
let mut h = DefaultHasher::new();
word.hash(&mut h);
salt.hash(&mut h);
let v = (h.finish() % n as u64) as u32;
```

Solange wir noch keine `k` distinkten Indizes gesammelt haben, salt++.
Ein lokaler `Vec<bool>` der Größe `n` dient als Bit-Set, um Duplikate
ohne `HashSet`-Allokation zu vermeiden. Anschließend `sort_unstable`.

Das Ergebnis ist:
- deterministisch (stabil über Calls innerhalb des Builds)
- exakt `k` aktive Bits pro Wort
- gleichmäßig über `[0, n)` verteilt

## SNN-Integration

```rust
pub fn inject_sdr(network: &mut Network, sdr_indices: &[u32], drive_na: f32) {
    for &idx in sdr_indices {
        if let Some(slot) = network.i_syn.get_mut(idx as usize) {
            *slot += drive_na;
        }
    }
}
```

Schreibt direkt in die post-synaptische Strom-Leitung der Ziel-Neuronen.
Der nächste `network.step` integriert das ganz normal unter den
LIF-Dynamiken — starker Drive lässt sie feuern, schwacher Drive nur
depolarisieren. `+=`-Semantik, damit mehrere Injektionen pro Schritt
korrekt akkumulieren.

## Tests

### `encoding.rs` (5 grün)

- **`shared_word_yields_exact_overlap`** — der Kern-Beweis:
  - `enc.encode("hello world").overlap(enc.encode_word("hello")) == 20`
  - `enc.encode("hello rust").overlap(enc.encode_word("hello"))  == 20`
  - `enc.encode("hello world").overlap(enc.encode("hello rust")) >= 20`,
    Überschuss ≤ 3 (Zufallskollisionen, Erwartungswert k²/n ≈ 0.2)
  - `enc.encode("foo bar").overlap` mit beiden ≤ 3 — keine semantische Nähe
- `deterministic_across_calls` — gleiche Eingabe ⇒ gleiche Indizes
- `punctuation_is_stripped` — `"hello, world!"` ≡ `"hello world"`
- `case_is_normalised` — `"Hello WORLD"` ≡ `"hello world"`
- `union_and_overlap_agree_on_simple_case` — Sanity auf 16-Bit-SDRs

### `injection.rs` (3 grün)

- **`injected_sdr_fires_exactly_its_neurons`** — der Kern-Beweis:
  - 2048 LIF-Neuronen, kein Wiring, keine externen Currents
  - `inject_sdr(net, &sdr.indices, 700.0)`
  - `let fired = net.step(&[]);`
  - `fired_sorted == sdr.indices` — die Feuermenge **ist exakt** das SDR-Set
- `untargeted_neurons_remain_at_rest` — alle Nicht-SDR-Neuronen bleiben
  bei `V_rest = -70 mV`, kein Leakage
- `weak_drive_only_depolarises` — bei 30 nA bleiben SDR-Neuronen unter
  Schwelle, sind aber sichtbar depolarisiert (V > -69.9 mV)

## Mathematik des Drives

LIF-Schrittformel: `dV ≈ (dt/τ_m) · R · I_syn`. Mit Defaults
(τ_m = 20 ms, R = 10 MΩ, dt = 0.1 ms):

| `drive_na` | dV pro 0.1-ms-Schritt | Effekt nach 1 Schritt          |
| ---------- | --------------------- | ------------------------------ |
| 30 nA      | 1.5 mV                | depolarisiert, kein Spike      |
| 400 nA     | 20 mV                 | Schwelle erreicht, feuert      |
| 700 nA     | 35 mV                 | sicheres Feuern in 1 Schritt   |

## Wo das hinführt

Mit dem Encoder-Stub wird Javis schreibfähig:

```
text input  →  TextEncoder  →  SDR  →  inject_sdr(input_region, …)
                                                │
                                                ▼
                                           Brain.step
                                                │
                                                ▼
                                  spikes propagieren in
                                  assoziative + memory-Regionen
                                                │
                                                ▼
                                          STDP konsolidiert
```

Das ist die Grundlage, jeden weiteren Input-Modus (Code, JSON, Embeddings)
über denselben SDR-Pfad zu schicken.

## Nächste Schritte

1. **Multi-Region-Assembly-Demo:** Text in einer Eingangs-Region
   einspeisen, eine zweite Region speichert über STDP. Test:
   gleiches Text-SDR zu einem späteren Zeitpunkt → die zweite Region
   reproduziert ihre damalige Antwort, *ohne* dass die Eingangs-Region
   nochmal komplett angesteuert wird.
2. **Recall-Decoder:** Umkehrung — Spike-Aktivität → SDR → Text-Kandidaten
   (über inverse Hash-Tabelle der bisher gesehenen Wörter).
3. **Token-Effizienz-Messung:** echtes Korpus, fester Query-Set,
   Vergleich: naives RAG vs. Javis (Tokens-an-LLM gezählt).
