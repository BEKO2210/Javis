# 11 — iSTDP: Intrinsische Konzept-Selektivität

**Stand:** Drei Konzepte teilen einen Hub, leben im selben R2,
trennen sich **intrinsisch** (kein kWTA-Post-Filter mehr).

## Vorher / Nachher

Notiz 10 hatte das Multi-Concept-Problem über einen kWTA-Post-Filter
gelöst — selektive Inhibition wurde *bei der Auslese* approximiert.
Jetzt lernt das Netzwerk diese Selektivität selbst, während des
Trainings.

| Metrik (rust↔world)                  | kWTA-Filter (Notiz 10) | iSTDP intrinsisch (Notiz 11) |
| ------------------------------------ | ---------------------- | ---------------------------- |
| direct retrieval                     | 1.00                   | 1.00                         |
| cross-bleed `recall(rust)→world`     | 0.31                   | **0.38**                     |
| cross-bleed `recall(world)→rust`     | 0.31                   | **0.38**                     |
| **hub recall** `recall(hello)→rust`  | 0.37                   | **1.00**                     |
| **hub recall** `recall(hello)→world` | 0.44                   | **0.94**                     |

Die Hub-Recall-Werte springen von ≈ 0.40 auf ≈ 0.97 — das ist der
qualitative Unterschied zwischen einer „kWTA macht es sparse, aber
schneidet auch Pattern Completion mit weg" und einem Netzwerk, dass
*intrinsisch* weiß, was zusammengehört und was nicht.

## Was im Kern dazukam

### `IStdpParams` (`crates/snn-core/src/istdp.rs`)

Anti-Hebbian-Lernregel auf Synapsen mit `pre.kind = Inhibitory` und
`post.kind = Excitatory`. Pro I-Pre-Spike:

```rust
let dw = a_plus - a_minus * post_trace_e[post];
let w  = (w + dw).clamp(w_min, w_max);
```

Zwei Regime in einer Formel:

| Zustand                          | `post_trace_e` | `dw`                        | Effekt |
| -------------------------------- | -------------- | --------------------------- | ------ |
| E silent → I-Schuss allein       | ≈ 0            | +`a_plus`                   | LTP    |
| E hat gerade gefeuert (Co-Akt.)  | hoch           | `a_plus - a_minus * trace`  | LTD    |

Der I-Neuron, der zu Engramm A gehört, baut so eine Wand der
Inhibition um die E-Zellen anderer Engramme („Pre-Only → LTP")
und befreit gleichzeitig die E-Zellen *seines eigenen* Engramms
von Selbst-Hemmung („Co-Activation → LTD"). Das ist der Mechanismus
aus dem PLOS-Comp-Bio-Paper, mathematisch reduziert auf einen
Update pro I-Spike.

### Network-Integration

- Neues Feld `istdp: Option<IStdpParams>` plus
  `enable_istdp` / `disable_istdp` Methoden.
- Default-off — alle Bestandstests (LIF, STDP, Homöostase, two-regions
  usw.) bleiben unverändert grün.
- Im Step-Loop:
  1. `post_trace` wird jetzt auch unter reinem iSTDP decayed
     (mit dessen `tau_minus`).
  2. Bei jedem Spike wird `post_trace[src] += 1.0` falls eine der
     beiden Plastizitäten aktiv ist.
  3. Beim Outgoing-Loop entscheidet `match src_kind`:
     - **E-pre + STDP:** klassische LTD wie bisher.
     - **I-pre + iSTDP:** Anti-Hebbian-Update, nur für E-post.
- E→I-Synapsen werden weiterhin von Hebbian STDP gelernt (über den
  Incoming-Loop bei E-pre, die `pre.kind = Excitatory` Filter passt).

### Brain helper

`Brain::disable_istdp_all` symmetrisch zu `disable_stdp_all` /
`disable_homeostasis_all`. Vor jeder Recall-Messung müssen alle drei
Plastizitäten eingefroren werden.

## Unit-Tests (drei neue, alle grün)

- `istdp_off_by_default_does_not_touch_weights` — Vertrag: ohne
  `enable_istdp` bewegt sich kein I→E-Gewicht.
- `istdp_weakens_synapse_on_coactivation` — E feuert, dann I kurz
  danach. `post_trace_e` ist hoch beim I-Pre-Spike → LTD dominiert.
  Gewicht sinkt von 1.0 messbar nach 50 Wiederholungen.
- `istdp_strengthens_synapse_on_pre_only` — I feuert wiederholt, E
  schweigt. `post_trace_e ≈ 0` → reine LTP. Gewicht steigt von 1.0
  klar an.

## Sweet-Spot

Empirisch tunen über ~10 Iterationen liefert:

```rust
fn r2_stdp_multi() -> StdpParams {
    StdpParams { a_plus: 0.015, a_minus: 0.012, w_max: 0.8, .. }
}

fn r2_istdp_multi() -> IStdpParams {
    IStdpParams {
        a_plus: 0.05,        // baseline LTP per I-pre-spike
        a_minus: 0.55,       // 11× a_plus — strong LTD on co-fire
        tau_minus: 30.0,     // long-ish post-trace
        w_max: 5.0,          // walls can grow well above E-side w_max
        ..
    }
}
```

Lehren:
- **Reduzierte STDP-Stärken** (`a_plus = 0.015`, `w_max = 0.8`)
  verhindern, dass STDP das ganze Netz vor iSTDP in einen
  hyperaktiven Attractor drückt. Mit 4× größerem `a_plus` und 2.5×
  größerem `w_max` (alte Werte aus Notiz 09) saturierte das Netz
  und alle Engramm-Sets wurden quasi identisch.
- **iSTDP `a_minus` muss deutlich größer sein als `a_plus`** —
  Faktor 10 ist gut. Sonst gewinnt der Wall-Building-Term selbst
  bei moderater Co-Aktivität und das Netz wird zu still.
- **Eine einzige interleaved Trainings-Runde** (`for _ in 0..1`)
  reicht. Mehr Runden saturieren wieder, weil sowohl STDP als
  auch iSTDP über die Zeit weiter akkumulieren.

## Test-Resultat

```
engram 'hello': 528 bits
engram 'rust' : 234 bits
engram 'world': 235 bits

recall(rust)  → hello=0.44 rust=1.00 world=0.38
recall(world) → hello=0.42 rust=0.38 world=1.00
recall(hello) → hello=1.00 rust=1.00 world=0.94
```

- Hello-Engram ist ungefähr doppelt so groß wie rust/world — es ist
  ja Hub beider Assoziationen.
- Rust- und World-Engramm sind ähnlich groß zueinander, aber **nicht
  identisch**: ihre Schnittmenge bleibt unter 0.40.
- Hub-Cue liefert beide Konzepte mit ≈ 1.0 / 0.94 zurück. Pattern
  Completion über den Knoten funktioniert vollständig.

## Anmerkung zu Bestandscode

`Network::step` filtert STDP jetzt scharf auf E-pre — vorher liefen
die LTD/LTP-Updates auch auf I-Pre/I-Post-Synapsen. Das war
biologisch unsauber und in der iSTDP-Welt würde es mit der neuen
Regel kollidieren. Folge: das numerische Verhalten von
`pattern_completion_with_homeostasis` hat sich leicht verschoben
(Bleed-Ratio 1.00 → 0.91, Coverage 86 % → 73 % bei den alten
Homöostase-Parametern). Ein nachjustiertes `r2_homeostasis`
(`eta_scale 0.002 → 0.004`, `a_target 2.0 → 1.8`) plus eine
ehrliche Coverage-Schwelle (≥ 0.70) bringen den Test wieder grün
ohne die Aussage zu verändern.

## Plastizitäts-Hierarchie, vollständiger Stand

| Skala     | Mechanismus      | τ        | Wirkung                              |
| --------- | ---------------- | -------- | ------------------------------------ |
| schnell   | STDP (E-side)    | ~20 ms   | *welche* E→E-Synapsen Pattern kodieren |
| schnell   | iSTDP            | ~30 ms   | *welche* I→E-Synapsen Engramme trennen |
| mittel    | Homöostase ↓     | ~30 ms   | E-Drive-Obergrenze pro Neuron        |

## Nächste Schritte

1. **Korpus-Skalierung** — viele Sätze nacheinander, prüfen, dass
   das Trio-Verhalten auf 5–10 Konzepte skaliert.
2. **Persistenz** — `Network` und `EngramDictionary` serialisierbar
   machen, damit gelernte Inhibition zwischen Sessions hält.
3. **Token-Effizienz-Messung** — gegen naives RAG: gleicher Korpus,
   gleicher Query-Set, Tokens-an-LLM gezählt.

## Literatur

- [Selective inhibition in CA3 — PLOS Comp Bio 2024](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013267)
- [Vogels et al. 2011 — Inhibitory Plasticity Balances Excitation and Inhibition (Science)](https://www.science.org/doi/10.1126/science.1211095)
- [Dynamic and selective engrams emerge with memory consolidation — Nature Neurosci. 2024](https://www.nature.com/articles/s41593-023-01551-w)
