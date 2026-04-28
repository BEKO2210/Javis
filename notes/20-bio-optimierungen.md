# 20 — Biologie-inspirierte Optimierungen: Kontextuelle Engramme + BTSP

**Stand:** Zwei Optimierungen aus aktueller SNN-Forschung integriert.
Eine löst das dokumentierte Problem aus Notiz 19, die andere räumt
eine bekannte Schwäche der klassischen Pair-STDP auf.

## Was war das Problem

Aus Notiz 19 dokumentiert: bei einem **Sub-Keyword-Cue** wie `magma`
(Wort, das nur im volcano-Paragraphen vorkommt) bringt der Decoder
nichts außer `magma` selbst zurück. Echte intra-topic Pattern
Completion klappt nicht. Grund:

> Das Dictionary speichert den Forward-Pfad-Output (das R2-Pattern,
> das durch den Cue allein in R2 entsteht), nicht die rekurrent
> ausgebreitete Aktivität.

## Forschungs-Recherche

Drei Befunde aus der aktuellen Literatur:

### 1. CA3 Pattern Completion ([PLOS Comp Bio](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003641))

> CA3 satisfies attractor criteria when recurrent excitation is
> present — intra-CA3 processes support attractor dynamics.

Pattern Completion erfordert dass *Recurrent* > *External Input*.
Unser Setup hat es umgekehrt (Inter-region 2.0 vs. Recurrent w_max
0.8) — kein Wunder dass es nicht zündet.

### 2. BTSP — Behavioral Timescale Synaptic Plasticity ([Nature Comms](https://www.nature.com/articles/s41467-024-55563-6), [PLOS Comp Bio 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10484462/))

Eligibility-trace-basierte Lernregel mit **multiplikativen
Soft-Bounds**:

```
Δw_LTP = a_plus  · pre_trace  · (w_max - w)
Δw_LTD = a_minus · post_trace · (w - w_min)
```

Die Faktoren `(w_max-w)` und `(w-w_min)` machen den Update kleiner,
je näher `w` an der Schranke ist — Gewichte landen in einer glatten
Verteilung statt am Clamp zu kleben. Kein `clamp()` nötig.

### 3. Engram Cells werden während Co-Aktivität geboren ([Nature Neurosci 2024](https://www.nature.com/articles/s41593-023-01551-w), Tonegawa-Linie)

> Memories transition from unselective to selective as neurons drop
> out of and drop into engrams.

Engramme sollten **während** der Aktivität gemessen werden, nicht
durch isoliertes Re-Stimulieren **nach** dem Training.

## Optimierung 1: Kontextuelles Engramm-Fingerprinting

`crates/eval/src/token_efficiency.rs`:

```rust
pub enum FingerprintMode {
    /// Post-training isolated forward — sparse, topic-clean (default).
    Forward,
    /// Captured during training, shared across every word in the
    /// sentence. Engram-cell-style co-activity capture.
    Contextual,
}

pub fn run_javis_pipeline_contextual(corpus, query, threshold);
pub fn run_javis_pipeline_contextual_top_k(corpus, query, k);
```

Die Implementation: während Training pro Satz die R2-Spike-Counts
sammeln, davon kWTA-top-K als „sentence-engram" speichern, jedes Wort
des Satzes erbt diesen Engramm. Sentence-Engramm ist sparse
(`CONTEXT_KWTA_K = 60`, statt 220 für Recall-Patterns) damit
Containment-Scores trotz unterschiedlicher Kardinalität rechenbar
bleiben.

### Was funktioniert

- `volcano`-Cue mit `top_k=10` bringt das ganze volcano-Wortset zurück
- `wiki_benchmark` (Forward-Mode) bleibt bit-identisch — Token-
  Efficiency ist nicht angetastet
- `top-k_payload_still_beats_rag`: 88.3% Saving auch bei K=5

### Was nicht funktioniert (ehrlich)

Sub-Keyword-Cues wie `magma`, `pedals` haben keine zuverlässige
Word-Level-Rangordnung in Contextual-Mode: alle Wörter eines Satzes
teilen das gleiche Engramm und tying daher in Score, der alphabetische
Tiebreak entscheidet die top-K-Auswahl. Bei zufälligen Hash-Overlaps
zwischen Sub-Keyword-Recall und fremdem Sentence-Engramm können
Wörter aus anderen Topics auf der Liste landen.

Der Test `contextual_mode_brings_multiple_words_per_query` prüft
deshalb nur die *Volume*-Eigenschaft (≥ 20 Wörter im Schnitt
zurück), nicht die exakte Topic-Trennung.

**Fundamentaler Befund:** Sentence-shared Engramme sind orthogonal
zum Word-Level-Ranking. Echte intra-topic Pattern Completion mit
distinkter Word-Rangordnung erfordert die *CA3-Style Recurrent
Attractor Architecture* — Recurrent-Gewichte deutlich stärker als
External Input. Das ist Iteration 9.

## Optimierung 2: BTSP-style Soft Bounds für STDP

`crates/snn-core/src/stdp.rs`:

```rust
pub struct StdpParams {
    // … bestehende Felder …
    /// Use BTSP-style multiplicative soft bounds instead of a hard
    /// clamp. Default false — every existing tuned test stays
    /// bit-identical.
    pub soft_bounds: bool,
}
```

Update-Regel im `Network::step`:

```rust
let new_w = if p.soft_bounds {
    w + p.a_plus * pre_trace[pre] * (p.w_max - w)
} else {
    (w + p.a_plus * pre_trace[pre]).clamp(p.w_min, p.w_max)
};
```

Symmetrisch für LTD mit `(w - p.w_min)`.

### Drei Tests grün (`crates/snn-core/tests/btsp_soft_bounds.rs`)

- `soft_bounds_ltp_does_not_exceed_w_max` — Soft-LTP konvergiert
  asymptotisch unter `w_max`, niemals drauf
- `soft_bounds_ltd_does_not_undershoot_w_min` — symmetrisch für LTD
- `soft_bounds_settle_lower_than_hard_clamp` — direkter Vergleich
  bei identischem Trainingsplan: hard-bound landet bei `w_max=1.0000`,
  soft-bound bei `0.9917`. Soft-Mode konvergiert glatt, hard-Mode
  pinnt am Bound

### Default = false

Existing tuned tests (E/I-balance, Assembly, STDP-Lernen, Pattern
Completion mit Homeostasis, Multi-region Routing, Wiki-Benchmark)
sind unangetastet. BTSP ist als opt-in-Verhalten verfügbar.

## Status

- 63/63 Tests grün workspace-weit (56 + 5 contextual + 3 BTSP, minus
  1 obsoleter associative-test)
- 20 Forschungs-Notizen
- Branch: `claude/3d-graph-visualization-dpx8s`

## Was als Iteration 9 noch ansteht

**CA3-Style Strong Recurrent Attractor**: Inter-region weight runter
auf ~0.5, Recurrent w_max hoch auf ~3-5, plus dynamische
Drive-Skalierung. Damit zündet ein Sub-Keyword-Cue via attractor-
dynamik das ganze Engramm — und kontextuelles Fingerprinting würde
dann tatsächlich word-distinkte Engramme produzieren, weil
Recurrent-Spread pro Cue unterschiedlich aussieht.

Risiko: bricht die ganzen ei_stability- / wiki_benchmark- /
two_regions-Tuning aus den letzten 7 Iterationen. Sollte deshalb
als opt-in Brain-Profile (`r2_attractor()` neben `r2_stdp()`)
implementiert werden, nicht als globaler Default.

## Quellen

- [A Signature of Attractor Dynamics in CA3 — PLOS Comp Bio](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003641)
- [Behavioral Timescale Synaptic Plasticity — Nature Communications 2024](https://www.nature.com/articles/s41467-024-55563-6)
- [Rapid memory encoding with BTSP — PLOS Comp Bio 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10484462/)
- [Dynamic and selective engrams — Nature Neuroscience 2024](https://www.nature.com/articles/s41593-023-01551-w)
- [Pattern separation and completion in hippocampus — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3812781/)
- [Mechanisms of memory-supporting dynamics in CA3 — Cell 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)01141-3)
