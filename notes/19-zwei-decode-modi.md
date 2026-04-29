# 19 — Zwei Decode-Modi: strict (sparse) + top-k (associative)

**Stand:** Der Decoder hat zwei Lese-Pfade auf demselben trainierten
Brain. *Strict* (Threshold) bleibt sparse und topic-rein — die
Token-Effizienz aus Notiz 12/18. *Top-k* gibt immer die K
relevantesten Engramme zurück, auch wenn die absoluten Scores
niedriger sind — füttert Auto-Suggest- und „related concepts"-Panels.

## Neue API

```rust
// crates/encoders/src/decode.rs
impl EngramDictionary {
    pub fn decode(&self, active: &[u32], min_overlap_ratio: f32)
        -> Vec<(String, f32)>;
    pub fn decode_top(&self, active: &[u32], k: usize)
        -> Vec<(String, f32)>;  // ← neu
}

// crates/eval/src/token_efficiency.rs
pub fn run_javis_pipeline_with_threshold(corpus, query, threshold);
pub fn run_javis_pipeline_top_k(corpus, query, k);  // ← neu
```

`decode_top` ist die robustere Wahl, wenn der richtige Threshold pro
Query unterschiedlich ist (was bei realistischem Korpus oft der
Fall ist):

- ein Hauptkeyword wie `volcano` macht ein dichtes Engramm — bei
  threshold 0.30 kommen so viele Wörter rein, dass auch
  Cross-Topic-Wörter über die Schwelle rutschen
- ein Nebenwort wie `magma` macht ein viel kleineres Engramm —
  bei demselben 0.30 kommt nichts außer dem Wort selbst zurück

`decode_top(active, 5)` gibt einfach die fünf besten Treffer
zurück, sortiert nach Score, ohne Threshold-Rätsel.

Refactor-Bonus: die Pipeline-Logik aus `run_javis_pipeline_*` ist
jetzt in `run_javis_recall_inner` extrahiert (privat). Beide
öffentlichen Decode-Wrapper sind drei Zeilen lang, kein Duplikat.

## Vier neue Tests in `eval/tests/associative_recall.rs`

### `top_k_top_one_is_always_the_cue`
Top-1 ist garantiert das Query-Wort selbst (Score 1.0 by
construction). Scores sind monoton fallend. Pinned die Sortier-
Reihenfolge.

### `top_k_payload_still_beats_rag`
Mit K=5 über alle fünf Wiki-Queries:

```
top-5 associative recall: rag=299  javis=35  saving=88.3%
```

Lockerer als die strenge 0.50-Schwelle (96.6 %), aber immer noch ein
deutlicher Gewinn gegen RAG. Das ist der Trade-off zwischen
Sparsamkeit und Detailtiefe.

### `strict_threshold_stays_topic_clean`
Bei threshold 0.50 ist *garantiert* kein Wort aus einem fremden
Topic im Decoded — über alle 5 Queries hinweg verifiziert. Das ist
der Vertrag, auf dem `wiki_benchmark::engrams_remain_separable`
aufbaut.

### `high_threshold_keeps_recall_minimal`
Default-Threshold bleibt bei ≤ 3 Wörtern pro Query. Regression-Guard
für die Token-Efficiency-Story.

Plus zwei neue Unit-Tests in `encoders/src/decode.rs`:
- `decode_top_returns_k_best_results` — Top-K-Verhalten
- `decode_top_breaks_ties_alphabetically` — Determinismus bei
  gleichem Score

## Ehrliche Beobachtung

Reines „intra-topic associative recall" — Cue ist ein Nebenwort und
das System ergänzt den Rest des Satzes — funktioniert **nicht**
zuverlässig mit der aktuellen Architektur:

- Die Engramme im Dictionary speichern den Forward-Pfad-Output (das
  R2-Pattern, das durch den Cue allein in R2 entsteht), nicht die
  rekurrent ausgebreitete Aktivität
- Bei einem Hauptkeyword ist der Engram dicht und überlappt mit
  vielen Co-Wörtern desselben Satzes
- Bei einem Nebenwort ist der Engram sparse und überlappt nur mit
  sich selbst

Top-k ist die pragmatische Antwort: wenn man immer K Treffer haben
will (z.B. für eine UI), funktioniert es. Wenn man wirklich
„intra-topic completion" will, brauchen wir Iteration 8 — Engramme
nach Training fingerprinten oder die rekurrente Spread-Antwort
explizit speichern.

## Status

- 56/56 Tests grün workspace-weit (51 + 4 associative recall + 2
  encoder-Unit-Tests, minus 1 weil der bestehende
  `wiki_benchmark` schon den Topic-Clean-Test enthielt — netto +5)
- 19 Forschungs-Notizen
- Branch unverändert: `claude/3d-graph-visualization-dpx8s`

## Was als nächstes spannend wäre

- **Engram-Fingerprints nach Training**, nicht aus Forward-Pfad —
  würde echte assoziative Recall ermöglichen. Konsequenz: das
  Dictionary wäre nicht mehr ein-zu-eins mit Forward-SDR, müsste
  per Wort separat trainiert werden
- **Frontend-Toggle „strict ↔ associative"** — UI-Schalter im
  Side-Panel, der zwischen den beiden Decode-Modi wechselt; im
  3D-Brain könnte man dabei sehen wie sich das Recall-Set
  vergrößert / verkleinert
- **Persistierter Wiki-Brain-Snapshot im Repo** — `brain-wiki.json`
  mitliefern, sodass `cargo run -p viz -- --snapshot brain-wiki.json`
  sofort mit voll trainiertem Wissen startet
