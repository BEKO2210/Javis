# 18 — Wikipedia-Skalierung: 5 Themen, 96 % Reduktion

**Stand:** Javis hält die Token-Effizienz auf einem realistischeren
Korpus. Mit fünf unabhängigen Wikipedia-Themen verarbeitet das Brain
mehr als doppelt so viel Vokabular wie der ursprüngliche Test, **und
die Token-Reduktion steigt sogar**: von 91 % (3-Themen-Korpus,
Notiz 12) auf 96.6 % im Mittel.

## Vorher / Nachher

| Korpus      | Themen | Mean RAG | Mean Javis | Mean Reduktion |
| ----------- | -----: | -------: | ---------: | -------------: |
| Notiz 12    |   3    |    27    |     2.3    |     91.3 %     |
| Notiz 18    |   5    |    60    |     2.0    |     **96.6 %** |

Warum besser? RAG-Chunks wachsen mit der Korpus-Tiefe (60 statt 27
Tokens pro Treffer), aber der Javis-Output bleibt minimal. Die
Asymmetrie wird ausgeprägter, je realistischer der Korpus wird —
genau das, was wir uns vom Token-Budget eines echten Vector-RAG-
Systems erhofft hatten.

## Korpus

`crates/eval/src/wiki_corpus.rs` enthält fünf kurze Snippets (~50 Wörter
jeweils) destilliert aus realen Wikipedia-Artikeln (CC BY-SA 4.0):

1. **Volcano** — Geologie / Tektonik
2. **Bicycle** — Transport / Mechanik
3. **Coffee** — Pflanze / Getränk / Caffein
4. **Photosynthesis** — Biologie / Chemie
5. **Eiffel Tower** — Architektur / Geschichte

Themen sind absichtlich sehr unterschiedlich. Genau weil sie keinen
gemeinsamen Hub haben (anders als „hello rust" und „hello world" aus
Notiz 10), ist die Engramm-Trennung sehr scharf.

## Pro-Query-Resultate

```
query='volcano'         rag=63  javis=2  saving=96.8%  → [volcano(1.00)]
query='bicycle'         rag=58  javis=2  saving=96.6%  → [bicycle(1.00)]
query='coffee'          rag=63  javis=2  saving=96.8%  → [coffee(1.00)]
query='photosynthesis'  rag=55  javis=2  saving=96.4%  → [photosynthesis(1.00)]
query='eiffel'          rag=60  javis=2  saving=96.7%  → [eiffel(1.00)]

aggregate: mean=96.6%  min=96.4%  max=96.8%
```

Jede Query gibt mit 1.00 Score genau das Query-Wort zurück. Mit der
Containment-Schwelle von 0.50 fällt alles andere weg — extrem sparse,
extrem präzise.

## Tests

Zwei neue Tests in `crates/eval/tests/wiki_benchmark.rs`:

### `javis_scales_to_a_five_topic_wiki_corpus`

Pro-Query-Vertrag:

- RAG findet das Topic
- Query-Wort kommt im Decoded zurück
- Token-Reduktion ≥ 70 %

Aggregat-Vertrag:

- Min ≥ 70 %, Mean ≥ 80 %

Ist-Werte: min 96.4 %, mean 96.6 % — beide weit drüber.

### `engrams_remain_separable_after_five_topic_training`

Beweist, dass Engramme nicht ineinander leaken: für jede der fünf
Topic-Queries prüft der Test, dass im Decoded-Set **kein anderes
Topic-Keyword** auftaucht. Z.B. „volcano" darf nicht „bicycle"
oder „coffee" mit zurückgeben.

Resultat: **null Verletzungen** über alle 5×4 = 20 Cross-Topic-
Vergleiche. Die Kombination von **iSTDP + asymmetrische Homöostase
+ kWTA + Decode-Threshold 0.50** ergibt eine Engramm-Topologie, in
der unabhängige Themen keine Brücken zueinander bauen.

## Refactor-Bonus

`token_efficiency::run_benchmark_on(corpus, query)` ist die neue
parametrisierbare API. `run_benchmark(query)` bleibt als Wrapper für
den 3-Themen-Default-Korpus. Dadurch konnten die 3 alten Benchmark-
Tests (`token_efficiency`) unverändert bleiben.

## Bedeutung

In Zahlen, was das für eine echte LLM-Pipeline bedeutet:

```
Eingabe-Korpus     Naive RAG sendet      Javis sendet      Saving
─────────────────────────────────────────────────────────────────
3 Paragraphen           ~27 Tokens/q       ~2 Tokens/q       91%
5 Wikipedia-Themen      ~60 Tokens/q       ~2 Tokens/q       96.6%
Großer Korpus           Skaliert linear    bleibt minimal    →
```

Die zentrale Beobachtung: **Javis-Output skaliert mit der Anzahl
relevanter Konzepte, nicht mit der Korpus-Größe.** Genau das ist die
versprochene Eigenschaft.

## Status

- 51/51 Tests grün workspace-weit (49 + 2 neue Wiki-Tests)
- 18 Forschungs-Notizen
- Branch `claude/3d-graph-visualization-dpx8s`, alle Commits gepusht

## Was als nächstes spannend wäre

- **Längere Wiki-Artikel** (volle Sektionen statt Intro-Snippets) —
  beweist, dass der Saving-Vorteil mit Korpus-Tiefe weiter wächst
- **Live-HTTP-Loader** mit `reqwest` + Wikipedia-API, optional hinter
  einem Feature-Flag, sodass `cargo test` offline bleibt
- **Cross-Domain-Pattern-Completion** — Recall mit „energy" sollte
  Volcano (thermal energy) und Photosynthesis (chemical energy)
  beide aktivieren, aber mit der aktuellen Decode-Threshold 0.50
  ist nur exakt-match möglich. Ein zweiter, lockerer Decode-Modus
  („associative") würde das öffnen.
- **Persistierte Wiki-Brain-Snapshots im Repo** — `brain-wiki.json`
  als 2 MB-Datei mitliefern, sodass `cargo run -p viz --
  --snapshot brain-wiki.json` sofort mit voll trainiertem Wissen
  startet.
