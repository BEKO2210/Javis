# 09 — Recall-Decoder: R2-Aktivität → Textkandidaten

**Stand:** Javis kann lesen *und* schreiben. Der Read-Pfad ist
geschlossen.

## Das Mapping-Problem

R1 wird von einem deterministischen Hash adressiert: ein Wort →
fixierte Bit-Indizes. R2 hingegen wächst seine Engramme *emergent* —
welche Neuronen für ein Konzept feuern, hängt vom zufälligen
recurrent-Wiring und der STDP-Verlaufsgeschichte ab. Es gibt keinen
inversen Hash.

Lösung: ein **Engramm-Dictionary**. Vor dem assoziativen Training
präsentieren wir jedes interessante Konzept einzeln, lesen die
R2-Antwort als „Fingerprint" ab und legen sie als
`HashMap<String, Vec<u32>>` ab. Beim Decodieren vergleichen wir das
live feuernde R2-Set mit allen Fingerprints.

## Decoder-Architektur

`crates/encoders/src/decode.rs` — `EngramDictionary`:

```rust
pub fn learn_concept(&mut self, word: &str, active_r2_indices: &[u32]);
pub fn decode(&self, active_r2_indices: &[u32], min_overlap_ratio: f32)
    -> Vec<(String, f32)>;
```

Die Score-Formel ist bewusst asymmetrisch:

```
score(word) = |recall ∩ stored(word)| / |stored(word)|
```

Das ist „**wie viel des gespeicherten Engramms ist im Recall-Set
enthalten**" — exakt das, was Pattern-Completion auslösen sollte.
Symmetrische Maße (Jaccard) hätten das beim Recall typisch viel
größere `recall`-Set gegen das kleine `stored`-Set bestraft.

Implementiert als Linear-Merge-Schnittmenge auf vorsortierten Vektoren:
O(|a| + |b|) pro Vergleich, kein `HashSet`-Overhead. Ergebnis-Liste
ist absteigend nach Score sortiert.

Sieben Unit-Tests für den Dictionary-Code allein:

- leeres Dictionary → leere Kandidaten
- Perfekt-Match → Score 1.0
- 50 % Überlappung → Score 0.5
- `min_overlap_ratio` filtert schwache Matches raus
- Ergebnisse absteigend sortiert
- Duplikate im Input werden gehandhabt
- `learn_concept` überschreibt bei Wiederholung

## Integrationstest `decoder_retrieves_completed_pattern`

Pipeline durch alle Schichten:

| Phase           | STDP | Homöo | Cue           | Output                     |
| --------------- | ---- | ----- | ------------- | -------------------------- |
| Fingerprint     | off  | off   | "hello"       | → `dict["hello"]`          |
| Fingerprint     | off  | off   | "rust"        | → `dict["rust"]`           |
| Training        | on   | on    | "hello rust"  | Engramm bildet sich        |
| Cool-down       | off  | off   | —             | Aktivität klingt ab        |
| Recall          | off  | off   | **"hello"**   | feuernde R2-Neuronen       |
| Decode          | —    | —     | —             | `dict.decode(recall, 0.5)` |

Der Recall-Cue enthält das Wort „rust" gar nicht. Wenn der Decoder
„rust" trotzdem zurückgibt, hat das Netzwerk die Assoziation aus
dem Training assoziativ rekonstruiert.

### Ergebnis

```
fingerprints: hello=149 bits, rust=148 bits
recall set  = 902 bits
candidates  = [("hello", 0.953), ("rust", 0.939)]
```

- `hello` (direkt im Cue) → **95.3 %** der gespeicherten Bits
- `rust` (im Cue **abwesend**) → **93.9 %** der gespeicherten Bits
- Beide weit über der 70-%-Schwelle.

Der Decoder spuckt **„rust" aus, ohne dass es jemals im Recall-Eingang
stand**. Das ist Pattern Completion + Decoding in einem geschlossenen
Lese-Pfad.

## Was sich gegenüber Notiz 08 änderte

- `TRAINING_MS` von 150 auf 250 ms hochgezogen. STDP braucht etwas
  mehr Zeit, damit die assoziative Verknüpfung tief genug ist, um
  in der Recall-Phase die volle „rust"-Repräsentation auszulösen.
  Bei 150 ms war Coverage von „rust" nur 58.8 %.
- Decoder ist eine reine read-only-Schicht — er ändert nichts am SNN.

## Was im Test absichtlich nicht passiert

- Der Test legt das Engramm-Dictionary **vor** dem assoziativen
  Training an. Das ist die saubere Demonstration: Fingerprints sind
  noch unabhängig vom Engramm. Eine produktive Pipeline würde
  Fingerprints fortlaufend aktualisieren oder kontextabhängig
  speichern.
- Der Decoder benutzt nur Containment, nicht Häufigkeit. Wir könnten
  Bits gewichten (TF-IDF-artig), aber das macht erst Sinn, wenn das
  Vokabular in die Tausende geht.

## Die ganze Lese-/Schreib-Pipeline jetzt

```
       ENCODE                          DECODE
       ──────                          ──────
text                                       text-Kandidaten
  │                                          ▲
  ▼  TextEncoder::encode                     │
SDR                                          │  EngramDictionary::decode
  │                                          │
  ▼  inject_sdr(R1, …, drive_nA)             │
R1 Spikes ──► Brain.step ──► R2 Spikes ──────┘
                AER-Routing       (recurrent: STDP + Homöostase)
```

## Nächste Schritte

1. **Mehrere überlappende Sätze** — Engramme orthogonal halten,
   prüfen, dass „hello rust" und „hello python" verschiedene
   `rust`/`python`-Antworten produzieren.
2. **Token-Effizienz-Messung** — der eigentliche End-to-End-Beweis:
   Korpus reinpacken, Query-Set anlegen, Tokens-an-LLM zählen
   gegen naives RAG.
3. **Persistenz** — Engramm-Dictionary + Synapsen-Gewichte
   serialisierbar machen, damit das Wissen zwischen Sessions hält.
