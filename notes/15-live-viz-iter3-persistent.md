# 15 — Live-Visualisierung, Iteration 3: persistentes Brain + Live-Training

**Stand:** Der Server hält jetzt **eine** Brain-Instanz, die zwischen
Anfragen lebt. Der User kann live Sätze trainieren und danach mit dem
gelernten Wissen sprechen — ohne Reload, ohne Re-Init.

## Was Iteration 3 ändert

### Neuer `AppState` (`crates/viz/src/state.rs`)

Eine `Arc<Mutex<Inner>>`-Struktur, in der das Brain für die Lebensdauer
des Servers persistiert. `Inner` hält:

- `brain: Brain` — das SNN, einmal beim Server-Start aufgebaut
- `dict: EngramDictionary` — wächst mit jedem neuen Wort, das trainiert wird
- `encoder: TextEncoder` — fixer Stoppwort-Encoder
- `known_words: HashSet<String>` — Vokabular-Tracking
- `trained_sentences: Vec<String>` — RAG-Lookup-Quelle (siehe unten)

Drei öffentliche Operationen:

```rust
state.run_train(sentence, Some(tx)).await;  // STDP+iSTDP+Homöo + neue Wörter ins Dict
state.run_recall(query, tx).await;          // freezes Plastizität, Decode + RAG-Vergleich
state.reset().await;                        // ganzes Brain wieder leer
```

`run_train` macht in einer Transaktion:
1. Engagiert STDP, iSTDP, Homöostase
2. Encodiert den Satz, fährt den Cue 150 ms lang durchs Brain
3. Cool-down 50 ms
4. Friert Plastizität wieder ein
5. Identifiziert neue Wörter, fingerprinted jedes als kWTA-Top-220
6. Pusht den Satz in `trained_sentences`

`run_recall` ist plastizitäts-frei (Messung darf Gewichte nicht
ändern), schickt Spike-Stream über das WS, decoded gegen das **aktuelle**
Dictionary, und der RAG-Vergleich sucht in `trained_sentences` —
also die gleiche Wissensbasis, die das Brain selbst hat.

### Neue Server-Routes (`server.rs`)

Eine einzige Route `/ws` mit Action-Param:

```
/ws?action=recall&query=rust
/ws?action=train&text=Cats are mammals.
/ws?action=reset
```

Alle drei streamen Events über den selben Kanal — das Frontend muss
nicht zwischen Endpoints unterscheiden.

### Auto-Bootstrap im Binary (`main.rs`)

Beim Server-Start wird der Default-Korpus (Rust / Python / Cpp) im
Brain trainiert. Print-Line `ready (3 sentences, ~30 concepts)` zeigt
den Stand. Damit hat der allererste Recall sofort etwas zu finden, der
User kann dann beliebig draufeinander trainieren.

### Frontend

Zwei Forms im Header:

- **train** (breiter Input) — „teach the brain a sentence…", Button
  *Learn*. Schickt den Text als Trainings-Cue.
- **recall** — Wort-Input, Button *Ask* (Akzent-Farbe). Schickt als
  Recall-Query.
- **reset brain** — wischt das ganze Wissen.

Der 3D-Brain bleibt während Training und Recall sichtbar, beobachtet
denselben Brain-State. Spike-Animationen aus Iteration 2 sind
unverändert.

## Wie sich das anfühlt im Live-Demo

```
1.  Page lädt → 3D-Brain rendert
2.  Auto-Bootstrap-Korpus ist beim Server-Start schon einmal durchs
    Brain gelaufen, also sieht man kurz die letzte Bootstrap-Phase.
3.  User tippt im Recall-Feld "rust" → Spikes wandern von R1 nach R2,
    Side-Panel zeigt 92 % Token-Saving und "rust" als Konzept
4.  User tippt im Train-Feld "Cats are mammals" → 1500 ms Animation:
    erst Cue, dann Cool-down, dann Fingerprint-Phase. Phase-Anzeige
    sagt "ready — 4 sentences, 32 concepts learnt"
5.  User tippt im Recall-Feld "cats" → das Brain antwortet mit
    "cats mammals", obwohl es das Wort nie zuvor gesehen hatte
    (trainiert nur in diesem Browser-Tab)
```

Das ist die Sequenz für ein 30-Sekunden-Video.

## Smoke-Tests

`tests/smoke.rs` hat jetzt zwei Tests:

- **`train_then_recall_streams_decoded`** — bringt den Server live hoch,
  trainiert einen Satz über `/ws?action=train`, danach Recall über
  `/ws?action=recall`. Asserts:
  - das gefragte Wort taucht in den Decoded-Kandidaten auf
  - die Token-Reduction über die Leitung ≥ 70 %
- **`reset_clears_dictionary`** — trainiert einen Satz, prüft Stats,
  triggert `/ws?action=reset`, prüft, dass `state.stats()` 0 / 0
  zurückgibt.

Plus alle 40 alten Tests aus snn-core / encoders / eval. **42/42
grün workspace-weit.**

## Was Iteration 4 noch bringen wird

- **`crates/llm`** mit Claude-API-Adapter
- **„Send to LLM"-Knopf** im Decoded-Panel: schickt RAG-Payload und
  Javis-Payload parallel zur API, zeigt beide Antworten + reale
  Token-Bill nebeneinander
- Damit ist nicht nur die *Token-Einsparung* sichtbar, sondern auch
  dass die LLM-Antwort mit dem Javis-Payload tatsächlich brauchbar
  bleibt — der Beweis, dass die 92 % Einsparung nicht „auf Kosten der
  Antwortqualität" geht
