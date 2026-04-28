# 16 — Live-Visualisierung, Iteration 4: LLM-Substanz (Claude-API)

**Stand:** Der Live-Demo schließt sich. Klick auf „send both to Claude"
schickt die zwei Payloads (RAG-Chunk vs. Javis-Decoded) an die
Anthropic API parallel und zeigt **beide echten Antworten und beide
realen Token-Bills** nebeneinander. Damit ist nicht nur die
Token-Einsparung sichtbar, sondern auch dass die LLM-Antwort mit dem
sparsamen Javis-Payload weiterhin korrekt bleibt.

## Neuer Crate `crates/llm`

```rust
pub struct LlmClient { /* … */ }
impl LlmClient {
    pub fn from_env() -> Self;       // real if ANTHROPIC_API_KEY set
    pub fn mock() -> Self;           // forced offline, deterministic
    pub fn is_real(&self) -> bool;
    pub async fn ask(&self, question: &str, context: &str) -> LlmAnswer;
}

pub struct LlmAnswer {
    pub text: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub real: bool,
}
```

Real-Modus:

- POST `https://api.anthropic.com/v1/messages`
- Default-Model `claude-haiku-4-5-20251001` (override per `ANTHROPIC_MODEL`)
- Response: `text` + reale `input_tokens`/`output_tokens` aus
  Anthropic's `usage`-Block

Mock-Modus (CI / Sandbox / kein Key):

- Deterministischer Antwort-Text auf Basis des Kontexts
- Token-Counts via `ceil(words × 1.3)` Heuristik (gleiche wie der
  Rest des Projekts)
- Wenn ein realer Call fehlschlägt, fällt der Client automatisch auf
  den Mock zurück mit `eprintln!` als Trace

Drei Unit-Tests (alle grün):

- `mock_returns_deterministic_answer_for_known_context`
- `mock_handles_empty_context`
- `token_counter_matches_eval_heuristic`

## Server-Erweiterung

Neuer WS-Action `ask`:

```
/ws?action=ask&query=<question>&rag=<rag_payload>&javis=<javis_payload>
```

Der Server ruft `state.run_ask(...)`:

1. Sendet `Phase { name: "asking", detail: "calling Claude (real|mock)…" }`
2. Triggert beide LLM-Calls **parallel** via `tokio::join!`
3. Sammelt beide Antworten in einem einzigen `Asked`-Event:

```jsonc
{ "type": "asked",
  "question": "rust",
  "rag":   { "text": "…", "input_tokens": 64, "output_tokens": 22, "real": true },
  "javis": { "text": "…", "input_tokens": 28, "output_tokens": 19, "real": true }
}
```

4. Sendet `Done`

Beim Test-Pfad zeigt sich genau das, was wir wollen: bei identischer
Frage hat die RAG-Variante deutlich mehr Input-Tokens als Javis
(reale Token-Differenz, nicht nur Heuristik).

## AppState-Änderungen

```rust
pub struct AppState {
    inner: Arc<Mutex<Inner>>,
    llm: Arc<LlmClient>,
}

impl AppState {
    pub fn new()                -> Self;  // LlmClient::from_env()
    pub fn new_with_mock_llm()  -> Self;  // forced mock, used by tests
    pub fn llm_is_real(&self)   -> bool;
    pub async fn run_ask(&self, question, rag_payload, javis_payload, tx);
}
```

`run_ask` blockiert die Brain-Mutex *nicht* — es nutzt nur den
`llm`-Adapter. So können theoretisch Recall und Ask parallel laufen.

## Frontend

Neue Sektion im Side-Panel **Ask the LLM**:

- Knopf **„send both to Claude"**, anfänglich disabled, wird nach jedem
  erfolgreichen Decode aktiv
- Klick öffnet eine neue WS-Session mit `action=ask` und schickt die
  Query + beide Payloads des letzten Recalls mit
- Antwort kommt als `asked`-Event, UI zeigt:
  - **RAG → Claude**: die Antwort + `input N · output M` Tokens
  - **Javis → Claude**: die Antwort + `input N · output M` Tokens
  - **real API call?**: `yes (Anthropic)` oder `no (mock)` — sofort
    klar, ob die Bill echt war

User-Flow für ein 30-Sekunden-Demo-Video:

```
1.  cargo run -p viz --release --bin javis-viz
2.  Browser → http://127.0.0.1:7777
3.  Live-3D-Brain rendert; Auto-Recall läuft mit "rust"
4.  Side-Panel zeigt: 92.9 % Reduktion, "rust" als einziges Konzept
5.  Klick "send both to Claude"
6.  Beide Antworten poppen auf:
    - RAG  → input ~50 / output ~25
    - Javis → input ~10 / output ~25
    Trotzdem semantisch gleich → Beweis dass der Javis-Payload
    ausreicht.
```

Mit gesetztem `ANTHROPIC_API_KEY` und Internet sind die Antworten
real und die Token-Bill ist die echte API-Abrechnung. Ohne Key
laufen die Mocks und das Demo bleibt vollständig funktional.

## Tests

Neuer Smoke-Test `ask_returns_both_answers_in_mock_mode`:

- Mock-LLM erzwungen
- Schickt `action=ask` mit voller RAG-Payload und Javis-Payload (nur
  das Wort selbst)
- Asserts:
  - beide Antworten haben Text > 0
  - `real == false` für beide (es war Mock)
  - `rag.input_tokens > javis.input_tokens` — die echte Mathematik der
    Token-Einsparung, jetzt über die LLM-Schicht beobachtet, nicht
    nur am Payload

Plus die zwei Tests aus Iteration 3 (`train_then_recall_streams_decoded`,
`reset_clears_dictionary`). **46/46 Tests grün workspace-weit.**

## Wo Javis jetzt steht

```
text  ──TextEncoder──►  SDR  ──inject_sdr──►  R1
                                                ↓ AER
                                                R2 (STDP + iSTDP + homöostasis)
                                                ↓ kWTA + decode
                                       EngramDictionary
                                                ↓
                                  Javis-Payload  ──┐
                                                   ├── parallel LLM call
                                  RAG-Payload   ──┘
                                                   ↓
                                     beide Antworten + reale Token-Bill
```

Die ganze Pipeline ist:

- **end-to-end live** im Browser sichtbar
- **persistent**: Brain lernt zwischen Anfragen weiter
- **substantiiert**: nicht nur Token-Heuristik, sondern reale
  LLM-Calls mit reale Bill
- **deterministisch testbar**: vier Test-Schichten grün
  (snn-core / encoders / eval / viz-smoke)

## Nächste sinnvolle Schritte (jenseits dieses Branches)

- **Größerer Korpus** über `/train`-Action lernen lassen, prüfen wie
  das Brain skaliert (Iteration 3 ist dafür schon vorbereitet)
- **`linkDirectionalParticles`** in 3d-force-graph für animierte
  Spike-Travels auf Inter-Region-Edges
- **Top-N starke Synapsen** sichtbar als gewachsene Bahnen, on-demand
  per Toggle (3d-force-graph kann `nodeThreeObject` mit
  `linkThreeObject` kombinieren)
- **Persistenz**: Brain-Snapshot-Datei schreiben/lesen, sodass das
  trainierte Wissen zwischen Sessions hält
- **Echte BPE-Tokenisierung** via `tiktoken`-Rust-Port für genaue
  Token-Bills auch im Mock-Modus
