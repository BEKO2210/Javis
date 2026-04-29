# 23 — Production Polish (CI, doc-tests, examples)

**Stand:** Funktional ist Javis seit Iteration 10 fertig — 96.7 %
Token-Reduktion vs naiver RAG, 91 grüne Tests, null Clippy-Warnings.
Iteration 11 hat keine neuen Features mehr eingeführt; sie bringt das
Repo nur in einen Zustand, in dem es ohne Erklärung *anderer Leute*
laufen würde, und in dem versehentliche Regressionen automatisch
bemerkt werden.

## 1. GitHub Actions CI

Vier Jobs in `.github/workflows/ci.yml`, alle parallel pro Push und PR:

| Job | Befehl | Zweck |
| --- | --- | --- |
| `fmt` | `cargo fmt --all -- --check` | Stylebreak fängt zur build-time, nicht im Review |
| `clippy` | `cargo clippy --all-targets -- -D warnings` | Jede neue Warnung scheitert den PR |
| `test` | `cargo test --release --workspace -- --test-threads=2` | Vollständige Test-Suite gegen Optimization-Code |
| `doc-test` | `cargo test --doc --workspace` | Doku im Code wird zur ausführbaren Spezifikation |

`--test-threads=2` ist gesetzt, weil ein paar Brain-Tests
deterministische Drive-RNGs benutzen und Cross-Test-Parallelism nicht
mögen. Lokal genauso.

Der Workflow zieht keine externen Actions außer
`actions/checkout@v4` und `dtolnay/rust-toolchain@stable` — keine
Lieferketten-Risiken durch nichtoffizielle Steps.

## 2. Doc-Tests auf der öffentlichen API

Drei Snippets, die kompiliert *und* ausgeführt werden:

```rust
//! crates/snn-core/src/lib.rs
//! ```
//! use snn_core::{Network, LifNeuron, LifParams};
//! let mut net = Network::new(0.1);
//! let a = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
//! let b = net.add_neuron(LifNeuron::excitatory(LifParams::default()));
//! net.connect(a, b, 1.0);
//! assert_eq!(net.synapses.len(), 1);
//! ```
```

```rust
//! crates/encoders/src/lib.rs (zweimal: forward & decode_top)
```

Diese Tests laufen automatisch im CI-Job `doc-test`. Wenn die API
gebrochen wird, scheitert der Build sofort — die README- und
Iterations-Notes-Snippets bleiben gültig.

## 3. End-to-End-Beispiel

`crates/eval/examples/hello_javis.rs` — das, was man als „Erstkontakt"
zeigen will:

```sh
$ cargo run --release -p eval --example hello_javis
=== Javis vs naive RAG (5 Wikipedia topics) ===

  rust       :  RAG  82 tok    Javis  2 tok   savings 97.6%
  ferris     :  RAG  61 tok    Javis  2 tok   savings 96.7%
  …
  Total: 299 tokens RAG → 10 tokens Javis = 96.7 % saving
```

Keine externen Dependencies nötig (das Beispiel benutzt nur die
bestehenden eval-Datasets). Damit gibt es einen einzigen Befehl, der
beweist *„dieses Repo macht, was die README sagt"* — ohne Browser,
WebSocket, oder Anthropic-Key.

## 4. CHANGELOG.md

Loosely Keep-a-Changelog. Iterations 8–10 sind explizit dokumentiert
mit `Added` / `Changed` / `Removed` / `Result`-Sektionen, frühere
Iterationen als Übersichtstabelle mit Notes-Verweisen.

Wichtig: Jede `## Iteration N`-Section verlinkt auf die korrespondiere
`notes/NN-*.md` für die vollständige Begründung. Die CHANGELOG ist
*Index*, nicht *Replikat*.

## 5. README-Updates

- CI-Badge aus dem neuen Workflow oben in der README.
- Test-Statistik auf 91/91 inklusive Doc-Tests-Zeile aktualisiert.
- Notes-Index erweitert um Einträge 21 und 22.
- Quick-Start-Sektion bekommt den `cargo run --example hello_javis`-
  Befehl als ersten Schritt — vor Browser-Demo und vor LLM-Demo.

## Was Iteration 11 explizit *nicht* macht

- **Keine semver-stable API.** `pub fn` heißt momentan „export für
  Tests / Beispiele", nicht „garantierte Schnittstelle für externe
  Crates". Vor einem `0.2`-Release müsste eine Audit-Runde her, was
  wirklich öffentlich bleiben soll.
- **Kein Release auf crates.io.** Ohne LICENSE-Klärung kein Push.
- **Keine MSRV-Garantie.** CI läuft auf `stable`. Das reicht für ein
  Hobby-Repo; ein Library-Konsument bräuchte eine fixe MSRV.
- **Keine Performance-Benchmarks im CI.** Die `eval`-Tests prüfen
  Korrektheit, nicht Geschwindigkeit. `cargo bench` existiert (noch)
  nicht — könnte ein eigener Schritt werden.

## Workspace-Status nach Iteration 11

```
$ cargo fmt --all -- --check       # 0 diff
$ cargo clippy --all-targets       # 0 warnings
$ cargo test --release --workspace # 92 passed; 0 failed
                                   # (89 unit/integration + 3 doc-tests)
```

Das ist der Zustand, in dem ein neuer Mitwirkender `git clone &&
cargo test` macht und sofort sieht: *läuft, ist sauber, Doku stimmt*.
Genau das war das Ziel.
