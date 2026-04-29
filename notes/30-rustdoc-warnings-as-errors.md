# 30 — `cargo doc` als CI-Gate

**Stand:** Teil D von Iteration 13 (Supply-Chain). Damit ist die
Iteration abgeschlossen. Davor konnten Doc-Comments still kaputt
gehen — ein umbenanntes `pub`-Item, ein gestorbener intra-doc Link,
ein verbogenes Codeblock-Attribut. Jetzt scheitert CI bei jedem
einzelnen davon.

## Was rustdoc warnt

`RUSTDOCFLAGS="-D warnings"` bei `cargo doc` hebelt alle rustdoc-
Lints von `warn` auf `deny`. Die wichtigsten:

| Lint | Was er fängt |
| --- | --- |
| `rustdoc::broken_intra_doc_links` | `[Foo]` zeigt auf nichts, weil `Foo` umbenannt/gelöscht wurde |
| `rustdoc::private_intra_doc_links` | Public-API-Doku verlinkt auf private Items |
| `rustdoc::invalid_codeblock_attributes` | ` ```rust,igore ` (typo) statt `ignore` |
| `rustdoc::invalid_html_tags` | `<priv>` als HTML interpretiert, kein bekannter Tag |
| `rustdoc::missing_crate_level_docs` | Crate-Root hat keinen `//!`-Block |
| `rustdoc::redundant_explicit_links` | `[Foo](Foo)` (wenn `[Foo]` reicht) |

Sie sind in `rustc` schon implementiert; CI muss nur sagen
„behandle sie als Fehler".

## CI-Job

```yaml
docs:
  name: cargo doc (zero warnings)
  runs-on: ubuntu-latest
  env:
    RUSTDOCFLAGS: "-D warnings"
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - uses: Swatinem/rust-cache@v2
    - run: cargo doc --workspace --no-deps --all-features
```

`--no-deps` heißt: nur unsere fünf Crates dokumentieren, nicht den
ganzen Tree. Sonst würden wir auch `axum`s und `tokio`s
intra-doc-Links erzwingen, die uns nichts angehen.

## Status: nichts zu fixen

Lokal ausgeführt, vor dem CI-Push:

```sh
$ RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
…
   Generated /home/user/Javis/target/doc/encoders/index.html and 5 other files
```

```sh
$ RUSTDOCFLAGS="-D rustdoc::all" cargo doc --workspace --no-deps --all-features
…
   Generated /home/user/Javis/target/doc/encoders/index.html and 5 other files
```

Beide grün ohne eine einzige Änderung. Heißt: die Doc-Comments, die
über die letzten 29 Iterationen entstanden sind, halten alle ihre
Versprechen. Keine kaputten Links, keine ungültigen Codeblock-
Attribute. Das war eine angenehme Überraschung — bei vielen Repos
landet dieser Job mit zweistelliger Warnungszahl.

## Was nicht aktiviert wurde

- **`missing_docs`** als deny-Lint: würde verlangen, dass jedes
  einzelne `pub fn`, `pub struct`, `pub enum` einen `///`-Block hat.
  Wir haben absichtlich pragmatische Doku — die *interessanten*
  Items (Network, Brain, AppState, run_train) sind dokumentiert,
  die offensichtlichen Helper (kleine Getter, Default-Impls) nicht.
  Voller Sweep wäre viel Mechanik für wenig Wert.
- **`rustdoc::missing_doc_code_examples`**: nur sinnvoll für
  publishbare Crates. Wir sind `publish = false`.
- **Kein `--document-private-items`**: lokale Doku-Generierung mit
  diesem Flag ist hilfreich beim Entwickeln, aber CI würde dann
  auch jede private `fn` mit Lints belegen — Overkill.

## Was Iteration 13 als ganzes geliefert hat

| Teil | Liefergegenstand | Notes |
| ---: | --- | --- |
| A | `cargo-deny` (advisories + licenses + bans + sources) | 27 |
| B | MSRV gepinnt auf Rust 1.86 | 28 |
| C | Dependabot (cargo + github-actions, grouped weekly) | 29 |
| D | `cargo doc -D warnings` als CI-Gate | 30 |

Damit sind alle vier Säulen der Supply-Chain-Hygiene gesetzt:
*Wer sind wir?* (MSRV), *was ziehen wir rein?* (cargo-deny),
*was kommt nach?* (Dependabot), *wie sehen wir es?* (rustdoc).
