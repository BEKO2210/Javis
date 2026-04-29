# 28 — MSRV festgesetzt auf Rust 1.86

**Stand:** Teil B von Iteration 13 (Supply-Chain). Ohne MSRV-Pin
konnte jede neue stabile Rust-Version unsere Build-Anforderungen
versehentlich nach oben drücken. Jetzt ist `rust-version = "1.86"`
ein Vertrag — neue Code-Features in unserem eigenen Workspace, die
1.86 nicht kennt, scheitern in CI bevor sie gemerged werden.

## Wie wir auf 1.86 gekommen sind

Empirische Messung gegen unseren tatsächlichen Dep-Tree, nicht durch
Raten. Reihenfolge:

| Toolchain | Ergebnis |
| --- | --- |
| 1.82 | ❌ `indexmap-2.14` braucht `edition2024` (stabilisiert in 1.85) |
| 1.85 | ❌ `icu_properties_data-2.2` braucht `rustc 1.86` |
| 1.86 | ✅ alles compiles, 98/98 tests grün |

Damit ist 1.86 die untere Grenze, ohne dass wir
`cargo update --precise <ver>`-Pins für transitive Deps schreiben
müssten. Höher zu gehen wäre Selbstbeschneidung, niedriger zu gehen
ist nicht möglich ohne Dep-Surgery.

## Eine Code-Änderung musste passieren

`snn-core/src/network.rs:434` benutzte `u64::is_multiple_of(...)`.
Das ist erst seit 1.87 stabil. Ersetzt durch das idiomatische
Äquivalent `% != 0`:

```rust
// vorher (1.87+)
if self.step_counter.is_multiple_of(h.apply_every as u64) { … }

// nachher (1.86-kompatibel)
if self.step_counter % (h.apply_every as u64) == 0 { … }
```

Semantisch identisch, ein Token kürzer ohne den Method-Call. Hätte
nicht passieren dürfen, aber Rust 1.94 (unser Daily-Driver) hat das
neue Feature zugelassen ohne dass `rust-version` es noch zurückgehalten
hätte — der einzige Grund, MSRV überhaupt formal festzulegen.

## Manifest-Mechanik

`[workspace.package]`:

```toml
rust-version = "1.86"
```

Damit das auch wirklich in jedes Member-Crate erbt, muss jede
`[package]`-Section explizit `rust-version.workspace = true` haben.
Cargo macht das nicht automatisch — vergisst man es, hat das Crate
keinen MSRV-Pin und der CI-Job schlägt erst auf, wenn jemand 1.85-
Code schreibt. Alle fünf Member-Crates haben es jetzt.

## CI-Job

```yaml
msrv:
  name: msrv (1.86)
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@1.86
    - uses: Swatinem/rust-cache@v2
    - run: cargo build --workspace --locked
    - run: cargo test --workspace --locked -- --test-threads=2
```

`dtolnay/rust-toolchain@1.86` pinpointet exakt diese Version. Kein
`stable`, kein Drift. `--locked` heißt: genau die `Cargo.lock`-
Versionen, kein Resolver-Update bei jedem Run.

## Was nicht passiert ist

- **Kein `rust-toolchain.toml`-File.** Das würde `cargo` zwingen,
  immer 1.86 zu benutzen, auch lokal. Das ist Overkill für ein
  Repo, in dem die meiste Entwicklung mit dem aktuellen Stable
  läuft. Der MSRV-Job in CI reicht als Gate.
- **Kein automatisches Bump-on-stable-bump.** Wenn Rust 1.95 raus
  kommt, ändert sich an unserem MSRV-Pin nichts. Bumps sind
  expliziter Akt mit CHANGELOG-Eintrag und einer Begründung
  („Crate X braucht 1.NN", oder „wir wollen Feature Y benutzen").
- **Kein „supports older Rust"-Werbespruch.** 1.86 ist nicht alt;
  es ist die jüngste Version, mit der unser Stack noch baut. Wer
  ein „echtes" altes MSRV will (1.74, 1.75), müsste die Hälfte
  der Deps austauschen.

## Status

- `rust-version = "1.86"` workspace-weit
- Alle 5 Member-Crates haben `rust-version.workspace = true`
- `cargo +1.86 build --workspace --locked` ✅
- `cargo +1.86 test --workspace --locked` → 98/98 ✅
- `cargo +stable test --workspace` → 98/98 ✅ (keine Regression)
- `cargo deny check` → exit 0 (kein Fallout durch Manifest-Änderungen)
