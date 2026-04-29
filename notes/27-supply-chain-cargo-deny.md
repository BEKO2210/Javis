# 27 — Supply-Chain-Hygiene mit `cargo-deny`

**Stand:** Erster Schritt von Iteration 13 (Supply-Chain). Davor lief
gar keine automatisierte Lizenz- oder CVE-Prüfung; wir haben nur
gehofft, dass keine GPL-Dependency ins Mid-Tree wandert oder ein
RUSTSEC-Advisory uns trifft. Jetzt scheitert CI bei beidem.

## Was `cargo-deny` macht

Vier Checks pro Run, alle in einem CI-Job zusammengefasst:

| Check | Was scheitert |
| --- | --- |
| **advisories** | Bekannte CVE / yanked / unmaintained crates aus der RustSec-DB |
| **licenses** | License nicht auf der allow-Liste |
| **bans** | Wildcard-Versions in Manifests, optional Duplicate-Versions |
| **sources** | Crates aus unbekannter Registry oder Git-Repo |

Konfiguration in der Repo-Root-`deny.toml`. Eine Datei, alle Regeln.

## License-Allowlist

Permissive only — kein Copyleft. Aktuell im Tree gefundene Lizenzen,
genau zehn:

```
MIT, Apache-2.0,
BSD-2-Clause, BSD-3-Clause,
ISC, BSL-1.0,
CDLA-Permissive-2.0, Unicode-3.0,
Unlicense, Zlib
```

Wenn ein neues Dep eine Lizenz reinbringt, die nicht auf der Liste ist
(z. B. MPL-2.0 von einer Mozilla-Crate), schlägt CI fehl. Lösung dann:

1. Alternative finden, die ohne diese Lizenz auskommt, ODER
2. Bewusste Aufnahme in die `allow`-Liste mit Rechtfertigung im
   Commit-Message und ggf. einem `[[licenses.exceptions]]`-Eintrag,
   wenn die License nur für *eine* spezifische Crate akzeptiert wird.

`unused-allowed-license = "deny"` hält die Liste tight — Einträge, die
gar nichts mehr matchen, fallen auf, statt vor sich hin zu rotten.

## Advisories

`version = 2` aktiviert das post-0.16-Schema, das `vulnerability`,
`unmaintained` und `yanked` zu einer Severity-Liste mergt. Default ist
`Low`, also auch leise yanked-Crates sind ein Failure. `ignore = []`
heißt explizit: kein Advisory wird unterdrückt — wenn wir mal eines
ignorieren *müssen*, gibt's einen `{ id = "RUSTSEC-…", reason = "…" }`-
Eintrag mit Begründung.

## Bans

```toml
multiple-versions = "warn"
wildcards = "deny"
allow-wildcard-paths = true
```

- **`wildcards = "deny"`** würde normalerweise unsere `path =
  "../snn-core"` Workspace-Deps als Wildcards flaggen, weil sie keine
  Version-Constraint haben. `allow-wildcard-paths = true` macht das ok
  — aber nur für nicht-publizierte Crates. Damit das greift, hat
  `[workspace.package]` jetzt `publish = false`, und jedes Member-
  Crate hat `publish.workspace = true` in seinem `[package]`.
- **`multiple-versions = "warn"`** statt `deny`: das Rust-Ökosystem
  produziert ständig Duplikate (z. B. `rand 0.8` aus
  `tokio-tungstenite` neben `rand 0.9` aus `metrics-exporter-
  prometheus`). Hartes Verbot wäre ein endloser Kampf gegen
  Upstreams. Stattdessen sehen wir die Warnungen im CI-Log und
  räumen *gezielt* auf, wenn eine Crate für > 30 % des Compile-Time-
  Anstiegs verantwortlich ist.

## Sources

```toml
unknown-registry = "deny"
unknown-git = "deny"
```

Nur crates.io. Kein Random-Git-Dep, kein Custom-Mirror. Wenn ein
unveröffentlichtes Patch wirklich nötig wird, muss der Eintrag
explizit in `allow-git` / `allow-org`. Das macht Supply-Chain-
Compromise-Versuche durch typo-squatted git-URLs unmöglich, ohne dass
der PR-Reviewer es offen sieht.

## CI-Integration

Neuer Job in `.github/workflows/ci.yml`:

```yaml
deny:
  name: cargo-deny (licenses, advisories, bans, sources)
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: EmbarkStudios/cargo-deny-action@v2
      with:
        command: check
        arguments: --workspace --all-features
```

`EmbarkStudios/cargo-deny-action` cached die RustSec-DB zwischen
Runs — kalt < 30 s, warm < 5 s. Vom selben Maintainer wie
cargo-deny selbst.

## Lokal reproduzieren

```sh
cargo install --locked cargo-deny   # einmalig
cargo deny check                    # alle vier checks
cargo deny check licenses           # nur licenses
cargo deny check bans               # nur bans
```

Genau dieselbe `deny.toml`, identisches Verhalten zu CI.

## Status nach Iteration 13a

- `cargo deny check` → exit 0 (4/4 sections ok)
- 9 duplicate-version Warnungen (informational, kein Failure)
- Workspace-`publish = false` als Default; einzelne Crates können das
  später überschreiben, wenn wir auf crates.io publishen wollen.
- 98/98 Tests grün, 0 Clippy-Warnings, fmt clean.

## Was noch kommt (Iter-13 Restplan)

- **B** — MSRV festlegen + CI-Job, der gegen genau diese Toolchain baut
- **C** — Dependabot config (`.github/dependabot.yml`)
- **D** — `cargo doc -D warnings` als CI-Gate
