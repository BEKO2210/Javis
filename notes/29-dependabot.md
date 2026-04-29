# 29 — Dependabot

**Stand:** Teil C von Iteration 13 (Supply-Chain). Mit `cargo-deny`
weiß CI, ob unsere aktuellen Deps sauber sind. Ohne Dependabot weiß
niemand automatisch, *dass* eine neuere Version existiert. Jetzt
geht jeden Montagmorgen ein Bot durch und macht PRs.

## Zwei Ecosystems

`.github/dependabot.yml` trackt zwei Supply-Chain-Quellen:

| Ecosystem | Was wird beobachtet |
| --- | --- |
| `cargo` | Jede `Cargo.toml` im Workspace (Root + crates/*) |
| `github-actions` | Jeder `uses: foo@vN`-Eintrag in `.github/workflows/*` |

Beide laufen `weekly` (Mo, 06:00 Europe/Berlin) — eine Welle Updates
pro Woche statt täglich Lärm.

## Grouped Updates

Default-Verhalten von Dependabot ist *eine PR pro Update*. Bei einem
Workspace mit ~80 transitiven Crates wären das schnell 5-10 PRs pro
Woche. Wir gruppieren stattdessen:

```yaml
groups:
  cargo-minor-patch:
    update-types: [minor, patch]
  tracing:
    patterns: ["tracing", "tracing-*"]
  tokio-stack:
    patterns: ["tokio", "tokio-*", "tower", "tower-*", "axum", "axum-*", "hyper", "hyper-*"]
```

Drei Konsequenzen:

1. **Patch + Minor in einer PR.** Pro Woche maximal eine „house
   keeping"-PR mit allen kleinen Bumps zusammen. Schneller zu
   reviewen, schneller zu mergen.
2. **Stack-Bündel.** Wenn `tracing-core` bumpt, kommt
   `tracing-subscriber` meist mit; jetzt landen sie in einer PR
   statt zwei. Gleiches für tokio/tower/axum/hyper, die oft
   co-released werden.
3. **Major-Bumps bleiben solo.** Breaking changes wollen
   individuelles Review + CHANGELOG-Eintrag. Gruppen-Logik
   filtert sie explizit raus (`update-types` enthält kein
   `major`).

## Limits

```yaml
open-pull-requests-limit: 5    # cargo
open-pull-requests-limit: 3    # github-actions
```

Falls jemand 2 Wochen lang nicht mergt: maximal 5 offene cargo-PRs,
3 actions-PRs. Backpressure statt Flood.

## Commit-Messages

```yaml
commit-message:
  prefix: "deps"      # cargo
  prefix: "ci"        # github-actions
  include: scope
```

PRs landen als `deps(cargo): bump foo from 0.5 to 0.6` bzw.
`ci(actions): bump actions/checkout from 4.1 to 4.2`. Direkt
filterbar in `git log --grep '^deps'`.

## Was nicht gemacht wurde

- **Keine `auto-merge`-Konfiguration.** Dependabot kann automatisch
  mergen wenn alle Checks grün sind, aber das ist gefährlich für ein
  Projekt mit echtem Production-Anspruch — supply-chain attacks
  durchkriechen sonst ungesehen. Wir wollen, dass jemand die PR
  *anschaut*, bevor sie ins main fließt.
- **Kein `security-updates: only`.** Würde Dependabot zwingen, nur
  bei tatsächlichen RustSec-Advisories zu reagieren. Klingt
  aufgeräumt, lässt aber die normalen Maintenance-Updates liegen
  — und die sind oft, was Advisories *vermeidet*.
- **Kein `assignees` / `reviewers`.** Solo-Projekt; PRs landen in
  der Inbox, der Owner reviewt sie. Bei Team-Setup würde hier eine
  Liste stehen.
- **Keine `ignore`-Regel für Major-Bumps von z. B. axum.** Wir
  *wollen* gesagt bekommen, dass axum 0.8 raus ist. Wir mergen es
  nur nicht ohne Tests-Migration.

## Kein Code-Test

Dependabot-Configs sind deklarativ. Es gibt keinen sinnvollen
Unit-Test — die einzige Verifikation ist YAML-Parsability
(`python3 -c "import yaml; yaml.safe_load(open('.github/dependabot.yml'))"`)
und der Service-Status auf GitHub: nach dem Merge sollte unter
*Insights → Dependency graph → Dependabot* der grüne „active"-Status
auftauchen, und Mo morgens kommt die erste PR.

## Status

- `.github/dependabot.yml` erstellt, YAML-valide
- 2 Ecosystems × wöchentlicher Lauf, mit Grouping
- 98/98 Tests grün (unverändert), `cargo deny check` grün, fmt clean
