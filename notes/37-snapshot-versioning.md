# 37 — Snapshot-Schema-Versioning

**Stand:** Iteration 19. Vor diesem Patch lag `SNAPSHOT_VERSION = 1`
ohne Migrationspfad — der erste echte Schema-Change hätte alle
existierenden Snapshots unloadable gemacht. Vom user-facing Standpunkt
wäre das ein Daten-Verlust gewesen, vom Operations-Standpunkt ein
Lock-In („wir können nichts mehr ändern, weil wir Production-Snapshots
nicht brechen wollen").

Iteration 19 baut den Migrationspfad ein und bumped das Schema
gleichzeitig auf v2, damit das Framework auch im Code-Pfad
exerciert wird (nicht nur als ungenutztes Scaffolding).

## Schema-Änderung v1 → v2

v2 fügt einen mandatory `metadata`-Block hinzu:

```rust
#[derive(Default, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    pub created_at_unix: u64,    // when this snapshot was first written
    pub javis_version: String,   // build-id of the writing Javis
}
```

Genuine Use-Cases:

- **Ops-Triage**: ein 30 MB-File auf der Platte sagt nichts. Mit
  `created_at_unix` + `javis_version` weiß jemand sofort, *wer*
  das wann geschrieben hat.
- **Compatibility-Audits**: wenn ein Snapshot nach 2 Jahren wieder
  geladen wird, ist `javis_version` der Hinweis, ob der
  Brain-Topology-Code in der Zwischenzeit substantiell verändert
  wurde.

## Migrationspfad

```rust
type MigrationFn = fn(serde_json::Value) -> Result<serde_json::Value, String>;

const MIGRATIONS: &[(u32, MigrationFn)] = &[
    (1, migrate_v1_to_v2),
    // future: (2, migrate_v2_to_v3),
];
```

Loading:

```rust
let value: Value = serde_json::from_slice(&bytes)?;
let from_version = value.get("version")?.as_u64()? as u32;
let value = migrate_snapshot(value, from_version)?;  // walks chain
let snap: Snapshot = serde_json::from_value(value)?; // canonical parse
```

`migrate_snapshot()`:
1. Wenn `from_version > SNAPSHOT_VERSION` → Fehler („cannot
   downgrade"), klare Operator-Botschaft.
2. Sonst: Loop, bei jedem Schritt das passende Element aus
   `MIGRATIONS` finden und anwenden, `version` im JSON inkrementieren,
   bis `current == SNAPSHOT_VERSION`.
3. Returns the migrated `Value`, ready for canonical parse.

`migrate_v1_to_v2()`:
- Injiziert `metadata: { created_at_unix: 0, javis_version: "migrated-from-v1" }`
  in die Top-Level-JSON-Object.
- Damit kann der downstream-Parser das v2-Schema strict serde-en.

## Loading-Logging

`load_from_file` unterscheidet jetzt zwei Cases:

```
INFO snapshot loaded                       # native v2 file
INFO snapshot loaded after schema migration from_version=1 to_version=2
```

Operator weiß sofort, dass ein Migrate stattgefunden hat. Das ist
nützlich beim Container-Restart nach einem Build-Bump: ein einziger
Log-Line bestätigt, dass die alten Snapshots korrekt aufgelesen
wurden.

## Tests

`crates/viz/tests/snapshot_migration.rs`, vier Tests:

1. **`current_version_snapshot_roundtrips`** — schreibe nativen v2,
   lese ihn wieder, sentences/words gleich; prüfe dass `metadata`
   im File ist und nicht leer.

2. **`v1_snapshot_loads_through_migration`** — synthetisiere ein
   v1-File (nimm einen v2, entferne `metadata`, setze
   `version = 1`), lade es, prüfe dass die Brain-Stats korrekt
   geladen sind. Verifiziert das Migrations-Framework
   end-to-end.

3. **`future_version_is_rejected`** — synthetisiere einen Snapshot
   mit `version = 999`, prüfe dass Load mit klarer Error-Message
   scheitert.

4. **`missing_version_field_is_rejected`** — `{"hello": "world"}`
   ist kein Snapshot; Loader weigert sich.

## Künftige Migrations: How-To

Wenn ein v3 ansteht:

1. Bumpe `SNAPSHOT_VERSION` auf 3.
2. Schreibe `fn migrate_v2_to_v3(Value) -> Result<Value, String>` —
   manipuliert die Value so, dass sie als v3 deserialisierbar ist.
3. Ergänze `MIGRATIONS` um `(2, migrate_v2_to_v3)`.
4. Aktualisiere die `Snapshot`-Struktur auf das v3-Layout.
5. Ergänze in den Tests einen synthetisierten v2 → v3 Migration-Run,
   eventuell auch v1 → v3 (Multi-Step) ohne extra Code, weil die
   Chain das automatisch macht.

Punkt 5 ist der eigentliche Wert dieses Frameworks: jede neue
Schema-Version testet automatisch *alle* älteren Versionen, weil
der Chain-Walk die ältesten erst auf v(k-1) hochmigriert.

## Was nicht passiert

- **Kein Downgrade-Path.** v3-snapshots können nicht auf einer
  v2-Build geladen werden. Das ist Absicht: ohne Schema-Übersicht
  weiß ein älteres Build nicht, wie es ein neueres Field
  „rückwärts" interpretieren soll, und Silently-Discard wäre
  schlimmer als ein klarer Fehler.
- **Keine Schema-Compatibility-Tests im CI.** Synthetisierte
  v1-Snapshots leben im Repo nicht. Wenn das mal nötig ist, wären
  `tests/fixtures/v1.snapshot.json` etc. der Ort.
- **Keine Migrations-Backups.** Ein Operator, der vor einem
  Migrate-Run Sorge hat, sollte das File selbst per `cp` sichern.
  Der Server überschreibt es bei Save unbarmherzig.

## Status

- 104/104 Tests grün (100 + 4 Migration-Tests)
- 0 Clippy-Warnings, fmt clean
- Schema-Version 2 lebt; v1-Files werden korrekt aufmigriert
- Framework testet automatisch alle künftigen Versionen
