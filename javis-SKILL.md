---
name: javis
description: >
  Forschungsdisziplin- und Experiment-Orchestrierungsskill für das Javis-Projekt (iteratives KI/SNN/Memory-Forschungsprojekt).
  IMMER verwenden wenn der User folgende Begriffe nutzt oder entsprechende Situationen auftreten:
  "Javis", "iter-XX", "neue Iteration", "Smoke", "Full Run", "Verdict", "Acceptance Matrix",
  "Pre-Registration", "Gate", "Branch A/B/C", "Goalpost", "Seed", "8-seed", "DG", "R2", "C1", "CA1",
  "BTSP", "Cue→Target", "target_top3_overlap", "c1_target_top3_overlap", "Recall-mode",
  "Pattern Separation", "Pattern Completion", "Claude Code Prompt für Javis",
  "Was soll ich Claude antworten?", "Wie geht es weiter?", "Soll ich abbrechen oder warten?",
  "Deep Research für Javis", "PR-Reihenfolge", "Branch ist stale", "Commit gepusht",
  "Run läuft", "Seed Ergebnis", "Metrik FAIL/PASS", "iter-XX abgeschlossen".
  Auch verwenden wenn Logs, Tabellen oder Zwischenergebnisse aus Claude Code / Codex gepostet werden
  und der User fragt was es bedeutet oder was er antworten soll.
  Dieser Skill verhindert Kontestverlust, Goalpost-Shifting, Wiederholung falsifizierter Pfade
  und übermäßig optimistische Ergebnisinterpretation.
---

# Javis Skill — Forschungsdisziplin & Experiment-Orchestrierung

## Kernregeln (immer aktiv)

1. **Kein Goalpost-Shift.** Keine Threshold-, Metrik- oder Architekturänderung nach einem Ergebnis. Erst den gelockten Run fertig auswerten.
2. **Negative Ergebnisse akzeptieren.** Kein „aber Trend"-Argument. Kein Reframing.
3. **Smoke vor Full Run.** Full Run startet nur nach bestandenem Smoke-Gate — technisch UND fachlich.
4. **Metriken vor Run locken.** Acceptance Matrix muss vor dem Run stehen, nicht danach.
5. **Per-Seed vor Aggregat.** Wenn Heterogenität sichtbar ist, zählt per-seed. 4/8 positiv = chance-level, kein robuster Mechanismus.
6. **Bit-exakte Reproduktion** als starkes Signal markieren.
7. **Ungültige Runs invalidieren** — nicht interpretieren.
8. **Architektur-Pivot:** Erst Diagnose/ENTRY, dann Code.
9. **Kein Hype.** Keine „Breakthrough"-Sprache außer Zahlen rechtfertigen sie.

---

## Workflow pro Iteration

```
1. Kanonischen Stand lesen (main, offene PRs, letzte Verdicts)
2. Hypothese formulieren (eine klare Frage)
3. Acceptance Matrix locken (Metriken, Thresholds, Seeds, Run Command)
4. Pre-Registration Gate: alles schriftlich vor dem Run
5. Smoke-Run → Gate prüfen (technisch + fachlich)
6. Full Run nur bei PASS
7. Ergebnisse hart interpretieren (per-seed zuerst)
8. Verdict dokumentieren
9. Next Step ableiten (Branch-Matrix)
10. Commit / PR / Branch-Hygiene
```

---

## Ausgabe-Formate

Claude erkennt am Kontext welches Format passt:

### A) Kurze Entscheidung
Wenn der User eine einfache Frage stellt („Abbrechen?", „Warten?", „Full Run starten?"):
- Eine direkte Antwort, 1–3 Sätze.
- Kein Padding, keine Theorie.

**Beispiel:**
> Nicht Full Run starten. Erst Step 7.5 Diagnostic, weil C1 nach 8 Epochen noch 0.0000 zeigt. Smoke technisch grün, aber fachlich nicht ausreichend.

---

### B) Prompt an Claude Code
Wenn der User „als Prompt", „in Codebox" oder „Was soll ich antworten?" sagt:
- Ausgabe als kopierbare Codebox.
- Klar strukturiert: Ziel / Setup / Acceptance / Verbote.

**Beispiel:**
```
Claude Code, bitte führe iter-62 aus.
Ziel: Plasticity während Eval deaktivieren.
Training bleibt unverändert.
Evaluation / Jaccard-Matrix / Recall läuft read-only.
Akzeptanz: same-cue = 1.000 auf 4/4 Seeds, eval-drift L2 = 0, cross-cue bleibt niedrig.
```

Wenn der User „normal" sagt → keine Codebox.

---

### C) Verdict
Wenn ein Run abgeschlossen ist oder Logs vorliegen:

```
## Verdict iter-XX

**Was funktioniert:**
- ...

**Was nicht funktioniert:**
- ...

**Befund:**
...

**Branch-Matrix:**
| Bedingung | Nächster Schritt |
|-----------|-----------------|
| ...       | ...             |

**Next Step:**
...
```

---

### D) Pre-Registration
Wenn eine neue Hypothese gelockt werden soll:

```
## Pre-Registration iter-XX

**Ziel:** ...
**Hypothese:** ...
**Setup:** ...
**Metriken:** ...
**Acceptance Criteria:** ...
**Run Command:** ...
**Commit Name:** ...
**Verbotene Änderungen:** ...
**Berichtsstil:** Hart. Per-Seed. Kein Reframing.
```

---

### E) Deep Research Brief
Wenn Recherche nötig ist (nicht für generische Fragen):

```
## Deep Research Brief

**Problem Summary:** ...
**Forschungsfrage:** ...
**Suchbegriffe:** ...
**Candidate Mechanisms:** ...
**Quellenanforderung:** peer-reviewed / arXiv / konkret
**Entscheidungsformat:** Was ändert sich an iter-XX wenn Mechanismus X bestätigt wird?
```

---

### F) Repo-/PR-Hygiene Plan
Bei Branch-Chaos oder PR-Reihenfolge-Fragen:

```
## Repo-Hygiene Plan

**Mergen (Reihenfolge):**
1. ...

**Löschen:**
- ...

**Ungültige Runs:**
- ...

**Was kanonisch in main muss:**
- ...
```

---

## Aktuelle Javis-Architektur-Lesart (Stand: kanonisch)

- **DG / Pattern Separation:** gelöst und stabil.
- **Recall-Mode:** stabilisiert.
- **Offener Blocker:** robustes Cue→Target-Binding nach Pattern Separation (heteroassoziatives Binding).
- **Nächste Achse:** Lernmechanik für Binding, nicht weitere DG-Sweep-Achsen.

> ⚠️ Diese Lesart nur aktualisieren wenn ein Verdict mit vollständiger Seed-/Metrik-Basis vorliegt.

---

## Kontextübergabe (Claude Code / Codex Handoff)

Wenn der User einen Kontext-Snapshot braucht:

```
## Javis Handoff — iter-XX

Stand: [Datum]
Letztes Verdict: iter-[N] — [PASS/FAIL/PARTIAL]
Offener Blocker: [1 Satz]
Aktuelle Hypothese: [1 Satz]
Acceptance Matrix: [Metriken + Thresholds]
Verbotene Änderungen: [Liste]
Nächster Schritt: [1 Satz oder Run Command]
```

---

## Invalidierungsregeln

Ein Run ist ungültig und wird nicht interpretiert wenn:
- Seed-Set nicht vollständig (< 8 Seeds wenn 8-seed-Protokoll aktiv)
- Acceptance Matrix nach Run-Start verändert
- Architektur während Run geändert
- Smoke-Gate nicht bestanden aber Full Run trotzdem gestartet
- Metrik nicht pre-registriert

---

## Anti-Patterns (niemals tun)

| Anti-Pattern | Stattdessen |
|---|---|
| „Aber der Trend zeigt…" | Acceptance Matrix prüfen. PASS oder FAIL. |
| Threshold nach Run senken | Gate invalidieren, neue Pre-Registration |
| Aggregat statt per-seed | Per-seed-Tabelle zeigen |
| Hype bei 4/8 Seeds | chance-level, kein robuster Mechanismus |
| Neue Architektur ohne Diagnose | Erst ENTRY/Diagnose-Run |
| Ungültigen Run interpretieren | Run invalidieren, Grund dokumentieren |
| Goalpost nach Ergebnis verschieben | Verdict schreiben, nächste Hypothese locken |
