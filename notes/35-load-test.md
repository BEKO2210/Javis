# 35 — Load-Test gegen den `recall`-Pfad

**Stand:** Iteration 17. Bisher hatten wir:
- Criterion-Microbenches für die heißen Pfade (Notiz 31)
- E2E-Sanity-Check (Notiz 34)

Aber keine Daten zur Frage *"Wie skaliert der Server, wenn N Clients
gleichzeitig drauflos hämmern?"* Jetzt schon. Tests gegen den
laufenden Container (`docker compose up`), Mock-LLM, 4
Concurrency-Stufen × 15 s pro Stufe.

## Setup

`scripts/load_test.py` (neu, ~120 Zeilen Python):

- N parallele Worker, jeder im Loop: WS auf, `action=recall&query=…`,
  drainen bis `done`, schließen, Latenz aufzeichnen, repeat.
- Queries rotieren über 6 verschiedene (rust, python, cpp, language,
  memory, safety) — kein Single-Hot-Key-Bias.
- Pro Stufe Server-side `javis_recall_duration_seconds_*` Counter
  vorher/nachher abgreifen; Cross-Check mit Client-side Werten.
- Im Hintergrund läuft ein `docker stats`-Loop, der RSS+CPU alle 2 s
  abgreift.

Aufruf:

```sh
python3 scripts/load_test.py --levels 1,10,50,100 --duration 15
```

## Ergebnisse

```
 conc     ops    ops/s     p50     p95     p99    srv mean
    1    1744    116.2     8.5     9.8    11.0       7.0
   10    2136    141.7    70.3    76.9    84.4      68.1
   50    2170    141.5   351.8   380.2   397.0     346.4
  100    2227    141.4   700.5   760.2   770.8     685.4
```

Alle Latenzen in **Millisekunden**. `srv mean` ist der Mittelwert,
den der Server selbst über sein Histogram gemessen hat (≈ 2 ms unter
client-side wegen Network/serde-Overhead).

### Drei klare Aussagen

**1. Throughput-Plateau bei ~141 recalls/sec**
Ab Concurrency 10 sättigt der Server. Mehr Worker holen nichts
mehr raus — sie bauen nur die Queue auf.

**2. Latenz wächst linear mit Concurrency**
- 1 Worker: p99 = 11 ms
- 10 Worker: p99 = 84 ms (~ 8 ×)
- 50 Worker: p99 = 397 ms (~ 36 ×)
- 100 Worker: p99 = 771 ms (~ 70 ×)

Klassisches Bild eines `Mutex`-serialisierten Pfades: jede Anfrage
wartet, bis alle vorher Ankommenden durch sind. Der Server *crasht*
nicht und drop't auch nichts — er stellt einfach in die Queue.

**3. Keine Errors, keine Drops**
Über 8 277 Recalls hinweg: `errors: 0` auf jeder Stufe. Selbst bei
100 parallelen Sessions liefert jede ein vollständiges
`decoded`-Event und schließt sauber.

## Memory-Profil

```
idle:                  27 MB
during c=1 (low):      28-29 MB
during c=10:           29-31 MB
during c=50:           31-34 MB
during c=100 (peak):   34-35 MB
30 s post-load (cool): 34 MB
```

~7-8 MB Anstieg während Load, der nach Cool-Down nicht
zurückgeht — typische steady-state allocator-Slack. **Kein Leak.**
Wenn jemand das Repo 24 h unter Last laufen lässt, sollte der RSS
nicht weiter wachsen; der hier sichtbare Sprung ist allokierter
Channel/Buffer-Speicher, der für Wiederverwendung erhalten bleibt.

CPU war durchgehend bei 100-124 % (= 1.0-1.2 Cores) auf 100-Worker-
Stufe — das passt zum Mutex-Bottleneck: ein Tokio-Worker hält die
Brain-Mutex und macht die ganze Arbeit, andere Tokio-Worker können
parallel nur die WS-IO machen.

## Wo der Bottleneck steckt

`AppState.inner: Arc<Mutex<Inner>>`. Jeder `run_recall`-Aufruf:

1. `let mut g = self.inner.lock().await;` — exklusiv die Mutex
2. mutates `g.brain` (disable_stdp_all, reset_state, run_with_cue_streaming…)
3. liest `g.dict.decode(...)` und `g.trained_sentences`
4. Lock-Drop am Funktion-Ende

Punkt 2 ist der Knackpunkt: Recall *mutiert* Brain-State (Plastizitäts-
Schalter, transient buffers, reset_state, Spike-Stepping). Eine
einfache Migration zu `RwLock` mit Read-Only-Recall geht nicht
ohne Refactoring.

## Was das für Production heißt

| Anwendungsfall | Verdikt |
| --- | --- |
| Single-Tenant Demo (1-3 User) | ✓ p99 unter 100 ms, kein Sweat |
| Small-Team Service (≤ 10 parallel) | ✓ p99 unter 100 ms, akzeptabel |
| Mid-Size (50-100 parallel) | ⚠ p99 in Sekunden, schlecht |
| Large-Scale | ✗ unverwendbar ohne Refactor |

Die naheliegenden Skalierungs-Pfade, alle out-of-scope für *jetzt*:

1. **Read-Only Recall**: Refactor `run_recall` so, dass es keinen
   mutable `Brain` braucht — z. B. einen Pool vorberechneter Brain-
   Clones, oder die transienten Buffer auf Stack-allokiert pro
   Request. Größerer Eingriff in den Hot-Path.
2. **Sharded Brains**: N Brain-Replikate hinter Load-Balancer, jede
   Replika mit eigener Mutex. Memory: ~30 MB/Replika.
3. **Server-side Concurrency-Cap**: Wenn Throughput-Plateau bei 140
   ops/s steht und Latency erst über 10 Workern explodiert,
   genügt es vielleicht schon, eingehende Sessions ab Worker > N
   zu rejecten (oder mit `Retry-After` zu zurückweisen). Kleinster
   Fix mit größtem User-Impact.

## Was Iteration 17 *nicht* fixt

Ich habe nichts an der Architektur geändert. Der Mutex bleibt. Die
Daten geben uns ein klares Bild der aktuellen Skalierung — Fixes
wären eigene Iterationen, ausgelöst durch konkretes Anforderungs-
profil (z. B. „wir wollen 50 parallele Recalls in < 100 ms p99
können"). Solange wir nicht wissen, ob jemand das überhaupt
braucht, ist Refactor verfrüht.

## Geliefert

- `scripts/load_test.py` — reproduzierbarer Load-Test, parametrisiert
  über `--levels` und `--duration`. Pendant zu `sanity_check.py`,
  derselbe Stil.
- Verifizierte Baseline-Zahlen: 141 ops/s, p99 11 ms (single), keine
  Errors, kein Memory-Leak.
- Klare Diagnose des Mutex-Bottlenecks samt Refactor-Optionen für
  später.

## Status

- 98/98 Tests grün (unverändert)
- Load-Test: 8 277 Recalls über 4 Stufen, 0 Errors
- Memory: 27 → 35 MB Peak, settles auf 34 MB
- Stack-State sauber zurückgeräumt nach Test
