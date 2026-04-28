# 04 — Multi-Region-Architektur und Address-Event-Routing

**Stand:** Javis verlässt die Mikro-Ebene. Mehrere `Region`s leben in einem
`Brain` und kommunizieren ausschließlich über asynchrone Spike-Events mit
realistischen axonalen Verzögerungen.

## Neue Module

### `Region` (`src/region.rs`)

Dünner Wrapper um `Network` mit Namen — bewusst minimal. Eine Region ist
*per definitionem* ein eigenständiges Netzwerk; alles, was zwischen
Regionen passiert, gehört in `Brain`.

### `Brain` (`src/brain.rs`)

- Hält `Vec<Region>` und `Vec<InterEdge>`.
- Pro Region und Source-Neuron eine Liste ausgehender Inter-Edge-IDs
  (`outgoing[region][neuron] -> Vec<u32>`) — analoge Adjazenz wie
  intra-Region.
- Eine flache Queue `pending: Vec<PendingEvent>` für noch nicht
  zugestellte Events.
- `step(externals)` macht pro Schritt:
  1. **Liefere alle fälligen Events:** für jedes `ev` mit
     `ev.arrive_at <= t` wird `regions[dst].network.i_syn[dst_neuron] +=
     ev.weight`. Das nutzt direkt die normale post-synaptische Pipeline
     der Ziel-Region (mit τ_syn-Decay).
  2. **Step jede Region** lokal (interne Dynamik + STDP).
  3. **Schedule neue Events:** für jeden lokalen Spike, der ein
     Inter-Edge hat, wird ein `PendingEvent` mit
     `arrive_at = t + delay_ms` und `weight = sign · edge.weight` in die
     Queue geschoben — `sign` aus dem Dale-Typ des Senders.

### Address-Event-Ansatz

Das ist das gleiche Prinzip wie AER in neuromorpher Hardware (SpiNNaker,
Loihi): zwischen Modulen wandern keine kontinuierlichen Ströme, sondern
gepackte „Events" mit (Ziel-Adresse, Gewicht, Ankunftszeit). Das skaliert
linear mit Spike-Volumen — bei sparsen Verbindungen Größenordnungen
billiger als dichte Matrix-Multiplikation pro Schritt.

## Test 6 — Zwei-Regionen-Signal-Transfer

### Setup

- 2 Regionen, je 1000 Neuronen, 80/20 E/I, `p_connect = 0.1` intern,
  STDP an, identisches Wiring-Rezept wie in Notiz 03.
- 1 % der R1-E-Neuronen sind Projektions-Neuronen (8 Stück).
  Jeder Sender hat einen Fan-Out von 15 zu zufälligen R2-E-Neuronen.
- Inter-Region-Gewicht: 12.0 (deutlich stärker als intra-E mit 0.2,
  damit ein einzelnes Event eine erkennbare Wahrscheinlichkeit hat,
  das Ziel zu treiben).
- Delay: uniform 2–5 ms.
- R1: Poisson-Drive 80 Hz / 80 nA.
- R2: kein externer Input.

### Ergebnis

```
control (keine Inter-Edges)
  R1: 2867 E + 690 I spikes  →  3.58 Hz E
  R2: 0 + 0 spikes           →  0.00 Hz (tot, wie erwartet)

wired (1 % R1-E projizieren in R2)
  R1: 2867 E + 690 I spikes  →  3.58 Hz E   (identisch zum control,
                                              R2 hat keinen Rückkanal)
  R2: 540 E + 0 I spikes     →  0.68 Hz E
  events_delivered = 555
```

### Interpretation

- **Signal kommt rüber:** R2 geht von 0 auf 540 Spikes. Diese 540 Spikes
  werden ausschließlich durch die 555 zugestellten Inter-Region-Events
  ausgelöst (R2 hat keinen externen Drive).
- **R2 bleibt im asynchron-irregulären Regime** (0.68 Hz, weit unter
  unserer 30-Hz-Runaway-Schwelle). Die importierte Aktivität wird durch
  R2's interne Inhibition gedämpft, statt durchzubrennen.
- **R1 völlig unbeeinflusst** vom angeschlossenen R2 — 1 zu 1 derselbe
  Spike-Count wie im Control. Das bestätigt die einseitige Topologie
  (R1 → R2, kein Rückkanal in diesem Test).
- Bei dieser sehr sparsamen Kopplung schweigt die R2-I-Population (0 Hz).
  Das ist kein Bug, sondern direkte Folge der niedrigen E-Aktivität —
  unter ~1 Hz hat die I-Population zu wenig Eingangsstrom, um Schwellen
  zu erreichen. Bei stärkerer Inter-Region-Kopplung würde I sichtbar
  mitspielen.

## Hardwaremäßiges Bild

Wir haben jetzt das Grundgerüst, um echte Multi-Modal-Architekturen zu
bauen:

```
[Encoder-Region]  →  [Assoziativer Cortex]  →  [Memory-Region]
                                ↑                     │
                                └─────────────────────┘
```

Jede Region läuft mit eigenem E/I-Balance-Tuning, eigenem STDP-Regime,
eigenen Hyperparametern. Inter-Region-Wege sind sparsam und mit
biologisch realistischen Delays — exakt wie der echte Cortex.

## Nächste Schritte

1. **Encoder-Stub** — Text/Code → Sparse Distributed Representation,
   simple Hash-basiert. Erstmal eine Region als „Eingangs-Cortex".
2. **Multi-Region-Assembly-Demo** — wir trainieren über zwei Regionen
   hinweg eine Assoziation: Pattern in R1 → Pattern in R2. Der Beweis
   wäre: nach dem Training löst R1-Pattern ohne externen R2-Drive das
   gespeicherte R2-Pattern aus.
3. **Threading** — `Region::step` lässt sich nach kleinen Anpassungen
   parallel über `rayon` ausführen. Inter-Region-Pending-Queue müsste
   dann thread-safe werden (Mutex oder per-Region-Mailboxes).
