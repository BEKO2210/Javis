# 38 — Read-only Recall: 2.5× Throughput, flat Server-Latenz

**Stand:** Iteration 20. Notiz 35 hat den `Mutex`-Bottleneck im
Recall-Pfad gemessen: 141 ops/s Plateau, p99-Latenz wuchs linear mit
Concurrency (771 ms bei 100 parallel). Iteration 18 (Concurrency-Cap)
hat den Server vor dem Verstopfen geschützt; jetzt heben wir den
Bottleneck selbst auf.

## Architektur-Refactor

Vor diesem Patch: Recall hielt eine exklusive `Mutex<Inner>`-Lock
während der gesamten Brain-Simulation (~7 ms wall pro Recall). Alle
parallelen Recalls serialisierten.

Nach diesem Patch:
1. `Brain` und `Network` sind `Sync` und werden während Recall **nur
   gelesen** (synapse weights, topology, neuron config — alles
   immutable).
2. Die *transiente* Simulation (Membranspannungen, Synaptic-Currents,
   Zeit, Event-Queue) lebt in einem neuen `BrainState` /
   `NetworkState`, das jede Recall-Session lokal allokiert.
3. `AppState.inner` ist jetzt `Arc<RwLock<Inner>>` statt `Mutex<Inner>`.
   Train hält die Write-Lock; Recall hält nur die Read-Lock —
   beliebig viele parallele Recalls können gleichzeitig laufen.

## Neue API in `snn-core`

```rust
pub struct NetworkState { v, refractory_until, last_spike,
                          i_syn, i_syn_nmda, i_syn_gaba,
                          time, step_counter, synapse_events }

impl Network {
    pub fn fresh_state(&self) -> NetworkState;
    pub fn step_immutable(&self, &mut NetworkState, external) -> Vec<usize>;
}

pub struct BrainState { regions: Vec<NetworkState>,
                        pending: PendingQueue,
                        time, events_delivered }

impl Brain {
    pub fn fresh_state(&self) -> BrainState;
    pub fn step_immutable(&self, &mut BrainState, externals) -> Vec<Vec<usize>>;
}
```

Vier Korrektheits-Tests (`crates/snn-core/tests/immutable_step_equivalence.rs`):

1. **`network_step_immutable_matches_step_when_plasticity_off`** —
   identische Spike-Outputs Schritt für Schritt; identische
   Membranpotentiale und Synaptic-Currents am Ende.
2. **`brain_step_immutable_matches_step_when_plasticity_off`** —
   gleiche Garantie für die Multi-Region-Pfad inkl. Inter-Region-
   Heap.
3. **`brain_step_immutable_does_not_mutate_brain`** — nach 200
   Schritten read-only sind Synapse-Weights, Inter-Edge-Weights, und
   `brain.time` *bit-identisch* zum Vor-Zustand. Nur `state.time`
   wandert.
4. **`network_state_lazy_nmda_gaba_allocation`** — NMDA/GABA-Buffer
   nur dann allokiert, wenn das Netzwerk Synapsen dieser Sorte hat.
   Default-AMPA-only-Netze zahlen keinen Memory-Aufpreis.

`step_immutable` ignoriert Plastizität *unconditionally* — die
Brain-Plastizitäts-Konfiguration darf bleiben, wo sie ist; im
Recall-Pfad spielt sie keine Rolle. Damit fällt das
`disable_stdp_all() / disable_istdp_all() / disable_homeostasis_all()`-
Voodoo aus dem alten `run_recall` weg, das ohnehin nur deshalb da
war, weil die mutierende `step()` sonst weights verändert hätte.

## Migration in `viz::AppState`

```rust
pub struct AppState {
    inner: Arc<RwLock<Inner>>,   // ← war Mutex<Inner>
    ...
}

pub async fn run_recall(&self, query, tx) {
    let g = self.inner.read().await;        // ← nur Read-Lock
    let mut state = g.brain.fresh_state();   // per-call, ~30 KB
    let counts = run_with_cue_streaming_immutable(
        &g.brain, &mut state, ..., &tx,
    ).await;
    // ... decode, send Decoded event, drop read lock ...
}
```

Train, reset, snapshot-load nehmen weiter die Write-Lock — sie
mutieren echt. Stats und snapshot-save nehmen Read-Lock.

`tokio::sync::RwLock` lässt beliebig viele Reader gleichzeitig durch;
ein Writer wartet auf alle aktiven Reader und blockiert dann neue
Reader bis er fertig ist (write-starvation-frei).

## Load-Test-Resultate (`docker compose up`, 15s pro Stufe)

```
                     before (Mutex)        after (RwLock + immutable)
  conc | ops/s |      p50  /   p99    |   ops/s |   p50  /   p99   | srv mean
  -----+-------+----------------------+---------+------------------+----------
     1 |   116 |    8.5  /    11.0 ms |    112  |   8.8  /  10.8 ms |  7.3 ms
    10 |   142 |   70.3  /    84.4 ms |    357  |  27.4  /  47.8 ms |  9.3 ms
    50 |   142 |  351.8  /   397.0 ms |    359  | 141.0  / 244.0 ms |  9.3 ms
   100 |   141 |  700.5  /   771.0 ms |    358  | 270.4  / 564.0 ms |  9.4 ms
```

Drei klare Gewinne:

1. **Throughput 142 → 358 ops/s** = 2.52 × bei mittlerer/hoher
   Concurrency. Single-Tenant-Speed bleibt unverändert (Single-
   Worker-Recall hat sowieso keinen Lock-Wettkampf).
2. **Server-side mean Latenz: konstant ~9 ms** auf jeder Stufe.
   Vorher 7→68→346→685 ms — die Brain-Step-Serialisierung ist weg.
3. **Client-side p99 trotzdem nicht null bei 100 parallel** —
   verbleibende 564 ms = ~358 ops/s bei 100 Clients = 3.58 ops/s
   pro Client = ~280 ms wall pro Request. Das ist Tokio-Worker-
   Queueing der TCP-Verbindungen, nicht mehr der Brain. Mehr CPU-
   Cores würden direkt skalieren.

## Production-Verdikt aktualisiert

| Anwendungsfall | Vor Iter 20 | Nach Iter 20 |
| --- | --- | --- |
| ≤ 3 parallel | ✅ | ✅ |
| 10 parallel | ✅ p99 84 ms | ✅ p99 48 ms |
| 50 parallel | ⚠ p99 397 ms | ✅ p99 244 ms |
| 100 parallel | ✗ p99 771 ms | ⚠ p99 564 ms |
| 200+ parallel (mit cap raise) | ✗ | ⚠ skaliert mit CPU |

Die obere Schranke ist jetzt **CPU-cores im Container**, nicht
mehr Mutex-Wartezeit. 4-Core-Host = ~360 ops/s. 16-Core-Host =
realistischerweise ~1400 ops/s ohne weitere Code-Änderungen.

## Was sich nicht ändert

- **Train-Pfad:** unangetastet. Mutiert echt (`Brain::step`) und
  hält Write-Lock — niemand will gleichzeitig zu Train trainieren.
- **Snapshot-Format:** unverändert. NetworkState lebt nur per
  Recall-Call, ist nicht Teil des Files.
- **Korrektheit:** vier neue Tests beweisen Spike-Bit-Identität
  zwischen mutating und immutable Pfad. Keine numerische Drift.

## Was Iter 20 nicht macht

- **Kein async unter dem read-lock-hold.** `tx.send(...).await` im
  Recall-Loop läuft, während wir die Read-Lock halten. Tokio's
  RwLock blockiert in dem Moment KEINE anderen Reader, nur Writer
  — also ok. Aber Writer müssen warten, bis der langsamste Reader
  fertig ist. Wenn Train-Latenz wichtig wird, könnte man die
  Trained-Sentences (für RAG-Lookup) und Encoder cheap clonen,
  read lock früh droppen, Sim ohne Lock laufen lassen. Out of
  scope hier.
- **Kein per-thread-Brain-Pool.** Wir teilen einen einzigen
  `Arc<Brain>` per immutable read. Wenn der Compute-Pfad mal
  CPU-cache-sensibel wird, könnte ein Pool von replizierten
  Brains pro Tokio-Worker noch mehr Throughput bringen.

## Status

- 108/108 Tests grün (104 + 4 immutable-step Equivalence-Tests)
- 0 Clippy-Warnings, fmt clean
- Docker-Image baut + bringt sich healthy hoch
- Load-Test bestätigt 2.5× Throughput, flat Server-Latenz
