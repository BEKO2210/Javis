# 22 — Min-Heap-Queue, AMPA/NMDA/GABA-Channels, null Lints

**Stand:** Die drei in Notiz 21 als "noch offen" markierten Punkte
sind erledigt. Brain skaliert auf große Multi-Region-Systeme,
heterogenes synaptisches Decay ist verfügbar (default off, opt-in
für realistische Synapsen-Mischungen), die Codebase ist
clippy-clean.

## 1. `PendingQueue` als BinaryHeap (min-heap auf `arrive_at`)

Vor Iteration 10 lag `Brain.pending: Vec<PendingEvent>`. Pro Step
wurde der ganze Vec gedraint, due Events delivered, alle anderen in
ein neues `keep` Vec geschoben — `O(P)` pro Step, egal wie viele
Events tatsächlich fällig waren.

Neu: `Brain.pending: PendingQueue` mit `BinaryHeap<HeapEntry>`
backing. `push` ist `O(log P)`, `drain_due(t)` pop't nur die fälligen
Events in `O(k log P)` mit `k = due events`. Bei wachsenden
Multi-Region-Systemen ist das die richtige asymptotische Form.

### Wichtige Design-Entscheidungen

- **`f32::total_cmp` statt `u64`-Mikrosekunden**. Mein erster Versuch
  konvertierte `arrive_at` ms → u64 μs für den Heap-Vergleich. Das
  brach den `multiple_overlapping_concepts_coexist`-Test, weil bei
  Float-Drift (`t = 0.1f32 + 0.1 + … + 0.1` driftet von 0.3) der
  truncate-on-cast nicht exakt dasselbe Verhalten wie `f32 <=` hat.
  Lösung: `total_cmp` benutzen, `arrive_at` als f32 belassen — der
  `<=` cutoff in `drain_due` ist bit-identisch zum alten Vec-code.

- **Sequence-Tiebreak.** `HeapEntry` trägt eine monotone
  Insertions-Sequenznummer. Bei gleichem `arrive_at` poppen ältere
  Events zuerst (FIFO). Damit ist die Delivery-Order deterministic
  und entspricht exakt der Insertion-Order der ursprünglichen Vec-
  Implementation. Alle alten Tests bleiben bit-identisch.

### API

```rust
pub struct PendingQueue { /* private */ }

impl PendingQueue {
    pub fn new() -> Self;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn clear(&mut self);
    pub fn push(&mut self, ev: PendingEvent);
    pub fn drain_due(&mut self, t_ms: f32) -> DrainDue<'_>;
}
```

`Brain.pending` ist weiterhin `pub`, also Teil der API. Anybody who
held a `Vec<PendingEvent>` would have to migrate, but inside the
workspace nobody does.

## 2. AMPA / NMDA / GABA — heterogenes synaptisches Decay

Vor Iteration 10 hatte `Network` ein einziges `tau_syn_ms`-Feld. Alle
Synapsen decayten gleich schnell. Biologisch unrealistisch:

| Receptor | τ typisch (ms) |
| -------- | -------------: |
| AMPA     | 2–5            |
| NMDA     | 50–150         |
| GABA-A   | 5–30           |

Neu: `Synapse.kind: SynapseKind { Ampa, Nmda, Gaba }`. `Network` hat
drei separate `i_syn`-Vektoren und drei Decay-Konstanten. Beim
Spike-Delivery landet das Gewicht im richtigen Channel:

```rust
let kind = self.synapses[eid].kind;
let channel = self.ensure_channel(kind);
channel[post] += sign * w;
```

LIF-Integration summiert über alle aktiven Channels:

```rust
let mut total = ext + self.i_syn[idx];
if let Some(v) = self.i_syn_nmda.get(idx) { total += *v; }
if let Some(v) = self.i_syn_gaba.get(idx) { total += *v; }
n.step(t, dt, total);
```

### Lazy Allocation

NMDA / GABA-Buffers werden erst beim ersten Spike der jeweiligen
Kind angelegt (`ensure_channel`). Default-AMPA-only Netze haben
weiterhin nur **einen** decay-loop pro step — keine Performance-
Regression.

`set_synaptic_taus(ampa, nmda, gaba)` setzt alle drei auf einmal,
mit positiver-finiter-Validierung.

### Snapshot-Backwards-Compat

`Synapse.kind` hat `#[serde(default)]` → pre-iteration-10-Snapshots
ohne `kind`-Feld lesen sich als AMPA wieder ein. Test
`snapshot_without_kind_field_defaults_to_ampa` verifiziert das.

## 3. Null Clippy-Lints

Vor Iteration 10: 14 unique clippy-warnings (manche dupliziert über
test files). Jetzt:

```sh
$ cargo clippy --all-targets --release 2>&1 | grep -c '^warning'
0
```

Behoben:
- `let mut x = Foo::default(); x.field = …` → struct-init mit
  `..Foo::default()` (8 Stellen in Tests + Examples)
- `len() > 0` → `!is_empty()` (2 Stellen in viz smoke)
- `for k in 0..n_proj { let src = r1_exc[k]; }` →
  `for &src in &r1_exc[..n_proj]` (1 Stelle in two_regions)
- `(N as f32 * x) as f32` → `N as f32 * x` (1 Stelle in two_regions)

## 12 neue Tests in `crates/snn-core/tests/heap_and_channels.rs`

### PendingQueue

- `pending_queue_drains_in_chronological_order` — older tied-arrival
  events pop first
- `pending_queue_respects_cutoff` — only events with
  `arrive_at <= cutoff`
- `pending_queue_clear_resets_sequence` — clear restores FIFO behavior
- `pending_queue_handles_many_events_efficiently` — 1000 events,
  drain order is monotone
- `brain_uses_heap_pending_queue_underneath` — Brain end-to-end
  with the new queue still delivers events

### SynapseKind / channels

- `default_synapse_is_ampa`
- `nmda_channel_decays_slower_than_ampa` — analytische Verifikation:
  - NMDA τ=100 → exp(-0.1) ≈ 0.905
  - GABA τ=10  → exp(-1.0) ≈ 0.368
  - AMPA τ=5   → exp(-2.0) ≈ 0.135
  - Reihenfolge `NMDA > GABA > AMPA` strict
- `nmda_synapse_routes_to_nmda_channel` — verifiziert dass NMDA
  delivery den NMDA-Buffer und nicht den AMPA-Buffer trifft
- `snapshot_round_trip_preserves_synapse_kind` — JSON roundtrip
- `snapshot_without_kind_field_defaults_to_ampa` — backward-compat
- `set_synaptic_taus_validates_inputs` — happy path
- `set_synaptic_taus_rejects_negative_nmda` — should_panic

## Workspace-Status

- **91/91 Tests grün** (79 + 12 neue)
- **0 clippy-warnings** workspace-weit (`cargo clippy --all-targets`)
- **0 dead code** mehr in lib (alle Hauptpfad-Felder werden gelesen)
- **API-stabil** — kein bestehender Test musste angepasst werden,
  Snapshots der vorigen Iteration laden mit Defaults

## Was Iteration 10 explizit *nicht* macht

- `Synapse.kind` ist eine *Channel*-Auswahl, kein Plastizitäts-
  Schalter. STDP / iSTDP arbeiten weiter auf den Synapsengewichten
  unabhängig von kind. Wenn man heterosynaptische Plastizitäts-
  Regeln pro Channel-Typ haben will, ist das ein eigener Schritt.
- AMPA / NMDA / GABA werden algorithmisch genau gleich behandelt,
  nur mit unterschiedlichen `τ`. Keine NMDA-Magnesium-Block-Modelle
  oder dergleichen — dafür wäre ein zweiter Lookup auf der
  Postsynapse nötig.
- Per-Synapse individuelle τ (nicht nur die drei Standard-Klassen)
  ist immer noch nicht möglich. Wer das braucht, kann sein eigenes
  Channel-Set einrichten — aber dann skaliert die `decay`-Schleife
  mit der Channel-Zahl, nicht mehr mit O(N).
