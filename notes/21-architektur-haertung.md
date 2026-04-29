# 21 — Architektur-Härtung: dead code, bounds checks, lints

**Stand:** Drei echte Architekturmängel beim Code-Audit gefunden und
behoben. 16 neue Regressionstests pinnen die Verträge fest, kein
bestehender Test bricht.

## Was war kaputt

### 1. `Synapse.tau_syn` war dead code

`crates/snn-core/src/synapse.rs` hatte ein `tau_syn: f32` Feld, das
beim Erstellen jeder Synapse auf `5.0` gesetzt wurde. Der Decay-Loop
in `Network::step` ignorierte es jedoch komplett:

```rust
// crates/snn-core/src/network.rs
let decay_psc = (-dt / 5.0_f32).exp();   // ← hardcoded, nicht aus Synapse!
```

Das Feld war seit Iteration 1 (Notiz 01 dokumentierte sogar die
Vereinfachung) vergessen worden. Die Doc-Kommentare in `synapse.rs`
**logen** über das Verhalten:

> "When the pre-synaptic neuron fires, `weight` is added to the
> post-synaptic neuron's `i_syn`, which then decays with `tau_syn`."

`tau_syn` wurde nie gelesen.

### 2. `Network::connect` und `Brain::connect` ohne Bounds-Checks

```rust
// vorher
pub fn connect(&mut self, pre: usize, post: usize, weight: f32) -> usize {
    let id = self.synapses.len();
    self.synapses.push(Synapse::new(pre, post, weight));
    self.outgoing[pre].push(id as u32);   // ← panic bei pre >= len()
    self.incoming[post].push(id as u32);  // ← panic bei post >= len()
    id
}
```

Bei einer falschen Index-Eingabe stürzte das Programm mit einem
nichtssagenden `index out of bounds` ab — die wahre Quelle (Caller-
Code mit dem falschen Index) blieb unklar. NaN- und Infinity-Gewichte
wurden stillschweigend akzeptiert und konnten später Plastizitäts-
Updates korrumpieren.

`Brain::connect` hatte das gleiche Problem für vier Indizes
(src_region, src_neuron, dst_region, dst_neuron), zusätzlich keine
Validierung von `delay_ms > 0`.

### 3. Diverse Clippy-Lints in lib code

Nicht kritisch, aber kosmetische Sünden:
- `map_or(true, …)` → idiomatisch `is_none_or(…)`
- `step_counter % h.apply_every == 0` → `is_multiple_of(...)`
- `format!("static string")` ohne Interpolation
- `let mut s = StdpParams::default(); s.field = …;` → struct-init
  syntax mit `..Default::default()`
- `impl Default for FingerprintMode` → `#[derive(Default)]` mit
  `#[default]` annotation

## Was sich geändert hat

### Synapse → Network

`Synapse.tau_syn` ist entfernt. Stattdessen hält `Network` ein
**globales** `tau_syn_ms: f32` (default 5 ms) das alle Synapsen
gleichermaßen verwendet:

```rust
pub struct Network {
    // …
    #[serde(default = "default_tau_syn_ms")]
    pub tau_syn_ms: f32,
}

impl Network {
    pub fn set_tau_syn_ms(&mut self, ms: f32) {
        assert!(ms > 0.0 && ms.is_finite(), "…");
        self.tau_syn_ms = ms;
    }
}
```

Ein-Synapse-pro-Synapse-τ wäre biologisch korrekter, aber
performance-teurer (eigener Decay pro Synapse statt pro Neuron). Per-
Network ist die einfachere und immer noch realistische Wahl —
biologisch entspricht das z. B. „dieses Cortex-Areal modelliert
AMPA-Synapsen mit τ ≈ 5 ms".

**Snapshot-Kompatibilität**: das neue Feld hat
`#[serde(default = "default_tau_syn_ms")]`. Pre-iteration-9-Snapshots
(in denen das Feld fehlt) laden mit dem Default-Wert — getestet durch
`snapshot_without_tau_syn_uses_default`.

### Bounds-checked `connect`

Beide `connect`-Methoden validieren jetzt vorher:

```rust
// Network::connect
assert!(pre < n,    "Network::connect: pre {pre} out of bounds (only {n} neurons)");
assert!(post < n,   "Network::connect: post {post} out of bounds (only {n} neurons)");
assert!(weight.is_finite(), "Network::connect: weight must be finite, got {weight}");
assert!(id < u32::MAX as usize, "Network::connect: synapse count exceeds u32 capacity");

// Brain::connect
assert!(src_region < n_regions, "…");
assert!(dst_region < n_regions, "…");
assert!(src_neuron < n_src,     "…");
assert!(dst_neuron < n_dst,     "…");
assert!(weight.is_finite(),     "…");
assert!(delay_ms > 0.0 && delay_ms.is_finite(), "…");
```

Saubere panic-Messages mit Variablen-Werten — Caller weiß sofort,
welcher Wert falsch war.

## 16 neue Hardening-Tests (`crates/snn-core/tests/hardening.rs`)

### tau_syn-Verhalten

- `default_tau_syn_is_5_ms` — klassisch
- `tau_syn_setter_changes_psc_decay` — analytische Verifikation:
  bei τ=20 ms ist `i_syn(10ms) ≈ exp(-0.5) ≈ 0.607`; bei τ=5 ms
  ist es `exp(-2) ≈ 0.135`. Beide Werte werden auf ±0.05
  matched
- `set_tau_syn_rejects_zero` / `_negative` — `should_panic`

### Bounds-Checks

Sieben `should_panic` Tests, jeweils mit dem erwarteten String-Match:

- `network_connect_rejects_pre_out_of_bounds` (`"out of bounds"`)
- `network_connect_rejects_post_out_of_bounds`
- `network_connect_rejects_nan_weight` (`"weight must be finite"`)
- `network_connect_rejects_inf_weight`
- `brain_connect_rejects_unknown_src_region` (`"src_region"`)
- `brain_connect_rejects_unknown_dst_region`
- `brain_connect_rejects_oob_src_neuron`
- `brain_connect_rejects_oob_dst_neuron`
- `brain_connect_rejects_zero_delay` (`"delay_ms must be positive"`)
- `brain_connect_rejects_nan_weight`

Plus ein Smoke-Test `brain_connect_accepts_valid_input` der die Happy-
Path-Kompatibilität sichert.

### Snapshot-Migration

- `snapshot_without_tau_syn_uses_default` — serialisiert ein
  Network, entfernt das `tau_syn_ms`-Feld aus dem JSON, lädt zurück.
  Erwartet: `tau_syn_ms == 5.0` (default), brain ist runnable.
  Verifiziert dass alte Snapshot-Dateien aus früheren Iterationen
  noch laden.

## Workspace-Status

- **79/79 Tests grün** (63 + 16 neue Hardening-Tests)
- **0 dead code** mehr — `Synapse.tau_syn` ist entfernt, alle
  Felder werden gelesen
- **Clippy-Lints in lib-code gefixt**: `is_none_or`, `is_multiple_of`,
  derive-Default, struct-init pattern. Test-Code-Lints sind
  teilweise belassen (clippy ist dort weniger streng zu nehmen weil
  Tests oft mit Mut zu Lesbarkeit gegenüber Idiomatik geschrieben
  sind)
- **Keine API-Bruch** für korrekte Aufrufer — falsche Aufrufer panicen
  jetzt mit klaren Fehlermeldungen statt mit kryptischen
  out-of-bounds-Stack-Traces

## Was noch offen wäre

1. **Pending-Event-Queue als BinaryHeap** — `Brain::step` macht
   aktuell O(P) drain-and-rebuild jeden Schritt. Für riesige
   Multi-Region-Systeme mit vielen Inter-Region-Edges wäre eine
   priority queue schneller. Aktuell kein Bottleneck (sparse traffic).
2. **Synapse mit eigenem τ** — falls man heterogenes synaptic decay
   für AMPA + NMDA + GABA modellieren möchte. Dann müsste τ wieder
   auf Synapse-Ebene und der decay-Loop pro Synapse statt pro
   Neuron laufen. Aktuell überflüssig.
3. **Test-Code-Clippy-Lints** — die `let mut s = X::default(); s.field = …`-
   Patterns in Tests könnten auf `..X::default()` umgestellt werden.
   Stilistisch, kein Funktionsproblem.
