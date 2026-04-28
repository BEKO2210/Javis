# 07 — Homeostatische synaptische Skalierung

**Stand:** Gegenmittel zum Generalisierungs-Bleeding aus Notiz 06. Eine
zweite, langsame Plastizitätsregel hält die mittlere Feuerrate jedes
Neurons in einem Zielbereich, ohne die von STDP gelernten *relativen*
Gewichtsmuster zu zerstören.

## Idee

STDP entscheidet **welche** Synapsen ein Pattern kodieren. Homöostase
entscheidet **wie viel Gesamtdrive** ein Neuron langfristig akzeptiert.
Die Regel ist *multiplikativ* — alle eingehenden exzitatorischen
Gewichte eines Post-Neurons werden mit demselben Faktor skaliert. Damit
bleibt das Verhältnis zwischen ihnen mathematisch erhalten, während die
absolute Größenordnung der Feuerrate folgt.

## Implementierung

### Activity Trace im `LifNeuron`

Neue Zustandsvariable `activity_trace: f32`:
- pro Spike: `+1.0`
- pro Schritt: `*= exp(-dt / tau_homeo_ms)`

Equilibrium unter konstanter Rate `r` Hz:
`A_trace ≈ r · tau_homeo / 1000`. Mit `tau_homeo = 2000 ms` heißt
`a_target = 5` ein Ziel von ~2.5 Hz.

### `HomeostasisParams`

```rust
pub struct HomeostasisParams {
    pub eta_scale: f32,      // 0.0 = aus
    pub a_target: f32,
    pub tau_homeo_ms: f32,
    pub apply_every: u32,    // Skalierungs-Pass alle N Schritte
}
```

Default: `eta_scale = 0.0` → Homöostase ist aus, sofern nicht explizit
aktiviert. Damit bleiben alle bisherigen Tests unverändert grün.

### `Network`-Integration

- Pro Schritt: Trace-Decay + Spike-Inkrement (nur wenn aktiviert).
- Alle `apply_every` Schritte: `apply_synaptic_scaling(...)` läuft. Pro
  Post-Neuron `i`:

  ```
  factor_i = max(0, 1 + eta * (A_target - A_trace_i))
  w_ij     = clamp(w_ij * factor_i, w_min, w_max)   für excitatorische pre j
  ```

- Inhibitorische incoming Synapsen werden *nicht* skaliert. Das bewahrt
  die Stabilitäts-Garantie der E/I-Balance-Mechanik aus Notiz 03.
- `factor.max(0.0)`: Schutz gegen pathologisch hohe Traces. Ohne den
  Guard würde ein extrem hyperaktives Neuron einen negativen Faktor
  produzieren, der durch die Clamp-Untergrenze alle eingehenden Gewichte
  uniform auf 0 setzen würde — und damit das *relative* Pattern
  zerstören. Der Guard hält das Verhalten in jedem Regime sauber.

### Datenstrukturen

Wie beim Rest des Cores: flach, `Vec<…>`-basiert. `activity_trace`
sitzt in `LifNeuron` (parallel zu `v`, `refractory_until`),
`step_counter: u64` in `Network`. GPU-portierungs-fest.

## Tests (drei neue, alle grün)

### `homeostasis_off_by_default_does_not_touch_weights`

Vertrag: ohne `enable_homeostasis` und ohne STDP bewegt sich kein
Gewicht — egal wie aktiv Neuronen sind. Genau dieses Verhalten schützt
die 19 vorherigen Tests vor Regression.

### `homeostasis_scales_down_hyperactive_neuron`

Ein einziges Post-Neuron, durch konstanten 5-nA-Strom fortwährend
gefeuert. Mit aggressivem `eta_scale = 0.01`, `a_target = 1.0`,
`tau_homeo = 500 ms`. Nach 5 simulierten Sekunden:

```
post.activity_trace = 45.4   (etwa 90 Hz, weit über target)
weight: 1.0 → 1.0e-45         (effektiv null)
```

Das Neuron wird sich selbst in eine Hülle aus stummen Synapsen
einsperren — genau die gewünschte Regulation.

### `homeostasis_preserves_relative_weights`

Identisches Setup, aber zwei eingehende Synapsen mit `w₁ = 2.0` und
`w₂ = 0.5` (Verhältnis 4 : 1). Sanftes `eta_scale = 0.0001`, damit der
Faktor numerisch stabil bleibt (kein f32-Underflow).

```
strong: 2.0   → 0.272   (10× kleiner)
weak:   0.5   → 0.068   (10× kleiner)
ratio:  4.000 → 4.000   (drift = 0.00e0, exakt)
```

Multiplikative Skalierung in Reinkultur: beide Gewichte verlieren
denselben Anteil, das von STDP gelernte Verhältnis ist immun.

## Wirkung in der Pipeline

Damit haben wir die zwei nötigen Plastizitäts-Skalen sauber getrennt:

| Ebene        | Mechanismus | Zeitkonstante  | Was sie ändert                  |
| ------------ | ----------- | -------------- | ------------------------------- |
| schnell      | STDP        | ~20 ms (τ±)    | *welche* Synapse stark wird     |
| langsam      | Homöostase  | ~2000 ms       | *wie viel* Gesamtdrive zulässig |

Im Pattern-Completion-Test aus Notiz 06 ist Homöostase noch aus —
deswegen das Bleeding (target_assembly=461, post_recall=827). Mit
eingeschalteter Homöostase erwarten wir, dass post_recall sich der
Größe von target_assembly nähert: das Engramm bleibt scharf, das Netz
sparsam.

## Nächste Schritte

1. **Pattern-Completion-Test mit Homöostase** — gleiches Protokoll
   wie 06, aber `eta_scale > 0` in R2. Ziel: post_recall ≤ 1.5 ×
   target_assembly bei gleicher Coverage.
2. **Recall-Decoder** — aktivierte R2-Neuronen → SDR → Text-Kandidaten.
3. **Token-Effizienz-Messung** — endgültige Validierung des
   Javis-Versprechens.
