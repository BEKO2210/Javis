# 39 — Profile zuerst, dann optimieren: 1.5× auf `step_immutable`

**Stand:** Iteration 21. Notiz 38 hat den `Mutex`-Bottleneck gelöst,
das Server-Plateau lag danach bei ~9 ms pro Recall, hardware-bound.
Die naheliegende Frage: *wo* in den 9 ms? Bevor irgendjemand SIMD-
Intrinsics anfasst, ein ehrlicher Profile-Run.

## Profiler-Stack

`perf` ist auf der Sandbox-VM nicht installiert (nicht-systemd-init,
linux-tools-paket nicht im Mirror). `cargo-flamegraph` läuft, aber
braucht perf als Backend. `valgrind --tool=callgrind` würde
funktionieren, aber instrumentiert mit ~50× Overhead — die
Phasen-Verhältnisse sind dann verzerrt.

Stattdessen: hand-instrumented Profiler im
`crates/snn-core/examples/profile_step_immutable.rs`. Drei
`Instant::now()`-Klammern um die drei Phasen von `step_immutable`,
5 000 Steps Tight Loop, Mean / p50 / p99 + Anteile am Total.

## Profile-Befund (2000 Neuronen, 200 angetrieben, 0.91 Spikes/Step)

```
              mean ns        p50         p99       share
  decay         300         256         414      3.8%
  lif          7132        6785       16779     91.8%
  deliver       342         326        1627      4.4%
  ----------------------------------------------------
  step total   7766 ns                          100.0%
```

**LIF-Integration ist der Bottleneck mit 92 %.** Decay-Loops sind
schon vom Auto-Vectorizer SIMD-isiert (3 Multiplikations-Loops über
2000 f32 in 300 ns = ~22 Multiplikationen pro Cache-Line, klar
memory-bandwidth-limited). Spike-Delivery ist billig, weil bei 0.91
Spikes/Step nur ~180 outgoing-edges pro Step gewalkt werden.

## Was die LIF-Loop teuer machte

Originaler Code hatte pro Iteration:
- `external.get(idx).copied().unwrap_or(0.0)` — bounds-check + Option
- `state.i_syn[idx]` — bounds-check
- `state.i_syn_nmda.get(idx)` — Option, kann `None` (empty buffer)
- `state.i_syn_gaba.get(idx)` — dito
- 4 weitere Vec-Indizierungen mit jeweils bounds-check
- 2 Branches (refractory + threshold)

Die `Option::get()`-Pattern verhindern, dass der Auto-Vectorizer
die Loop in SIMD-Lanes zerlegt — er weiß nicht, ob die Buffer-
Length konstant ist über die Loop hinweg.

## Optimierung: Pre-Summed Channel Buffer

```rust
state.total_input: Vec<f32>  // ← neu, in NetworkState

match (ext_full, nmda_active, gaba_active) {
    (true, false, false) => {
        for i in 0..n { total_input[i] = external[i] + i_syn[i]; }
    }
    (true, true, false) => {
        for i in 0..n { total_input[i] = external[i] + i_syn[i] + nmda[i]; }
    }
    // ... 4 specialised inner loops, 1 slow fallback ...
}
```

Vor dem LIF-Loop: einmalige 3-Wege-Fall-Unterscheidung; danach ein
straight-line Loop ohne `Option`-Branches. Die LIF-Loop selbst liest
nur noch eine Slice (`total_input[idx]`) statt drei.

## Bench-Ergebnis

Criterion-Lauf gegen den vor-Optimierungs-Baseline:

```
network_step_immutable/100   time: [305 ns 307 ns 308 ns]  -44%
network_step_immutable/500   time: [1.56 µs 1.66 µs 1.81 µs]  -41%
network_step_immutable/1000  time: [3.58 µs 3.73 µs 3.95 µs]  -36%
network_step_immutable/2000  time: [9.16 µs 9.32 µs 9.51 µs]  -33%
                                                    p < 0.05
```

| Größe | Vorher | Nachher | Speedup |
| ---: | ---: | ---: | ---: |
| 100  | 545 ns | 307 ns | **1.78×** |
| 500  | 2.81 µs | 1.66 µs | 1.69× |
| 1000 | 5.85 µs | 3.73 µs | 1.57× |
| 2000 | 13.79 µs | 9.32 µs | 1.48× |

p < 0.05 in jedem Fall. Statistically significant, nicht-trivial groß.

Der Speedup nimmt mit n leicht ab — bei 2000 Neuronen wird die
LIF-State-Daten (v + refr + last + params, ~48 B/Neuron = 96 KB) zu
groß für L1, dann limitiert L2-Bandwidth statt CPU. Auto-Vec hilft
am meisten, wenn die Daten in L1 passen.

## Was Iter 21 explizit *nicht* tut

- **Keine `std::simd`-Intrinsics.** Die Auto-Vectorisierung des
  LLVM-Compilers war deutlich besser als ich nach dem ersten Profile
  erwartet hatte — der Pre-Sum-Buffer reicht aus, um 1.5× zu holen,
  ohne Rust-nightly oder Crate-Dependencies. SIMD-Intrinsics sind
  für eine zukünftige Iteration sinnvoll, falls dann noch mehr
  rauszuholen ist (data-layout-Refactor von `LifNeuron`).
- **Keine LifNeuron-AoSoA.** `LifNeuron` hat nach wie vor `params`
  und transientes `v/refr/last/activity_trace` in einer Struct.
  Eine AoS→SoA-Migration würde Cache-Lokalität weiter verbessern,
  ist aber ein invasiver Refactor mit großem Test-Surface. Bei
  50% Speedup als „kleine" Optimierung fühlt sich AoSoA out of
  scope an, bis es nochmal eng wird.
- **Pipeline-Latenz** (encoder, decoder, RAG-Search, JSON-Serde,
  Channel-Send) wurde nicht angefasst. Der Server-Mean von ~10 ms
  in der Docker-Load-Test-Umgebung ist daher gleich geblieben — die
  ~2 ms Step-Sparung verschwinden in Run-to-Run-Variance der
  shared-VM. Der saubere Win lebt in den Criterion-Zahlen.

## Methodik-Lektion

ChatGPT hatte in der Vorab-Diskussion gefragt: *"Hast du schon einen
Profiler über den neuen Read-Only-Pfad laufen lassen, um zu
bestätigen, ob wirklich die LIF-Mathematik das Limit ist, oder
vielleicht doch Cache-Misses beim Iterieren durch die Adjazenzlisten?"*

Genau die richtige Frage. Wir hätten direkt zu SIMD-Intrinsics
gegriffen und ungefähr dasselbe Ergebnis gehabt — aber 200 Zeilen
Unsafe-Code, eine `wide`-Crate-Dependency, und einen Wartungs-Tax
für Operations, die das nicht lesen wollen. Stattdessen: einen
einzigen Pre-Sum-Buffer und 4 kleine Spezialisierungs-Loops.

## Status

- 108/108 Tests grün (equivalence-Tests bestätigen, dass die Spike-
  Outputs immer noch bit-identisch zum mutating-Pfad sind)
- 0 Clippy-Warnings, fmt clean
- 1.5× Speedup auf `Network::step_immutable` (p < 0.05, alle
  Größenstufen)
- Profile + Bench + Optimierung dokumentiert für die nächsten
  Iterationen
