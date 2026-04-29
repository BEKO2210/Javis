# 31 — Performance-Benchmarks mit Criterion

**Stand:** Iteration 14. Davor hatten wir keine Möglichkeit, einen
Performance-Regress zu erkennen — eine Optimierung in `Network::step`
oder `encoder::encode` konnte rückgängig gemacht werden, ohne dass
jemand es merkte. Jetzt gibt es Baselines mit konkreten Zahlen, die
neue Code-Änderungen lokal vergleichen können.

## Drei Bench-Files

| File | Wo | Was |
| --- | --- | --- |
| `crates/snn-core/benches/network_step.rs` | `network_step_passive`, `network_step_stdp` | Single-Region-Step, mit/ohne STDP, Größen 100/500/1000 |
| `crates/snn-core/benches/brain_step.rs` | `brain_step_two_region` | Multi-Region-Step inkl. PendingQueue-Heap, Größen 200/500/1000 |
| `crates/encoders/benches/encode_decode.rs` | `encode_word`, `encode_sentence`, `decode_strict`, `decode_top_k` | Per-Request-Pfad: Text → SDR und SDR → Wörter, Vokabular 10/100/1000 |

Alle benutzen Criterion 0.8 mit `harness = false`. Drei `[[bench]]`-
Einträge im jeweiligen `Cargo.toml`. `criterion = { version = "0.8",
default-features = false }` — keine `rayon`-Parallelisierung, kein
`html_reports` als zwingender Featureset.

## Lokale Ausführung

```sh
# Alle drei Suiten:
cargo bench --workspace

# Einzeln:
cargo bench -p snn-core --bench network_step
cargo bench -p snn-core --bench brain_step
cargo bench -p encoders --bench encode_decode

# Schnell-Smoke (Sample-Size 10, Mess-Zeit 0.5 s):
cargo bench -p encoders -- --warm-up-time 0.3 --measurement-time 0.5 --sample-size 10
```

Criterion schreibt JSON-Roh-Daten und HTML-Reports nach
`target/criterion/`. Zwei aufeinanderfolgende Runs vergleicht es
automatisch und meldet Regressionen mit p-Werten.

## Baseline-Zahlen (lokal, Linux x86_64, Release-Profile)

| Benchmark | Zeit | Anmerkung |
| --- | ---: | --- |
| `network_step_passive/100` | < 1 µs | dichte 100-Neuron-Net |
| `network_step_passive/1000` | 3.19 µs | sparse p=0.1 |
| `network_step_stdp/1000` | 3.41 µs | + Plastizitäts-Overhead ~7 % |
| `brain_step_two_region/500` | 3.90 µs | inkl. Inter-Region-Heap |
| `brain_step_two_region/1000` | 7.65 µs | |
| `encode_sentence` | 21.5 µs | 18-Wort-Satz |
| `decode_strict/vocab_1000` | 253 µs | 200-Index Query |
| `decode_top_k/vocab_1000` | 340 µs | k=5 |

Die Zahlen sind als Größenordnungs-Anker gedacht, nicht als
Vertrag — auf anderer Hardware sind sie absolut anders, aber die
*Verhältnisse* sollten erhalten bleiben.

## CI-Job: nur compile-check

```yaml
benches:
  name: bench compile-check
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - uses: Swatinem/rust-cache@v2
    - run: cargo bench --workspace --no-run
```

`--no-run`: kompiliert die Bench-Binaries, führt sie nicht aus.
Damit fängt CI alles, was den Bench-Code kaputt macht (umbenannter
Symbol-Name in der API, geänderte Signatur), aber misst keine Zeit.

**Warum keine echten Bench-Zahlen in CI?** GitHub-Actions-Runner sind
geteilte VMs. Variance auf `cargo bench`-Werten ist auf shared-cloud-
infrastructure regelmäßig 20–40 %, oft mit Sprüngen, weil ein
Nachbar-Workload anläuft. Das wäre nutzlos für Regression-Detection
— jeder dritte Push würde False-Positiv-Alarmen.

Wenn ernsthafte Performance-Validierung kommen soll, gibt es zwei
Optionen, beide nicht jetzt:

1. Self-hosted Runner mit deaktiviertem CPU-Boost und gepinnten
   Cores, dann kann `cargo bench` mit `criterion-compare` oder
   `bencher.dev` Trends tracken.
2. `cargo-codspeed` benutzen, das spezielle Hardware bei Codspeed
   benutzt und PR-Comments mit Bench-Diffs postet. Kostenpflichtig
   für privates Repos, gratis für OSS.

Für ein Hobby-Repo mit production-Anspruch reicht der Compile-Check.

## Was die Benchmarks nicht abdecken

- **Keine `cold-start`-Messung.** Alle Benches haben Warmup-
  Schleifen vorne dran, sodass das Membrane-Potential nicht
  bei -65 mV beginnt. Production-typischer Code-Pfad mit dem ersten
  Spike-Burst ist nicht gemessen.
- **Keine End-to-End-Recall-Latenz.** Wir messen `step` und
  `decode` separat; das volle „WebSocket-Request → Decoded-Event"
  könnte zusätzliche Overheads zeigen (Mutex-Contention,
  Channel-Backpressure). Die `viz` Crate hat keine Benches; das
  wäre der nächste sinnvolle Schritt, falls Latenz-Profiling
  irgendwann kritisch wird.
- **Keine Memory-Profiles.** Criterion misst nur Wall-Time. Für
  Allocation-Counts wäre `dhat` oder `heaptrack` das Werkzeug.

## Status

- 3 Bench-Files, 7 separate Bench-Funktionen, 3 verschiedene Größen-
  Skalierungen
- `cargo bench --workspace --no-run` lokal: alle compilen, alle
  laufen tatsächlich durch
- `cargo bench --workspace -- --measurement-time 0.5 --sample-size 10`
  liefert die obigen Baseline-Zahlen in ~30 s
- Neuer CI-Job `benches` (Compile-Check, nicht-blocking-mäßig
  schnell)
- 98/98 Tests grün, 0 Clippy-Warnings, fmt clean
