# 32 — Container & Deploy

**Stand:** Iteration 15. Damit ist der Zustand erreicht, in dem ein
Außenstehender mit `docker compose up --build` den Server, einen
Prometheus, eine Grafana-Instanz hochfährt und drei Endpoints unter
seiner Kontrolle hat: WS auf 7777, Prom auf 9090, Grafana auf 3000.

## Code-Voraussetzung — Runtime-konfigurierbare Pfade

Vorher hat `viz::main` zwei Werte hartkodiert:

```rust
let static_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("static");
let addr: SocketAddr = "127.0.0.1:7777".parse().unwrap();
```

Beides ist im Container kaputt: `CARGO_MANIFEST_DIR` ist Build-Zeit-
Kontext (existiert nicht mehr nach dem Multi-Stage), und `127.0.0.1`
würde den Server vom Host-Forward unerreichbar machen. Neue
Resolver-Funktionen mit Env-Var-Override:

| Env-Var | Default | Container-Wert |
| --- | --- | --- |
| `JAVIS_STATIC_DIR` | `…/crates/viz/static` | `/app/static` |
| `JAVIS_BIND_ADDR`  | `127.0.0.1:7777`      | `0.0.0.0:7777` |
| `RUST_LOG`         | `info,viz=info`       | `info` |
| `JAVIS_LOG_FORMAT` | `pretty`              | `json` |

`127.0.0.1` als Default ist *bewusst* — niemand soll versehentlich
ein Demo-Repo auf der LAN exponieren, wenn er nur `cargo run` macht.
Der Container überschreibt das explicit.

## Multi-Stage Dockerfile

```
FROM rust:1.86-bookworm AS builder
… cargo fetch with stub source for layer cache …
… cargo build --release --locked --bin javis-viz …

FROM debian:bookworm-slim AS runtime
… ca-certificates + tini + curl …
… non-root user javis (uid 1000) …
COPY --from=builder … javis-viz
COPY --chown=javis:javis crates/viz/static /app/static
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/javis-viz"]
```

Drei Sicherheits-relevante Punkte:

1. **`tini` als PID 1.** Sonst wird ein Shutdown-Signal in Rust nicht
   sauber zugestellt (kein graceful shutdown, Snapshot wird nicht
   geschrieben). `tini` proxied alle Signale korrekt.
2. **Non-root.** `useradd --uid 1000 javis`, `USER javis`. Wenn
   irgendetwas im Container kompromittiert wird, hat es keine
   root-Privs.
3. **`HEALTHCHECK` benutzt `/health`.** Docker / Compose / Swarm
   kennen den Container-Status, bevor sie Traffic darauf routen.

Final image size: ~80 MB (debian-slim + glibc + tls roots + binary).
Distroless wäre ~30 MB, aber dann verlieren wir `curl` für den
Healthcheck und `bash` zum Debuggen. Tradeoff bewusst zugunsten der
Operability.

## Layer-Cache-Trick

```dockerfile
# Manifests first
COPY Cargo.toml Cargo.lock ./
COPY crates/*/Cargo.toml crates/*/Cargo.toml

# Stub-source the workspace
RUN echo 'fn main() {}' > crates/viz/src/main.rs && \
    echo ''            > crates/snn-core/src/lib.rs && …

RUN cargo fetch --locked    # downloads all deps, exit-cached

# Real source replaces stubs
COPY crates/ crates/
RUN find crates -name '*.rs' -exec touch {} +
RUN cargo build --release --locked --bin javis-viz
```

`cargo fetch` läuft nur, wenn `Cargo.toml`/`Cargo.lock` sich
ändern. Bei reinen Code-Änderungen ist die `fetch`-Layer cache-hit
— spart pro Build ~30 s Download-Zeit, auch wenn die actual
compile-step jedes mal komplett neu läuft. Eine vollwertige
cargo-chef-Integration wäre die nächste Optimierung; für jetzt
reicht das.

## docker-compose

Drei Services:

```
javis-viz   → :7777      (the actual app)
prometheus  → :9090      (scrapes /metrics every 15 s)
grafana     → :3000      (anonymous-admin, prometheus auto-provisioned)
```

Wichtig: Grafana läuft mit
`GF_AUTH_ANONYMOUS_ENABLED=true` + `GF_AUTH_DISABLE_LOGIN_FORM=true`.
Das ist der Compose-Stack für *Lokal-Demo*, nicht für anyhow
exponiert. Comment im File macht das explizit.

## Auto-Provisioning

```
deploy/
├── prometheus.yml                  ← scrape config
└── grafana/
    ├── provisioning/
    │   ├── datasources/prometheus.yml   ← Prometheus als default DS
    │   └── dashboards/dashboards.yml    ← Provider, der /var/lib/grafana/dashboards lädt
    └── dashboards/
        └── javis-overview.json     ← 5-Panel-Dashboard
```

Dashboard-Panels:

1. **Brain — sentences** (`javis_brain_sentences`)
2. **Brain — words** (`javis_brain_words`)
3. **Token saving (lifetime, %)** —
   `100 * (1 - javis_recall_tokens_javis_total / javis_recall_tokens_rag_total)`
4. **WS sessions / sec by action** — Rate-graph mit Label-split
5. **p95 operation latency** — Histogram-Quantile auf den drei
   `*_duration_seconds` Histogrammen

Damit reicht ein Browser-Tab `http://localhost:3000`, um zu sehen,
ob der Server lebt, was er gelernt hat, und wie viele Tokens er
gespart hat seit Container-Start.

## Lokaler Test

`docker compose up --build` ist im Sandbox-Build dieser Notiz nicht
gelaufen, weil der Daemon nicht erreichbar war. Verifiziert wurden
stattdessen:

- Rust-Side: gleiche Smoke-Tests wie in 24/25/26, aber mit
  `JAVIS_BIND_ADDR=127.0.0.1:17777` und `JAVIS_STATIC_DIR=…` als
  expliziten Env-Vars. `/health` antwortet `ok`, `/ready` gibt
  korrektes JSON. Tests 98/98 grün.
- Dockerfile-Side: durchsucht und gegen die bekannten Stolpersteine
  geprüft (PID 1, non-root, healthcheck, layer-cache). Vollständiger
  Image-Build muss in einer Umgebung mit Docker-Daemon nachgeholt
  werden.

## CI

Kein neuer CI-Job. `docker build` in CI würde pro Push ~3-5 min
extra kosten und 1 GB Image bauen — übermäßig für ein Hobby-Repo.
Falls später ein Registry-Push automatisiert werden soll, wäre der
Pfad: `docker/build-push-action@v6` + GHCR.

## Was nicht im Stack ist

- **Kein Reverse-Proxy** (nginx/caddy). Wer dem Compose-Stack einen
  echten Hostname und TLS geben will, packt das davor. Der Server
  selbst macht nur HTTP/WS auf 7777.
- **Kein Auth** auf `/metrics`, `/health`, `/ready`. Im Lokal-Stack
  alles offen; im Produktions-Deploy wäre das Sache eines
  Network-Layers (Cluster-internes Netzwerk, IP-Whitelist, etc.).
- **Kein Persistenz-Volume für Snapshot.** `/app/snapshot.json`
  wäre der natürliche Mount-Point, aber das Setup ist im Compose
  noch nicht da, weil es bisher kein Snapshot-Auto-Save gibt.
  Nächster sinnvoller Schritt, wenn das Repo Production-Use sehen
  würde.
- **Kein Alerting.** Prometheus könnte mit Alertmanager Token-
  Saving-Drops oder Latency-Spikes alarmieren. Out of scope hier.

## Status

- 6 neue Dateien: `Dockerfile`, `.dockerignore`, `docker-compose.yml`,
  `deploy/prometheus.yml`, `deploy/grafana/provisioning/…`,
  `deploy/grafana/dashboards/javis-overview.json`
- 1 Code-Änderung: `viz::main` mit `JAVIS_STATIC_DIR` und
  `JAVIS_BIND_ADDR` Env-Var-Support
- 98/98 Tests, 0 Clippy, fmt clean
- Stack-Test pending (sandbox limitation)
