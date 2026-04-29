# 33 — Docker-Stack lokal verifiziert + Snapshot-Volume

**Stand:** Follow-up zu Iteration 15. Notiz 32 war "geschrieben aber
nicht gestartet" — jetzt hat der ganze Stack einen `docker compose
up` Test hinter sich, plus ein paar Bugs sind aufgefallen und
behoben, plus ein Snapshot-Volume macht den Brain-State
restart-fest.

## Drei Bugs im ursprünglichen Dockerfile

### 1. Fehlende Bench-Stubs

Iteration 14 hat `[[bench]]`-Einträge in `crates/snn-core/Cargo.toml`
und `crates/encoders/Cargo.toml` eingefügt. Die Layer-Cache-Strategie
in der Dockerfile stub-sourced nur `lib.rs` / `main.rs`, aber:

```
error: failed to parse manifest at `…/encoders/Cargo.toml`
Caused by:
  can't find `encode_decode` bench at `benches/encode_decode.rs`
```

Cargo refuses, ein Manifest mit deklarierten `[[bench]]`/`[[bin]]`-
Targets zu parsen, deren Files nicht existieren. Fix: drei
zusätzliche `echo 'fn main() {}' > crates/.../benches/...rs` Lines.

### 2. Bind-Adresse default war loopback

`viz::main` hat `127.0.0.1:7777` hartkodiert. Im Container hört der
Server damit nur auf der Container-internen Loopback — kein
Port-Forward funktioniert.

Fix: `JAVIS_BIND_ADDR` Env-Var, default bleibt loopback (für
`cargo run` Sicherheit), Container setzt `0.0.0.0:7777`.

### 3. Static-Dir an Build-Zeit-Pfad gekoppelt

`PathBuf::from(env!("CARGO_MANIFEST_DIR"))` ist ein Build-Zeit-
Konstruktor, der im Container auf einen nicht-existenten Pfad zeigt.
Fix: `JAVIS_STATIC_DIR` Env-Var, Container setzt `/app/static`.

(2) und (3) waren schon in der ersten Notiz-32-Runde gefixt; (1) ist
jetzt auch dazu.

## Sandbox-Workaround: optionales CA-Bundle

Test-Build im Sandbox-Environment scheiterte nicht am Code, sondern
am TLS-Interception-Proxy:

```
[60] SSL peer certificate or SSH remote key was not OK
     (SSL certificate problem: self signed certificate in chain)
```

Der Container traut der Sandbox-CA nicht, weil das Standard-
ca-certificates-Bundle vom Debian-Image die nicht enthält. Fix
ist BuildKit-Secret-basiert:

```dockerfile
RUN --mount=type=secret,id=hostca,target=/tmp/hostca.crt,required=false \
    if [ -s /tmp/hostca.crt ]; then \
        cp /tmp/hostca.crt /usr/local/share/ca-certificates/sandbox-ca.crt && \
        update-ca-certificates; \
    fi \
 && cargo fetch --locked
```

`required=false` heißt: in normalen Environments (CI, Cloud, Desktop
mit echtem TLS) ist der Schritt ein No-Op. Im Sandbox kommt das
Bundle als Secret rein, wird zur Trust-Chain hinzugefügt, der
darauf folgende `cargo fetch` läuft normal durch.

`docker-compose.yml` deklariert das Secret zentral:

```yaml
secrets:
  hostca:
    file: /etc/ssl/certs/ca-certificates.crt
```

Wer das File nicht hat, kann den Eintrag rauskommentieren — der
Build läuft dann ohne Secret, der `required=false`-Pfad greift.

## Snapshot-Volume

Vor diesem Patch ging der Brain-State bei jedem `docker compose
restart` verloren. Jetzt:

```yaml
javis-viz:
  command: ["--snapshot", "/app/data/brain.snapshot.json"]
  volumes:
    - javis-data:/app/data
```

Plus im Dockerfile:

```dockerfile
RUN mkdir -p /app/data && chown javis:javis /app/data
```

Damit das Volume-Mount auf den richtigen Owner trifft (sonst kann der
Non-Root-User die Datei nicht schreiben).

## Lokaler End-to-End-Test

Mit `docker compose up --build`:

```
NAME              STATUS                    PORTS
javis-grafana     Up 27 seconds             0.0.0.0:3000->3000/tcp
javis-prometheus  Up 28 seconds             0.0.0.0:9090->9090/tcp
javis-viz         Up 28 seconds (healthy)   0.0.0.0:7777->7777/tcp
```

| Probe | Ergebnis |
| --- | --- |
| `curl localhost:7777/health` | `ok` |
| `curl localhost:7777/ready` | `{"status":"ready","sentences":3,"words":43,"llm":"mock"}` |
| Prometheus `/api/v1/targets` | `javis-viz` `up`, scrape `http://javis-viz:7777/metrics` |
| PromQL `javis_brain_words` | `43` (live-aktuell, env=local, instance=javis-viz:7777) |
| Grafana datasources | `Prometheus  prometheus  http://prometheus:9090` (auto-provisioned) |

**Snapshot-Persistenz verifiziert:**

```
# Vor restart
$ docker exec javis-viz ls -la /app/data/    →  (leer)

# nach `docker compose restart javis-viz`
$ docker exec javis-viz ls -la /app/data/
-rw-r--r-- 1 javis javis 29557339  brain.snapshot.json

# Logs zeigen save-then-load
"snapshot saved","bytes":29557339,"elapsed_ms":190
"snapshot loaded","bytes":29557339,"sentences":3,"words":43
```

29.5 MB für 3 Sätze + 43 Wörter; Brain-Topology dominiert das (1000
+ 2000 Neuronen, p=0.1 sparse Synapsen, Inter-Region-Edges).

## Image-Größe

```
javis-viz:local   148 MB   (with secret-CA + tools)
```

`debian:bookworm-slim` als Basis (~80 MB), dazu glibc-Erweiterungen
für `reqwest`/`rustls`, dann `ca-certificates` + `tini` + `curl`,
am Ende der Rust-Binary. 148 MB ist im erwartbaren Bereich für
einen async-Tokio-Server mit TLS — distroless könnte auf ~30 MB
runter, kostet aber `curl` für den Healthcheck und `ls/cat` zum
Debuggen.

## Was noch nicht passiert

- **Kein Push auf eine Registry.** Image lebt nur lokal. Nächster
  Schritt wäre `docker login ghcr.io` + `docker push
  ghcr.io/beko2210/javis-viz:0.0.1` plus eine GitHub-Action mit
  `docker/build-push-action@v6`.
- **Keine Image-Vulnerability-Scans.** `trivy image javis-viz:local`
  lokal ausführbar; CI-Integration wäre konsistent mit Iter 13
  (Supply-Chain), aber nicht jetzt.

## Status

- 3 Bugs gefixt: Bench-Stubs, optionales CA-Secret, Snapshot-Volume
- `docker compose up --build` lokal grün, alle drei Services
  healthy, Prometheus scrapt erfolgreich, Grafana auto-provisioned
- 98/98 Tests im Workspace grün, fmt clean, 0 Clippy-Warnings
