# 34 — End-to-End Sanity-Check + Grafana-UID-Bug

**Stand:** Iteration 16 (Sanity, kein neues Feature). Stack hochgefahren,
echte WebSocket-Operationen gegen den laufenden Container gefahren,
Metriken + Grafana + Snapshot-Persistenz im Live-Flow überprüft. Ein
echter Bug ist dabei aufgefallen — Grafana-Datasource-UID — und
gefixt.

## Was getestet wurde

`scripts/sanity_check.py` (neu) macht in einem Run:

1. `/ready` lesen (Anfangszustand)
2. WS `train` — neuer Satz "Lava is liquid molten rock …"
3. WS `recall query=lava` — der frisch trainierte Begriff muss wieder
   raus kommen
4. WS `recall query=rust` — Bootstrap-Vokabular muss überleben
5. WS `ask query=rust rag=… javis=…` (Mock-Modus)
6. 5 parallele `recall`-Sessions — verifiziert dass die `tokio::Mutex`
   sauber serialisiert ohne Deadlock
7. `/ready` + `/metrics` lesen, Counter / Histogramme prüfen,
   Lifetime-Token-Saving ausrechnen

Alles in einem Python-Skript, läuft in ~1 s, exit 0 = grün.

### Lokaler Run gegen `docker compose up`

```
INITIAL  /ready: sentences=3 words=43 llm=mock

--- TRAIN ---
  events: init=1  phase=3  step=150
--- RECALL 'lava' ---
  decoded candidates: ['lava']
  rag_tokens=21  javis_tokens=2  reduction=90.5%
--- RECALL 'rust' ---
  decoded candidates: ['rust']
--- ASK (mock) ---
  rag:   real=False input=43 output=23
  javis: real=False input=26 output=16
--- 5 PARALLEL RECALLS ---
  5 concurrent recalls completed in 69 ms
--- FINAL ---
  /ready: sentences=4 words=53
  ws_sessions: train=1 recall=7 ask=1
  lifetime tokens: rag=189 javis=14 saving=92.6%
  duration histogram counts: train=4 recall=7

ALL SANITY CHECKS PASSED.
```

`train_count = 4` ist 3 (Bootstrap-Korpus) + 1 (Sanity-Train);
`ws_sessions_total{action="train"} = 1` weil Bootstrap kein
WebSocket benutzt — die Counter widersprechen sich nicht, sie messen
verschiedene Sachen (Operations vs WS-Sessions).

## Bug-Fund: Grafana zeigte „Datasource not found"

In Grafana-API direkt geprüft:

```sh
$ curl -s http://admin:admin@localhost:3000/api/datasources | jq .
[{ "name": "Prometheus", "uid": "PBFA97CFB590B2093", … }]
```

Aber das Dashboard-JSON hatte:

```json
"datasource": { "type": "prometheus", "uid": "Prometheus" }
```

Die UID `"Prometheus"` (Großbuchstabe-P, der String) war meine
Annahme — Grafana ignoriert das aber und vergibt eine eigene
Hash-UID, wenn die Datasource-Provisioning kein `uid:`-Feld setzt.
Resultat: jeder Panel-Query hätte „Datasource not found" gezeigt.

Gegencheck mit der falschen UID:

```sh
$ curl -G ".../proxy/uid/Prometheus/api/v1/query" --data-urlencode 'query=javis_brain_words'
{"message":"Unable to find datasource","traceID":""}
```

### Fix

Pin die UID explizit in der Datasource-Provisioning:

```yaml
datasources:
  - name: Prometheus
    uid: prometheus     # ← neu, vorher nicht gesetzt
    type: prometheus
    …
```

Plus in allen Panels des Dashboard-JSON `"uid": "prometheus"`
(Kleinbuchstabe).

Nach `docker compose restart grafana`:

```sh
$ curl -G ".../proxy/uid/prometheus/api/v1/query" --data-urlencode 'query=javis_brain_words'
{ "result": [{ "value": [..., "53"] }] }
```

5 Panels resolved jetzt sauber, alle Queries gegen die echte
Datasource. Diese Klasse von Bug ist *genau* der Grund, warum man
solche Stack-Setups end-to-end durchprobieren muss — Tests im Workspace
hätten das nie gefangen.

## Snapshot-Persistenz im Live-Flow

Nicht nur „start/stop saves", sondern *eine echte Sequenz*:

1. Frischer Stack (volume war geleert): brain ist nur Bootstrap
2. WS-Train mit „Mountains form when …" → state in-memory
3. `docker compose restart javis-viz`
4. Server liest Snapshot, sagt: `sentences=5, words=62`
5. WS-Recalls auf rust, lava, mountains:

```
recall('rust')      → ['rust']       reduction 92.9%
recall('lava')      → ['lava']       reduction 90.5%
recall('mountains') → ['mountains']  reduction 86.7%
```

Bootstrap-Vokabular *und* live-trainierte Wörter überleben den
Restart, jeweils mit > 86% Token-Reduktion. Engram-Dictionary,
SDR-Encoder-State und Brain-Topology-Snapshot funktionieren komplett
durch.

Snapshot-Datei Größe: 29.6 MB für 5 Sätze + 62 Wörter. Brain-Topology
dominiert, nicht das Vocabulary.

## Frontend-Sanity

```
GET /             → 200 OK, content-type text/html, 9.8 KB index
GET /main.js      → 200 OK, frontend code
```

Beobachtung: Das Frontend zieht `three@0.160.0` und `3d-force-graph@1.73.4`
von `unpkg.com` (CDN). Heißt: die Backend-Container ist self-contained,
aber der Browser braucht Internet, um die 3D-Viz zu laden. Für eine
echt-Air-Gapped-Deployment müssten die JS-Libs in `crates/viz/static/`
gebundlet werden. Für das aktuelle Hobby-Demo-Projekt ist das ok,
würde aber eine Notiz im README-„Production"-Abschnitt rechtfertigen.

## Was geliefert wurde

- `scripts/sanity_check.py` (neu, ~200 Zeilen Python). Reproduzierbarer
  Smoke-Test, läuft gegen `localhost:7777` per Default, `JAVIS_HOST`
  Env-Var für andere Targets.
- `deploy/grafana/provisioning/datasources/prometheus.yml`: `uid: prometheus`
  gepinnt.
- `deploy/grafana/dashboards/javis-overview.json`: alle Panel-`uid:`
  von `"Prometheus"` auf `"prometheus"`.

## Was sonst aufgefallen ist (nicht jetzt gefixt)

| Beobachtung | Status |
| --- | --- |
| Frontend-CDN-Deps via unpkg.com | nicht air-gapped — bewusst, README-Note zeitnah |
| `train completed` log fehlt im Bootstrap | by design (Bootstrap-Trains laufen vor `tracing` init) |
| Snapshot-Datei wächst mit Brain-Größe, nicht mit Vocabulary | erwartet — Brain-Topologie ist dominant |
| `tokio::Mutex`-Serialisierung der WS-Sessions ist sauber | 5 parallel = 69 ms, kein Deadlock |

## Status

- 98/98 Tests grün, 0 Clippy, fmt clean (unverändert)
- Sanity-Skript exit 0 nach 1 Bug-Fix
- 1 echter Bug gefunden + behoben (Grafana datasource UID)
- 1 lebendige Sequenz Train-→-Restart-→-Recall durchgespielt: alles
  überlebt
