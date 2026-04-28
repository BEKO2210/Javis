// Javis live visualisation — Iteration 2: 3D brain.
//
// Two anatomical "lobes" arranged in space: R1 (input cortex) on the
// left, R2 (memory cortex) on the right with inhibitory neurons
// embedded inside it. Spikes paint their neuron bright for ~200 ms and
// fade back to base colour. Wire format is unchanged from iteration 1.

const $ = (sel) => document.querySelector(sel);
const log = (msg) => {
  const el = $("#log");
  const line = document.createElement("div");
  line.textContent = msg;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
};

// ---------------------------------------------------------------------------
// Brain layout
// ---------------------------------------------------------------------------

const COLORS = {
  r1: new THREE.Color("#62d6ff"),
  r2e: new THREE.Color("#ffd166"),
  r2i: new THREE.Color("#ff5c8a"),
  spike: new THREE.Color("#ffffff"),
};

// Centres and radii of the two anatomical lobes.
const LAYOUT = {
  r1: { cx: -260, cy: 0, cz: 0, radius: 70 },
  r2: { cx: 140, cy: 0, cz: 0, radius: 130 },
};

// Fibonacci-sphere distribution: gives an even, organic-looking
// scatter on the surface of a sphere.
function fibSpherePoint(idx, total, radius, jitter = 0) {
  const phi = Math.acos(1 - (2 * (idx + 0.5)) / total);
  const theta = Math.PI * (1 + Math.sqrt(5)) * idx;
  const r = radius * (1 - jitter * Math.random());
  return {
    x: r * Math.sin(phi) * Math.cos(theta),
    y: r * Math.sin(phi) * Math.sin(theta),
    z: r * Math.cos(phi),
  };
}

// Each region carries its own colour and centre; we pre-compute fixed
// positions so the lobes hold their shape (no force-layout drift).
function buildGraph(init) {
  const nodes = [];
  for (let i = 0; i < init.r1_size; i++) {
    const p = fibSpherePoint(i, init.r1_size, LAYOUT.r1.radius, 0.05);
    nodes.push({
      id: `r1-${i}`,
      region: "r1",
      fx: LAYOUT.r1.cx + p.x,
      fy: LAYOUT.r1.cy + p.y,
      fz: LAYOUT.r1.cz + p.z,
    });
  }
  for (let i = 0; i < init.r2_excitatory; i++) {
    const p = fibSpherePoint(i, init.r2_excitatory, LAYOUT.r2.radius, 0.04);
    nodes.push({
      id: `r2e-${i}`,
      region: "r2e",
      fx: LAYOUT.r2.cx + p.x,
      fy: LAYOUT.r2.cy + p.y,
      fz: LAYOUT.r2.cz + p.z,
    });
  }
  // I-cells on a smaller inner shell — anatomically interneurons sit
  // inside excitatory tissue.
  for (let i = 0; i < init.r2_inhibitory; i++) {
    const p = fibSpherePoint(
      i,
      init.r2_inhibitory,
      LAYOUT.r2.radius * 0.55,
      0.15,
    );
    nodes.push({
      id: `r2i-${i}`,
      region: "r2i",
      fx: LAYOUT.r2.cx + p.x,
      fy: LAYOUT.r2.cy + p.y,
      fz: LAYOUT.r2.cz + p.z,
    });
  }
  return { nodes, links: [] };
}

// ---------------------------------------------------------------------------
// 3D rendering
// ---------------------------------------------------------------------------

let Graph = null;
let nodeById = new Map(); // id -> { node, mat, base, lastSpike }

function initBrain(init) {
  const data = buildGraph(init);

  if (Graph) {
    // Tear the old graph down so a re-init (new query) starts fresh.
    Graph._destructor && Graph._destructor();
    $("#brain3d").innerHTML = "";
  }
  nodeById = new Map();

  Graph = ForceGraph3D()(document.getElementById("brain3d"))
    .backgroundColor("#04060c")
    .showNavInfo(false)
    .nodeRelSize(2)
    .nodeOpacity(0.92)
    .graphData(data)
    .nodeThreeObject((node) => {
      const base = COLORS[node.region].clone();
      const radius = node.region === "r2i" ? 1.3 : 1.6;
      const geom = new THREE.SphereGeometry(radius, 8, 6);
      const mat = new THREE.MeshBasicMaterial({
        color: base.clone(),
        transparent: true,
        opacity: 0.85,
      });
      const mesh = new THREE.Mesh(geom, mat);
      nodeById.set(node.id, {
        node,
        mat,
        base: base.clone(),
        lastSpike: -Infinity,
      });
      return mesh;
    })
    .nodeThreeObjectExtend(false);

  // Camera framing.
  Graph.cameraPosition({ x: 0, y: 80, z: 480 }, { x: 0, y: 0, z: 0 }, 0);

  // Disable charge-force so positions stay where we set them.
  const fg = Graph.d3Force("charge");
  if (fg) fg.strength(0);
  const link = Graph.d3Force("link");
  if (link) link.strength(0);
}

// Spike animation tick — runs every animation frame, decays the glow on
// every node we've recently illuminated.
const SPIKE_DECAY_MS = 220;

function animateSpikes() {
  const now = performance.now();
  for (const entry of nodeById.values()) {
    if (entry.lastSpike < 0) continue;
    const age = now - entry.lastSpike;
    if (age > SPIKE_DECAY_MS) {
      entry.mat.color.copy(entry.base);
      entry.mat.opacity = 0.85;
      entry.lastSpike = -1;
      continue;
    }
    const t = 1 - age / SPIKE_DECAY_MS;
    entry.mat.color.copy(entry.base).lerp(COLORS.spike, t);
    entry.mat.opacity = 0.85 + 0.15 * t;
  }
  requestAnimationFrame(animateSpikes);
}
requestAnimationFrame(animateSpikes);

function flashNode(id) {
  const entry = nodeById.get(id);
  if (!entry) return;
  entry.lastSpike = performance.now();
}

// ---------------------------------------------------------------------------
// Live stats
// ---------------------------------------------------------------------------

let recentR1 = 0;
let recentR2 = 0;
let lastSampleMs = 0;
let simTms = 0;

function tickStats(t_ms, r1Count, r2Count) {
  recentR1 += r1Count;
  recentR2 += r2Count;
  simTms = t_ms;
  $("#sim-t").textContent = `${Math.round(t_ms)} ms`;
  if (simTms - lastSampleMs >= 250) {
    const window_s = (simTms - lastSampleMs) / 1000;
    const r1ps = window_s > 0 ? Math.round(recentR1 / window_s) : 0;
    const r2ps = window_s > 0 ? Math.round(recentR2 / window_s) : 0;
    $("#rate-r1").textContent = r1ps.toLocaleString();
    $("#rate-r2").textContent = r2ps.toLocaleString();
    recentR1 = 0;
    recentR2 = 0;
    lastSampleMs = simTms;
  }
}

// ---------------------------------------------------------------------------
// Event handling
// ---------------------------------------------------------------------------

let r2eSize = 0;

function onStep(ev) {
  // R1 indices are global within R1.
  for (const id of ev.r1) flashNode(`r1-${id}`);
  // R2 indices are global within R2 (excitatory first, then inhibitory).
  for (const id of ev.r2) {
    if (id < r2eSize) flashNode(`r2e-${id}`);
    else flashNode(`r2i-${id - r2eSize}`);
  }
  tickStats(ev.t_ms, ev.r1.length, ev.r2.length);
}

// Last decoded payload, kept around so the "Send to LLM" button can
// fire an `ask` request without re-running recall.
let lastDecoded = null;

function onDecoded(ev) {
  const decodedDiv = $("#decoded");
  decodedDiv.innerHTML = "";
  if (ev.candidates.length === 0) {
    decodedDiv.textContent = "no concepts above threshold";
  } else {
    for (const c of ev.candidates) {
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = `${c.word} · ${c.score.toFixed(2)}`;
      decodedDiv.appendChild(pill);
    }
  }
  $("#reduction").textContent = `−${ev.reduction_pct.toFixed(1)}%`;
  $("#rag-tok").textContent = `${ev.rag_tokens} tokens`;
  $("#javis-tok").textContent = `${ev.javis_tokens} tokens`;
  $("#rag-text").textContent = ev.rag_payload || "—";
  $("#javis-text").textContent = ev.javis_payload || "—";

  lastDecoded = {
    query: ev.query,
    rag: ev.rag_payload || "",
    javis: ev.javis_payload || "",
  };
  $("#send-llm").disabled = false;
  // Fresh recall hides any previous LLM answer pair.
  $("#llm-pair").hidden = true;

  log(
    `decoded query="${ev.query}" → ${ev.candidates.length} candidates, ` +
      `reduction ${ev.reduction_pct.toFixed(1)}%`,
  );
}

function onAsked(ev) {
  $("#llm-pair").hidden = false;
  $("#rag-answer").textContent = ev.rag.text;
  $("#javis-answer").textContent = ev.javis.text;
  $("#rag-llm-tok").textContent =
    `in ${ev.rag.input_tokens} · out ${ev.rag.output_tokens}`;
  $("#javis-llm-tok").textContent =
    `in ${ev.javis.input_tokens} · out ${ev.javis.output_tokens}`;
  $("#llm-real").textContent = ev.rag.real ? "yes (Anthropic)" : "no (mock)";
  log(
    `asked "${ev.question}" → RAG ${ev.rag.input_tokens}→${ev.rag.output_tokens} ` +
      `· Javis ${ev.javis.input_tokens}→${ev.javis.output_tokens}`,
  );
}

function handleEvent(ev) {
  switch (ev.type) {
    case "init":
      r2eSize = ev.r2_excitatory;
      initBrain(ev);
      log(
        `init R1=${ev.r1_size}  R2=${ev.r2_size} ` +
          `(${ev.r2_excitatory}E + ${ev.r2_inhibitory}I)`,
      );
      break;
    case "phase":
      $("#phase").textContent = `${ev.name} — ${ev.detail}`;
      log(`phase: ${ev.name} (${ev.detail})`);
      break;
    case "step":
      onStep(ev);
      break;
    case "decoded":
      onDecoded(ev);
      break;
    case "asked":
      onAsked(ev);
      break;
    case "done":
      log("session done");
      break;
  }
}

// ---------------------------------------------------------------------------
// WebSocket session
// ---------------------------------------------------------------------------

let socket = null;

function startAction(action, paramName, paramValue) {
  if (socket) {
    socket.close();
    socket = null;
  }
  if (action === "recall") {
    $("#decoded").textContent = "—";
    $("#reduction").textContent = "—";
    $("#rag-tok").textContent = "—";
    $("#javis-tok").textContent = "—";
    $("#rag-text").textContent = "—";
    $("#javis-text").textContent = "—";
  }
  $("#phase").textContent = "connecting…";
  recentR1 = 0;
  recentR2 = 0;
  lastSampleMs = 0;
  simTms = 0;

  let url = `ws://${location.host}/ws?action=${action}`;
  if (paramName && paramValue) {
    url += `&${paramName}=${encodeURIComponent(paramValue)}`;
  }
  socket = new WebSocket(url);
  socket.onopen = () =>
    log(`ws ${action}${paramValue ? ` "${paramValue}"` : ""}`);
  socket.onclose = () => log("ws closed");
  socket.onerror = () => log("ws error");
  socket.onmessage = (msg) => {
    try {
      handleEvent(JSON.parse(msg.data));
    } catch (e) {
      /* ignore malformed frames */
    }
  };
}

$("#train-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("#train-text").value.trim();
  if (!text) return;
  startAction("train", "text", text);
  $("#train-text").value = "";
});

$("#recall-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#recall-text").value.trim();
  if (!q) return;
  startAction("recall", "query", q);
});

$("#reset").addEventListener("click", () => {
  startAction("reset");
  $("#log").innerHTML = "";
  lastDecoded = null;
  $("#send-llm").disabled = true;
  $("#llm-pair").hidden = true;
});

$("#send-llm").addEventListener("click", () => {
  if (!lastDecoded) return;
  if (socket) {
    socket.close();
    socket = null;
  }
  $("#phase").textContent = "asking the LLM…";
  $("#rag-answer").textContent = "…";
  $("#javis-answer").textContent = "…";
  $("#llm-pair").hidden = false;
  let url =
    `ws://${location.host}/ws?action=ask` +
    `&query=${encodeURIComponent(lastDecoded.query)}` +
    `&rag=${encodeURIComponent(lastDecoded.rag)}` +
    `&javis=${encodeURIComponent(lastDecoded.javis)}`;
  socket = new WebSocket(url);
  socket.onopen = () => log(`ws ask "${lastDecoded.query}"`);
  socket.onclose = () => log("ws closed");
  socket.onerror = () => log("ws error");
  socket.onmessage = (msg) => {
    try {
      handleEvent(JSON.parse(msg.data));
    } catch (e) {
      /* ignore */
    }
  };
});

// Default: an immediate recall against the bootstrap-trained corpus.
startAction("recall", "query", "rust");

// Keep `startSession` as a thin alias so older callers keep working.
function startSession(query) {
  startAction("recall", "query", query);
}
