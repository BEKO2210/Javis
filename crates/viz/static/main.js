// Javis live visualisation.
// Iteration 1: 2D spike raster on canvas + live stats. Iteration 2 will
// swap the canvas for a 3D brain (Three.js / 3d-force-graph).

const $ = (sel) => document.querySelector(sel);
const log = (msg) => {
  const el = $("#log");
  const line = document.createElement("div");
  line.textContent = msg;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
};

// Canvas spike raster -------------------------------------------------------

const canvas = $("#raster");
const ctx = canvas.getContext("2d");

let totalNeurons = 1; // updated on init
let r1Size = 1;
let r2eSize = 1;
let r2iSize = 1;
let firstR2 = 0;
let firstR2i = 0;

let pixelRatio = 1;

function resizeCanvas() {
  pixelRatio = window.devicePixelRatio || 1;
  const stage = $("#stage");
  const w = stage.clientWidth;
  const h = stage.clientHeight;
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  canvas.width = Math.floor(w * pixelRatio);
  canvas.height = Math.floor(h * pixelRatio);
  ctx.fillStyle = "#0a0e18";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

let xCursor = 0;
let xWindowMs = 600; // visible time window
let pxPerMs = 0;

function setupRaster(init) {
  r1Size = init.r1_size;
  r2eSize = init.r2_excitatory;
  r2iSize = init.r2_inhibitory;
  totalNeurons = r1Size + r2eSize + r2iSize;
  firstR2 = r1Size;
  firstR2i = r1Size + r2eSize;
  ctx.fillStyle = "#0a0e18";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  pxPerMs = canvas.width / xWindowMs;
  xCursor = 0;
}

function plotSpikes(t_ms, r1, r2) {
  // Fade older content slightly.
  ctx.fillStyle = "rgba(10, 14, 24, 0.04)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const x = Math.floor(xCursor * pxPerMs);
  // wrap
  if (xCursor >= xWindowMs) {
    ctx.fillStyle = "#0a0e18";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    xCursor = 0;
  }
  const dotR1 = "#62d6ff";
  const dotR2e = "#ffd166";
  const dotR2i = "#ff5c8a";

  const xpos = Math.floor(xCursor * pxPerMs);
  const dot = 2 * pixelRatio;
  // R1 band: top third
  for (const id of r1) {
    const y = Math.floor((id / r1Size) * canvas.height * 0.30);
    ctx.fillStyle = dotR1;
    ctx.fillRect(xpos, y, dot, dot);
  }
  // R2-E band: middle, R2-I band: bottom
  for (const id of r2) {
    let y;
    if (id < firstR2i - firstR2) {
      // excitatory portion of r2 (id is local to r2)
    }
  }
  // Brain reports r2 indices as global within R2 (0..R2_N).
  // We split E (first 80%) and I (last 20%).
  for (const id of r2) {
    const r2Idx = id; // R2-local
    if (r2Idx < r2eSize) {
      const y =
        canvas.height * 0.32 +
        (r2Idx / r2eSize) * canvas.height * 0.5;
      ctx.fillStyle = dotR2e;
      ctx.fillRect(xpos, Math.floor(y), dot, dot);
    } else {
      const i = r2Idx - r2eSize;
      const y =
        canvas.height * 0.84 + (i / Math.max(r2iSize, 1)) * canvas.height * 0.14;
      ctx.fillStyle = dotR2i;
      ctx.fillRect(xpos, Math.floor(y), dot, dot);
    }
  }

  // Sweeping cursor.
  ctx.fillStyle = "rgba(255,255,255,0.08)";
  ctx.fillRect(xpos + dot, 0, 1, canvas.height);

  xCursor += 1; // 1ms per Step batch
}

// Live stats ----------------------------------------------------------------

let recentR1 = 0;
let recentR2 = 0;
let recentMs = 0;
let lastSampleMs = 0;
let simTms = 0;

function tickStats(t_ms, r1Count, r2Count) {
  recentR1 += r1Count;
  recentR2 += r2Count;
  recentMs += 1; // 1 ms batches
  simTms = t_ms;
  $("#sim-t").textContent = `${Math.round(t_ms)} ms`;

  // Refresh rate readout every ~250 ms of sim time.
  if (simTms - lastSampleMs >= 250) {
    const window_s = (simTms - lastSampleMs) / 1000;
    const r1ps = window_s > 0 ? Math.round(recentR1 / window_s) : 0;
    const r2ps = window_s > 0 ? Math.round(recentR2 / window_s) : 0;
    $("#rate-r1").textContent = `${r1ps.toLocaleString()}`;
    $("#rate-r2").textContent = `${r2ps.toLocaleString()}`;
    recentR1 = 0;
    recentR2 = 0;
    lastSampleMs = simTms;
  }
}

// WebSocket session ---------------------------------------------------------

let socket = null;

function startSession(query) {
  if (socket) {
    socket.close();
    socket = null;
  }
  $("#decoded").textContent = "—";
  $("#reduction").textContent = "—";
  $("#rag-tok").textContent = "—";
  $("#javis-tok").textContent = "—";
  $("#rag-text").textContent = "—";
  $("#javis-text").textContent = "—";
  $("#phase").textContent = "connecting…";
  $("#log").innerHTML = "";
  recentR1 = 0;
  recentR2 = 0;
  lastSampleMs = 0;
  simTms = 0;

  const url = `ws://${location.host}/ws?query=${encodeURIComponent(query)}`;
  socket = new WebSocket(url);
  socket.onopen = () => log(`ws connected — query="${query}"`);
  socket.onclose = () => log("ws closed");
  socket.onerror = (e) => log("ws error");
  socket.onmessage = (msg) => {
    let ev;
    try {
      ev = JSON.parse(msg.data);
    } catch (e) {
      return;
    }
    handleEvent(ev);
  };
}

function handleEvent(ev) {
  switch (ev.type) {
    case "init":
      setupRaster(ev);
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
      plotSpikes(ev.t_ms, ev.r1, ev.r2);
      tickStats(ev.t_ms, ev.r1.length, ev.r2.length);
      break;
    case "decoded": {
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
      log(
        `decoded query="${ev.query}" → ${ev.candidates.length} candidates, ` +
          `reduction ${ev.reduction_pct.toFixed(1)}%`,
      );
      break;
    }
    case "done":
      log("session done");
      break;
  }
}

$("#cue-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#cue").value.trim();
  if (q) startSession(q);
});

// Auto-start with the default query so the first visit shows life.
startSession($("#cue").value.trim() || "rust");
