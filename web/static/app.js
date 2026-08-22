/* voxelgym live view client. Binary packet:
 * [u32 LE header_len][header JSON (4-padded)][rgb 128*128*3]
 * [chase 128*128*3][seg 128*128 u16][lidar 16*256 f32]            */

const LIDAR_C = 16, LIDAR_A = 256;
let RES = 256;              // follows the server's render resolution
let palette = null;
const SKY = [0x78, 0xA6, 0xFF];

const $ = (id) => document.getElementById(id);
const conn = $("conn"), hud = $("hud");
const ctxView = $("view").getContext("2d");
const ctxChase = $("chase").getContext("2d");
const ctxSeg = $("seg").getContext("2d");
const ctxLidar = $("lidar").getContext("2d");
const ctxBars = $("bars").getContext("2d");

let imgView, imgChase, imgSeg;
const imgLidar = ctxLidar.createImageData(LIDAR_A, LIDAR_C);
function setRes(r) {
  if (r === RES && imgView) return;
  RES = r;
  for (const [c, ctx] of [["view", ctxView], ["chase", ctxChase], ["seg", ctxSeg]]) {
    $(c).width = r; $(c).height = r;
    ctx.imageSmoothingEnabled = false;
  }
  imgView = ctxView.createImageData(r, r);
  imgChase = ctxChase.createImageData(r, r);
  imgSeg = ctxSeg.createImageData(r, r);
}
setRes(RES);

/* --- channel painters --- */
function expand(bytes, off) {
  const src = new Uint8Array(bytes, off, RES * RES * 3);
  const out = new Uint8ClampedArray(RES * RES * 4);
  for (let i = 0, j = 0; i < src.length; i += 3, j += 4) {
    out[j] = src[i]; out[j + 1] = src[i + 1]; out[j + 2] = src[i + 2]; out[j + 3] = 255;
  }
  return out;
}
function putRgb(ctx, img, bytes, off) {
  img.data.set(expand(bytes, off));
  ctx.putImageData(img, 0, 0);
}
function putSeg(bytes, off) {
  if (!palette) return;
  const seg = new Uint16Array(bytes, off, RES * RES);
  const out = imgSeg.data;
  for (let i = 0; i < seg.length; i++) {
    const id = seg[i];
    const c = id === 0xffff ? SKY : (palette[id] || [255, 0, 255]);
    out[i * 4] = c[0]; out[i * 4 + 1] = c[1]; out[i * 4 + 2] = c[2]; out[i * 4 + 3] = 255;
  }
  ctxSeg.putImageData(imgSeg, 0, 0);
}
function heat(r) {
  if (r <= 0) return [0, 0, 0];
  const t = Math.min(1, r / 16); // saturate at 16 cells: near ground bright
  return [Math.round(255 * Math.min(1, 2 * t)),
          Math.round(255 * Math.min(1, 2 * (1 - Math.abs(t - 0.5) * 2)) * (t < 0.5 ? t * 2 : 1)),
          Math.round(255 * Math.max(0, 1 - 2 * t))];
}
function putLidar(bytes, off) {
  const rng = new Float32Array(bytes, off, LIDAR_C * LIDAR_A);
  const out = imgLidar.data;
  for (let c = 0; c < LIDAR_C; c++) {
    for (let a = 0; a < LIDAR_A; a++) {
      // channel 0 = lowest elevation -> bottom row
      const src = c * LIDAR_A + a;
      const dst = ((LIDAR_C - 1 - c) * LIDAR_A + a) * 4;
      const col = heat(rng[src]);
      out[dst] = col[0]; out[dst + 1] = col[1]; out[dst + 2] = col[2]; out[dst + 3] = 255;
    }
  }
  ctxLidar.putImageData(imgLidar, 0, 0);
}

/* --- agent avatar projected into the chase cam ---
 * Ports camera_rays(): fwd = [-sin y cos p, -sin p, cos y cos p]
 * (pitch positive = down); right/up same construction as the renderer. */
function camBasis(yawDeg, pitchDeg) {
  const y = yawDeg * Math.PI / 180, p = pitchDeg * Math.PI / 180;
  const fwd = [-Math.sin(y) * Math.cos(p), -Math.sin(p), Math.cos(y) * Math.cos(p)];
  let right = [fwd[2], 0, -fwd[0]];
  const rl = Math.hypot(right[0], right[2]) || 1;
  right = [right[0] / rl, 0, right[2] / rl];
  const up = [
    fwd[1] * right[2] - fwd[2] * right[1],
    fwd[2] * right[0] - fwd[0] * right[2],
    fwd[0] * right[1] - fwd[1] * right[0],
  ];
  return { fwd, right, up };
}
function drawAvatar(chase) {
  const { fwd, right, up } = camBasis(chase.yaw, chase.pitch);
  const v = chase.agent.map((c, i) => c - chase.eye[i]);
  const zc = v[0] * fwd[0] + v[1] * fwd[1] + v[2] * fwd[2];
  if (zc < 0.5) return;
  const xc = v[0] * right[0] + v[1] * right[1] + v[2] * right[2];
  const yc = v[0] * up[0] + v[1] * up[1] + v[2] * up[2];
  const px = (xc / zc + 1) * RES / 2;          // fov 90 -> half = 1
  const py = (1 - yc / zc) * RES / 2;
  const h = (1.8 / zc) * RES / 2;              // agent is 1.8 tall
  const w = (0.62 / zc) * RES / 2;  // body + head: simple steve-like figure (annotation layer, not sim truth)
  ctxChase.fillStyle = "#e06030";
  ctxChase.fillRect(px - w / 2, py - h * 0.18, w, h * 0.68);         // body
  ctxChase.fillStyle = "#e0b080";
  ctxChase.fillRect(px - w * 0.38, py - h * 0.5, w * 0.76, h * 0.32); // head
  ctxChase.fillStyle = "#40485a";
  ctxChase.fillRect(px - w / 2, py + h * 0.32, w * 0.46, h * 0.18);   // legs
  ctxChase.fillRect(px + w * 0.04, py + h * 0.32, w * 0.46, h * 0.18);
}

/* --- episode bars + banner --- */
function drawBars(results) {
  const W = ctxBars.canvas.width, H = ctxBars.canvas.height;
  ctxBars.clearRect(0, 0, W, H);
  if (!results || !results.length) return;
  const max = Math.max(0.5, ...results.map(r => Math.abs(r.reward)));
  const bw = Math.max(2, Math.floor(W / 48) - 1);
  results.slice(-48).forEach((r, i) => {
    const x = i * (bw + 1);
    const bh = Math.max(2, (Math.abs(r.reward) / max) * (H - 8));
    ctxBars.fillStyle = r.ok ? "#3fae6a" : "#c0503f";
    ctxBars.fillRect(x, H - 4 - bh, bw, bh);
  });
}
let lastResultEp = null, bannerTimer = null;
function showBanner(h) {
  // fires when a NEW result lands — during the terminal-frame hold, so the
  // banner overlays the actual success/failure scene, not the next episode
  const r = h.last_result;
  if (!r || r.ep === lastResultEp) return;
  if (lastResultEp === null) { lastResultEp = r.ep; return; } // stale on connect
  lastResultEp = r.ep;
  const b = $("banner");
  b.className = `banner show ${r.ok ? "win" : "fail"}`;
  b.textContent = `episode ${r.ep} — ${r.ok ? "SUCCESS" : "FAILED"}  ${r.reward > 0 ? "+" : ""}${r.reward}`;
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => b.classList.remove("show"), 2600);
}

/* --- hud --- */
function fmtAction(a) {
  if (!a || a.move === undefined) return "-";
  const keys = ["move", "jump", "sneak", "yaw", "pitch", "mine", "place", "use", "hotbar", "craft"];
  return keys.filter(k => a[k]).map(k => `${k}=${a[k]}`).join(" ") || "idle";
}
function show(h) {
  const rate = h.wins + h.losses ? (100 * h.wins / (h.wins + h.losses)).toFixed(0) : "-";
  hud.textContent =
    `task     ${h.task}   seed ${h.seed}\n` +
    `episode  ${h.episode}   win ${h.wins} / loss ${h.losses}  (${rate}%)\n` +
    `tick     ${h.tick}   hp ${h.hp}   pos ${h.pos.join(", ")}\n` +
    `reward   ${h.ep_reward}\n` +
    `stage    ${h.stage || "-"}\n` +
    `action   ${fmtAction(h.action)}\n` +
    `policy   ${h.policy}${h.paused ? "  [PAUSED]" : ""}   speed ${h.speed}${h.showcase ? "   ✨ showcase" : ""}`;
  drawBars(h.results);
  showBanner(h);
  const taskSel = $("task");
  if (document.activeElement !== taskSel) {
    const want = h.showcase ? "__showcase__" : h.task;
    if (taskSel.value !== want) { taskSel.value = want; showDesc(); }
  }
  const seedIn = $("seed");
  if (document.activeElement !== seedIn && +seedIn.value !== h.seed) seedIn.value = h.seed;
}

/* --- socket --- */
let ws;
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { conn.textContent = "live"; conn.className = "pill ok"; };
  ws.onclose = () => {
    conn.textContent = "reconnecting…"; conn.className = "pill warn";
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      const m = JSON.parse(ev.data);
      if (m.type === "palette") palette = m.colors;
      return;
    }
    const buf = ev.data;
    const hlen = new DataView(buf, 0, 4).getUint32(0, true);
    const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
    if (head.res) setRes(head.res);
    let off = 4 + hlen;
    putRgb(ctxView, imgView, buf, off); off += RES * RES * 3;
    putRgb(ctxChase, imgChase, buf, off); off += RES * RES * 3;
    if (head.chase) drawAvatar(head.chase);
    putSeg(buf, off); off += RES * RES * 2;
    putLidar(buf, off);
    show(head);
  };
}
connect();

/* --- controls --- */
const send = (m) => ws && ws.readyState === 1 && ws.send(JSON.stringify(m));
const taskSel = $("task");
let taskDescs = {};
function showDesc() {
  $("taskdesc").textContent =
    taskSel.value === "__showcase__"
      ? "cycles through 10 tasks, one episode each — the full curriculum"
      : (taskDescs[taskSel.value] || "");
}
fetch("/api/tasks").then(r => r.json()).then(({ tasks }) => {
  const sc = document.createElement("option");
  sc.value = "__showcase__"; sc.textContent = "✨ showcase (cycle all)";
  taskSel.appendChild(sc);
  for (const t of tasks) {
    const o = document.createElement("option");
    o.value = t.name; o.textContent = t.name;
    taskSel.appendChild(o);
    taskDescs[t.name] = t.desc;
  }
  taskSel.value = "__showcase__";
  showDesc();
});
taskSel.onchange = () => {
  if (taskSel.value === "__showcase__") send({ cmd: "set_showcase", on: true });
  else send({ cmd: "set_task", task: taskSel.value });
  showDesc();
};
$("seed").onchange = (e) => send({ cmd: "set_seed", seed: +e.target.value });
const speed = $("speed");
speed.oninput = () => {
  $("speedv").textContent = speed.value;
  send({ cmd: "set_speed", speed: +speed.value });
};
$("quality").onchange = (e) => send({ cmd: "set_quality", q: +e.target.value });
$("policy").onchange = (e) => send({ cmd: "set_policy", policy: e.target.value });
$("pause").onclick = (e) => {
  const paused = e.target.textContent.includes("resume");
  send({ cmd: paused ? "resume" : "pause" });
  e.target.textContent = paused ? "⏸ pause" : "▶ resume";
};
$("reset").onclick = () => send({ cmd: "reset" });
