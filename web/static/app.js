/* voxelgym live view client. Binary packet:
 * [u32 LE header_len][header JSON][rgb 128*128*3][chase 128*128*3]
 * [seg 128*128 u16][lidar 16*256 f32]                                 */

const RES = 128, LIDAR_C = 16, LIDAR_A = 256;
let palette = null;
const SKY = [0x78, 0xA6, 0xFF];

const conn = document.getElementById("conn");
const hud = document.getElementById("hud");
const ctxView = document.getElementById("view").getContext("2d");
const ctxChase = document.getElementById("chase").getContext("2d");
const ctxSeg = document.getElementById("seg").getContext("2d");
const ctxLidar = document.getElementById("lidar").getContext("2d");
const ctxSpark = document.getElementById("spark").getContext("2d");

const imgView = ctxView.createImageData(RES, RES);
const imgChase = ctxChase.createImageData(RES, RES);
const imgSeg = ctxSeg.createImageData(RES, RES);
const imgLidar = ctxLidar.createImageData(LIDAR_A, LIDAR_C);

/* expand 3-channel to RGBA */
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
/* turbo-ish blue->green->yellow->red heatmap; range 0 = black.
 * Color saturates at 16 cells so nearby ground reads bright. */
function heat(r, max = 16) {
  if (r <= 0) return [0, 0, 0];
  const t = Math.min(1, r / max);
  return [Math.round(255 * Math.min(1, 2 * t)),
          Math.round(255 * Math.min(1, 2 * (1 - Math.abs(t - 0.5) * 2)) * (t < 0.5 ? t * 2 : 1)),
          Math.round(255 * Math.max(0, 1 - 2 * t))];
}
function putLidar(bytes, off) {
  const rng = new Float32Array(bytes, off, LIDAR_C * LIDAR_A);
  const out = imgLidar.data;
  for (let c = 0; c < LIDAR_C; c++) {
    for (let a = 0; a < LIDAR_A; a++) {
      // row 0 = lowest elevation -> draw flipped so up is up
      const src = c * LIDAR_A + a;
      const dst = ((LIDAR_C - 1 - c) * LIDAR_A + a) * 4;
      const col = heat(rng[src]);
      out[dst] = col[0]; out[dst + 1] = col[1]; out[dst + 2] = col[2]; out[dst + 3] = 255;
    }
  }
  ctxLidar.putImageData(imgLidar, 0, 0);
}
function drawSpark(hist) {
  const w = ctxSpark.canvas.width, h = ctxSpark.canvas.height;
  ctxSpark.clearRect(0, 0, w, h);
  if (!hist || hist.length < 2) return;
  const max = Math.max(1, ...hist.map(Math.abs));
  ctxSpark.strokeStyle = "#7ee2a8";
  ctxSpark.beginPath();
  hist.forEach((r, i) => {
    const x = (i / (hist.length - 1)) * w;
    const y = h - 8 - (r / max) * (h - 16);
    i ? ctxSpark.lineTo(x, y) : ctxSpark.moveTo(x, y);
  });
  ctxSpark.stroke();
}
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
    `policy   ${h.policy}${h.paused ? "  [PAUSED]" : ""}   speed ${h.speed}`;
  drawSpark(h.reward_hist);
  // keep controls in sync with server state (page reload, auto-advance)
  const taskSel = document.getElementById("task");
  if (document.activeElement !== taskSel && taskSel.value !== h.task) taskSel.value = h.task;
  const seedIn = document.getElementById("seed");
  if (document.activeElement !== seedIn && +seedIn.value !== h.seed) seedIn.value = h.seed;
}

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
    let off = 4 + hlen;
    putRgb(ctxView, imgView, buf, off); off += RES * RES * 3;
    putRgb(ctxChase, imgChase, buf, off); off += RES * RES * 3;
    /* chase cam follows the agent every frame (4 behind, 3.2 up, pitch 25
     * down): the agent projects to a fixed box near center — the renderer
     * draws blocks only, so overlay the agent as a spectator marker */
    ctxChase.strokeStyle = "#ff4444";
    ctxChase.lineWidth = 2;
    ctxChase.strokeRect(57, 53, 14, 32);
    ctxChase.beginPath(); ctxChase.moveTo(64, 49); ctxChase.lineTo(64, 53);
    ctxChase.stroke();
    putSeg(buf, off); off += RES * RES * 2;
    putLidar(buf, off);
    show(head);
  };
}
connect();

/* controls */
const send = (m) => ws && ws.readyState === 1 && ws.send(JSON.stringify(m));
const taskSel = document.getElementById("task");
fetch("/api/tasks").then(r => r.json()).then(({ tasks }) => {
  for (const t of tasks) {
    const o = document.createElement("option");
    o.value = o.textContent = t;
    taskSel.appendChild(o);
  }
});
taskSel.onchange = () => send({ cmd: "set_task", task: taskSel.value });
document.getElementById("seed").onchange = (e) => send({ cmd: "set_seed", seed: +e.target.value });
const speed = document.getElementById("speed");
speed.oninput = () => {
  document.getElementById("speedv").textContent = speed.value;
  send({ cmd: "set_speed", speed: +speed.value });
};
document.getElementById("policy").onchange = (e) => send({ cmd: "set_policy", policy: e.target.value });
document.getElementById("pause").onclick = (e) => {
  const paused = e.target.textContent.includes("resume");
  send({ cmd: paused ? "resume" : "pause" });
  e.target.textContent = paused ? "⏸ pause" : "▶ resume";
};
document.getElementById("reset").onclick = () => send({ cmd: "reset" });
