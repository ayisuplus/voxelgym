import { readFileSync } from "node:fs";

import { afterEach, expect, test, vi } from "vitest";


const WEB_ROOT = process.cwd();


function installPage() {
  const html = readFileSync(`${WEB_ROOT}/static/index.html`, "utf8")
    .replace(/<script src="\/static\/app\.js"><\/script>/, "");
  document.open();
  document.write(html);
  document.close();

  const contexts = new Map();
  HTMLCanvasElement.prototype.getContext = function getContext() {
    if (!contexts.has(this.id)) {
      const context = {
        canvas: this,
        clearRect: vi.fn(),
        createImageData: vi.fn((width, height) => ({
          data: new Uint8ClampedArray(width * height * 4),
          width,
          height,
        })),
        fillRect: vi.fn(),
        putImageData: vi.fn((image) => {
          context.lastImage = new Uint8ClampedArray(image.data);
        }),
        fillStyle: "",
        globalAlpha: 1,
        imageSmoothingEnabled: true,
      };
      contexts.set(this.id, context);
    }
    return contexts.get(this.id);
  };
  return contexts;
}


class FakeWebSocket {
  static instances = [];

  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.sent = [];
    FakeWebSocket.instances.push(this);
  }

  open() {
    this.readyState = 1;
    this.onopen();
  }

  closeFromServer() {
    this.readyState = 3;
    this.onclose();
  }

  send(message) {
    this.sent.push(JSON.parse(message));
  }

  message(data) {
    this.onmessage({ data });
  }
}


function makePacket(overrides = {}) {
  const res = 2;
  const header = {
    res,
    task: "alpha",
    showcase: false,
    seed: 9,
    episode: 2,
    wins: 1,
    losses: 1,
    tick: 42,
    hp: 18,
    pos: [1, 2, 3],
    ep_reward: 3.5,
    results: [
      { ep: 1, ok: false, reward: -1 },
      { ep: 2, ok: true, reward: 3.5 },
    ],
    last_result: { ep: 1, ok: false, reward: -1 },
    action: { move: 1, jump: 1, craft: 0 },
    stage: "0: Navigate",
    policy: "expert",
    paused: false,
    speed: 2,
    chase: { eye: [0, 0, -4], agent: [0, 0, 0], yaw: 0, pitch: 0 },
    ...overrides,
  };
  const encoded = new TextEncoder().encode(JSON.stringify(header));
  const headerLength = Math.ceil(encoded.length / 4) * 4;
  const rgbLength = res * res * 3;
  const segLength = res * res * 2;
  const lidarLength = 16 * 256 * 4;
  const buffer = new ArrayBuffer(4 + headerLength + rgbLength * 2 + segLength + lidarLength);
  const view = new DataView(buffer);
  view.setUint32(0, headerLength, true);
  new Uint8Array(buffer, 4, encoded.length).set(encoded);
  new Uint8Array(buffer, 4 + encoded.length, headerLength - encoded.length).fill(32);
  let offset = 4 + headerLength;
  new Uint8Array(buffer, offset, rgbLength).set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  offset += rgbLength;
  new Uint8Array(buffer, offset, rgbLength).fill(40);
  offset += rgbLength;
  new Uint16Array(buffer, offset, 4).set([0xffff, 1, 99, 0]);
  offset += segLength;
  const lidar = new Float32Array(buffer, offset, 16 * 256);
  lidar.fill(8);
  lidar[0] = 0;
  lidar[15 * 256] = 16;
  return buffer;
}


afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  FakeWebSocket.instances = [];
});


test("real page bootstraps, controls the socket, and renders every packet channel", async () => {
  vi.useFakeTimers();
  const contexts = installPage();
  vi.stubGlobal("WebSocket", FakeWebSocket);
  vi.stubGlobal("fetch", vi.fn(async () => ({
    json: async () => ({ tasks: [
      { name: "alpha", desc: "Alpha description" },
      { name: "beta", desc: "Beta description" },
    ] }),
  })));

  await import("../static/app.js");
  await Promise.resolve();
  await Promise.resolve();

  expect(fetch).toHaveBeenCalledWith("/api/tasks");
  expect([...document.querySelectorAll("#task option")].map((option) => option.value))
    .toEqual(["__showcase__", "alpha", "beta"]);
  expect(document.querySelector("#taskdesc").textContent).toContain("full curriculum");

  const socket = FakeWebSocket.instances[0];
  expect(socket.url).toBe("ws://voxel.test/ws");
  expect(socket.binaryType).toBe("arraybuffer");
  socket.open();
  expect(document.querySelector("#conn").textContent).toBe("live");

  socket.message(JSON.stringify({
    type: "palette",
    colors: [[12, 13, 14], [21, 22, 23]],
  }));
  socket.message(makePacket());

  expect(document.querySelector("#view").width).toBe(2);
  expect(contexts.get("view").lastImage.slice(0, 4)).toEqual(
    new Uint8ClampedArray([1, 2, 3, 255]),
  );
  expect(contexts.get("chase").fillRect).toHaveBeenCalledTimes(4);
  expect(contexts.get("seg").lastImage.slice(0, 16)).toEqual(
    new Uint8ClampedArray([
      0x78, 0xa6, 0xff, 255,
      21, 22, 23, 255,
      255, 0, 255, 255,
      12, 13, 14, 255,
    ]),
  );
  expect(contexts.get("lidar").lastImage.slice(0, 4)).toEqual(
    new Uint8ClampedArray([255, 0, 0, 255]),
  );
  const lowestElevationStart = 15 * 256 * 4;
  expect(contexts.get("lidar").lastImage.slice(lowestElevationStart, lowestElevationStart + 4)).toEqual(
    new Uint8ClampedArray([0, 0, 0, 255]),
  );
  expect(document.querySelector("#hud").textContent).toContain("tick     42");
  expect(document.querySelector("#task").value).toBe("alpha");
  expect(document.querySelector("#taskdesc").textContent).toBe("Alpha description");
  expect(document.querySelector("#seed").value).toBe("9");
  expect(contexts.get("bars").fillRect).toHaveBeenCalledTimes(2);

  socket.message(makePacket({
    episode: 3,
    last_result: { ep: 2, ok: true, reward: 3.5 },
  }));
  expect(document.querySelector("#banner").className).toBe("banner show win");
  expect(document.querySelector("#banner").textContent).toContain("SUCCESS  +3.5");
  socket.message(makePacket({
    episode: 3,
    last_result: { ep: 2, ok: true, reward: 3.5 },
  }));
  await vi.advanceTimersByTimeAsync(2600);
  expect(document.querySelector("#banner").classList.contains("show")).toBe(false);

  const task = document.querySelector("#task");
  task.value = "beta";
  task.onchange();
  task.value = "__showcase__";
  task.onchange();
  const seed = document.querySelector("#seed");
  seed.value = "12";
  seed.onchange({ target: seed });
  const speed = document.querySelector("#speed");
  speed.value = "8";
  speed.oninput();
  const quality = document.querySelector("#quality");
  quality.value = "4";
  quality.onchange({ target: quality });
  const policy = document.querySelector("#policy");
  policy.value = "random";
  policy.onchange({ target: policy });
  document.querySelector("#pause").onclick({ target: document.querySelector("#pause") });
  document.querySelector("#pause").onclick({ target: document.querySelector("#pause") });
  document.querySelector("#reset").onclick();

  expect(socket.sent).toEqual([
    { cmd: "set_task", task: "beta" },
    { cmd: "set_showcase", on: true },
    { cmd: "set_seed", seed: 12 },
    { cmd: "set_speed", speed: 8 },
    { cmd: "set_quality", q: 4 },
    { cmd: "set_policy", policy: "random" },
    { cmd: "pause" },
    { cmd: "resume" },
    { cmd: "reset" },
  ]);
  expect(document.querySelector("#speedv").textContent).toBe("8");

  socket.closeFromServer();
  expect(document.querySelector("#conn").textContent).toBe("reconnecting…");
  await vi.advanceTimersByTimeAsync(1000);
  expect(FakeWebSocket.instances).toHaveLength(2);
});
