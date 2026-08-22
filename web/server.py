"""voxelgym demo server: stream live sim frames to a browser.

Architecture: one asyncio loop owns the env (steps are microseconds; a
frame's renders cost ~5 ms — fine inside the loop at 15 fps). Clients
connect over WebSocket and receive binary packets:

    [u32 LE header_len][header JSON][rgb][chase_rgb][seg][lidar_range]

where rgb/chase_rgb are 128x128x3 u8, seg is 128x128 u16, lidar_range is
(16, 256) f32. The header carries HUD state (task/seed/tick/reward/hp/
action/expert stage/episode stats/reward history).

Client -> server control messages are JSON text: set_task, set_seed,
set_speed (ticks per displayed frame), pause/resume, reset, set_policy
("expert" | "random").

Run:  .venv/Scripts/python.exe web/server.py   (then open :8000)
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import numpy as np  # noqa: E402
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from voxelgym.experts import make_expert  # noqa: E402
from voxelgym.env import VoxelGymEnv  # noqa: E402
from voxelgym.tasks import make_task, task_names  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
RES = 128
LIDAR = {"channels": 16, "azimuth": 256, "min_elev": -20.0, "max_elev": 10.0,
         "max_range": 48.0, "every": 1}
FPS = 15


class Sim:
    """Owns the env + policy. Mutated only from the asyncio loop."""

    def __init__(self):
        self.task_name = "navigate_to_target"
        self.seed = 0
        self.speed = 8  # sim ticks per displayed frame
        self.paused = False
        self.policy = "expert"
        self.episode = 0
        self.wins = 0
        self.losses = 0
        self.reward_hist = deque(maxlen=240)
        self.env = None
        self.expert = None
        self.last_action = None
        self._rng = np.random.default_rng(7)
        self.reset_episode(self.seed)

    def reset_episode(self, seed: int):
        self.seed = seed
        self.task = make_task(self.task_name)
        self.env = VoxelGymEnv(task=self.task, seed=seed, lidar=dict(LIDAR))
        self.env.reset(seed=seed)
        self.expert = make_expert(self.task_name, self.task, seed=seed)
        self.ep_reward = 0.0

    def action(self, world) -> dict:
        if self.policy == "expert":
            try:
                return self.expert.act(world)
            except Exception:
                pass  # fall through to random on expert errors in demo mode
        return {k: int(self._rng.integers(0, n)) for k, n in
                zip(("move", "jump", "sneak", "yaw", "pitch", "mine",
                     "place", "use", "hotbar", "craft"),
                    (5, 2, 2, 24, 9, 2, 2, 2, 9, 8))}

    def run_frames(self):
        """Advance `speed` ticks; auto-reset on episode end."""
        for _ in range(self.speed):
            a = self.action(self.env.world)
            self.last_action = a
            _, r, term, trunc, _ = self.env.step(a)
            self.ep_reward += r
            self.reward_hist.append(r)
            if term or trunc:
                ok = bool(term) and not self.env.world.dead()
                self.wins += ok
                self.losses += (not ok)
                self.episode += 1
                self.reset_episode(self.seed + 1)
                break

    def hud(self, last_action: dict | None) -> dict:
        w = self.env.world
        stage = None
        stages = getattr(self.expert, "stages", None)
        idx = getattr(self.expert, "_idx", None)
        if stages and idx is not None and idx < len(stages):
            stage = f"{idx}: {stages[idx].name}"
        return {
            "task": self.task_name,
            "seed": self.seed,
            "episode": self.episode,
            "wins": self.wins,
            "losses": self.losses,
            "tick": w.tick(),
            "hp": w.hp(),
            "pos": [round(v, 1) for v in w.agent_pos()],
            "ep_reward": round(self.ep_reward, 3),
            "reward_hist": list(self.reward_hist),
            "action": last_action or {},
            "stage": stage,
            "policy": self.policy,
            "paused": self.paused,
            "speed": self.speed,
        }


SIM = Sim()
CLIENTS: set[WebSocket] = set()


def build_packet(sim: Sim, hud: dict) -> bytes:
    w = sim.env.world
    rgb, _, seg = w.render()
    # chase cam: 4 cells behind (MC fwd = (-sin yaw, 0, cos yaw)), 3.2 up
    x, y, z, yaw = (v for v in w.obs_pose()[:4])
    yr = math.radians(float(yaw))
    eye = (float(x) + math.sin(yr) * 4.0, float(y) + 3.2, float(z) - math.cos(yr) * 4.0)
    chase, _, _ = w.render_pose(eye, float(yaw), 25.0)
    rng, _, _ = w.lidar_scan(
        channels=LIDAR["channels"], azimuth_steps=LIDAR["azimuth"],
        min_elev_deg=LIDAR["min_elev"], max_elev_deg=LIDAR["max_elev"],
        max_range=LIDAR["max_range"], frame_idx=w.tick())
    head = json.dumps(hud).encode()
    head += b" " * ((-len(head)) % 4)  # keep the f32 section 4-aligned
    parts = [
        struct.pack("<I", len(head)),
        head,
        rgb.tobytes(),
        chase.tobytes(),
        seg.astype(np.uint16).tobytes(),
        rng.astype(np.float32).tobytes(),
    ]
    return b"".join(parts)


async def broadcast(packet: bytes):
    dead = []
    for ws in CLIENTS:
        try:
            await ws.send_bytes(packet)
        except Exception:
            dead.append(ws)
    for ws in dead:
        CLIENTS.discard(ws)


async def sim_loop():
    frame = 1.0 / FPS
    while True:
        t0 = time.perf_counter()
        if not SIM.paused:
            SIM.run_frames()
        if CLIENTS:
            hud = SIM.hud(SIM.last_action)
            await broadcast(build_packet(SIM, hud))
        dt = time.perf_counter() - t0
        await asyncio.sleep(max(0.005, frame - dt))


def apply_cmd(msg: dict):
    cmd = msg.get("cmd")
    if cmd == "set_task":
        name = msg["task"]
        if name in task_names():
            SIM.task_name = name
            SIM.reset_episode(int(msg.get("seed", SIM.seed)))
    elif cmd == "set_seed":
        SIM.reset_episode(int(msg["seed"]))
    elif cmd == "set_speed":
        SIM.speed = max(1, min(500, int(msg["speed"])))
    elif cmd == "pause":
        SIM.paused = True
    elif cmd == "resume":
        SIM.paused = False
    elif cmd == "reset":
        SIM.reset_episode(SIM.seed)
    elif cmd == "set_policy":
        if msg["policy"] in ("expert", "random"):
            SIM.policy = msg["policy"]


def create_app() -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/tasks")
    async def tasks():
        return {"tasks": task_names()}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        CLIENTS.add(websocket)
        # one-time palette for seg colorization
        pal = SIM.env.world.palette().tolist()
        await websocket.send_text(json.dumps({"type": "palette", "colors": pal}))
        try:
            while True:
                data = await websocket.receive_text()
                apply_cmd(json.loads(data))
        except WebSocketDisconnect:
            pass
        finally:
            CLIENTS.discard(websocket)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.on_event("startup")
    async def _start():
        asyncio.create_task(sim_loop())

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
