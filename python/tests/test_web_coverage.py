import asyncio
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from web import server


class StubWorld:
    def palette(self):
        return np.array([[0, 0, 0], [10, 20, 30]], dtype=np.uint8)


class StubSim:
    def __init__(self):
        self.env = SimpleNamespace(world=StubWorld())
        self.seed = 7
        self.showcase = False
        self.show_idx = 3
        self.task_name = "alpha"
        self.speed = 2
        self.paused = False
        self.res = 256
        self.policy = "expert"
        self.reset_calls = []

    def reset_episode(self, seed):
        self.seed = seed
        self.reset_calls.append(seed)


def test_create_app_owns_injected_dependencies_during_lifespan():
    sim = StubSim()
    app = server.create_app(sim=sim, start_loop=False)

    with TestClient(app):
        assert app.state.sim is sim
        assert app.state.clients == set()
        assert app.state.sim_task is None

    assert app.state.clients == set()


def test_create_app_cancels_and_awaits_background_loop(monkeypatch):
    lifecycle = []

    async def fake_loop(sim, clients):
        lifecycle.append(("started", sim, clients))
        try:
            await asyncio.Event().wait()
        finally:
            lifecycle.append(("stopped", sim, clients))

    monkeypatch.setattr(server, "sim_loop", fake_loop)
    sim = StubSim()
    app = server.create_app(sim=sim)

    with TestClient(app):
        assert lifecycle == [("started", sim, app.state.clients)]

    assert lifecycle[-1] == ("stopped", sim, app.state.clients)


class SimWorld:
    def __init__(self):
        self.is_dead = False

    def dead(self):
        return self.is_dead

    def tick(self):
        return 12

    def hp(self):
        return 17

    def agent_pos(self):
        return (1.25, 2.5, -3.75)


class FakeEnv:
    def __init__(self, task, seed, lidar):
        self.task = task
        self.seed = seed
        self.lidar = lidar
        self.world = SimWorld()
        self.reset_calls = []
        self.steps = []
        self.outcome = (None, 1.25, False, False, {})

    def reset(self, seed):
        self.reset_calls.append(seed)

    def step(self, action):
        self.steps.append(action)
        return self.outcome


class FakeExpert:
    def __init__(self):
        self.stages = [SimpleNamespace(op=SimpleNamespace())]
        self._idx = 0
        self.should_raise = False

    def act(self, world):
        if self.should_raise:
            raise RuntimeError("expert failed")
        return {"move": 1}


def make_sim(monkeypatch):
    monkeypatch.setattr(server, "make_task", lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(server, "VoxelGymEnv", FakeEnv)
    monkeypatch.setattr(server, "make_expert", lambda *args, **kwargs: FakeExpert())
    monkeypatch.setattr(server, "random_action", lambda rng: {"jump": 1})
    return server.Sim()


def test_sim_runs_terminal_hold_showcase_and_reports_hud(monkeypatch):
    sim = make_sim(monkeypatch)
    assert sim.action(sim.env.world) == {"move": 1}
    sim.expert.should_raise = True
    assert sim.action(sim.env.world) == {"jump": 1}

    sim.expert.should_raise = False
    sim.env.outcome = (None, 2.5, True, False, {})
    sim.run_frames()
    assert sim.hold == 45
    assert sim.wins == 1
    assert sim.losses == 0
    assert sim.last_result == {"ep": 0, "ok": True, "reward": 2.5}

    hud = sim.hud(sim.last_action)
    assert hud["stage"] == "0: SimpleNamespace"
    assert hud["tick"] == 12
    assert hud["hp"] == 17
    assert hud["pos"] == [1.2, 2.5, -3.8]
    assert hud["holding"] is True

    sim.hold = 1
    old_task = sim.task_name
    sim.run_frames()
    assert sim.episode == 1
    assert sim.seed == 1
    assert sim.task_name != old_task
    assert sim.hold == 0


def test_sim_records_truncation_as_a_loss_and_handles_missing_stage(monkeypatch):
    sim = make_sim(monkeypatch)
    sim.showcase = False
    sim.env.outcome = (None, -0.25, False, True, {})
    sim.run_frames()
    sim.expert.stages = []

    assert sim.losses == 1
    assert sim.last_result == {"ep": 0, "ok": False, "reward": -0.25}
    assert sim.hud(None)["stage"] is None


def test_apply_cmd_covers_controls_validation_and_clamping(monkeypatch):
    monkeypatch.setattr(server, "task_names", lambda: ["alpha", "beta"])
    sim = StubSim()

    server.apply_cmd(sim, {"cmd": "set_task", "task": "missing"})
    assert sim.reset_calls == []
    server.apply_cmd(sim, {"cmd": "set_task", "task": "beta", "seed": 9})
    assert (sim.task_name, sim.showcase, sim.reset_calls[-1]) == ("beta", False, 9)

    server.apply_cmd(sim, {"cmd": "set_showcase", "on": False})
    assert sim.showcase is False
    server.apply_cmd(sim, {"cmd": "set_showcase"})
    assert (sim.showcase, sim.show_idx, sim.task_name) == (True, 0, server.SHOWCASE[0])

    for message in [
        {"cmd": "set_seed", "seed": "11"},
        {"cmd": "set_speed", "speed": 0},
        {"cmd": "pause"},
        {"cmd": "resume"},
        {"cmd": "reset"},
        {"cmd": "set_quality", "q": 4},
        {"cmd": "set_policy", "policy": "random"},
    ]:
        server.apply_cmd(sim, message)

    assert sim.seed == 11
    assert sim.speed == 1
    assert sim.paused is False
    assert sim.res == 512
    assert sim.policy == "random"

    server.apply_cmd(sim, {"cmd": "set_speed", "speed": 999})
    server.apply_cmd(sim, {"cmd": "set_quality", "q": 99})
    server.apply_cmd(sim, {"cmd": "set_policy", "policy": "unsafe"})
    server.apply_cmd(sim, {"cmd": "unknown"})
    assert (sim.speed, sim.res, sim.policy) == (500, 256, "random")


class PacketWorld:
    def __init__(self, hit=1.0):
        self.hit = hit
        self.render_poses = []

    def obs_pose(self):
        return (1, 2, 3, 0, 10, 0)

    def render_pose(self, eye, yaw, pitch, width, height):
        self.render_poses.append((eye, yaw, pitch, width, height))
        value = 7 if len(self.render_poses) == 1 else 9
        rgb = np.full((height, width, 3), value, dtype=np.uint8)
        seg = np.array([[1, 2], [65535, 0]], dtype=np.uint16)
        return rgb, np.zeros_like(rgb), seg, np.zeros((height, width), dtype=np.float32)

    def cast_ray(self, origin, direction, distance):
        self.ray = (origin, direction, distance)
        return self.hit

    def lidar_scan(self, **kwargs):
        self.lidar_kwargs = kwargs
        values = np.arange(16 * 256, dtype=np.float32).reshape(16, 256)
        return values, np.zeros_like(values, dtype=np.uint16), np.zeros_like(values, dtype=np.uint8)

    def tick(self):
        return 23


def decode_packet(packet):
    header_len = struct.unpack_from("<I", packet)[0]
    header = json.loads(packet[4:4 + header_len])
    return header_len, header, packet[4 + header_len:]


def test_build_packet_has_aligned_protocol_sections_and_collision_safe_camera():
    world = PacketWorld(hit=1.0)
    sim = SimpleNamespace(env=SimpleNamespace(world=world), res=2, cam_yaw=180.0)

    packet = server.build_packet(sim, {"tick": 23})
    header_len, header, body = decode_packet(packet)

    assert header_len % 4 == 0
    assert header["res"] == 2
    assert header["chase"]["yaw"] == 0.0
    assert body[:12] == bytes([7] * 12)
    assert body[12:24] == bytes([9] * 12)
    assert np.frombuffer(body, dtype=np.uint16, count=4, offset=24).tolist() == [1, 2, 65535, 0]
    assert np.frombuffer(body, dtype=np.float32, offset=32)[-1] == 4095
    assert len(packet) == 4 + header_len + 12 + 12 + 8 + 16 * 256 * 4
    assert world.lidar_kwargs["frame_idx"] == 23

    origin, _, _ = world.ray
    chase_eye = world.render_poses[1][0]
    pulled_distance = np.linalg.norm(np.subtract(chase_eye, origin))
    assert pulled_distance == pytest.approx(0.7)


def test_build_packet_smooths_camera_when_ray_does_not_hit():
    world = PacketWorld(hit=-1.0)
    sim = SimpleNamespace(env=SimpleNamespace(world=world), res=2, cam_yaw=20.0)

    _, header, _ = decode_packet(server.build_packet(sim, {}))

    assert header["chase"]["yaw"] == 14.0
    assert np.linalg.norm(np.subtract(world.render_poses[1][0], world.ray[0])) > 4


class Socket:
    def __init__(self, fail=False):
        self.fail = fail
        self.packets = []

    async def send_bytes(self, packet):
        if self.fail:
            raise ConnectionError("gone")
        self.packets.append(packet)


def test_broadcast_sends_packet_and_prunes_failed_clients():
    live = Socket()
    dead = Socket(fail=True)
    clients = {live, dead}

    asyncio.run(server.broadcast(b"frame", clients))

    assert live.packets == [b"frame"]
    assert clients == {live}


def test_broadcast_uses_a_snapshot_when_clients_change_during_send():
    clients = set()
    late = Socket()

    class JoiningSocket(Socket):
        async def send_bytes(self, packet):
            await super().send_bytes(packet)
            clients.add(late)

    joining = JoiningSocket()
    clients.add(joining)

    asyncio.run(server.broadcast(b"frame", clients))

    assert joining.packets == [b"frame"]
    assert late.packets == []
    assert clients == {joining, late}


def test_sim_loop_advances_and_broadcasts_until_cancelled(monkeypatch):
    sim = SimpleNamespace(paused=False, last_action={"move": 1}, calls=0)

    def run_frames():
        sim.calls += 1

    sim.run_frames = run_frames
    sim.hud = lambda action: {"action": action}
    monkeypatch.setattr(server, "build_packet", lambda active, hud: b"packet")

    async def stop_after_packet(packet, clients):
        assert packet == b"packet"
        raise asyncio.CancelledError

    monkeypatch.setattr(server, "broadcast", stop_after_packet)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(server.sim_loop(sim, {Socket()}))
    assert sim.calls == 1


def test_sim_loop_logs_frame_errors_and_keeps_running(monkeypatch, capsys):
    sim = SimpleNamespace(
        paused=False,
        last_action=None,
        run_frames=lambda: (_ for _ in ()).throw(ValueError("bad frame")),
    )
    delays = []

    async def controlled_sleep(delay):
        delays.append(delay)
        if len(delays) == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(server.asyncio, "sleep", controlled_sleep)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(server.sim_loop(sim, set()))

    assert delays[0] == 0.5
    assert delays[1] >= 0.005
    assert "[sim_loop] ValueError: bad frame" in capsys.readouterr().err


def test_http_and_websocket_routes_keep_public_protocol(monkeypatch):
    class AlphaTask:
        """Alpha task description.

        More detail is intentionally omitted from the API.
        """

    monkeypatch.setattr(server, "task_names", lambda: ["alpha"])
    monkeypatch.setattr(server, "make_task", lambda name: AlphaTask())
    sim = StubSim()
    app = server.create_app(sim=sim, start_loop=False)

    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert "live training view" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/tasks").json() == {
            "tasks": [{"name": "alpha", "desc": "Alpha task description."}]
        }

        with client.websocket_connect("/ws") as ws:
            assert ws.receive_json() == {
                "type": "palette",
                "colors": [[0, 0, 0], [10, 20, 30]],
            }
            assert len(app.state.clients) == 1
            ws.send_json({"cmd": "pause"})
            assert sim.paused is True

        assert app.state.clients == set()
