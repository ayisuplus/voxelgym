"""Deterministic coverage tests for data, training, vector, and replay seams."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from voxelgym import baseline, datasets, recorder, replay, vec
from voxelgym.env import ACTION_KEYS


def _frame_rows(count: int, *, rendered: bool = True, offset: int = 0):
    rows = []
    for tick in range(count):
        value = offset + tick
        rows.append(
            {
                "tick": tick,
                **{key: (value + i) % 2 for i, key in enumerate(ACTION_KEYS)},
                "swap": 0,
                "reward": float(tick),
                "done": tick == count - 1,
                "voxel_win": b"vox",
                "inv": b"inv",
                "rgb": np.full((128, 128, 3), value, np.uint8).tobytes() if rendered else None,
                "depth": np.full((128, 128), value + 0.5, np.float16).tobytes() if rendered else None,
                "seg": np.full((128, 128), value, np.uint16).tobytes() if rendered else None,
                "world_ckpt": None,
            }
        )
    return rows


def _write_shard(path, rows):
    pq.write_table(pa.Table.from_pylist(rows, schema=recorder.SCHEMA), path)


@pytest.mark.ml
def test_sequence_dataset_splits_decodes_and_reuses_one_shard_cache(tmp_path, monkeypatch):
    _write_shard(tmp_path / "a.parquet", _frame_rows(3, offset=1))
    _write_shard(tmp_path / "b.parquet", _frame_rows(4, offset=10))

    train = datasets.VoxelSequenceDataset(str(tmp_path), seq_len=2, split="train", test_frac=0.5)
    test = datasets.VoxelSequenceDataset(str(tmp_path), seq_len=2, split="test", test_frac=0.5)
    assert len(train) == 3
    assert len(test) == 2

    reads = []
    original = pq.read_table

    def tracked(path):
        reads.append(str(path))
        return original(path)

    monkeypatch.setattr(datasets.pq, "read_table", tracked)
    rgb, actions, depth, seg = train[0]
    assert tuple(rgb.shape) == (2, 128, 128, 3)
    assert tuple(actions.shape) == (2, len(ACTION_KEYS))
    assert tuple(depth.shape) == (2, 128, 128)
    assert tuple(seg.shape) == (2, 128, 128)
    assert rgb[0, 0, 0].tolist() == [10, 10, 10]
    assert depth.dtype.is_floating_point
    train[1]
    assert len(reads) == 1, "adjacent windows must reuse the decoded episode"


@pytest.mark.ml
def test_sequence_dataset_reports_missing_data(tmp_path):
    with pytest.raises(AssertionError, match="no parquet shards"):
        datasets.VoxelSequenceDataset(str(tmp_path))

    _write_shard(tmp_path / "headless.parquet", _frame_rows(2, rendered=False))
    ds = datasets.VoxelSequenceDataset(str(tmp_path), seq_len=1)
    with pytest.raises(ValueError, match="no rgb frames"):
        ds[0]


def test_dataset_export_wrapper_and_cli_exit_codes(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_episode(task, seed, **kwargs):
        calls.append((task, seed, kwargs))
        return seed % 2 == 0, seed + 1, 99, str(tmp_path / f"{seed}.parquet")

    import voxelgym.experts as experts

    monkeypatch.setattr(experts, "run_episode", fake_episode)
    datasets.export("collect_log", 2, str(tmp_path), render=0, seed0=4, epsilon=0.25, scale=2.0)
    assert [c[1] for c in calls] == [4, 5]
    assert calls[0][2] == {"record_dir": str(tmp_path), "render": False, "epsilon": 0.25, "scale": 2.0}
    assert "expert success 1/2" in capsys.readouterr().out

    monkeypatch.setattr(datasets, "export", lambda *a, **kw: calls.append((a, kw)))
    assert datasets.main(["export", "--task", "collect_log", "--episodes", "1", "--out", str(tmp_path)]) == 0

    monkeypatch.setattr(datasets, "baseline", lambda *a, **kw: 0.5)
    assert datasets.main(["baseline", "--data", str(tmp_path), "--limit-steps", "1", "--channels", "rgbd"]) == 0
    monkeypatch.setattr(datasets, "baseline", lambda *a, **kw: 1.0)
    assert datasets.main(["baseline", "--data", str(tmp_path)]) == 1


@pytest.mark.ml
def test_pack_actions_and_model_components_have_expected_shapes():
    import torch

    actions = np.zeros((1, 2, 10), dtype=np.uint8)
    actions[0, 1, [0, 1, 3]] = [2, 1, 3]
    packed = baseline.pack_actions(actions)
    assert packed.tolist() == [[0, (2 + 5 + 30) % 63]]
    assert packed.dtype == np.int64

    model = baseline._build_model(in_ch=3)
    x = torch.zeros(2, 3, 64, 64)
    online = model.online.encoder(x)
    target = model.target_encoder(x)
    assert tuple(online.shape) == (2, 1024)
    assert torch.allclose(online, target)
    pred = model.online(online.reshape(2, 1, -1), torch.zeros(2, 1, dtype=torch.long))
    assert tuple(pred.shape) == (2, 1, 1024)
    assert tuple(model.decoder(online).shape) == (2, 3, 64, 64)
    with torch.no_grad():
        next(model.online.encoder.parameters()).add_(1)
    before = next(model.target_encoder.parameters()).clone()
    model.ema_update()
    assert not torch.equal(before, next(model.target_encoder.parameters()))


class _SplitDataset:
    def __init__(self, _data, seq_len, split):
        self.shards = [f"{split}.parquet"]
        self.split = split

    def _load(self, _index):
        n = 6
        rgb = np.stack([np.full((128, 128, 3), i * 20, np.uint8) for i in range(n)])
        actions = np.zeros((n, 10), np.uint8)
        actions[:, 0] = np.arange(n) % 5
        depth = np.stack([np.full((128, 128), i + 1, np.float16) for i in range(n)])
        return rgb, actions, depth, np.zeros((n, 128, 128), np.uint16)


def test_load_split_builds_strided_windows_and_requires_depth(monkeypatch):
    monkeypatch.setattr(baseline, "VoxelSequenceDataset", _SplitDataset)
    rgbs, acts, wins, deps = baseline._load_split("unused", 2, "train", stride=2, with_depth=True)
    assert len(rgbs) == len(acts) == len(deps) == 1
    assert wins == [(0, 0), (0, 1), (0, 2), (0, 3)]

    class NoDepth(_SplitDataset):
        def _load(self, index):
            rgb, actions, _depth, seg = super()._load(index)
            return rgb, actions, None, seg

    monkeypatch.setattr(baseline, "VoxelSequenceDataset", NoDepth)
    with pytest.raises(ValueError, match="needs depth"):
        baseline._load_split("unused", 2, "train", with_depth=True)


@pytest.mark.parametrize("channels", ["rgb", "rgbd"])
@pytest.mark.ml
def test_run_baseline_trains_rgb_and_rgbd_with_tiny_model(
    monkeypatch, capsys, channels
):
    import torch
    import torch.nn as nn

    monkeypatch.setattr(baseline, "VoxelSequenceDataset", _SplitDataset)

    class Encoder(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(channels, 4)

        def forward(self, x):
            return self.fc(self.pool(x).flatten(1))

    class Online(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.encoder = Encoder(channels)
            self.act = nn.Embedding(64, 4)
            self.head = nn.Linear(8, 4)

        def forward(self, latents, action_ids, h=None):
            del h
            return self.head(torch.cat([latents, self.act(action_ids)], dim=-1))

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)

        def forward(self, z):
            return torch.sigmoid(self.fc(z))[:, :, None, None].expand(-1, -1, 64, 64)

    class Tiny(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.online = Online(channels)
            self.target_encoder = Encoder(channels)
            self.target_encoder.load_state_dict(self.online.encoder.state_dict())
            for p in self.target_encoder.parameters():
                p.requires_grad = False
            self.decoder = Decoder()

        def ema_update(self):
            for target, online in zip(self.target_encoder.parameters(), self.online.encoder.parameters()):
                target.data.mul_(0.9).add_(online.data, alpha=0.1)

    monkeypatch.setattr(baseline, "_build_model", lambda in_ch: Tiny(in_ch))
    ratio = baseline.run_baseline(
        "train", steps=1, batch=2, seq_len=2, lr=1e-3, limit_steps=1,
        stride=2, channels=channels, transfer_data="transfer",
    )
    assert np.isfinite(ratio)
    out = capsys.readouterr().out
    assert f"channels={channels}" in out and "TRANSFER (transfer)" in out


@pytest.mark.ml
def test_run_baseline_rejects_empty_window_sets(monkeypatch):
    monkeypatch.setattr(baseline, "_load_split", lambda *a, **kw: ([], [], [], []))
    with pytest.raises(RuntimeError, match="not enough data"):
        baseline.run_baseline("x", 1, 1, 2, 1e-3, 1)
    with pytest.raises(AssertionError):
        baseline.run_baseline("x", 1, 1, 2, 1e-3, 1, channels="invalid")


class _WorkerPipe:
    def __init__(self, incoming):
        self.incoming = iter(incoming)
        self.sent = []
        self.closed = False

    def recv(self):
        return next(self.incoming)

    def send(self, value):
        self.sent.append(value)

    def close(self):
        self.closed = True


def test_vector_worker_protocol(monkeypatch):
    class Batch:
        def __init__(self, specs):
            self.specs = specs

        def step_batch_np(self, rows):
            return [False] * len(rows)

        def obs_voxels_batch(self):
            return np.zeros((2, 1), np.uint16)

        def obs_inventory_batch(self):
            return np.ones((2, 1), np.uint16)

        def obs_pose_batch(self):
            return np.full((2, 1), 2.0)

        def obs_raycast_batch(self):
            return np.full((2, 1), 3, np.uint16)

        def hashes(self):
            return [10, 11]

    monkeypatch.setitem(sys.modules, "voxelgym_rs", SimpleNamespace(PyWorldBatch=Batch))
    actions = np.zeros((2, 10), np.uint8)
    pipe = _WorkerPipe([("step", actions), ("bench", np.stack([actions, actions])), ("hashes", None), ("close", None)])
    vec._worker(pipe, [(1, "flat"), (2, "flat")])
    obs, dead = pipe.sent[0]
    assert set(obs) == {"voxels", "inventory", "pose", "raycast"}
    assert dead.tolist() == [False, False]
    assert isinstance(pipe.sent[1], float) and pipe.sent[1] >= 0
    assert pipe.sent[2] == [10, 11]
    assert pipe.closed


def test_sharded_vector_env_splits_aggregates_and_closes(monkeypatch):
    created = []

    class Parent:
        def __init__(self, index):
            self.index = index
            self.response = None

        def send(self, message):
            cmd, payload = message
            if cmd == "step":
                n = len(payload)
                self.response = ({"pose": np.full((n, 2), self.index)}, np.full(n, self.index % 2, bool))
            elif cmd == "hashes":
                self.response = [self.index * 10 + i for i in range(2 if self.index == 0 else 1)]

        def recv(self):
            return self.response

    class Process:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False
            self.joined = False
            self.terminated = False

        def start(self):
            self.started = True

        def join(self, timeout):
            self.joined = timeout == 5

        def is_alive(self):
            return self.kwargs["args"][1][0][0] == 0

        def terminate(self):
            self.terminated = True

    class Context:
        def Pipe(self):
            parent = Parent(len(created))
            created.append(parent)
            return parent, object()

        def Process(self, **kwargs):
            return Process(**kwargs)

    monkeypatch.setattr(vec.mp, "get_context", lambda mode: Context())
    env = vec.ShardedVectorEnv(3, num_shards=2, seed=0)
    assert env._sizes == [2, 1]
    obs, dead = env.step(np.zeros((3, 10), np.int64))
    assert obs["pose"].tolist() == [[0, 0], [0, 0], [1, 1]]
    assert dead.tolist() == [False, False, True]
    assert env.hashes() == [0, 1, 10]
    with pytest.raises(AssertionError):
        env.step(np.zeros((2, 10), np.uint8))
    assert env.__enter__() is env
    env.__exit__()
    env.close()
    assert env._closed and all(p.joined for p in env._procs)
    assert env._procs[0].terminated


class _RecordedWorld:
    def __init__(self, tick=600):
        self._tick = tick

    def tick(self):
        return self._tick

    def snapshot(self):
        return b"snapshot"

    def obs_voxels_bytes(self):
        return b"voxels"

    def obs_inventory_bytes(self):
        return b"inventory"


def test_recorder_flushes_frames_and_writes_sidecar(tmp_path, monkeypatch):
    rec = recorder.Recorder(str(tmp_path), "probe", 7, render=True)
    rec.FLUSH_EVERY = 1
    frames = (
        np.zeros((2, 3, 3), np.uint8),
        np.ones((2, 3), np.float16),
        np.full((2, 3), 2, np.uint16),
    )
    rec.log(_RecordedWorld(), tuple(range(10)), 1.25, True, frames, swap=4)
    path = rec.save(0xCAFE)
    rows = pq.read_table(path).to_pylist()
    assert rows[0]["world_ckpt"] == b"snapshot"
    assert rows[0]["swap"] == 4 and rows[0]["rgb"] == frames[0].tobytes()
    sidecar = json.loads((tmp_path / (rec._stem + ".json")).read_text())
    assert sidecar["final_hash"] == 0xCAFE and sidecar["steps"] == 1
    rec._flush()  # empty flush is a no-op

    empty = recorder.Recorder(str(tmp_path), "empty", 1)
    empty_path = empty.save(0)
    assert pq.read_metadata(empty_path).num_rows == 0


def test_replay_swap_checkpoint_mismatch_and_cli(monkeypatch, tmp_path, capsys):
    rows = [{**{k: 0 for k in ACTION_KEYS}, "swap": 3, "world_ckpt": b"bad", "tick": 1}]
    monkeypatch.setattr(replay, "_load", lambda path: ({"task": "collect_log", "seed": 2, "final_hash": 9}, rows))

    class World:
        def __init__(self, hash_value):
            self.hash_value = hash_value
            self.swaps = []

        def swap_to_hotbar(self, item):
            self.swaps.append(item)

        def step(self, action):
            self.action = action

        def hash(self):
            return self.hash_value

    live = World(1)
    scratch = World(2)
    scratch.restore = lambda snapshot: None

    class Env:
        def __init__(self, **kwargs):
            self.world = live

        def reset(self, **kwargs):
            return None

    monkeypatch.setitem(sys.modules, "voxelgym_rs", SimpleNamespace(PyWorld=lambda *a: scratch))
    monkeypatch.setattr("voxelgym.tasks.make_task", lambda name: SimpleNamespace(preset="flat"))
    monkeypatch.setattr("voxelgym.env.VoxelGymEnv", Env)
    assert not replay.verify("x.parquet")
    assert live.swaps == [3]
    assert "CHECKPOINT MISMATCH" in capsys.readouterr().out

    monkeypatch.setattr(replay, "verify", lambda path: path == "good.parquet")
    assert replay.main(["good.parquet"]) == 2
    assert replay.main(["good.parquet", "--verify"]) == 0
    assert replay.main(["bad.parquet", "--verify"]) == 1
