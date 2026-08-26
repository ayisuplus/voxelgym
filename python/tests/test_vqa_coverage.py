"""Tiny CPU tests for VQA encoders, data joins, training, and reporting."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytestmark = pytest.mark.ml
pytest.importorskip("torch", reason="VQA model and train modules require the ML extra")

from voxelgym.vqa import model as vqa_model
from voxelgym.vqa import train


@pytest.mark.ml
def test_real_modality_and_question_encoders_return_contract_shapes():
    import torch

    conv = vqa_model.ConvEncoder(3, out=7)
    assert tuple(conv(torch.zeros(2, 3, 64, 64)).shape) == (2, 7)
    lidar = vqa_model.LidarEncoder(out=9)
    assert tuple(lidar(torch.zeros(2, 1, 16, 256)).shape) == (2, 9)
    voxel = vqa_model.VoxelEncoder(out=11)
    cells = torch.zeros(2, 2, 3, 2, dtype=torch.long)
    cells[1] = 1 | (3 << 12)
    assert tuple(voxel(cells).shape) == (2, 11)

    question = vqa_model.QuestionEncoder(vocab_size=4, dim=5)
    q_ids = torch.tensor([[2, 3], [0, 0]])
    q_mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    encoded = question(q_ids, q_mask)
    assert tuple(encoded.shape) == (2, 5)
    assert torch.equal(encoded[1], torch.zeros(5)), "empty questions must not divide by zero"


@pytest.mark.parametrize("arm, components", list(vqa_model.ARM_COMPONENTS.items()))
@pytest.mark.ml
def test_every_vqa_arm_routes_only_its_declared_modalities(monkeypatch, arm, components):
    import torch
    import torch.nn as nn

    class TinyVisual(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return torch.ones(x.shape[0], 256)

    monkeypatch.setattr(vqa_model, "ConvEncoder", TinyVisual)
    monkeypatch.setattr(vqa_model, "LidarEncoder", TinyVisual)
    monkeypatch.setattr(vqa_model, "VoxelEncoder", TinyVisual)
    model = vqa_model.VQAModel(set(components), {"door_state": 2}, vocab_size=5)
    batch = {
        "rgb": torch.full((2, 3, 4, 4), 255, dtype=torch.uint8),
        "depth": torch.full((2, 1, 4, 4), 96.0),
        "normals": torch.zeros(2, 3, 4, 4),
        "lidar_range": torch.full((2, 16, 256), 96.0),
        "voxels": torch.zeros(2, 2, 2, 2, dtype=torch.int32),
        "q_ids": torch.tensor([[2], [3]]),
        "q_mask": torch.ones(2, 1),
    }
    z = model.encode(batch)
    assert tuple(z.shape) == (2, 256)
    assert tuple(model.head("door_state", z).shape) == (2, 2)
    assert model.arm_set == set(components)


def _write_vqa_fixture(path):
    path.mkdir()
    items = [
        {"id": "task/0/0", "task": "task", "seed": 0, "family": "door_state", "q_en": "", "answer": 0},
        {"id": "task/2/0", "task": "task", "seed": 2, "family": "door_state", "q_en": "Open door?", "answer": 1},
        {"id": "task/3/0", "task": "task", "seed": 3, "family": "door_state", "q_en": "door open", "answer": 1},
    ]
    (path / "manifest.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    n = len(items)
    np.savez_compressed(
        path / "task.npz",
        id=np.array([i["id"] for i in items]),
        rgb=np.zeros((n, 4, 4, 3), np.uint8),
        depth=np.ones((n, 4, 4), np.float16),
        normals=np.zeros((n, 4, 4, 3), np.float16),
        lidar_range=np.ones((n, 16, 256), np.float16),
        voxels=np.zeros((n, 2, 2, 2), np.uint16),
    )
    return items


@pytest.mark.ml
def test_dataset_cache_split_vocab_unknown_and_all_component_collate(tmp_path, monkeypatch):
    data = tmp_path / "vqa"
    items = _write_vqa_fixture(data)
    ds = train.Dataset(str(data))
    arrays = ds._task_arrays("task")
    assert ds._task_arrays("task") is arrays
    assert ds._index["task"]["task/3/0"] == 2

    train_items, test_items = train.split_items(ds.items)
    assert [it["seed"] for it in test_items] == [0]
    assert [it["seed"] for it in train_items] == [2, 3]
    monkeypatch.setattr(train, "VOCAB_CAP", 4)
    vocab = train.build_vocab(train_items)
    assert vocab == {"<pad>": 0, "<unk>": 1, "door": 2, "open": 3}
    assert train.encode_q("MISSING door!", vocab) == [1, 2]

    # Include the empty test question: collate must emit a length-one, all-zero mask.
    batch, labels = train.collate(ds, [items[0]], ("rgb", "depth", "lidar", "voxels"), vocab)
    assert set(batch) == {"rgb", "depth", "normals", "lidar_range", "voxels", "q_ids", "q_mask"}
    assert tuple(batch["rgb"].shape) == (1, 3, 4, 4)
    assert tuple(batch["depth"].shape) == (1, 1, 4, 4)
    assert tuple(batch["voxels"].shape) == (1, 2, 2, 2)
    assert batch["q_mask"].tolist() == [[0.0]] and labels.tolist() == [0]


@pytest.mark.ml
def test_train_arm_runs_one_step_and_groups_family_evaluation(tmp_path, monkeypatch):
    import torch
    import torch.nn as nn

    data = tmp_path / "vqa"
    _write_vqa_fixture(data)
    ds = train.Dataset(str(data))
    train_items, test_items = train.split_items(ds.items)
    vocab = train.build_vocab(train_items)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, 4)
            self.heads = nn.ModuleDict({name: nn.Linear(4, 16) for name in train.FAMILY_ORDER})

        def encode(self, batch):
            value = batch["q_mask"].sum(1, keepdim=True)
            return self.proj(value)

        def head(self, family, z):
            return self.heads[family](z)

    monkeypatch.setattr(train, "build_vqa_model", lambda *a, **kw: TinyModel())
    result = train.train_arm(
        ds, "voxels", train_items, test_items, vocab,
        steps=10, batch=2, limit_steps=1, eval_every=1, seed=4,
    )
    assert result["arm"] == "voxels"
    assert result["first_loss"] is not None
    assert result["acc"]["door_state"] in (0.0, 1.0)
    assert result["majority"]["door_state"] == 0.0
    assert result["history"] == [(1, result["first_loss"])]


def _table_result(arm, accuracy):
    acc = {family: accuracy for family in train.DERIVABLE}
    acc["hazard_near"] = accuracy
    return {"arm": arm, "acc": acc}


def test_accuracy_table_prints_pass_fail_missing_and_prior_markers(capsys):
    majority = {family: 0.5 for family in train.DERIVABLE}
    majority["hazard_near"] = 0.5
    passing = train.print_table([_table_result("rgb", 0.8)], majority)
    assert passing == "PASS"
    out = capsys.readouterr().out
    assert "ACCEPTANCE" in out and "prior-only" in out and "PASS" in out

    weak = _table_result("voxels", 0.6)
    weak["acc"][train.DERIVABLE[0]] = None
    assert train.print_table([weak], majority) == "FAIL"
    assert "FAIL" in capsys.readouterr().out


def test_train_cli_dispatches_single_and_all_arms(tmp_path, monkeypatch, capsys):
    data = tmp_path / "vqa"
    _write_vqa_fixture(data)
    calls = []

    def fake_train(ds, arm, train_items, test_items, vocab, **kwargs):
        calls.append((arm, kwargs["limit_steps"]))
        return {
            "arm": arm, "acc": {"door_state": 1.0}, "majority": {"door_state": 0.0},
            "first_loss": 0.1, "history": [], "macro": 1.0, "macro_derivable": 1.0,
        }

    monkeypatch.setattr(train, "train_arm", fake_train)
    monkeypatch.setattr(train, "print_table", lambda results, majority: "PASS")
    assert train.main(["--data", str(data), "--arm", "voxels", "--limit-steps", "1"]) == 0
    assert calls == [("voxels", 1)]
    calls.clear()
    assert train.main(["--data", str(data), "--arm", "all", "--limit-steps", "2"]) == 0
    assert [arm for arm, _ in calls] == list(vqa_model.ARM_COMPONENTS)
    assert "per-family items" in capsys.readouterr().out

    monkeypatch.setattr(train, "print_table", lambda results, majority: "FAIL")
    assert train.main(["--data", str(data), "--arm", "all", "--limit-steps", "2"]) == 1
