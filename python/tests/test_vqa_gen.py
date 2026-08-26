"""VQA generator determinism: identical args -> byte-identical manifest."""

import json

import numpy as np

from voxelgym.vqa import gen


def test_gen_manifest_byte_identical(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    tasks = ["logic_probe"]
    gen.run(tasks, 2, str(a))
    gen.run(tasks, 2, str(b))
    ma = (a / "manifest.jsonl").read_bytes()
    mb = (b / "manifest.jsonl").read_bytes()
    assert ma == mb, "manifest not byte-identical across runs"
    rows = [json.loads(l) for l in ma.decode("utf-8").splitlines()]
    assert rows, "empty manifest"
    # every row joins to a tensor index in the task npz, and carries the
    # plan's schema
    npz = np.load(a / "logic_probe.npz")
    ids = set(npz["id"].tolist())
    fams = set()
    for r in rows:
        assert r["id"] in ids
        assert {"id", "task", "seed", "tick", "family", "q_en", "q_zh", "answer", "needs"} <= set(r)
        fams.add(r["family"])
    # circuit-task episode coverage: logic_probe is a circuit task -> 2x episodes
    assert len({(r["seed"]) for r in rows}) == 4
    # the circuit-only families must be present on a circuit task
    assert {"lamp_state", "lever_combo"} <= fams
