"""VQA train + eval harness.

    python -m voxelgym.vqa.train --data data/vqa --arm <arm> [--steps 3000 --batch 64]
    python -m voxelgym.vqa.train --data data/vqa --arm all        # full 5-arm table

Split BY EPISODE: sample ids are task/seed/tick; seeds 0-1 of each task are
test (no frame leakage across splits), the rest train. The question vocab is
built from TRAIN questions only. Loss = cross-entropy on each sample's own
family head. Majority baseline (per-family most-frequent train label) is
printed beside model accuracy.

Acceptance gate (printed at the end of an --arm all run): best-arm macro
accuracy over families with needs != {"prior"} >= majority macro + 25 pts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np

from .families import FAMILIES
from .model import ARM_COMPONENTS, build_vqa_model

TEST_SEEDS = (0, 1)
VOCAB_CAP = 256
FAMILY_ORDER = [f.name for f in FAMILIES]
NEEDS = {f.name: f.needs for f in FAMILIES}
DERIVABLE = [f.name for f in FAMILIES if f.needs != frozenset({"prior"})]


# ---------------- data ----------------


class Dataset:
    """Manifest rows joined to per-task npz tensors by sample id."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.items = []
        with open(os.path.join(data_dir, "manifest.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(l) for l in f]
        self._npz: dict[str, dict[str, np.ndarray]] = {}
        self._index: dict[str, dict[str, int]] = {}
        for r in rows:
            self.items.append(r)

    def _task_arrays(self, task: str):
        if task not in self._npz:
            npz = np.load(os.path.join(self.data_dir, f"{task}.npz"))
            self._npz[task] = {k: npz[k] for k in npz.files}
            self._index[task] = {sid: i for i, sid in enumerate(self._npz[task]["id"].tolist())}
        return self._npz[task]


def split_items(items):
    train, test = [], []
    for it in items:
        (test if it["seed"] in TEST_SEEDS else train).append(it)
    return train, test


_TOKEN_RE = re.compile(r"[\w-]+")


def build_vocab(train_items) -> dict[str, int]:
    freq: Counter = Counter()
    for it in train_items:
        freq.update(_TOKEN_RE.findall(it["q_en"].lower()))
    toks = sorted(freq, key=lambda t: (-freq[t], t))[: VOCAB_CAP - 2]
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({t: i + 2 for i, t in enumerate(toks)})
    return vocab


def encode_q(q: str, vocab: dict[str, int]) -> list[int]:
    return [vocab.get(t, 1) for t in _TOKEN_RE.findall(q.lower())]


def collate(ds: Dataset, items, comps: tuple[str, ...], vocab: dict[str, int]):
    import torch

    batch: dict[str, torch.Tensor] = {}
    if "rgb" in comps:
        rgb = np.stack([ds._task_arrays(it["task"])["rgb"][ds._index[it["task"]][it["id"]]] for it in items])
        batch["rgb"] = torch.from_numpy(rgb).permute(0, 3, 1, 2)
    if "depth" in comps:
        dep = np.stack([ds._task_arrays(it["task"])["depth"][ds._index[it["task"]][it["id"]]] for it in items])
        nrm = np.stack([ds._task_arrays(it["task"])["normals"][ds._index[it["task"]][it["id"]]] for it in items])
        batch["depth"] = torch.from_numpy(dep.astype(np.float32)).unsqueeze(1)
        batch["normals"] = torch.from_numpy(nrm).permute(0, 3, 1, 2)
    if "lidar" in comps:
        lid = np.stack([ds._task_arrays(it["task"])["lidar_range"][ds._index[it["task"]][it["id"]]] for it in items])
        batch["lidar_range"] = torch.from_numpy(lid)
    if "voxels" in comps:
        vox = np.stack([ds._task_arrays(it["task"])["voxels"][ds._index[it["task"]][it["id"]]] for it in items])
        batch["voxels"] = torch.from_numpy(vox.astype(np.int32))
    q_ids = [encode_q(it["q_en"], vocab) for it in items]
    L = max(1, max(len(q) for q in q_ids))
    qarr = np.zeros((len(items), L), dtype=np.int64)
    mask = np.zeros((len(items), L), dtype=np.float32)
    for i, q in enumerate(q_ids):
        qarr[i, : len(q)] = q
        mask[i, : len(q)] = 1.0
    batch["q_ids"] = torch.from_numpy(qarr)
    batch["q_mask"] = torch.from_numpy(mask)
    labels = torch.tensor([it["answer"] for it in items], dtype=torch.long)
    return batch, labels


# ---------------- train / eval ----------------


def train_arm(ds: Dataset, arm: str, train_items, test_items, vocab,
              steps: int = 3000, batch: int = 64, lr: float = 3e-4,
              limit_steps: int | None = None, eval_every: int = 1000, seed: int = 0):
    import torch
    import torch.nn.functional as F

    comps = ARM_COMPONENTS[arm]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_vqa_model(set(comps), FAMILY_ORDER, vocab)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # per-family majority from TRAIN labels (the baseline to beat)
    maj_label = {}
    for fam in FAMILY_ORDER:
        c = Counter(it["answer"] for it in train_items if it["family"] == fam)
        if c:
            maj_label[fam] = c.most_common(1)[0][0]
    maj_accs = {}
    for fam in FAMILY_ORDER:
        fam_items = [it for it in test_items if it["family"] == fam]
        if fam_items and fam in maj_label:
            maj_accs[fam] = sum(it["answer"] == maj_label[fam] for it in fam_items) / len(fam_items)

    def evaluate(items):
        model.eval()
        per_fam: dict[str, tuple[int, int]] = defaultdict(lambda: [0, 0])
        with torch.no_grad():
            for i in range(0, len(items), 256):
                chunk = items[i : i + 256]
                b, labels = collate(ds, chunk, comps, vocab)
                z = model.encode(b)
                by_fam: dict[str, list[int]] = defaultdict(list)
                for j, it in enumerate(chunk):
                    by_fam[it["family"]].append(j)
                for fam, idxs in by_fam.items():
                    idx = torch.tensor(idxs, dtype=torch.long)
                    pred = model.head(fam, z[idx]).argmax(-1)
                    correct = int((pred == labels[idx]).sum())
                    per_fam[fam][0] += correct
                    per_fam[fam][1] += len(idxs)
        model.train()
        return {f: (c / n if n else None) for f, (c, n) in per_fam.items()}

    def macro(accs, fams):
        vals = [accs[f] for f in fams if accs.get(f) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    n_steps = limit_steps or steps
    t0 = time.time()
    first_loss = None
    history = []
    model.train()
    for step in range(1, n_steps + 1):
        idx = rng.integers(0, len(train_items), size=batch)
        chunk = [train_items[i] for i in idx]
        b, labels = collate(ds, chunk, comps, vocab)
        z = model.encode(b)
        by_fam: dict[str, list[int]] = defaultdict(list)
        for j, it in enumerate(chunk):
            by_fam[it["family"]].append(j)
        loss = 0.0
        for fam, idxs in by_fam.items():
            idx_t = torch.tensor(idxs, dtype=torch.long)
            loss = loss + F.cross_entropy(model.head(fam, z[idx_t]), labels[idx_t])
        loss = loss / max(1, len(by_fam))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
        if step % 100 == 0 or step == 1:
            rate = step / (time.time() - t0)
            print(f"[{arm}] step {step}/{n_steps} loss {loss.item():.4f} ({rate:.1f} steps/s)", flush=True)
            history.append((step, loss.item()))
        if step % eval_every == 0:
            accs = evaluate(test_items)
            per_fam = " ".join(f"{f}={accs[f]:.2f}" for f in FAMILY_ORDER if accs.get(f) is not None)
            print(f"[{arm}] [eval@{step}] macro={macro(accs, FAMILY_ORDER):.3f} "
                  f"macro_derivable={macro(accs, DERIVABLE):.3f} "
                  f"majority_derivable={macro(maj_accs, DERIVABLE):.3f}\n[{arm}]   {per_fam}", flush=True)

    accs = evaluate(test_items)
    return {
        "arm": arm,
        "acc": accs,
        "majority": maj_accs,
        "first_loss": first_loss,
        "history": history,
        "macro": macro(accs, FAMILY_ORDER),
        "macro_derivable": macro(accs, DERIVABLE),
    }


def print_table(results: list[dict], majority: dict[str, float]):
    arms = [r["arm"] for r in results]
    fams = [f for f in FAMILY_ORDER if any(r["acc"].get(f) is not None for r in results)]
    header = f"{'family':<14}" + "".join(f"{a:>26}" for a in arms) + f"{'majority':>10}"
    print("\n==== VQA per-family x per-arm test accuracy (split: seeds 0-1 per task) ====")
    print(header)
    for fam in fams:
        row = f"{fam:<14}"
        for r in results:
            v = r["acc"].get(fam)
            row += f"{(v * 100):>25.1f}%" if v is not None else f"{'-':>26}"
        m = majority.get(fam)
        row += f"{(m * 100):>9.1f}%" if m is not None else f"{'-':>10}"
        marker = "" if fam in DERIVABLE else "  (prior-only: excluded from gate)"
        print(row + marker)

    def macro_over(fams_, get):
        vals = [get(f) for f in fams_]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    row = f"{'MACRO(derivable)':<14}"
    for r in results:
        row += f"{macro_over(DERIVABLE, lambda f: r['acc'].get(f)) * 100:>25.1f}%"
    maj_macro = macro_over(DERIVABLE, lambda f: majority.get(f))
    row += f"{maj_macro * 100:>9.1f}%"
    print(row)
    best = max(results, key=lambda r: macro_over(DERIVABLE, lambda f: r["acc"].get(f)))
    best_macro = macro_over(DERIVABLE, lambda f: best["acc"].get(f))
    margin = (best_macro - maj_macro) * 100
    verdict = "PASS" if margin >= 25.0 else "FAIL"
    print(f"ACCEPTANCE: best-arm derivable macro = {best_macro * 100:.1f}% ({best['arm']}) "
          f"vs majority {maj_macro * 100:.1f}% -> margin {margin:+.1f} pts (gate +25): {verdict}")
    return verdict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/vqa")
    ap.add_argument("--arm", default="all", choices=list(ARM_COMPONENTS) + ["all"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit-steps", type=int, default=None,
                    help="override --steps (smoke runs)")
    args = ap.parse_args(argv)

    ds = Dataset(args.data)
    train_items, test_items = split_items(ds.items)
    vocab = build_vocab(train_items)
    print(f"items: {len(ds.items)} (train {len(train_items)}, test {len(test_items)}), "
          f"vocab {len(vocab)}", flush=True)
    fam_counts = Counter(it["family"] for it in ds.items)
    print("per-family items:", dict(sorted(fam_counts.items())), flush=True)

    arms = list(ARM_COMPONENTS) if args.arm == "all" else [args.arm]
    results = []
    for arm in arms:
        t0 = time.time()
        r = train_arm(ds, arm, train_items, test_items, vocab,
                      steps=args.steps, batch=args.batch, limit_steps=args.limit_steps)
        r["wall_s"] = time.time() - t0
        results.append(r)
        print(f"[{arm}] done in {r['wall_s']:.0f}s; macro_derivable={r['macro_derivable']:.3f}", flush=True)
    verdict = print_table(results, results[0]["majority"])
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
