"""VQA baseline: per-modality encoders + concat fusion + one head per
question family. CPU-sized; an in-repo validation harness like
baseline.py, NOT a production VLM.

Arms (CLI names decompose into encoder components):
  rgb                          -> ConvEncoder(3ch)
  rgb+depth                    -> + ConvEncoder(4ch: depth/96 + normals)
  rgb+depth+lidar              -> + 2D CNN over the (1,16,256) range image
  rgb+depth+lidar+voxels       -> + cell-embedding mean over (21,11,21)
  voxels                       -> voxels alone

seg/raycast/world-state are label SOURCES, never model inputs (they would
leak the answer).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ARM_COMPONENTS = {
    "rgb": ("rgb",),
    "rgb+depth": ("rgb", "depth"),
    "rgb+depth+lidar": ("rgb", "depth", "lidar"),
    "rgb+depth+lidar+voxels": ("rgb", "depth", "lidar", "voxels"),
    "voxels": ("voxels",),
}

N_BLOCKS = 29  # registry size (ids.py): block ids 0..28


class ConvEncoder(nn.Module):
    """4-conv stack; input (B,C,128,128) pooled to 64x64 by the caller.
    Copied from voxelgym/baseline.py:30-45 (no shared utility exists; the
    BYOL wrapper and unit-sphere normalize are baseline-specific and not
    copied)."""

    def __init__(self, in_ch: int, out: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.ELU(),   # 64->32
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ELU(),      # 32->16
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ELU(),     # 16->8
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ELU(),    # 8->4
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, out),
        )

    def forward(self, x):
        return self.net(x)


class LidarEncoder(nn.Module):
    """2D CNN over the (1,16,256) range image (range/48 clamp [0,1] done by
    the caller)."""

    def __init__(self, out: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=(2, 2), padding=1), nn.ELU(),  # (32,8,128)
            nn.Flatten(),
            nn.Linear(32 * 8 * 128, out),
        )

    def forward(self, x):
        return self.net(x)


class VoxelEncoder(nn.Module):
    """cell = id | state<<12. Plan spec: Embedding(29,16)(id) +
    Linear(16,16)(state/15); state/15 is scalar per cell, so the 16-d input
    is realized as one_hot(state, 16) — a learned linear map either way."""

    def __init__(self, out: int = 256):
        super().__init__()
        self.id_emb = nn.Embedding(N_BLOCKS, 16)
        self.st_fc = nn.Linear(16, 16)
        self.mlp = nn.Sequential(nn.Linear(16, 128), nn.ELU(), nn.Linear(128, out))

    def forward(self, cells):  # (B,21,11,21) int64 raw cells
        bid = cells & 0xFFF
        st = (cells >> 12).clamp(0, 15)
        h = self.id_emb(bid) + self.st_fc(F.one_hot(st, 16).float())
        return self.mlp(h.mean(dim=(1, 2, 3)))


class QuestionEncoder(nn.Module):
    """Whitespace vocab (cap 256, <pad>=0, <unk>=1), Embedding(64),
    masked mean-pool."""

    def __init__(self, vocab_size: int, dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)

    def forward(self, q_ids, q_mask):
        e = self.emb(q_ids) * q_mask.unsqueeze(-1)
        return e.sum(1) / q_mask.sum(1, keepdim=True).clamp(min=1)


class VQAModel(nn.Module):
    def __init__(self, arms: set[str], families: dict[str, int], vocab_size: int):
        super().__init__()
        self.arm_set = set(arms)
        if "rgb" in arms:
            self.rgb = ConvEncoder(3)
        if "depth" in arms:
            self.depth = ConvEncoder(4)
        if "lidar" in arms:
            self.lidar = LidarEncoder()
        if "voxels" in arms:
            self.voxels = VoxelEncoder()
        self.q = QuestionEncoder(vocab_size)
        d = 64 + 256 * len(self.arm_set)
        self.fuse = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 512), nn.ELU(), nn.Linear(512, 256),
        )
        self.heads = nn.ModuleDict({f: nn.Linear(256, n) for f, n in families.items()})

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        if "rgb" in self.arm_set:
            x = batch["rgb"].float() / 255.0
            parts.append(self.rgb(F.avg_pool2d(x, 2)))
        if "depth" in self.arm_set:
            d = batch["depth"].float() / 96.0  # metric cells; baseline.py forward_lat norm
            x = torch.cat([d, batch["normals"].float()], dim=1)
            parts.append(self.depth(F.avg_pool2d(x, 2)))
        if "lidar" in self.arm_set:
            r = (batch["lidar_range"].float() / 48.0).clamp(0, 1).unsqueeze(1)
            parts.append(self.lidar(r))
        if "voxels" in self.arm_set:
            parts.append(self.voxels(batch["voxels"].long()))
        parts.append(self.q(batch["q_ids"], batch["q_mask"]))
        return self.fuse(torch.cat(parts, dim=-1))

    def head(self, family: str, z: torch.Tensor) -> torch.Tensor:
        return self.heads[family](z)


def build_vqa_model(arms: set[str], families: list[str], vocab: dict) -> VQAModel:
    from .families import FAMILY_BY_NAME

    n_classes = {name: len(FAMILY_BY_NAME[name].classes) for name in families}
    return VQAModel(arms, n_classes, len(vocab))
