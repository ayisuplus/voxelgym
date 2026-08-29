"""RSSM-lite baseline: does the recorded action-conditioned video carry
learnable dynamics? Validation tool, not a deliverable model.

Architecture (contract): CNN encoder 128^2 -> 1024-dim latent, GRU(512),
action embedding(64); decoder head predicts the NEXT frame's latent.
Loss = latent MSE. Control = copy-last-latent baseline. Acceptance: test
latent-MSE ratio model/copy < 0.9.

Implementation notes:
- the latent target is stop-gradient (BYOL-style). Without an anchor-free
  pure-MSE objective the encoder could collapse to a constant and make the
  ratio vacuous; stop-grad prevents that while keeping the loss a pure
  latent MSE.
- frames are center-cropped... not cropped: downsampled x2 by area pooling
  to keep CPU training tractable; the encoder sees (B*T, 3, 64, 64).
"""

from __future__ import annotations

import time

import numpy as np

from .datasets import VoxelSequenceDataset


def _build_model(in_ch: int = 3):
    import torch
    import torch.nn as nn

    class Encoder(nn.Module):
        def __init__(self, latent=1024):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.ELU(),  # 64->32
                nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ELU(),  # 32->16
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ELU(), # 16->8
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ELU(),# 8->4
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, latent),  # 64px input path
            )

        def forward(self, x):
            # unit-sphere latents: keeps latent MSE scale-stable (bounded in
            # [0, 4]); unnormalized latents blew up in the first long run
            return torch.nn.functional.normalize(self.net(x), dim=-1)

    class Decoder(nn.Module):
        """64x64 RGB reconstruction head — anchors the latents (random
        unanchored latents concentrate: copy-last then beats any model)."""

        def __init__(self, latent=1024):
            super().__init__()
            self.fc = nn.Linear(latent, 256 * 4 * 4)
            self.net = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ELU(),  # 4->8
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ELU(),   # 8->16
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ELU(),    # 16->32
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),               # 32->64
            )

        def forward(self, lat):
            h = self.fc(lat).reshape(-1, 256, 4, 4)
            return torch.sigmoid(self.net(h))

    class RSSMLite(nn.Module):
        def __init__(self, latent=1024, hidden=512, act_emb=64):
            super().__init__()
            self.encoder = Encoder(latent)
            self.act = nn.Embedding(64, act_emb)  # packed action id
            self.inp = nn.Linear(latent + act_emb, hidden)
            self.gru = nn.GRUCell(hidden, hidden)
            self.head = nn.Sequential(nn.Linear(hidden, latent), nn.LayerNorm(latent))

        def forward(self, latents, action_ids, h=None):
            # latents: (T, B, latent); action_ids: (T, B) int64
            outs = []
            for t in range(latents.shape[0]):
                a = self.act(action_ids[t])
                x = torch.cat([latents[t], a], dim=-1)
                h = self.gru(self.inp(x), h)
                outs.append(self.head(h))
            return torch.stack(outs)

    class BYOLModel(nn.Module):
        """Online encoder + GRU + predictor, with an EMA target encoder.
        Keeps the loss a pure latent MSE while preventing collapse."""

        def __init__(self, latent=1024, hidden=512, act_emb=64, tau=0.99):
            super().__init__()
            self.online = RSSMLite(latent, hidden, act_emb)
            self.decoder = Decoder(latent)
            self.target_encoder = Encoder(latent)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
            self.tau = tau
            self.ema_init()

        @torch.no_grad()
        def ema_init(self):
            for tp, op in zip(self.target_encoder.parameters(), self.online.encoder.parameters()):
                tp.data.copy_(op.data)

        @torch.no_grad()
        def ema_update(self):
            for tp, op in zip(self.target_encoder.parameters(), self.online.encoder.parameters()):
                tp.mul_(self.tau).add_(op.detach(), alpha=1 - self.tau)

    return BYOLModel()


# action dict (10 discrete fields) -> single id for the embedding
def pack_actions(a: np.ndarray) -> np.ndarray:
    # fields: move(5), jump(2), sneak(2), yaw(24), pitch(9), mine(2), place(2), use(2), hotbar(9), craft(8)
    a = a.astype(np.int64)
    ids = (
        a[..., 0] * 1
        + a[..., 1] * 5
        + a[..., 3] * 10          # yaw
        + a[..., 4] * 240         # pitch
        + a[..., 5] * 2160        # mine
        + a[..., 6] * 4320        # place
        + a[..., 7] * 8640        # use
        + a[..., 8] * 17280       # hotbar
        + a[..., 9] * 17280 * 9   # craft
    )
    return (ids % 63).astype(np.int64)  # 64-entry embedding table


def _load_split(data: str, seq_len: int, split: str, stride: int = 1, with_depth: bool = False):
    """Decode all episodes of a split into memory (the export is ~100 MB) and
    build the flat window index. `stride` subsamples ticks within a window —
    raw 20-TPS footage is near-static per tick, which makes any copy-last
    baseline unbeatable; stride 4 gives real per-frame motion."""
    ds = VoxelSequenceDataset(data, seq_len=seq_len, split=split)
    rgbs, acts, deps = [], [], []
    for si in range(len(ds.shards)):
        rgb, actions, depth, _ = ds._load(si)
        rgbs.append(rgb)
        acts.append(actions)
        if with_depth:
            if depth is None:
                raise ValueError("rgbd ablation needs depth frames in the export")
            deps.append(depth)
    wins = []
    span = (seq_len - 1) * stride + 1
    for ei, rgb in enumerate(rgbs):
        for s in range(0, len(rgb) - span + 1):
            wins.append((ei, s))
    return rgbs, acts, wins, deps


def run_baseline(data: str, steps: int, batch: int, seq_len: int, lr: float, limit_steps: int | None,
                 stride: int = 4, channels: str = "rgb", transfer_data: str | None = None,
                 device: str = "auto", dtype: str = "bf16"):
    """channels: "rgb" (3ch) or "rgbd" (4th channel = metric depth /96 cells).
    The ablation's whole point: identical data, windows, seed, steps — the
    ONLY difference is whether the encoder sees depth."""
    import torch

    assert channels in ("rgb", "rgbd")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch_device = torch.device(device)
    use_bf16 = dtype == "bf16" and torch_device.type == "cuda"
    want_depth = channels == "rgbd"
    torch.manual_seed(0)
    tr_rgb, tr_act, tr_wins, tr_dep = _load_split(data, seq_len, "train", stride, want_depth)
    te_rgb, te_act, te_wins, te_dep = _load_split(data, seq_len, "test", stride, want_depth)
    print(f"train windows: {len(tr_wins)}, test windows: {len(te_wins)} (stride={stride}, channels={channels})")
    if not tr_wins or not te_wins:
        raise RuntimeError("not enough data for the requested seq_len/split")

    def grab(rgbs, deps, acts, wins, idxs):
        rgb = np.stack([rgbs[ei][s : s + seq_len * stride : stride] for ei, s in idxs])  # (B,T,H,W,3)
        act = np.stack([acts[ei][s : s + seq_len * stride : stride] for ei, s in idxs])  # (B,T,10)
        if want_depth:
            dep = np.stack([deps[ei][s : s + seq_len * stride : stride] for ei, s in idxs])
            dep = dep.astype(np.float32)[..., None] / 96.0  # (B,T,H,W,1) metric cells -> ~[0,1]
        else:
            dep = None
        return rgb, dep, act

    in_ch = 4 if want_depth else 3
    model = _build_model(in_ch).to(torch_device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    rng = np.random.default_rng(0)

    def forward_lat(x_rgb, x_dep, act):
        # (B,T,H,W,C) -> (T,B,C,64,64) -> online/target latents (T,B,1024)
        if want_depth:
            x_in = np.concatenate([x_rgb.astype(np.float32) / 255.0, x_dep], axis=-1)
        else:
            x_in = x_rgb.astype(np.float32) / 255.0
        x = torch.from_numpy(x_in).permute(1, 0, 4, 2, 3).float().to(torch_device)
        T, B = x.shape[0], x.shape[1]
        x = torch.nn.functional.avg_pool2d(x.reshape(T * B, in_ch, 128, 128), 2)
        lat_on = model.online.encoder(x).reshape(T, B, -1)
        with torch.no_grad():
            lat_tg = model.target_encoder(x).reshape(T, B, -1)
        a_ids = torch.from_numpy(pack_actions(act)).permute(1, 0).to(torch_device)  # (T,B)
        return x, lat_on, lat_tg, a_ids

    n_steps = limit_steps or steps
    t0 = time.time()

    def eval_ratio():
        model.eval()
        model_err = 0.0
        copy_err = 0.0
        n = 0
        with torch.no_grad():
            for i in range(0, len(te_wins), batch):
                idxs = te_wins[i : i + batch]
                rgb, dep, act = grab(te_rgb, te_dep, te_act, te_wins, idxs)
                with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16,
                                    enabled=use_bf16):
                    _x, lat_on, lat_tg, a_ids = forward_lat(rgb, dep, act)
                    pred = model.online(lat_on[:-1], a_ids[1:])
                    model_err += torch.nn.functional.mse_loss(pred, lat_tg[1:], reduction="sum").item()
                    copy_err += torch.nn.functional.mse_loss(lat_tg[:-1], lat_tg[1:], reduction="sum").item()
                n += lat_tg[1:].numel()
        model.train()
        return (model_err / n) / (copy_err / n), model_err / n, copy_err / n

    model.train()
    for step in range(1, n_steps + 1):
        idxs = [tr_wins[i] for i in rng.integers(0, len(tr_wins), batch)]
        rgb, dep, act = grab(tr_rgb, tr_dep, tr_act, tr_wins, idxs)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16,
                            enabled=use_bf16):
            x, lat_on, lat_tg, a_ids = forward_lat(rgb, dep, act)
            pred = model.online(lat_on[:-1], a_ids[1:])  # predict next latents
            recon = model.decoder(lat_on.reshape(-1, lat_on.shape[-1]))
            loss = mse(pred, lat_tg[1:]) + 0.5 * torch.nn.functional.mse_loss(recon, x[:, :3])
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.ema_update()
        if step % 100 == 0 or step == 1:
            rate = step / (time.time() - t0)
            print(f"step {step}/{n_steps} loss {loss.item():.4f} ({rate:.1f} steps/s)", flush=True)
        if step % 1000 == 0:
            ratio, me, ce = eval_ratio()
            print(f"  [eval@{step}] ratio={ratio:.3f} model={me:.5f} copy={ce:.5f}", flush=True)

    ratio, me, ce = eval_ratio()
    print(f"test latent MSE: model {me:.5f}  copy {ce:.5f}  channels={channels}")
    if transfer_data:
        # resolution-transfer probe: the frozen model (trained on `data`) is
        # evaluated on a DIFFERENT dataset's windows (e.g. 0.5 m cells). The
        # ratio vs copy-last is self-normalizing, so the two numbers are
        # directly comparable across datasets.
        tr_rgb, tr_act, tr_wins2, tr_dep = _load_split(transfer_data, seq_len, "test", stride, want_depth)
        if not tr_wins2:
            # tiny transfer set: use its train split instead
            tr_rgb, tr_act, tr_wins2, tr_dep = _load_split(transfer_data, seq_len, "train", stride, want_depth)
        model.eval()
        me2 = ce2 = 0.0
        n2 = 0
        with torch.no_grad():
            for i in range(0, len(tr_wins2), batch):
                idxs = tr_wins2[i : i + batch]
                rgb, dep, act = grab(tr_rgb, tr_dep, tr_act, tr_wins2, idxs)
                with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16,
                                    enabled=use_bf16):
                    _x, lat_on, lat_tg, a_ids = forward_lat(rgb, dep, act)
                    pred = model.online(lat_on[:-1], a_ids[1:])
                    me2 += torch.nn.functional.mse_loss(pred, lat_tg[1:], reduction="sum").item()
                    ce2 += torch.nn.functional.mse_loss(lat_tg[:-1], lat_tg[1:], reduction="sum").item()
                n2 += lat_tg[1:].numel()
        ratio2 = (me2 / n2) / (ce2 / n2)
        print(f"TRANSFER ({transfer_data}): model {me2/n2:.5f} copy {ce2/n2:.5f} ratio {ratio2:.3f}")
    return ratio
