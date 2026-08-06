#!/usr/bin/env python3
"""
fw_unet2d.py — 2-D complex-field U-Net surrogate (PART 3): model + training.

Architecture is the `phase_c_train_colab_v3` U-Net, adapted: cin = 9 (the
dataset_3d featurization contract), cout = 2 (field_re, field_im of the
phase-reduced field Ũ). Trains on the fw_dataset shards; runs on Apple MPS / CUDA
/ CPU. The same file is Colab-importable for GPU training at scale.

  python fw_unet2d.py --data fw_data_test --epochs 60 --base 32
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
IN_CH, OUT_CH = 9, 2


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))


class UNet(nn.Module):
    """4-level U-Net. cout=2 → (re, im). H,W must be multiples of 16."""

    def __init__(self, base=32, in_ch=IN_CH, out_ch=OUT_CH):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList()
        prev = in_ch
        for c in ch:
            self.enc.append(block(prev, c)); prev = c
        self.pool = nn.MaxPool2d(2)
        self.mid = block(prev, prev * 2)
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        prev = prev * 2
        for c in reversed(ch):
            self.up.append(nn.ConvTranspose2d(prev, c, 2, 2))
            self.dec.append(block(2 * c, c)); prev = c
        self.head = nn.Conv2d(prev, out_ch, 1)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x); skips.append(x); x = self.pool(x)
        x = self.mid(x)
        for u, d in zip(self.up, self.dec):
            x = d(torch.cat([u(x), skips.pop()], 1))
        return self.head(x)


class ShardDS(Dataset):
    def __init__(self, files):
        xs, ys = [], []
        for f in files:
            d = np.load(f); xs.append(d["x"]); ys.append(d["y"])
        self.x = np.concatenate(xs); self.y = np.concatenate(ys)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (torch.from_numpy(self.x[i].astype(np.float32, copy=False)),
                torch.from_numpy(self.y[i].astype(np.float32, copy=False)))


def complex_loss(pred, tgt, amp_w=0.5):
    """MSE on (re, im) plus an amplitude term so |U| is well fit, not just re/im."""
    mse = nn.functional.mse_loss(pred, tgt)
    amp_p = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + 1e-8)
    amp_t = torch.sqrt(tgt[:, 0] ** 2 + tgt[:, 1] ** 2 + 1e-8)
    amp = nn.functional.mse_loss(amp_p, amp_t)
    return mse + amp_w * amp, mse.item(), amp.item()


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(data_dir, epochs=60, base=32, bs=8, lr=1e-3, val_frac=0.2, out=None):
    files = sorted(glob.glob(str(Path(data_dir) / "shard_*.npz")))
    if not files:
        raise FileNotFoundError(f"no shards in {data_dir}")
    ds = ShardDS(files)
    cin, cout = ds.x.shape[1], ds.y.shape[1]               # infer (9 or 10 in; 2 out)
    n = len(ds); nv = max(1, int(val_frac * n))
    g = torch.Generator().manual_seed(0)
    tr, va = torch.utils.data.random_split(ds, [n - nv, nv], generator=g)
    dev = pick_device()
    m = UNet(base=base, in_ch=cin, out_ch=cout).to(dev)
    print(f"device={dev}  cin={cin} cout={cout}  "
          f"params={sum(p.numel() for p in m.parameters())/1e6:.2f}M  "
          f"train={len(tr)} val={len(va)}")
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tl = DataLoader(tr, bs, shuffle=True)
    vl = DataLoader(va, bs)
    best = float("inf")
    for ep in range(epochs):
        m.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss, _, _ = complex_loss(m(x), y)
            loss.backward(); opt.step()
        sched.step()
        m.eval(); vmse = 0.0; nv2 = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                vmse += nn.functional.mse_loss(m(x), y).item() * len(x); nv2 += len(x)
        vmse /= max(nv2, 1)
        best = min(best, vmse)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d}  val_mse={vmse:.4f}  (best {best:.4f})")
    out = Path(out) if out else HERE / "fw_unet2d.pt"
    torch.save({"state": m.state_dict(), "base": base, "in_ch": cin,
                "out_ch": cout}, out)
    print(f"saved {out}  best_val_mse={best:.4f}")
    return m, best


def load_model(path, dev=None):
    ck = torch.load(path, map_location="cpu")
    m = UNet(base=ck["base"], in_ch=ck["in_ch"], out_ch=ck["out_ch"])
    m.load_state_dict(ck["state"]); m.eval()
    return m.to(dev or pick_device())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="fw_data_test")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    train(args.data, epochs=args.epochs, base=args.base, bs=args.bs, lr=args.lr,
          out=args.out)


if __name__ == "__main__":
    main()
