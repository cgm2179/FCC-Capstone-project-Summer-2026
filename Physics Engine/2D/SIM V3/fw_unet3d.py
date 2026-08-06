#!/usr/bin/env python3
"""
fw_unet3d.py — 3-D complex-field U-Net surrogate (Part B).

Reuses `models_3d.UNet3D` (the repo's 3-D surrogate architecture) but drops its
final sigmoid — the PL surrogate output [0,1] can't represent signed re/im — and
reuses the fw_unet2d training loop (ShardDS + complex_loss). Predicts the 3-D
complex field Ũ; reconstruct U and animate in the three.js viewer.

  python fw_unet3d.py --data fw_data3d --epochs 25 --base 16
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import _bootstrap as B
sys.path.insert(0, str(B.SIM3D))
from models_3d import UNet3D                                # noqa: E402
from fw_unet2d import ShardDS, complex_loss, pick_device    # noqa: E402

HERE = Path(__file__).resolve().parent


class UNet3DField(UNet3D):
    """UNet3D without the sigmoid (re/im are signed)."""

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        d3 = self.d3(self._up(b, e3))
        d2 = self.d2(self._up(d3, e2))
        d1 = self.d1(self._up(d2, e1))
        return self.out(d1)


def load_model3d(path, dev=None):
    ck = torch.load(path, map_location="cpu")
    m = UNet3DField(cin=ck["in_ch"], cout=ck["out_ch"], base=ck["base"])
    m.load_state_dict(ck["state"]); m.eval()
    return m.to(dev or pick_device())


def train(data, epochs=25, base=16, bs=4, lr=1e-3, val_frac=0.2, out=None):
    files = sorted(glob.glob(str(Path(data) / "shard_*.npz")))
    ds = ShardDS(files)
    cin, cout = ds.x.shape[1], ds.y.shape[1]
    n = len(ds); nv = max(1, int(val_frac * n))
    g = torch.Generator().manual_seed(0)
    tr, va = torch.utils.data.random_split(ds, [n - nv, nv], generator=g)
    dev = pick_device()
    m = UNet3DField(cin=cin, cout=cout, base=base).to(dev)
    print(f"device={dev}  cin={cin} cout={cout}  "
          f"params={sum(p.numel() for p in m.parameters())/1e6:.2f}M  "
          f"train={len(tr)} val={len(va)}")
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    tl = DataLoader(tr, bs, shuffle=True); vl = DataLoader(va, bs)
    best = float("inf")
    for ep in range(epochs):
        m.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss, _, _ = complex_loss(m(x), y); loss.backward(); opt.step()
        m.eval(); vmse = 0.0; nv2 = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                vmse += nn.functional.mse_loss(m(x), y).item() * len(x); nv2 += len(x)
        vmse /= max(nv2, 1); best = min(best, vmse)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d}  val_mse={vmse:.4f}  (best {best:.4f})")
    out = Path(out) if out else HERE / "fw_unet3d.pt"
    torch.save({"state": m.state_dict(), "base": base, "in_ch": cin, "out_ch": cout}, out)
    print(f"saved {out}  best_val_mse={best:.4f}")
    return m, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="fw_data3d")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    train(a.data, epochs=a.epochs, base=a.base, out=a.out)


if __name__ == "__main__":
    main()
