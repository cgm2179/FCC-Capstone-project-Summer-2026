#!/usr/bin/env python3
"""
fw_dataset3d.py — 3-D FDTD boxes for the 3-D complex U-Net surrogate (Part B).

Same recipe as fw_dataset.py but in 3-D: run FullWaveScene on an indoor
sub-volume (it's dimension-agnostic), extract the complex phasor U(x,y,z), crop
cubic boxes, phase-reduce Ũ=U·e^{+jkd}, featurize (9 ch, 3-D). Feeds fw_unet3d.

  python fw_dataset3d.py --band LTE_B71_617 --n-tx 2 --box 24
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # engine root, for _bootstrap
import _bootstrap as B
import dataset_3d as D3
import fullwave2d as FW
from bands_v3 import get
from fullwave2d import FullWaveScene

HERE = Path(__file__).resolve().parent.parent
C0 = 299_792_458.0


def featurize3d(classes, tx, d_m, ff, ncls=6):
    D = classes.shape
    x = np.zeros((9,) + D, np.float32)
    for c in range(ncls):
        x[c] = (classes == c)
    ii, jj, kk = np.mgrid[0:D[0], 0:D[1], 0:D[2]]
    sig = D3.TX_SIGMA_CELLS
    x[ncls] = np.exp(-(((ii - tx[0]) ** 2 + (jj - tx[1]) ** 2 + (kk - tx[2]) ** 2)
                       / (2.0 * sig * sig)))
    x[ncls + 1] = ff
    x[ncls + 2] = np.log10(np.maximum(d_m, 1.0)) / D3.LOGDIST_DIVISOR
    return x


def subvol_field(band, tx_m, npw, region_m, crossings, max_cells):
    """One 3-D FDTD over an indoor sub-volume → (fine classes, U, tx, h)."""
    grid = np.load(B.MATERIAL_GRID)
    cs = float(json.loads(B.MANIFEST.read_text())["cell_size_m"])
    h = band.cell_size_m(npw); zoom = cs / h
    ix, iz = int(tx_m[0] / cs), int(tx_m[1] / cs)
    r = int(region_m / cs / 2)
    x0, x1 = max(0, ix - r), min(grid.shape[0], ix + r)
    z0, z1 = max(0, iz - r), min(grid.shape[2], iz + r)
    sub = grid[x0:x1, :, z0:z1]
    while (sub.shape[0] * zoom) * (sub.shape[1] * zoom) * (sub.shape[2] * zoom) > max_cells:
        zoom *= 0.9
    fine = ndimage.zoom(sub, zoom, order=0).astype(np.int8)
    tx = tuple(int(np.clip(v, 1, fine.shape[i] - 2)) for i, v in enumerate(
        ((ix - x0) * zoom, (grid.shape[1] // 2) * zoom, (iz - z0) * zoom)))
    sim = FullWaveScene(fine, h, band.f_mhz, tx, source="cw")
    steps = int(round(crossings * max(fine.shape) * h / C0 / sim.dt))
    res = FW.simulate(sim, steps, warmup_frac=0.6, extract_phasor=True,
                      show_tqdm=True, desc=f"3D {band.label}")
    if not res["finite"]:
        raise FloatingPointError("3-D field blew up")
    return fine, res["phasor"], tx, h


def generate3d(band_label="LTE_B71_617", n_tx=2, boxes_per=20, box=24, npw=5.0,
               region_m=7.0, crossings=2.0, max_cells=1_500_000, out_dir=None, seed=1):
    rng = np.random.default_rng(seed)
    out = Path(out_dir) if out_dir else HERE / "fw_data3d"
    out.mkdir(parents=True, exist_ok=True)
    norm = D3.load_norm(json.loads(B.MANIFEST.read_text()))
    band = get(band_label); k = 2 * np.pi / band.wavelength_m
    ff = norm.freq_feature(band.f_mhz)
    cs = float(json.loads(B.MANIFEST.read_text())["cell_size_m"])
    txc = np.argwhere(np.load(B.VALID_TX_MASK).any(axis=1))
    sid = 0
    for _ in range(n_tx):
        tc = txc[rng.integers(len(txc))]
        fine, U, tx, h = subvol_field(band, (tc[0] * cs, tc[1] * cs), npw, region_m,
                                      crossings, max_cells)
        Dx, Dy, Dz = U.shape; txf = np.array(tx, float)
        ref = np.percentile(np.abs(U), 99.0) or 1.0
        xs, ys = [], []
        for _ in range(boxes_per):
            x0 = int(rng.integers(0, max(1, Dx - box))); y0 = int(rng.integers(0, max(1, Dy - box)))
            z0 = int(rng.integers(0, max(1, Dz - box))); b = min(box, Dx, Dy, Dz)
            sl = (slice(x0, x0 + b), slice(y0, y0 + b), slice(z0, z0 + b))
            ii, jj, kk = np.mgrid[x0:x0 + b, y0:y0 + b, z0:z0 + b]
            d_m = np.sqrt((ii - txf[0]) ** 2 + (jj - txf[1]) ** 2 + (kk - txf[2]) ** 2) * h
            Ut = U[sl] * np.exp(1j * k * d_m)
            ys.append((np.stack([Ut.real, Ut.imag]) / ref).astype(np.float32))
            xs.append(featurize3d(fine[sl], (txf[0] - x0, txf[1] - y0, txf[2] - z0), d_m, ff))
        np.savez_compressed(out / f"shard_{sid:03d}.npz", x=np.stack(xs), y=np.stack(ys),
                            h_m=h, ref=ref, band=band.label)
        print(f"  shard {sid:03d} boxes={len(xs)} x{xs[0].shape}")
        sid += 1
    (out / "dataset_meta.json").write_text(json.dumps(dict(
        spec="fw-unet3d-v1", box=box, bands=[band_label], n_shards=sid), indent=2))
    print(f"wrote {sid} shards -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--n-tx", type=int, default=2)
    ap.add_argument("--box", type=int, default=24)
    ap.add_argument("--out", default="fw_data3d")
    a = ap.parse_args()
    generate3d(a.band, n_tx=a.n_tx, box=a.box, out_dir=a.out)


if __name__ == "__main__":
    main()
