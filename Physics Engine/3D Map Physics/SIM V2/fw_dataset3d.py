#!/usr/bin/env python3
"""
fw_dataset3d.py — Phase B: 3-D FDTD boxes for the pure-JAX 3-D field U-Net.

SIM V2 port of `2D/SIM V3/fw_dataset3d.py`. The FDTD teacher runs on the JAX/XLA
hot path (`simulate_jax`). Fixes the exploration-flagged issues:
  * per-BAND output subdir (no more `sid` overwrite across bands),
  * enforce a UNIFORM cube box (skip a Tx whose fine volume is thinner than `box`,
    so `ShardDS` can concatenate all shards),
  * optional PHYSICAL directivity: a fraction of Tx get a class-3 backplane behind
    them (`antenna_patterns_3D.place_backplane_3d`) → directional teacher fields;
    directivity is captured for free by the 6 material one-hot channels (the
    backplane is class 3), so the input contract stays 9-ch.

Shard schema (spec "fw-unet3d-v1"), one shard per Tx, in out/<band>/ :
    x=(N,9,B,B,B) f32, y=(N,2,B,B,B) f32, h_m (), ref (), band ()

  python fw_dataset3d.py --band LTE_B71_617 --n-tx 4 --box 24 --npw 8 --directional-frac 0.4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import bootstrap as B
import dataset_3d as D3
import antenna_patterns_3D as AP3
from bands_v3 import get
from fullwave2d import FullWaveScene
from fw_fdtd_jax import simulate_jax

HERE = Path(__file__).resolve().parent
C0 = 299_792_458.0


def featurize3d(classes, tx, d_m, ff, ncls=6):
    """9-channel input: 6 material one-hot + Tx Gaussian blob + freq feat + log-dist.
    Identical contract to SIM V3 (dataset_3d.INPUT_CHANNELS)."""
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


def subvol_field(band, tx_m, npw, region_m, crossings, max_cells,
                 directional=False, boresight_az=None, kind="panel", dtype="float32"):
    """One 3-D JAX FDTD over an indoor sub-volume → (fine classes, U, tx, h).
    If `directional`, drop a class-3 backplane behind the source first."""
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
    if directional:
        az = float(np.random.default_rng().uniform(0, 360)) if boresight_az is None else boresight_az
        bore = AP3.boresight_vec_from_angles(az, 0.0)
        fine, _ = AP3.place_backplane_3d(fine, tx, bore, kind=kind, offset_cells=2,
                                         radius_cells=max(4, int(0.8 / h)))
    sim = FullWaveScene(fine, h, band.f_mhz, tx, source="cw")
    steps = int(round(crossings * max(fine.shape) * h / C0 / sim.dt))
    res = simulate_jax(sim, steps, warmup_frac=0.6, extract_phasor=True, dtype=dtype)
    if not res["finite"]:
        raise FloatingPointError("3-D field blew up")
    return fine, res["phasor"], tx, h


def generate3d(band_label="LTE_B71_617", n_tx=2, boxes_per=20, box=24, npw=8.0,
               region_m=7.0, crossings=2.0, max_cells=2_000_000, out_dir=None,
               seed=1, directional_frac=0.0):
    rng = np.random.default_rng(seed)
    out = Path(out_dir) if out_dir else HERE / "fw_data3d" / band_label   # per-band dir
    out.mkdir(parents=True, exist_ok=True)
    norm = D3.load_norm(json.loads(B.MANIFEST.read_text()))
    band = get(band_label); k = 2 * np.pi / band.wavelength_m
    ff = norm.freq_feature(band.f_mhz)
    cs = float(json.loads(B.MANIFEST.read_text())["cell_size_m"])
    txc = np.argwhere(np.load(B.VALID_TX_MASK).any(axis=1))
    sid = 0
    for _ in range(n_tx):
        tc = txc[rng.integers(len(txc))]
        directional = rng.random() < directional_frac
        fine, U, tx, h = subvol_field(band, (tc[0] * cs, tc[1] * cs), npw, region_m,
                                      crossings, max_cells, directional=directional)
        Dx, Dy, Dz = U.shape
        if min(Dx, Dy, Dz) < box:
            print(f"  [skip] tx {tuple(tc)}: fine {U.shape} thinner than box={box} "
                  f"(raise npw or region_m)")
            continue
        txf = np.array(tx, float)
        ref = np.percentile(np.abs(U), 99.0) or 1.0
        xs, ys = [], []
        for _ in range(boxes_per):
            x0 = int(rng.integers(0, Dx - box + 1)); y0 = int(rng.integers(0, Dy - box + 1))
            z0 = int(rng.integers(0, Dz - box + 1))
            sl = (slice(x0, x0 + box), slice(y0, y0 + box), slice(z0, z0 + box))
            ii, jj, kk = np.mgrid[x0:x0 + box, y0:y0 + box, z0:z0 + box]
            d_m = np.sqrt((ii - txf[0]) ** 2 + (jj - txf[1]) ** 2 + (kk - txf[2]) ** 2) * h
            Ut = U[sl] * np.exp(1j * k * d_m)                       # phase-reduce Ũ
            ys.append((np.stack([Ut.real, Ut.imag]) / ref).astype(np.float32))
            xs.append(featurize3d(fine[sl], (txf[0] - x0, txf[1] - y0, txf[2] - z0), d_m, ff))
        np.savez_compressed(out / f"shard_{sid:03d}.npz", x=np.stack(xs), y=np.stack(ys),
                            h_m=h, ref=ref, band=band.label,
                            directional=bool(directional))
        print(f"  shard {sid:03d} boxes={len(xs)} x{xs[0].shape} "
              f"{'[directional]' if directional else '[omni]'}")
        sid += 1
    (out / "dataset_meta.json").write_text(json.dumps(dict(
        spec="fw-unet3d-v1", box=box, bands=[band_label], n_shards=sid,
        in_channels=list(D3.INPUT_CHANNELS), out_channels=["field_re", "field_im"],
        phase_reduced=True, npw=npw, directional_frac=directional_frac), indent=2))
    print(f"wrote {sid} shards -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--n-tx", type=int, default=2)
    ap.add_argument("--boxes-per", type=int, default=20)
    ap.add_argument("--box", type=int, default=24)
    ap.add_argument("--npw", type=float, default=8.0)
    ap.add_argument("--region-m", type=float, default=7.0)
    ap.add_argument("--max-cells", type=int, default=2_000_000)
    ap.add_argument("--directional-frac", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    generate3d(a.band, n_tx=a.n_tx, boxes_per=a.boxes_per, box=a.box, npw=a.npw,
               region_m=a.region_m, max_cells=a.max_cells, out_dir=a.out,
               directional_frac=a.directional_frac)


if __name__ == "__main__":
    main()
