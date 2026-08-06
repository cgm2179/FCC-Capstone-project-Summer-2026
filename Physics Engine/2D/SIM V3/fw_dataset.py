#!/usr/bin/env python3
"""
fw_dataset.py — training data for the full-wave U-Net surrogate (PART 3).

Ground truth is the complex steady-state field U(x) from FullWaveScene (FDTD).
A per-box FDTD only carries a field when the Tx is INSIDE the box, so instead we
run ONE larger FDTD per Tx and CROP many boxes from it — far boxes then show
realistic weak illumination from the Tx direction, and it amortizes the solve.

Per box we store:
  input  x[9,H,W]  = 6 material one-hot + tx_blob + freq_feat + log_distance
                     (the dataset_3d.py featurization contract, verbatim)
  target y[2,H,W]  = (Re, Im) of the PHASE-REDUCED field  Ũ = U·e^{+j k d}
                     (free-space ramp removed → smooth, tileable), normalized.

Scenes:
  indoor   whole 7th-floor plane per Tx (voxelize.py grid)          — cheap fields
  outdoor  a REGION crop of the NoMa city plane around each tower   — small regions
           (buildings = barrier class 3); citywide is infeasible.

  python fw_dataset.py --scene indoor  --bands LTE_B71_617 --n-tx 6 --box 128
  python fw_dataset.py --scene outdoor --bands NR_n41_2506 --region-m 40 --n-tx 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import _bootstrap as B
import dataset_3d as D3               # INPUT_CHANNELS, Norm, TX_SIGMA_CELLS, LOGDIST_DIVISOR
import fullwave2d as FW
import plane_extract as PE
from bands_v3 import get
from fullwave2d import FullWaveScene
from plane_extract import Plane

HERE = Path(__file__).resolve().parent
C0 = 299_792_458.0


def _featurize(classes, tx_local, d_m, freq_feat, n_classes=6):
    """9-channel input tensor for one box (dataset_3d contract)."""
    H, W = classes.shape
    x = np.zeros((len(D3.INPUT_CHANNELS), H, W), np.float32)
    for c in range(n_classes):                              # material one-hot
        x[c] = (classes == c)
    ii, jj = np.mgrid[0:H, 0:W]
    sig = D3.TX_SIGMA_CELLS
    x[n_classes] = np.exp(-(((ii - tx_local[0]) ** 2 + (jj - tx_local[1]) ** 2)
                            / (2.0 * sig * sig)))           # tx_blob (0 if Tx far)
    x[n_classes + 1] = freq_feat                            # freq_feat (broadcast)
    x[n_classes + 2] = np.log10(np.maximum(d_m, 1.0)) / D3.LOGDIST_DIVISOR
    return x


def _run_field(classes, h, tx_ij, f_mhz, crossings):
    """FDTD on a fine class grid → complex phasor U (shared by both scenes)."""
    sim = FullWaveScene(classes, h, f_mhz, tx_ij, source="cw")
    steps = int(round(crossings * max(classes.shape) * h / C0 / sim.dt))
    res = FW.simulate(sim, steps, warmup_frac=0.6, extract_phasor=True,
                      show_tqdm=True, desc=f"field {f_mhz:.0f}MHz")
    if not res["finite"]:
        raise FloatingPointError("field blew up")
    return res["phasor"]


def _indoor_field(band, tx_m, height_frac, npw, crossings, max_cells):
    p = PE.extract_plane(band, height_frac=height_frac, n_per_wavelength=npw,
                         tx_m=tx_m, max_cells=max_cells)
    return p, _run_field(p.classes, p.h_m, p.tx_idx, band.f_mhz, crossings)


def _outdoor_field(city_grid, cs, band, tx_m, height_m, npw, region_m,
                   crossings, max_cells):
    """Slice + crop a city REGION around the tower, resample to λ/N, run FDTD."""
    ny = city_grid.shape[1]
    y = max(0, min(ny - 1, int(round(height_m / cs))))
    plane = city_grid[:, y, :]                              # (X, Z), 1 m
    ix, iz = int(tx_m[0] / cs), int(tx_m[1] / cs)
    h = band.cell_size_m(npw)
    zoom = cs / h
    half = int(region_m / cs / 2)
    while (2 * half * zoom) ** 2 > max_cells and half > 8:   # keep it feasible
        half = int(half * 0.85)
    x0, x1 = max(0, ix - half), min(plane.shape[0], ix + half)
    z0, z1 = max(0, iz - half), min(plane.shape[1], iz + half)
    fine = ndimage.zoom(plane[x0:x1, z0:z1], zoom, order=0).astype(np.int8)
    txf = [int((ix - x0) * zoom), int((iz - z0) * zoom)]
    txf[0] = max(1, min(fine.shape[0] - 2, txf[0]))
    txf[1] = max(1, min(fine.shape[1] - 2, txf[1]))
    if fine[txf[0], txf[1]] != 0:                            # snap out of a building
        air = np.argwhere(fine == 0)
        j = np.argmin(np.abs(air[:, 0] - txf[0]) + np.abs(air[:, 1] - txf[1]))
        txf = [int(air[j, 0]), int(air[j, 1])]
    p = Plane(fine, h, (fine.shape[0] * h, fine.shape[1] * h), tuple(txf),
              y * cs, npw, dict(scene="outdoor"))
    return p, _run_field(fine, h, tuple(txf), band.f_mhz, crossings)


def generate(band_labels, scene="indoor", n_tx=3, boxes_per_field=80, box=128,
             height_frac=0.5, n_per_wavelength=8.0, region_m=40.0, crossings=2.5,
             max_cells=1_500_000, out_dir=None, seed=0):
    rng = np.random.default_rng(seed)
    out = Path(out_dir) if out_dir else HERE / f"fw_data_{scene}"
    out.mkdir(parents=True, exist_ok=True)
    # freq window is scene-independent → always the indoor manifest's norm
    norm = D3.load_norm(json.loads(B.MANIFEST.read_text()))

    if scene == "outdoor":
        cman = json.loads((B.CITY_DIR / "manifest_3d.json").read_text())
        cs = float(cman["cell_size_m"])
        city = np.load(B.require(B.CITY_DIR / "material_grid.npy"))
        tx_cells = np.argwhere(city[:, max(0, int(round(height_frac *
                    city.shape[1]))), :] == 0)               # street (air) cells
        height_m = height_frac * city.shape[1] * cs
    else:
        cman = json.loads(B.MANIFEST.read_text())
        cs = float(cman["cell_size_m"])
        vtx = np.load(B.VALID_TX_MASK)
        tx_cells = np.argwhere(vtx.any(axis=1))              # (x,z) AP locations

    shard_id = 0
    for label in band_labels:
        band = get(label)
        k = 2.0 * np.pi / band.wavelength_m
        ff = norm.freq_feature(band.f_mhz)
        for _ in range(n_tx):
            tc = tx_cells[rng.integers(len(tx_cells))]
            tx_m = (float(tc[0]) * cs, float(tc[1]) * cs)
            if scene == "outdoor":
                p, U = _outdoor_field(city, cs, band, tx_m, height_m,
                                      n_per_wavelength, region_m, crossings, max_cells)
            else:
                p, U = _indoor_field(band, tx_m, height_frac, n_per_wavelength,
                                     crossings, max_cells)
            H, W = U.shape
            h = p.h_m
            txf = np.array(p.tx_idx, float)
            ref = np.percentile(np.abs(U), 99.0) or 1.0
            xs, ys = [], []
            for _ in range(boxes_per_field):
                if H <= box or W <= box:
                    x0 = y0 = 0; b = min(box, H, W)
                else:
                    x0 = int(rng.integers(0, H - box)); y0 = int(rng.integers(0, W - box)); b = box
                sl = (slice(x0, x0 + b), slice(y0, y0 + b))
                ii, jj = np.mgrid[x0:x0 + b, y0:y0 + b]
                d_m = np.hypot(ii - txf[0], jj - txf[1]) * h
                Utilde = U[sl] * np.exp(1j * k * d_m)        # phase-reduce (smooth)
                y = (np.stack([Utilde.real, Utilde.imag]) / ref).astype(np.float32)
                x = _featurize(p.classes[sl], (txf[0] - x0, txf[1] - y0), d_m, ff)
                xs.append(x); ys.append(y)
            X = np.stack(xs); Y = np.stack(ys)
            np.savez_compressed(out / f"shard_{shard_id:03d}.npz", x=X, y=Y,
                                band=band.label, f_mhz=band.f_mhz, h_m=h, ref=ref)
            print(f"  shard {shard_id:03d} {scene} {band.label} tx=({int(tc[0])},{int(tc[1])}) "
                  f"boxes={len(xs)} x{X.shape[1:]}")
            shard_id += 1

    (out / "dataset_meta.json").write_text(json.dumps(dict(
        spec="fw-unet2d-v1", scene=scene, box=box, in_channels=list(D3.INPUT_CHANNELS),
        out_channels=["field_re", "field_im"], phase_reduced=True,
        bands=list(band_labels), n_shards=shard_id, height_frac=height_frac,
        n_per_wavelength=n_per_wavelength, region_m=region_m,
        logdist_divisor=D3.LOGDIST_DIVISOR, tx_sigma_cells=D3.TX_SIGMA_CELLS,
        norm=dict(freq_log_lo_mhz=norm.freq_log_lo_mhz,
                  freq_log_hi_mhz=norm.freq_log_hi_mhz)), indent=2))
    print(f"wrote {shard_id} shards -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=["indoor", "outdoor"], default="indoor")
    ap.add_argument("--bands", nargs="+", default=["LTE_B71_617"])
    ap.add_argument("--n-tx", type=int, default=3)
    ap.add_argument("--boxes-per-field", type=int, default=80)
    ap.add_argument("--box", type=int, default=128)
    ap.add_argument("--n-per-wavelength", type=float, default=8.0)
    ap.add_argument("--region-m", type=float, default=40.0, help="outdoor region size")
    ap.add_argument("--max-cells", type=int, default=1_500_000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.bands, scene=args.scene, n_tx=args.n_tx,
             boxes_per_field=args.boxes_per_field, box=args.box,
             n_per_wavelength=args.n_per_wavelength, region_m=args.region_m,
             max_cells=args.max_cells, out_dir=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
