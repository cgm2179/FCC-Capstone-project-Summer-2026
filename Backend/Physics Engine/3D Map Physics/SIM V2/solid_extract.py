#!/usr/bin/env python3
"""
solid_extract.py — turn the shipped coarse 3-D voxel scene into a FINE 3-D grid the
full-wave solver can step. The 3-D generalization of `2D/SIM V3/plane_extract.py`.

The indoor scene (SIM V1 3D/voxelize.py) is `material_grid.npy`, shape
(X, Y_up, Z) = (262, 17, 132) at 0.30 m. Unlike the 2-D slicer, we keep ALL THREE
axes: NEAREST-resample the whole coarse volume up to the FDTD cell h = λ/N (order-0
so material-class ids are preserved), with a `max_cells` guard that coarsens N
rather than allocating a billion voxels. Registration (metres↔cells) is preserved.

  python solid_extract.py --band LTE_B71_617 --npw 8
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

import bootstrap as B
from bands_v3 import Band, get


@dataclass
class Solid:
    classes: np.ndarray      # (nx, ny, nz) int8 fine material-class grid
    h_m: float               # fine cell size (m)
    extent_m: tuple          # (Lx, Ly, Lz) physical size (m)
    tx_idx: tuple            # (ix, iy, iz) source cell in the fine grid
    n_eff: float             # effective cells-per-wavelength actually used
    meta: dict


def load_scene():
    grid = np.load(B.require(B.MATERIAL_GRID))
    valid_tx = np.load(B.require(B.VALID_TX_MASK))
    manifest = json.loads(B.require(B.MANIFEST).read_text())
    return grid, valid_tx, manifest


def _snap_to_air(classes, idx):
    """Move a source index out of any solid cell to the nearest air (class 0) in 3-D."""
    ix, iy, iz = idx
    if classes[ix, iy, iz] == 0:
        return ix, iy, iz
    air = np.argwhere(classes == 0)
    if not len(air):
        return ix, iy, iz
    j = np.argmin(np.abs(air[:, 0] - ix) + np.abs(air[:, 1] - iy) + np.abs(air[:, 2] - iz))
    return int(air[j, 0]), int(air[j, 1]), int(air[j, 2])


def extract_solid(band: Band, n_per_wavelength=8.0, crop_m=None, tx_m=None,
                  max_cells=8_000_000) -> Solid:
    """Resample the whole coarse volume to the FDTD cell for `band`.

    crop_m  (x0,x1,y0,y1,z0,z1) metres to restrict the domain (None = full).
    tx_m    (x,y,z) metres for the source; default = valid-Tx centroid (mid-height).
    max_cells  guard: coarsen N (×0.85) until the fine volume fits, and warn.
    """
    grid, valid_tx, manifest = load_scene()
    cs = float(manifest["cell_size_m"])                 # 0.30 m
    o = np.zeros(3, int)                                 # crop origin (coarse cells)

    if crop_m is not None:
        cx0, cx1, cy0, cy1, cz0, cz1 = crop_m
        o = np.array([int(cx0 / cs), int(cy0 / cs), int(cz0 / cs)])
        sub = grid[o[0]:int(np.ceil(cx1 / cs)), o[1]:int(np.ceil(cy1 / cs)),
                   o[2]:int(np.ceil(cz1 / cs))]
        vtx = valid_tx[o[0]:int(np.ceil(cx1 / cs)), o[1]:int(np.ceil(cy1 / cs)),
                       o[2]:int(np.ceil(cz1 / cs))]
    else:
        sub, vtx = grid, valid_tx

    # cell-size budget + 3-D guard
    h = band.cell_size_m(n_per_wavelength)
    n_eff = float(n_per_wavelength)
    zoom = cs / h
    while (sub.shape[0] * zoom) * (sub.shape[1] * zoom) * (sub.shape[2] * zoom) > max_cells:
        n_eff *= 0.85
        h = band.cell_size_m(n_eff)
        zoom = cs / h
    if n_eff < n_per_wavelength:
        print(f"  [guard] volume would exceed {max_cells:,} cells; dropped "
              f"cells/wavelength {n_per_wavelength:.0f} -> {n_eff:.1f} "
              f"(h {h*100:.2f} cm). Crop for full resolution.")

    classes = ndimage.zoom(sub, zoom, order=0).astype(np.int8)  # order-0: preserve ids
    nx, ny, nz = classes.shape
    extent = (nx * h, ny * h, nz * h)

    # transmitter placement
    if tx_m is not None:
        idx = (int(round((tx_m[0] - o[0] * cs) / h)),
               int(round((tx_m[1] - o[1] * cs) / h)),
               int(round((tx_m[2] - o[2] * cs) / h)))
    else:
        src = vtx if vtx.any() else (sub == 0)          # fall back to interior air
        c = np.argwhere(src).mean(axis=0)
        idx = tuple(int(round(c[a] * zoom)) for a in range(3))
    idx = tuple(int(np.clip(idx[a], 1, classes.shape[a] - 2)) for a in range(3))
    idx = _snap_to_air(classes, idx)

    meta = dict(cell_size_coarse_m=cs, zoom=float(zoom), crop_m=crop_m,
                grid_coarse=list(sub.shape), band=band.label, f_mhz=band.f_mhz,
                crop_origin_cells=o.tolist())
    return Solid(classes=classes, h_m=h, extent_m=extent, tx_idx=idx,
                 n_eff=n_eff, meta=meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--npw", type=float, default=8.0)
    ap.add_argument("--max-cells", type=int, default=8_000_000)
    a = ap.parse_args()
    s = extract_solid(get(a.band), n_per_wavelength=a.npw, max_cells=a.max_cells)
    print(f"solid {s.classes.shape} = {s.classes.size/1e6:.2f} M cells  h={s.h_m*100:.2f} cm  "
          f"extent={s.extent_m[0]:.1f}x{s.extent_m[1]:.1f}x{s.extent_m[2]:.1f} m  "
          f"tx={s.tx_idx}  n_eff={s.n_eff:.1f}  occ={100*(s.classes>0).mean():.1f}%")
