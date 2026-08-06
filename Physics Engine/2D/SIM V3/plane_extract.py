#!/usr/bin/env python3
"""
plane_extract.py — turn the shipped 3-D voxel scene into a fine 2-D floor plane
the full-wave solver can step.

The indoor scene (SIM V1 3D/voxelize.py) is material_grid.npy, shape
(X, Y_up, Z) = (262, 17, 132) at cell_size_m = 0.30 m, axes per manifest_3d.json.
A HORIZONTAL floor plane is therefore grid[:, y, :] (X by Z) at a fixed height y.

Two jobs:
  1. slice the plane at a chosen height, optionally crop to a sub-region;
  2. NEAREST-resample the coarse 0.30 m classes up to the FDTD cell h = lambda/N
     (mm-scale) so the wave is resolved -- classes are ids, so order-0 only.

Registration (metres <-> cells) is preserved so the plane lines up with the 3-D
scene the Unity view renders.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

import _bootstrap as B
from bands_v3 import Band


@dataclass
class Plane:
    classes: np.ndarray      # (nx, nz) int8 fine material-class grid
    h_m: float               # fine cell size (m)
    extent_m: tuple          # (Lx, Lz) physical size (m)
    tx_idx: tuple            # (ix, iz) source cell in the fine grid
    height_m: float          # plane height above floor (m)
    n_eff: float             # effective cells-per-wavelength actually used
    meta: dict


def load_scene():
    """material_grid (X,Y,Z) int8, valid_tx mask, and the manifest dict."""
    grid = np.load(B.require(B.MATERIAL_GRID))
    valid_tx = np.load(B.require(B.VALID_TX_MASK))
    manifest = json.loads(B.require(B.MANIFEST).read_text())
    return grid, valid_tx, manifest


def extract_plane(band: Band, height_m=None, height_frac=0.5,
                  n_per_wavelength=10.0, crop_m=None, tx_m=None,
                  max_cells=6_000_000) -> Plane:
    """Slice + resample the floor plane for `band`.

    height_m       plane height (m); defaults to height_frac * ceiling.
    crop_m         (x0, x1, z0, z1) metres to restrict the domain (None = full).
    tx_m           (x, z) metres for the source; default = valid-Tx centroid.
    max_cells      guard: if the fine grid would exceed this, coarsen h (drop
                   effective N) and warn instead of allocating tens of millions.
    """
    grid, valid_tx, manifest = load_scene()
    cs = float(manifest["cell_size_m"])                # 0.30 m
    ny = grid.shape[1]
    ceiling = float(manifest.get("ceiling_height_m", ny * cs))

    if height_m is None:
        height_m = height_frac * ceiling
    y = int(round(height_m / cs))
    y = max(0, min(ny - 1, y))

    plane = grid[:, y, :]                               # (nx, nz) coarse, X by Z
    vtx = valid_tx[:, y, :]

    # crop (metres -> coarse cells) before resampling
    x0 = z0 = 0
    if crop_m is not None:
        cx0, cx1, cz0, cz1 = crop_m
        x0, x1 = int(cx0 / cs), int(np.ceil(cx1 / cs))
        z0, z1 = int(cz0 / cs), int(np.ceil(cz1 / cs))
        plane = plane[x0:x1, z0:z1]
        vtx = vtx[x0:x1, z0:z1]

    # cell-size budget + guard
    h = band.cell_size_m(n_per_wavelength)
    n_eff = n_per_wavelength
    zoom = cs / h
    while (plane.shape[0] * zoom) * (plane.shape[1] * zoom) > max_cells:
        n_eff *= 0.85                                  # coarsen: fewer cells/lambda
        h = band.cell_size_m(n_eff)
        zoom = cs / h
    if n_eff < n_per_wavelength:
        print(f"  [guard] grid would exceed {max_cells:,} cells; dropped "
              f"cells/wavelength {n_per_wavelength:.0f} -> {n_eff:.1f} "
              f"(h {h*100:.2f} cm). Crop for full resolution.")

    # NEAREST resample (order=0) so material-class ids are preserved exactly
    classes = ndimage.zoom(plane, zoom, order=0).astype(np.int8)
    nx, nz = classes.shape
    extent = (nx * h, nz * h)

    # transmitter: explicit metres, else centroid of valid-Tx cells on the plane
    if tx_m is not None:
        ix = int(round((tx_m[0] - x0 * cs) / h))
        iz = int(round((tx_m[1] - z0 * cs) / h))
    else:
        src = vtx if vtx.any() else (plane == 0)       # fall back to interior air
        cx, cz = np.argwhere(src).mean(axis=0)
        ix, iz = int(round(cx * zoom)), int(round(cz * zoom))
    ix = max(1, min(nx - 2, ix))
    iz = max(1, min(nz - 2, iz))
    # nudge the source out of any solid cell it landed in (must radiate from air)
    if classes[ix, iz] != 0:
        air = np.argwhere(classes == 0)
        if len(air):
            j = np.argmin(np.abs(air[:, 0] - ix) + np.abs(air[:, 1] - iz))
            ix, iz = int(air[j, 0]), int(air[j, 1])

    meta = dict(cell_size_coarse_m=cs, y_index=y, ceiling_m=ceiling,
                zoom=zoom, crop_m=crop_m, grid_coarse=list(plane.shape),
                band=band.label, f_mhz=band.f_mhz)
    return Plane(classes=classes, h_m=h, extent_m=extent, tx_idx=(ix, iz),
                 height_m=y * cs, n_eff=n_eff, meta=meta)


if __name__ == "__main__":
    from bands_v3 import get
    p = extract_plane(get("LTE_B71_617"), height_frac=0.5)
    print(f"plane {p.classes.shape}  h={p.h_m*100:.2f} cm  "
          f"extent={p.extent_m[0]:.1f}x{p.extent_m[1]:.1f} m  "
          f"tx={p.tx_idx}  height={p.height_m:.2f} m  n_eff={p.n_eff:.1f}")
