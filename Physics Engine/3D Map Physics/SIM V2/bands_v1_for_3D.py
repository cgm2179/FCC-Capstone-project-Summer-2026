#!/usr/bin/env python3
"""
bands_v1_for_3D.py — real RF bands for the SIM V2 3-D full-wave engine.

Re-exports the SIM V3 `Band`/catalog verbatim (single source of truth — the bands
don't change with dimensionality) and adds a **3-D** cell-budget estimate. The
budget matters far more in 3-D: cells scale as N³ (not N²), so the "whole floor at
λ/10" verdict flips to "crop" for almost every band above the smallest.
"""
from __future__ import annotations

import bootstrap  # noqa: F401  (side effect: sys.path → SIM V3)

# single source of truth: reuse the SIM V3 band catalog
from bands_v3 import Band, BANDS_V3, get, C0, estimate as estimate_2d  # noqa: F401


def estimate_3d(band: Band, extent_m, n_per_wavelength: float = 10.0,
                max_cells: float = 40_000_000):
    """Grid the band implies over a 3-D domain `extent_m` = (Lx, Ly, Lz) metres.

    Returns cell size, per-axis grid, total cells, and a whole-volume verdict.
    The 3-D memory footprint (float32, ~6 field/medium arrays + complex phasor) is
    ~ total_cells * 40 bytes — reported so callers can size a solve up front.
    """
    lx, ly, lz = (float(extent_m[0]), float(extent_m[1]), float(extent_m[2]))
    h = band.cell_size_m(n_per_wavelength)
    nx, ny, nz = int(round(lx / h)), int(round(ly / h)), int(round(lz / h))
    total = nx * ny * nz
    return dict(
        band=band.label, f_mhz=band.f_mhz, wavelength_m=band.wavelength_m,
        h_m=h, nx=nx, ny=ny, nz=nz, total_cells=total,
        est_bytes=int(total * 40),
        whole_volume_ok=total <= max_cells,
        verdict=("whole volume" if total <= max_cells else "crop advised"),
    )


if __name__ == "__main__":
    # 3-D budget over the shipped 7th-floor extent (~78.6 x 5.1 x 39.6 m)
    for b in BANDS_V3[:8]:
        for npw in (4.0, 8.0, 10.0):
            e = estimate_3d(b, (78.6, 5.1, 39.6), npw)
            print(f"{b.label:14s} λ/{npw:<4.0f} h={e['h_m']*100:5.2f}cm "
                  f"grid={e['nx']:4d}x{e['ny']:3d}x{e['nz']:4d} "
                  f"{e['total_cells']/1e6:8.1f} M  {e['est_bytes']/1e9:6.2f} GB  {e['verdict']}")
