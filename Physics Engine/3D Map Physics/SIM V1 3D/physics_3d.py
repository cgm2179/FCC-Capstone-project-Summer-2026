#!/usr/bin/env python3
"""
physics_3d.py — 3-D geometry adapter over the validated SIM/physics_v2.py.

The electromagnetics is NOT reimplemented. Blocks A–D and the CrossingLUT in
SIM/physics_v2.py are geometry-agnostic (they take incidence angle, thickness,
wedge angles), so they are imported and reused verbatim — that is the
non-corner-cutting move. This module adds only the *3-D geometry* the 2-D
engine_v2.py supplied in 2-D:

  - wall_normals_3d          unit boundary normals from a blurred-occupancy grad
  - measure_nominal_widths_3d median raster run-width per class (thickness cal)
  - slowness/speed field     Re(√ε_r)/c per voxel + barrier mask, for the eikonal
  - r_slab                   Airy slab reflection (for the Stage-3 module)

Material table: reuses physics_v2.MATERIALS7 — its ids 0..5 already line up with
the voxel grid (air, drywall→plasterboard, concrete, core→concrete,
furniture→wood/per-metre, glass→low-E), including the calibrated e_ref/gamma_e.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

# Locate the validated 2-D physics_v2.py. The repo reorg moved SIM to
# Physics Engine/2D/SIM, so search a few candidate locations (real path first,
# then legacy sibling layouts) and put the first that exists on sys.path.
_here = Path(__file__).resolve().parent
for _c in (_here.parent.parent / "2D" / "SIM",   # Physics Engine/2D/SIM (real)
           _here.parent.parent / "SIM",          # legacy: SIM beside the 3D group
           _here.parent / "SIM"):                # legacy: SIM beside SIM V1 3D
    if (_c / "physics_v2.py").exists():
        if str(_c) not in sys.path:
            sys.path.insert(0, str(_c))
        break
import physics_v2 as P   # noqa: E402  (validated Block A–D kernels + CrossingLUT)

C_LIGHT = 299792458.0

# --- reused verbatim from physics_v2 (no reimplementation) ------------------
MATERIALS = P.MATERIALS7
permittivity = P.permittivity
fresnel_coeffs = P.fresnel_coeffs
electrical_thickness = P.electrical_thickness
slab_transmission_coherent = P.slab_transmission_coherent      # Refraction_3D consumer
slab_transmission_incoherent = P.slab_transmission_incoherent
CrossingLUT = P.CrossingLUT


def wall_normals_3d(grid, sigma=1.0):
    """Unit boundary normals (nx, ny, nz) from the gradient of a Gaussian-blurred
    occupancy mask. Deep inside solids / open air the gradient vanishes → (0,0,0)
    and the caller falls back to normal incidence. 3-D port of
    engine_v2.wall_normals (axes X, Y, Z)."""
    occ = (grid > 0).astype(np.float32)
    sm = ndimage.gaussian_filter(occ, sigma, mode="nearest")
    gx, gy, gz = np.gradient(sm)
    mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    inv = np.where(mag > 1e-4, 1.0 / np.maximum(mag, 1e-9), 0.0).astype(np.float32)
    return (-gx * inv).astype(np.float32), (-gy * inv).astype(np.float32), (-gz * inv).astype(np.float32)


def measure_nominal_widths_3d(grid, n_classes, cell_size_m):
    """Median rasterised run-width (m) per class, scanned along all three axes.
    Connects raster run length to real construction thickness (t_ref/w_ref).
    3-D port of engine_v2.measure_nominal_widths."""
    out = np.zeros(n_classes, np.float32)
    for c in range(1, n_classes):
        runs = []
        hit0 = grid == c
        if not hit0.any():
            out[c] = cell_size_m
            continue
        for ax in range(3):
            hit = np.moveaxis(hit0, ax, -1)
            pad = np.zeros(hit.shape[:-1] + (1,), bool)
            h = np.concatenate([pad, hit, pad], axis=-1).astype(np.int8)
            d = np.diff(h, axis=-1)
            starts = np.argwhere(d == 1)
            ends = np.argwhere(d == -1)
            if len(starts):
                runs.append(ends[:, -1] - starts[:, -1])
        out[c] = float(np.median(np.concatenate(runs))) * cell_size_m if runs else cell_size_m
    return out


def refractive_index_by_class(materials, f_mhz, air_n=1.0):
    """n = Re(√ε_r) per class at f (CT7/RT1). Air/vacuum → air_n."""
    n = np.full(len(materials), air_n, np.float32)
    for m in materials:
        if m.get("p2040") not in (None, "vacuum"):
            n[m["id"]] = float(np.real(np.sqrt(permittivity(m["p2040"], f_mhz)[0])))
    return n


def speed_field(grid, f_mhz, materials=MATERIALS, barrier_classes=(3,), air_n=1.0):
    """Wave-speed field c/n per voxel (m/s) + barrier mask for the eikonal.
    Penetrable classes get finite c/n (front transmits, bends, lags = refraction);
    opaque classes are masked so the front routes around them (CTR1/CT6)."""
    n_by_class = refractive_index_by_class(materials, f_mhz, air_n)
    speed = (C_LIGHT / np.maximum(n_by_class[grid], 1e-3)).astype(np.float64)
    mask = np.isin(grid, list(barrier_classes))
    return speed, mask


def r_slab(eps_r, theta_i, thickness_m, f_mhz):
    """Airy slab reflection coefficients (R_te, R_tm); shares the denominator
    with physics_v2.slab_transmission_coherent. Used by the Stage-3 reflection
    module (image sources)."""
    r_te, r_tm = fresnel_coeffs(eps_r, theta_i)
    q = electrical_thickness(eps_r, theta_i, thickness_m, f_mhz)
    e2 = np.exp(-2j * q)

    def _R(r):
        return r * (1 - e2) / (1 - r * r * e2)

    return _R(r_te), _R(r_tm)
