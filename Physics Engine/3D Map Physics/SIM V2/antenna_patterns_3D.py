#!/usr/bin/env python3
"""
antenna_patterns_3D.py — 3-D (azimuth × elevation) antenna patterns + a physical
3-D backplane placement, for shaping the emitted full-wave lobes.

Extends `2D/SIM V3/antenna_patterns.py` (22 families, azimuth-only) with the 3GPP
**vertical** pattern and the az×el combination, and generalizes the 2-D panel
backplane (`fw_bs_catalog._place_source`) to a 3-D conductive reflector so lobes
emerge from the full-wave solve itself. Two directivity paths, matching the plan:

  * place_backplane_3d(...)  — physical: a rigid (class-3) reflector behind the
    source ⇒ directivity is IN the FDTD field (ground-truth teacher).
  * apply_directivity(U, ...) — analytic: multiply an omni solve by the az×el gain
    (cheap augmentation; one omni solve reused for every antenna kind).

Boresight comes from the 3-D antenna geometry (`Antenna_Type_3D.boresight_world`)
or from (azimuth, tilt) via `boresight_vec_from_angles`. All pure NumPy config.
"""
from __future__ import annotations

import numpy as np

import bootstrap  # noqa: F401  (side effect: sys.path → SIM V3)
import antenna_patterns as AP2D          # azimuth base (params, pattern_atten_db, is_directional)
from antenna_patterns import params, is_directional, PATTERNS  # noqa: F401

# per-family vertical (elevation) HPBW °, and the vertical side-lobe floor
VERT_HPBW = {
    "dish": 10.0, "phased_array": 15.0, "horn": 40.0,
    "yagi": 50.0, "quad": 55.0, "lpda": 60.0,
    "panel": 10.0, "patch": 65.0, "slot": 20.0, "pifa": 90.0, "inverted_f": 90.0,
    "loop": 80.0, "fractal": 90.0, "helical": 60.0, "metamaterial": 90.0,
    "omni": 90.0, "monopole": 45.0, "whip": 45.0, "handheld": 90.0,
    "dipole": 78.0, "discone": 60.0, "biconical": 60.0,
}
DEFAULT_VERT = 15.0
SLA_V = 30.0                              # vertical side-lobe attenuation floor (dB)


def vert_hpbw(kind) -> float:
    return VERT_HPBW.get((kind or "panel").lower(), DEFAULT_VERT)


def pattern3d_atten_db(kind, az_deg, el_deg, boresight_az=0.0, tilt_deg=0.0,
                       hpbw_az=None, hpbw_el=None, am_db=None, sla_v=SLA_V):
    """Combined 3GPP az×el off-boresight attenuation (dB ≥ 0).
    A = min(A_h(φ) + A_v(θ), max(Am, SLA_v)); near-omni-in-azimuth families still
    get a vertical donut (dipole/omni) via A_v."""
    Ah = AP2D.pattern_atten_db(kind, az_deg, boresight_az, hpbw_deg=hpbw_az, am_db=am_db)
    hv = hpbw_el if hpbw_el is not None else vert_hpbw(kind)
    dv = np.asarray(el_deg, float) - tilt_deg
    Av = np.minimum(12.0 * (dv / hv) ** 2, sla_v)
    am = am_db if am_db is not None else params(kind)[1]
    return np.minimum(Ah + Av, max(am, sla_v))


def pattern3d_gain(kind, az_deg, el_deg, **kw):
    """0..1 linear amplitude (the surrogate's directivity input channel, in 3-D)."""
    return 10.0 ** (-pattern3d_atten_db(kind, az_deg, el_deg, **kw) / 20.0)


def boresight_vec_from_angles(azimuth_deg, tilt_deg=0.0):
    """Unit boresight vector in (X, Y_up, Z): azimuth in the XZ plane, tilt above
    horizon. Matches the scene's Y-up convention."""
    phi, th = np.radians(azimuth_deg), np.radians(tilt_deg)
    return np.array([np.cos(th) * np.sin(phi), np.sin(th), np.cos(th) * np.cos(phi)])


def _cell_angles(shape, tx):
    """az (deg, XZ bearing) and el (deg, above horizontal) of every cell rel. to tx."""
    ii, jj, kk = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    dx, dy, dz = ii - tx[0], jj - tx[1], kk - tx[2]
    az = np.degrees(np.arctan2(dx, dz))
    el = np.degrees(np.arctan2(dy, np.sqrt(dx * dx + dz * dz) + 1e-9))
    return az, el


def apply_directivity(U, tx, kind, boresight_az=0.0, tilt_deg=0.0, **kw):
    """Analytic augmentation: multiply an omni complex field U by the az×el gain."""
    az, el = _cell_angles(U.shape, tx)
    g = pattern3d_gain(kind, az, el, boresight_az=boresight_az, tilt_deg=tilt_deg, **kw)
    return U * g


def place_backplane_3d(classes, tx, boresight_vec, kind="panel", offset_cells=2,
                       radius_cells=6, barrier_class=3):
    """Put a rigid (class `barrier_class`) reflector patch behind `tx`, perpendicular
    to `boresight_vec`, so a directional family's lobe emerges from the FDTD solve.
    Near-omni families get nothing (returns unchanged). Returns (new_classes, n_set)."""
    if not is_directional(kind):
        return classes, 0
    out = classes.copy()
    b = np.asarray(boresight_vec, float)
    b = b / (np.linalg.norm(b) + 1e-12)
    tx = np.asarray(tx, float)
    center = tx - offset_cells * b                       # behind the source
    r = int(radius_cells)
    lo = np.floor(center - r - 1).astype(int)
    hi = np.ceil(center + r + 1).astype(int)
    lo = np.maximum(lo, 0); hi = np.minimum(hi, np.array(classes.shape) - 1)
    if np.any(hi <= lo):
        return out, 0
    ii, jj, kk = np.mgrid[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    rel = np.stack([ii - center[0], jj - center[1], kk - center[2]], axis=-1)
    along = rel @ b                                       # signed distance along boresight
    lateral = np.sqrt(np.maximum((rel * rel).sum(-1) - along * along, 0.0))
    mask = (np.abs(along) <= 0.7) & (lateral <= r)        # thin plane, disk of radius r
    sub = out[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    sub[mask] = barrier_class
    out[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = sub
    return out, int(mask.sum())


if __name__ == "__main__":
    # panel is directional (front≫back); omni is flat in azimuth
    print("panel front (0,0):", round(float(pattern3d_gain("panel", 0, 0)), 3),
          " back (180,0):", round(float(pattern3d_gain("panel", 180, 0)), 4),
          " up (0,45):", round(float(pattern3d_gain("panel", 0, 45)), 4))
    print("dipole az flat:", round(float(pattern3d_gain("dipole", 90, 0)), 3),
          " overhead null:", round(float(pattern3d_gain("dipole", 0, 85)), 4))
    print("boresight az=90,tilt=-10 ->", np.round(boresight_vec_from_angles(90, -10), 3))
