#!/usr/bin/env python3
"""
indoor_georef.py — indoor model ↔ metres ↔ voxel ↔ floor-plan px registration.

The indoor 7th-floor geometry has no lat/lon; it is anchored to real metres via a
floor-plan-pixel scale (`manifest_3d.json:scale_reference` → `m_per_unit`), and a
px↔voxel affine (`registration_3d.json`) places walk-test measurements into the
voxel grid. This module centralizes those transforms (pure coordinate math —
NumPy/Python config, no JAX).

Chain:  model-units --×m_per_unit--> metres ;  metres --/cell_size--> voxel ;
        floorplan px --affine_px_to_vox--> voxel  (registration_v2, anchored scale)

  python indoor_georef.py            # print the registration + a round-trip check
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import bootstrap as B


def load_manifest() -> dict:
    return json.loads(B.require(B.MANIFEST).read_text())


def load_registration() -> dict:
    return json.loads(B.require(B.REGISTRATION).read_text())


# ---- scale accessors ---------------------------------------------------------
def m_per_unit(man=None) -> float:
    man = man or load_manifest()
    return float(man["m_per_unit"])


def cell_size_m(man=None) -> float:
    man = man or load_manifest()
    return float(man["cell_size_m"])


def model_units_to_m(units, man=None) -> np.ndarray:
    return np.asarray(units, float) * m_per_unit(man)


def m_to_model_units(metres, man=None) -> np.ndarray:
    return np.asarray(metres, float) / m_per_unit(man)


# ---- metres <-> voxel (per-axis isotropic cell) ------------------------------
def m_to_vox(xyz_m, man=None) -> np.ndarray:
    return np.asarray(xyz_m, float) / cell_size_m(man)


def vox_to_m(vox, man=None) -> np.ndarray:
    return np.asarray(vox, float) * cell_size_m(man)


# ---- floor-plan px <-> voxel (registration_v2 anchored affine) ---------------
def _affine(reg=None):
    """Return the chosen 2x3 px->vox affine [[a,0,c],[0,e,f]]: vx=a·px_x+c, vz=e·px_y+f."""
    reg = reg or load_registration()
    A = np.asarray(reg["affine_px_to_vox"], float)   # already the disambiguated choice
    return A


def px_to_vox(px_x, px_y, reg=None):
    """Floor-plan pixel (records frame) -> (vx, vz) voxel indices (axis0=X, axis1=Z)."""
    A = _affine(reg)
    vx = A[0, 0] * np.asarray(px_x, float) + A[0, 2]
    vz = A[1, 1] * np.asarray(px_y, float) + A[1, 2]
    return vx, vz


def vox_to_px(vx, vz, reg=None):
    """Inverse of px_to_vox."""
    A = _affine(reg)
    px_x = (np.asarray(vx, float) - A[0, 2]) / A[0, 0]
    px_y = (np.asarray(vz, float) - A[1, 2]) / A[1, 1]
    return px_x, px_y


# ---- walk-test Rx placement --------------------------------------------------
def rx_voxel(px_x, px_y, reg=None):
    """Place a walk-test Rx (measured at floor-plan px) into the voxel grid at the
    registered receiver height (`rx_y_vox`). Returns (vx, vy, vz) rounded ints."""
    reg = reg or load_registration()
    vx, vz = px_to_vox(px_x, px_y, reg)
    vy = int(reg.get("rx_y_vox", 5))
    return int(round(float(vx))), vy, int(round(float(vz)))


def lonlat_to_vox_indoor(lon, lat):
    """Best-effort lon/lat -> voxel via floorplan_meta (affine_px_to_lonlat inverse)
    then px->vox. Requires the floorplan_meta.json referenced by registration_3d.
    Returns (vx, vz) or raises if the meta file is unavailable."""
    reg = load_registration()
    fm_path = Path(reg.get("source", {}).get("floorplan_meta", ""))
    if not fm_path.exists():
        raise FileNotFoundError(
            f"floorplan_meta not found ({fm_path}); indoor lon/lat needs the "
            f"affine_px_to_lonlat from STEP_1. Use px_to_vox / rx_voxel instead.")
    fm = json.loads(fm_path.read_text())
    A = np.asarray(fm["affine_px_to_lonlat"], float)      # px -> lonlat
    # invert the 2x3 affine (linear 2x2 + translation) to map lonlat -> px
    M, t = A[:, :2], A[:, 2]
    px = np.linalg.solve(M, np.asarray([lon, lat], float) - t)
    return px_to_vox(px[0], px[1], reg)


if __name__ == "__main__":
    man, reg = load_manifest(), load_registration()
    print(f"grid_shape={man['grid_shape']}  cell_size_m={cell_size_m(man)}  "
          f"m_per_unit={m_per_unit(man):.6f}  ceiling={man.get('ceiling_height_m')}")
    print(f"affine px->vox:\n{_affine(reg)}")
    # round-trip a floor-plan pixel through vox and back
    for px in [(575, 257), (100, 100)]:
        vx, vz = px_to_vox(*px, reg)
        bx, bz = vox_to_px(vx, vz, reg)
        print(f"px{px} -> vox({vx:.1f},{vz:.1f}) -> px({bx:.1f},{bz:.1f})")
    print("rx_voxel(575,257) =", rx_voxel(575, 257, reg))
