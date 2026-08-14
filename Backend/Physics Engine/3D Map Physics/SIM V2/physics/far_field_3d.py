#!/usr/bin/env python3
"""
far_field_3d.py — the 3-D analytic FAR-field engine (already 3-D in the repo).

Thin wrapper over `SIM V1 3D/engine_3d.py::SceneV3`, the geometrical-optics /
Motley-Keenan + eikonal path-loss engine that already returns `(nf, NX, NY, NZ)`.
The hybrid solver (`hybrid_field_3d`) uses this for the region beyond the full-wave
near-box, and `run_outdoor3d` uses it as the primary (full-wave over a city is
infeasible). Pure reuse — no dimensional changes needed.

  from far_field_3d import load, far_pathloss
  scene, man = load()
  PL = far_pathloss(scene, tx=(120, 8, 90), f_mhz=617.0)   # (NX,NY,NZ) dB
"""
from __future__ import annotations

import numpy as np

import bootstrap  # noqa: F401  (side effect: sys.path → SIM V1 3D)
import engine_3d as E3


def load(sim_dir=None, barrier_classes=(3,)):
    """(SceneV3, manifest) for the indoor scene (defaults to SIM V1 3D).
    `engine_3d.load_scene` is a module-level function; pathloss needs no skfmm
    (only arrival_time does)."""
    if sim_dir is None:
        return E3.load_scene(barrier_classes=barrier_classes)
    return E3.load_scene(sim_dir=sim_dir, barrier_classes=barrier_classes)


def far_pathloss(scene, tx, f_mhz):
    """Analytic path-loss volume PL(x,y,z) in dB for a Tx voxel + frequency."""
    return np.asarray(scene.pathloss_map(tx, f_mhz))


def far_pathloss_band(scene, tx, band_label="LTE_B71_617"):
    from bands_v3 import get
    return far_pathloss(scene, tx, get(band_label).f_mhz)


def arrival_time(scene, tx, f_mhz):
    """First-arrival time volume (eikonal), if the analytic engine exposes it."""
    return np.asarray(scene.arrival_time(tx, f_mhz))


if __name__ == "__main__":
    scene, man = load()
    NX, NY, NZ = man["grid_shape"]
    tx = (NX // 2, NY // 2, NZ // 2)
    PL = far_pathloss(scene, tx, 617.0)
    print(f"far-field PL {PL.shape}  dB range [{np.nanmin(PL):.1f}, {np.nanmax(PL):.1f}]  tx={tx}")
