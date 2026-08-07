#!/usr/bin/env python3
"""
nearfield_3d.py — near/far field zones + close-in correction for the 3-D hybrid.

`2D/SIM V3/nearfield.py` is already dimension-agnostic (it operates on distance
arrays), so this re-exports it verbatim and adds a 3-D near-box sizer used by
`hybrid_field_3d` to choose the full-wave sub-volume radius.
"""
from __future__ import annotations

import numpy as np

import bootstrap  # noqa: F401  (side effect: sys.path → SIM V3)
from nearfield import (  # noqa: F401  re-export the dim-agnostic zone math
    C0, wavelength_m, zones, fspl_db, apply_close_in, schantz_relative_power,
)


def near_box_radius_cells(f_mhz, h_m, radius_m=6.0, min_wavelengths=6.0):
    """Cells for the full-wave near-box radius: at least `min_wavelengths` λ (dense
    multipath), else `radius_m`, whichever is larger — clamped to whole cells."""
    lam = wavelength_m(f_mhz)
    r_m = max(float(radius_m), float(min_wavelengths) * lam)
    return int(np.ceil(r_m / float(h_m)))


if __name__ == "__main__":
    for f in (617.0, 2442.0, 3700.0):
        z = zones(f)
        r = near_box_radius_cells(f, z["wavelength_m"] / 8.0)
        print(f"{f:6.0f} MHz  λ={z['wavelength_m']*100:5.1f} cm  "
              f"near-box r≈{r} cells (@λ/8)")
