#!/usr/bin/env python3
"""
Spatial_Physics.py — spatial operator + medium for the full-wave sandbox
(Wave Behavior / Wave Generation).

This is the method-of-lines SPATIAL half of the scalar wave equation

    d^2 u / dt^2  =  c(x)^2 * laplacian(u)

It owns everything that does NOT depend on time:
  * laplacian()      n-dimensional 2nd-order stencil (works in 2-D and 3-D)
  * speed_field()    material-class grid -> wave-speed field c(x)
  * rigid_mask()     perfectly-reflecting classes (metal / service core)

Time integration, sources, and the run loop live in Time_Domain_Physics.py.

Why this is NOT the eikonal / path-loss engine
-----------------------------------------------
The production engine (SIM V1 3D/engine_3d.py) solves the geometrical-optics
LIMIT of this equation -- eikonal T + transport PL, two STATIC grids. Here we
step the actual field u(x, t) forward in time, so reflection, refraction,
diffraction and interference EMERGE from c(x) and the boundaries instead of
being added as separate terms. That fidelity is paid for with the CFL time loop
-- see the resolution budget at the top of Time_Domain_Physics.py.

"Model of choosing"
-------------------
c(x) is built from the SAME material_grid.npy that SIM V1 3D/voxelize.py
produces from any OBJ, so pointing this at a new building is just
"re-run voxelize.py, load the grid here." (Classes below match
manifest_3d.json's material ids.)
"""
from __future__ import annotations

import numpy as np

C0 = 299_792_458.0  # m/s, vacuum wave speed

# Relative permittivity per material CLASS -- real part only, order-of-magnitude
# (ITU-R P.2040-3 ballpark). This is a SANDBOX medium, not the calibrated
# physics_v2 table; index = material class id in material_grid.npy.
EPS_R = {
    0: 1.0,   # air
    1: 2.5,   # drywall_partition
    2: 6.0,   # concrete_masonry
    3: 6.0,   # core_service_area  (also flagged rigid below)
    4: 2.0,   # furniture_clutter
    5: 5.5,   # glass
}
RIGID_CLASSES = (3,)  # perfectly reflecting (metal / service core) -> u = 0


def laplacian(u, inv_h2):
    """n-D 2nd-order Laplacian via np.roll. Works unchanged for 2-D or 3-D u.

    np.roll wraps the boundary rows, so the outermost cell couples to the far
    edge. That 1-cell contamination MUST be killed by the absorbing sponge in
    the time loop -- never run this with bare (undamped) edges.
    """
    lap = np.zeros_like(u)
    for ax in range(u.ndim):
        lap += np.roll(u, 1, axis=ax) + np.roll(u, -1, axis=ax)
    lap -= 2.0 * u.ndim * u
    return lap * inv_h2


def speed_field(M, eps_r=EPS_R):
    """material-class grid M (int) -> wave-speed field c(x) = C0 / sqrt(eps_r).

    Unknown classes default to air (C0). Refraction (bending) and partial
    reflection (impedance contrast) both fall out of the spatial gradient of the
    returned c -- no extra code.
    """
    c = np.full(M.shape, C0, dtype=np.float64)
    for cls, er in eps_r.items():
        c[M == cls] = C0 / np.sqrt(er)
    return c


def rigid_mask(M, rigid_classes=RIGID_CLASSES):
    """bool grid of cells held at u = 0 every step (perfect reflector, Dirichlet).

    Gives hard-wall reflection and full shadowing; the wavefront diffracts
    around the mask's edges on its own.
    """
    m = np.zeros(M.shape, dtype=bool)
    for cls in rigid_classes:
        m |= (M == cls)
    return m
