#!/usr/bin/env python3
"""
Absorption_3D — absorbed-power density (where the energy goes).

Like refraction, absorption is not a separate power contribution: the CrossingLUT already
folds Im(q) bulk absorption into the direct path loss. So this module is a DIAGNOSTIC
decomposition — it isolates HOW MUCH of the transmitted power each material deposits (as
heat) rather than reflects, and WHERE, for the "absorption" visualization. It is marked
`combine_as = DIAGNOSTIC` and is never summed into received power (that energy has already
left the channel; adding it back would violate conservation).

    absorbed_frac(material, f) = 1 - exp(2 * Im(q))          q = electrical_thickness(...)
                                                             (Im(q) <= 0; lossless -> 0)
    absorbed_density(voxel)    = P_direct(voxel) * absorbed_frac(material(voxel), f)

i.e. the direct power arriving AT a material voxel times the fraction that voxel's medium
absorbs on one crossing. Air voxels absorb nothing. The result is a per-voxel linear-power
map carried in `p_incoh` (E is zero), so the browser's mechanism view renders it directly.

Reuses `physics_3d.electrical_thickness` / `permittivity` (the Im(q) absorption model from
physics_v2 Block B) and the material table's `per_metre` / `fill_fraction` for bulk clutter.

usage:
    python Absorption_3D.py --test        # self-test (physics_v2.py --test convention)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from _bootstrap import SIM3D, contracts, engine_3d, physics_3d  # noqa: F401

MECHANISM = "absorption"
DEFAULTS = {"theta_i": 0.0}


def absorbed_fraction(materials, f_mhz, theta_i=0.0) -> np.ndarray:
    """Per-class single-crossing bulk-absorbed power fraction at frequency f (0..1).

    From the imaginary electrical thickness: the bulk transmission factor is exp(2 Im q),
    so the absorbed fraction is 1 - exp(2 Im q). Air / lossless media -> 0. Bulk-clutter
    classes (per_metre, e.g. furniture) use their solid-equivalent thickness
    t_ref * fill_fraction.
    """
    frac = np.zeros(len(materials), np.float32)
    for m in materials:
        p2040 = m.get("p2040")
        if p2040 in (None, "vacuum"):
            continue                                   # air: no absorption
        thick = float(m.get("t_ref_m", 0.0))
        if m.get("per_metre"):
            thick *= float(m.get("fill_fraction", 1.0))
        if thick <= 0.0:
            continue
        eps = physics_3d.permittivity(p2040, f_mhz)[0]
        q = physics_3d.electrical_thickness(eps, theta_i, thick, f_mhz)
        a2 = float(np.exp(2.0 * np.imag(q)))           # bulk power transmission (<= 1)
        frac[m["id"]] = max(0.0, 1.0 - a2)
    return frac


def solve(scene, tx, *, bands=None, theta_i=0.0, backend="numpy") -> "contracts.FieldGrid":
    """Absorbed-power density for one Tx (DIAGNOSTIC; never summed into received power)."""
    if backend != "numpy":
        raise NotImplementedError(f"backend {backend!r} — absorption is cheap; CPU only")

    freqs = np.asarray(scene.freqs if bands is None else bands, float)
    idx = [scene.band_index(f) for f in freqs]
    NX, NY, NZ = scene.NX, scene.NY, scene.NZ

    pl = scene.pathloss_maps(tx)[idx].astype(np.float32)       # direct power arriving AT voxel
    absorbed = np.zeros((len(freqs), NX, NY, NZ), np.float32)
    for i, f in enumerate(freqs):
        frac_by_class = absorbed_fraction(scene.materials, float(f), theta_i)   # (n_class,)
        frac_vox = frac_by_class[scene.M]                                       # (NX,NY,NZ)
        incident = np.power(10.0, -pl[i] / 10.0)                                # linear power
        absorbed[i] = (incident * frac_vox).astype(np.float32)

    fg = contracts.FieldGrid(
        E=np.zeros((contracts.NPOL, len(freqs), NX, NY, NZ), np.complex64),
        tau_first=np.full((len(freqs), NX, NY, NZ), np.inf, np.float32),
        mechanism=MECHANISM, freqs_mhz=freqs, tx_vox=tuple(int(v) for v in tx),
        p_incoh=absorbed,
        meta={"kind": "diagnostic",
              "quantity": "absorbed power density = P_direct * (1 - exp(2 Im q))",
              "note": "energy deposited in materials; NOT summed into received power"})
    fg.combine_as = contracts.DIAGNOSTIC
    return fg


# --------------------------------------------------------------------------- self-test
def _selftest() -> int:
    sys.path.insert(0, str(SIM3D / "tests3d"))
    import synth3d as S

    ok, fails = 0, []

    def check(name, cond, msg=""):
        nonlocal ok
        if cond:
            ok += 1
            print(f"  PASS  {name}")
        else:
            fails.append(name)
            print(f"  FAIL  {name}  {msg}")

    # ---- per-class fraction: air absorbs nothing, a lossy dielectric absorbs some ----
    M, man = S.slab_scene_3d(n=(40, 16, 16), mat=2, x0=16, x1=20)   # concrete slab
    sc = engine_3d.SceneV3(M, man)
    f = float(man["freqs_mhz"][0])
    frac = absorbed_fraction(sc.materials, f)
    check("air absorbs nothing", float(frac[0]) == 0.0)
    check("concrete absorbs a positive fraction", 0.0 < float(frac[2]) < 1.0,
          f"frac[concrete] = {float(frac[2]):.4f}")
    check("absorbed fraction is bounded 0..1", bool((frac >= 0).all() and (frac <= 1).all()))

    # ---- the field ----
    tx = (3, M.shape[1] // 2, M.shape[2] // 2)
    fg = solve(sc, tx, bands=[f])
    check("is declared DIAGNOSTIC", fg.combine_as == contracts.DIAGNOSTIC)
    check("E is zero (not a propagating field)", not np.any(np.abs(fg.E)))
    check("p_incoh populated", fg.p_incoh is not None)
    ap = fg.p_incoh[0]
    check("absorbed density finite and non-negative",
          bool(np.isfinite(ap).all() and (ap >= 0).all()))
    slabmask = (M == 2)
    airmask = (M == 0)
    check("absorption happens in the slab, not in air",
          float(ap[slabmask].sum()) > 0 and float(ap[airmask].sum()) == 0.0,
          f"slab {float(ap[slabmask].sum()):.3e}, air {float(ap[airmask].sum()):.3e}")
    # absorbed can never exceed the incident power at the same voxel
    incident = np.power(10.0, -sc.pathloss_maps(tx)[sc.band_index(f)] / 10.0)
    check("absorbed <= incident everywhere",
          bool((ap <= incident + 1e-20).all()))

    # ---- combiner must ignore it (diagnostic, not received power) ----
    import Combine_3D as CMB
    import Path_Loss_3D as PLM
    direct = PLM.solve(sc, tx, bands=[f])
    cf_d = CMB.combine([direct], floor_at_fspl=False)
    cf_da = CMB.combine([direct, fg], floor_at_fspl=False)
    check("adding absorption does not change received PL",
          float(np.nanmax(np.abs(cf_d.PL_db - cf_da.PL_db))) < 1e-6,
          f"max dPL {float(np.nanmax(np.abs(cf_d.PL_db - cf_da.PL_db))):.2e} dB")
    check("absorption is recorded per-mechanism (for the viz)",
          "absorption" in cf_da.per_mech_lin)

    print(f"\n{ok} passed, {len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--test", action="store_true", help="run the module self-test")
    a = ap.parse_args()
    raise SystemExit(_selftest() if a.test else (ap.print_help() or 0))
