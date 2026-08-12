#!/usr/bin/env python3
"""
Path_Loss_3D — geometric spreading + the direct/transmitted coherent field.

The first of the six mechanism modules and the spine the others attach to. It contains
essentially NO new physics: `SceneV3` already computes both halves correctly, so this is
an adapter that packages them into the complex-field contract.

    magnitude  <-  SceneV3.pathloss_maps   (fspl_1m + 10 n log10 d  +  per-crossing
                                            Fresnel/Airy slab loss from the CrossingLUT)
    phase      <-  SceneV3.arrival_time    (3-D eikonal on slowness Re(sqrt(eps_r))/c,
                                            i.e. the true excess path length including
                                            in-wall slowdown and Fermat corner-wrap)

    E = 10**(-PL/20) * exp(-j 2 pi f tau)

Because both inputs are the engine's own outputs, `Combine_3D.to_legacy_volumes` of this
module alone must reproduce `SceneV3.pathloss_maps` exactly — that identity is the M1
acceptance gate (tests3d/test_combiner.py) and is what proves the complex spine is a
strict superset of the working engine rather than a parallel implementation that drifts.

usage:
    python Path_Loss_3D.py --test        # self-test (physics_v2.py --test convention)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from _bootstrap import SIM3D, contracts, engine_3d  # noqa: F401

MECHANISM = "path_loss"
DEFAULTS = {"pol": "te", "phase_from": "eikonal"}


def solve(scene, tx, *, bands=None, pol="te", phase_from="eikonal",
          backend="numpy") -> "contracts.FieldGrid":
    """Direct/transmitted coherent field for one Tx.

    scene       engine_3d.SceneV3
    tx          (x, y, z) voxel coordinates
    bands       frequencies in MHz (default: all scene bands)
    phase_from  'eikonal'   -> tau from SceneV3.arrival_time (refraction included)
                'geometric' -> tau = d/c (vacuum reference)
    """
    if backend != "numpy":
        raise NotImplementedError(f"backend {backend!r} — path loss is cheap; CPU only")

    freqs = np.asarray(scene.freqs if bands is None else bands, float)
    idx = [scene.band_index(f) for f in freqs]

    pl = np.ascontiguousarray(scene.pathloss_maps(tx)[idx]).astype(np.float32)

    geo = scene.geometric_time(tx).astype(np.float32)          # d/c lower bound
    if phase_from == "geometric":
        tau = np.broadcast_to(geo, pl.shape).copy()
    else:
        # The eikonal is frequency dependent through Re(sqrt(eps_r(f))), so solve per band.
        tau = np.empty(pl.shape, np.float32)
        for i, f in enumerate(freqs):
            tau[i] = scene.arrival_time(tx, float(f))
        # The eikonal masks barrier classes, so it returns inf inside/behind them -- but
        # the CrossingLUT path-loss model still reports a finite level there. Gating the
        # MAGNITUDE on eikonal reachability would zero the field where the engine says
        # there is signal, breaking the superset identity. So keep the amplitude and fall
        # back to the geometric d/c phase wherever no eikonal arrival exists.
        bad = ~np.isfinite(tau)
        if bad.any():
            tau[bad] = np.broadcast_to(geo, pl.shape)[bad]

    fg = contracts.FieldGrid.from_pl_tau(
        pl, tau, freqs, MECHANISM, tx, pol=pol,
        meta={"phase_from": phase_from, "engine": "SceneV3",
              "note": "magnitude from CrossingLUT, phase from the 3-D eikonal"})
    fg.combine_as = contracts.COHERENT
    return fg


def direct_pl_db(scene, tx, *, bands=None) -> np.ndarray:
    """Path loss only (NF, NX, NY, NZ) — the engine's own result, unwrapped."""
    freqs = np.asarray(scene.freqs if bands is None else bands, float)
    idx = [scene.band_index(f) for f in freqs]
    return np.ascontiguousarray(scene.pathloss_maps(tx)[idx]).astype(np.float32)


def free_space_field(scene, tx, *, bands=None, pol="te") -> "contracts.FieldGrid":
    """Vacuum reference: FSPL magnitude with geometric d/c phase.

    Mode 1 of the four propagation modes, and the invariant every other result is
    checked against (a passive channel can never beat free space).
    """
    freqs = np.asarray(scene.freqs if bands is None else bands, float)
    gx, gy, gz = np.meshgrid(np.arange(scene.NX), np.arange(scene.NY),
                             np.arange(scene.NZ), indexing="ij")
    d = np.sqrt((gx - tx[0]) ** 2 + (gy - tx[1]) ** 2 + (gz - tx[2]) ** 2) * scene.cell
    d = np.maximum(d, scene.d0).astype(np.float32)
    pl = np.stack([scene.fspl1[scene.band_index(f)] + 10.0 * scene.n_exp * np.log10(d)
                   for f in freqs]).astype(np.float32)
    tau = (d / contracts.C0).astype(np.float32)
    fg = contracts.FieldGrid.from_pl_tau(pl, tau, freqs, "free_space", tx, pol=pol,
                                         meta={"reference": True})
    fg.combine_as = contracts.COHERENT
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

    M, man = S.free_space_scene(n=(24, 24, 24))
    sc = engine_3d.SceneV3(M, man)
    tx = S.centre(M.shape)
    bands = man["freqs_mhz"][:2]
    fg = solve(sc, tx, bands=bands)

    check("shape", fg.E.shape == (2, 2) + M.shape, str(fg.E.shape))
    check("dtype complex64", fg.E.dtype == np.complex64, str(fg.E.dtype))
    check("TM empty for te-only", not np.any(fg.E[1]))

    pl_engine = direct_pl_db(sc, tx, bands=bands)
    err = float(np.nanmax(np.abs(fg.pathloss_db() - pl_engine)))
    check("|E|^2 == PL", err < 0.01, f"max err {err:.5f} dB")

    amp = np.abs(fg.E[0])
    check("amplitude finite", bool(np.isfinite(amp).all()))
    live = amp[0] > 0
    check("phase sweeps with distance",
          float(np.ptp(np.angle(fg.E[0][0][live]))) > 1.0)

    fg_g = solve(sc, tx, bands=bands[:1], phase_from="geometric")
    m = np.isfinite(fg.tau_first[0]) & np.isfinite(fg_g.tau_first[0]) & (fg_g.tau_first[0] > 0)
    rel = np.abs(fg.tau_first[0][m] - fg_g.tau_first[0][m]) / fg_g.tau_first[0][m]
    check("eikonal == d/c in vacuum", float(np.median(rel)) < 0.05,
          f"median rel {float(np.median(rel)):.4f}")

    # free-space reference must never be beaten by the real solve
    fs = free_space_field(sc, tx, bands=bands)
    worse = (fg.pathloss_db() >= fs.pathloss_db() - 0.05)
    check("never beats free space", float(worse.mean()) > 0.999,
          f"{100*(1-worse.mean()):.3f}% of cells below FSPL")

    print(f"\n{ok} passed, {len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--test", action="store_true", help="run the module self-test")
    a = ap.parse_args()
    raise SystemExit(_selftest() if a.test else (ap.print_help() or 0))
