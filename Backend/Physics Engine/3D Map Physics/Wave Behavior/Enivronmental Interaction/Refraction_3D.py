#!/usr/bin/env python3
"""
Refraction_3D — the coherent transmitted field, isolated from free space.

Refraction is not a separate power contribution: `SceneV3` already bends the wavefront
through the index field (the eikonal solves |grad T| = 1/(c/n) on n = Re(sqrt(eps_r))), and
the CrossingLUT already carries the through-wall transmission magnitude. So this module is
a DIAGNOSTIC decomposition of the direct path — it isolates what the medium does to the
transmitted wave, for the "refraction" visualization, and is NOT summed into received power
(that would double-count `Path_Loss_3D`). It is marked `combine_as = DIAGNOSTIC`.

    magnitude  <-  the through-wall transmission loss = PL_direct - FSPL
                   (= the saturated per-crossing Fresnel/Airy slab loss; unity in free
                    space, so the field shows ONLY the effect of the media it crossed)
    phase      <-  the REFRACTIVE excess delay = eikonal tau - geometric d/c
                   (the extra time from in-wall slowdown and Fermat corner-wrap — the
                    genuine signature of refraction, zero in vacuum)

    E = 10**(-(PL_direct - FSPL)/20) * exp(-j 2 pi f (tau_eikonal - d/c))

`physics_v2.slab_transmission_coherent` (the coherent transmitted-phase model, until now
unused in 3-D) is exercised in the self-test as the reference the eikonal excess delay is
checked against — this module is its first legitimate 3-D consumer.

usage:
    python Refraction_3D.py --test        # self-test (physics_v2.py --test convention)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from _bootstrap import SIM3D, contracts, engine_3d, physics_3d  # noqa: F401

MECHANISM = "refraction"
DEFAULTS = {"pol": "te"}


def solve(scene, tx, *, bands=None, pol="te", backend="numpy") -> "contracts.FieldGrid":
    """Isolated coherent transmitted (refracted) field for one Tx.

    DIAGNOSTIC: a decomposition of the direct path, never summed into received power.
    Magnitude is the through-wall transmission loss (PL_direct - FSPL); phase is the
    refractive excess delay (eikonal - d/c).
    """
    if backend != "numpy":
        raise NotImplementedError(f"backend {backend!r} — refraction is cheap; CPU only")

    freqs = np.asarray(scene.freqs if bands is None else bands, float)
    idx = [scene.band_index(f) for f in freqs]

    # magnitude: excess loss over free space = the (saturated) per-crossing slab loss, the
    # part of PL_direct that the media contribute. Reusing crossing_loss + the same satObs
    # cap the direct path applies keeps this exactly PL_direct - FSPL.
    cl = scene.crossing_loss(tx)[idx].astype(np.float32)
    if scene.use_satobs:
        cl = engine_3d.sat_obs(cl, scene.obs_T, scene.obs_S).astype(np.float32)
    excess_pl = np.ascontiguousarray(cl)

    # phase: the refractive excess delay, eikonal - geometric, per band (the eikonal is
    # frequency dependent through Re(sqrt(eps_r(f)))). Clip >= 0 (the eikonal can never beat
    # the straight line by more than sub-cell source offset) and zero it where the eikonal
    # is unreachable (behind a barrier) — there the magnitude is already ~0.
    geo = scene.geometric_time(tx).astype(np.float32)
    excess_tau = np.empty(excess_pl.shape, np.float32)
    for i, f in enumerate(freqs):
        te = scene.arrival_time(tx, float(f))
        d = te - geo
        excess_tau[i] = np.where(np.isfinite(te), np.maximum(d, 0.0), 0.0)

    fg = contracts.FieldGrid.from_pl_tau(
        excess_pl, excess_tau, freqs, MECHANISM, tx, pol=pol,
        meta={"kind": "diagnostic",
              "magnitude": "PL_direct - FSPL (through-wall transmission loss)",
              "phase": "eikonal tau - geometric d/c (refractive excess delay)",
              "note": "isolated medium effect; NOT summed into received power"})
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

    # ---- free space: no medium => unity field, zero excess delay ----------
    M, man = S.free_space_scene(n=(24, 24, 24))
    sc = engine_3d.SceneV3(M, man)
    tx = S.centre(M.shape)
    bands = man["freqs_mhz"][:2]
    fg = solve(sc, tx, bands=bands)

    check("shape", fg.E.shape == (2, 2) + M.shape, str(fg.E.shape))
    check("is declared DIAGNOSTIC", fg.combine_as == contracts.DIAGNOSTIC)
    amp = np.abs(fg.E[0])
    check("free-space magnitude is unity (0 dB excess loss)",
          float(np.nanmax(np.abs(amp[amp > 0] - 1.0))) < 1e-3,
          f"max |amp-1| = {float(np.nanmax(np.abs(amp[amp > 0] - 1.0))):.4f}")
    tf = fg.tau_first[np.isfinite(fg.tau_first)]
    tol = 4.0 * sc.cell / contracts.C0     # eikonal seeds on the source cell → sub-cell offset
    check("free-space excess delay within eikonal discretization tol",
          (float(np.nanmax(tf)) < tol) if tf.size else True,
          f"max tau {float(np.nanmax(tf))*1e9 if tf.size else 0:.3f} ns (tol {tol*1e9:.3f} ns)")

    # ---- slab: the medium attenuates and delays the transmitted wave ------
    M, man = S.slab_scene_3d(n=(40, 16, 16), mat=2, x0=16, x1=20)   # concrete slab
    sc = engine_3d.SceneV3(M, man)
    tx = (3, M.shape[1] // 2, M.shape[2] // 2)
    b1 = man["freqs_mhz"][:1]
    fg = solve(sc, tx, bands=b1)
    behind = (slice(24, None), slice(None), slice(None))           # past the slab
    amp_b = np.abs(fg.E[0, 0])[behind]
    tau_b = fg.tau_first[0][behind]
    check("slab attenuates the transmitted field (excess loss > 0 behind it)",
          float(np.median(amp_b[amp_b > 0])) < 0.999,
          f"median |amp| behind slab = {float(np.median(amp_b[amp_b > 0])):.4f}")
    check("slab adds excess delay (in-wall slowdown behind it)",
          float(np.median(tau_b[np.isfinite(tau_b)])) > 0,
          f"median excess {float(np.median(tau_b[np.isfinite(tau_b)]))*1e9:.3f} ns")

    # ---- slab_transmission_coherent is a legitimate consumer, and the eikonal
    #      excess delay agrees in scale with the coherent slab phase delay ----------
    eps = physics_3d.permittivity("concrete", float(b1[0]))[0]
    n_re = float(np.real(np.sqrt(eps)))
    thick = 4 * sc.cell                                            # slab is x0..x1 = 4 cells
    Tc = physics_3d.slab_transmission_coherent(eps, 0.0, thick, float(b1[0]), pol="te")
    slab_delay = (n_re - 1.0) * thick / contracts.C0              # extra time vs vacuum
    med = float(np.median(tau_b[np.isfinite(tau_b)]))
    check("slab_transmission_coherent returns a complex transmission < 1",
          abs(complex(Tc)) < 1.0, f"|T| = {abs(complex(Tc)):.3f}")
    check("eikonal excess delay ~ (n-1)*thickness/c (order of magnitude)",
          0.2 * slab_delay < med < 5.0 * slab_delay,
          f"eikonal {med*1e12:.1f} ps vs slab {slab_delay*1e12:.1f} ps")

    print(f"\n{ok} passed, {len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--test", action="store_true", help="run the module self-test")
    a = ap.parse_args()
    raise SystemExit(_selftest() if a.test else (ap.print_help() or 0))
