#!/usr/bin/env python3
"""
Combine_3D — the physics-exact field combiner.

Sums the six mechanisms' contributions into a total field. Deliberately NOT learned: the
combination rule is cheap and exactly known, so letting a network approximate it would
add error and destroy energy bookkeeping for nothing. Keeping it analytic also keeps the
learned pieces (per-mechanism surrogates) small and well-posed.

RULES
    1. P_coh = sum_pol | sum_mech E |^2      mechanisms declaring COHERENT
    2. P_inc = sum_mech p_incoh              diffuse scattering (mutually uncorrelated)
    3. P_tot = P_coh + P_inc ;  PL = -10 log10(P_tot)
    4. T_first = min over contributions      (preserves the exact semantic the existing
                                              wavefront-sweep animation consumes)
    5. FSPL floor: a passive obstructed channel cannot deliver more than free space.

BANDWIDTH — not optional
    A single-tone coherent sum produces deep nulls that no 20/100 MHz receiver ever
    observes; a sim that shows them would fail against the scanner for the wrong reason.
    `bandwidth_hz` averages |sum E(f_k)|^2 over n_sub sub-carriers across the channel.
    Delays are already known, so this costs n_sub complex exponentials — nearly free.
    This is the "reintroduce the statistical structure" lever as a single parameter.

LEGACY BRIDGE
    `to_legacy_volumes()` returns exactly the (PL, T) arrays `export_pl_volume.py`
    already serializes, so improved physics reaches the browser with ZERO exporter or
    frontend changes. `test_combiner.py` asserts that combining path-loss-only reproduces
    `SceneV3.pathloss_maps` to < 0.01 dB — the proof that this is a strict superset of
    the working engine.

usage:
    python Combine_3D.py --test
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

import numpy as np

from _bootstrap import SIM3D, contracts, engine_3d  # noqa: F401

C0 = contracts.C0


# --------------------------------------------------------------------------- combine
def combine(contribs: Sequence["contracts.FieldGrid"], *,
            bandwidth_hz: float | None = None, n_sub: int = 8,
            floor_at_fspl: bool = True, scene=None, tx=None) -> "contracts.CombinedField":
    """Coherently+incoherently sum mechanism contributions.

    contribs      FieldGrids for the SAME tx / bands / grid
    bandwidth_hz  if set, average power over n_sub sub-carriers across the channel
    floor_at_fspl clamp PL so no cell beats free space (needs `scene` and `tx`)
    """
    if not contribs:
        raise ValueError("combine() needs at least one contribution")
    ref = contribs[0]
    freqs = np.asarray(ref.freqs_mhz, float)
    for c in contribs[1:]:
        if c.E.shape != ref.E.shape:
            raise ValueError(f"shape mismatch: {c.mechanism} {c.E.shape} vs {ref.E.shape}")
        if not np.allclose(c.freqs_mhz, freqs):
            raise ValueError(f"band mismatch on {c.mechanism}")

    coh = [c for c in contribs if c.combine_as == contracts.COHERENT]
    inc = [c for c in contribs if c.combine_as == contracts.INCOHERENT]

    # --- 1. coherent sum of complex fields --------------------------------
    E_tot = np.zeros_like(ref.E)
    for c in coh:
        E_tot += c.E

    if bandwidth_hz and coh:
        # Sub-carrier average: re-phase each contribution by its own delay at f_k. The
        # magnitude is held at the band-centre value (loss varies slowly across a channel
        # compared with phase), so only the interference pattern is averaged.
        P_coh = np.zeros(ref.E.shape[1:], np.float64)
        offs = np.linspace(-0.5, 0.5, int(n_sub)) * float(bandwidth_hz)
        for df in offs:
            acc = np.zeros_like(ref.E)
            for c in coh:
                tau = np.nan_to_num(c.tau_first, nan=0.0, posinf=0.0)
                acc += c.E * np.exp(-1j * 2.0 * np.pi * df * tau).astype(np.complex64)
            P_coh += np.sum(np.abs(acc).astype(np.float64) ** 2, axis=0)
        P_coh /= float(n_sub)
    else:
        # float64: deep-shadow |E|^2 ~1e-54 underflows float32 (see contracts.power)
        P_coh = np.sum(np.abs(E_tot).astype(np.float64) ** 2, axis=0)

    # --- 2. incoherent (diffuse) power ------------------------------------
    P_inc = np.zeros_like(P_coh)
    for c in inc + [c for c in coh if c.p_incoh is not None]:
        if c.p_incoh is not None:
            P_inc += np.asarray(c.p_incoh, np.float64)

    P_tot = P_coh + P_inc

    # --- 3. per-mechanism power (for the frontend contribution-share view) -
    per_mech = {}
    for c in contribs:
        per_mech[c.mechanism] = c.power()

    # --- 4. earliest arrival across all mechanisms ------------------------
    # Diagnostics (refraction/absorption) carry an excess-delay or no-arrival tau that is
    # not a real first arrival, so they must not participate in T_first.
    T = np.full(ref.shape, np.inf, np.float32)
    for c in contribs:
        if c.combine_as == contracts.DIAGNOSTIC:
            continue
        t = np.asarray(c.tau_first, np.float32)
        t = np.nanmin(t, axis=0) if t.ndim == 4 else t
        T = np.minimum(T, np.where(np.isfinite(t), t, np.inf))

    with np.errstate(divide="ignore"):
        PL = (-10.0 * np.log10(np.maximum(P_tot, 1e-300))).astype(np.float32)

    # --- 5. FSPL floor -----------------------------------------------------
    if floor_at_fspl and scene is not None and tx is not None:
        import Path_Loss_3D as PLM
        fs = PLM.free_space_field(scene, tx, bands=freqs).pathloss_db()
        PL = np.maximum(PL, fs).astype(np.float32)

    return contracts.CombinedField(
        E_total=E_tot, P_lin=P_tot, PL_db=PL, T_first=T, per_mech_lin=per_mech,
        freqs_mhz=freqs, tx_vox=ref.tx_vox,
        meta={"mechanisms": [c.mechanism for c in contribs],
              "coherent": [c.mechanism for c in coh],
              "incoherent": [c.mechanism for c in inc],
              "bandwidth_hz": bandwidth_hz, "n_sub": int(n_sub) if bandwidth_hz else 0,
              "fspl_floored": bool(floor_at_fspl and scene is not None)})


# --------------------------------------------------------------------------- bridge
def to_legacy_volumes(cf: "contracts.CombinedField") -> tuple[np.ndarray, np.ndarray]:
    """(PL (NF,NX,NY,NZ) dB, T (NX,NY,NZ) seconds) — exactly what export_pl_volume.py
    serializes today. This is the zero-change path to the browser."""
    return np.asarray(cf.PL_db, np.float32), np.asarray(cf.T_first, np.float32)


def rms_delay_spread(cf: "contracts.CombinedField", contribs) -> np.ndarray:
    """Power-weighted RMS delay spread (NF, NX, NY, NZ) seconds.

    The observable that tests the COMBINER rather than the path-loss model, and the one
    the scanner's unused `Ref Signal - Delay Spread` column (5,951 samples) can check.
    """
    # float64 throughout: tau ~ 1e-7 s, so E[t^2] - E[t]^2 in float32 loses ~7 digits to
    # catastrophic cancellation and reports picoseconds of spread for a single path.
    P = np.zeros(cf.PL_db.shape, np.float64)
    m1 = np.zeros_like(P)
    m2 = np.zeros_like(P)
    for c in contribs:
        p = c.power().astype(np.float64)
        t = np.nan_to_num(np.asarray(c.tau_first, np.float64), posinf=0.0)
        P += p
        m1 += p * t
        m2 += p * t * t
    Ps = np.maximum(P, 1e-300)
    var = np.maximum(m2 / Ps - (m1 / Ps) ** 2, 0.0)
    return np.sqrt(var).astype(np.float32)


def k_factor_db(cf: "contracts.CombinedField", dominant="path_loss") -> np.ndarray:
    """Rician K = dominant path power / diffuse power, in dB (NF, NX, NY, NZ)."""
    dom = cf.per_mech_lin.get(dominant)
    if dom is None:
        raise KeyError(f"{dominant!r} not among {list(cf.per_mech_lin)}")
    rest = np.maximum(cf.P_lin - dom, 1e-300)
    with np.errstate(divide="ignore"):
        return (10.0 * np.log10(np.maximum(dom, 1e-300) / rest)).astype(np.float32)


# --------------------------------------------------------------------------- self-test
def _selftest() -> int:
    sys.path.insert(0, str(SIM3D / "tests3d"))
    import synth3d as S
    import Path_Loss_3D as PLM

    ok, fails = 0, []

    def check(name, cond, msg=""):
        nonlocal ok
        if cond:
            ok += 1
            print(f"  PASS  {name}")
        else:
            fails.append(name)
            print(f"  FAIL  {name}  {msg}")

    # ---- THE M1 GATE: path-loss-only must reproduce the engine exactly ----
    for scene_name, (M, man) in [
            ("free space", S.free_space_scene(n=(24, 24, 24))),
            ("slab", S.slab_scene_3d(n=(32, 16, 16), mat=2, x0=14, x1=17))]:
        sc = engine_3d.SceneV3(M, man)
        tx = (4, M.shape[1] // 2, M.shape[2] // 2)
        bands = man["freqs_mhz"][:2]
        fg = PLM.solve(sc, tx, bands=bands)
        cf = combine([fg], floor_at_fspl=False)
        pl_new, t_new = to_legacy_volumes(cf)
        pl_engine = PLM.direct_pl_db(sc, tx, bands=bands)
        err = float(np.nanmax(np.abs(pl_new - pl_engine)))
        check(f"superset gate [{scene_name}]", err < 0.01, f"max err {err:.5f} dB")

    # ---- structural -------------------------------------------------------
    M, man = S.free_space_scene(n=(20, 20, 20))
    sc = engine_3d.SceneV3(M, man)
    tx = S.centre(M.shape)
    bands = man["freqs_mhz"][:2]
    fg = PLM.solve(sc, tx, bands=bands)
    cf = combine([fg], floor_at_fspl=False)

    check("T finite near tx", np.isfinite(cf.T_first).any())
    check("T causal", float(np.nanmin(cf.T_first)) >= 0.0)
    check("per-mech recorded", "path_loss" in cf.per_mech_lin)
    check("contribution share == 1 for a single mechanism",
          float(np.nanmax(np.abs(cf.contribution_fraction("path_loss") - 1.0))) < 1e-5)

    # ---- coherent addition is real: two identical fields => +6 dB ---------
    fg2 = PLM.solve(sc, tx, bands=bands)
    fg2.mechanism = "copy"
    cf2 = combine([fg, fg2], floor_at_fspl=False)
    delta = np.nanmedian(cf.PL_db - cf2.PL_db)
    check("coherent doubling = 6.02 dB", abs(float(delta) - 6.0206) < 0.01,
          f"got {float(delta):.4f} dB")

    # ---- incoherent addition: same power added as p_incoh => +3 dB -------
    fg3 = PLM.solve(sc, tx, bands=bands)
    fg3.mechanism = "diffuse"
    fg3.p_incoh = fg.power()
    fg3.E = np.zeros_like(fg3.E)
    fg3.combine_as = contracts.INCOHERENT
    cf3 = combine([fg, fg3], floor_at_fspl=False)
    d3 = np.nanmedian(cf.PL_db - cf3.PL_db)
    check("incoherent doubling = 3.01 dB", abs(float(d3) - 3.0103) < 0.01,
          f"got {float(d3):.4f} dB")

    # ---- bandwidth averaging must not change a single-contribution result -
    cf_bw = combine([fg], bandwidth_hz=20e6, n_sub=8, floor_at_fspl=False)
    check("bandwidth avg is a no-op for one path",
          float(np.nanmax(np.abs(cf_bw.PL_db - cf.PL_db))) < 1e-3)

    # ---- FSPL floor -------------------------------------------------------
    cf_f = combine([fg], floor_at_fspl=True, scene=sc, tx=tx)
    fs = PLM.free_space_field(sc, tx, bands=bands).pathloss_db()
    check("FSPL floor holds", bool((cf_f.PL_db >= fs - 1e-3).all()))

    print(f"\n{ok} passed, {len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="physics-exact field combiner")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    raise SystemExit(_selftest() if a.test else (ap.print_help() or 0))
