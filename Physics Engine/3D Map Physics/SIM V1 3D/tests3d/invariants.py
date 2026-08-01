"""
Reusable physics invariants — the assertions any correct propagation result must satisfy.

These are mechanism-agnostic: they apply to the engine, to each of the six mechanism
modules, to the combiner, and (as differentiable penalties) to the ML surrogate. Four of
them are lifts of checks already written in `phase_d_calibrate.sanity()` and the 2D
training notebook's D.3 block — pulled up here so there is one implementation.

Anything that violates these is wrong regardless of how plausible the picture looks.
"""
from __future__ import annotations

import numpy as np

C0 = 299_792_458.0


def _finite(a):
    a = np.asarray(a)
    return a[np.isfinite(a)]


# --------------------------------------------------------------------------- basic
def assert_finite(*arrays, names=None):
    for i, a in enumerate(arrays):
        a = np.asarray(a)
        bad = ~np.isfinite(a)
        if bad.any():
            nm = (names[i] if names else f"array[{i}]")
            raise AssertionError(f"{nm}: {bad.sum()} non-finite of {a.size}")


def assert_above_fspl(pl_db, fspl_db, tol_db=0.1, max_frac=1e-3, name="PL"):
    """A passive obstructed channel cannot beat free space.

    Mirrors the FSPL floor already applied in engine_v2.pathloss_maps.
    """
    pl_db, fspl_db = np.asarray(pl_db), np.asarray(fspl_db)
    m = np.isfinite(pl_db) & np.isfinite(fspl_db)
    viol = (pl_db[m] < fspl_db[m] - tol_db)
    frac = viol.mean() if viol.size else 0.0
    if frac > max_frac:
        worst = (fspl_db[m] - pl_db[m]).max()
        raise AssertionError(
            f"{name}: {100*frac:.3f}% of cells below FSPL (max {worst:.2f} dB) "
            f"- exceeds {100*max_frac:.3f}%")
    return frac


def assert_band_ordering(pl_by_band, freqs_mhz, min_frac=0.99, name="PL"):
    """Higher frequency must not propagate better: PL(f_lo) <= PL(f_hi)."""
    pl = np.asarray(pl_by_band)
    order = np.argsort(np.asarray(freqs_mhz, float))
    for a, b in zip(order[:-1], order[1:]):
        lo, hi = pl[a], pl[b]
        m = np.isfinite(lo) & np.isfinite(hi)
        frac = (lo[m] <= hi[m] + 1e-6).mean() if m.any() else 1.0
        if frac < min_frac:
            raise AssertionError(
                f"{name}: band ordering violated {freqs_mhz[a]:.0f}->{freqs_mhz[b]:.0f} MHz "
                f"- only {100*frac:.2f}% ordered (need {100*min_frac:.2f}%)")


def assert_causal(tau_s, d_m, cell_m, slack_cells=3.0, name="T"):
    """Nothing arrives faster than light: tau >= d/c (minus raster slack)."""
    tau, d = np.asarray(tau_s), np.asarray(d_m)
    m = np.isfinite(tau) & np.isfinite(d)
    slack = slack_cells * cell_m / C0
    viol = tau[m] < (d[m] / C0 - slack)
    if viol.any():
        worst = ((d[m] / C0) - tau[m]).max()
        raise AssertionError(
            f"{name}: {viol.sum()} superluminal cells (worst {worst*1e9:.2f} ns early)")


def assert_monotone_los(pl_db, tx, axis=0, min_frac=0.95, name="PL"):
    """Along a clear line of sight, path loss must increase with distance."""
    pl = np.asarray(pl_db)
    sl = [slice(None)] * pl.ndim
    for a in range(pl.ndim):
        if a != axis:
            sl[a] = int(tx[a])
    ray = pl[tuple(sl)]
    fwd = ray[int(tx[axis]) + 1:]
    fwd = fwd[np.isfinite(fwd)]
    if fwd.size < 3:
        return 1.0
    frac = (np.diff(fwd) >= -1e-6).mean()
    if frac < min_frac:
        raise AssertionError(
            f"{name}: LOS not monotone along axis {axis} - only {100*frac:.1f}% "
            f"non-decreasing (need {100*min_frac:.1f}%)")
    return frac


# --------------------------------------------------------------------------- field
def assert_field_matches_power(E, pl_db, tol_db=0.01, name="E"):
    """The complex-field contract: -10*log10(sum_pol |E|^2) == PL_dB.

    This is what keeps the complex spine a strict superset of the working engine.
    """
    E = np.asarray(E)
    p = np.sum(np.abs(E) ** 2, axis=0) if E.ndim > np.asarray(pl_db).ndim else np.abs(E) ** 2
    with np.errstate(divide="ignore"):
        pl_from_E = -10.0 * np.log10(np.maximum(p, 1e-300))
    m = np.isfinite(pl_from_E) & np.isfinite(pl_db)
    if not m.any():
        raise AssertionError(f"{name}: nothing finite to compare")
    err = np.abs(pl_from_E[m] - np.asarray(pl_db)[m]).max()
    if err > tol_db:
        raise AssertionError(f"{name}: |E|^2 disagrees with PL by {err:.4f} dB (tol {tol_db})")
    return err


def assert_reciprocity(solve_fn, a, b, tol_db=0.5, name="PL"):
    """PL(a->b) must equal PL(b->a). Catches asymmetric ray sampling."""
    ab = float(np.asarray(solve_fn(a))[tuple(b)])
    ba = float(np.asarray(solve_fn(b))[tuple(a)])
    if not (np.isfinite(ab) and np.isfinite(ba)):
        raise AssertionError(f"{name}: non-finite reciprocity pair {ab}, {ba}")
    if abs(ab - ba) > tol_db:
        raise AssertionError(
            f"{name}: reciprocity broken {a}->{b}={ab:.2f} dB vs {b}->{a}={ba:.2f} dB "
            f"(delta {abs(ab-ba):.2f} > {tol_db})")
    return abs(ab - ba)


def assert_energy_closes(p_in, p_out, p_abs, tol_frac=0.01, name="energy"):
    """P_in == P_out + P_absorbed. The scattering split must not manufacture power."""
    tot = p_out + p_abs
    if p_in <= 0:
        raise AssertionError(f"{name}: non-positive input power {p_in}")
    err = abs(tot - p_in) / p_in
    if err > tol_frac:
        raise AssertionError(
            f"{name}: not conserved - in={p_in:.6g} out+abs={tot:.6g} ({100*err:.3f}% off)")
    return err


def assert_continuous_across_boundary(field, boundary_idx, axis=0, tol_db=0.5,
                                      name="total field"):
    """UTD's whole purpose: the total field is continuous across a shadow boundary.

    A jump here means the diffracted term is not correctly cancelling the GO
    discontinuity - usually a wrong wedge face assignment.
    """
    f = np.asarray(field)
    lo = np.take(f, boundary_idx - 1, axis=axis)
    hi = np.take(f, boundary_idx + 1, axis=axis)
    m = np.isfinite(lo) & np.isfinite(hi)
    if not m.any():
        raise AssertionError(f"{name}: nothing finite at boundary")
    jump = np.abs(hi[m] - lo[m]).max()
    if jump > tol_db:
        raise AssertionError(
            f"{name}: discontinuous across shadow boundary - {jump:.2f} dB jump (tol {tol_db})")
    return jump


# --------------------------------------------------------------------------- summary
def summarize(pl_db, name="PL") -> dict:
    v = _finite(pl_db)
    if v.size == 0:
        return {"name": name, "n_finite": 0}
    return {"name": name, "n_finite": int(v.size),
            "min": float(v.min()), "median": float(np.median(v)),
            "max": float(v.max()), "mean": float(v.mean())}
