"""
Core SceneV3 physics — the invariants the engine must satisfy before any mechanism
module is layered on top. These run against ANALYTIC synthetic scenes where the right
answer is known in closed form.
"""
from __future__ import annotations

import numpy as np
import pytest

import engine_3d
import invariants as inv
import synth3d as S


# --------------------------------------------------------------------------- vacuum
def test_free_space_is_exactly_fspl():
    """Mode 1 (vacuum): with no matter, PL must reduce to the spreading term exactly."""
    M, man = S.free_space_scene(n=(40, 40, 40))
    sc = engine_3d.SceneV3(M, man)
    tx = S.centre(M.shape)

    pl = sc.pathloss_maps(tx)                       # (nf, NX, NY, NZ)
    d = S.distance_grid(M.shape, tx, man["cell_size_m"])

    for i, f in enumerate(man["freqs_mhz"]):
        want = S.fspl_db(d, f)
        got = pl[i]
        m = np.isfinite(got) & (d > man["cell_size_m"])
        err = np.abs(got[m] - want[m]).max()
        assert err < 0.01, f"{f:.0f} MHz: free-space PL deviates from FSPL by {err:.4f} dB"


def test_free_space_no_absorption_no_crossings():
    M, man = S.free_space_scene(n=(32, 32, 32))
    sc = engine_3d.SceneV3(M, man)
    xl = sc.crossing_loss(S.centre(M.shape))
    assert np.nanmax(np.abs(xl)) < 1e-6, "air produced crossing loss"


# --------------------------------------------------------------------------- invariants
def test_band_ordering_free_space():
    M, man = S.free_space_scene(n=(32, 32, 32))
    sc = engine_3d.SceneV3(M, man)
    inv.assert_band_ordering(sc.pathloss_maps(S.centre(M.shape)), man["freqs_mhz"])


def test_band_ordering_through_slab():
    """Higher frequency must lose at least as much through matter, not less."""
    M, man = S.slab_scene_3d(n=(48, 24, 24), mat=2, x0=22, x1=25)
    sc = engine_3d.SceneV3(M, man)
    inv.assert_band_ordering(sc.pathloss_maps((8, 12, 12)), man["freqs_mhz"], min_frac=0.98)


def test_reciprocity_through_slab():
    """PL(a->b) == PL(b->a) across an obstruction - catches asymmetric ray sampling."""
    M, man = S.slab_scene_3d(n=(48, 20, 20), mat=2, x0=22, x1=25)
    sc = engine_3d.SceneV3(M, man)
    a, b = (6, 10, 10), (40, 10, 10)
    f = man["freqs_mhz"][1]
    delta = inv.assert_reciprocity(lambda t: sc.pathloss_map(t, f), a, b, tol_db=1.0)
    assert np.isfinite(delta)


def test_slab_adds_loss_over_free_space():
    """A wall must cost something, and the cost must be finite."""
    n, x0, x1 = (48, 20, 20), 22, 25
    tx, rx = (6, 10, 10), (40, 10, 10)
    f = 3500.0

    M0, man = S.free_space_scene(n=n)
    M1, _ = S.slab_scene_3d(n=n, mat=2, x0=x0, x1=x1)
    pl0 = engine_3d.SceneV3(M0, man).pathloss_map(tx, f)[rx]
    pl1 = engine_3d.SceneV3(M1, man).pathloss_map(tx, f)[rx]

    excess = float(pl1 - pl0)
    assert excess > 0.5, f"concrete slab added only {excess:.2f} dB"
    assert excess < 120.0, f"concrete slab added an implausible {excess:.1f} dB"


def test_monotone_los_free_space():
    M, man = S.free_space_scene(n=(48, 24, 24))
    sc = engine_3d.SceneV3(M, man)
    tx = (8, 12, 12)
    inv.assert_monotone_los(sc.pathloss_map(tx, 3500.0), tx, axis=0)


def test_no_cell_beats_free_space():
    """The FSPL floor: obstruction can never help."""
    M, man = S.slab_scene_3d(n=(48, 24, 24), mat=2, x0=22, x1=25)
    sc = engine_3d.SceneV3(M, man)
    tx = (6, 12, 12)
    d = S.distance_grid(M.shape, tx, man["cell_size_m"])
    inv.assert_above_fspl(sc.pathloss_map(tx, 3500.0), S.fspl_db(d, 3500.0))


# --------------------------------------------------------------------------- timing
def test_arrival_time_is_causal():
    """T >= d/c everywhere: the eikonal cannot outrun light."""
    M, man = S.free_space_scene(n=(32, 32, 32))
    sc = engine_3d.SceneV3(M, man)
    tx = S.centre(M.shape)
    T = sc.arrival_time(tx, 3500.0)
    d = S.distance_grid(M.shape, tx, man["cell_size_m"])
    inv.assert_causal(T, d, man["cell_size_m"])


def test_arrival_time_free_space_matches_geometric():
    """In vacuum the eikonal must reduce to d/c."""
    M, man = S.free_space_scene(n=(32, 32, 32))
    sc = engine_3d.SceneV3(M, man)
    tx = S.centre(M.shape)
    T = sc.arrival_time(tx, 3500.0)
    G = sc.geometric_time(tx)
    m = np.isfinite(T) & np.isfinite(G) & (G > 0)
    rel = np.abs(T[m] - G[m]) / G[m]
    assert np.median(rel) < 0.05, f"eikonal deviates from d/c in vacuum (median {np.median(rel):.3f})"


def test_dense_material_slows_arrival():
    """Refraction is implicit in the eikonal: a slab must DELAY arrival, never advance it."""
    n, tx, f = (48, 20, 20), (6, 10, 10), 3500.0
    M0, man = S.free_space_scene(n=n)
    M1, _ = S.slab_scene_3d(n=n, mat=2, x0=22, x1=25)
    T0 = engine_3d.SceneV3(M0, man).arrival_time(tx, f)
    T1 = engine_3d.SceneV3(M1, man).arrival_time(tx, f)
    far = (slice(30, None), slice(None), slice(None))
    m = np.isfinite(T0[far]) & np.isfinite(T1[far])
    assert (T1[far][m] >= T0[far][m] - 1e-12).mean() > 0.98, \
        "slab advanced the wavefront - slowness field is wrong"
