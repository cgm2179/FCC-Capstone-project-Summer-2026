"""
M2 — Diffraction_3D (UTD wedge diffraction on the voxel grid).

The mechanism that fills geometric shadow. The physics is reused verbatim from the 2D
engine's Kouyoumjian-Pathak implementation; what is new is the 3-D edge finding, the
Keller-cone angle beta0, and keeping D COMPLEX so the combiner can interfere.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

import Combine_3D as CMB
import Diffraction_3D as DIF
import engine_3d
import Path_Loss_3D as PLM
import synth3d as S

sys.path.insert(0, str(__import__("_bootstrap").SIM2D))
import physics_v2 as P2  # noqa: E402


def _half_wall(n=(48, 12, 44), x0=24, x1=26, zmax=20, cls=3):
    """A wall that stops part-way, leaving a genuine vertical knife edge."""
    M = np.zeros(n, np.int8)
    M[x0:x1, :, :zmax] = cls
    _m, man = S.free_space_scene(n=n)
    sc = engine_3d.SceneV3(M, man)
    sc.inside = (M == 0)
    return M, man, sc


# --------------------------------------------------------------------------- physics
def test_utd_coefficient_is_complex():
    """The whole point: D carries phase. Routing through utd_pathloss_db would take
    |D| and make interference impossible."""
    D = P2.utd_coefficient(np.pi, np.pi / 2, 10.0, 10.0, 2 * np.pi / 0.1227, n=2.0)
    assert np.iscomplexobj(D) and abs(complex(np.asarray(D).ravel()[0])) > 0


def test_knife_edge_cross_check_monotone():
    """ITU knife-edge loss must grow with the Fresnel parameter — the independent
    check on the UTD path that the 2D --test also asserts."""
    j = P2.knife_edge_j_db(np.linspace(0.5, 3.0, 12))
    assert np.all(np.diff(j) > 0)


# --------------------------------------------------------------------------- edges
def test_flat_wall_has_no_diffracting_edge():
    """A flat face subtends exactly pi of air (n=1.0), below n_min — so an infinite
    wall must produce NO edges. Catches an over-eager corner detector."""
    n = (40, 12, 32)
    M = np.zeros(n, np.int8)
    M[20:22, :, :] = 3                      # spans the whole slice: no ends, no corners
    edges = DIF.find_diffracting_edges_3d(M, M == 0, y_band=(2, 10), y_step=3)
    assert len(edges) == 0, f"flat wall produced {len(edges)} spurious edges"


def test_half_wall_end_is_found_as_near_knife_edge():
    """A wall END is a half-plane: n should approach 2.0."""
    M, _man, sc = _half_wall()
    edges = DIF.find_diffracting_edges_3d(M, sc.inside, y_band=(2, 10), y_step=3)
    assert edges, "no edge found at a wall end"
    assert max(e["n"] for e in edges) > 1.5, \
        f"wall end should read as near-knife-edge, got n={[round(e['n'],2) for e in edges]}"


def test_edge_anchors_are_in_air():
    """The relay must be anchored OUT of the structure: anchoring on the corner voxel
    charges the ray for the very wall it is bending around (~15 dB for concrete)."""
    M, _man, sc = _half_wall()
    for e in DIF.find_diffracting_edges_3d(M, sc.inside, y_band=(2, 10), y_step=3):
        assert M[e["ax"], e["ay"], e["az"]] == 0, "relay anchor is inside structure"


def test_wedge_parameter_in_valid_range():
    M, _man, sc = _half_wall()
    for e in DIF.find_diffracting_edges_3d(M, sc.inside, y_band=(2, 10), y_step=3):
        assert 1.15 <= e["n"] <= 2.0


# --------------------------------------------------------------------------- field
@pytest.fixture(scope="module")
def diffraction_case():
    M, man, sc = _half_wall()
    tx = (6, 6, 8)
    bands = man["freqs_mhz"][:1]
    edges = DIF.find_diffracting_edges_3d(M, sc.inside, y_band=(2, 10), y_step=3)
    direct = PLM.solve(sc, tx, bands=bands)
    diff = DIF.solve(sc, tx, bands=bands, edges=edges, n_edges=8)
    return M, sc, tx, bands, direct, diff


def test_field_is_finite_and_causal(diffraction_case):
    _M, _sc, _tx, _b, _d, diff = diffraction_case
    assert np.isfinite(np.abs(diff.E)).all()
    t = diff.tau_first[np.isfinite(diff.tau_first)]
    assert t.size and (t >= 0).all()


def test_phase_is_carried(diffraction_case):
    """arg(D) - k(s'+s) must actually vary, else the combiner cannot interfere."""
    _M, _sc, _tx, _b, _d, diff = diffraction_case
    live = np.abs(diff.E[0, 0]) > 0
    assert live.any() and float(np.ptp(np.angle(diff.E[0, 0][live]))) > 1.0


def test_diffraction_puts_energy_in_the_shadow(diffraction_case):
    """The reason the module exists: some shadow voxels must improve when it is added."""
    M, sc, tx, bands, direct, diff = diffraction_case
    only = CMB.combine([direct], floor_at_fspl=False).PL_db[0]
    both = CMB.combine([direct, diff], floor_at_fspl=False).PL_db[0]
    xs = np.arange(M.shape[0])[:, None, None]
    zs = np.arange(M.shape[2])[None, None, :]
    sh = (M == 0) & (xs > 30) & (zs < 16) & np.isfinite(only)
    assert sh.sum() > 100
    improved = (both[sh] < only[sh] - 1.0)
    assert improved.any(), "diffraction added no energy anywhere in the shadow"
    assert float(np.max(only[sh] - both[sh])) > 1.0


def test_los_region_barely_perturbed(diffraction_case):
    """Diffraction must not meaningfully disturb a well-lit line-of-sight region."""
    M, sc, tx, bands, direct, diff = diffraction_case
    only = CMB.combine([direct], floor_at_fspl=False).PL_db[0]
    both = CMB.combine([direct, diff], floor_at_fspl=False).PL_db[0]
    xs = np.arange(M.shape[0])[:, None, None]
    los = (M == 0) & (xs < 20) & np.isfinite(only) & np.isfinite(both)
    assert float(np.median(np.abs(only[los] - both[los]))) < 1.0


def test_combines_without_breaking_the_spine(diffraction_case):
    """Adding diffraction must not corrupt the (PL, T) contract the exporters consume."""
    _M, _sc, _tx, bands, direct, diff = diffraction_case
    cf = CMB.combine([direct, diff], floor_at_fspl=False)
    pl, t = CMB.to_legacy_volumes(cf)
    assert pl.shape == (len(bands),) + direct.shape and pl.dtype == np.float32
    assert t.shape == direct.shape
    assert np.isfinite(pl).any() and set(cf.per_mech_lin) == {"path_loss", "diffraction"}
