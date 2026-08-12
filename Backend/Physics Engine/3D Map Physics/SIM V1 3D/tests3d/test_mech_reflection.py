"""
M2 — Reflection_3D (specular multipath by image sources).

The mechanism that creates interference structure. On a voxel grid the floor and ceiling
bounces dominate indoors, which is why M0.4 (concrete floor slab + real ceiling) had to
land before this module could be trusted.
"""
from __future__ import annotations

import numpy as np
import pytest

import Combine_3D as CMB
import engine_3d
import Path_Loss_3D as PLM
import Reflection_3D as RFL
import synth3d as S


@pytest.fixture(scope="module")
def two_plate():
    M, man = S.two_plate_scene_3d(n=(40, 20, 24), gap_cells=10, mat=2)
    sc = engine_3d.SceneV3(M, man)
    sc.inside = (M == 0)
    return M, man, sc, (8, 10, 12), man["freqs_mhz"][:1]


# --------------------------------------------------------------------------- geometry
def test_finds_floor_and_ceiling(two_plate):
    _M, _man, sc, _tx, _b = two_plate
    planes = RFL.find_reflector_planes(sc, min_area_cells=20)
    y = [p for p in planes if p.axis == 1]
    assert len(y) >= 2, f"floor+ceiling not both found: {[p.name for p in planes]}"


def test_image_source_is_a_mirror(two_plate):
    _M, _man, sc, tx, _b = two_plate
    p = RFL.find_reflector_planes(sc, min_area_cells=20)[0]
    img = RFL.image_source(tx, p)
    assert abs(img[p.axis] - (2 * p.coord - tx[p.axis])) < 1e-9
    for a in (0, 1, 2):
        if a != p.axis:
            assert img[a] == tx[a], "mirroring must not move the off-axis components"


def test_free_space_has_no_reflectors():
    """No matter, no faces — the plane finder must not invent reflectors."""
    M, man = S.free_space_scene(n=(24, 24, 24))
    sc = engine_3d.SceneV3(M, man)
    assert RFL.find_reflector_planes(sc, min_area_cells=10) == []


# --------------------------------------------------------------------------- field
def test_reflected_field_is_finite_and_phase_carrying(two_plate):
    _M, _man, sc, tx, bands = two_plate
    fg = RFL.solve(sc, tx, bands=bands)
    assert np.isfinite(np.abs(fg.E)).all()
    live = np.abs(fg.E[0, 0]) > 0
    assert live.any() and float(np.ptp(np.angle(fg.E[0, 0][live]))) > 1.0


def test_reflection_arrives_after_the_direct_ray(two_plate):
    """A bounce is geometrically longer than the straight path — always."""
    _M, _man, sc, tx, bands = two_plate
    fg = RFL.solve(sc, tx, bands=bands)
    geo = sc.geometric_time(tx)
    live = np.abs(fg.E[0, 0]) > 0
    m = np.isfinite(fg.tau_first[0]) & np.isfinite(geo) & live
    assert m.any() and (fg.tau_first[0][m] >= geo[m] - 1e-12).mean() > 0.98


def test_creates_interference_not_a_uniform_lift(two_plate):
    """Coherent addition must produce BOTH constructive and destructive regions. A
    power-summed implementation would only ever reduce path loss, which is exactly the
    behaviour that makes a map look rendered rather than measured."""
    _M, _man, sc, tx, bands = two_plate
    direct = PLM.solve(sc, tx, bands=bands)
    refl = RFL.solve(sc, tx, bands=bands)
    one = CMB.combine([direct], floor_at_fspl=False).PL_db[0]
    both = CMB.combine([direct, refl], floor_at_fspl=False).PL_db[0]
    live = np.abs(refl.E[0, 0]) > 0
    d = (one - both)[np.isfinite(one) & np.isfinite(both) & live]
    assert (d > 0.1).any(), "no constructive interference anywhere"
    assert (d < -0.1).any(), "no destructive interference anywhere (power-summed?)"


def test_reflection_coefficient_is_complex_and_bounded(two_plate):
    """|R| <= 1 for a passive interface, and it must carry phase."""
    _M, _man, sc, _tx, bands = two_plate
    p = RFL.find_reflector_planes(sc, min_area_cells=20)[0]
    cos_th = np.linspace(0.05, 1.0, 25)
    R = RFL.reflection_coeff(sc, p, cos_th, bands[0])
    assert np.iscomplexobj(R)
    assert np.all(np.abs(R) <= 1.0 + 1e-6), f"|R| exceeded 1: max {np.abs(R).max():.4f}"
    assert float(np.ptp(np.angle(R))) > 1e-3, "R carries no phase"


# --------------------------------------------------------------------------- backends
def test_torch_matches_numpy(two_plate):
    """The GPU path must reproduce the reference to <= 0.01 dB. Both call the same
    reflection_coeff so this compares implementations, not two different physics."""
    torch = pytest.importorskip("torch")
    _M, _man, sc, tx, bands = two_plate
    planes = RFL.find_reflector_planes(sc, min_area_cells=20)
    a = RFL.solve(sc, tx, bands=bands, planes=planes, backend="numpy")
    b = RFL.solve(sc, tx, bands=bands, planes=planes, backend="torch")
    err = float(np.nanmax(np.abs(a.pathloss_db() - b.pathloss_db())))
    assert err <= 0.01, f"torch/numpy parity {err:.5f} dB on {b.meta.get('device')}"


def test_combines_without_breaking_the_spine(two_plate):
    _M, _man, sc, tx, bands = two_plate
    direct = PLM.solve(sc, tx, bands=bands)
    refl = RFL.solve(sc, tx, bands=bands)
    cf = CMB.combine([direct, refl], floor_at_fspl=False)
    pl, t = CMB.to_legacy_volumes(cf)
    assert pl.shape == (len(bands),) + direct.shape and t.shape == direct.shape
    assert set(cf.per_mech_lin) == {"path_loss", "reflection"}
