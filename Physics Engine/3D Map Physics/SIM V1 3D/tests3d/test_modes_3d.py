"""
M3 — the four propagation modes.

The load-bearing test here is the VACUUM GATE: in an all-air scene the engine must
reproduce closed-form free space. It is the only configuration whose answer is known
exactly, so it is the only place a solver bug cannot hide behind "well, buildings are
complicated". Everything else in this file guards the two things that make a mode
different from the indoor default — which scene it builds, and where it puts the source.

The O2I tests deliberately assert the DERIVATION rather than the number. The 2D manifest
carried `bs_preset.bearing_deg = 135` with a comment calling it a demo placeholder; the
whole point of M3's O2I mode is that the bearing now falls out of the floor plan's
georeference and the known Forte Hall coordinates, so a test that just pinned 237.0 would
re-create exactly the failure mode it replaced.
"""
from __future__ import annotations

import numpy as np
import pytest

import modes_3d as MD
import Path_Loss_3D as PLM

C0 = MD.C0


# --------------------------------------------------------------------------- registry
def test_four_modes_in_plan_order():
    assert MD.list_modes() == ["vacuum", "indoor", "o2i", "outdoor"]


def test_describe_never_raises_and_reports_availability():
    """The browser calls describe() for every mode, including ones with no scene built."""
    for k in MD.list_modes():
        d = MD.describe(k)
        assert d["key"] == k
        assert d["summary"] and d["label"]
        assert isinstance(d["available"], bool)
        if not d["available"]:
            assert d["unavailable_reason"], f"{k}: unavailable with no reason given"


def test_unknown_mode_is_a_hard_error():
    with pytest.raises(KeyError):
        MD.get_mode("indor")


def test_each_mode_declares_a_source_model():
    for k in MD.list_modes():
        assert MD.get_mode(k).source in ("point", "plane_wave", "none")
    # only O2I is a plane wave; if that changes, the browser's Tx controls are wrong
    assert [k for k in MD.list_modes() if MD.get_mode(k).source == "plane_wave"] == ["o2i"]


# --------------------------------------------------------------------------- vacuum
@pytest.fixture(scope="module")
def vac():
    bands = [2442.0, 3500.0]
    sc, man = MD.build_scene("vacuum", n=(30, 30, 30), bands=bands)
    tx = (15, 15, 15)
    return sc, tx, bands, PLM.solve(sc, tx, bands=bands)


def test_vacuum_is_all_air(vac):
    sc, *_ = vac
    assert not sc.M.any(), "vacuum scene has material in it"
    assert sc.inside.all()


def test_vacuum_reproduces_closed_form_fspl(vac):
    """PL == 20log10(d) + 20log10(f) - 27.55. The gate."""
    sc, tx, bands, fg = vac
    d = np.linalg.norm(sc.P - np.asarray(tx, np.float32), axis=1).reshape(sc.M.shape)
    d_m = np.maximum(d * sc.cell, sc.d0)
    pl = fg.pathloss_db()
    for i, f in enumerate(bands):
        ana = 20 * np.log10(d_m) + 20 * np.log10(f) - 27.55
        err = float(np.nanmax(np.abs(pl[i] - ana)))
        assert err < 0.05, f"{f} MHz: engine deviates from free space by {err:.4f} dB"


def test_vacuum_matches_the_engines_own_free_space_reference(vac):
    sc, tx, bands, fg = vac
    fs = PLM.free_space_field(sc, tx, bands=bands)
    err = float(np.nanmax(np.abs(fg.pathloss_db() - fs.pathloss_db())))
    assert err < 1e-3, f"solve() and free_space_field() disagree by {err:.6f} dB"


def test_vacuum_is_causal(vac):
    """Nothing outruns c. Slack is the fast-marching discretization, not physics."""
    sc, tx, bands, fg = vac
    d = np.linalg.norm(sc.P - np.asarray(tx, np.float32), axis=1).reshape(sc.M.shape)
    t_direct = (d * sc.cell) / C0
    slack = 1.5 * sc.cell / C0
    worst = float(np.min(fg.tau_first[0] - t_direct))
    assert worst >= -slack, f"arrives {-worst * 1e9:.2f} ns early"


def test_vacuum_band_ordering(vac):
    """Higher frequency, more free-space loss — everywhere, with no exceptions."""
    _sc, _tx, bands, fg = vac
    pl = fg.pathloss_db()
    delta = pl[1] - pl[0]
    expect = 20 * np.log10(bands[1] / bands[0])
    assert np.allclose(delta, expect, atol=0.05), \
        f"band gap {np.median(delta):.3f} dB, expected {expect:.3f}"


def test_vacuum_defaults_to_the_production_grid():
    """A 48-cube would not line up with the indoor scene it exists to be compared to."""
    import json
    from pathlib import Path
    man = json.loads((Path(MD.HERE) / "manifest_3d.json").read_text())
    sc, _ = MD.build_scene("vacuum", bands=[2442.0])
    assert list(sc.M.shape) == list(man["grid_shape"])


# --------------------------------------------------------------------------- o2i
@pytest.fixture(scope="module")
def geom():
    return MD.forte_hall_geometry()


def test_o2i_geometry_is_derived_from_the_georeference(geom):
    """Recompute the range independently from the raw lon/lat, not from the return value."""
    lon0, lat0 = geom["origin_lonlat"]
    tx_lon, tx_lat = geom["tx_lonlat"]
    R = MD.R_EARTH
    e = np.radians(tx_lon - lon0) * R * np.cos(np.radians(lat0))
    n = np.radians(tx_lat - lat0) * R
    bx, by = geom["building_centre_local_m"]
    d = float(np.hypot(e - bx, n - by))
    assert abs(d - geom["distance_m"]) < 1.0, "reported range is not the derived range"
    assert 400 < d < 430, f"Forte Hall should be ~415 m away, got {d:.1f}"


def test_o2i_bearing_is_not_the_placeholder(geom):
    """The 135 deg in the 2D manifest was explicitly a demo placeholder."""
    assert abs(geom["arrival_bearing_deg"] - 237.0) < 1.0
    assert abs(geom["arrival_bearing_deg"] - 135.0) > 50.0
    assert abs(((geom["propagation_bearing_deg"] - geom["arrival_bearing_deg"]) % 360)
               - 180.0) < 1e-6, "propagation and arrival bearings must be opposite"


def test_o2i_outdoor_leg_fspl_matches_the_measured_bands(geom):
    lo = MD.fspl_db(geom["distance_m"], 619.0)
    hi = MD.fspl_db(geom["distance_m"], 3710.0)
    assert abs(lo - 80.7) < 0.5, f"{lo:.1f} dB at 619 MHz"
    assert abs(hi - 96.2) < 0.5, f"{hi:.1f} dB at 3710 MHz"


def test_3gpp_penetration_classes():
    assert MD.O2I_DB == {"low_loss": 15.0, "high_loss": 28.0}
    assert MD.O2I_DB["high_loss"] - MD.O2I_DB["low_loss"] == 13.0


@pytest.mark.slow
def test_facade_only_lights_the_side_facing_the_base_station(scene):
    """A wave from 237 deg and one from 57 deg must illuminate DISJOINT facades.

    This is the property that makes O2I directional at all. If the normal estimate were
    broken, every envelope voxel would light for every bearing and the indoor map would be
    the same whatever direction the cell is in.
    """
    g = MD.forte_hall_geometry()
    near, _ = MD.facade_sources_3d(scene, g["arrival_bearing_deg"])
    far, _ = MD.facade_sources_3d(scene, g["propagation_bearing_deg"])
    assert len(near) > 100, f"only {len(near)} lit facade voxels"
    assert len(far) > 100
    a = {tuple(map(int, p)) for p in near}
    b = {tuple(map(int, p)) for p in far}
    assert not (a & b), f"{len(a & b)} voxels lit from both sides — normals are wrong"


@pytest.mark.slow
def test_facade_thinning_stays_spread_along_the_arrival_direction(scene):
    """max_sources must sample the whole lit facade, not one end of it.

    Array order is X-major, so a plain stride would over-sample one edge and leave the
    other dark — which would tilt the whole indoor map.
    """
    g = MD.forte_hall_geometry()
    full, proj_full = MD.facade_sources_3d(scene, g["arrival_bearing_deg"])
    few, proj_few = MD.facade_sources_3d(scene, g["arrival_bearing_deg"], max_sources=32)
    assert len(few) == 32
    assert proj_few.min() == pytest.approx(proj_full.min(), abs=1e-3)
    assert proj_few.max() == pytest.approx(proj_full.max(), abs=1e-3)


# --------------------------------------------------------------------------- outdoor
def test_outdoor_uses_a_barrier_class_for_buildings():
    m = MD.get_mode("outdoor")
    assert 3 in m.barrier_classes, \
        "buildings must be an eikonal barrier or the front walks through them"
    assert "scattering" not in m.mechanisms, \
        "diffuse scattering is indoor-calibrated; enabling it at 1 m cells is extrapolation"


def test_outdoor_reports_how_to_build_its_scene_when_missing():
    d = MD.describe("outdoor")
    if not d["available"]:
        assert "voxelize_city" in d["unavailable_reason"]


# --------------------------------------------------------------------------- wiring
def test_source_for_returns_the_right_shape_per_mode(scene):
    s = MD.source_for("indoor", scene, tx=(66, 5, 54))
    assert s["kind"] == "point" and s["tx_vox"] == (66, 5, 54)
    o = MD.source_for("o2i", scene)
    assert o["kind"] == "plane_wave"
    assert "arrival_bearing_deg" in o and "distance_m" in o


def test_entry_points_take_the_default_stack_from_the_registry():
    """The exporters must not carry their own idea of what a mode includes.

    `precompute_volumes` used to hold `DEFAULT_MECHANISMS = "path_loss,reflection,
    diffraction"`, so a default indoor solve there was missing the scattering the registry
    lists — two entry points, two different meanings of "indoor".
    """
    import precompute_volumes as PV

    for k in MD.list_modes():
        assert PV.resolve_mechanisms(k) == list(MD.get_mode(k).mechanisms), \
            f"{k}: the batch exporter disagrees with the registry"
    assert "scattering" in PV.resolve_mechanisms("indoor")
    assert not hasattr(PV, "DEFAULT_MECHANISMS"), \
        "a second definition of the default stack is back beside the registry"


def test_explicit_mechanisms_flag_still_wins():
    """Including when it spells exactly what the old hardcoded default said — comparing
    the argument against a default string read that as "no flag given"."""
    import precompute_volumes as PV

    assert PV.resolve_mechanisms("indoor", "path_loss,reflection") == \
        ["path_loss", "reflection"]
    assert PV.resolve_mechanisms("indoor", "path_loss,reflection,diffraction") == \
        ["path_loss", "reflection", "diffraction"]
    assert PV.resolve_mechanisms("o2i", "path_loss") == ["path_loss"]


def test_single_tx_exporter_resolves_through_the_same_helper():
    """export_pl_volume shares the batch code path, so it must share the resolution too.

    Asserted on the source because the choice happens inside `main()`, after a scene build
    and a full solve that no unit test can afford.
    """
    import inspect

    import export_pl_volume as EX

    assert "PV.resolve_mechanisms" in inspect.getsource(EX.main)
    assert not hasattr(EX, "DEFAULT_MECHANISMS")


def test_manifest_carries_the_mode_and_the_o2i_derivation(manifest):
    assert manifest.get("mode") in MD.list_modes()
    modes = manifest.get("modes")
    assert modes, "manifest has no modes block — the browser cannot explain the selector"
    g = modes["o2i"]["geometry"]
    assert abs(g["distance_m"] - 415.9) < 2.0
    assert abs(g["arrival_bearing_deg"] - 237.0) < 1.0
    assert "derivation" in g, "the geometry must record how it was obtained"
