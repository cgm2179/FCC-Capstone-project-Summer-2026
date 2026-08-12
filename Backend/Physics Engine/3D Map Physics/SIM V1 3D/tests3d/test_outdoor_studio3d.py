"""Smoke checks for the Outdoor 3-D Full-Wave Studio bake (foundational scene)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# This file lives in SIM V1 3D/tests3d/ (TESTS3D symlink).
SIM3D = Path(__file__).resolve().parents[1]
WEB = SIM3D / "web" / "outdoor_studio3d"


@pytest.fixture(scope="module")
def scene():
    p = WEB / "scene.json"
    if not p.exists():
        pytest.skip("outdoor_studio3d scene not baked")
    return json.loads(p.read_text())


def test_scene_assets_exist(scene):
    assert (WEB / "material_grid.bin").exists()
    assert (WEB / "buildings.json").exists()
    assert scene["cell_size_m"] == 2.0
    assert scene["grid_shape"][0] > 100 and scene["grid_shape"][2] > 100


def test_material_classes(scene):
    nx, ny, nz = scene["grid_shape"]
    raw = (WEB / "material_grid.bin").read_bytes()
    assert len(raw) == nx * ny * nz
    g = np.frombuffer(raw, dtype=np.uint8).reshape(nx, ny, nz)
    present = set(int(c) for c in np.unique(g))
    assert 0 in present
    assert 2 in present, "concrete core missing"
    assert 5 in present, "glass facade missing"
    # foliage is best-effort (Overpass); do not hard-fail if absent
    ids = {m["id"] for m in scene["materials"]}
    assert ids == {0, 2, 4, 5}


def test_live_base_stations_placed(scene):
    live = [s for s in scene["stations"] if s["status"] == "live"]
    assert len(live) >= 5
    nx, ny, nz = scene["grid_shape"]
    for s in live:
        assert s["in_grid"], s["site_id"]
        assert 0 <= s["vx"] < nx and 0 <= s["vz"] < nz
        assert 0 <= s["vy"] < ny
        assert s["roof_height_m"] > 0


def test_stations_match_outdoor_2d_lonlat(scene):
    """Antenna XZ must sit on the Outdoor 2-D / BASE_STATIONS pin (no building hop)."""
    import sys
    sys.path.insert(0, str(SIM3D))
    import voxelize_city as VC

    man = {
        "epsg": 3857,
        "merc_scale": scene["merc_scale"],
        "merc_anchor": scene["merc_anchor"],
        "origin_units": scene["origin_units"],
        "cell_size_m": scene["cell_size_m"],
    }
    nx, _ny, nz = scene["grid_shape"]
    live = [s for s in scene["stations"] if s.get("status") == "live" and s.get("in_grid")]
    assert live
    for s in live:
        vx, vz = VC.lonlat_to_vox(man, float(s["lon"]), float(s["lat"]))
        assert abs(s["vx"] - round(vx)) <= 1, (s["site_id"], s["vx"], vx)
        assert abs(s["vz"] - round(vz)) <= 1, (s["site_id"], s["vz"], vz)
        cr = s.get("cad_roof") or {}
        world = cr.get("world")
        assert world is not None, s["site_id"]
        assert abs(world[0] - (vx - nx / 2)) < 0.75, (s["site_id"], world[0], vx)
        assert abs(world[2] - (vz - nz / 2)) < 0.75, (s["site_id"], world[2], vz)
        assert cr.get("xy_source") == "lonlat_2d"


def test_live_stations_on_osm_footprints(scene):
    """Each live Tx should land on (or within ~10 m of) an OSM extrusion of matching height."""
    bpath = WEB / "buildings.json"
    if not bpath.exists():
        pytest.skip("buildings.json missing")
    buildings = json.loads(bpath.read_text())["buildings"]
    cell = float(scene["cell_size_m"])
    live = [s for s in scene["stations"] if s.get("status") == "live" and s.get("in_grid")]
    assert len(live) >= 5
    for s in live:
        xm, zm = s["vx"] * cell, s["vz"] * cell
        prefer = float(s["roof_height_m"])
        best = None
        for b in buildings:
            cx = sum(b["x"]) / len(b["x"])
            cz = sum(b["z"]) / len(b["z"])
            d = ((cx - xm) ** 2 + (cz - zm) ** 2) ** 0.5
            score = (d + 0.15 * abs(float(b["h"]) - prefer), d, float(b["h"]))
            if best is None or score < best:
                best = score
        assert best[1] < 10.0, (s["site_id"], best)
        assert abs(best[2] - prefer) < 8.0 or best[1] < 6.0, (s["site_id"], best)


def test_cad_forte_align_present(scene):
    cad = scene.get("cad") or {}
    assert cad.get("align_site_id") == "BS7_forte_hall"
    off = cad.get("align_offset_local_m")
    assert isinstance(off, list) and len(off) == 3
    # Non-trivial fix point (metres in cube-local frame)
    assert abs(off[0]) + abs(off[2]) > 1.0


def test_excluded_sites(scene):
    by_id = {s["site_id"]: s for s in scene["stations"]}
    assert by_id["BS1_901_n_capitol"]["in_grid"] is False
    assert by_id["BS3_400_ny_ave"]["status"] == "na_bankrupt"


def test_fdtd_foundation_budget(scene):
    bud = scene["fdtd_foundation"]
    assert bud["indoor_scale_0p3m"]["n_cells"] > 1e9
    lam = bud["lambda_over_8"]
    assert lam[0]["n_cells"] > bud["indoor_scale_0p3m"]["n_cells"]
    assert "Gcell" in lam[0]["n_cells_label"] or "Tcell" in lam[0]["n_cells_label"]


def test_buildings_polyhedra():
    b = json.loads((WEB / "buildings.json").read_text())
    assert b["n"] >= 100
    sample = b["buildings"][0]
    assert len(sample["x"]) >= 3 and sample["h"] > 0
