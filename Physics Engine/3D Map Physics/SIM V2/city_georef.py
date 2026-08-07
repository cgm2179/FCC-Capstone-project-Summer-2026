#!/usr/bin/env python3
"""
city_georef.py — georeference the NoMa city scene by the FCC HQ anchor.

Moved into SIM V2 from `2D/SIM V3/city_georef.py` (adapted to the SIM V2 `bootstrap`
shim). The NoMa buildings OBJ was exported with FCC HQ (45 L St NE) at local origin
(0,0); the voxelized `manifest_3d.json` was missing `epsg`/`merc_anchor`, so
`voxelize_city.lonlat_to_vox` failed. This computes `merc_anchor = EPSG:3857(FCC HQ)`
and injects it, making lat/lon ↔ voxel exact. Verified vs base_stations.csv < 1 m.

  python city_georef.py --patch     # write epsg+merc_anchor into the city manifest (.bak kept)
  python city_georef.py --verify     # place the real base stations, check distances
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import bootstrap as B

sys.path.insert(0, str(B.SIM3D))
import voxelize_city as VC  # noqa: E402

FCC_LONLAT = (-77.0074055, 38.90358466666667)   # 45 L St NE — FCC HQ / O2I origin
CITY_MANIFEST = B.CITY_DIR / "manifest_3d.json"
CATALOG = B.STEP2.parent.parent / "Cross_Validation" / "base_station_catalog" / "base_stations.csv"


def georef_manifest(man: dict) -> dict:
    """Return a copy of `man` with the FCC-HQ EPSG:3857 anchor injected."""
    ax, az = VC.lonlat_to_merc(*FCC_LONLAT)
    out = dict(man)
    out["epsg"] = 3857
    out["merc_anchor"] = [ax, az]
    out["georef"] = {
        "anchor": "FCC_HQ_45_L_St_NE", "anchor_lonlat": list(FCC_LONLAT),
        "method": "Blender local origin = FCC HQ; merc_anchor = EPSG:3857(FCC). "
                  "Verified vs base_stations.csv distance_m to < 1 m.",
    }
    return out


def load_georef_city(city_dir=None):
    """City material grid + a georeferenced manifest (non-invasive, in-memory)."""
    cdir = Path(city_dir) if city_dir else B.CITY_DIR
    man = georef_manifest(json.loads((cdir / "manifest_3d.json").read_text()))
    grid = np.load(B.require(cdir / "material_grid.npy"))
    tvp = cdir / "terrain_v.npy"
    if tvp.exists():
        man["_terrain_v"] = np.load(tvp)
    return grid, man


def _live_sites():
    sites = {}
    for r in csv.DictReader(CATALOG.open()):
        if r["status"] != "live" or not r["lat"]:
            continue
        sites.setdefault(r["site_id"], (float(r["lon"]), float(r["lat"]),
                                        float(r["distance_m"]), r["name"]))
    return sites


def verify():
    grid, man = load_georef_city()
    cs = float(man["cell_size_m"]); NX, NY, NZ = grid.shape
    fx, fz = VC.lonlat_to_vox(man, *FCC_LONLAT)
    print(f"FCC HQ -> vox ({fx:.0f},{fz:.0f})  grid {NX}x{NZ}")
    ok = True
    for sid, (lon, lat, dcat, name) in _live_sites().items():
        vx, vz = VC.lonlat_to_vox(man, lon, lat)
        d = np.hypot(vx - fx, vz - fz) * cs
        inb = 0 <= vx < NX and 0 <= vz < NZ
        err = d - dcat
        ok &= inb and abs(err) < 5.0
        print(f"{sid:16s} ({vx:6.0f},{vz:6.0f}) {str(inb)[0]}  {d:7.1f} {dcat:7.1f} {err:+6.1f}")
    print("VERIFY:", "OK (<5 m on all)" if ok else "FAILED")
    return ok


def patch():
    man = json.loads(CITY_MANIFEST.read_text())
    if man.get("merc_anchor"):
        print("already georeferenced (merc_anchor present)"); return
    shutil.copy(CITY_MANIFEST, str(CITY_MANIFEST) + ".bak")
    CITY_MANIFEST.write_text(json.dumps(georef_manifest(man), indent=2))
    print(f"patched {CITY_MANIFEST}  (.bak kept)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.patch:
        patch()
    if a.verify or not a.patch:
        verify()
