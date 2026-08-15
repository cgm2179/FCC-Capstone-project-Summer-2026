#!/usr/bin/env python3
"""
fetch_terrain_opentopo.py — fetch a DEM for a scene bbox from the OpenTopography API and save it
as the terrain heightmap that `voxelize_city.py --terrain` consumes (real outdoor ground level).

NEEDS AN OPENTOPOGRAPHY API KEY (free): register at https://portal.opentopography.org/ and pass
it via --key or the OPENTOPOGRAPHY_API_KEY env var. (The public API returns HTTP 401 without one.)
Global DEMs: SRTMGL1 (30 m), SRTMGL3 (90 m), COP30 (Copernicus 30 m), NASADEM, AW3D30, …

usage:
  export OPENTOPOGRAPHY_API_KEY=...          # or pass --key
  python fetch_terrain_opentopo.py --bbox 38.900 38.912 -77.016 -77.004 --out city/NoMa_dem.npy
  python voxelize_city.py --terrain city/NoMa_dem.npy      # buildings on real terrain
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # SIM V1 3D root (siblings live there)
import argparse
import os
from pathlib import Path

import numpy as np

API = "https://portal.opentopography.org/API/globaldem"


def parse_aaigrid(text):
    """Parse an ESRI ASCII Grid (AAIGrid) into a 2-D float array (metres); NODATA -> NaN."""
    lines = text.splitlines()
    hdr, i = {}, 0
    keys = {"ncols", "nrows", "xllcorner", "yllcorner", "xllcenter", "yllcenter",
            "cellsize", "nodata_value"}
    while i < len(lines) and lines[i].split() and lines[i].split()[0].lower() in keys:
        k, v = lines[i].split()[:2]
        hdr[k.lower()] = float(v)
        i += 1
    nc, nr = int(hdr["ncols"]), int(hdr["nrows"])
    nod = hdr.get("nodata_value", -9999.0)
    vals = np.fromstring(" ".join(lines[i:]), sep=" ")[: nc * nr].reshape(nr, nc)
    return np.where(vals == nod, np.nan, vals)


def fetch_dem(south, north, west, east, demtype="SRTMGL1", key=None):
    """Return a 2-D DEM (metres) for the bbox from OpenTopography (AAIGrid output)."""
    key = key or os.environ.get("OPENTOPOGRAPHY_API_KEY")
    if not key:
        raise SystemExit("OpenTopography API key required: --key or OPENTOPOGRAPHY_API_KEY "
                         "(free at https://portal.opentopography.org/).")
    import urllib.parse
    import urllib.request
    q = urllib.parse.urlencode(dict(demtype=demtype, south=south, north=north, west=west,
                                    east=east, outputFormat="AAIGrid", API_Key=key))
    with urllib.request.urlopen(API + "?" + q, timeout=120) as r:
        if getattr(r, "status", 200) != 200:
            raise SystemExit(f"OpenTopography returned {r.status} — check the key / bbox / quota.")
        text = r.read().decode("utf-8", "replace")
    return parse_aaigrid(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", type=float, nargs=4, required=True,
                    metavar=("SOUTH", "NORTH", "WEST", "EAST"),
                    help="lat/lon bounding box: south north west east (degrees)")
    ap.add_argument("--demtype", default="SRTMGL1",
                    help="OpenTopography DEM id (default SRTMGL1 = 30 m)")
    ap.add_argument("--key", default=None,
                    help="OpenTopography API key (or the OPENTOPOGRAPHY_API_KEY env var)")
    ap.add_argument("--out", required=True, help="output .npy heightmap (metres)")
    a = ap.parse_args()
    s, n, w, e = a.bbox
    dem = fetch_dem(s, n, w, e, a.demtype, a.key)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, dem.astype(np.float32))
    print(f"saved {dem.shape} DEM ({a.demtype}) · elev {np.nanmin(dem):.0f}..{np.nanmax(dem):.0f} m"
          f" -> {out}\nnext: python voxelize_city.py --terrain {out}")


if __name__ == "__main__":
    main()
