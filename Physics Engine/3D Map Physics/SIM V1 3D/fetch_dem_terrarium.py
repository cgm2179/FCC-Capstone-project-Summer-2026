#!/usr/bin/env python3
"""
fetch_dem_terrarium.py — free, keyless DEM for a city grid from AWS Terrarium tiles.

Mapzen/AWS "terrarium" elevation tiles (s3://elevation-tiles-prod) encode ground
elevation in RGB: metres = R*256 + G + B/256 - 32768. This fetches the tiles that
cover a georeferenced city grid's bbox, decodes them, and samples a coarse DEM
ORIENTED TO THE GRID (grid[0,0] -> dem[0,0]) so voxelize_city.load_terrain / the
--terrain hook zoom it straight onto the (nx,nz) grid.

  python fetch_dem_terrarium.py --city-dir city/NoMa_DC_osm --out city/NoMa_DC_osm/dem_noma.npy

Needs network (matplotlib for PNG decode). Anchor = FCC HQ, same as city_georef.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import ssl
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE))
import voxelize_city as VC  # noqa: E402

FCC_LONLAT = (-77.0074055, 38.90358466666667)


def _tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    s = math.radians(lat)
    y = (1.0 - math.log(math.tan(s) + 1.0 / math.cos(s)) / math.pi) / 2.0 * n
    return x, y


def fetch_dem(city_dir, out, zoom=14, coarse=(140, 90)):
    import matplotlib.image as mpimg
    man = json.loads((Path(city_dir) / "manifest_3d.json").read_text())
    man["epsg"] = 3857
    man["merc_anchor"] = list(VC.lonlat_to_merc(*FCC_LONLAT))
    nx, ny, nz = man["grid_shape"]
    corners = [VC.vox_to_lonlat(man, vx, vz) for vx in (0, nx) for vz in (0, nz)]
    lons = [c[0] for c in corners]; lats = [c[1] for c in corners]
    x0f, y1f = _tile_xy(min(lons), min(lats), zoom)
    x1f, y0f = _tile_xy(max(lons), max(lats), zoom)
    tx0, tx1 = int(x0f), int(x1f); ty0, ty1 = int(y0f), int(y1f)
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    mos = {}
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            u = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{tx}/{ty}.png"
            req = urllib.request.Request(u, headers={"User-Agent": "rf-sim-dem/1.0"})
            img = mpimg.imread(io.BytesIO(urllib.request.urlopen(req, timeout=60, context=ctx).read()))
            a = (img[:, :, :3] * 255.0).round()
            mos[(tx, ty)] = a[:, :, 0] * 256 + a[:, :, 1] + a[:, :, 2] / 256 - 32768
    print(f"fetched {len(mos)} terrarium tiles z{zoom}")

    def elev(lon, lat):
        x, y = _tile_xy(lon, lat, zoom); tx, ty = int(x), int(y)
        E = mos.get((tx, ty))
        if E is None:
            return np.nan
        return float(E[min(int((y - ty) * 256), 255), min(int((x - tx) * 256), 255)])

    CX, CZ = coarse
    dem = np.zeros((CX, CZ))
    for i, vx in enumerate(np.linspace(0, nx, CX)):
        for j, vz in enumerate(np.linspace(0, nz, CZ)):
            lo, la = VC.vox_to_lonlat(man, vx, vz)
            dem[i, j] = elev(lo, la)
    dem = np.nan_to_num(dem, nan=float(np.nanmedian(dem)))
    np.save(out, dem)
    print(f"DEM {dem.shape} · min {dem.min():.1f} m · max {dem.max():.1f} m · "
          f"relief {dem.max()-dem.min():.1f} m -> {out}")
    return dem


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city-dir", required=True, help="grid dir with manifest_3d.json (frame)")
    ap.add_argument("--out", default=None, help="output DEM .npy (default <city-dir>/dem.npy)")
    ap.add_argument("--zoom", type=int, default=14)
    a = ap.parse_args()
    out = a.out or str(Path(a.city_dir) / "dem.npy")
    fetch_dem(a.city_dir, out, zoom=a.zoom)


if __name__ == "__main__":
    main()
