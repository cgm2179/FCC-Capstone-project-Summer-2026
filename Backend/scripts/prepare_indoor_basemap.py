#!/usr/bin/env python3
"""Build the indoor OSM basemap for the Map Coverage under-layer (PR #25).

The floor plan is rotated -7.33 deg vs north (Data/floorplan_meta.js), so a plain crop
would misalign. This fetches/stitches an OpenStreetMap raster covering the plan's lon/lat
footprint, then AFFINE-WARPS it into the exact 1150x515 plan pixel frame, so it drops
straight under `floorPlanImage` in Plotly (x in [0,1150], y in [0,515]).

Modes:
  (default)                              fetch + stitch OSM tiles          [needs network]
  --src IMG --src-bbox lonmin,latmin,lonmax,latmax   use a north-up raster [offline]
  --placeholder                          synthesize a labeled north-up map [offline, dev]

Output (base64-embedded, <script>-loadable like records_data.js):
  Data/basemap_image.js  ->  window.INDOOR_BASEMAP = {
      source, width:1150, height:515, bbox_lonlat, corners_lonlat, zoom, source_mode }

Reuses the georeference in Data/floorplan_meta.js; mirrors the crop/embed pattern of
FCC_Walk_Outdoor_Indoor_Full/Outdoor_Walk_Test_7-24/scripts/prepare_basemap.py.
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
REPO = Path(__file__).resolve().parent.parent
META_JS = REPO / "Data" / "floorplan_meta.js"
OUT_JS = REPO / "Data" / "basemap_image.js"
FORTE_HALL = (-77.011420, 38.901550)
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
UA = "indoor-walk-test-basemap/1.0 (RF coverage cross-validation; contact via repo)"
MARGIN_FRAC = 0.6      # extra ground around the plan footprint, as a fraction of its span
MAX_TILES = 64         # safety cap on tiles fetched


# --------------------------------------------------------------------------- geometry
def load_meta() -> dict:
    txt = META_JS.read_text(encoding="utf-8")
    return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])


def plan_corners_lonlat(meta):
    """4 floor-plan corners (px 0,0 / W,0 / W,H / 0,H) in lon/lat via affine_px_to_lonlat."""
    A = meta["affine_px_to_lonlat"]
    H, W = meta["grid_shape"]
    pts_px = [(0, 0), (W, 0), (W, H), (0, H)]
    ll = [(A[0][0] * px + A[0][1] * py + A[0][2],
           A[1][0] * px + A[1][1] * py + A[1][2]) for px, py in pts_px]
    return pts_px, ll, W, H


def padded_bbox(corners_ll, frac):
    lons = [c[0] for c in corners_ll]; lats = [c[1] for c in corners_ll]
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    dlon = (lon_max - lon_min) * frac; dlat = (lat_max - lat_min) * frac
    return lon_min - dlon, lat_min - dlat, lon_max + dlon, lat_max + dlat


def deg2num(lon, lat, z):
    lat_r = math.radians(lat)
    n = 2 ** z
    xt = (lon + 180.0) / 360.0 * n
    yt = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return xt, yt


def pick_zoom(bbox, target_px=1600, tile=256, zmax=19, zmin=14):
    lon_min, lat_min, lon_max, lat_max = bbox
    for z in range(zmax, zmin - 1, -1):
        x0, _ = deg2num(lon_min, lat_max, z)
        x1, _ = deg2num(lon_max, lat_min, z)
        if abs(x1 - x0) * tile <= target_px:
            return z
    return zmin


# --------------------------------------------------------------------------- raster sources
def fetch_osm_mosaic(bbox, z):
    """Stitch OSM tiles covering bbox at zoom z. Returns (img, src_px_fn)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    xa, ya = deg2num(lon_min, lat_max, z)   # top-left
    xb, yb = deg2num(lon_max, lat_min, z)   # bottom-right
    x0, y0 = int(math.floor(xa)), int(math.floor(ya))
    x1, y1 = int(math.floor(xb)), int(math.floor(yb))
    nx, ny = (x1 - x0 + 1), (y1 - y0 + 1)
    if nx * ny > MAX_TILES:
        raise SystemExit(f"{nx*ny} tiles > cap {MAX_TILES}; lower zoom or margin")
    mosaic = Image.new("RGB", (nx * 256, ny * 256), (235, 235, 235))
    for xi in range(x0, x1 + 1):
        for yi in range(y0, y1 + 1):
            url = TILE_URL.format(z=z, x=xi, y=yi)
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                tile = Image.open(io.BytesIO(r.read())).convert("RGB")
            mosaic.paste(tile, ((xi - x0) * 256, (yi - y0) * 256))
            time.sleep(0.05)   # be polite to the tile server

    def src_px(lon, lat):
        xt, yt = deg2num(lon, lat, z)
        return (xt - x0) * 256.0, (yt - y0) * 256.0
    return mosaic, src_px


def load_src_mosaic(src_path, src_bbox):
    """A user-supplied north-up raster covering src_bbox (lonmin,latmin,lonmax,latmax)."""
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    lon_min, lat_min, lon_max, lat_max = src_bbox

    def src_px(lon, lat):   # linear over bbox, north-up (y increases downward)
        u = (lon - lon_min) / (lon_max - lon_min)
        v = (lat_max - lat) / (lat_max - lat_min)
        return u * w, v * h
    return img, src_px


def placeholder_mosaic(bbox, size=1400):
    """Synthetic north-up map (offline dev): grid + Forte Hall marker + plan footprint."""
    lon_min, lat_min, lon_max, lat_max = bbox
    span = max(lon_max - lon_min, lat_max - lat_min)
    w = size
    h = max(1, int(size * (lat_max - lat_min) / (lon_max - lon_min)))
    img = Image.new("RGB", (w, h), (238, 236, 230))
    d = ImageDraw.Draw(img)

    def px(lon, lat):
        return ((lon - lon_min) / (lon_max - lon_min) * w,
                (lat_max - lat) / (lat_max - lat_min) * h)

    step = 10 ** math.floor(math.log10(span / 4))     # ~nice grid spacing in degrees
    lon = math.ceil(lon_min / step) * step
    while lon < lon_max:
        x, _ = px(lon, lat_max)
        d.line([(x, 0), (x, h)], fill=(210, 208, 200), width=1)
        lon += step
    lat = math.ceil(lat_min / step) * step
    while lat < lat_max:
        _, y = px(lon_min, lat)
        d.line([(0, y), (w, y)], fill=(210, 208, 200), width=1)
        lat += step
    # Forte Hall marker + label, and a north arrow
    fx, fy = px(*FORTE_HALL)
    d.ellipse([fx - 7, fy - 7, fx + 7, fy + 7], fill=(200, 40, 40))
    d.text((fx + 10, fy - 6), "Forte Hall (Tx)", fill=(150, 20, 20))
    d.line([(30, 60), (30, 20)], fill=(40, 40, 40), width=3)
    d.polygon([(30, 12), (24, 26), (36, 26)], fill=(40, 40, 40))
    d.text((22, 64), "N", fill=(40, 40, 40))
    d.text((10, h - 24), "PLACEHOLDER basemap (offline) — run without --placeholder for real OSM tiles",
           fill=(120, 60, 60))

    def src_px(lon, lat):
        return px(lon, lat)
    return img, src_px


# --------------------------------------------------------------------------- warp + emit
def warp_to_plan(mosaic, src_px, corners_px, corners_ll, W, H):
    """Affine-warp the north-up mosaic into the plan pixel frame (W x H).

    PIL AFFINE maps OUT(x,y) -> SRC(x,y) via (a,b,c,d,e,f):
        src_x = a*ox + b*oy + c ,  src_y = d*ox + e*oy + f
    Fit that from the 4 plan corners: OUT = plan px, SRC = mosaic px of the corner's lon/lat.
    (Composition of two affines — floor-plan affine and the locally-affine mercator map —
    so 4 corners determine it exactly to <0.01% over this ~200 m span.)
    """
    src_pts = [src_px(lon, lat) for (lon, lat) in corners_ll]
    O = np.array([[ox, oy, 1.0] for (ox, oy) in corners_px])
    sx = np.array([p[0] for p in src_pts]); sy = np.array([p[1] for p in src_pts])
    cx, *_ = np.linalg.lstsq(O, sx, rcond=None)
    cy, *_ = np.linalg.lstsq(O, sy, rcond=None)
    coeffs = (cx[0], cx[1], cx[2], cy[0], cy[1], cy[2])
    out = mosaic.transform((W, H), Image.AFFINE, coeffs,
                           resample=Image.BICUBIC, fillcolor=(235, 235, 235))
    return out


def emit_js(img, corners_ll, bbox, zoom, mode, fmt="JPEG"):
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, "JPEG", quality=85)
        mime = "image/jpeg"
    else:
        img.save(buf, "PNG")
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    payload = {
        "source": f"data:{mime};base64,{b64}",
        "width": img.width, "height": img.height,
        "bbox_lonlat": list(bbox),
        "corners_lonlat": [[round(lo, 7), round(la, 7)] for lo, la in corners_ll],
        "zoom": zoom, "source_mode": mode,
    }
    OUT_JS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JS.open("w", encoding="utf-8") as f:
        f.write("// Generated by scripts/prepare_indoor_basemap.py — OSM under-layer for Map Coverage.\n")
        f.write("// Warped into the floor-plan pixel frame (1150x515); drops under floorPlanImage.\n")
        f.write("window.INDOOR_BASEMAP = {\n")
        f.write(f'  source: "{payload["source"]}",\n')
        for k in ("width", "height", "zoom"):
            f.write(f"  {k}: {json.dumps(payload[k])},\n")   # None -> null (valid JS)
        f.write(f'  source_mode: "{mode}",\n')
        f.write(f"  bbox_lonlat: {json.dumps(payload['bbox_lonlat'])},\n")
        f.write(f"  corners_lonlat: {json.dumps(payload['corners_lonlat'])}\n")
        f.write("};\n")
    return OUT_JS.stat().st_size


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--placeholder", action="store_true",
                    help="synthesize an offline labeled map (dev / no network)")
    ap.add_argument("--src", type=Path, help="north-up raster to use instead of OSM tiles")
    ap.add_argument("--src-bbox", help="lonmin,latmin,lonmax,latmax for --src")
    ap.add_argument("--margin", type=float, default=MARGIN_FRAC,
                    help=f"ground margin around the plan (default {MARGIN_FRAC})")
    args = ap.parse_args()

    meta = load_meta()
    corners_px, corners_ll, W, H = plan_corners_lonlat(meta)
    bbox = padded_bbox(corners_ll, args.margin)
    print(f"plan {W}x{H}px; corners lon/lat:")
    for (px, py), (lo, la) in zip(corners_px, corners_ll):
        print(f"   px({px:>4.0f},{py:>3.0f}) -> ({lo:.6f}, {la:.6f})")
    print(f"bbox (+{args.margin:.0%} margin): {tuple(round(b,6) for b in bbox)}")

    if args.placeholder:
        mosaic, src_px = placeholder_mosaic(bbox); zoom, mode, fmt = None, "placeholder", "PNG"
    elif args.src:
        if not args.src_bbox:
            raise SystemExit("--src requires --src-bbox lonmin,latmin,lonmax,latmax")
        sb = tuple(float(x) for x in args.src_bbox.split(","))
        mosaic, src_px = load_src_mosaic(args.src, sb); zoom, mode, fmt = None, "src", "JPEG"
    else:
        zoom = pick_zoom(bbox)
        print(f"fetching OSM tiles at zoom {zoom} …")
        mosaic, src_px = fetch_osm_mosaic(bbox, zoom); mode, fmt = "osm", "JPEG"

    out = warp_to_plan(mosaic, src_px, corners_px, corners_ll, W, H)
    size = emit_js(out, corners_ll, bbox, zoom, mode, fmt=fmt)
    print(f"wrote {OUT_JS.relative_to(REPO)} ({size/1e3:.0f} KB, mode={mode}, {out.width}x{out.height})")


if __name__ == "__main__":
    main()
