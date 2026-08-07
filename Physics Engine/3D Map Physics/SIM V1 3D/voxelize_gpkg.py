#!/usr/bin/env python3
"""
voxelize_gpkg.py — real OSM/VIDA **building GeoPackage** → georeferenced city grid.

Sibling of voxelize_city.py, but the buildings come from a lon/lat **.gpkg**
(millions of real footprints with per-building height) instead of a prepared OBJ.
It reads only the footprints inside the scene bbox (pyogrio pushes the spatial
filter to GDAL, so the 13.9 M-feature / 4 GB file never loads whole), projects
each polygon to the SAME EPSG:3857 scaled-local frame the rest of SIM V3 uses,
rasterizes footprints → **barrier class 3** with a per-building **height** column
fill (2.5-D), and emits a grid `city_georef.load_georef_city()` / `fw_bs_catalog`
consume UNCHANGED.

Default frame = a **drop-in overlay** of the existing NoMa OBJ grid
(`--match <city dir>`): same `origin_units` X/Z, `cell_size_m`, `merc_scale`,
`nx`, `nz`, and the FCC-HQ anchor — so every base station lands at the identical
voxel it does today, only now on real OSM buildings with real heights. `--bbox-
lonlat` builds a fresh frame around the FCC anchor instead.

Height: the VIDA `height` field is a string like `"HHT:4.29"` (metres); parsed by
taking the trailing number, with the batch median filling the ~5 % that lack one.

Reuses voxelize_city: `lonlat_to_merc` / `vox_to_lonlat` (projection), `make_masks`
(exterior-air + rooftop-Tx masks), `city_manifest` (physics/materials/freqs copied
from the indoor manifest), and the 2.5-D column-fill logic.

  # drop-in overlay of the current NoMa grid, on real buildings (run with a
  # geopandas env — e.g. a venv that has geopandas+pyogrio+shapely):
  python voxelize_gpkg.py --gpkg ~/Downloads/building.032010.gpkg
  # fresh 1.2 km tile around the FCC anchor at 1 m:
  python voxelize_gpkg.py --bbox-lonlat -77.020 38.898 -76.995 38.910 --cell 1
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import voxelize_city as VC  # noqa: E402  (projection + masks + manifest, reused)

FCC_LONLAT = (-77.0074055, 38.90358466666667)   # 45 L St NE — the SIM V3 anchor origin
BUILDING_CLASS = 3
DEFAULT_GPKG = Path.home() / "Downloads" / "building.032010.gpkg"
DEFAULT_MATCH = HERE / "city" / "NoMa_DC_buildings"   # overlay this grid by default


def parse_height(val, default, floor_h=3.0):
    """VIDA `height` field → metres. The field encodes several formats:
      HHT:x         exact height in metres (DEM/lidar) — always preferred
      H:n           storey count → n·floor_h metres
      HBET:a-b      storey RANGE (e.g. 'HBET:1-5' generic low-rise) → midpoint·floor_h
      H:n+HHT:x     both present → the exact HHT wins
    Missing / unparseable → `default` (the batch median)."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    s = str(val)
    if not s or s == "nan":
        return default
    m = re.search(r"HHT:\s*([-+]?\d*\.?\d+)", s)          # exact metres wins
    if m:
        return float(m.group(1))
    m = re.search(r"HBET:\s*(\d+)\s*-\s*(\d+)", s)         # storey range → midpoint
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2.0 * floor_h
    m = re.search(r"\bH:\s*(\d+)", s)                      # storey count
    if m:
        return int(m.group(1)) * floor_h
    return default


def lonlat_to_vox_xz(lon, lat, merc_scale, anchor, ox, oz, cell):
    """Vectorized lon/lat (arrays) → (vx, vz) voxel coords in the scaled-local frame.
    Same math as voxelize_city.lonlat_to_vox, but array-friendly for polygon rings."""
    mx = -VC.R_MERC * np.radians(lon)                       # X = -easting (render runs west)
    mz = VC.R_MERC * np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))
    vx = (merc_scale * (mx - anchor[0]) - ox) / cell
    vz = (merc_scale * (mz - anchor[1]) - oz) / cell
    return vx, vz


def rasterize_heightmap(gdf, heights, nx, nz, merc_scale, anchor, ox, oz, cell):
    """Per-(x,z) roof height (voxel-Y) from real footprints. Each polygon is
    projected to voxel space and its interior cells set to max(current, height)."""
    import shapely
    hm = np.full((nx, nz), -1.0)
    geoms = gdf.geometry.values
    for geom, h in zip(geoms, heights):
        if geom is None or geom.is_empty:
            continue
        hy = h / cell
        parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in parts:
            ring = np.asarray(poly.exterior.coords)          # (n, 2) lon, lat
            vx, vz = lonlat_to_vox_xz(ring[:, 0], ring[:, 1],
                                      merc_scale, anchor, ox, oz, cell)
            x0, x1 = int(np.floor(vx.min())), int(np.ceil(vx.max()))
            z0, z1 = int(np.floor(vz.min())), int(np.ceil(vz.max()))
            x0, x1 = max(0, x0), min(nx, x1)
            z0, z1 = max(0, z0), min(nz, z1)
            if x0 >= x1 or z0 >= z1:
                continue
            polyv = shapely.Polygon(np.column_stack([vx, vz]))
            gx, gz = np.meshgrid(np.arange(x0, x1) + 0.5,
                                 np.arange(z0, z1) + 0.5, indexing="ij")
            inside = shapely.contains_xy(polyv, gx.ravel(), gz.ravel()).reshape(gx.shape)
            sub = hm[x0:x1, z0:z1]
            np.maximum(sub, np.where(inside, hy, -1.0), out=sub)
    return hm


def columns_to_grid(hm, ny, model="penetrable", core=2, facade=5, barrier=3,
                    terrain_v=None):
    """2.5-D column fill (voxelize_city.build_25d logic): footprints solid
    ground→roof; ring-only footprints and courtyards filled so nothing leaks.

    model='penetrable' paints a 1-cell GLASS `facade` skin + CONCRETE `core`
    interior — both lossy dielectrics, so the wave penetrates/attenuates (real
    O2I) and reflects off facades, instead of the old rigid `barrier` (perfect
    reflector). model='barrier' keeps the single rigid class (old behaviour).
    terrain_v (nx,nz) voxel ground heights: columns extrude from the terrain
    surface instead of a flat y=0."""
    foot = hm >= 0.0
    filled = ndimage.binary_fill_holes(foot)
    need = filled & ~foot
    if need.any():
        lbl, nlab = ndimage.label(filled)
        cmax = ndimage.maximum(np.where(foot, hm, -1.0), lbl, np.arange(1, nlab + 1))
        hm = hm.copy()
        hm[need] = cmax[lbl[need] - 1]
        foot = filled
    nx, nz = hm.shape
    gnd = (np.clip(terrain_v, 0, ny - 1).astype(np.int32) if terrain_v is not None
           else np.zeros((nx, nz), np.int32))
    top = np.clip(gnd + np.floor(np.clip(hm, 0, ny - 1)).astype(np.int32), 0, ny - 1)
    M = np.zeros((nx, ny, nz), np.int8)
    if model == "barrier":
        for iy in range(ny):
            M[:, iy, :][foot & (top >= iy) & (gnd <= iy)] = barrier
        return M
    interior = ndimage.binary_erosion(foot, iterations=1)     # concrete core
    skin = foot & ~interior                                   # 1-cell glass facade
    for iy in range(ny):
        col = (top >= iy) & (gnd <= iy)
        M[:, iy, :][interior & col] = core
        M[:, iy, :][skin & col] = facade
    return M


def foliage_to_grid(M, hm_fol, ny, foliage=4, terrain_v=None):
    """Overlay foliage/tree columns (class `foliage`, a dilute lossy medium) onto
    an existing grid wherever there's no building. hm_fol = per-(x,z) canopy top
    (voxel height, -1 = none)."""
    foot = hm_fol >= 0.0
    if not foot.any():
        return M
    nx, nz = hm_fol.shape
    gnd = (np.clip(terrain_v, 0, ny - 1).astype(np.int32) if terrain_v is not None
           else np.zeros((nx, nz), np.int32))
    top = np.clip(gnd + np.floor(np.clip(hm_fol, 0, ny - 1)).astype(np.int32), 0, ny - 1)
    for iy in range(ny):
        col = foot & (top >= iy) & (gnd <= iy)
        sl = M[:, iy, :]
        sl[col & (sl == 0)] = foliage                        # only fill air (buildings win)
    return M


def frame_from_match(match_dir):
    """Read origin_units X/Z, cell, merc_scale, nx, nz from an existing city grid
    so the new grid overlays it exactly (drop-in for load_georef_city)."""
    man = json.loads((Path(match_dir) / "manifest_3d.json").read_text())
    nx, _, nz = man["grid_shape"]
    return dict(cell=float(man["cell_size_m"]), merc_scale=float(man["merc_scale"]),
                merc_lat=float(man.get("merc_lat", 38.9)),
                ox=float(man["origin_units"][0]), oz=float(man["origin_units"][2]),
                nx=int(nx), nz=int(nz), matched=str(match_dir))


def frame_from_bbox(lon0, lat0, lon1, lat1, cell, merc_lat, anchor, pad_m):
    """Build a fresh scaled-local frame covering a lon/lat bbox (+pad) around the anchor."""
    merc_scale = math.cos(math.radians(merc_lat))
    xs, zs = [], []
    for lo in (lon0, lon1):
        for la in (lat0, lat1):
            mx, mz = VC.lonlat_to_merc(lo, la)
            xs.append(merc_scale * (mx - anchor[0]))
            zs.append(merc_scale * (mz - anchor[1]))
    ox, oz = min(xs) - pad_m, min(zs) - pad_m
    nx = int(math.ceil((max(xs) - min(xs) + 2 * pad_m) / cell)) + 1
    nz = int(math.ceil((max(zs) - min(zs) + 2 * pad_m) / cell)) + 1
    return dict(cell=cell, merc_scale=merc_scale, merc_lat=merc_lat,
                ox=ox, oz=oz, nx=nx, nz=nz, matched=None)


def bbox_lonlat_of_frame(fr, anchor):
    """lon/lat AABB of the frame (for the pyogrio read filter + the manifest)."""
    corners = []
    for vx in (0, fr["nx"]):
        for vz in (0, fr["nz"]):
            mx = (vx * fr["cell"] + fr["ox"]) / fr["merc_scale"] + anchor[0]
            mz = (vz * fr["cell"] + fr["oz"]) / fr["merc_scale"] + anchor[1]
            corners.append(VC.merc_to_lonlat(mx, mz))
    lons = [c[0] for c in corners]; lats = [c[1] for c in corners]
    return min(lons), min(lats), max(lons), max(lats)


def fetch_foliage(read_bbox, fr, anchor, foliage_height, cell):
    """Per-(x,z) canopy-top heightmap from OSM vegetation (Overpass): park /
    wood / forest polygons at `foliage_height`, plus individual tree nodes.
    Returns None if nothing found / the query fails."""
    import ssl, json as _json, urllib.request, urllib.parse
    import shapely
    lon0, lat0, lon1, lat1 = read_bbox
    q = (f'[out:json][timeout:90];('
         f'way["leisure"="park"]({lat0},{lon0},{lat1},{lon1});'
         f'way["natural"="wood"]({lat0},{lon0},{lat1},{lon1});'
         f'way["landuse"="forest"]({lat0},{lon0},{lat1},{lon1});'
         f'node["natural"="tree"]({lat0},{lon0},{lat1},{lon1}););out geom;')
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request("https://overpass-api.de/api/interpreter",
                                 data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": "rf-sim-voxelizer/1.0"})
    with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
        els = _json.load(r)["elements"]
    nx, nz, ms, ox, oz = fr["nx"], fr["nz"], fr["merc_scale"], fr["ox"], fr["oz"]
    hm = np.full((nx, nz), -1.0); hy = foliage_height / cell
    npoly = ntree = 0
    for e in els:
        if e["type"] == "way" and e.get("geometry") and len(e["geometry"]) >= 3:
            lon = np.array([p["lon"] for p in e["geometry"]])
            lat = np.array([p["lat"] for p in e["geometry"]])
            vx, vz = lonlat_to_vox_xz(lon, lat, ms, anchor, ox, oz, cell)
            x0, x1 = max(0, int(vx.min())), min(nx, int(vx.max()) + 1)
            z0, z1 = max(0, int(vz.min())), min(nz, int(vz.max()) + 1)
            if x0 >= x1 or z0 >= z1:
                continue
            polyv = shapely.Polygon(np.column_stack([vx, vz]))
            gx, gz = np.meshgrid(np.arange(x0, x1) + .5, np.arange(z0, z1) + .5, indexing="ij")
            ins = shapely.contains_xy(polyv, gx.ravel(), gz.ravel()).reshape(gx.shape)
            sub = hm[x0:x1, z0:z1]; np.maximum(sub, np.where(ins, hy, -1.0), out=sub); npoly += 1
        elif e["type"] == "node":
            vx, vz = lonlat_to_vox_xz(np.array([e["lon"]]), np.array([e["lat"]]),
                                      ms, anchor, ox, oz, cell)
            ix, iz = int(vx[0]), int(vz[0])
            if 0 <= ix < nx and 0 <= iz < nz:
                hm[max(0, ix - 1):ix + 2, max(0, iz - 1):iz + 2] = hy; ntree += 1
    print(f"  foliage: {npoly} veg polygons + {ntree} trees from OSM Overpass")
    return hm if (npoly or ntree) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpkg", default=str(DEFAULT_GPKG), help="building GeoPackage (lon/lat)")
    ap.add_argument("--layer", default="building")
    ap.add_argument("--match", default=str(DEFAULT_MATCH),
                    help="existing city dir to overlay exactly (default: NoMa OBJ grid). "
                         "Ignored if --bbox-lonlat is given.")
    ap.add_argument("--bbox-lonlat", type=float, nargs=4, default=None,
                    metavar=("LON0", "LAT0", "LON1", "LAT1"),
                    help="build a fresh frame for this bbox around the FCC anchor instead")
    ap.add_argument("--cell", type=float, default=1.0, help="voxel size m (fresh-frame mode)")
    ap.add_argument("--merc-lat", type=float, default=38.9, help="Mercator scale latitude")
    ap.add_argument("--pad-m", type=float, default=100.0, help="fresh-frame bbox padding (m)")
    ap.add_argument("--default-height", type=float, default=None,
                    help="metres for footprints lacking a height (default: batch median)")
    ap.add_argument("--floor-height", type=float, default=3.0,
                    help="metres per storey for H:/HBET: storey-count heights (default 3.0)")
    ap.add_argument("--building-class", type=int, default=BUILDING_CLASS)
    ap.add_argument("--material-model", choices=["penetrable", "barrier"], default="penetrable",
                    help="penetrable = concrete core (2) + glass facade (5) lossy dielectrics "
                         "(real O2I); barrier = single rigid class 3 (old perfect reflector)")
    ap.add_argument("--core-class", type=int, default=2, help="concrete core class (penetrable)")
    ap.add_argument("--facade-class", type=int, default=5, help="glass facade class (penetrable)")
    ap.add_argument("--foliage-class", type=int, default=4, help="foliage/tree class")
    ap.add_argument("--foliage", action="store_true",
                    help="fetch OSM vegetation (parks/wood/trees, Overpass) and add as foliage")
    ap.add_argument("--foliage-height", type=float, default=8.0, help="canopy height m")
    ap.add_argument("--terrain", default=None, help="DEM .npy (metres) — ground follows terrain")
    ap.add_argument("--ground", action="store_true",
                    help="lay a 1-cell ground sheet (concrete/asphalt = core class) at the surface")
    ap.add_argument("--out", default=None, help="output dir (default city/NoMa_DC_osm)")
    ap.add_argument("--tx-band", type=float, nargs=2, default=(4.0, 12.0))
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    import geopandas as gpd
    anchor = list(VC.lonlat_to_merc(*FCC_LONLAT))
    cls = int(args.building_class)

    if args.bbox_lonlat is not None:
        fr = frame_from_bbox(*args.bbox_lonlat, args.cell, args.merc_lat, anchor, args.pad_m)
        print(f"fresh frame · bbox {args.bbox_lonlat} +{args.pad_m:.0f} m")
    else:
        fr = frame_from_match(args.match)
        print(f"drop-in overlay of {fr['matched']}")
    cell, merc_scale, ox, oz, nx, nz = (fr["cell"], fr["merc_scale"],
                                        fr["ox"], fr["oz"], fr["nx"], fr["nz"])
    read_bbox = bbox_lonlat_of_frame(fr, anchor)
    print(f"  frame  cell={cell} m  merc_scale={merc_scale:.6f}  grid {nx}x{nz}  "
          f"origin_units_xz=({ox:.2f},{oz:.2f})")
    print(f"  read bbox lon/lat {tuple(round(x, 5) for x in read_bbox)}")

    t0 = time.time()
    gdf = gpd.read_file(args.gpkg, layer=args.layer, bbox=read_bbox, engine="pyogrio")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    print(f"  read {len(gdf):,} footprints in {time.time()-t0:.1f}s")

    raw = [parse_height(v, np.nan, args.floor_height) for v in gdf["height"].tolist()] \
        if "height" in gdf.columns else [np.nan] * len(gdf)
    raw = np.asarray(raw, float)
    valid = np.isfinite(raw)
    dflt = args.default_height if args.default_height is not None else \
        (float(np.nanmedian(raw[valid])) if valid.any() else 8.0)
    heights = np.where(valid, raw, dflt)
    print(f"  height: {100*valid.mean():.1f}% real · median {np.nanmedian(raw[valid]):.1f} m · "
          f"default {dflt:.1f} m · max {heights.max():.1f} m")

    # optional real terrain (DEM): ground follows it, buildings extrude from it
    terrain_v = None
    if args.terrain:
        terrain_v = VC.load_terrain(Path(args.terrain), nx, nz, cell)
        relief = int(terrain_v.max())
        print(f"  terrain: DEM applied · +{relief} voxels ({relief*cell:.0f} m relief)")
    ny = int(math.ceil(heights.max() / cell)) + (int(terrain_v.max()) if terrain_v is not None else 0) + 2

    t0 = time.time()
    hm = rasterize_heightmap(gdf, heights, nx, nz, merc_scale, anchor, ox, oz, cell)
    M = columns_to_grid(hm, ny, model=args.material_model, core=args.core_class,
                        facade=args.facade_class, barrier=cls, terrain_v=terrain_v)

    # optional foliage/trees from OSM (Overpass): parks/wood/forest polygons + tree nodes
    n_fol = 0
    if args.foliage:
        try:
            hm_fol = fetch_foliage(read_bbox, fr, anchor, args.foliage_height, cell)
            if hm_fol is not None:
                M = foliage_to_grid(M, hm_fol, ny, foliage=args.foliage_class, terrain_v=terrain_v)
                n_fol = int((M == args.foliage_class).any(0).sum())
        except Exception as e:
            print(f"  (foliage skipped: {e})")

    # optional ground sheet (concrete/asphalt) at the surface — below the 2-D slice,
    # matters for a 3-D solve / ground reflection, not the current 2-D surrogate.
    if args.ground:
        gnd = (np.clip(terrain_v, 0, ny - 1) if terrain_v is not None
               else np.zeros((nx, nz), int))
        for iy in range(ny):
            sl = M[:, iy, :]; sl[(gnd == iy) & (sl == 0)] = args.core_class

    inside, valid_tx = VC.make_masks(M, cls, tuple(args.tx_band), cell)
    print(f"  {args.material_model} fill + masks in {time.time()-t0:.1f}s · grid {nx}x{ny}x{nz}")

    out = Path(args.out) if args.out else HERE / "city" / "NoMa_DC_osm"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "material_grid.npy", M)
    np.save(out / "inside_mask.npy", inside)
    np.save(out / "valid_tx_mask.npy", valid_tx)
    if terrain_v is not None:                                 # per-(x,z) ground voxel height
        np.save(out / "terrain_v.npy", terrain_v.astype(np.int16))

    classes_present = sorted(int(c) for c in np.unique(M))
    template = json.loads((HERE / "manifest_3d.json").read_text())
    man = VC.city_manifest(
        template, source_gpkg=str(args.gpkg), source_layer=args.layer,
        domain="city_outdoor", mode="2.5d-gpkg",
        material_model=args.material_model, classes_present=classes_present,
        core_class=args.core_class, facade_class=args.facade_class,
        foliage_class=(args.foliage_class if args.foliage else None),
        building_class=cls, cell_size_m=round(cell, 6), m_per_unit=1.0, merc_lat=fr["merc_lat"],
        merc_scale=round(merc_scale, 6), epsg=3857,
        merc_anchor=[round(anchor[0], 3), round(anchor[1], 3)],
        anchor_lonlat=list(FCC_LONLAT),
        origin_units=[round(ox, 4), 0.0, round(oz, 4)],
        grid_shape=[nx, ny, nz],
        ceiling_height_m=round(float(heights.max()), 3),
        room_band=[1, ny - 1], terrain=bool(args.terrain),
        bbox_lonlat=[round(x, 7) for x in read_bbox],
        overlay_of=fr["matched"], n_buildings=int(len(gdf)),
    )
    (out / "manifest_3d.json").write_text(json.dumps(man, indent=2))

    nvox = nx * ny * nz
    print(f"  classes present {classes_present} · solid {100*(M>0).sum()/nvox:.1f}% · "
          f"foliage cols {n_fol:,} · inside(air) {inside.sum():,} · valid_tx {int(valid_tx.sum()):,}")
    print(f"saved material_grid.npy, inside_mask.npy, valid_tx_mask.npy, "
          f"manifest_3d.json → {out}")

    if not args.no_preview:
        try:
            _preview(M, cls, cell, out)
        except Exception as e:
            print(f"  (preview skipped: {e})")


def _preview(M, cls, cell, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nx, ny, nz = M.shape
    solid = M == cls
    top = np.where(solid, np.arange(ny)[None, :, None], -1).max(1) * cell
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(np.where(top < 0, np.nan, top).T, origin="lower",
                   cmap="viridis", aspect="equal")
    ax.set_title(f"OSM roof heights · {out.name} · {cell} m · grid {nx}x{ny}x{nz}")
    ax.set_xlabel("X (vox)"); ax.set_ylabel("Z (vox)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="height (m)")
    fig.tight_layout()
    p = out / "preview_voxel.png"
    fig.savefig(p, dpi=120)
    print(f"  preview → {p}")


if __name__ == "__main__":
    main()
