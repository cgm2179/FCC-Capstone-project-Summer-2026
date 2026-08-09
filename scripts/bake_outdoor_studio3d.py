#!/usr/bin/env python3
"""Bake the Outdoor 3-D Full-Wave Studio scene (foundational FDTD setup).

Builds a georeferenced NoMa DC voxel city from the committed OSM/VIDA GeoPackage
(penetrable materials matching Outdoor 2-D):

  0 air / street
  2 concrete core     — heavy (O2I)
  4 foliage / trees   — light (dilute)
  5 glass facade      — light

Browser studio grid defaults to 2.0 m (same as Outdoor 2-D / city demo) over a
base-station cluster crop so the asset stays shippable. Manifest records the
true indoor-scale (0.3 m) and λ/8 FDTD cell budgets for the later full-wave
scene — those counts are intentionally massive.

Outputs (committed under SIM3D/web/outdoor_studio3d/):
  material_grid.bin   — uint8 XYZ contiguous (x major, then y, then z)
  scene.json          — manifest + base stations + FDTD budgets + georef
  buildings.json      — extruded OSM footprints (polyhedra for three.js)
  preview_xy.png      — verification top-down (optional)

Usage (from repo root, venv active):
  python scripts/bake_outdoor_studio3d.py
  python scripts/bake_outdoor_studio3d.py --cell 2 --pad-m 280 --no-foliage
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIM3D = ROOT / "Physics Engine" / "3D Map Physics" / "SIM V1 3D"
# Prefer the SIM3D symlink at repo root when present.
if (ROOT / "SIM3D" / "voxelize_gpkg.py").exists():
    SIM3D = ROOT / "SIM3D"
sys.path.insert(0, str(SIM3D))

import voxelize_city as VC  # noqa: E402
import voxelize_gpkg as VG  # noqa: E402

REGISTRY = ROOT / "Physics Engine" / "Cross_Validation" / "base_station_catalog" / "sites_registry.json"
GPKG = SIM3D / "city_src" / "NoMa_DC_buildings.gpkg"
OUT_DIR = SIM3D / "web" / "outdoor_studio3d"
CITY_CACHE = SIM3D / "city" / "NoMa_DC_outdoor_studio3d"

# Outdoor material legend (matches Data/noma_city2d.js / user request)
OUTDOOR_MATERIALS = [
    {"id": 0, "name": "air / street", "color": "#eef2ef", "loss": "—", "loss_db": 0.0},
    {"id": 2, "name": "concrete core", "color": "#555555", "loss": "heavy (O2I)", "loss_db": 20.0},
    {"id": 4, "name": "foliage / trees", "color": "#2e9e3f", "loss": "light (dilute)", "loss_db": 0.3, "loss_per_m_db": 0.3},
    {"id": 5, "name": "glass facade", "color": "#3a86ff", "loss": "light", "loss_db": 3.0},
]

# Surveyed / user-confirmed rooftop heights (AGL m) — override unreliable OSM placeholders.
HEIGHT_OVERRIDES = {
    "BS7_forte_hall": 20.0,       # manual; OSM hits wrong adjacent building
    "BS5_1140_n_capitol": 33.0,   # osm-w67093724, 11 levels
    "BS6_1005_n_capitol": 50.7,   # osm-w535751723, 14 levels
    "BS3_400_ny_ave": 50.7,       # Meridian Phase 2 (excluded, informational)
}

C0 = 299792458.0
INDOOR_CELL_M = 0.3


def load_sites(path: Path):
    reg = json.loads(path.read_text())
    return reg["sites"]


def station_bbox(sites, pad_deg=0.0):
    lons, lats = [], []
    for s in sites:
        if s.get("lat") is None or s.get("lon") is None:
            continue
        # include excluded-but-located sites so the crop covers the full surveyed set
        lons.append(float(s["lon"]))
        lats.append(float(s["lat"]))
    return (min(lons) - pad_deg, min(lats) - pad_deg,
            max(lons) + pad_deg, max(lats) + pad_deg)


def fdtd_budgets(extent_m, height_m, freqs_mhz=(617.0, 2412.0, 3700.0)):
    """Massive full-wave cell counts for the later FDTD scene (foundational metadata)."""
    Lx, Lz = extent_m
    Ly = height_m
    indoor = {
        "cell_m": INDOOR_CELL_M,
        "grid": [int(math.ceil(Lx / INDOOR_CELL_M)),
                 int(math.ceil(Ly / INDOOR_CELL_M)),
                 int(math.ceil(Lz / INDOOR_CELL_M))],
    }
    indoor["n_cells"] = int(np.prod(indoor["grid"]))
    indoor["note"] = "Same cell size as Indoor 3-D Full-Wave Studio — not the browser asset."
    bands = []
    for f in freqs_mhz:
        h = (C0 / (f * 1e6)) / 8.0
        g = [int(math.ceil(Lx / h)), int(math.ceil(Ly / h)), int(math.ceil(Lz / h))]
        n = int(np.prod(g))
        bands.append({
            "f_mhz": f, "cell_m": round(h, 6), "npw": 8,
            "grid": g, "n_cells": n,
            "n_cells_label": _fmt_cells(n),
        })
    return {"indoor_scale_0p3m": indoor, "lambda_over_8": bands,
            "domain_m": [round(Lx, 1), round(Ly, 1), round(Lz, 1)]}


def _fmt_cells(n: int) -> str:
    if n >= 1e15:
        return f"{n / 1e15:.2f} Pcell"
    if n >= 1e12:
        return f"{n / 1e12:.2f} Tcell"
    if n >= 1e9:
        return f"{n / 1e9:.2f} Gcell"
    if n >= 1e6:
        return f"{n / 1e6:.2f} Mcell"
    return f"{n:,} cells"


def _nearest_solid_column(M, ix, iz, max_r=40):
    """Snap a lon/lat pin that misses a footprint to the nearest solid column."""
    nx, ny, nz = M.shape
    if 0 <= ix < nx and 0 <= iz < nz and (M[ix, :, iz] > 0).any():
        return ix, iz, 0
    for r in range(1, max_r + 1):
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if abs(dx) != r and abs(dz) != r:
                    continue
                x, z = ix + dx, iz + dz
                if 0 <= x < nx and 0 <= z < nz and (M[x, :, z] > 0).any():
                    return x, z, r
    return ix, iz, -1


def place_stations(sites, man, M):
    """Lon/lat → voxel; snap Tx to rooftop air just above the building column."""
    nx, ny, nz = M.shape
    cell = float(man["cell_size_m"])
    out = []
    for s in sites:
        if s.get("lat") is None or s.get("lon") is None:
            out.append({**_site_public(s), "in_grid": False, "reason": "no_coords"})
            continue
        vx, vz = VC.lonlat_to_vox(man, float(s["lon"]), float(s["lat"]))
        ix, iz = int(round(vx)), int(round(vz))
        if not (0 <= ix < nx and 0 <= iz < nz):
            out.append({**_site_public(s), "in_grid": False, "reason": "outside_bbox",
                        "vx": ix, "vz": iz})
            continue
        ix, iz, snap_r = _nearest_solid_column(M, ix, iz)
        col = M[ix, :, iz]
        solid = np.where(col > 0)[0]
        if solid.size:
            roof_y = int(solid.max())
            roof_m = (roof_y + 1) * cell
            src = "osm_column" if snap_r == 0 else f"osm_column_snap_{snap_r}"
        else:
            roof_y = 0
            roof_m = 0.0
            src = "ground"
        h_ov = HEIGHT_OVERRIDES.get(s["site_id"])
        if h_ov is not None:
            roof_m = float(h_ov)
            roof_y = min(ny - 2, max(0, int(round(h_ov / cell)) - 1))
            src = "survey_override"
        ty = min(ny - 1, roof_y + 1)
        if col[ty] != 0:
            for yy in range(ty, ny):
                if col[yy] == 0:
                    ty = yy
                    break
        out.append({
            **_site_public(s),
            "in_grid": True,
            "vx": ix, "vy": ty, "vz": iz,
            "x_m": round(ix * cell, 2),
            "y_m": round(ty * cell, 2),
            "z_m": round(iz * cell, 2),
            "roof_height_m": round(roof_m, 2),
            "height_source": src,
            "snap_cells": snap_r,
        })
    return out


def _site_public(s):
    carriers = []
    for c in s.get("carriers") or []:
        carriers.append({
            "carrier": c.get("carrier"),
            "pcis": c.get("pcis") or [],
            "bands": c.get("bands") or [],
        })
    return {
        "site_id": s["site_id"],
        "name": s.get("name"),
        "address": s.get("address"),
        "lat": s.get("lat"),
        "lon": s.get("lon"),
        "status": s.get("status"),
        "role": s.get("role"),
        "carriers": carriers,
    }


def export_buildings_json(gdf, heights, fr, anchor, cell, out_path: Path, max_buildings=2500):
    """Extruded building polyhedra (footprint rings + height) for three.js Buildings view."""
    import shapely
    ms, ox, oz = fr["merc_scale"], fr["ox"], fr["oz"]
    buildings = []
    # Prefer taller / larger buildings first if we have to cap.
    areas = []
    for geom in gdf.geometry.values:
        try:
            areas.append(float(geom.area) if geom is not None else 0.0)
        except Exception:
            areas.append(0.0)
    order = np.argsort(-np.asarray(areas))
    for k in order[:max_buildings]:
        geom = gdf.geometry.values[k]
        h = float(heights[k])
        if geom is None or geom.is_empty or h <= 0:
            continue
        parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in parts:
            ring = np.asarray(poly.exterior.coords)
            if len(ring) < 3:
                continue
            vx, vz = VG.lonlat_to_vox_xz(ring[:, 0], ring[:, 1], ms, anchor, ox, oz, cell)
            # Store in metres in the scaled-local frame (grid origin at 0,0)
            xs = (vx * cell).round(2).tolist()
            zs = (vz * cell).round(2).tolist()
            buildings.append({"h": round(h, 2), "x": xs, "z": zs})
    out_path.write_text(json.dumps({"cell_m": cell, "n": len(buildings), "buildings": buildings}))
    return len(buildings)


def preview(M, cell, stations, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nx, ny, nz = M.shape
    # roof height where any solid
    solid = M > 0
    top = np.where(solid, np.arange(ny)[None, :, None], -1).max(1) * cell
    # class map at mid height of each column
    cls = np.zeros((nx, nz), np.int8)
    for ix in range(nx):
        for iz in range(nz):
            col = M[ix, :, iz]
            s = np.where(col > 0)[0]
            if s.size:
                cls[ix, iz] = col[s[len(s) // 2]]
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.matplotlib.colors.ListedColormap(
        ["#eef2ef", "#bbbbbb", "#555555", "#888888", "#2e9e3f", "#3a86ff"])
    ax.imshow(np.where(top < 0, np.nan, top).T, origin="lower", cmap="viridis",
              aspect="equal", alpha=0.85)
    for st in stations:
        if not st.get("in_grid"):
            continue
        ax.plot(st["vx"], st["vz"], "r*" if st["status"] == "live" else "ko",
                markersize=10 if st["status"] == "live" else 6)
        ax.text(st["vx"] + 2, st["vz"] + 2, st["site_id"].replace("_", "\n"),
                color="white", fontsize=6, va="bottom")
    ax.set_title(f"Outdoor studio3d · {cell} m · {nx}×{ny}×{nz}")
    ax.set_xlabel("vx"); ax.set_ylabel("vz")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", type=float, default=2.0, help="studio voxel size m (default 2.0)")
    ap.add_argument("--pad-m", type=float, default=280.0, help="pad around BS bbox (m)")
    ap.add_argument("--gpkg", default=str(GPKG))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--city-cache", default=str(CITY_CACHE))
    ap.add_argument("--no-foliage", action="store_true")
    ap.add_argument("--foliage-height", type=float, default=8.0)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--max-buildings", type=int, default=2500)
    args = ap.parse_args()

    import geopandas as gpd

    sites = load_sites(REGISTRY)
    lon0, lat0, lon1, lat1 = station_bbox(sites)
    print(f"BS cluster lon/lat bbox ({lon0:.5f},{lat0:.5f})–({lon1:.5f},{lat1:.5f})")

    anchor = list(VC.lonlat_to_merc(*VG.FCC_LONLAT))
    fr = VG.frame_from_bbox(lon0, lat0, lon1, lat1, args.cell, 38.9, anchor, args.pad_m)
    cell, merc_scale, ox, oz, nx, nz = (fr["cell"], fr["merc_scale"],
                                        fr["ox"], fr["oz"], fr["nx"], fr["nz"])
    read_bbox = VG.bbox_lonlat_of_frame(fr, anchor)
    print(f"frame cell={cell} m  grid {nx}×?×{nz}  origin_xz=({ox:.1f},{oz:.1f})")
    print(f"read bbox {tuple(round(x, 5) for x in read_bbox)}")

    t0 = time.time()
    gdf = gpd.read_file(args.gpkg, layer="building", bbox=read_bbox, engine="pyogrio")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    print(f"read {len(gdf):,} footprints in {time.time() - t0:.1f}s")

    raw = [VG.parse_height(v, np.nan, 3.0) for v in gdf["height"].tolist()] \
        if "height" in gdf.columns else [np.nan] * len(gdf)
    raw = np.asarray(raw, float)
    valid = np.isfinite(raw)
    dflt = float(np.nanmedian(raw[valid])) if valid.any() else 12.0
    heights = np.where(valid, raw, dflt)

    # Apply surveyed height overrides by nearest footprint to known sites.
    for s in sites:
        h_ov = HEIGHT_OVERRIDES.get(s["site_id"])
        if h_ov is None or s.get("lat") is None:
            continue
        # find nearest footprint centroid
        best_i, best_d = None, 1e9
        for i, geom in enumerate(gdf.geometry.values):
            if geom is None:
                continue
            c = geom.centroid
            d = (float(c.x) - float(s["lon"])) ** 2 + (float(c.y) - float(s["lat"])) ** 2
            if d < best_d:
                best_d, best_i = d, i
        if best_i is not None and best_d < (0.0004 ** 2):  # ~40 m in deg²-ish
            heights[best_i] = h_ov
            print(f"  height override {s['site_id']}: footprint#{best_i} → {h_ov} m")

    ny = int(math.ceil(float(heights.max()) / cell)) + 2
    t0 = time.time()
    hm = VG.rasterize_heightmap(gdf, heights, nx, nz, merc_scale, anchor, ox, oz, cell)
    M = VG.columns_to_grid(hm, ny, model="penetrable", core=2, facade=5, barrier=3)

    # Ensure every building keeps a concrete core (never all-glass thin footprints).
    # columns_to_grid already erodes; if a footprint is 1-cell wide the whole column
    # is facade — paint the lowest solid cell as core so FDTD/eikonal still see mass.
    skin = M == 5
    core = M == 2
    only_glass = skin.any(axis=1) & ~core.any(axis=1)  # (nx,nz)
    if only_glass.any():
        for ix, iz in zip(*np.where(only_glass)):
            ys = np.where(M[ix, :, iz] == 5)[0]
            if ys.size:
                M[ix, ys[0], iz] = 2
        print(f"  core-backed {int(only_glass.sum())} thin glass-only columns")

    n_fol = 0
    if not args.no_foliage:
        try:
            hm_fol = VG.fetch_foliage(read_bbox, fr, anchor, args.foliage_height, cell)
            if hm_fol is not None:
                M = VG.foliage_to_grid(M, hm_fol, ny, foliage=4)
                n_fol = int((M == 4).any(axis=1).sum())
        except Exception as e:
            print(f"  (foliage skipped: {e})")

    inside, valid_tx = VC.make_masks(M, 3, (4.0, 12.0), cell)
    print(f"fill+masks in {time.time() - t0:.1f}s · grid {nx}×{ny}×{nz} · "
          f"foliage cols {n_fol:,}")

    city = Path(args.city_cache)
    city.mkdir(parents=True, exist_ok=True)
    np.save(city / "material_grid.npy", M)
    np.save(city / "inside_mask.npy", inside)
    np.save(city / "valid_tx_mask.npy", valid_tx)

    template = json.loads((SIM3D / "manifest_3d.json").read_text())
    # Replace indoor material table with outdoor 4-class legend for the studio.
    template = dict(template)
    template["materials"] = [
        {"id": m["id"], "name": m["name"].replace(" / ", "_").replace(" ", "_"),
         "loss_db": m.get("loss_db", 0.0),
         "loss_per_m_db": m.get("loss_per_m_db", 0.0),
         "color": m["color"]}
        for m in OUTDOOR_MATERIALS
    ]
    man = VC.city_manifest(
        template, source_gpkg=str(args.gpkg), source_layer="building",
        domain="city_outdoor", mode="2.5d-gpkg-studio3d",
        material_model="penetrable",
        classes_present=sorted(int(c) for c in np.unique(M)),
        core_class=2, facade_class=5, foliage_class=(4 if n_fol else None),
        building_class=3, cell_size_m=round(cell, 6), m_per_unit=1.0,
        merc_lat=38.9, merc_scale=round(merc_scale, 6), epsg=3857,
        merc_anchor=[round(anchor[0], 3), round(anchor[1], 3)],
        anchor_lonlat=list(VG.FCC_LONLAT),
        origin_units=[round(ox, 4), 0.0, round(oz, 4)],
        grid_shape=[nx, ny, nz],
        ceiling_height_m=round(float(heights.max()), 3),
        room_band=[1, ny - 1], terrain=False,
        bbox_lonlat=[round(x, 7) for x in read_bbox],
        n_buildings=int(len(gdf)),
        outdoor_materials=OUTDOOR_MATERIALS,
        studio="outdoor_fw_3d",
    )
    (city / "manifest_3d.json").write_text(json.dumps(man, indent=2))

    stations = place_stations(sites, man, M)
    for st in stations:
        tag = "LIVE" if st.get("status") == "live" and st.get("in_grid") else st.get("status", "?")
        if st.get("in_grid"):
            print(f"  {st['site_id']:22s} [{tag}] cell ({st['vx']},{st['vy']},{st['vz']}) "
                  f"roof {st['roof_height_m']} m ({st['height_source']})")
        else:
            print(f"  {st['site_id']:22s} [{tag}] SKIP {st.get('reason')}")

    extent_m = (nx * cell, nz * cell)
    height_m = ny * cell
    budgets = fdtd_budgets(extent_m, height_m)
    print(f"FDTD foundation · domain {budgets['domain_m']} m")
    print(f"  indoor-scale 0.3 m → {_fmt_cells(budgets['indoor_scale_0p3m']['n_cells'])}")
    for b in budgets["lambda_over_8"]:
        print(f"  λ/8 @{b['f_mhz']:.0f} MHz ({b['cell_m']*1000:.1f} mm) → {b['n_cells_label']}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # XYZ contiguous uint8 for fetch()
    grid_u8 = np.ascontiguousarray(M.astype(np.uint8))
    (out / "material_grid.bin").write_bytes(grid_u8.tobytes())
    n_build = export_buildings_json(gdf, heights, fr, anchor, cell,
                                    out / "buildings.json", args.max_buildings)

    scene = {
        "version": 1,
        "studio": "outdoor_fw_3d",
        "domain": "city_outdoor",
        "cell_size_m": cell,
        "grid_shape": [nx, ny, nz],
        "origin_units": man["origin_units"],
        "epsg": 3857,
        "merc_scale": man["merc_scale"],
        "merc_anchor": man["merc_anchor"],
        "anchor_lonlat": man["anchor_lonlat"],
        "bbox_lonlat": man["bbox_lonlat"],
        "materials": OUTDOOR_MATERIALS,
        "physics": {
            "0": {"n": 1.0, "gPerF": 0.0, "rigid": False},
            "2": {"n": 2.2944, "gPerF": 0.42812, "rigid": False},
            "3": {"n": 2.2944, "gPerF": 0.0, "rigid": True},
            "4": {"n": 1.411, "gPerF": 0.00294, "rigid": False},
            "5": {"n": 2.512, "gPerF": 0.04076, "rigid": False},
        },
        "wall_db_per_voxel": [0, 2.5, 18.0, 30.0, 1.5, 3.0],
        "stations": stations,
        "fdtd_foundation": budgets,
        "assets": {
            "material_grid": "material_grid.bin",
            "buildings": "buildings.json",
            "dtype": "uint8",
            "order": "x-y-z",
        },
        "n_buildings_mesh": n_build,
        "notes": (
            "Browser studio grid is 2 m (Outdoor 2-D parity). "
            "fdtd_foundation records the massive indoor-scale / λ/8 cell budgets "
            "for the later full-wave scene. Transmitters are known base stations "
            "from sites_registry.json (BS1 TV + BS3 Starry excluded from live Tx)."
        ),
    }
    (out / "scene.json").write_text(json.dumps(scene, indent=2))

    if not args.no_preview:
        try:
            preview(M, cell, stations, out / "preview_xy.png")
        except Exception as e:
            print(f"  (preview skipped: {e})")

    nvox = nx * ny * nz
    print(f"saved → {out}")
    print(f"  material_grid.bin {(nvox / 1e6):.2f} MB · buildings {n_build} · "
          f"solid {100 * (M > 0).sum() / nvox:.1f}%")
    print(f"  city cache → {city}")


if __name__ == "__main__":
    # fix typo guard — place_stations had a bad token in draft; ensure clean exec
    main()
