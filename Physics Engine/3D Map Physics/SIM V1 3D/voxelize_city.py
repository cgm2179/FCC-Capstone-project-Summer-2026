#!/usr/bin/env python3
"""
voxelize_city.py — outdoor *city* voxelizer (SIM V1 3D · city/outdoor domain).

Sibling of voxelize.py, but for a georeferenced OSM / BlenderGIS **city** model
(many buildings, streets between them) instead of one indoor floor. The engine
(engine_3d.py · SceneV3) is reused unchanged — this only produces the grid,
masks and manifest it consumes. Differences from the indoor voxelizer:

  * SCALE. The OBJ is already in metres (BlenderGIS Web-Mercator, 1 BU = 1 m),
    so m_per_unit = 1.0 — it is NOT registered to a floor-plan width. Optional
    Web-Mercator latitude correction: projected horizontal distances are
    inflated by 1/cos(lat) while extruded heights are true metres, so
    --merc-lat compresses X/Z by cos(lat) and leaves Y (height) alone.

  * OCCUPANCY — two modes:
      2.5d  extruded-footprint COLUMN FILL (default). A per-(x,z) roof
            heightmap from the projected triangles, then each footprint column
            is filled solid ground->roof. Fast, low memory, exact for the
            prismatic OSM building extrusions.
      3d    TRUE VOLUMETRIC. Surface-rasterize every triangle, seal the ground
            under footprints, flood-fill exterior air from the domain boundary,
            and solidify the enclosed interior. Respects real 3-D shape
            (setbacks, sloped/stepped roofs, overhangs) for detailed meshes.

  * MATERIAL. Every building -> one BARRIER class (default 3 = core/concrete).
    Class 3 is both high-loss in the CrossingLUT AND the class engine_3d's
    eikonal routes AROUND (speed_field barrier_classes=(3,)) -> real
    street-canyon NLOS shadows. OSM faces carry no RF material names, so the
    indoor per-usemtl map does not apply; all faces are buildings.

  * MASKS (outdoor semantics — the inverse of the indoor interior fill):
      inside_mask    exterior OPEN AIR, flood-filled from the domain boundary
                     (streets + everything above the rooftops) = the
                     propagation / evaluation space.
      valid_tx_mask  candidate transmitters: rooftop-adjacent air (macro /
                     small-cell mounts); falls back to a low outdoor band if
                     there are no roofs.

Outputs into --out:  material_grid.npy · inside_mask.npy · valid_tx_mask.npy ·
manifest_3d.json (physics/freqs/materials copied from the indoor manifest so
`engine_3d.load_scene(sim_dir=<out>)` just works).

Meshes: .obj and .stl load natively (no dependency); .ply/.gltf/.glb/.dae/.fbx
load via trimesh (pip install trimesh; FBX also needs an assimp backend). The
up-axis is auto-detected (smallest-extent axis) or forced with --up-axis.

usage:
  python "SIM V1 3D/voxelize_city.py"                       # NoMa OBJ · 2.5d · 1 m
  python "SIM V1 3D/voxelize_city.py" --mode 3d --cell 2
  python "SIM V1 3D/voxelize_city.py" --mesh city.glb --up-axis z --out city/Foo
  python SIM3D/voxelize_city.py --cell 2 --origin-m 800 0 600 \
      --size-m 256 64 256 --out city_demo/NoMa_DC_tile --no-preview
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in (HERE, *HERE.parents) if (p / "Data" / "models").is_dir()),
            HERE.parent)

DEFAULT_OBJ = "Data/models/NoMa_DC/NoMa_DC_buildings.obj"
BUILDING_CLASS = 3           # core/concrete: high crossing loss AND eikonal barrier
MERC_LAT_DEFAULT = 38.90     # NoMa, Washington DC (Web-Mercator scale reference)
# SceneV3 holds a dense (N,3) float32 coord array and ray-marches per voxel, so
# cost/RAM scale with the voxel count. Warn past this (≈0.5 GB just for coords).
ENGINE_VOXEL_WARN = 40_000_000

# ------------------------------------------------------------- georeference (EPSG:3857 anchor cube)
# The NoMa render is a LOCAL metre frame (verts near the origin) PLUS a small "anchor" cube placed at
# the ABSOLUTE Web-Mercator (EPSG:3857) coordinate of the local origin — the prepped OBJ has `o Cube`
# at `v 8572395 0 4707866` (|easting| / height / northing = the FCC block). So
#     local = epsg3857(lon,lat) - merc_anchor
# makes lon/lat <-> voxel exact with no BlenderGIS-origin guessing (verified: puts all six donors
# inside the city and reproduces Forte->FCC = 416 m vs the measured 415). One sign quirk: EPSG X here
# is WEST-positive (X = -easting), mirroring the indoor floor-plan registration flip. Scenes store true
# ground metres (local X/Z compressed by merc_scale = cos(merc_lat)); these helpers read the manifest
# (merc_anchor + merc_scale + origin_units + cell) and undo it consistently.
R_MERC = 6_378_137.0            # EPSG:3857 sphere radius
ANCHOR_MIN_M = 1e6              # verts past this |X| are the absolute-Mercator anchor, not local geometry


def lonlat_to_merc(lon, lat):
    """(x, z) in ABSOLUTE EPSG:3857 metres. x = -easting (this render's X runs west); z = northing."""
    x = -R_MERC * math.radians(lon)
    z = R_MERC * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, z


def merc_to_lonlat(x, z):
    lon = -math.degrees(x / R_MERC)
    lat = math.degrees(2.0 * math.atan(math.exp(z / R_MERC)) - math.pi / 2.0)
    return lon, lat


def lonlat_to_vox(man, lon, lat, height_m=None):
    """(vx, vz) [or (vx, vy, vz) if height_m] voxel coords for a lon/lat in a georeferenced city scene.

    Needs `man['epsg']==3857` + `man['merc_anchor']`. local = epsg3857(lon,lat) - anchor;
    scaled = merc_scale * local; vox = (scaled - origin_units)/cell (height uses the unscaled Y origin)."""
    if man.get("epsg") != 3857 or "merc_anchor" not in man:
        raise ValueError("lonlat_to_vox needs a georeferenced EPSG:3857 city scene (re-voxelize with the anchor cube)")
    merc = float(man.get("merc_scale", 1.0)); cell = float(man["cell_size_m"])
    o = man["origin_units"]; anch = man["merc_anchor"]
    ox, oz = lonlat_to_merc(lon, lat)
    vx = (merc * (ox - anch[0]) - o[0]) / cell
    vz = (merc * (oz - anch[1]) - o[2]) / cell
    if height_m is None:
        return vx, vz
    return vx, (float(height_m) - o[1]) / cell, vz


def vox_to_lonlat(man, vx, vz):
    """Inverse of lonlat_to_vox (horizontal only)."""
    merc = float(man.get("merc_scale", 1.0)); cell = float(man["cell_size_m"])
    o = man["origin_units"]; anch = man["merc_anchor"]
    ox = (vx * cell + o[0]) / merc + anch[0]
    oz = (vz * cell + o[2]) / merc + anch[1]
    return merc_to_lonlat(ox, oz)


def parse_obj(path):
    """(verts (V,3) float64, tris (T,3) int32). Polygons fan-triangulated;
    only vertex indices are used (every face is a building)."""
    verts, tris = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    s = tok.split("/")[0]
                    if s:
                        i = int(s)
                        idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):
                    tris.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(verts, np.float64), np.asarray(tris, np.int32)


def parse_stl(path):
    """(verts (V,3), tris (T,3)) from an STL — binary or ASCII, no dependency.
    STL is a triangle soup (no shared vertices); that's fine for rasterization."""
    data = Path(path).read_bytes()
    if len(data) >= 84:                                  # binary if size matches the header count
        n = int.from_bytes(data[80:84], "little")
        if len(data) == 84 + n * 50:
            dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            v = np.frombuffer(data, dtype=dt, count=n, offset=84)["v"].reshape(-1, 3)
            return v.astype(np.float64), np.arange(3 * n, dtype=np.int32).reshape(-1, 3)
    verts = [[float(x) for x in s.split()[1:4]]          # ASCII: "vertex x y z"
             for s in data.decode("utf-8", "replace").splitlines()
             if s.strip().startswith("vertex")]
    v = np.asarray(verts, np.float64)
    return v, np.arange(len(v), dtype=np.int32).reshape(-1, 3)


def load_via_trimesh(path):
    """(verts, tris) for .ply/.gltf/.glb/.dae/.fbx/.3ds… via trimesh (optional).
    force='mesh' flattens scene graphs and bakes node transforms to world coords."""
    try:
        import trimesh
    except ImportError:
        raise SystemExit(
            f"{Path(path).suffix} needs trimesh:  pip install trimesh"
            f"  (FBX/3DS also need an assimp backend: pip install pyassimp).\n"
            f"  .obj and .stl load natively with no extra dependency.")
    m = trimesh.load(str(path), force="mesh", process=False)
    if getattr(m, "faces", None) is None or len(m.faces) == 0:
        raise SystemExit(f"{Path(path).name}: no triangles found after loading.")
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int32)


LOADERS = {".obj": parse_obj, ".stl": parse_stl}


def load_mesh(path, up_axis="auto"):
    """Dispatch by extension, then move the up-axis to column 1 (internal Y-up).
    up_axis='auto' picks the smallest-extent axis as height (buildings/terrain
    are always far wider than tall). Returns (verts Y-up, tris, up_name)."""
    ext = Path(path).suffix.lower()
    verts, tris = LOADERS.get(ext, load_via_trimesh)(path)
    if verts.size == 0 or tris.size == 0:
        raise SystemExit(f"{Path(path).name}: empty mesh (no vertices/faces).")
    if up_axis == "auto":
        u = int(np.argmin(verts.max(0) - verts.min(0)))
    else:
        u = {"x": 0, "y": 1, "z": 2}[up_axis]
    if u != 1:                                           # reorder columns -> [horiz, up, horiz]
        others = [i for i in (0, 1, 2) if i != u]
        verts = verts[:, [others[0], u, others[1]]]
    return verts, tris, "xyz"[u]


def tri_sample_points(tri, step=0.5):
    """Dense barycentric samples covering one triangle (coords in VOXEL units).
    Spacing ~`step` voxels so every crossed voxel is hit. tri: (3,3)."""
    e = float(np.linalg.norm(np.diff(np.vstack([tri, tri[0]]), axis=0), axis=1).max())
    n = max(2, int(math.ceil(e / step)))
    a = np.linspace(0.0, 1.0, n)
    u, v = np.meshgrid(a, a)
    keep = (u + v) <= 1.0
    u, v = u[keep], v[keep]
    w = 1.0 - u - v
    return w[:, None] * tri[0] + u[:, None] * tri[1] + v[:, None] * tri[2]


def exterior_air(M):
    """Boolean mask of open air connected to the domain boundary (6-conn flood
    fill). Excludes solids and any fully-enclosed interior pockets."""
    air = M == 0
    lbl, n = ndimage.label(air)                    # 6-connectivity (default SE)
    if n == 0:
        return air
    border = np.unique(np.concatenate([
        lbl[0].ravel(), lbl[-1].ravel(),
        lbl[:, 0].ravel(), lbl[:, -1].ravel(),
        lbl[:, :, 0].ravel(), lbl[:, :, -1].ravel()]))
    border = border[border > 0]
    return np.isin(lbl, border)


def build_25d(verts_vox, tris, shape, cls, terrain_v=None):
    """2.5-D column fill. Roof heightmap from projected triangles (Y is up),
    footprint holes filled per-building, then each column solid ground->roof.

    terrain_v (nx,nz) optional: per-(x,z) GROUND voxel height from a real DEM. When given, the
    ground follows the terrain EVERYWHERE (not a flat plane under footprints only) and buildings
    extrude from the terrain surface — this is the outdoor 'real ground level'."""
    nx, ny, nz = shape
    hm = np.full(nx * nz, -1.0, np.float64)        # roof height (voxel-Y) per (x,z)
    for t in tris:
        pts = tri_sample_points(verts_vox[t])
        ijk = np.floor(pts).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < np.array(shape)), axis=1)
        if not ok.any():
            continue
        pts = pts[ok]
        ijk = ijk[ok]
        ix = ijk[:, 0]
        iz = ijk[:, 2]
        np.maximum.at(hm, ix * nz + iz, pts[:, 1])
    hm = hm.reshape(nx, nz)

    foot = hm >= 0.0
    filled = ndimage.binary_fill_holes(foot)       # solidify ring-only footprints
    need = filled & ~foot
    if need.any():                                  # interior cells w/o a roof face
        lbl, nlab = ndimage.label(filled)
        cmax = ndimage.maximum(np.where(foot, hm, -1.0), lbl, np.arange(1, nlab + 1))
        hm = hm.copy()
        hm[need] = cmax[lbl[need] - 1]
        foot = filled

    hcol = np.floor(np.clip(hm, 0, ny - 1)).astype(np.int32)   # roof voxel index (building height)
    M = np.zeros((nx, ny, nz), np.int8)
    if terrain_v is None:
        for iy in range(ny):                        # y-loop keeps peak RAM at one slice
            M[:, iy, :][foot & (hcol >= iy)] = cls
    else:
        # ground follows the terrain EVERYWHERE; buildings extrude from the terrain surface
        gv = np.clip(terrain_v, 0, ny - 1).astype(np.int32)
        topcol = np.clip(gv + np.where(foot, hcol, 0), 0, ny - 1)   # terrain (+ building) top
        for iy in range(ny):
            M[:, iy, :][topcol >= iy] = cls
    return M


def build_3d(verts_vox, tris, shape, cls):
    """True volumetric: surface-rasterize the actual triangles, seal open
    bottoms, close <=1-voxel raster pinholes, and solidify air enclosed by the
    shells. Preserves real 3-D structure (setbacks, sloped roofs, voids) on
    watertight / detailed meshes. Returns (M, watertight_ratio): open meshes
    (raw OSM extrusions) stay partly hollow -> low ratio -> caller warns."""
    nx, ny, nz = shape
    M = np.zeros((nx, ny, nz), np.int8)
    for t in tris:
        pts = tri_sample_points(verts_vox[t])
        ijk = np.floor(pts).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < np.array(shape)), axis=1)
        ijk = ijk[ok]
        if ijk.size:
            M[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = cls

    # reference volume: each footprint column solid to its surface top (y-loop
    # keeps peak RAM at one slice) — the watertightness yardstick.
    top = np.full((nx, nz), -1, np.int32)
    for iy in range(ny):
        top[M[:, iy, :] > 0] = iy
    foot = top >= 0
    ref = int((top[foot] + 1).sum())

    M[:, 0, :][foot] = cls                          # seal open bottoms at ground
    solid = ndimage.binary_closing(M > 0, iterations=1)  # seal <=1-voxel raster pinholes
    M[solid] = cls                                  # (else the flood-fill leaks inside)
    interior = (M == 0) & ~exterior_air(M)          # enclosed voids -> solid
    M[interior] = cls
    return M, float((M > 0).sum() / max(ref, 1))


def load_terrain(path, nx, nz, cell_m):
    """Load a DEM heightmap (.npy 2-D elevations in metres) and resample to the (nx,nz) grid,
    returning per-(x,z) GROUND voxel heights (normalised so the lowest point is 0). This is what
    replaces the flat artificial ground with real terrain — the outdoor 'ground level'. Fetch a
    DEM for the scene bbox with fetch_terrain_opentopo.py (needs an OpenTopography API key)."""
    dem = np.load(path).astype(np.float64)
    if dem.ndim != 2:
        raise SystemExit(f"--terrain {path}: expected a 2-D heightmap, got shape {dem.shape}")
    zy, zx = nx / dem.shape[0], nz / dem.shape[1]
    dem_g = ndimage.zoom(dem, (zy, zx), order=1)[:nx, :nz]
    dem_g = dem_g - float(np.nanmin(dem_g))                 # lowest point -> 0
    return np.nan_to_num(dem_g / cell_m).round().astype(np.int32)   # metres -> voxel heights


def make_masks(M, cls, tx_band_m, cell_m):
    """inside = exterior open air (eval space); valid_tx = rooftop-adjacent air
    (fallback: outdoor air in a low mounting band)."""
    outside = exterior_air(M)
    below_solid = np.zeros_like(M, bool)            # a building voxel directly below
    below_solid[:, 1:, :] = M[:, :-1, :] > 0
    roof_air = outside & below_solid
    if roof_air.sum() < 8:                          # no usable roofs -> low air band
        lo = max(1, int(round(tx_band_m[0] / cell_m)))
        hi = int(round(tx_band_m[1] / cell_m))
        band = np.zeros(M.shape[1], bool)
        band[lo:hi + 1] = True
        roof_air = outside & band[None, :, None]
    return outside, roof_air


def city_manifest(template, **over):
    """Copy physics/freqs/materials from the indoor manifest; override geometry
    so engine_3d.load_scene(sim_dir=<out>) reads one source of truth."""
    keep = ("materials", "physics", "freqs_mhz", "freq_loss_mult", "norm",
            "speed_of_light_mps", "tx")
    man = {k: template[k] for k in keep if k in template}
    man.update(version="sim3d-city-v1.0", axes=["X", "Y_up", "Z"])
    man.update(over)
    return man


def _selftest():
    """Georeference round-trips (pure math) + donor placement in the NoMa_FCC_tile if built."""
    ok, fails = 0, []

    def check(name, cond, msg=""):
        nonlocal ok
        if cond:
            ok += 1; print(f"  PASS  {name}")
        else:
            fails.append(name); print(f"  FAIL  {name}  {msg}")

    for lon, lat in [(-77.01142, 38.90155), (-77.0074, 38.90359)]:
        x, z = lonlat_to_merc(lon, lat); rlon, rlat = merc_to_lonlat(x, z)
        check(f"merc round-trip {lon},{lat}", abs(rlon - lon) < 1e-9 and abs(rlat - lat) < 1e-9,
              f"{rlon},{rlat}")
    man = dict(epsg=3857, merc_scale=math.cos(math.radians(38.9)), cell_size_m=3.0,
               origin_units=[-674.8, 0.0, -495.3], merc_anchor=[8572395.0, 4707865.5])
    vx, vz = lonlat_to_vox(man, -77.0074, 38.90359)
    rlon, rlat = vox_to_lonlat(man, vx, vz)
    check("lonlat->vox->lonlat round-trip", abs(rlon + 77.0074) < 1e-6 and abs(rlat - 38.90359) < 1e-6,
          f"{rlon},{rlat}")

    tile = HERE / "city" / "NoMa_FCC_tile"
    if (tile / "manifest_3d.json").exists():
        m2 = json.loads((tile / "manifest_3d.json").read_text())
        nx, ny, nz = np.load(tile / "material_grid.npy").shape
        allin = all(0 <= (v := lonlat_to_vox(m2, lon, lat))[0] < nx and 0 <= v[1] < nz
                    for lon, lat in [(-77.01142, 38.90155), (-77.0074, 38.90359), (-77.00848, 38.90309)])
        check("NoMa_FCC_tile: Forte/FCC/BS6 land inside", allin)
    print(f"\n{ok} passed, {len(fails)} failed")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", "--obj", dest="mesh", default=None,
                    help=f".obj/.stl (native) or .ply/.gltf/.glb/.dae/.fbx (needs trimesh) "
                         f"(default {DEFAULT_OBJ})")
    ap.add_argument("--up-axis", choices=["auto", "x", "y", "z"], default="auto",
                    help="which axis is up (default auto = smallest-extent axis)")
    ap.add_argument("--out", default=None, help="output dir (default city/<mesh stem>)")
    ap.add_argument("--mode", choices=["2.5d", "3d"], default="2.5d",
                    help="occupancy mode (default 2.5d column fill)")
    ap.add_argument("--cell", type=float, default=1.0,
                    help="cubic voxel size in metres (default 1.0)")
    ap.add_argument("--origin-m", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="crop origin in mesh metres after Web-Mercator scaling; with "
                         "--size-m defines the AABB lower corner")
    ap.add_argument("--size-m", type=float, nargs=3, default=None, metavar=("SX", "SY", "SZ"),
                    help="crop size in metres; voxels outside origin/size are discarded")
    ap.add_argument("--building-class", type=int, default=BUILDING_CLASS,
                    help=f"RF class for buildings (default {BUILDING_CLASS} = concrete barrier)")
    ap.add_argument("--merc-lat", type=float, default=MERC_LAT_DEFAULT,
                    help="Web-Mercator latitude for horizontal scale correction; 0 disables")
    ap.add_argument("--tx-band", type=float, nargs=2, default=(4.0, 12.0),
                    help="fallback Tx height band (m) if no roofs (default 4 12)")
    ap.add_argument("--terrain", default=None,
                    help="DEM heightmap .npy (metres) — outdoor ground follows REAL terrain "
                         "instead of a flat plane (2.5d mode). Fetch for the scene bbox with "
                         "fetch_terrain_opentopo.py (needs an OpenTopography API key).")
    ap.add_argument("--bbox-lonlat", type=float, nargs=4, default=None,
                    metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                    help="crop to a lon/lat box (needs an absolute EPSG:3857 mesh); converts to the "
                         "Mercator AABB crop. E.g. the 7/24 walk: -77.0144 38.900 -77.0005 38.908")
    ap.add_argument("--bbox-pad-m", type=float, default=100.0,
                    help="metres of padding around --bbox-lonlat (default 100; keeps donor rooftops "
                         "+ first-bounce reflectors inside the tile)")
    ap.add_argument("--no-preview", action="store_true", help="skip the preview PNG")
    ap.add_argument("--test", action="store_true", help="run the georeference self-test (no voxelize)")
    args = ap.parse_args()
    if args.test:
        raise SystemExit(_selftest())

    mesh_path = Path(args.mesh) if args.mesh else ROOT / DEFAULT_OBJ
    if not mesh_path.is_absolute():
        mesh_path = (ROOT / mesh_path)
    out = Path(args.out) if args.out else HERE / "city" / mesh_path.stem
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    cls = int(args.building_class)
    if (args.origin_m is None) != (args.size_m is None):
        raise SystemExit("--origin-m and --size-m must be provided together")

    print(f"loading {mesh_path.name} ({mesh_path.suffix.lower()}) ...")
    verts, tris, up = load_mesh(mesh_path, args.up_axis)
    print(f"  {len(verts)} verts, {len(tris)} triangles  "
          f"up-axis={up}{' (auto)' if args.up_axis == 'auto' else ''}")

    # separate the absolute-Mercator ANCHOR cube (the georeference marker) from the local city.
    # It sits at EPSG:3857 metres (|coord| >> the local frame), so it would both wreck the AABB and
    # add a phantom building; drop it and keep its centre as the local origin's Mercator coordinate.
    cube_v = (np.abs(verts) > ANCHOR_MIN_M).any(axis=1)
    merc_anchor = None
    if cube_v.any():
        T = verts[cube_v].mean(0)
        merc_anchor = [float(T[0]), float(T[2])]
        keep = ~cube_v
        remap = -np.ones(len(verts), np.int64); remap[keep] = np.arange(int(keep.sum()))
        tris = remap[tris[keep[tris].all(1)]]
        verts = verts[keep]
        alon, alat = merc_to_lonlat(T[0], T[2])
        print(f"  anchor cube -> local origin lon {alon:.5f} lat {alat:.5f} "
              f"({int(cube_v.sum())} anchor verts dropped, {len(verts)} local verts kept)")
    epsg3857 = merc_anchor is not None

    # ---- scale: metres already; optional Web-Mercator horizontal correction ---
    merc = math.cos(math.radians(args.merc_lat)) if args.merc_lat else 1.0
    if merc != 1.0:
        verts = verts.copy()
        verts[:, 0] *= merc            # X horizontal
        verts[:, 2] *= merc            # Z horizontal (Y is up = true metres)
    lo = verts.min(0)
    span = verts.max(0) - lo
    full_lo, full_span = lo.copy(), span.copy()

    if args.bbox_lonlat is not None:
        if not epsg3857:
            raise SystemExit("--bbox-lonlat needs the EPSG:3857 anchor cube (absolute-Mercator marker) in the mesh")
        lon0, lat0, lon1, lat1 = args.bbox_lonlat
        cx, cz = [], []
        for lo_ in (lon0, lon1):
            for la_ in (lat0, lat1):
                ox, oz = lonlat_to_merc(lo_, la_)               # absolute EPSG:3857
                cx.append(merc * (ox - merc_anchor[0]))         # -> local -> scaled (true-metre) frame
                cz.append(merc * (oz - merc_anchor[1]))
        pad = float(args.bbox_pad_m)
        args.origin_m = [min(cx) - pad, float(full_lo[1]), min(cz) - pad]
        args.size_m = [max(cx) - min(cx) + 2 * pad, float(full_span[1]), max(cz) - min(cz) + 2 * pad]
        print(f"  --bbox-lonlat {args.bbox_lonlat} +{pad:.0f} m -> crop "
              f"origin={[round(x, 1) for x in args.origin_m]} size={[round(x, 1) for x in args.size_m]}")

    crop = args.origin_m is not None
    if crop:
        lo = np.asarray(args.origin_m, np.float64)
        span = np.asarray(args.size_m, np.float64)
        if np.any(span <= 0):
            raise SystemExit("--size-m values must be positive")
        hi = lo + span
        tri_v = verts[tris]
        tlo = tri_v.min(axis=1)
        thi = tri_v.max(axis=1)
        keep = np.all((thi >= lo) & (tlo <= hi), axis=1)
        print(f"  full scaled bounds origin={np.round(full_lo, 2).tolist()} "
              f"span={np.round(full_span, 2).tolist()}")
        print(f"  crop origin={np.round(lo, 2).tolist()} size={np.round(span, 2).tolist()} "
              f"keeps {int(keep.sum()):,}/{len(tris):,} triangles")
        tris = tris[keep]
        if len(tris) == 0:
            raise SystemExit("crop contains no mesh triangles; choose a different --origin-m/--size-m")
    pitch = args.cell                  # m_per_unit = 1.0 -> pitch(units) == cell(m)
    nx, ny, nz = (np.ceil(span / pitch).astype(int) + 1).tolist()
    nvox = nx * ny * nz
    print(f"  merc_lat={args.merc_lat} (scale {merc:.4f})  cell={args.cell} m  "
          f"grid={nx}x{ny}x{nz} = {nvox:,} voxels  ({nvox/1e6:.0f} MB int8)")
    print(f"  extent  {span[0]*1:.0f} x {span[2]:.0f} m footprint, {span[1]:.0f} m tall")
    if nvox > ENGINE_VOXEL_WARN:
        print(f"  ! {nvox:,} voxels is large for a dense SceneV3 solve "
              f"(coords alone ~{nvox*12/1e9:.1f} GB). The grid saves fine, but for "
              f"per-Tx engine runs use a coarser --cell (2-3 m) or a cropped tile.")

    terrain_v = None
    if args.terrain:
        tpath = Path(args.terrain)
        if not tpath.is_absolute():
            tpath = ROOT / tpath
        terrain_v = load_terrain(tpath, nx, nz, args.cell)
        extra = int(terrain_v.max())
        ny += extra                    # grow the domain to fit terrain + tallest building
        print(f"  terrain: ground follows the DEM · +{extra} voxels ({extra*args.cell:.0f} m relief)"
              f" · grid Y now {ny}")
        if args.mode == "3d":
            print("  ! terrain is applied in 2.5d fill only; --mode 3d ignores it for now.")

    verts_vox = (verts - lo) / pitch
    t0 = time.time()
    if args.mode == "2.5d":
        M = build_25d(verts_vox, tris, (nx, ny, nz), cls, terrain_v=terrain_v)
    else:
        M, watertight = build_3d(verts_vox, tris, (nx, ny, nz), cls)
        if watertight < 0.8:
            print(f"  ! mesh looks ~{100*(1-watertight):.0f}% non-watertight — 3D shells "
                  f"stay partly hollow and CAN leak the eikonal barrier. For open "
                  f"OSM/GIS extrusions use --mode 2.5d (guaranteed-solid prisms).")
    inside, valid_tx = make_masks(M, cls, tuple(args.tx_band), args.cell)
    print(f"  {args.mode} occupancy + masks in {time.time()-t0:.1f}s")

    np.save(out / "material_grid.npy", M)
    np.save(out / "inside_mask.npy", inside)
    np.save(out / "valid_tx_mask.npy", valid_tx)

    template = json.loads((HERE / "manifest_3d.json").read_text())
    try:
        src_rel = str(mesh_path.relative_to(ROOT))
    except ValueError:
        src_rel = str(mesh_path)

    geo = {}
    if epsg3857:                                    # bake the lon/lat anchor so BS placement is exact
        def _ll(sx, sz):   # scaled-local (X,Z) -> lon/lat: undo merc_scale, add the anchor back
            return merc_to_lonlat(sx / merc + merc_anchor[0], sz / merc + merc_anchor[1])
        olon, olat = _ll(lo[0], lo[2])
        cs = [_ll(lo[0] + ex, lo[2] + ez) for ex in (0.0, float(span[0])) for ez in (0.0, float(span[2]))]
        lons = [c[0] for c in cs]; lats = [c[1] for c in cs]
        geo = dict(epsg=3857, merc_R=R_MERC,
                   merc_anchor=[round(merc_anchor[0], 3), round(merc_anchor[1], 3)],
                   origin_lonlat=[round(olon, 7), round(olat, 7)],
                   bbox_lonlat=[round(min(lons), 7), round(min(lats), 7),
                                round(max(lons), 7), round(max(lats), 7)])
        print(f"  EPSG:3857 anchor · scene bbox lon/lat {geo['bbox_lonlat']}")

    man = city_manifest(
        template,
        source_obj=src_rel, domain="city_outdoor", mode=args.mode,
        **geo,
        building_class=cls, cell_size_m=round(args.cell, 6), m_per_unit=1.0,
        merc_lat=args.merc_lat, merc_scale=round(merc, 6),
        crop_origin_m=([round(float(x), 4) for x in lo] if crop else None),
        crop_size_m=([round(float(x), 4) for x in span] if crop else None),
        full_origin_m=[round(float(x), 4) for x in full_lo],
        full_span_m=[round(float(x), 4) for x in full_span],
        terrain=bool(args.terrain), terrain_source=(str(args.terrain) if args.terrain else None),
        grid_shape=[nx, ny, nz],
        origin_units=[round(float(x), 4) for x in lo],
        ceiling_height_m=round(float(span[1]), 3),          # domain top (tallest bldg)
        room_band=[1, ny - 1],
    )
    (out / "manifest_3d.json").write_text(json.dumps(man, indent=2))

    solid = int((M == cls).sum())
    print(f"  buildings: {100*solid/nvox:.1f}% of volume solid")
    print(f"  inside (exterior air): {int(inside.sum()):,} voxels "
          f"({100*inside.sum()/nvox:.1f}%)")
    print(f"  valid Tx (rooftop/mount): {int(valid_tx.sum()):,} voxels")
    print(f"saved material_grid.npy, inside_mask.npy, valid_tx_mask.npy, "
          f"manifest_3d.json -> {out}")

    if not args.no_preview:
        try:
            _preview(M, cls, args, out)
        except Exception as e:                              # preview is non-essential
            print(f"  (preview skipped: {e})")


def _preview(M, cls, args, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nx, ny, nz = M.shape
    solid = M == cls
    top = np.where(solid, np.arange(ny)[None, :, None], -1).max(1) * args.cell  # (nx,nz) roof
    iz = nz // 2
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    im = ax[0].imshow(np.where(top < 0, np.nan, top).T, origin="lower", cmap="viridis", aspect="equal")
    ax[0].set_title(f"roof height map · {args.mode} · {args.cell} m"); ax[0].set_xlabel("X"); ax[0].set_ylabel("Z")
    fig.colorbar(im, ax=ax[0], shrink=0.8, label="height (m)")
    ax[1].imshow(solid[:, :, iz].T, origin="lower", cmap="Greys", aspect="auto")
    ax[1].set_title(f"vertical section Z={iz}"); ax[1].set_xlabel("X"); ax[1].set_ylabel("Y (up)")
    fig.suptitle(f"voxelize_city · {out.name} · grid {nx}x{ny}x{nz}", fontsize=12)
    fig.tight_layout()
    p = out / "preview_voxel.png"
    fig.savefig(p, dpi=110)
    print(f"  preview -> {p}")


if __name__ == "__main__":
    main()
