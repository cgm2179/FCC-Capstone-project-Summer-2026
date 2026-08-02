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
        ix = np.clip(np.floor(pts[:, 0]).astype(np.int64), 0, nx - 1)
        iz = np.clip(np.floor(pts[:, 2]).astype(np.int64), 0, nz - 1)
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
    ap.add_argument("--no-preview", action="store_true", help="skip the preview PNG")
    args = ap.parse_args()

    mesh_path = Path(args.mesh) if args.mesh else ROOT / DEFAULT_OBJ
    if not mesh_path.is_absolute():
        mesh_path = (ROOT / mesh_path)
    out = Path(args.out) if args.out else HERE / "city" / mesh_path.stem
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    cls = int(args.building_class)

    print(f"loading {mesh_path.name} ({mesh_path.suffix.lower()}) ...")
    verts, tris, up = load_mesh(mesh_path, args.up_axis)
    print(f"  {len(verts)} verts, {len(tris)} triangles  "
          f"up-axis={up}{' (auto)' if args.up_axis == 'auto' else ''}")

    # ---- scale: metres already; optional Web-Mercator horizontal correction ---
    merc = math.cos(math.radians(args.merc_lat)) if args.merc_lat else 1.0
    if merc != 1.0:
        verts = verts.copy()
        verts[:, 0] *= merc            # X horizontal
        verts[:, 2] *= merc            # Z horizontal (Y is up = true metres)
    lo = verts.min(0)
    span = verts.max(0) - lo
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
    man = city_manifest(
        template,
        source_obj=src_rel, domain="city_outdoor", mode=args.mode,
        building_class=cls, cell_size_m=round(args.cell, 6), m_per_unit=1.0,
        merc_lat=args.merc_lat, merc_scale=round(merc, 6),
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
