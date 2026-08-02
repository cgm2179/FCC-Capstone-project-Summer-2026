#!/usr/bin/env python3
"""
Phase A-3D — Voxelize the OBJ into a 3-D material grid (SIM V1 3D).

Pure NumPy (no mesh library). Parses the OBJ (v / usemtl / f), maps each face's
material name to an RF class via manifest_3d.json's `obj_material_map`, and
surface-rasterizes every triangle into a voxel grid. Produces:

  material_grid.npy  (nx, ny, nz) int8   material class per voxel (0 = air)
  inside_mask.npy    (nx, ny, nz) bool    interior air (flood-fill from faces)
  valid_tx_mask.npy  (nx, ny, nz) bool    interior air at plausible Tx heights

Axes are (X, Y-up, Z). Scale: the OBJ X-span is registered to the floor-plan
width (manifest `scale_reference`) → m_per_unit, giving a realistic ceiling
height. grid_shape / cell_size_m / m_per_unit / origin are written back into
manifest_3d.json so the engine and dataset stages read one source of truth.

usage:
  python "SIM V1 3D/voxelize.py"              # cell from manifest.target_cell_size_m
  python "SIM V1 3D/voxelize.py" --cell 0.5   # coarser + faster (fewer voxels)
"""
import argparse
import sys
import json
import re
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
# The repo reorg moved this file deep under "Physics Engine/", so the model tree
# (Data/) is no longer HERE.parent. Resolve the repo root as the nearest ancestor
# that actually contains Data/models (manifest["source_obj"] is relative to it).
ROOT = next((p for p in (HERE, *HERE.parents) if (p / "Data" / "models").is_dir()),
            HERE.parent)


def parse_obj(path):
    """Return (verts (V,3) float64, faces list of (i0, i1, i2, matname)).
    Polygons are fan-triangulated; only vertex indices are used."""
    verts, faces, cur = [], [], ""
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("usemtl"):
                parts = line.split(maxsplit=1)
                cur = parts[1].strip() if len(parts) > 1 else ""
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    s = tok.split("/")[0]
                    if s == "":
                        continue
                    i = int(s)
                    idx.append(i - 1 if i > 0 else len(verts) + i)  # 1-based / neg
                for k in range(1, len(idx) - 1):                    # fan
                    faces.append((idx[0], idx[k], idx[k + 1], cur))
    return np.asarray(verts, np.float64), faces


def class_for(name, rules):
    low = name.lower()
    for r in rules:
        if re.search(r["match"], low):
            return int(r["class"])
    return 0


def rasterize_triangle(tri, cls, M, origin, pitch, oversample=2.0):
    """Surface-rasterize one triangle into M[ix,iy,iz] by dense barycentric
    sampling at ~pitch/oversample spacing. Sets voxels to `cls` (caller passes
    triangles in ascending material priority so higher classes overwrite)."""
    edges = np.vstack([tri[1] - tri[0], tri[2] - tri[1], tri[0] - tri[2]])
    e = float(np.linalg.norm(edges, axis=1).max())
    n = max(2, int(np.ceil(oversample * e / pitch)))
    a = np.linspace(0.0, 1.0, n)
    u, v = np.meshgrid(a, a)
    keep = (u + v) <= 1.0
    u, v = u[keep], v[keep]
    w = 1.0 - u - v
    pts = w[:, None] * tri[0] + u[:, None] * tri[1] + v[:, None] * tri[2]
    ijk = np.floor((pts - origin) / pitch).astype(np.int64)
    ok = np.all((ijk >= 0) & (ijk < np.array(M.shape)), axis=1)
    ijk = ijk[ok]
    if ijk.size:
        M[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = cls


def interior_air(M):
    """Interior mask for a single floor. The mesh has a floor slab (dense low
    layers) and room-height walls/façade but no ceiling slab, so:
      1. find the floor slab (dense layers) → room band starts just above it;
      2. project the room-height structure to an X-Z footprint (façade + walls);
      3. fill that footprint's enclosed holes → the open floor area;
      4. interior = air inside that footprint, at room heights.
    Filling holes in the façade ring is robust to the open top and to door gaps
    (a 1-voxel closing seals rasterization pinholes; the façade is the real
    enclosure). Returns (inside, room_lo, room_hi)."""
    air = M == 0
    solid = ~air
    ny = M.shape[1]
    prof = solid.mean(axis=(0, 2))                          # per-height solid frac
    slab_idx = np.nonzero(prof > 0.5)[0]                    # dense floor/ceiling layers

    if slab_idx.size >= 2:
        # A detailed mesh has BOTH a floor slab and a ceiling slab, so "the last dense
        # layer" is the ceiling, not the floor. The occupiable room is the largest
        # vertical GAP between dense slabs; that identification works for any number of
        # slabs (floor / carpet / ceiling / plenum) and in either order.
        k = int(np.argmax(np.diff(slab_idx)))
        room_lo = int(slab_idx[k]) + 1
        room_hi = int(slab_idx[k + 1]) - 1
    elif slab_idx.size == 1:
        room_lo = int(slab_idx[0]) + 1
        room_hi = int(np.nonzero(prof > 0.02)[0].max()) if (prof > 0.02).any() else ny - 1
    else:
        room_lo = 1
        room_hi = int(np.nonzero(prof > 0.02)[0].max()) if (prof > 0.02).any() else ny - 1
    if room_hi <= room_lo:                                  # degenerate -> open top
        room_hi = int(np.nonzero(prof > 0.02)[0].max()) if (prof > 0.02).any() else ny - 1
    wall2d = solid[:, room_lo:room_hi + 1, :].any(axis=1)   # (nx,nz) façade+partitions
    filled = ndimage.binary_fill_holes(ndimage.binary_closing(wall2d, iterations=1))
    interior2d = filled & ~wall2d                           # enclosed open floor area
    band = np.zeros(ny, bool)
    band[room_lo:room_hi + 1] = True
    inside = air & interior2d[:, None, :] & band[None, :, None]
    return inside, room_lo, room_hi


def seal_ceiling(M, room_hi, cls=2, thickness=1):
    """Add a ceiling slab above the room band.

    The source meshes model a floor slab and room-height walls but no ceiling, so the
    top grid layer comes out 100% air. For a ceiling-mounted AP the ceiling bounce is
    one of the two dominant specular paths, so an open top is not a small error --
    that path is simply missing. Only fills where the footprint is enclosed, so the
    slab follows the building outline rather than capping the whole bounding box.
    """
    ny = M.shape[1]
    y0 = min(int(room_hi) + 1, ny - 1)
    y1 = min(y0 + int(thickness), ny)
    if y0 >= ny:
        return M, 0
    # If the mesh already models a ceiling (dense layer just above the room), leave it
    # alone -- overwriting it would replace real material classes with a flat slab.
    if float((M[:, y0, :] != 0).mean()) > 0.5:
        return M, 0
    footprint = ndimage.binary_fill_holes(
        ndimage.binary_closing((M != 0).any(axis=1), iterations=1))
    added = 0
    for y in range(y0, y1):
        layer = M[:, y, :]
        fill = footprint & (layer == 0)
        layer[fill] = cls
        added += int(fill.sum())
    return M, added


def reclass_floor_slab(M, room_lo, cls=2, layers=None):
    """Reclassify the floor slab to structural concrete.

    A slab tagged drywall carries eps' ~2.9 instead of ~5.24, which gives the wrong
    reflection coefficient for the floor bounce -- the other dominant indoor path.
    """
    hi = int(room_lo) if layers is None else min(int(layers), int(room_lo))
    sub = M[:, :hi, :]
    changed = int(((sub != 0) & (sub != cls)).sum())
    sub[sub != 0] = cls
    return M, changed


def tx_clearance(inside, cells=2):
    """Erode the interior mask so transmitters cannot be placed inside/touching walls.

    Without this `valid_tx_mask` is identical to `inside_mask`, which lets the dataset
    sample Tx positions flush against (or inside) structure.
    """
    if cells <= 0:
        return inside.copy()
    eroded = ndimage.binary_erosion(inside, iterations=int(cells))
    return eroded if eroded.any() else inside.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=float, default=None,
                    help="target cubic voxel size in metres (default: manifest)")
    ap.add_argument("--obj", default=None,
                    help="override source .obj (default: manifest source_obj)")
    ap.add_argument("--material-map", choices=["manifest", "sandbox"], default="manifest",
                    help="'sandbox' uses sandbox_material_map.py (253-material rules "
                         "+ people/prop exclusion) instead of manifest obj_material_map")
    ap.add_argument("--seal-ceiling", action="store_true",
                    help="add a ceiling slab above the room band (fixes the open top)")
    ap.add_argument("--ceiling-class", type=int, default=2)
    ap.add_argument("--floor-slab-class", type=int, default=None,
                    help="reclassify the floor slab to this class (2 = concrete)")
    ap.add_argument("--tx-clearance", type=int, default=0,
                    help="erode valid_tx_mask by N voxels off walls")
    ap.add_argument("--max-faces", type=int, default=None,
                    help="rasterize only the first N triangles (fast pipeline smoke test)")
    ap.add_argument("--out", default=None,
                    help="write grids+manifest to this directory instead of overwriting "
                         "the live scene (recommended when re-voxelizing)")
    args = ap.parse_args()
    out_dir = Path(args.out).expanduser().resolve() if args.out else HERE
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((HERE / "manifest_3d.json").read_text())
    obj_path = Path(args.obj).expanduser().resolve() if args.obj else ROOT / manifest["source_obj"]
    cell_m = args.cell or manifest["target_cell_size_m"]

    classify_fn = None
    if args.material_map == "sandbox":
        sys.path.insert(0, str(HERE))
        import sandbox_material_map as SMM
        classify_fn = SMM.classify
        print("material map: sandbox_material_map.py "
              f"({len(SMM.RULES)} rules, {len(SMM.EXCLUDE)} exclusions)")

    print(f"parsing {obj_path.name} ...")
    verts, faces = parse_obj(obj_path)
    print(f"  {len(verts)} verts, {len(faces)} triangles")

    # ---- scale: register OBJ X-span to the floor-plan width -----------------
    lo, hi = verts.min(0), verts.max(0)
    span = hi - lo
    sref = manifest["scale_reference"]
    d_real_x = sref["floorplan_w_px"] * sref["m_per_px"]     # metres
    m_per_unit = d_real_x / span[0]
    pitch = cell_m / m_per_unit                              # OBJ units / voxel
    nx, ny, nz = (np.ceil(span / pitch).astype(int) + 1).tolist()
    ceiling_m = span[1] * m_per_unit
    print(f"  m_per_unit={m_per_unit:.6f}  cell={cell_m} m  grid={nx}x{ny}x{nz}"
          f"  ceiling={ceiling_m:.2f} m")

    # ---- rasterize, ascending material priority (higher overwrites) --------
    rules = manifest["obj_material_map"]
    priority = manifest["material_priority"]                 # low -> high
    rank = {c: i for i, c in enumerate(priority)}
    if args.max_faces:
        faces = faces[:args.max_faces]
        print(f"  --max-faces: rasterizing only {len(faces)} triangles (smoke test)")
    by_class, n_excluded = {}, 0
    for i0, i1, i2, name in faces:
        if classify_fn is not None:
            cls, _why = classify_fn(name)
            if cls is None:                       # person / desk prop -> not structure
                n_excluded += 1
                continue
        else:
            cls = class_for(name, rules)
        by_class.setdefault(cls, []).append((i0, i1, i2))
    if n_excluded:
        print(f"  excluded {n_excluded} triangles (people/props, not building material)")

    M = np.zeros((nx, ny, nz), np.int8)
    origin = lo
    for cls in sorted(by_class, key=lambda c: rank.get(c, 0)):
        if cls == 0:
            continue
        for (i0, i1, i2) in by_class[cls]:
            rasterize_triangle(verts[[i0, i1, i2]], cls, M, origin, pitch)
    print(f"  rasterized {sum(len(v) for v in by_class.values())} triangles")

    # ---- masks -------------------------------------------------------------
    inside, room_lo, room_hi = interior_air(M)

    # ---- scene-fidelity fixes (opt-in; see tests3d/test_scene_sanity.py) ----
    if args.floor_slab_class is not None:
        M, n = reclass_floor_slab(M, room_lo, cls=args.floor_slab_class)
        print(f"  floor slab -> class {args.floor_slab_class}: {n} voxels reclassified")
    if args.seal_ceiling:
        M, n = seal_ceiling(M, room_hi, cls=args.ceiling_class)
        print(f"  sealed ceiling (class {args.ceiling_class}): {n} voxels added")
        inside, room_lo, room_hi = interior_air(M)      # re-derive after sealing

    valid_tx = tx_clearance(inside, args.tx_clearance)
    if args.tx_clearance:
        print(f"  valid_tx clearance {args.tx_clearance} voxels: "
              f"{inside.sum()} -> {valid_tx.sum()}")

    # ---- save --------------------------------------------------------------
    np.save(out_dir / "material_grid.npy", M)
    np.save(out_dir / "inside_mask.npy", inside)
    np.save(out_dir / "valid_tx_mask.npy", valid_tx)

    tot = M.size
    hist = {int(c): round(float((M == c).sum()) / tot, 4)
            for c in range(len(manifest["materials"]))}
    manifest.update(
        grid_shape=[nx, ny, nz],
        cell_size_m=round(cell_m, 6),
        m_per_unit=round(m_per_unit, 8),
        origin_units=[round(float(x), 4) for x in origin],
        ceiling_height_m=round(ceiling_m, 3),
        room_band=[room_lo, room_hi],
    )
    if args.obj:
        manifest["source_obj"] = str(obj_path)
    if args.material_map == "sandbox":
        manifest.setdefault("provenance", {})["material_map"] = "sandbox_material_map.py"
    (out_dir / "manifest_3d.json").write_text(json.dumps(manifest, indent=2))

    print("class fractions:", hist)
    print(f"interior air: {inside.sum()} voxels "
          f"({100*inside.sum()/tot:.1f}% of volume)")
    print(f"valid Tx    : {valid_tx.sum()} voxels")
    print(f"saved material_grid.npy, inside_mask.npy, valid_tx_mask.npy "
          f"and updated manifest_3d.json")


if __name__ == "__main__":
    main()
