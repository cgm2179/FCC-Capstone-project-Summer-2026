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
    slab = prof > 0.5                                       # dense floor/ceiling layers
    room_lo = int(np.nonzero(slab)[0].max()) + 1 if slab.any() else 1
    room_hi = int(np.nonzero(prof > 0.02)[0].max()) if (prof > 0.02).any() else ny - 1
    wall2d = solid[:, room_lo:room_hi + 1, :].any(axis=1)   # (nx,nz) façade+partitions
    filled = ndimage.binary_fill_holes(ndimage.binary_closing(wall2d, iterations=1))
    interior2d = filled & ~wall2d                           # enclosed open floor area
    band = np.zeros(ny, bool)
    band[room_lo:room_hi + 1] = True
    inside = air & interior2d[:, None, :] & band[None, :, None]
    return inside, room_lo, room_hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=float, default=None,
                    help="target cubic voxel size in metres (default: manifest)")
    args = ap.parse_args()

    manifest = json.loads((HERE / "manifest_3d.json").read_text())
    obj_path = ROOT / manifest["source_obj"]
    cell_m = args.cell or manifest["target_cell_size_m"]

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
    by_class = {}
    for i0, i1, i2, name in faces:
        by_class.setdefault(class_for(name, rules), []).append((i0, i1, i2))

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
    valid_tx = inside          # any interior room voxel is a candidate transmitter

    # ---- save --------------------------------------------------------------
    np.save(HERE / "material_grid.npy", M)
    np.save(HERE / "inside_mask.npy", inside)
    np.save(HERE / "valid_tx_mask.npy", valid_tx)

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
    (HERE / "manifest_3d.json").write_text(json.dumps(manifest, indent=2))

    print("class fractions:", hist)
    print(f"interior air: {inside.sum()} voxels "
          f"({100*inside.sum()/tot:.1f}% of volume)")
    print(f"valid Tx    : {valid_tx.sum()} voxels")
    print(f"saved material_grid.npy, inside_mask.npy, valid_tx_mask.npy "
          f"and updated manifest_3d.json")


if __name__ == "__main__":
    main()
