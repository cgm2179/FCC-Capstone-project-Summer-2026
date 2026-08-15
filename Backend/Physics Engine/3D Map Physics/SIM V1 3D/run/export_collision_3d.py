#!/usr/bin/env python3
"""
export_collision_3d.py — building-shell static collision boxes for the SIM V1
3D antenna-placement sandbox (web/antenna_sandbox.html).

material_grid.npy has ~160k solid (non-air) voxels — one static rigid body per
voxel is unnecessary for a browser physics engine (cannon-es/Rapier) and would
make broadphase pointlessly expensive. This does a 2-pass greedy merge of
solid voxels into axis-aligned boxes:
  pass 1: run-length-encode along X for every (y,z) column
  pass 2: merge adjacent Z rows that share the same (y, x0, x1)

MERGED PER CLASS, NOT PER SOLID/AIR
This used to merge a boolean solid mask, which threw the material class away —
so the browser had nothing to colour by and drew the whole building one flat
grey. Runs are now broken at class boundaries and every box carries its class,
which is what lets the 3-D tab render drywall, concrete, the core, furniture
and glass as distinct materials. The cost is a modest box-count increase
(concrete no longer merges into the drywall run beside it); the benefit is that
the render finally shows the scene the RF engine is actually solving on.

The palette travels with the boxes so the browser needs no second lookup, and
comes from manifest_3d.json — which mirrors the 2D manifest, so the floor plan,
the 2D sim and the 3D sim all render the same 6-class scheme.

Boxes are emitted in meters in the same corner-anchored frame
Construct_Transmitter_3D.py uses for transmitter position_m: box (i,j,k) voxel
range maps to [i0*cell_size_m, i1*cell_size_m) etc., Y up. A thin fallback
ground pad is added at y=0 since the voxelized floor slab (layer 0 is only
~1% solid — see layer histogram) doesn't fully cover the building footprint,
so a dropped object outside real walls still lands on something instead of
free-falling forever.

usage:
  python "SIM V1 3D/export_collision_3d.py"
  python "SIM V1 3D/export_collision_3d.py" --sim-dir city_demo/NoMa_DC_tile \
      --out web/collision_outdoor_3d.js --global SIM3D_COLLISION_OUTDOOR
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # SIM V1 3D root (siblings live there)
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent


def merge_voxels_by_class(M):
    """2-pass greedy merge (X-run then Z-run) of the material grid into
    axis-aligned index-space boxes, ONE CLASS PER BOX.

    Returns [(x0, x1, y0, y1, z0, z1, cls), ...]. A run stops at a class change
    as well as at air, so a concrete column beside a drywall partition stays two
    boxes instead of collapsing into one anonymous solid.
    """
    nx, ny, nz = M.shape
    boxes = []
    for y in range(ny):
        # pass 1: RLE along X per z, keyed by (span, class) -> [z, z, ...]
        spans = {}
        for z in range(nz):
            col = M[:, y, z]
            i = 0
            while i < nx:
                c = int(col[i])
                if c == 0:
                    i += 1
                    continue
                j = i
                while j < nx and int(col[j]) == c:
                    j += 1
                spans.setdefault((i, j, c), []).append(z)
                i = j
        # pass 2: merge consecutive z for the same (x0, x1, class)
        for (x0, x1, c), zs in spans.items():
            zs.sort()
            i = 0
            while i < len(zs):
                j = i
                while j + 1 < len(zs) and zs[j + 1] == zs[j] + 1:
                    j += 1
                boxes.append((x0, x1, y, y + 1, zs[i], zs[j] + 1, c))
                i = j + 1
    return boxes


def boxes_to_meters(boxes, cell_size_m):
    """voxel-index boxes -> ([cx,cy,cz,hx,hy,hz] meter boxes, [class, ...])
    (center + half-extents), corner-anchored at (0,0,0)."""
    out, classes = [], []
    for x0, x1, y0, y1, z0, z1, cls in boxes:
        cx, cy, cz = (x0 + x1) / 2.0 * cell_size_m, (y0 + y1) / 2.0 * cell_size_m, (z0 + z1) / 2.0 * cell_size_m
        hx, hy, hz = (x1 - x0) / 2.0 * cell_size_m, (y1 - y0) / 2.0 * cell_size_m, (z1 - z0) / 2.0 * cell_size_m
        out.append([round(cx, 4), round(cy, 4), round(cz, 4), round(hx, 4), round(hy, 4), round(hz, 4)])
        classes.append(int(cls))
    return out, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", default=str(HERE),
                    help="directory containing manifest_3d/material_grid")
    ap.add_argument("--out", default=str(HERE / "web" / "collision_3d.js"),
                    help="output JS file")
    ap.add_argument("--global", dest="global_name", default="SIM3D_COLLISION",
                    help="window global to assign")
    args = ap.parse_args()

    sim_dir = Path(args.sim_dir).expanduser().resolve()
    man = json.loads((sim_dir / "manifest_3d.json").read_text())
    M = np.load(sim_dir / "material_grid.npy")
    cell = man["cell_size_m"]
    nx, ny, nz = M.shape

    solid = M != 0
    boxes = merge_voxels_by_class(M)
    meter_boxes, classes = boxes_to_meters(boxes, cell)

    # Ground pad: the voxelized floor slab does not cover the whole footprint, so a
    # dropped object outside real walls would free-fall forever. Class -1 marks it as
    # scaffolding rather than building, so the renderer can hide it.
    ground_pad_m = 0.05
    ground_box = [nx * cell / 2.0, -ground_pad_m / 2.0, nz * cell / 2.0,
                  nx * cell / 2.0, ground_pad_m / 2.0, nz * cell / 2.0]
    meter_boxes.append([round(v, 4) for v in ground_box])
    classes.append(-1)

    per_class = {}
    for c in classes:
        per_class[c] = per_class.get(c, 0) + 1
    names = {m["id"]: m["name"] for m in man["materials"]}

    out = dict(
        cell_size_m=cell,
        grid_shape=man["grid_shape"],
        extent_m=[round(nx * cell, 3), round(ny * cell, 3), round(nz * cell, 3)],
        box_format=["cx", "cy", "cz", "half_x", "half_y", "half_z"],
        n_boxes=len(meter_boxes),
        boxes=meter_boxes,
        # one class per box; -1 = the synthetic ground pad, not part of the building
        classes=classes,
        materials=[{"id": m["id"], "name": m["name"], "color": m.get("color")}
                   for m in man["materials"]],
        class_note="classes[i] is the material id of boxes[i]; -1 = synthetic ground pad",
    )
    p = Path(args.out).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"window.{args.global_name} = " + json.dumps(out) + ";\n")
    print(f"wrote {p} ({len(meter_boxes)} boxes, {p.stat().st_size / 1e3:.0f} KB) "
          f"from {int(solid.sum())} solid voxels")
    for c in sorted(per_class):
        label = "ground pad" if c < 0 else names.get(c, f"class {c}")
        print(f"  {label:20s} {per_class[c]:6d} boxes")


if __name__ == "__main__":
    main()
