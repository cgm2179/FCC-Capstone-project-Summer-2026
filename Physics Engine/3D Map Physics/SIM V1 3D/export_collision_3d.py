#!/usr/bin/env python3
"""
export_collision_3d.py — building-shell static collision boxes for the SIM V1
3D antenna-placement sandbox (web/antenna_sandbox.html).

material_grid.npy has ~99k solid (non-air) voxels — one static rigid body per
voxel is unnecessary for a browser physics engine (cannon-es/Rapier) and would
make broadphase pointlessly expensive. This does a 2-pass greedy merge of
solid voxels into axis-aligned boxes:
  pass 1: run-length-encode along X for every (y,z) column         -> ~10.8k boxes
  pass 2: merge adjacent Z rows that share the same (y, x0, x1)    -> ~3.3k boxes
(measured on the current 262x11x118 grid). Collision only cares whether a
voxel is solid, not which material class it is, so classes are merged
together (unlike measure_nominal_widths_3d in physics_3d.py, which measures
per-class thickness for the RF model).

Boxes are emitted in meters in the same corner-anchored frame
Construct_Transmitter_3D.py uses for transmitter position_m: box (i,j,k) voxel
range maps to [i0*cell_size_m, i1*cell_size_m) etc., Y up. A thin fallback
ground pad is added at y=0 since the voxelized floor slab (layer 0 is only
~1% solid — see layer histogram) doesn't fully cover the building footprint,
so a dropped object outside real walls still lands on something instead of
free-falling forever.

usage: python "SIM V1 3D/export_collision_3d.py"
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def merge_solid_voxels(solid):
    """2-pass greedy merge (X-run then Z-run) of a 3-D boolean solid mask into
    axis-aligned index-space boxes [x0,x1) x [y,y+1) x [z0,z1)."""
    nx, ny, nz = solid.shape
    boxes = []
    for y in range(ny):
        # pass 1: RLE along X per z -> {(x0, x1): [z, z, ...]}
        spans = {}
        for z in range(nz):
            col = solid[:, y, z]
            i = 0
            while i < nx:
                if not col[i]:
                    i += 1
                    continue
                j = i
                while j < nx and col[j]:
                    j += 1
                spans.setdefault((i, j), []).append(z)
                i = j
        # pass 2: merge consecutive z for the same (x0, x1) span
        for (x0, x1), zs in spans.items():
            zs.sort()
            i = 0
            while i < len(zs):
                j = i
                while j + 1 < len(zs) and zs[j + 1] == zs[j] + 1:
                    j += 1
                boxes.append((x0, x1, y, y + 1, zs[i], zs[j] + 1))
                i = j + 1
    return boxes


def boxes_to_meters(boxes, cell_size_m):
    """[x0,x1,y0,y1,z0,z1) voxel-index boxes -> [cx,cy,cz,hx,hy,hz] meter boxes
    (center + half-extents), corner-anchored at (0,0,0)."""
    out = []
    for x0, x1, y0, y1, z0, z1 in boxes:
        cx, cy, cz = (x0 + x1) / 2.0 * cell_size_m, (y0 + y1) / 2.0 * cell_size_m, (z0 + z1) / 2.0 * cell_size_m
        hx, hy, hz = (x1 - x0) / 2.0 * cell_size_m, (y1 - y0) / 2.0 * cell_size_m, (z1 - z0) / 2.0 * cell_size_m
        out.append([round(cx, 4), round(cy, 4), round(cz, 4), round(hx, 4), round(hy, 4), round(hz, 4)])
    return out


def main():
    man = json.loads((HERE / "manifest_3d.json").read_text())
    M = np.load(HERE / "material_grid.npy")
    cell = man["cell_size_m"]
    nx, ny, nz = M.shape

    solid = M != 0
    boxes = merge_solid_voxels(solid)
    meter_boxes = boxes_to_meters(boxes, cell)

    ground_pad_m = 0.05
    ground_box = [nx * cell / 2.0, -ground_pad_m / 2.0, nz * cell / 2.0,
                  nx * cell / 2.0, ground_pad_m / 2.0, nz * cell / 2.0]
    meter_boxes.append([round(v, 4) for v in ground_box])

    out = dict(
        cell_size_m=cell,
        grid_shape=man["grid_shape"],
        extent_m=[round(nx * cell, 3), round(ny * cell, 3), round(nz * cell, 3)],
        box_format=["cx", "cy", "cz", "half_x", "half_y", "half_z"],
        n_boxes=len(meter_boxes),
        boxes=meter_boxes,
    )
    (HERE / "web").mkdir(exist_ok=True)
    p = HERE / "web" / "collision_3d.js"
    p.write_text("window.SIM3D_COLLISION = " + json.dumps(out) + ";\n")
    print(f"wrote {p} ({len(meter_boxes)} boxes, {p.stat().st_size / 1e3:.0f} KB) "
          f"from {int(solid.sum())} solid voxels")


if __name__ == "__main__":
    main()
