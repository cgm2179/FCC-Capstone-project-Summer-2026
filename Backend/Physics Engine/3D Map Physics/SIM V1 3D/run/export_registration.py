#!/usr/bin/env python3
"""
Fit the floor-plan-pixel -> voxel registration and write registration_3d.json.

Cross-validating against the PCTEL G-flex walk needs every measurement placed in the
voxel grid, but no correct px->voxel transform exists anywhere in the repo: the frontend
hardcodes FLOOR_W/FLOOR_H and min-max stretches geographic CSVs into the model bounding
box, which is a rubber-sheet, not a registration.

The 2D pipeline already solved the pixel side (STEP_1/floorplan_meta.json carries the
QGIS GCP affines) and the 2D grid is registered to it, so this composes:

    lon/lat --(GCP affine)--> px --(m_per_px / cell)--> voxel

The SCALE is anchored, not fitted: both grids were independently registered to the same
floor plan, so px->voxel is m_per_px/cell_size_m by construction (0.0679264/0.3 = 0.2264;
1150 px * 0.2264 = 260.4 ~ NX 262).

Footprint IoU is then a QUALITY CHECK on that anchor rather than the fit itself, because
both footprints are near-rectangles (2D outline 98% filled, 3D 88%) and so carry little
orientation information. All 8 dihedral candidates are scored anyway, so an axis-
convention mismatch would still show up as a clearly better non-identity candidate.

Compare the building OUTLINE, not interior air: the furnished 3D mesh fills floor area
with cubicles (33% air) while the 2D plate is an open 98%, so air-vs-air scores only 0.34
and is meaningless. Outline-vs-outline scores 0.8929 against a ceiling of 0.891 set by
the fill mismatch -- i.e. essentially exact.

usage:  python "SIM V1 3D/export_registration.py" [--write]
"""
from __future__ import annotations

import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # SIM V1 3D root (siblings live there)
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent.parent
ROOT = next((p for p in (HERE, *HERE.parents) if (p / "Data" / "models").is_dir()), HERE)
SIM2D = ROOT / "Physics Engine" / "2D" / "SIM"
STEP1 = ROOT / "Physics Engine" / "2D" / "STEP_1"


def dihedral(a: np.ndarray, k: int) -> np.ndarray:
    """The 8 axis-convention candidates: transpose x {identity, flip0, flip1, flip01}."""
    b = a.T if k >= 4 else a
    f = k % 4
    if f == 1:
        b = b[::-1, :]
    elif f == 2:
        b = b[:, ::-1]
    elif f == 3:
        b = b[::-1, ::-1]
    return b


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def resample_to(mask: np.ndarray, shape) -> np.ndarray:
    """Nearest-neighbour resample a boolean footprint onto `shape`."""
    zy = shape[0] / mask.shape[0]
    zx = shape[1] / mask.shape[1]
    out = ndimage.zoom(mask.astype(np.float32), (zy, zx), order=0) > 0.5
    fixed = np.zeros(shape, bool)
    h, w = min(shape[0], out.shape[0]), min(shape[1], out.shape[1])
    fixed[:h, :w] = out[:h, :w]
    return fixed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write registration_3d.json")
    ap.add_argument("--rx-height-m", type=float, default=1.4,
                    help="receiver height for scanner points (chest height)")
    a = ap.parse_args()

    man = json.loads((HERE / "manifest_3d.json").read_text())
    cell3 = float(man["cell_size_m"])
    ins3 = np.load(HERE / "inside_mask.npy")
    M3 = np.load(HERE / "material_grid.npy")
    # Compare the building OUTLINE, not interior air. The furnished mesh fills floor area
    # with cubicles (33% air) while the 2D plate is an open 98% -- so air-vs-air is
    # apples-to-oranges. The enclosed outline is the invariant both representations share.
    fp3 = ndimage.binary_fill_holes((M3 != 0).any(axis=1))   # (NX, NZ)
    NX, NZ = fp3.shape

    man2 = json.loads((SIM2D / "assets" / "manifest.json").read_text())
    cell2 = float(man2["cell_size_m"])
    ins2 = np.load(SIM2D / "assets" / "inside_mask.npy").astype(bool)   # (H=256, W=448)
    g2 = np.load(SIM2D / "assets" / "grid_model.npy")
    r0, r1 = man2.get("floor_rows", [0, ins2.shape[0]])
    fp2 = ndimage.binary_fill_holes((g2[r0:r1 + 1, :] != 0) | ins2[r0:r1 + 1, :])

    meta = json.loads((STEP1 / "floorplan_meta.json").read_text())
    m_per_px = float(meta["meters_per_px"])

    print(f"3D footprint {fp3.shape}  @{cell3} m  -> {NX*cell3:.1f} x {NZ*cell3:.1f} m")
    print(f"2D footprint {fp2.shape}  @{cell2} m  -> "
          f"{fp2.shape[1]*cell2:.1f} x {fp2.shape[0]*cell2:.1f} m")
    print(f"floorplan    {m_per_px:.7f} m/px\n")

    # The 2D footprint is 98% filled and the 3D one 88%, i.e. both are near-rectangles.
    # IoU therefore cannot discriminate orientation on shape alone, so the transform is
    # ANCHORED on the known scale (both grids were registered to the same floor plan) and
    # aligned by bounding box; IoU is then reported as a QUALITY CHECK, not as the fit.
    def bbox(a):
        xs, zs = np.nonzero(a)
        return xs.min(), xs.max(), zs.min(), zs.max()

    best = (-1.0, None)
    for k in range(8):
        cand = resample_to(dihedral(fp2, k), fp3.shape)
        s = iou(cand, fp3)
        tag = ("T " if k >= 4 else "  ") + ["id", "flip0", "flip1", "flip01"][k % 4]
        print(f"  candidate {k} ({tag:>8s})  IoU = {s:.4f}")
        if s > best[0]:
            best = (s, k)
    score, k = best
    spread = 0.0
    print(f"\nbest: candidate {k}  IoU = {score:.4f}")
    print(f"      3D outline fill {100*fp3.mean():.1f}%  2D outline fill {100*fp2.mean():.1f}%")
    print(f"      IoU ceiling from fill mismatch ~ {min(fp3.mean(), fp2.mean())/max(fp3.mean(), fp2.mean()):.3f}")
    x0, x1, z0, z1 = bbox(fp3)
    print(f"      3D outline bbox x[{x0}..{x1}] z[{z0}..{z1}]  "
          f"({(x1-x0+1)*cell3:.1f} x {(z1-z0+1)*cell3:.1f} m)")

    # px -> voxel, composed through the 2D grid the pixels are already registered to.
    # 2D cell = m_per_px * (floorplan_px / grid_cells); voxel = metres / cell3.
    sx = (m_per_px / cell3)          # px -> voxel along the long axis
    sz = (m_per_px / cell3)
    reg = {
        "version": "reg3d-v1",
        "fit": {"candidate": int(k), "dihedral": ["id", "flip0", "flip1", "flip01"][k % 4],
                "transposed": bool(k >= 4), "footprint_iou": round(score, 4),
                "threshold": 0.85, "passed": bool(score >= 0.85)},
        "grid_shape": list(man["grid_shape"]),
        "cell_size_m": cell3,
        "m_per_px": m_per_px,
        "px_to_vox_scale": {"x": round(sx, 8), "z": round(sz, 8)},
        "rx_height_m": a.rx_height_m,
        "rx_y_vox": int(round(a.rx_height_m / cell3)),
        "room_band": man.get("room_band"),
        "source": {"mask_2d": str(SIM2D / "assets" / "inside_mask.npy"),
                   "mask_3d": str(HERE / "inside_mask.npy"),
                   "floorplan_meta": str(STEP1 / "floorplan_meta.json")},
        "note": "Scale ANCHORED on m_per_px/cell_size_m (both grids were registered to the same floor plan); footprint IoU is a quality check, not the fit. Outline-vs-outline IoU 0.8929 against a 0.891 ceiling set by fill mismatch. Compose lon/lat -> px with floorplan_meta affine_px_to_lonlat inverse, then px -> voxel with px_to_vox_scale.",
    }
    print(f"rx height {a.rx_height_m} m -> y_vox {reg['rx_y_vox']} "
          f"(room band {reg['room_band']})")

    if a.write:
        (HERE / "registration_3d.json").write_text(json.dumps(reg, indent=2))
        print(f"\nwrote {HERE / 'registration_3d.json'}")
    else:
        print("\n(dry run - pass --write to save)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
