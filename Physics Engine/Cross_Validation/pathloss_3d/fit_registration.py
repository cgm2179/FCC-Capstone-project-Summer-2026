#!/usr/bin/env python3
"""PR #27 — fit a proper px→voxel registration for the 3D scene.

`registration_3d.json` stores a scale only (no offset/rotation) and was fit against the
256×448 SIM mask, so mapping the measured records (1150×515 floor-plan frame) onto the 3D
voxel footprint (262×132) reaches only IoU≈0.41 and scrambles validation. This fits a full
2-D affine px→vox by matching the two inside footprints via image moments (PCA): align the
principal axes + centroids + extents, disambiguate the 4 axis-sign flips by IoU, then polish
the translation. Saves `registration_fit.json` (a 2×3 affine used by validate_3d.py).

usage: python fit_registration.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SIM3D = REPO / "Physics Engine" / "3D Map Physics" / "SIM V1 3D"
OUTSIDE_2D = REPO / "Physics Engine" / "2D" / "STEP_2" / "outside_mask.npy"
OUT = HERE / "registration_fit.json"


def pca(pts):
    """pts (N,2) → centroid (2,), eigvals (major,minor), eigvecs cols (2,2)."""
    c = pts.mean(0)
    X = pts - c
    cov = X.T @ X / len(pts)
    w, V = np.linalg.eigh(cov)
    o = np.argsort(w)[::-1]
    return c, w[o], V[:, o]


def rasterize(vx, vz, NX, NZ):
    m = np.zeros((NX, NZ), bool)
    ix = np.clip(np.round(vx).astype(int), 0, NX - 1)
    iz = np.clip(np.round(vz).astype(int), 0, NZ - 1)
    m[ix, iz] = True
    return m


def fit():
    ins3 = np.load(SIM3D / "inside_mask.npy")
    foot3 = ins3.any(axis=1)                       # (NX,NZ) indoor AIR at any height
    NX, NZ = foot3.shape
    # Match like-for-like: 2D INDOOR AIR (walkable rooms), not ~outside (which is the whole
    # rectangle minus the border). Both masks are then ~room-shaped and comparable.
    out2 = np.load(OUTSIDE_2D)
    mat2 = np.load(REPO / "Physics Engine" / "2D" / "STEP_1" / "material_grid.npy")
    ins2 = (mat2 == 0) & (~out2)                   # (515,1150) indoor air (records frame)
    print(f"mask coverage — 2D indoor-air {100*ins2.mean():.1f}%  3D air-footprint {100*foot3.mean():.1f}%")
    ys, xs = np.nonzero(ins2)
    p2 = np.stack([xs, ys], 1).astype(float)       # (px_x, px_y)
    ii = np.argwhere(foot3)
    p3 = ii.astype(float)                          # (vx, vz)

    c2, w2, V2 = pca(p2)
    c3, w3, V3 = pca(p3)
    scale = np.sqrt(w3 / w2)

    best = None
    for f0 in (1, -1):
        for f1 in (1, -1):
            A = V3 @ np.diag(scale * np.array([f0, f1])) @ V2.T
            t = c3 - A @ c2
            v = (A @ p2.T).T + t
            iou = _iou(rasterize(v[:, 0], v[:, 1], NX, NZ), foot3)
            if best is None or iou > best[0]:
                best = (iou, A, t, (f0, f1))
    iou, A, t, flips = best

    # polish translation by a small local search (±6 vox)
    for _ in range(2):
        base = _iou(rasterize((A @ p2.T).T[:, 0] + t[0], (A @ p2.T).T[:, 1] + t[1], NX, NZ), foot3)
        vt = (A @ p2.T).T
        for dtx in range(-6, 7):
            for dtz in range(-6, 7):
                io = _iou(rasterize(vt[:, 0] + t[0] + dtx, vt[:, 1] + t[1] + dtz, NX, NZ), foot3)
                if io > base:
                    base, best_dt = io, (dtx, dtz)
        if base > iou:
            t = t + np.array(best_dt, float); iou = base
        else:
            break

    affine = [[float(A[0, 0]), float(A[0, 1]), float(t[0])],
              [float(A[1, 0]), float(A[1, 1]), float(t[1])]]
    OUT.write_text(json.dumps({
        "version": "regfit-v1",
        "note": "px(floor-plan 1150x515) -> voxel (NX,NZ): vx=a*px+b*py+c, vz=d*px+e*py+f. "
                "Moment/PCA fit of the 2D inside footprint to the 3D voxel footprint, "
                "flip-disambiguated + translation-polished by IoU.",
        "grid_shape": [int(NX), int(NZ)], "flips": list(flips),
        "footprint_iou": round(float(iou), 3),
        "affine_px_to_vox": affine,
    }, indent=2))
    print(f"fitted px->vox affine  footprint IoU={iou:.3f}  (scale-only was ~0.41)  flips={flips}")
    print(f"  affine = {affine}")
    print(f"  -> {OUT}")
    return affine, iou


def _iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


if __name__ == "__main__":
    fit()
