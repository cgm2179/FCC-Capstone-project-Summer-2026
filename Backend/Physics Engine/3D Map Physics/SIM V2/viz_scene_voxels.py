#!/usr/bin/env python3
"""
viz_scene_voxels.py — three honest views of "what the 3-D grid the surrogate sees looks like":

  (a) the REAL CAD, triangles colored by material (true full detail, no voxel loss);
  (b) the 0.3 m scene voxel grid (the coarse whole-floor medium — the old blocky view);
  (c) a room-sized crop voxelized at λ/10 + adaptive super — the conformal effective εr the
      surrogate actually consumes (~5 mm geometry detail), the antidote to the blocky grid.

1 mm whole-floor is impossible (~1.6e16 cells); the wave cell is wavelength-bound, and detail
comes from supersampling into the effective medium (see voxelize_conformal.auto_super).

  python viz_scene_voxels.py --band LTE_B71_617 --crop 30 36 18 24 --out renders/scene_voxels.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import bootstrap as B
import cad_materials as CM
import voxelize_conformal as VC
from bands_v3 import get

HERE = Path(__file__).resolve().parent
# 6-class palette: 0 air · 1 drywall · 2 concrete · 3 metal · 4 wood · 5 glass
CLASS_COLORS = np.array([[1, 1, 1], [0.85, 0.72, 0.53], [0.6, 0.6, 0.62],
                         [0.20, 0.45, 0.85], [0.55, 0.35, 0.18], [0.35, 0.8, 0.85]])
CLASS_NAMES = ["air", "drywall", "concrete", "metal", "wood", "glass"]


def _sub(rng, n, k):
    return rng.choice(n, min(k, n), replace=False)


def _orient(ax, ex):
    """Plot with height (Y) as the vertical axis and true proportions. `ex` = (X,Y,Z) metres."""
    ax.set_xlim(0, ex[0]); ax.set_ylim(0, ex[2]); ax.set_zlim(0, ex[1])
    ax.set_box_aspect((ex[0], ex[2], max(ex[1], 0.15 * max(ex[0], ex[2]))))  # floor is wide/flat


def view_cad(ax, max_tris=120000):
    V, F, TM, names = VC._parsed(VC.OBJ)
    mpu = float(__import__("json").loads(B.MANIFEST.read_text())["m_per_unit"])
    Vm = (V - V.min(0)) * mpu                                        # metres, Y up
    tri_cls = np.array([CM.class_of(nm) for nm in names])[TM]        # per-triangle 6-class id
    keep = tri_cls > 0                                               # drop air/excluded tris
    F, tri_cls = F[keep], tri_cls[keep]
    sel = _sub(np.random.default_rng(0), len(F), max_tris)
    polys = Vm[F[sel]][:, :, [0, 2, 1]]                              # (x, z→depth, y→height)
    ax.add_collection3d(Poly3DCollection(polys, facecolors=CLASS_COLORS[tri_cls[sel]],
                                         edgecolors="none", alpha=0.5, linewidths=0))
    _orient(ax, Vm.max(0))
    ax.set_title(f"(a) Real CAD · materials\n({len(F):,} solid tris, {min(max_tris,len(F))//1000}k shown)",
                 fontsize=10)


def _scatter_grid(ax, mask, values, cmap, cell_m, title, sfrac=None, rng=None, vmin=None, vmax=None):
    ijk = np.argwhere(mask)
    v = values[mask]
    if sfrac and len(ijk) > sfrac:
        i = (rng or np.random.default_rng(0)).choice(len(ijk), sfrac, replace=False)
        ijk, v = ijk[i], v[i]
    p = ijk * cell_m
    ax.scatter(p[:, 0], p[:, 2], p[:, 1], c=v, cmap=cmap, s=2, marker="s",   # x, z→depth, y→height
               alpha=0.5, linewidths=0, vmin=vmin, vmax=vmax)
    _orient(ax, np.array(mask.shape) * cell_m)
    ax.set_title(title, fontsize=10)
    return len(ijk)


def view_coarse(ax, max_pts=120000):
    from matplotlib.colors import ListedColormap
    grid = np.load(B.MATERIAL_GRID)                                  # 0.3 m scene (int8, 6-class)
    cs = float(__import__("json").loads(B.MANIFEST.read_text())["cell_size_m"])
    n = _scatter_grid(ax, grid > 0, grid.astype(float), ListedColormap(CLASS_COLORS), cs,
                      f"(b) Scene voxel grid @ {cs*100:.0f} cm  ({grid.shape})",
                      sfrac=max_pts, vmin=0, vmax=5)
    print(f"  (b) coarse grid {grid.shape} @ {cs*100:.0f}cm  non-air cells shown≈{n}")


def view_fine(ax, band_label, crop, max_pts=120000, super="auto"):
    ceil = float(__import__("json").loads(B.MANIFEST.read_text())["ceiling_height_m"])
    x0, x1, z0, z1 = crop
    band = get(band_label); h = band.cell_size_m(10.0)
    sup = VC.auto_super(h) if super == "auto" else int(super)
    o = VC.bake(band_label, crop_m=(x0, x1, 0.0, ceil, z0, z1), cell_m=h, super=super)
    eps = o["epsr_eff"]
    n = _scatter_grid(ax, eps > 1.05, eps, "viridis", h,
                      f"(c) Conformal εr @ λ/10 = {h*100:.1f} cm, super={sup}\n"
                      f"{eps.shape} — what the surrogate consumes", sfrac=max_pts)
    print(f"  (c) fine crop {eps.shape} @ {h*100:.1f}cm super={sup}  "
          f"εr[{eps.min():.2f},{eps.max():.2f}]  cells={eps.size/1e6:.2f}M  shown≈{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--crop", type=float, nargs=4, default=[30, 36, 18, 24],
                    metavar=("X0", "X1", "Z0", "Z1"), help="room crop in metres (X,Z)")
    ap.add_argument("--fine-super", default="auto",
                    help='super for the fine crop; "auto" (=8 @617) or an int (lower=faster on CPU)')
    ap.add_argument("--out", default=str(HERE / "renders" / "scene_voxels.png"))
    a = ap.parse_args()
    fine_super = a.fine_super if a.fine_super == "auto" else int(a.fine_super)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(19, 6.2))
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.set_zlabel("height y (m)")
        ax.view_init(elev=22, azim=-60)
    print("rendering (c) fine crop …"); view_fine(fig.axes[2], a.band, a.crop, super=fine_super)
    print("rendering (a) CAD …"); view_cad(fig.axes[0])
    print("rendering (b) coarse grid …"); view_coarse(fig.axes[1])
    # material legend
    from matplotlib.patches import Patch
    fig.legend([Patch(color=CLASS_COLORS[c]) for c in range(1, 6)], CLASS_NAMES[1:],
               loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle("SIM V2 3-D — what the surrogate sees: real CAD → 0.3 m grid → λ/10 conformal εr",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
