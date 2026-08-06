#!/usr/bin/env python3
"""
validate_vs_eikonal.py — the SIM V3 -> coverage/validation bridge.

Full-wave and the production eikonal engine model different limits, so they will
never be dB-identical. But where the geometry dominates (LOS + first walls) they
should AGREE in shape, and where they DIVERGE (deep shadow, diffraction) is
exactly where the ray engine is weakest and the full-wave field is most useful.
This script quantifies that with a rank (Spearman) correlation on the SAME plane.

Convention: SceneV3 gives path LOSS in dB (higher = weaker); the FDTD RMS map is
field STRENGTH in dB (higher = stronger). We therefore correlate SceneV3 PL with
(-field_db) and expect a POSITIVE rho.

  python validate_vs_eikonal.py --band LTE_B71_617 --out out/LTE_B71_617_full
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr

import _bootstrap as B
import engine_3d as E3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--out", required=True, help="SIM V3 run dir with field_db.npy")
    ap.add_argument("--exclude-r-m", type=float, default=3.0,
                    help="drop cells within this radius of the Tx (near field)")
    args = ap.parse_args()

    run = Path(args.out)
    field_db = np.load(run / "field_db.npy")               # fine (nx,nz), strength dB
    meta = json.loads((run / "run_meta.json").read_text())
    pm = meta["plane_meta"]
    y = int(pm["y_index"])
    zoom = float(pm["zoom"])
    f_mhz = float(meta["f_mhz"])

    # --- SceneV3 path loss on the SAME plane ------------------------------
    scene, manifest = E3.load_scene()
    # Tx in coarse cell coords, recovered from the fine index we simulated
    txf = meta["tx_idx"]
    tx_cell = np.array([txf[0] / zoom, y, txf[1] / zoom], np.float32)
    print(f"SceneV3 PL: tx_cell={tx_cell.round(1).tolist()}  f≈{f_mhz} MHz "
          f"(band {scene.freqs[scene.band_index(f_mhz)]:.0f})")
    PL = scene.pathloss_map(tx_cell, f_mhz)                 # (NX,NY,NZ) dB
    PL_plane = PL[:, y, :]                                  # coarse (NX,NZ)

    # interior air on the coarse plane (compare only where both are meaningful)
    grid = np.load(B.MATERIAL_GRID)
    air = (grid[:, y, :] == 0)
    try:
        inside = np.load(B.SCENE_DIR / "inside_mask.npy")[:, y, :]
        air &= inside
    except FileNotFoundError:
        pass

    # --- bring the fine FDTD strength down to the coarse plane ------------
    fdb_coarse = ndimage.zoom(field_db, (PL_plane.shape[0] / field_db.shape[0],
                                         PL_plane.shape[1] / field_db.shape[1]),
                              order=1)
    fdtd_loss = -fdb_coarse                                 # higher = weaker, like PL

    # near-Tx exclusion
    xx, zz = np.meshgrid(np.arange(air.shape[0]), np.arange(air.shape[1]),
                         indexing="ij")
    r_m = np.hypot(xx - tx_cell[0], zz - tx_cell[2]) * scene.cell
    mask = air & (r_m > args.exclude_r_m) & np.isfinite(PL_plane) & np.isfinite(fdtd_loss)

    a = PL_plane[mask]
    b = fdtd_loss[mask]
    rho, p = spearmanr(a, b)
    print(f"n={mask.sum()} interior cells  Spearman(SceneV3 PL, FDTD loss) "
          f"rho={rho:+.3f}  (p={p:.1e})")

    result = dict(band=args.band, f_mhz=f_mhz, n_cells=int(mask.sum()),
                  spearman_rho=float(rho), p_value=float(p),
                  note="positive rho = agrees in shape; divergence is the "
                       "diffraction/deep-shadow regime the ray engine misses")
    (run / "validation_vs_eikonal.json").write_text(json.dumps(result, indent=2))
    _plot(run, PL_plane, fdb_coarse, mask, a, b, rho, args.band)
    print(f"outputs -> {run}/validation_vs_eikonal.{{json,png}}")


def _plot(run, PL_plane, fdb_coarse, mask, a, b, rho, band):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    pl_show = np.where(mask, PL_plane, np.nan)
    fd_show = np.where(mask, fdb_coarse, np.nan)
    im0 = ax[0].imshow(pl_show.T, origin="lower", cmap="magma_r", aspect="equal")
    ax[0].set_title(f"SceneV3 eikonal PL (dB)  {band}")
    fig.colorbar(im0, ax=ax[0], shrink=0.8)
    im1 = ax[1].imshow(fd_show.T, origin="lower", cmap="viridis", aspect="equal")
    ax[1].set_title("SIM V3 full-wave RMS strength (dB)")
    fig.colorbar(im1, ax=ax[1], shrink=0.8)
    ax[2].hexbin(a, b, gridsize=40, cmap="Greys", mincnt=1)
    ax[2].set_xlabel("SceneV3 path loss (dB)")
    ax[2].set_ylabel("FDTD loss = -strength (dB)")
    ax[2].set_title(f"rank agreement  rho = {rho:+.3f}")
    fig.tight_layout()
    fig.savefig(run / "validation_vs_eikonal.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
