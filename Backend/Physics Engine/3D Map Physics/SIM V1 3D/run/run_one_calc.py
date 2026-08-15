#!/usr/bin/env python3
"""
run_one_calc.py — one full-physics calculation (PL + eikonal T), SIM V1 3D.

Loads the voxelized scene, places one ceiling-height transmitter, and computes
BOTH co-registered grids with SceneV3:
  PL(x,y,z)  direct through-wall path loss — real angle+thickness Fresnel/Airy
             transmission + Im(q) absorption (physics_v2 kernels via CrossingLUT).
             (Diffraction/reflection/scattering arrive in later stages.)
  T(x,y,z)   first-arrival time from the 3-D eikonal fast-march (refraction +
             corner wraparound baked in).

Outputs (SIM V1 3D/preview/):
  pl_volume.npy, t_volume.npy       float32 (nx,ny,nz)
  pl_one_calc.png                   PL slices (top) + eikonal T slices (bottom)
  one_calc_meta.json                tx, freq, PL/T stats + validation

usage:
  python "SIM V1 3D/run_one_calc.py"                 # 3500 MHz, auto Tx
  python "SIM V1 3D/run_one_calc.py" --freq 2442
  python "SIM V1 3D/run_one_calc.py" --tx 128 9 55
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # SIM V1 3D root (siblings live there)
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine_3d import load_scene
import physics_3d as P3

HERE = Path(__file__).resolve().parent.parent


def pick_tx(valid, want):
    ijk = np.argwhere(valid)
    target = np.array([want[i] * valid.shape[i] for i in range(3)])
    return ijk[np.argmin(((ijk - target) ** 2).sum(1))].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=float, default=3500.0)
    ap.add_argument("--tx", type=float, nargs=3, default=None)
    args = ap.parse_args()

    scene, man = load_scene()
    inside = scene.inside
    valid = np.load(HERE / "valid_tx_mask.npy")
    nx, ny, nz = scene.M.shape
    cell = man["cell_size_m"]
    tx = np.array(args.tx, float) if args.tx else pick_tx(valid, (0.5, 0.85, 0.5))
    print(f"grid {nx}x{ny}x{nz}  cell {cell} m  tx(voxel)={tx.astype(int).tolist()}  "
          f"f={args.freq:.0f} MHz  barrier classes={scene.barrier_classes}")

    PL = scene.pathloss_map(tx, args.freq)
    T = scene.arrival_time(tx, args.freq)                     # seconds
    Tgeo = scene.geometric_time(tx)
    np.save(HERE / "preview" / "pl_volume.npy", PL)
    np.save(HERE / "preview" / "t_volume.npy", T)

    # ---- verification ------------------------------------------------------
    d_m = np.maximum(np.linalg.norm(scene.P - tx.astype(np.float32), axis=1) * cell,
                     man["physics"]["d0_m"]).reshape(PL.shape)
    fspl = scene.fspl1[scene.band_index(args.freq)] + 10 * man["physics"]["n_exp"] * np.log10(d_m)
    excess = (PL - fspl)[inside]
    reach = np.isfinite(T) & inside
    slack = (T - Tgeo)[reach]
    tol_T = 3.0 * cell / P3.C_LIGHT
    pin = PL[inside]
    corr = float(np.corrcoef(np.clip(pin, 0, 150), np.log10(d_m[inside]))[0, 1])
    stats = dict(
        tx_voxel=[int(v) for v in tx], freq_mhz=args.freq,
        pl_min=round(float(pin.min()), 1), pl_median=round(float(np.median(pin)), 1),
        pl_max=round(float(pin.max()), 1),
        excess_over_fspl_min=round(float(excess.min()), 2),
        T_max_ns=round(float(T[reach].max() * 1e9), 1),
        T_minus_dc_min_ns=round(float(slack.min() * 1e9), 3),
        corr_pl_vs_logdist=round(corr, 3),
        note="direct-only PL; deep shadows fill once diffraction (Stage 2) is added",
    )
    print("PL(interior) min/median/max =", stats["pl_min"], stats["pl_median"],
          stats["pl_max"], "dB   excess>=", stats["excess_over_fspl_min"])
    print("T max =", stats["T_max_ns"], "ns   T-d/c min =", stats["T_minus_dc_min_ns"],
          f"ns (tol {tol_T*1e9:.2f})   corr(PL,logd) =", stats["corr_pl_vs_logdist"])
    assert excess.min() >= -0.5, "PL below free space — bug"
    assert slack.min() > -tol_T, "eikonal beat straight-line beyond tolerance — bug"
    (HERE / "preview" / "one_calc_meta.json").write_text(json.dumps(stats, indent=2))

    # ---- preview: PL (top row) + eikonal T (bottom row) --------------------
    rlo, rhi = man.get("room_band", [1, ny - 1])
    ys = sorted({rlo, (rlo + rhi) // 2, rhi})
    iz = int(round(tx[2]))
    plm = np.where(inside, PL, np.nan)
    tms = np.where(reach, T * 1e9, np.nan)                    # ns
    pl_cmap = plt.get_cmap("viridis").copy(); pl_cmap.set_bad("#e9edeb")
    t_cmap = plt.get_cmap("plasma").copy(); t_cmap.set_bad("#e9edeb")
    plvmax = min(float(np.nanpercentile(plm, 99)), 150.0)
    tvmax = float(np.nanpercentile(tms, 99))

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for a, y in zip(ax[0], ys):
        im = a.imshow(plm[:, y, :].T, origin="lower", cmap=pl_cmap, vmin=43, vmax=plvmax, aspect="equal")
        if abs(y - tx[1]) <= 1:
            a.plot(tx[0], tx[2], "r*", ms=15, mec="white")
        a.set_title(f"PL  z≈{y*cell:.1f} m (iy {y})"); a.set_xlabel("X"); a.set_ylabel("Z")
        fig.colorbar(im, ax=a, shrink=0.8, label="PL (dB)")

    ymid = ys[len(ys) // 2]
    im = ax[1, 0].imshow(plm[:, :, iz].T, origin="lower", cmap=pl_cmap, vmin=43, vmax=plvmax, aspect="auto")
    ax[1, 0].plot(tx[0], tx[1], "r*", ms=15, mec="white")
    ax[1, 0].set_title(f"PL  vertical (Z={iz})"); ax[1, 0].set_xlabel("X"); ax[1, 0].set_ylabel("Y")
    fig.colorbar(im, ax=ax[1, 0], shrink=0.8, label="PL (dB)")

    im = ax[1, 1].imshow(tms[:, ymid, :].T, origin="lower", cmap=t_cmap, vmax=tvmax, aspect="equal")
    ax[1, 1].contour(tms[:, ymid, :].T, levels=8, colors="white", linewidths=0.4, alpha=0.6)
    ax[1, 1].plot(tx[0], tx[2], "c*", ms=15, mec="black")
    ax[1, 1].set_title(f"eikonal T  z≈{ymid*cell:.1f} m (ns)"); ax[1, 1].set_xlabel("X"); ax[1, 1].set_ylabel("Z")
    fig.colorbar(im, ax=ax[1, 1], shrink=0.8, label="T (ns)")

    im = ax[1, 2].imshow(tms[:, :, iz].T, origin="lower", cmap=t_cmap, vmax=tvmax, aspect="auto")
    ax[1, 2].plot(tx[0], tx[1], "c*", ms=15, mec="black")
    ax[1, 2].set_title(f"eikonal T  vertical (Z={iz})"); ax[1, 2].set_xlabel("X"); ax[1, 2].set_ylabel("Y")
    fig.colorbar(im, ax=ax[1, 2], shrink=0.8, label="T (ns)")

    fig.suptitle(f"SIM V1 3D — full-physics PL + eikonal T · {args.freq:.0f} MHz · "
                 f"Tx {tx.astype(int).tolist()}", fontsize=13)
    fig.tight_layout()
    out = HERE / "preview" / "pl_one_calc.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
