#!/usr/bin/env python3
"""PR #27 — cross-validate the 3D O2I prediction against measured RSRP, per PCI.

Samples the receiver-height slice of a 3D prediction (pathloss_3d/out/metrics3d_bs_*.npz)
at each measured point — px → voxel via registration_3d.json (`px_to_vox_scale`, `rx_y_vox`,
footprint IoU 0.89) — and reports RMSE / bias / Spearman ρ + a single-constant level
calibration (the EIRP+O2I offset). JSON report + PNG (residual + scatter).

usage:
  python validate_3d.py --pci 397 --band 13
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PL3D_OUT = REPO / "Physics Engine" / "Cross_Validation" / "pathloss_3d" / "out"
REG3D = REPO / "Physics Engine" / "3D Map Physics" / "SIM V1 3D" / "registration_3d.json"
REPORTS = HERE / "reports"
import validate as V   # load_records, measured_points, spearman_rho


def sample2(grid, a, b):
    """Bilinear sample of grid at (a=axis0, b=axis1)."""
    A0, A1 = grid.shape
    a = np.clip(np.asarray(a, float), 0, A0 - 1.001)
    b = np.clip(np.asarray(b, float), 0, A1 - 1.001)
    a0, b0 = np.floor(a).astype(int), np.floor(b).astype(int)
    a1, b1 = a0 + 1, b0 + 1
    fa, fb = a - a0, b - b0
    return (grid[a0, b0] * (1 - fa) * (1 - fb) + grid[a1, b0] * fa * (1 - fb)
            + grid[a0, b1] * (1 - fa) * fb + grid[a1, b1] * fa * fb)


def validate(pci=397, band=13, network="lte", npz=None):
    if npz is None:
        cands = sorted(PL3D_OUT.glob(f"metrics3d_bs_{pci}_*MHz.npz"))
        if not cands:
            raise SystemExit(f"no 3D prediction for pci={pci}; run predict_o2i_3d.py --pci {pci}")
        npz = cands[0]
    d = np.load(npz, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    frame = str(d["frame"]) if "frame" in d.files else "vox"

    recs = V.measured_points(V.load_records(), network, band, pci)
    if not recs:
        raise SystemExit(f"no measured points for {network} band {band} pci {pci}")
    px = np.array([r["px"] for r in recs]); py = np.array([r["py"] for r in recs])
    meas = np.array([r["rsrp"] for r in recs])

    if frame == "px":                    # accurate-geometry field, floor-plan px frame (Commit 2)
        pred = V.sample_bilinear(d["rsrp"], px, py)       # rsrp is (H,W)=(py,px)
        plot_x, plot_y = px, py
    else:                                # legacy voxel frame -> px→vox registration (Commit 1)
        reg = json.loads(REG3D.read_text()); rx_y = int(reg["rx_y_vox"])
        aff = reg.get("affine_px_to_vox"); sl = d["rsrp"][:, rx_y, :]
        if aff is not None:
            (a, b, c), (d0, e, f) = aff
            vx = a * px + b * py + c; vz = d0 * px + e * py + f
        else:
            sx = reg["px_to_vox_scale"]["x"]; sz = reg["px_to_vox_scale"]["z"]
            vx, vz = px * sx, py * sz
        pred = sample2(sl, vx, vz)
        plot_x, plot_y = vx, vz

    resid = meas - pred
    bias = float(resid.mean()); rmse_raw = float(np.sqrt((resid ** 2).mean()))
    rmse_cal = float(resid.std()); rho = V.spearman_rho(meas, pred)
    pred_cal = pred + bias

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = dict(dim="3D", site_id=meta.get("site_id"), pci=pci, band=band, network=network,
                  freq_mhz=meta.get("freq_mhz"), bearing_deg=meta.get("bearing_deg"),
                  elev_m=meta.get("elev_m"), n_points=len(recs),
                  measured_mean=round(float(meas.mean()), 2),
                  predicted_mean_raw=round(float(pred.mean()), 2),
                  level_calibration_db=round(bias, 2), rmse_raw_db=round(rmse_raw, 2),
                  rmse_after_level_cal_db=round(rmse_cal, 2),
                  spearman_rho=round(rho, 3) if rho is not None else None)
    tag = f"{pci}_{meta.get('freq_mhz'):.0f}MHz" + (
        f"_b{int(meta['bearing_deg'])}" if "_b" in npz.stem else "")
    (REPORTS / f"validation3d_{tag}.json").write_text(json.dumps(report, indent=2))
    _plot(plot_x, plot_y, meas, pred_cal, resid - bias, report, REPORTS / f"validation3d_{tag}.png")

    print(f"[3D] PCI {pci} B{band} {meta.get('freq_mhz'):.0f} MHz  n={len(recs)}  "
          f"arrival {meta.get('bearing_deg')}°")
    print(f"  measured mean {meas.mean():.2f}  predicted mean {pred.mean():.2f}  "
          f"level cal {bias:+.2f} dB")
    print(f"  RMSE raw {rmse_raw:.2f} -> after level cal {rmse_cal:.2f} dB   Spearman ρ {rho:.3f}")
    print(f"  -> {REPORTS / f'validation3d_{tag}.json'}")
    return report


def _plot(vx, vz, meas, pred_cal, resid_cal, report, out_png):
    fig, (axm, axs) = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [2, 1]})
    lim = max(6.0, float(np.percentile(np.abs(resid_cal), 95)))
    sc = axm.scatter(vx, vz, c=resid_cal, cmap="RdBu_r", vmin=-lim, vmax=lim, s=24,
                     edgecolors="#333", linewidths=0.3)
    axm.set_aspect("equal"); axm.set_xlabel("voxel X"); axm.set_ylabel("voxel Z")
    axm.set_title(f"[3D] measured − predicted (level-cal) — PCI {report['pci']} · "
                  f"{report['freq_mhz']:.0f} MHz · arrival {report['bearing_deg']}°\n"
                  f"RMSE {report['rmse_after_level_cal_db']} dB · ρ {report['spearman_rho']} · "
                  f"n {report['n_points']}")
    fig.colorbar(sc, ax=axm, label="residual (dB)", shrink=0.85)
    axs.scatter(pred_cal, meas, s=20, alpha=0.7, edgecolors="#333", linewidths=0.3)
    lo = min(pred_cal.min(), meas.min()) - 2; hi = max(pred_cal.max(), meas.max()) + 2
    axs.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    axs.set_xlim(lo, hi); axs.set_ylim(lo, hi); axs.set_aspect("equal")
    axs.set_xlabel("predicted RSRP (level-cal, dBm)"); axs.set_ylabel("measured RSRP (dBm)")
    axs.set_title(f"level cal {report['level_calibration_db']:+.1f} dB"); axs.legend(fontsize=8)
    axs.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pci", type=int, default=397)
    ap.add_argument("--band", type=int, default=13)
    ap.add_argument("--network", default="lte")
    ap.add_argument("--npz", default=None)
    args = ap.parse_args()
    validate(pci=args.pci, band=args.band, network=args.network,
             npz=Path(args.npz) if args.npz else None)


if __name__ == "__main__":
    main()
