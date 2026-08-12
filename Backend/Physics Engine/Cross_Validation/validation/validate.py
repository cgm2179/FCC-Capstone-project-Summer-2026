#!/usr/bin/env python3
"""PR #26 — cross-validate predicted vs measured RSRP, per PCI.

Samples a 2D O2I prediction (pathloss_2d/out/metrics_bs_<pci>_<freq>.npz) at every
measured point for that (network, band, PCI) from Data/records_data.js, and reports:

  n, measured mean (sanity vs the catalog), predicted mean,
  bias (mean measured−predicted), RMSE (raw),
  RMSE after a single-constant LEVEL calibration (= the EIRP+O2I offset the
    cross-cutting calibration fits; this is std of the residual),
  Spearman ρ (rank agreement of the spatial shape — the part physics must get right).

Writes validation/reports/validation_<pci>_<freq>.json and a PNG (residual map on the
floor plan + measured-vs-predicted scatter).

usage:
  python validate.py --pci 396 --band 13
  python validate.py --pci 397 --band 13
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
PL2D_OUT = REPO / "Physics Engine" / "Cross_Validation" / "pathloss_2d" / "out"
RECORDS_JS = REPO / "Data" / "records_data.js"
REPORTS = HERE / "reports"


def load_records():
    txt = RECORDS_JS.read_text(encoding="utf-8")
    return json.loads(txt[txt.index("["):txt.rindex("]") + 1])


def measured_points(records, network, band, pci):
    pts = [r for r in records if r.get("network") == network and r.get("band") == band
           and r.get("pci") == pci and r.get("rsrp") is not None]
    return pts


def sample_bilinear(grid, px, py):
    """Bilinear sample of grid(H,W) at floating (px,py); px∈[0,W), py∈[0,H)."""
    H, W = grid.shape
    x = np.clip(np.asarray(px, float), 0, W - 1.001)
    y = np.clip(np.asarray(py, float), 0, H - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    fx, fy = x - x0, y - y0
    return (grid[y0, x0] * (1 - fx) * (1 - fy) + grid[y0, x1] * fx * (1 - fy)
            + grid[y1, x0] * (1 - fx) * fy + grid[y1, x1] * fx * fy)


def spearman_rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else None


def validate(pci=396, band=13, network="lte", npz=None):
    if npz is None:
        cands = sorted(PL2D_OUT.glob(f"metrics_bs_{pci}_*MHz.npz"))
        if not cands:
            raise SystemExit(f"no prediction for pci={pci}; run predict_o2i.py --pci {pci} --band {band}")
        npz = cands[0]
    d = np.load(npz, allow_pickle=True)
    rsrp_pred = d["rsrp"]
    meta = json.loads(str(d["meta"]))

    recs = measured_points(load_records(), network, band, pci)
    if not recs:
        raise SystemExit(f"no measured points for {network} band {band} pci {pci}")
    px = np.array([r["px"] for r in recs]); py = np.array([r["py"] for r in recs])
    meas = np.array([r["rsrp"] for r in recs])
    pred = sample_bilinear(rsrp_pred, px, py)

    resid = meas - pred                       # measured − predicted
    bias = float(resid.mean())                # the level offset (EIRP+O2I) to apply
    rmse_raw = float(np.sqrt((resid ** 2).mean()))
    rmse_cal = float(resid.std())             # RMSE after removing the constant level
    rho = spearman_rho(meas, pred)
    pred_cal = pred + bias

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = dict(
        site_id=meta.get("site_id"), pci=pci, band=band, network=network,
        freq_mhz=meta.get("freq_mhz"), distance_m=meta.get("distance_m"),
        bearing_deg=meta.get("bearing_deg"), n_points=len(recs),
        measured_mean=round(float(meas.mean()), 2),
        catalog_measured_mean=meta.get("measured_mean"),
        predicted_mean_raw=round(float(pred.mean()), 2),
        level_calibration_db=round(bias, 2),
        rmse_raw_db=round(rmse_raw, 2),
        rmse_after_level_cal_db=round(rmse_cal, 2),
        spearman_rho=round(rho, 3) if rho is not None else None,
        note="rmse_after_level_cal is the honest shape error; level_calibration_db is the "
             "single EIRP+O2I constant that best-fits the measured level (Forte-Hall calibration).",
    )
    (REPORTS / f"validation_{pci}_{meta.get('freq_mhz'):.0f}MHz.json").write_text(json.dumps(report, indent=2))
    _plot(px, py, meas, pred_cal, resid - bias, rsrp_pred, report,
          REPORTS / f"validation_{pci}_{meta.get('freq_mhz'):.0f}MHz.png")

    print(f"PCI {pci} B{band} {meta.get('freq_mhz'):.0f} MHz  n={len(recs)}")
    print(f"  measured mean {meas.mean():.2f} (catalog {meta.get('measured_mean')})  "
          f"predicted mean {pred.mean():.2f}")
    print(f"  level calibration (EIRP+O2I offset) = {bias:+.2f} dB")
    print(f"  RMSE raw {rmse_raw:.2f} dB  ->  after level cal {rmse_cal:.2f} dB   "
          f"Spearman ρ {rho:.3f}")
    print(f"  -> {REPORTS / f'validation_{pci}_{meta.get('freq_mhz'):.0f}MHz.json'}")
    return report


def _plot(px, py, meas, pred_cal, resid_cal, grid, report, out_png):
    H, W = grid.shape
    fig, (axm, axs) = plt.subplots(1, 2, figsize=(15, 5.4),
                                   gridspec_kw={"width_ratios": [2, 1]})
    # residual (calibrated) on the plan
    lim = max(6.0, float(np.percentile(np.abs(resid_cal), 95)))
    sc = axm.scatter(px, py, c=resid_cal, cmap="RdBu_r", vmin=-lim, vmax=lim,
                     s=22, edgecolors="#333", linewidths=0.3)
    axm.set_xlim(0, W); axm.set_ylim(H, 0); axm.set_aspect("equal")
    axm.set_title(f"Measured − predicted (level-calibrated) — PCI {report['pci']} · "
                  f"{report['freq_mhz']:.0f} MHz\nRMSE {report['rmse_after_level_cal_db']} dB · "
                  f"ρ {report['spearman_rho']} · n {report['n_points']}")
    axm.set_xlabel("floor-plan px"); axm.set_ylabel("px")
    fig.colorbar(sc, ax=axm, label="residual (dB)", shrink=0.85)
    # scatter measured vs calibrated prediction
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
    ap.add_argument("--pci", type=int, default=396)
    ap.add_argument("--band", type=int, default=13)
    ap.add_argument("--network", default="lte")
    ap.add_argument("--npz", default=None, help="explicit metrics_bs_*.npz (else auto by pci)")
    args = ap.parse_args()
    validate(pci=args.pci, band=args.band, network=args.network,
             npz=Path(args.npz) if args.npz else None)


if __name__ == "__main__":
    main()
