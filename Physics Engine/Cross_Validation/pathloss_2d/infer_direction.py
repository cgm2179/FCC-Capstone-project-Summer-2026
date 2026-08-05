#!/usr/bin/env python3
"""PR #26 extension — infer each PCI's coverage ARRIVAL direction from the measured
spatial pattern (RSRP / RSRQ / CINR·SINR / RSSI).

Motivation: the flat SW plane wave (Forte Hall site geometry, arrival 237°) tracks PCI 397
but anti-correlates PCI 396 — 396's indoor energy is stronger toward the NE. So which
direction does each sector's coverage actually enter from, and is the sticky-note
antenna→PCI mapping (396/397/398) right?

Method: for each candidate arrival bearing β, ray-march the Motley-Keenan indoor wall loss
along β from every measured point out to the façade (points-only, cheap), and correlate the
predicted shape (−loss) with each measured metric (Spearman ρ). The β that maximizes ρ is
the inferred arrival direction. Bearing→pixel uses the `affine_px_to_local_m` linear part
(ENU arrival unit → L⁻¹ → pixel), so β=237° reproduces the geometry-derived direction.

Per band (≥ min-n points) and per metric, aggregated n-weighted per PCI. Outputs:
  validation/reports/direction_<pci>.json   β* per metric + per-band, peak ρ, vs site 237°
  validation/reports/direction_<pci>.png    polar ρ(β) lobes + Cartesian ρ-vs-β curves

usage:
  python infer_direction.py                 # PCI 396, 397, 398
  python infer_direction.py --pci 396 --step-deg 2
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STEP1 = REPO / "Physics Engine" / "2D" / "STEP_1"
CATALOG = REPO / "Physics Engine" / "Cross_Validation" / "base_station_catalog" / "base_stations.csv"
REPORTS = REPO / "Physics Engine" / "Cross_Validation" / "validation" / "reports"
sys.path.insert(0, str(HERE))                                   # predict_o2i
sys.path.insert(0, str(REPO / "Physics Engine" / "Cross_Validation" / "validation"))  # validate
import predict_o2i as P2
import validate as V

METRICS = ["rsrp", "rsrq", "cinr", "rssi"]
MET_LABEL = {"rsrp": "RSRP", "rsrq": "RSRQ", "cinr": "SINR", "rssi": "RSSI"}
MET_COLOR = {"rsrp": "#1f77b4", "rsrq": "#2ca02c", "cinr": "#d62728", "rssi": "#9467bd"}


def point_loss(px, py, dir_px, grid, loss_db, loss_per_m, mpp, wall_sat_db=60.0, step_px=0.6):
    """Motley-Keenan indoor wall loss (dB) from each point marching along +dir to the façade."""
    H, W = grid.shape
    dx, dy = dir_px
    nrm = np.hypot(dx, dy) or 1.0
    dx, dy = dx / nrm, dy / nrm
    K = int(np.ceil(np.hypot(W, H) / step_px)) + 1
    s = np.arange(K, dtype=np.float32) * step_px
    px = np.asarray(px, np.float32); py = np.asarray(py, np.float32)
    sx = px[:, None] + s[None, :] * dx
    sy = py[:, None] + s[None, :] * dy
    valid = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    xi = np.clip(sx.round(), 0, W - 1).astype(np.int32)
    yi = np.clip(sy.round(), 0, H - 1).astype(np.int32)
    mats = np.where(valid, grid[yi, xi], 255)
    wall_ids = [m for m in np.unique(grid) if loss_db[m] > 0]
    bulk_ids = [m for m in np.unique(grid) if loss_per_m[m] > 0]
    extra = np.zeros(mats.shape[0], np.float32)
    for m in wall_ids:
        hit = mats == m
        runs = hit[:, 0].astype(np.int32) + (hit[:, 1:] & ~hit[:, :-1]).sum(1, dtype=np.int32)
        extra += loss_db[m] * runs
    sp = step_px * mpp
    for m in bulk_ids:
        extra += loss_per_m[m] * (mats == m).sum(1) * sp
    return wall_sat_db * -np.expm1(-extra / wall_sat_db)


def bearing_to_dir(beta_deg, Linv):
    e = np.radians(beta_deg)
    d = Linv @ np.array([np.sin(e), np.cos(e)])       # ENU arrival unit → pixel delta
    return d


def site_bearing(pci):
    try:
        with open(CATALOG, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["pci"] == str(pci) and r["bearing_deg"]:
                    return float(r["bearing_deg"])
    except FileNotFoundError:
        pass
    return 237.0


def infer_pci(pci, records, grid, loss_db, loss_per_m, mpp, Linv, betas, min_n):
    recs_all = [r for r in records if r.get("pci") == pci]
    bands = defaultdict(list)
    for r in recs_all:
        bands[(r["network"], r["band"])].append(r)

    per_band = {}
    for (net, band), recs in bands.items():
        if len(recs) < min_n:
            continue
        px = np.array([r["px"] for r in recs]); py = np.array([r["py"] for r in recs])
        vals = {m: np.array([r[m] for r in recs], float) for m in METRICS}
        rho = {m: np.zeros(len(betas)) for m in METRICS}
        for j, b in enumerate(betas):
            loss = point_loss(px, py, bearing_to_dir(b, Linv), grid, loss_db, loss_per_m, mpp)
            for m in METRICS:
                r_ = V.spearman_rho(vals[m], -loss)
                rho[m][j] = r_ if r_ is not None else 0.0
        per_band[f"{net}_B{band}"] = dict(n=len(recs), rho=rho)

    agg = {}
    for m in METRICS:
        num = np.zeros(len(betas)); den = 0
        for d in per_band.values():
            num += d["rho"][m] * d["n"]; den += d["n"]
        agg[m] = num / den if den else np.zeros(len(betas))
    return per_band, agg


def _plot(pci, betas, per_band, agg, site_b, best, out_png):
    fig = plt.figure(figsize=(13, 5.6))
    axp = fig.add_subplot(1, 2, 1, projection="polar")
    axp.set_theta_zero_location("N"); axp.set_theta_direction(-1)
    th = np.radians(betas)
    for m in METRICS:
        r = np.clip(agg[m], 0, 1)
        axp.plot(np.append(th, th[0]), np.append(r, r[0]), color=MET_COLOR[m],
                 lw=2, label=f"{MET_LABEL[m]} (β*={best[m]['beta']:.0f}°, ρ={best[m]['rho']:.2f})")
    axp.plot([np.radians(site_b)] * 2, [0, 1], "k--", lw=1.5, label=f"site geometry {site_b:.0f}°")
    axp.set_rlim(0, 1); axp.set_title(f"PCI {pci} — inferred arrival direction (ρ≥0 lobes)")
    axp.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8)

    axc = fig.add_subplot(1, 2, 2)
    for m in METRICS:
        axc.plot(betas, agg[m], color=MET_COLOR[m], lw=1.8, label=MET_LABEL[m])
    axc.axvline(site_b, color="k", ls="--", lw=1.2, label=f"site {site_b:.0f}°")
    axc.axhline(0, color="#999", lw=0.8)
    axc.set_xlabel("arrival bearing β (°, compass)"); axc.set_ylabel("Spearman ρ(metric, −loss)")
    axc.set_xlim(0, 360); axc.set_xticks(range(0, 361, 45)); axc.grid(alpha=0.3)
    axc.set_title(f"PCI {pci} — ρ vs bearing (aggregate over bands)")
    axc.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def run(pcis, step_deg=3.0, min_n=20):
    meta = P2.load_floorplan_meta()
    mpp = meta["meters_per_px"]
    grid = np.load(STEP1 / "material_grid.npy")
    mats = json.loads((STEP1 / "materials.json").read_text())
    loss_db, loss_per_m = P2.band_loss_vectors(mats, 751.0)          # geometry-dominated; B13 ref
    grid = P2.mk.consolidate_walls(grid, loss_db)
    Linv = np.linalg.inv(np.array(meta["affine_px_to_local_m"])[:, :2])
    betas = np.arange(0, 360, step_deg)
    records = V.load_records()
    REPORTS.mkdir(parents=True, exist_ok=True)

    summary = []
    for pci in pcis:
        per_band, agg = infer_pci(pci, records, grid, loss_db, loss_per_m, mpp, Linv, betas, min_n)
        if not per_band:
            print(f"PCI {pci}: no band with ≥{min_n} points; skipped"); continue
        site_b = site_bearing(pci)
        best = {m: {"beta": float(betas[int(np.argmax(agg[m]))]),
                    "rho": round(float(np.max(agg[m])), 3)} for m in METRICS}
        best_pb = {k: {"beta_rsrp": float(betas[int(np.argmax(d["rho"]["rsrp"]))]),
                       "rho_rsrp": round(float(np.max(d["rho"]["rsrp"])), 3), "n": d["n"]}
                   for k, d in per_band.items()}
        rep = dict(pci=pci, site_geometry_bearing_deg=site_b, step_deg=step_deg,
                   bands_used=list(per_band), best_by_metric=best, best_by_band_rsrp=best_pb)
        (REPORTS / f"direction_{pci}.json").write_text(json.dumps(rep, indent=2))
        _plot(pci, betas, per_band, agg, site_b, best, REPORTS / f"direction_{pci}.png")

        drsrp = abs(((best["rsrp"]["beta"] - site_b + 180) % 360) - 180)
        summary.append((pci, best, site_b, drsrp, best_pb))
        print(f"\nPCI {pci}  (site geometry {site_b:.0f}°)  bands: {', '.join(per_band)}")
        for m in METRICS:
            print(f"   {MET_LABEL[m]:5} β* = {best[m]['beta']:5.0f}°  peak ρ = {best[m]['rho']:+.2f}")
        print(f"   RSRP β* is {drsrp:.0f}° from the site geometry"
              + ("  ✓ matches SW arrival" if drsrp < 35 else "  ✗ NOT the flat SW arrival"))
        print("   per-band RSRP β*: " + " · ".join(
            f"{k} {v['beta_rsrp']:.0f}°(ρ{v['rho_rsrp']:+.2f},n{v['n']})" for k, v in best_pb.items()))

    print("\n===== antenna-hypothesis read =====")
    for pci, best, site_b, drsrp, _ in summary:
        tag = "SW arrival (site geometry)" if drsrp < 35 else f"arrives from ~{best['rsrp']['beta']:.0f}° (NOT SW)"
        print(f"  PCI {pci}: {tag}; RSRP peak ρ {best['rsrp']['rho']:+.2f}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pci", type=int, nargs="*", default=[396, 397, 398])
    ap.add_argument("--step-deg", type=float, default=3.0)
    ap.add_argument("--min-n", type=int, default=20)
    args = ap.parse_args()
    run(args.pci, step_deg=args.step_deg, min_n=args.min_n)


if __name__ == "__main__":
    main()
