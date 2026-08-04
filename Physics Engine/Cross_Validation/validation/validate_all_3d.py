#!/usr/bin/env python3
"""Commit 3 — validate the explicit 3D O2I across the LARGER donor sample (indoor).

Loops every catalog cell (network, band, pci) with n ≥ MIN_N, runs the explicit
accurate-geometry predictor + the px-frame validator, and emits a summary table with
Spearman ρ reported BEFORE the level calibration ("Path Loss first" — the shape the physics
must get right; the single EIRP+O2I level is fit afterwards).

Confidence:
  * BS7 Forte Hall is the calibrated ANCHOR — known coordinates + orientation, so both ρ
    (shape) and the level are meaningful.
  * The other donors (BS4/5/6) have uncertain antenna azimuth/identity, so ρ (shape agreement)
    is the honest metric; the absolute level is not.

usage:
  python validate_all_3d.py                  # all cells n>=20
  python validate_all_3d.py --min-n 30
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CATALOG = REPO / "Physics Engine" / "Cross_Validation" / "base_station_catalog" / "base_stations.csv"
PL3D = REPO / "Physics Engine" / "Cross_Validation" / "pathloss_3d"
REPORTS = HERE / "reports"
for p in (PL3D, HERE):
    sys.path.insert(0, str(p))
import predict_o2i_3d as PR       # noqa: E402
import validate_3d as V3          # noqa: E402

ANCHOR = "BS7_forte_hall"


def cells(min_n):
    """Unique (network, band, pci) → (best n, catalog row). One physical cell per key."""
    by = {}
    with open(CATALOG, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n = int(r["n_points"] or 0)
            if n < min_n or (r.get("status") or "").startswith("na"):
                continue
            key = (r["network"], int(r["band"]), int(r["pci"]))
            if key not in by or n > by[key][0]:
                by[key] = (n, r)
    return by


def run(min_n=20):
    rows = []
    for (net, band, pci), (n, r) in sorted(cells(min_n).items(), key=lambda kv: -kv[1][0]):
        try:
            npz = PR.predict(pci=pci, band=band, network=net)
            rep = V3.validate(pci=pci, band=band, network=net, npz=npz)
        except SystemExit as e:
            print(f"  skip {net} b{band} pci {pci}: {e}"); continue
        except Exception as e:                                     # noqa: BLE001
            print(f"  ERROR {net} b{band} pci {pci}: {e}"); continue
        rows.append(dict(site_id=r["site_id"], carrier=r["carrier"], network=net, band=band,
                         pci=pci, freq_mhz=rep.get("freq_mhz"), n=rep.get("n_points"),
                         spearman_rho=rep.get("spearman_rho"),
                         rmse_after_level_cal_db=rep.get("rmse_after_level_cal_db"),
                         level_calibration_db=rep.get("level_calibration_db"),
                         measured_mean=rep.get("measured_mean"),
                         anchor=(r["site_id"] == ANCHOR)))

    rows.sort(key=lambda x: (x["spearman_rho"] if x["spearman_rho"] is not None else -9), reverse=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "validation3d_summary.json").write_text(json.dumps(
        {"min_n": min_n, "n_cells": len(rows), "gate": "rho>0.75 aspirational; ~0.64 is the "
         "single-floor fidelity ceiling (== independent 2-D result)", "cells": rows}, indent=2))
    with open(REPORTS / "validation3d_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"\n{'='*92}\n3D O2I validation — {len(rows)} cells (n≥{min_n}), ρ BEFORE level cal, sorted by ρ\n{'='*92}")
    print(f"  {'site':16}{'carr':9}{'net':7}{'b':>3} {'freq':>6} {'pci':>5} {'n':>4} "
          f"{'ρ':>7} {'RMSE_cal':>9} {'level':>7}  anchor")
    for x in rows:
        rho = "  n/a" if x["spearman_rho"] is None else f"{x['spearman_rho']:+.3f}"
        print(f"  {x['site_id']:16}{x['carrier'][:8]:9}{x['network']:7}{x['band']:>3} "
              f"{x['freq_mhz']:>6.0f} {x['pci']:>5} {x['n']:>4} {rho:>7} "
              f"{x['rmse_after_level_cal_db']:>8.1f}d {x['level_calibration_db']:>+6.1f}  "
              f"{'ANCHOR' if x['anchor'] else ''}")
    anch = [x for x in rows if x["anchor"] and x["spearman_rho"] is not None]
    if anch:
        best = max(anch, key=lambda x: x["spearman_rho"])
        print(f"\nForte Hall anchor best ρ = {best['spearman_rho']:+.3f} (PCI {best['pci']}, "
              f"{best['freq_mhz']:.0f} MHz, n={best['n']})")
    print(f"-> {REPORTS/'validation3d_summary.csv'}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-n", type=int, default=20)
    run(min_n=ap.parse_args().min_n)


if __name__ == "__main__":
    main()
