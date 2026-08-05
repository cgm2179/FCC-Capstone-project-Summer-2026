#!/usr/bin/env python3
"""Phase 3 — validate outdoor path-loss predictions vs the 7/24 ground-level walk, per BS.

Samples a predicted outdoor ground-plane field (pathloss_outdoor/out/outdoor_*.npz) at every 7/24
walk point for the site's PCIs — lat/lon → tile voxel via `voxelize_city.lonlat_to_vox` (EPSG:3857
anchor) — and reports Spearman ρ (shape, BEFORE level calibration) + RMSE per PCI, grouped per base
station. The 7/24 walk is far richer than indoor (30–140 pts/PCI) and covers BS2/4/5/6/7 + Forte.

usage:
  python validate_outdoor.py --site BS7_forte_hall           # validate that site's field(s)
  python validate_outdoor.py --all
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SIM3D = REPO / "Physics Engine" / "3D Map Physics" / "SIM V1 3D"
PL_OUT = REPO / "Physics Engine" / "Cross_Validation" / "pathloss_outdoor" / "out"
TILE = SIM3D / "city" / "NoMa_FCC_tile"
WALK = REPO / "FCC_Walk_Outdoor_Indoor_Full" / "Outdoor_Walk_Test_7-24" / "records_data.js"
REPORTS = HERE / "reports"
sys.path.insert(0, str(SIM3D))
import voxelize_city as VC     # noqa: E402  (lonlat_to_vox)

# donor -> its PCIs (from the field intel + indoor/outdoor catalogs). One physical site per donor.
SITE_PCIS = {
    "BS7_forte_hall":     [396, 397, 398],
    "BS5_1140_n_capitol": [293, 185, 171, 172, 173, 67],
    "BS6_1005_n_capitol": [359, 255, 256, 193, 194, 411, 71, 169, 199, 54, 140, 146],
    "BS4_55_m_st":        [17, 489, 172, 272, 429, 382, 194, 300, 56, 214],
    "BS2_900_nj_ave":     [291, 82],
}


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if len(a) < 8:
        return None
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean(); d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else None


def sample2(grid, a, b):
    A0, A1 = grid.shape
    a = np.clip(np.asarray(a, float), 0, A0 - 1.001); b = np.clip(np.asarray(b, float), 0, A1 - 1.001)
    a0, b0 = np.floor(a).astype(int), np.floor(b).astype(int); fa, fb = a - a0, b - b0
    return (grid[a0, b0] * (1 - fa) * (1 - fb) + grid[a0 + 1, b0] * fa * (1 - fb)
            + grid[a0, b0 + 1] * (1 - fa) * fb + grid[a0 + 1, b0 + 1] * fa * fb)


def load_walk():
    t = WALK.read_text(encoding="utf-8")
    return json.loads(t[t.index("["):t.rindex("]") + 1])


def validate_field(npz_path, recs, man):
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    site = meta["site_id"]; field = np.asarray(d["rsrp_ground"], float)  # (NX,NZ)
    street = np.asarray(d["street"], bool)
    fld = np.where(street, field, np.nan)
    NX, NZ = fld.shape
    rows = []
    for pci in SITE_PCIS.get(site, []):
        pts = [r for r in recs if r.get("pci") == pci and r.get("avg_rsrp") is not None
               and r.get("latitude") is not None]
        if len(pts) < 8:
            continue
        lon = np.array([r["longitude"] for r in pts]); lat = np.array([r["latitude"] for r in pts])
        meas = np.array([r["avg_rsrp"] for r in pts])
        vx = np.empty(len(pts)); vz = np.empty(len(pts))
        for i in range(len(pts)):
            vx[i], vz[i] = VC.lonlat_to_vox(man, lon[i], lat[i])
        inb = (vx >= 0) & (vx < NX - 1) & (vz >= 0) & (vz < NZ - 1)
        if inb.sum() < 8:
            continue
        pred = sample2(fld, vx[inb], vz[inb]); meas_in = meas[inb]
        rho = spearman(meas_in, pred)
        ok = np.isfinite(pred)
        resid = meas_in[ok] - pred[ok]
        rows.append(dict(site_id=site, pci=pci, freq_mhz=meta["freq_mhz"],
                         mechanisms="+".join(meta["mechanisms"]), n=int(ok.sum()),
                         spearman_rho=None if rho is None else round(rho, 3),
                         rmse_after_level_cal_db=round(float(resid.std()), 2) if ok.sum() else None,
                         level_calibration_db=round(float(resid.mean()), 2) if ok.sum() else None,
                         measured_mean=round(float(meas_in[ok].mean()), 2) if ok.sum() else None))
    return rows


def run(sites=None):
    man = json.loads((TILE / "manifest_3d.json").read_text())
    recs = load_walk()
    fields = sorted(glob.glob(str(PL_OUT / "outdoor_*.npz")))
    if sites:
        fields = [f for f in fields if any(s in Path(f).name for s in sites)]
    if not fields:
        raise SystemExit("no predicted outdoor fields; run predict_outdoor.py first")
    rows = []
    for f in fields:
        rows += validate_field(Path(f), recs, man)
    rows.sort(key=lambda x: (x["spearman_rho"] if x["spearman_rho"] is not None else -9), reverse=True)

    REPORTS.mkdir(parents=True, exist_ok=True)
    bysite = defaultdict(list)
    for x in rows:
        if x["spearman_rho"] is not None:
            bysite[x["site_id"]].append(x["spearman_rho"])
    summ = dict(n_cells=len(rows), cells=rows,
                per_base_station={s: dict(cells=len(v), rho_median=round(float(np.median(v)), 3),
                                          rho_best=round(float(np.max(v)), 3))
                                  for s, v in bysite.items()})
    (REPORTS / "validation_outdoor_summary.json").write_text(json.dumps(summ, indent=2))

    print(f"\n{'='*84}\nOUTDOOR 3D path-loss validation vs 7/24 walk — ρ before level cal\n{'='*84}")
    print(f"  {'site':20}{'pci':>5}{'freq':>7}{'mech':>26}{'n':>5}{'ρ':>8}{'RMSE':>8}")
    for x in rows:
        rho = "  n/a" if x["spearman_rho"] is None else f"{x['spearman_rho']:+.3f}"
        print(f"  {x['site_id']:20}{x['pci']:>5}{x['freq_mhz']:>7.0f}{x['mechanisms']:>26}{x['n']:>5}"
              f"{rho:>8}{x['rmse_after_level_cal_db']:>7.1f}d")
    print(f"\n{'-'*60}\nPer base station:")
    for s, st in sorted(summ["per_base_station"].items(), key=lambda kv: -kv[1]["rho_median"]):
        print(f"  {s:20} cells={st['cells']:>2}  ρ median {st['rho_median']:+.2f}  best {st['rho_best']:+.2f}")
    print(f"-> {REPORTS/'validation_outdoor_summary.json'}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    run(sites=None if args.all else ([args.site] if args.site else None))


if __name__ == "__main__":
    main()
