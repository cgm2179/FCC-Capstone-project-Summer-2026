#!/usr/bin/env python3
"""
validate_scanner_3d.py — M5 gates against PCTEL / NR walk measurements (preview).

Full M5 (per-donor fixed effects, delay-spread combiner check, material
calibration) still needs a known-Tx walk and solid georef. This module ships the
pieces that can run *today* on the registered indoor records:

  V0  invariants — finite metrics, RSRP in a plausible window
  V1  Spearman ρ(sim_PL, −RSRP) ≥ 0.6 on held-out cells (shape gate)
  V2  RMSE after a single per-PCI offset ≤ 8 dB (preview; not full fixed effects)

usage:
  python validate_scanner_3d.py --records Data/records_data.js \\
      --sim-dir "Physics Engine/3D Map Physics/SIM V1 3D" \\
      --pl-volume web/volumes/pl_volume_tx_66-5-54.bin --txid tx_66-5-54

  # or from phase_c3 after training (surrogate PL volume in memory):
  from validate_scanner_3d import preview_gates
  report = preview_gates(records, pl_db_xyz, reg, eirp_dbm=20.0)
"""
from __future__ import annotations

import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # SIM V1 3D root (siblings live there)
import argparse
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent.parent          # repo root (…/indoor-walk-test)
MECH = HERE.parent / "Object and Tranmission" / "Reciever Objects"
import sys
for p in (HERE, MECH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from Construct_Reciever_3D import BandSpec, rsrp_from_pl
except ImportError:
    # sibling import when cwd is Reciever Objects
    from Construct_Reciever_3D import BandSpec, rsrp_from_pl  # type: ignore


def load_records_js(path: Path) -> list[dict]:
    """Parse Data/records_data.js (assigns a JS array literal)."""
    text = Path(path).read_text()
    m = re.search(r"=\s*(\[.*\])\s*;?\s*$", text, re.S)
    if not m:
        raise ValueError(f"{path} does not look like a JS array assignment")
    return json.loads(m.group(1))


def px_to_vox(px, py, reg: dict):
    """Floor-plan px → (i, k) voxel using registration_3d.json scale."""
    sx = float(reg["px_to_vox_scale"]["x"])
    sz = float(reg["px_to_vox_scale"]["z"])
    return int(round(px * sx)), int(round(py * sz))


def aggregate_cells(records, reg, *, min_samples: int = 3):
    """Median-combine ≥min_samples per (i, k, band) to suppress fast fading."""
    y_vox = int(reg.get("rx_y_vox", 5))
    nx, _, nz = reg["grid_shape"]
    buckets: dict[tuple, list] = {}
    for r in records:
        if r.get("rsrp") is None or r.get("px") is None:
            continue
        i, k = px_to_vox(float(r["px"]), float(r["py"]), reg)
        if not (0 <= i < nx and 0 <= k < nz):
            continue
        band = int(r.get("band") or 0)
        key = (i, y_vox, k, band, round(float(r["freq"]), 1))
        buckets.setdefault(key, []).append(float(r["rsrp"]))
    rows = []
    for (i, j, k, band, freq), vals in buckets.items():
        if len(vals) < min_samples:
            continue
        rows.append(dict(i=i, j=j, k=k, band=band, freq_mhz=freq,
                         rsrp_dbm=float(np.median(vals)), n=len(vals)))
    return rows


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan")
    return float(spearmanr(a[m], b[m]).correlation)


def preview_gates(records, pl_volume, reg: dict, *, eirp_dbm: float = 20.0,
                  bands=None, min_samples: int = 1, holdout: float = 0.2,
                  seed: int = 0) -> dict:
    """Run V0/V1/V2 preview against a PL volume shaped (nf, nx, ny, nz) or (nx,ny,nz).

    `bands` is the frequency list matching axis 0 of a 4-D volume. If PL is 3-D,
    every record uses that single map.
    """
    pl = np.asarray(pl_volume, np.float32)
    rows = aggregate_cells(records, reg, min_samples=min_samples)
    if not rows:
        return {"ok": False, "reason": "no records mapped into the grid",
                "n_records": len(records)}

    rng = np.random.default_rng(seed)
    hold = rng.random(len(rows)) < holdout

    sim_pl, meas_rsrp, freqs = [], [], []
    for r in rows:
        if pl.ndim == 4:
            if not bands:
                raise ValueError("4-D PL volume needs bands=")
            bi = int(np.argmin([abs(float(b) - r["freq_mhz"]) for b in bands]))
            v = float(pl[bi, r["i"], r["j"], r["k"]])
        else:
            v = float(pl[r["i"], r["j"], r["k"]])
        if not np.isfinite(v) or v >= 200:
            continue
        sim_pl.append(v)
        meas_rsrp.append(r["rsrp_dbm"])
        freqs.append(r["freq_mhz"])
    sim_pl = np.asarray(sim_pl, float)
    meas_rsrp = np.asarray(meas_rsrp, float)
    # align hold mask to kept rows — recompute simply
    hold = rng.random(len(sim_pl)) < holdout

    # Predicted RSRP from sim PL (single global EIRP; per-PCI offset fitted on train)
    band = BandSpec.from_freq_mhz(float(np.median(freqs)) if len(freqs) else 2000.0)
    pred_rsrp = rsrp_from_pl(sim_pl, eirp_dbm=eirp_dbm, band=band)

    # V1: spatial structure — high PL should mean low RSRP → ρ(sim_PL, −RSRP) ≥ 0.6
    rho = spearman(sim_pl[~hold] if (~hold).sum() >= 5 else sim_pl,
                   -meas_rsrp[~hold] if (~hold).sum() >= 5 else -meas_rsrp)

    # V2: one global offset (stand-in for per-donor fixed effects) on train, RMSE on hold
    train = ~hold if (~hold).any() and hold.any() else np.ones(len(sim_pl), bool)
    test = hold if hold.any() and (~hold).any() else train
    offset = float(np.median(meas_rsrp[train] - pred_rsrp[train]))
    resid = meas_rsrp[test] - (pred_rsrp[test] + offset)
    rmse = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else float("nan")

    v0 = bool(np.isfinite(pred_rsrp).all() and np.nanpercentile(pred_rsrp, 50) < -40)
    v1 = bool(np.isfinite(rho) and rho >= 0.6)
    v2 = bool(np.isfinite(rmse) and rmse <= 8.0)

    return {
        "ok": True,
        "n_records": len(records),
        "n_cells": len(sim_pl),
        "n_holdout": int(test.sum()),
        "eirp_dbm": float(eirp_dbm),
        "offset_db": offset,
        "gates": {
            "V0_invariants": {"pass": v0, "detail": "finite RSRP, median < -40 dBm"},
            "V1_spearman": {"pass": v1, "rho": rho, "threshold": 0.6},
            "V2_rmse_db": {"pass": v2, "rmse_db": rmse, "threshold": 8.0},
        },
        "note": "Preview only — full M5 needs per-donor FE, delay-spread, known-Tx walk.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(ROOT / "Data" / "records_data.js"))
    ap.add_argument("--sim-dir", default=str(HERE))
    ap.add_argument("--pl-volume", default=None,
                    help="float16 (nf,nx,ny,nz) or (nx,ny,nz) .bin from web/volumes/")
    ap.add_argument("--txid", default="tx_66-5-54")
    ap.add_argument("--eirp", type=float, default=20.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sim = Path(args.sim_dir)
    reg = json.loads((sim / "registration_3d.json").read_text())
    records = load_records_js(Path(args.records))

    vol_path = Path(args.pl_volume) if args.pl_volume else (
        sim / "web" / "volumes" / f"pl_volume_{args.txid}.bin")
    idx = json.loads((sim / "web" / "volumes" / "index.json").read_text())
    entry = next((e for e in idx["volumes"] if e["txid"] == args.txid), None)
    if entry is None:
        raise SystemExit(f"txid {args.txid} not in index.json")
    shape = tuple(entry["grid_shape"])
    bands = entry["bands"]
    raw = np.fromfile(vol_path, dtype=np.float16)
    nf = len(bands)
    pl = raw.reshape((nf, *shape)).astype(np.float32)

    report = preview_gates(records, pl, reg, eirp_dbm=args.eirp, bands=bands,
                           min_samples=3)
    out = Path(args.out) if args.out else sim / "validation_report_preview.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")

    # Optional scatter PNGs next to the report (skipped if matplotlib missing).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = aggregate_cells(records, reg, min_samples=3)
    if not rows or pl.ndim != 4:
        return
    bi = int(np.argmin([abs(float(b) - 1935.0) for b in bands]))
    xs, ys = [], []
    for r in rows:
        if abs(r["freq_mhz"] - bands[bi]) > 50:
            continue
        xs.append(float(pl[bi, r["i"], r["j"], r["k"]]))
        ys.append(r["rsrp_dbm"])
    if len(xs) < 5:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(xs, ys, s=8, alpha=0.5)
    ax.set_xlabel("sim PL (dB)"); ax.set_ylabel("meas RSRP (dBm)")
    ax.set_title(f"M5 preview · {args.txid} · band≈{bands[bi]} MHz")
    png = out.with_name(out.stem + "_scatter.png")
    fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
