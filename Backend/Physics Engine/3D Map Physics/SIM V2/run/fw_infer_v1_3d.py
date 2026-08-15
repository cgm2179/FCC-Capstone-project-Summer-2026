#!/usr/bin/env python3
"""
fw_infer_v1_3d.py — 3-D surrogate inference + the g2 validation gate.

SIM V2 3-D port of `2D/SIM V3/fw_infer.py`. Reuses `fw_unet3d.tiled_predict3d` for
overlap-tiled whole-volume prediction, and validates the surrogate against a FRESH
3-D FDTD solve of the same held-out scene (`fullwave3d`): envelope-dB RMSE, Spearman
ρ, complex coherence — the same three metrics as the 2-D gate.

  python fw_infer_v1_3d.py --ckpt fw_unet3d.pt --band LTE_B71_617 --validate
"""
from __future__ import annotations

import argparse

import numpy as np

import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # engine root, for bootstrap
import bootstrap  # noqa: F401
import fullwave3d as F3
import solid_extract as SE
from bands_v3 import get


def _db(a, ref):
    return 20.0 * np.log10(np.maximum(a, ref * 1e-6) / ref)


def predict_field(model, classes, tx, h, f_mhz, box=24, stride=18):
    """Whole-volume complex U via the 3-D surrogate (tiled)."""
    from fw_unet3d import tiled_predict3d
    return tiled_predict3d(model, classes, tx, h, f_mhz, box=box, stride=stride)


def validate(ckpt="fw_unet3d.pt", band_label="LTE_B71_617", npw=6.0, region_m=8.0,
             seed=99, max_cells=1_500_000):
    """g2 gate: surrogate vs a fresh 3-D FDTD on a held-out sub-volume."""
    from scipy.stats import spearmanr
    from fw_unet3d import load_model3d
    band = get(band_label)

    # held-out Tx sub-volume
    vtx = np.load(__import__("bootstrap").VALID_TX_MASK)
    import bootstrap as B
    man = __import__("json").loads(B.MANIFEST.read_text())
    cs = float(man["cell_size_m"])
    txc = np.argwhere(vtx.any(axis=1))
    tc = txc[np.random.default_rng(seed).integers(len(txc))]
    txm = (float(tc[0]) * cs, float(tc[1]) * cs)
    crop = (txm[0] - region_m, txm[0] + region_m, 0.0, man["grid_shape"][1] * cs,
            txm[1] - region_m, txm[1] + region_m)
    s = SE.extract_solid(band, n_per_wavelength=npw,
                         crop_m=crop, tx_m=(txm[0], (man["grid_shape"][1] // 2) * cs, txm[1]),
                         max_cells=max_cells)
    res = F3.solve3d(s.classes, s.h_m, band.f_mhz, s.tx_idx, crossings=2.0,
                     extract_phasor=True, dtype="float64")
    Utrue = res["phasor"]

    model = load_model3d(str((__import__("pathlib").Path(__file__).resolve().parent.parent / ckpt)))
    Upred = predict_field(model, s.classes, s.tx_idx, s.h_m, band.f_mhz)

    air = (s.classes == 0)
    at, ap = np.abs(Utrue), np.abs(Upred)
    reft = np.percentile(at[air][at[air] > 0], 99.0)
    refp = np.percentile(ap[air][ap[air] > 0], 99.0)
    et, ep = _db(at, reft), _db(ap, refp)
    m = air & (et > -60)
    rmse = float(np.sqrt(np.mean((et[m] - ep[m]) ** 2)))
    rho, _ = spearmanr(et[m], ep[m])
    a, b = Upred[m], Utrue[m]
    coh = float(np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"g2 3-D @ {band.label}  grid {s.classes.shape}  tx=({tc[0]},{tc[1]})")
    print(f"   envelope dB RMSE = {rmse:.2f} dB   Spearman = {rho:+.3f}   coherence = {coh:.3f}")
    return dict(rmse_db=rmse, spearman=float(rho), coherence=coh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="fw_unet3d.pt")
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--npw", type=float, default=6.0)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if a.validate:
        validate(a.ckpt, a.band, npw=a.npw)
    else:
        print("use --validate (g2 gate vs fresh 3-D FDTD)")


if __name__ == "__main__":
    main()
