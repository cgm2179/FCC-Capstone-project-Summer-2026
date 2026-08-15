#!/usr/bin/env python3
"""
run_indoor3d.py — indoor 7th-floor 3-D coverage pipeline (full-wave, JAX).

Extract the whole floor (or a crop) → antenna-shaped full-wave solve (reuses
`run_wave3d.solve_shaped`) → steady-state RMS field-strength → relative-dB coverage
volume + horizontal/vertical slice PNGs. The indoor product built on the SIM V2
engine (`run_outdoor3d` is the city analog).

  python run_indoor3d.py --band LTE_B71_617 --npw 4 --antenna omni
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # engine root, for bootstrap
import bootstrap  # noqa: F401
from bands_v3 import get
from run_wave3d import solve_shaped, export_volume

OUT = Path(__file__).resolve().parent.parent / "out"


def run(band_label="LTE_B71_617", npw=4.0, crossings=1.5, antenna="omni",
        azimuth=0.0, tilt=0.0, directivity="backplane", max_cells=6_000_000,
        out=None, plot=True):
    res = solve_shaped(band_label, npw=npw, crossings=crossings, antenna=antenna,
                       azimuth=azimuth, tilt=tilt, directivity=directivity,
                       max_cells=max_cells)
    U = res["phasor"]; s = res["solid"]
    air = (s.classes == 0)
    amp = np.abs(U)
    ref = np.percentile(amp[air][amp[air] > 0], 99.5) or 1.0
    db = 20.0 * np.log10(np.maximum(amp, ref * 1e-6) / ref)      # relative dB coverage
    out = Path(out) if out else OUT
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"indoor_db_{band_label}.npy", db.astype(np.float16))
    export_volume(res, band_label, out=out)
    print(f"indoor {s.classes.shape} [{res['backend']}] {res['dt_run']:.1f}s "
          f"{res['mcps']:.0f} Mcps  {res['directivity']}")

    if plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k0 = s.classes.shape[1] // 2
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        hz = np.where(air[:, k0, :], db[:, k0, :], np.nan)
        ax[0].imshow(hz.T, origin="lower", cmap="turbo", vmin=-40, vmax=0, aspect="equal")
        ax[0].set_title(f"horizontal slice — {band_label}")
        tx = res["solid"].tx_idx
        vt = np.where(air[:, :, tx[2]], db[:, :, tx[2]], np.nan)
        ax[1].imshow(vt.T, origin="lower", cmap="turbo", vmin=-40, vmax=0, aspect="auto")
        ax[1].set_title("vertical slice through Tx")
        fig.suptitle(f"SIM V2 indoor 3-D — {band_label} ({get(band_label).f_mhz:.0f} MHz)")
        fig.tight_layout(); fig.savefig(out / f"indoor_{band_label}.png", dpi=120); plt.close(fig)
        print(f"wrote {out / f'indoor_{band_label}.png'}")
    return db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--npw", type=float, default=4.0)
    ap.add_argument("--crossings", type=float, default=1.5)
    ap.add_argument("--antenna", default="omni")
    ap.add_argument("--azimuth", type=float, default=0.0)
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--directivity", default="backplane", choices=["backplane", "analytic", "none"])
    ap.add_argument("--max-cells", type=int, default=6_000_000)
    a = ap.parse_args()
    run(a.band, npw=a.npw, crossings=a.crossings, antenna=a.antenna, azimuth=a.azimuth,
        tilt=a.tilt, directivity=a.directivity, max_cells=a.max_cells)


if __name__ == "__main__":
    main()
