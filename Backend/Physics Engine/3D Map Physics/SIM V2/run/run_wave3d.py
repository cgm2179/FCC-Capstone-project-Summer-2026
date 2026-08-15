#!/usr/bin/env python3
"""
run_wave3d.py — the SIM V2 3-D wave driver: antenna geometry shapes the emitted lobes.

The 3-D shape the wave takes is determined by the solid geometry (walls/racks) AND
the transmitter antenna geometry. Directivity is applied two ways (plan: "both"):
  * backplane — a physical 3-D conductive reflector behind the source (lobes emerge
    from the full-wave solve); the ground-truth directional teacher.
  * analytic  — multiply an omni solve by the az×el pattern (cheap; any kind).

Extract scene (solid_extract) → shape source (antenna_patterns_3D) → solve
(fullwave3d, JAX) → export re/im volume + dB slices for the viewer.

  python run_wave3d.py --band LTE_B71_617 --antenna panel --azimuth 90 --tilt -5 --directivity backplane
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # engine root, for bootstrap
import bootstrap as B
import antenna_patterns_3D as AP3
import fullwave3d as F3
import solid_extract as SE
from bands_v3 import get

WEB_OUT = B._PHYS.parent / "Frontend" / "simulator"


def solve_shaped(band_label="LTE_B71_617", npw=6.0, crossings=1.5, antenna="omni",
                 azimuth=0.0, tilt=0.0, directivity="backplane", crop_m=None,
                 tx_m=None, max_cells=6_000_000, dtype="float32"):
    """Extract + antenna-shape + solve. Returns dict(U, solid, backend, dt_run, ...)."""
    band = get(band_label)
    s = SE.extract_solid(band, n_per_wavelength=npw, crop_m=crop_m, tx_m=tx_m,
                         max_cells=max_cells)
    classes, tx = s.classes, s.tx_idx
    note = "omni"
    if directivity == "backplane" and AP3.is_directional(antenna):
        bore = AP3.boresight_vec_from_angles(azimuth, tilt)
        classes, nset = AP3.place_backplane_3d(classes, tx, bore, kind=antenna,
                                               offset_cells=2,
                                               radius_cells=max(4, int(0.8 / s.h_m)))
        note = f"backplane({antenna}, az={azimuth}, tilt={tilt}, cells={nset})"
    res = F3.solve3d(classes, s.h_m, band.f_mhz, tx, crossings=crossings,
                     extract_phasor=True, dtype=dtype)
    U = res["phasor"]
    if directivity == "analytic":
        U = AP3.apply_directivity(U, tx, antenna, boresight_az=azimuth, tilt_deg=tilt)
        note = f"analytic({antenna}, az={azimuth}, tilt={tilt})"
    res.update(phasor=U, solid=s, antenna=antenna, directivity=note)
    return res


def export_volume(res, band_label, out=None, export_xz=130, export_y=16):
    """Write re/im cube + solid + JSON header for the CAD wave-mesh viewer."""
    from scipy import ndimage
    U = res["phasor"]; s = res["solid"]; band = get(band_label)
    tX, tY, tZ = min(export_xz, U.shape[0]), min(export_y, U.shape[1]), min(export_xz, U.shape[2])
    f = (tX / U.shape[0], tY / U.shape[1], tZ / U.shape[2])
    Ud = ndimage.zoom(U.real, f, order=1) + 1j * ndimage.zoom(U.imag, f, order=1)
    cls = ndimage.zoom(s.classes, f, order=0).astype(np.uint8)
    amax = float(np.percentile(np.abs(Ud), 99.5)) or 1.0
    out = Path(out) if out else WEB_OUT
    out.mkdir(parents=True, exist_ok=True)
    np.stack([(Ud.real / amax).astype(np.float32),
              (Ud.imag / amax).astype(np.float32)]).tofile(out / "fw_wave3d.bin")
    (cls > 0).astype(np.uint8).tofile(out / "fw_wave3d_solid.bin")
    (out / "fw_wave3d.json").write_text(json.dumps(dict(
        dims=list(Ud.shape), extent_m=[U.shape[i] * s.h_m for i in range(3)],
        band=band.label, f_mhz=band.f_mhz, antenna=res["antenna"],
        directivity=res["directivity"], dtype="float32",
        layout="[2(re,im), X, Y, Z]", source="fdtd-jax")))
    return out / "fw_wave3d.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--npw", type=float, default=6.0)
    ap.add_argument("--crossings", type=float, default=1.5)
    ap.add_argument("--antenna", default="omni")
    ap.add_argument("--azimuth", type=float, default=0.0)
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--directivity", default="backplane", choices=["backplane", "analytic", "none"])
    ap.add_argument("--max-cells", type=int, default=6_000_000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = solve_shaped(a.band, npw=a.npw, crossings=a.crossings, antenna=a.antenna,
                       azimuth=a.azimuth, tilt=a.tilt, directivity=a.directivity,
                       max_cells=a.max_cells)
    s = res["solid"]
    print(f"solve {s.classes.shape} ({s.classes.size/1e6:.2f} M) [{res['backend']}] "
          f"{res['dt_run']:.1f}s {res['mcps']:.0f} Mcps  {res['directivity']}")
    p = export_volume(res, a.band, out=a.out)
    print(f"exported wave volume -> {p}")


if __name__ == "__main__":
    main()
