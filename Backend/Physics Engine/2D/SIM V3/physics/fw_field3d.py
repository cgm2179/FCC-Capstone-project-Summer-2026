#!/usr/bin/env python3
"""
fw_field3d.py — 3-D full-wave complex field for the three.js animated viewer.

FullWaveScene is dimension-agnostic, so this runs the SAME FDTD on a 3-D indoor
sub-volume and extracts the steady-state complex phasor U(x,y,z). It exports a
compact volume (re/im, float16) + a JSON header that
`Frontend/simulator/fw_field3d_viewer.html` loads and animates as
Re{U·e^{jωt}} — the 2-D wave animation, now in 3-D, in three.js.

The surrogate path (fw_unet3d) predicts the same U in ms; this FDTD path is the
ground truth / clean demo source.

  python fw_field3d.py --band LTE_B71_617 --npw 5 --size 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import _bootstrap as B
import fullwave2d as FW
from bands_v3 import get
from fullwave2d import FullWaveScene

WEB_OUT = B._PHYS.parent / "Frontend" / "simulator"    # served by the static server


def field3d(band_label="LTE_B71_617", npw=5.0, region_m=(8.0, 5.0, 8.0),
            export_side=40, crossings=2.0):
    band = get(band_label)
    grid = np.load(B.MATERIAL_GRID)                     # (X, Y_up, Z) 0.3 m
    man = json.loads(B.MANIFEST.read_text())
    cs = float(man["cell_size_m"]); h = band.cell_size_m(npw); zoom = cs / h
    valid = np.load(B.VALID_TX_MASK)

    # a sub-volume centred on a valid Tx (full height, region in X/Z)
    tx = np.argwhere(valid); tc = tx[len(tx) // 2]
    rx, rz = int(region_m[0] / cs / 2), int(region_m[2] / cs / 2)
    x0, x1 = max(0, tc[0] - rx), min(grid.shape[0], tc[0] + rx)
    z0, z1 = max(0, tc[2] - rz), min(grid.shape[2], tc[2] + rz)
    sub = grid[x0:x1, :, z0:z1]
    fine = ndimage.zoom(sub, zoom, order=0).astype(np.int8)
    txf = tuple(int(v) for v in ((tc[0] - x0) * zoom, tc[1] * zoom, (tc[2] - z0) * zoom))
    txf = tuple(int(np.clip(txf[i], 1, fine.shape[i] - 2)) for i in range(3))

    sim = FullWaveScene(fine, h, band.f_mhz, txf, source="cw")
    steps = int(round(crossings * max(fine.shape) * h / 299_792_458.0 / sim.dt))
    print(f"3-D FDTD  grid {fine.shape} ({fine.size/1e6:.2f} M)  h={h*100:.1f}cm  "
          f"steps={steps}  CFL={sim.cfl():.3f}")
    res = FW.simulate(sim, steps, warmup_frac=0.6, extract_phasor=True,
                      show_tqdm=True, desc=f"3D {band.label}")
    if not res["finite"]:
        raise FloatingPointError("3-D field blew up")
    U = res["phasor"]

    # downsample to a browser-friendly cube; carry re/im (already phase-INCLUSIVE)
    f = np.array([export_side / s for s in U.shape])
    Ud = ndimage.zoom(U.real, f, order=1) + 1j * ndimage.zoom(U.imag, f, order=1)
    cls = ndimage.zoom(fine, f, order=0).astype(np.int8)
    amax = float(np.percentile(np.abs(Ud), 99.5)) or 1.0

    WEB_OUT.mkdir(parents=True, exist_ok=True)
    re = (Ud.real / amax).astype(np.float32)              # float32 → JS Float32Array
    im = (Ud.imag / amax).astype(np.float32)
    solid = (cls > 0).astype(np.uint8)
    np.stack([re, im]).tofile(WEB_OUT / "fw_field3d.bin")       # [2, X, Y, Z] f32
    solid.tofile(WEB_OUT / "fw_field3d_solid.bin")             # [X, Y, Z] u8
    (WEB_OUT / "fw_field3d.json").write_text(json.dumps(dict(
        dims=list(Ud.shape), h_m=h * (U.shape[0] / export_side), band=band.label,
        f_mhz=band.f_mhz, tx=[int(t * export_side / U.shape[i]) for i, t in enumerate(txf)],
        dtype="float32", layout="[2(re,im), X, Y, Z]",
        note="animate Re{(re+j im)*exp(j*2pi*phase)} over phase in [0,1)")))
    print(f"exported {Ud.shape} cube -> {WEB_OUT}/fw_field3d.{{bin,json}} "
          f"({(2*re.size*4)/1e6:.1f} MB)")
    return Ud


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--npw", type=float, default=5.0)
    ap.add_argument("--size", type=int, default=40)
    args = ap.parse_args()
    field3d(args.band, npw=args.npw, export_side=args.size)


if __name__ == "__main__":
    main()
