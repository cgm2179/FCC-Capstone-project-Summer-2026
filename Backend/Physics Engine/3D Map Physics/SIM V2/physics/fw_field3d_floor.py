#!/usr/bin/env python3
"""
fw_field3d_floor.py — 3-D full-wave complex field over the WHOLE 7th floor.

SIM V2 port of `2D/SIM V3/fw_field3d_floor.py`: whole-building solve at a
visualization resolution (~λ/4), on the JAX/XLA hot path. Exports a downsampled
re/im cube + metric extent for the CAD wave-mesh viewer, and prints the gen time
for the FDTD-vs-surrogate timing table.

  python fw_field3d_floor.py --band LTE_B71_617 --npw 4
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy import ndimage

import bootstrap as B
from bands_v3 import get
from fullwave2d import FullWaveScene
from fw_fdtd_jax import simulate_jax

WEB_OUT = B._PHYS.parent / "Frontend" / "simulator"
C0 = 299_792_458.0


def floor_field(band_label="LTE_B71_617", npw=4.0, crossings=2.0,
                export_xz=130, export_y=16, dtype="float32"):
    band = get(band_label)
    grid = np.load(B.MATERIAL_GRID)
    man = json.loads(B.MANIFEST.read_text())
    cs = float(man["cell_size_m"]); h = band.cell_size_m(npw); zoom = cs / h
    valid = np.load(B.VALID_TX_MASK)

    fine = ndimage.zoom(grid, zoom, order=0).astype(np.int8)
    tc = np.argwhere(valid); c = tc[len(tc) // 2]
    tx = tuple(int(np.clip(round(v * zoom), 1, fine.shape[i] - 2))
               for i, v in enumerate(c))
    print(f"3-D floor FDTD (JAX)  grid {fine.shape} ({fine.size/1e6:.1f} M cells)  "
          f"h={h*100:.1f} cm  tx={tx}")

    sim = FullWaveScene(fine, h, band.f_mhz, tx, source="cw")
    steps = int(round(crossings * max(fine.shape) * h / C0 / sim.dt))
    t0 = time.time()
    res = simulate_jax(sim, steps, warmup_frac=0.6, extract_phasor=True, dtype=dtype)
    gen_s = time.time() - t0
    if not res["finite"]:
        raise FloatingPointError("floor field blew up")
    U = res["phasor"]
    print(f"FDTD floor field [{res['backend']}]: {steps} steps in {gen_s:.1f} s "
          f"({res['mcps']:.0f} Mcell-updates/s)")

    tX, tY, tZ = min(export_xz, U.shape[0]), min(export_y, U.shape[1]), min(export_xz, U.shape[2])
    f = (tX / U.shape[0], tY / U.shape[1], tZ / U.shape[2])
    Ud = ndimage.zoom(U.real, f, order=1) + 1j * ndimage.zoom(U.imag, f, order=1)
    cls = ndimage.zoom(fine, f, order=0).astype(np.uint8)
    amax = float(np.percentile(np.abs(Ud), 99.5)) or 1.0
    txd = [int(t * f[i]) for i, t in enumerate(tx)]

    WEB_OUT.mkdir(parents=True, exist_ok=True)
    np.stack([(Ud.real / amax).astype(np.float32),
              (Ud.imag / amax).astype(np.float32)]).tofile(WEB_OUT / "fw_floor.bin")
    (cls > 0).astype(np.uint8).tofile(WEB_OUT / "fw_floor_solid.bin")
    (WEB_OUT / "fw_floor.json").write_text(json.dumps(dict(
        dims=list(Ud.shape), extent_m=[U.shape[i] * h for i in range(3)],
        band=band.label, f_mhz=band.f_mhz, tx=txd, dtype="float32",
        layout="[2(re,im), X, Y, Z]", source="fdtd-jax", gen_seconds=round(gen_s, 1))))
    print(f"exported {Ud.shape} cube ({(2*Ud.size*4)/1e6:.1f} MB) -> {WEB_OUT}/fw_floor.*")
    return dict(gen_seconds=gen_s, cells=int(fine.size), steps=steps, dims=list(Ud.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--npw", type=float, default=4.0)
    ap.add_argument("--crossings", type=float, default=2.0)
    args = ap.parse_args()
    floor_field(args.band, npw=args.npw, crossings=args.crossings)


if __name__ == "__main__":
    main()
