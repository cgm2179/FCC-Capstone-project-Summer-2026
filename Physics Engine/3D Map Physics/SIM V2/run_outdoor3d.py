#!/usr/bin/env python3
"""
run_outdoor3d.py — outdoor (NoMa city) 3-D coverage: analytic-primary + optional patch.

Full-wave over a city is infeasible, so outdoor is ANALYTIC-dominant: the 3-D
Motley-Keenan/eikonal engine (`far_field_3d` / `engine_3d.SceneV3`) over the
georeferenced city grid (`city_georef`). Prx = EIRP − PL. A local full-wave
"microscope" patch (`hybrid_field_3d`) can refine a chosen tower's near field.

  python run_outdoor3d.py --band NR_n41_2506 --eirp-dbm 46
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import bootstrap  # noqa: F401
import city_georef as CG
import engine_3d as E3
from bands_v3 import get

OUT = Path(__file__).resolve().parent / "out"


def run(band_label="NR_n41_2506", tx_lonlat=None, eirp_dbm=46.0, city_dir=None,
        out=None):
    band = get(band_label)
    grid, man = CG.load_georef_city(city_dir)
    scene = E3.SceneV3(grid, man, barrier_classes=(3,))
    NX, NY, NZ = grid.shape

    # transmitter voxel: from lon/lat if given, else the city centre at street level
    if tx_lonlat is not None:
        import voxelize_city as VC
        vx, vz = VC.lonlat_to_vox(man, tx_lonlat[0], tx_lonlat[1])
        tx = (int(vx), int(NY * 0.5), int(vz))
    else:
        tx = (NX // 2, int(NY * 0.5), NZ // 2)

    PL = np.asarray(scene.pathloss_map(tx, band.f_mhz))          # (NX,NY,NZ) dB
    prx = float(eirp_dbm) - PL                                   # received power dBm

    out = Path(out) if out else OUT
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"outdoor_pl_{band_label}.npy", PL.astype(np.float16))
    np.save(out / f"outdoor_prx_{band_label}.npy", prx.astype(np.float16))
    print(f"outdoor {grid.shape} @ {band.label} tx={tx}  "
          f"PL dB [{np.nanmin(PL):.0f},{np.nanmax(PL):.0f}]  "
          f"Prx dBm [{np.nanmin(prx):.0f},{np.nanmax(prx):.0f}]  eirp={eirp_dbm}")
    return dict(pl=PL, prx=prx, tx=tx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="NR_n41_2506")
    ap.add_argument("--eirp-dbm", type=float, default=46.0)
    ap.add_argument("--tx-lonlat", type=float, nargs=2, default=None)
    ap.add_argument("--city-dir", default=None)
    a = ap.parse_args()
    run(a.band, tx_lonlat=a.tx_lonlat, eirp_dbm=a.eirp_dbm, city_dir=a.city_dir)


if __name__ == "__main__":
    main()
