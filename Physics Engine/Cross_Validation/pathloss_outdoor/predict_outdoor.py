#!/usr/bin/env python3
"""Phase 2 — outdoor path loss per donor base station on the georeferenced NoMa tile.

The outdoor analog of the indoor O2I predictor, applying the BS4–BS7 lessons: place each donor
at its TRUE lat/lon/elevation (geometry drives ρ) and let reflection + diffraction fill the
street-canyon NLOS the direct field can't. Runs `precompute_volumes.solve_one` (point source →
path_loss [+ reflection + diffraction] → Combine_3D) on `city/NoMa_FCC_tile`, converts PL → RSRP
via `Construct_Reciever_3D`, and exports the ground-plane slice (+ meta) that `validate_outdoor.py`
samples against the 7/24 walk.

usage:
  python predict_outdoor.py --site BS7_forte_hall --freq 751          # path_loss only (fast smoke)
  python predict_outdoor.py --site BS7_forte_hall --freq 751 --mechanisms path_loss,reflection,diffraction
  python predict_outdoor.py --all --freq 751
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SIM3D = REPO / "Physics Engine" / "3D Map Physics" / "SIM V1 3D"
RX_DIR = REPO / "Physics Engine" / "3D Map Physics" / "Object and Tranmission" / "Reciever Objects"
TILE = SIM3D / "city" / "NoMa_FCC_tile"
OUT = HERE / "out"
for p in (SIM3D, RX_DIR):
    sys.path.insert(0, str(p))
import engine_3d                          # noqa: E402
import modes_3d as MD                     # noqa: E402
import precompute_volumes as PV           # noqa: E402
import Construct_Reciever_3D as RX        # noqa: E402

# Donor sites (lat, lon, antenna-height hint AGL). The hint is a fallback — city_base_station_vox
# snaps horizontally to the tallest rooftop within 30 m of the pin and mounts one voxel above it.
DONORS = {
    "BS7_forte_hall":     dict(lat=38.90155, lon=-77.01142, elev=20),
    "BS5_1140_n_capitol": dict(lat=38.90476, lon=-77.00920, elev=33),
    "BS6_1005_n_capitol": dict(lat=38.90309, lon=-77.00848, elev=50),
    "BS4_55_m_st":        dict(lat=38.90587, lon=-77.01110, elev=50),
    "BS2_900_nj_ave":     dict(lat=38.90197, lon=-77.01423, elev=21),
}


def predict(site_id, freq_mhz, mechanisms=("path_loss",), eirp_dbm=62.0, rx_h_m=1.5):
    if site_id not in DONORS:
        raise SystemExit(f"unknown site {site_id}; have {list(DONORS)}")
    scene, man = engine_3d.load_scene(TILE, barrier_classes=(3,))
    if scene.inside is None:
        scene.inside = np.load(TILE / "inside_mask.npy")
    d = DONORS[site_id]
    tx = MD.city_base_station_vox(man, d["lon"], d["lat"], d["elev"], scene=scene)
    NX, NY, NZ = scene.M.shape
    roof_h = int((scene.M[tx[0], :, tx[2]] > 0).sum()) * scene.cell

    print(f"{site_id}  {freq_mhz:.0f} MHz  tx voxel {tx} (roof {roof_h:.0f} m, antenna y={tx[1]*scene.cell:.0f} m)  "
          f"mechanisms={'+'.join(mechanisms)}")
    t0 = time.time()
    pl, t, cf, contribs = PV.solve_one(scene, tx, bands=[freq_mhz], mechanisms=list(mechanisms),
                                       backend="numpy", mode="outdoor")
    plb = np.asarray(pl[0], np.float32)                          # single requested band -> pl[0]
    metrics = RX.metrics_from_pl(plb, eirp_dbm=eirp_dbm, band=RX.BandSpec.from_freq_mhz(freq_mhz))
    rsrp = np.asarray(metrics["rsrp"], np.float32)
    print(f"  solved in {time.time()-t0:.1f}s")

    # pedestrian plane = lowest y-layer with real street air (y=0 is the solid ground datum)
    air_per_y = scene.inside.sum(axis=(0, 2))
    iy = int(np.argmax(air_per_y > 0.05 * NX * NZ))
    iy = max(iy, int(round(rx_h_m / scene.cell)))
    rsrp_g = rsrp[:, iy, :]                                      # (NX,NZ)
    street = scene.inside[:, iy, :]                              # air at street level
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{site_id}_{freq_mhz:.0f}MHz_{'_'.join(mechanisms)}"
    np.savez_compressed(OUT / f"outdoor_{tag}.npz",
                        rsrp_ground=rsrp_g, pl_ground=plb[:, iy, :].astype(np.float32),
                        street=street, tx=np.array(tx), rx_y=iy,
                        meta=json.dumps(dict(site_id=site_id, lon=d["lon"], lat=d["lat"],
                            elev_m=d["elev"], freq_mhz=freq_mhz, mechanisms=list(mechanisms),
                            eirp_dbm=eirp_dbm, rx_h_m=rx_h_m, tile=str(TILE.name),
                            grid_shape=[NX, NY, NZ], roof_h_m=roof_h)))
    _render(rsrp_g, scene.M[:, iy, :], street, tx, site_id, freq_mhz, mechanisms,
            OUT / f"outdoor_{tag}.png")

    sv = rsrp_g[street]; sv = sv[np.isfinite(sv)]
    print(f"  ground RSRP (street): median {np.median(sv):.1f}  p10 {np.percentile(sv,10):.1f}  "
          f"p90 {np.percentile(sv,90):.1f} dBm  ({int(street.sum()):,} street voxels)")
    print(f"  -> {OUT / f'outdoor_{tag}.npz'}")
    return OUT / f"outdoor_{tag}.npz"


def _render(rsrp_g, mat_g, street, tx, site_id, freq, mechs, out_png):
    shown = np.where(street, rsrp_g, np.nan).T                  # (NZ,NX)
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad((0.9, 0.9, 0.9))
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(shown, origin="lower", cmap=cmap, aspect="equal")
    walls = (mat_g > 0).T
    ov = np.zeros(walls.shape + (4,)); ov[walls] = (0, 0, 0, 0.45)
    ax.imshow(ov, origin="lower", aspect="equal")
    ax.plot(tx[0], tx[2], "r*", ms=16, mec="white", label=f"{site_id} Tx")
    ax.set_title(f"Outdoor ground RSRP — {site_id} · {freq:.0f} MHz · {'+'.join(mechs)}")
    ax.set_xlabel("voxel X"); ax.set_ylabel("voxel Z"); ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="predicted RSRP (dBm)", shrink=0.8)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="BS7_forte_hall")
    ap.add_argument("--all", action="store_true", help="predict every donor at --freq")
    ap.add_argument("--freq", type=float, default=751.0)
    ap.add_argument("--mechanisms", default="path_loss",
                    help="comma list: path_loss[,reflection][,diffraction]")
    ap.add_argument("--eirp-dbm", type=float, default=62.0)
    args = ap.parse_args()
    mechs = tuple(m.strip() for m in args.mechanisms.split(","))
    sites = list(DONORS) if args.all else [args.site]
    for s in sites:
        predict(s, args.freq, mechanisms=mechs, eirp_dbm=args.eirp_dbm)


if __name__ == "__main__":
    main()
