#!/usr/bin/env python3
"""PR #27 — 3D outdoor-to-indoor (O2I) predicted RSRP volume from a catalog base station.

Runs the implemented facade O2I (`modes_3d.bs_field_3d`): a distant macro cell lights the
building envelope; each lit facade voxel re-radiates inward through `SceneV3.crossing_loss`
(power-summed, disk-cached). Converts PL(x,y,z) → RSRP with `Construct_Reciever_3D`. Saves
the RSRP volume, the receiver-height slice (y_vox from registration_3d.json), and a PNG.

The additive level (EIRP/O2I) is calibratable (as in 2D); the spatial shape is the physics.

usage:
  python predict_o2i_3d.py --pci 397 --band 13                 # Forte Hall anchor
  python predict_o2i_3d.py --pci 396 --band 13 --bearing-deg 90  # diagnostic: reflected arrival
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SIM3D = REPO / "Physics Engine" / "3D Map Physics" / "SIM V1 3D"
RX_DIR = REPO / "Physics Engine" / "3D Map Physics" / "Object and Tranmission" / "Reciever Objects"
CATALOG = REPO / "Physics Engine" / "Cross_Validation" / "base_station_catalog" / "base_stations.csv"
REG3D = SIM3D / "registration_3d.json"
OUT = HERE / "out"
for p in (SIM3D, RX_DIR):
    sys.path.insert(0, str(p))
import modes_3d as MD                     # noqa: E402
import Construct_Reciever_3D as RX        # noqa: E402


def find_bs_row(pci, band, network, carrier=None):
    rows = []
    with open(CATALOG, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["pci"] == str(pci) and r["band"] == str(band)
                    and (network is None or r["network"] == network)
                    and (carrier is None or r["carrier"] == carrier)):
                rows.append(r)
    if not rows:
        raise SystemExit(f"no catalog row for pci={pci} band={band} network={network}")
    rows.sort(key=lambda r: (r["carrier_match"] != "exact", -int(r["n_points"] or 0)))
    return rows[0]


def predict(pci=397, band=13, network="lte", carrier=None,
            eirp_dbm=62.0, o2i_class="low_loss", max_sources=MD.DEFAULT_O2I_SOURCES,
            bearing_deg=None):
    reg = json.loads(REG3D.read_text())
    rx_y = int(reg["rx_y_vox"])

    row = find_bs_row(pci, band, network, carrier)
    lon, lat = float(row["lon"]), float(row["lat"])
    elev = float(row["building_height_m"]) if row["building_height_m"] else None
    freq_mhz = float(row["freq_mhz"]) if row["freq_mhz"] else 751.0

    scene, man = MD.build_scene("o2i")
    geom = MD.base_station_geometry(lon, lat, elev_m=elev)
    bearing = float(bearing_deg) if bearing_deg is not None else geom["arrival_bearing_deg"]
    dist = geom["distance_m"]

    print(f"BS {row['site_id']} PCI {pci} B{band} {freq_mhz:.0f} MHz  "
          f"d={dist:.0f} m  arrival {bearing:.0f}°  elev {elev} m "
          f"(angle {geom.get('elevation_angle_deg')}°)  o2i={o2i_class}")
    fg = MD.bs_field_3d(scene, bearing, bands=[freq_mhz], o2i_db=MD.O2I_DB[o2i_class],
                        max_sources=max_sources, distance_m=dist, progress=True)

    p = np.asarray(fg.p_incoh[0], np.float64)                 # (NX,NY,NZ) linear power
    pl = np.where(p > 0, -10.0 * np.log10(np.maximum(p, 1e-30)), np.inf).astype(np.float32)
    metrics = RX.metrics_from_pl(pl, eirp_dbm=eirp_dbm, band=RX.BandSpec.from_freq_mhz(freq_mhz))
    rsrp = np.asarray(metrics["rsrp"], np.float32)

    inside = getattr(scene, "inside", None)
    if inside is None:
        inside = np.load(SIM3D / "inside_mask.npy")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{pci}_{freq_mhz:.0f}MHz" + (f"_b{int(bearing_deg)}" if bearing_deg is not None else "")
    np.savez_compressed(OUT / f"metrics3d_bs_{tag}.npz",
                        rsrp=rsrp, pl=pl, inside=inside, rx_y_vox=rx_y,
                        px_to_vox=reg["px_to_vox_scale"],
                        meta=json.dumps(dict(site_id=row["site_id"], pci=pci, band=band,
                            network=network, lon=lon, lat=lat, elev_m=elev,
                            distance_m=dist, bearing_deg=bearing, freq_mhz=freq_mhz,
                            eirp_dbm=eirp_dbm, o2i_class=o2i_class,
                            elevation_angle_deg=geom.get("elevation_angle_deg"),
                            measured_mean=float(row["rsrp_mean"]) if row["rsrp_mean"] else None,
                            n_facade_sources=fg.meta.get("n_facade_sources"))))

    _render_slice(rsrp[:, rx_y, :], inside[:, rx_y, :], pci, freq_mhz, row, bearing,
                  OUT / f"slice3d_bs_{tag}.png")

    sl = rsrp[:, rx_y, :][inside[:, rx_y, :]]
    print(f"  facade sources: {fg.meta.get('n_facade_sources')}  rx slice y_vox={rx_y}")
    print(f"  predicted RSRP @ rx slice: median {np.median(sl):.1f}  p10 {np.percentile(sl,10):.1f}"
          f"  max {sl.max():.1f} dBm  (measured mean {row['rsrp_mean']} dBm)")
    print(f"  -> {OUT / f'metrics3d_bs_{tag}.npz'}")
    return OUT / f"metrics3d_bs_{tag}.npz"


def _render_slice(rsrp_slice, inside_slice, pci, freq, row, bearing, out_png):
    # rsrp_slice is (NX, NZ); show with NX horizontal (floor-plan width), NZ vertical
    shown = np.where(inside_slice, rsrp_slice, np.nan).T          # (NZ, NX) for imshow
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad((0.93, 0.93, 0.93))
    fig, ax = plt.subplots(figsize=(12, 5.4))
    im = ax.imshow(shown, cmap=cmap, origin="upper", aspect="auto")
    ax.set_title(f"Predicted RSRP (3D O2I, rx-height slice) — {row['site_id']} · PCI {pci} · "
                 f"{freq:.0f} MHz · arrival {bearing:.0f}°")
    ax.set_xlabel("voxel X (floor-plan width)"); ax.set_ylabel("voxel Z (floor-plan depth)")
    fig.colorbar(im, ax=ax, label="predicted RSRP (dBm)", shrink=0.85)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pci", type=int, default=397)
    ap.add_argument("--band", type=int, default=13)
    ap.add_argument("--network", default="lte")
    ap.add_argument("--carrier", default=None)
    ap.add_argument("--eirp-dbm", type=float, default=62.0)
    ap.add_argument("--o2i-class", default="low_loss")
    ap.add_argument("--max-sources", type=int, default=MD.DEFAULT_O2I_SOURCES)
    ap.add_argument("--bearing-deg", type=float, default=None,
                    help="override arrival bearing (deg); default = site geometry")
    args = ap.parse_args()
    predict(pci=args.pci, band=args.band, network=args.network, carrier=args.carrier,
            eirp_dbm=args.eirp_dbm, o2i_class=args.o2i_class, max_sources=args.max_sources,
            bearing_deg=args.bearing_deg)


if __name__ == "__main__":
    main()
