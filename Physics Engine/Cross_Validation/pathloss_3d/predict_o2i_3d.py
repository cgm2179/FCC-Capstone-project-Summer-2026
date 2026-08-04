#!/usr/bin/env python3
"""PR #27 follow-up (Commit 2) — EXPLICIT 3D O2I predicted RSRP on the ACCURATE floor geometry.

Why this shape
--------------
The 3D voxel scene (`Sandbox_Version_3D_Simulation_1.obj`) is a crude/generic model — a full
arrival-bearing sweep shows it caps Spearman ρ at ≈0.22 for ANY direction (grid fidelity, not
physics), while the fine floor-plan grid reaches 0.64 with the same Motley-Keenan physics. And
on a single floor with a near-horizontal donor (Forte elevation angle ≈ 0°) the O2I field is
physically an in-plane problem. So the explicit 3D O2I marches a single PARALLEL wavefront along
the 3D arrival direction (bearing + elevation) over the ACCURATE floor-plan grid, and validates
in the px frame — no registration error, 2D-grade fidelity, genuinely 3D-capable (the elevation
angle tilts the march; the RX-plane field extrudes to a volume for the OSM drape).

    PL(x,y) = outdoor_leg(log-distance) + O2I_penetration + indoor_walls(x,y)

`indoor_walls` (the parallel-ray Motley-Keenan march, reused verbatim from the validated 2-D
`o2i_indoor_loss`) is the only spatially-varying term → it alone sets ρ. The nine explicit
path-loss parameters are exposed via `PathLossConfig` (SIM V1.5): PLE n, σ, PL(d₀), ABG,
breakpoint, d₀, EIRP=Pt+Gt, cable loss L_c, fade margin M = z_σ·σ.

usage:
  python predict_o2i_3d.py --pci 397 --band 13                    # Forte Hall anchor
  python predict_o2i_3d.py --pci 396 --band 13
  python predict_o2i_3d.py --pci 359 --band 12 --network lte      # BS6, closest donor
"""
from __future__ import annotations
import argparse
import importlib.util
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
SIM3D_V15 = REPO / "Physics Engine" / "3D Map Physics" / "SIM V1.5 3D"
RX_DIR = REPO / "Physics Engine" / "3D Map Physics" / "Object and Tranmission" / "Reciever Objects"
PL2D = REPO / "Physics Engine" / "Cross_Validation" / "pathloss_2d"
OUT = HERE / "out"
for p in (SIM3D, SIM3D_V15, RX_DIR):
    sys.path.insert(0, str(p))
import modes_3d as MD                     # noqa: E402  (base_station_geometry: bearing + elevation)
import Construct_Reciever_3D as RX        # noqa: E402
from pathloss_config import PathLossConfig  # noqa: E402

# reuse the validated 2-D O2I march + material/geometry helpers verbatim (the ρ driver)
_spec = importlib.util.spec_from_file_location("predict_o2i_2d", PL2D / "predict_o2i.py")
P2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P2)


def predict(pci=397, band=13, network="lte", carrier=None, o2i_class="low_loss",
            bearing_deg=None, cfg=None, wall_sat_db=60.0, n_vert=12):
    cfg = cfg or PathLossConfig.outdoor(tx_power_dbm=49.0, tx_gain_dbi=13.0)   # EIRP 62 default

    meta = P2.load_floorplan_meta()
    mpp = meta["meters_per_px"]; H, W = meta["grid_shape"]
    grid = np.load(P2.STEP1 / "material_grid.npy")
    mats = json.loads((P2.STEP1 / "materials.json").read_text())

    row = P2.find_bs_row(pci, band, network, carrier)
    lat, lon = float(row["lat"]), float(row["lon"])
    d_bs = float(row["distance_m"]) if row["distance_m"] else 415.0
    freq_mhz = float(row["freq_mhz"]) if row["freq_mhz"] else 751.0
    elev = float(row["building_height_m"]) if row["building_height_m"] else None
    geom = MD.base_station_geometry(lon, lat, elev_m=elev)
    elev_ang = float(geom.get("elevation_angle_deg") or 0.0)

    loss_db, loss_per_m = P2.band_loss_vectors(mats, freq_mhz)
    grid_c = P2.mk.consolidate_walls(grid, loss_db)

    # 3-D arrival: horizontal px direction toward the donor + a vertical tilt from the
    # elevation angle. The horizontal projection drives the in-plane march (φ≈0 for Forte,
    # so cos φ ≈ 1); the field is the RX plane, extruded to a volume below for the drape.
    if bearing_deg is not None:
        Linv = np.linalg.inv(np.array(meta["affine_px_to_local_m"])[:, :2])
        er = np.radians(bearing_deg)
        d = Linv @ np.array([np.sin(er), np.cos(er)]); dir_px = (float(d[0]), float(d[1]))
        arr = f"override β={bearing_deg:.0f}°"
    else:
        bs_px, bs_py = P2.lonlat_to_px(lon, lat, meta)
        dir_px = (bs_px - W / 2.0, bs_py - H / 2.0); arr = "site geometry"

    indoor = P2.o2i_indoor_loss(grid_c, loss_db, loss_per_m, dir_px, mpp, wall_sat_db=wall_sat_db)
    outdoor_leg = P2.fspl_db(d_bs, freq_mhz) + 10.0 * (cfg.ple_n - 2.0) * np.log10(max(d_bs, 1.0))
    pl = (outdoor_leg + float(MD.O2I_DB[o2i_class]) + cfg.cable_loss_db + cfg.fade_margin_db
          + indoor).astype(np.float32)
    metrics = RX.metrics_from_pl(pl, eirp_dbm=cfg.eirp_dbm, band=RX.BandSpec.from_freq_mhz(freq_mhz))
    rsrp = np.asarray(metrics["rsrp"], np.float32)                      # (H,W) px frame

    outside = P2.mk.outside_mask(grid_c)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{pci}_{freq_mhz:.0f}MHz" + (f"_b{int(bearing_deg)}" if bearing_deg is not None else "")
    # extruded volume (n_vert layers @ ceiling); φ≈0 → layers ≈ equal. For the OSM drape / 3D view.
    ceiling_m = 2.85
    np.savez_compressed(
        OUT / f"metrics3d_bs_{tag}.npz",
        rsrp=rsrp, pl=pl, outside=outside, frame="px", grid_shape=[H, W],
        n_vert=int(n_vert), ceiling_m=ceiling_m,
        meta=json.dumps(dict(site_id=row["site_id"], pci=pci, band=band, network=network,
            lon=lon, lat=lat, elev_m=elev, distance_m=d_bs, bearing_deg=geom["arrival_bearing_deg"],
            arrival_source=arr, freq_mhz=freq_mhz, elevation_angle_deg=elev_ang,
            o2i_class=o2i_class, model="explicit_directional_o2i_accurate_geom",
            eirp_dbm=cfg.eirp_dbm, ple_n=cfg.ple_n, d0_m=cfg.d0_m,
            shadow_sigma_db=cfg.shadow_sigma_db, fade_margin_db=cfg.fade_margin_db,
            cable_loss_db=cfg.cable_loss_db, dir_px=list(dir_px),
            measured_mean=float(row["rsrp_mean"]) if row["rsrp_mean"] else None)))

    _render(rsrp, grid_c, outside, mpp, pci, freq_mhz, row, OUT / f"slice3d_bs_{tag}.png")

    ind = ~outside
    print(f"BS {row['site_id']} PCI {pci} B{band} {freq_mhz:.0f} MHz  d={d_bs:.0f} m  "
          f"arrival {geom['arrival_bearing_deg']:.0f}° ({arr})  elev-angle {elev_ang:.1f}°  o2i={o2i_class}")
    print(f"  outdoor leg {outdoor_leg:.1f} dB  indoor walls: median {np.median(indoor[ind]):.1f} "
          f"(max {indoor[ind].max():.1f}) dB   EIRP {cfg.eirp_dbm:.0f} dBm  PLE {cfg.ple_n}")
    print(f"  predicted RSRP indoor: median {np.median(rsrp[ind]):.1f}  p10 "
          f"{np.percentile(rsrp[ind],10):.1f}  max {rsrp[ind].max():.1f} dBm  "
          f"(measured mean {row['rsrp_mean']} dBm)")
    print(f"  -> {OUT / f'metrics3d_bs_{tag}.npz'}")
    return OUT / f"metrics3d_bs_{tag}.npz"


def _render(rsrp, grid, outside, mpp, pci, freq, row, out_png):
    H, W = grid.shape
    shown = np.where(outside, np.nan, rsrp)
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad((0.93, 0.93, 0.93))
    fig, ax = plt.subplots(figsize=(12, 5.6))
    im = ax.imshow(shown, cmap=cmap, extent=[0, W * mpp, H * mpp, 0])
    walls = np.isin(grid, [1, 2, 3, 5, 6, 7])
    ov = np.zeros((H, W, 4)); ov[walls] = (0, 0, 0, 0.55)
    ax.imshow(ov, extent=[0, W * mpp, H * mpp, 0])
    ax.set_title(f"Predicted RSRP (explicit 3D O2I, accurate geom) — {row['site_id']} · PCI {pci} · "
                 f"{freq:.0f} MHz · arrival {row['bearing_deg']}°")
    ax.set_xlabel("meters"); ax.set_ylabel("meters")
    fig.colorbar(im, ax=ax, label="predicted RSRP (dBm)", shrink=0.85)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pci", type=int, default=397)
    ap.add_argument("--band", type=int, default=13)
    ap.add_argument("--network", default="lte")
    ap.add_argument("--carrier", default=None)
    ap.add_argument("--o2i-class", default="low_loss")
    ap.add_argument("--bearing-deg", type=float, default=None,
                    help="override arrival bearing (deg); default = site geometry")
    ap.add_argument("--eirp-dbm", type=float, default=62.0, help="EIRP = Pt + Gt")
    ap.add_argument("--ple-n", type=float, default=None, help="path-loss exponent for the outdoor leg")
    ap.add_argument("--sigma-db", type=float, default=None, help="shadow-fading std dev σ")
    ap.add_argument("--reliability", default=None, help="80|90|95 -> fade margin M = z_σ·σ")
    ap.add_argument("--wall-sat-db", type=float, default=60.0)
    args = ap.parse_args()
    kw = dict(tx_power_dbm=args.eirp_dbm - 13.0, tx_gain_dbi=13.0)
    if args.ple_n is not None: kw["ple_n"] = args.ple_n
    if args.sigma_db is not None: kw["shadow_sigma_db"] = args.sigma_db
    if args.reliability is not None: kw["reliability"] = args.reliability
    cfg = PathLossConfig.outdoor(**kw)
    predict(pci=args.pci, band=args.band, network=args.network, carrier=args.carrier,
            o2i_class=args.o2i_class, bearing_deg=args.bearing_deg, cfg=cfg,
            wall_sat_db=args.wall_sat_db)


if __name__ == "__main__":
    main()
