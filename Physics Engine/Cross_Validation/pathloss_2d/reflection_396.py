#!/usr/bin/env python3
"""PR #27 (Phase 3) — tie PCI 396's inferred east arrival to a real reflector building.

PR #26 found 396's LTE-B13 energy is best explained by an arrival from the EAST (~90°),
not the direct SW Forte Hall path (237°): re-predicting at 90° flipped ρ −0.27 → +0.52.
This script formalizes that as a single specular reflection off a building EAST of the FCC:

    Forte Hall (SW)  →  reflector (E of the FCC, façade facing ~WSW)  →  FCC (arrives from E)

It (1) sweeps the 2-D arrival bearing for 396 to pin β*, (2) computes the reflector ray from
the FCC centroid + the façade azimuth the specular bounce requires, (3) validates 396 at β*
vs the direct 237°, and (4) writes a lon/lat map (Forte Hall, FCC, reflected ray, 396 points)
+ a JSON the OSM Buildings 3D view consumes to mark the reflector on real buildings.

usage: python reflection_396.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REPORTS = REPO / "Physics Engine" / "Cross_Validation" / "validation" / "reports"
FORTE_HALL = (-77.011420, 38.901550)     # lon, lat
sys.path.insert(0, str(HERE))                                             # predict_o2i, infer_direction
sys.path.insert(0, str(REPO / "Physics Engine" / "Cross_Validation" / "validation"))  # validate
import predict_o2i as P2
import infer_direction as ID
import validate as V


def px_to_lonlat(px, py, meta):
    A = meta["affine_px_to_lonlat"]
    return (A[0][0] * px + A[0][1] * py + A[0][2], A[1][0] * px + A[1][1] * py + A[1][2])


def enu_to_lonlat(e, n, lon0, lat0):
    R = 6371000.0
    lat = lat0 + np.degrees(n / R)
    lon = lon0 + np.degrees(e / (R * np.cos(np.radians(lat0))))
    return lon, lat


def main():
    meta = P2.load_floorplan_meta()
    mpp = meta["meters_per_px"]; H, W = meta["grid_shape"]
    grid = np.load(P2.STEP1 / "material_grid.npy")
    mats = json.loads((P2.STEP1 / "materials.json").read_text())
    loss_db, loss_per_m = P2.band_loss_vectors(mats, 751.0)
    grid = P2.mk.consolidate_walls(grid, loss_db)
    Linv = np.linalg.inv(np.array(meta["affine_px_to_local_m"])[:, :2])

    recs = V.measured_points(V.load_records(), "lte", 13, 396)
    px = np.array([r["px"] for r in recs]); py = np.array([r["py"] for r in recs])
    meas = np.array([r["rsrp"] for r in recs])

    betas = np.arange(0, 360, 2.0)
    rho = np.array([V.spearman_rho(meas, -ID.point_loss(px, py, ID.bearing_to_dir(b, Linv),
                                                        grid, loss_db, loss_per_m, mpp)) or 0.0
                    for b in betas])
    bstar = float(betas[int(np.argmax(rho))]); rho_star = float(rho.max())
    rho_direct = float(rho[int(np.argmin(np.abs(betas - 237.0)))])

    # geometry: FCC centroid, reflector ray (bearing bstar from the FCC), façade azimuth
    c_lon, c_lat = px_to_lonlat(W / 2.0, H / 2.0, meta)
    lon0, lat0 = meta["origin_lonlat"]
    def bearing_between(lon1, lat1, lon2, lat2):
        R = 6371000.0
        e = np.radians(lon2 - lon1) * R * np.cos(np.radians(lat1)); n = np.radians(lat2 - lat1) * R
        return float(np.degrees(np.arctan2(e, n)) % 360.0)
    # reflector candidates along the arrival ray east of the FCC
    cands = []
    for d in (80, 150, 250):
        e = np.sin(np.radians(bstar)) * d; n = np.cos(np.radians(bstar)) * d
        rlon, rlat = enu_to_lonlat(e, n, c_lon, c_lat)
        r_to_fh = bearing_between(rlon, rlat, *FORTE_HALL)
        r_to_c = bearing_between(rlon, rlat, c_lon, c_lat)
        facade = ((r_to_fh + r_to_c) / 2.0) % 360.0
        cands.append(dict(distance_m=d, lon=round(rlon, 6), lat=round(rlat, 6),
                          r_to_forte_deg=round(r_to_fh, 1), r_to_fcc_deg=round(r_to_c, 1),
                          facade_azimuth_deg=round(facade, 1)))

    report = dict(pci=396, band=13, network="lte",
                  inferred_arrival_bearing_deg=bstar, rho_at_inferred=round(rho_star, 3),
                  rho_at_direct_237=round(rho_direct, 3),
                  forte_hall_lonlat=list(FORTE_HALL), fcc_centroid_lonlat=[round(c_lon, 6), round(c_lat, 6)],
                  interpretation=("396 arrives from the E; a single specular bounce off a building "
                                  "EAST of the FCC with a WSW-facing façade (see facade_azimuth_deg) "
                                  "turns Forte Hall's SW signal into the observed east arrival."),
                  reflector_candidates=cands)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "reflection_396.json").write_text(json.dumps(report, indent=2))

    _map(px, py, meas, meta, c_lon, c_lat, bstar, cands, REPORTS / "reflection_396_map.png")

    print(f"396 LTE B13: inferred arrival β* = {bstar:.0f}°  ρ={rho_star:.3f}  "
          f"(direct 237° → ρ={rho_direct:.3f})")
    print(f"FCC centroid ({c_lon:.5f},{c_lat:.5f}); reflector ray heads {bstar:.0f}° (E).")
    for c in cands:
        print(f"  reflector @ {c['distance_m']} m E: ({c['lon']},{c['lat']}) "
              f"façade faces {c['facade_azimuth_deg']}° (WSW toward Forte Hall)")
    print(f"  -> {REPORTS / 'reflection_396.json'}")


def _map(px, py, meas, meta, c_lon, c_lat, bstar, cands, out_png):
    lon = np.array([px_to_lonlat(px[i], py[i], meta)[0] for i in range(len(px))])
    lat = np.array([px_to_lonlat(px[i], py[i], meta)[1] for i in range(len(px))])
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(lon, lat, c=meas, cmap="viridis", s=26, edgecolors="#333", linewidths=0.3,
                    label="396 measured (RSRP)")
    ax.plot(*FORTE_HALL, "r^", ms=13, label="Forte Hall (donor)")
    ax.plot(c_lon, c_lat, "ks", ms=9, label="FCC floor centroid")
    ax.plot([FORTE_HALL[0], c_lon], [FORTE_HALL[1], c_lat], "r--", lw=1, alpha=0.6,
            label="direct path 237° (weak fit)")
    for c in cands:
        ax.plot(c["lon"], c["lat"], "b*", ms=13)
        ax.plot([FORTE_HALL[0], c["lon"]], [FORTE_HALL[1], c["lat"]], "b-", lw=0.8, alpha=0.5)
        ax.plot([c["lon"], c_lon], [c["lat"], c_lat], "b-", lw=0.8, alpha=0.5)
    ax.plot([], [], "b*", label="reflector candidates (E, façade WSW)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"PCI 396 — reflected path hypothesis (arrival {bstar:.0f}° from the E)\n"
                 f"Forte Hall → E/NE reflector → FCC")
    ax.legend(fontsize=8, loc="best"); fig.colorbar(sc, ax=ax, label="measured RSRP (dBm)", shrink=0.8)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
