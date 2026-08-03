#!/usr/bin/env python3
"""PR #26 — 2D outdoor-to-indoor (O2I) path-loss + predicted RSRP on the floor grid,
driven by a base station from the catalog (Cross_Validation/base_station_catalog).

The donor is ~415 m away, so it arrives as an effectively plane wave from a fixed
bearing. We reuse the STEP_2 Motley-Keenan wall model but march PARALLEL rays along
that arrival bearing (from each cell out through the façade) instead of radial rays
from an interior Tx:

    PL(x,y)   = FSPL(d_bs, f) + indoor_walls(x,y)          # walls include the façade
    RSRP(x,y) = eirp_re + G_r − PL(x,y)                     # per-RE; see below

The spatially-varying physics is `indoor_walls(x,y)` (multi-wall + saturating obstruction
along the arrival direction). The additive **level** (EIRP + FSPL + O2I + per-RE offset)
is a single constant that the calibration step fits to the measured level — so a rough
EIRP default is fine here; validate.py reports both raw and level-calibrated error.

Geometry uses the same georeference as the catalog (Data/floorplan_meta.js): the donor
lon/lat → floor-plan pixels via the inverse affine gives the (far) source pixel, and the
arrival direction is `bs_px − grid_centre` (parallel across the small floor).

Outputs (Cross_Validation/pathloss_2d/out/):
  pl_bs_<pci>_<freq>MHz.npy        PL grid (float32 dB)
  metrics_bs_<pci>_<freq>MHz.npz   rsrp + pl grids + outside mask + meta
  map_bs_<pci>_<freq>MHz.png       quick-look RSRP heatmap

usage:
  python predict_o2i.py --pci 396 --band 13 --network lte      # Forte Hall anchor
  python predict_o2i.py --pci 396 --band 13 --eirp-dbm 62 --o2i-db 0
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STEP1 = REPO / "Physics Engine" / "2D" / "STEP_1"
STEP2 = REPO / "Physics Engine" / "2D" / "STEP_2"
CATALOG = REPO / "Physics Engine" / "Cross_Validation" / "base_station_catalog" / "base_stations.csv"
FLOORPLAN_META_JS = REPO / "Data" / "floorplan_meta.js"
OUT = HERE / "out"

# reuse consolidate_walls from the STEP_2 model
_spec = importlib.util.spec_from_file_location("mk", STEP2 / "motley_keenan.py")
mk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mk)

# frequency-dependent material-loss slope about 3.5 GHz (mirrors multiband_metrics.py)
LOSS_SLOPE_PER_GHZ = {1: 0.3, 2: 2.0, 3: 2.5, 5: 0.3, 6: 0.3, 7: 0.0}
BULK_SLOPE_PER_GHZ = {4: 0.05}
NOISE_FIGURE_DB = 7.0


def load_floorplan_meta() -> dict:
    txt = FLOORPLAN_META_JS.read_text(encoding="utf-8")
    return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])


def lonlat_to_px(lon, lat, meta):
    """Inverse of affine_px_to_lonlat (carries the −7.33° rotation) — same as georef.js."""
    A = meta["affine_px_to_lonlat"]
    a, b, c = A[0]; d, e, f = A[1]
    det = a * e - b * d
    X, Y = lon - c, lat - f
    return (e * X - b * Y) / det, (-d * X + a * Y) / det


def band_loss_vectors(mats, f_mhz):
    loss_db = np.zeros(256, np.float32)
    loss_per_m = np.zeros(256, np.float32)
    df = f_mhz / 1000.0 - 3.5
    for k, v in mats.items():
        i = int(k)
        loss_db[i] = max(v["loss_db"] + LOSS_SLOPE_PER_GHZ.get(i, 0) * df, 0.3 * v["loss_db"])
        ref = v.get("loss_per_m_db", 0.0)
        loss_per_m[i] = max(ref + BULK_SLOPE_PER_GHZ.get(i, 0) * df, 0.3 * ref)
    return loss_db, loss_per_m


def n_rb_for(bw_mhz, scs_khz):
    return int(bw_mhz * 1000 * 0.9 / (12 * scs_khz))


def fspl_db(d_m, f_mhz):
    return 20.0 * np.log10(max(d_m, 1e-6)) + 20.0 * np.log10(f_mhz) - 27.55


def o2i_indoor_loss(grid, loss_db, loss_per_m, dir_px, mpp,
                    wall_sat_db=60.0, step_px=0.6, chunk_rows=8):
    """Indoor obstruction loss (dB) from a plane wave arriving along `dir_px`.

    For each cell, march a parallel ray in +dir (toward the donor) until it leaves the
    grid, counting contiguous wall-material runs once (Motley-Keenan) plus per-metre bulk
    clutter, then saturate the total at wall_sat_db (corridor-waveguiding proxy)."""
    H, W = grid.shape
    dx, dy = np.asarray(dir_px, float)
    nrm = np.hypot(dx, dy) or 1.0
    dx, dy = dx / nrm, dy / nrm
    K = int(np.ceil(np.hypot(W, H) / step_px)) + 1
    s = (np.arange(K, dtype=np.float32) * step_px)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    wall_ids = [m for m in np.unique(grid) if loss_db[m] > 0]
    bulk_ids = [m for m in np.unique(grid) if loss_per_m[m] > 0]
    out = np.zeros((H, W), np.float32)
    spacing_m = step_px * mpp
    for r0 in range(0, H, chunk_rows):
        r1 = min(r0 + chunk_rows, H)
        sx = gx[r0:r1].ravel()[:, None] + s[None, :] * dx
        sy = gy[r0:r1].ravel()[:, None] + s[None, :] * dy
        valid = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
        xi = np.clip(sx.round(), 0, W - 1).astype(np.int32)
        yi = np.clip(sy.round(), 0, H - 1).astype(np.int32)
        mats = np.where(valid, grid[yi, xi], 255)        # 255 = off-grid sentinel
        extra = np.zeros(mats.shape[0], np.float32)
        for m in wall_ids:
            hit = mats == m
            runs = hit[:, 0].astype(np.int32) + (hit[:, 1:] & ~hit[:, :-1]).sum(axis=1, dtype=np.int32)
            extra += loss_db[m] * runs
        for m in bulk_ids:
            extra += loss_per_m[m] * (mats == m).sum(axis=1) * spacing_m
        extra = wall_sat_db * -np.expm1(-extra / wall_sat_db)
        out[r0:r1] = extra.reshape(r1 - r0, W)
    return out


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
    # prefer a row with a real measured level + geometry
    rows.sort(key=lambda r: (r["carrier_match"] != "exact", -int(r["n_points"] or 0)))
    return rows[0]


def predict(pci=396, band=13, network="lte", carrier=None,
            eirp_dbm=62.0, gr_db=0.0, o2i_db=0.0, wall_sat_db=60.0,
            bw_mhz=10.0, scs_khz=15.0):
    meta = load_floorplan_meta()
    mpp = meta["meters_per_px"]
    H, W = meta["grid_shape"]
    grid = np.load(STEP1 / "material_grid.npy")
    if grid.shape != (H, W):
        raise SystemExit(f"grid {grid.shape} != meta grid_shape {(H, W)}")
    mats = json.loads((STEP1 / "materials.json").read_text())

    row = find_bs_row(pci, band, network, carrier)
    lat, lon = float(row["lat"]), float(row["lon"])
    d_bs = float(row["distance_m"]) if row["distance_m"] else 415.0
    freq_mhz = float(row["freq_mhz"]) if row["freq_mhz"] else 751.0
    meas_mean = float(row["rsrp_mean"]) if row["rsrp_mean"] else None

    loss_db, loss_per_m = band_loss_vectors(mats, freq_mhz)
    grid_c = mk.consolidate_walls(grid, loss_db)

    # arrival direction in pixel space: donor pixel (far) minus grid centre
    bs_px, bs_py = lonlat_to_px(lon, lat, meta)
    dir_px = (bs_px - W / 2.0, bs_py - H / 2.0)

    indoor = o2i_indoor_loss(grid_c, loss_db, loss_per_m, dir_px, mpp, wall_sat_db=wall_sat_db)
    pl = (fspl_db(d_bs, freq_mhz) + o2i_db + indoor).astype(np.float32)

    n_rb = n_rb_for(bw_mhz, scs_khz)
    eirp_re = eirp_dbm - 10.0 * np.log10(12 * n_rb)
    rsrp = (eirp_re + gr_db - pl).astype(np.float32)

    outside = mk.outside_mask(grid_c)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{pci}_{freq_mhz:.0f}MHz"
    np.save(OUT / f"pl_bs_{tag}.npy", pl)
    np.savez_compressed(OUT / f"metrics_bs_{tag}.npz", rsrp=rsrp, pl=pl, outside=outside,
                        meta=json.dumps(dict(pci=pci, band=band, network=network,
                            site_id=row["site_id"], lon=lon, lat=lat, distance_m=d_bs,
                            bearing_deg=row["bearing_deg"], freq_mhz=freq_mhz,
                            eirp_dbm=eirp_dbm, o2i_db=o2i_db, n_rb=n_rb,
                            dir_px=list(dir_px), measured_mean=meas_mean)))

    _render(rsrp, grid_c, outside, mpp, pci, freq_mhz, row, meas_mean, OUT / f"map_bs_{tag}.png")

    ind = ~outside
    print(f"BS {row['site_id']} PCI {pci} B{band} {freq_mhz:.0f} MHz  "
          f"d={d_bs:.0f} m  arrival {row['bearing_deg']}°  dir_px=({dir_px[0]:.0f},{dir_px[1]:.0f})")
    print(f"  FSPL={fspl_db(d_bs, freq_mhz):.1f} dB  eirp_re={eirp_re:.1f} dBm  "
          f"indoor-loss median={np.median(indoor[ind]):.1f} dB (max {indoor[ind].max():.1f})")
    print(f"  predicted RSRP indoor: median {np.median(rsrp[ind]):.1f}  "
          f"p10 {np.percentile(rsrp[ind], 10):.1f}  max {rsrp[ind].max():.1f} dBm"
          + (f"   | measured mean {meas_mean:.1f} dBm" if meas_mean is not None else ""))
    print(f"  -> {OUT / f'metrics_bs_{tag}.npz'}")
    return OUT / f"metrics_bs_{tag}.npz"


def _render(rsrp, grid, outside, mpp, pci, freq, row, meas_mean, out_png):
    H, W = grid.shape
    shown = np.where(outside, np.nan, rsrp)
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad((0.93, 0.93, 0.93))
    fig, ax = plt.subplots(figsize=(12, 5.6))
    im = ax.imshow(shown, cmap=cmap, extent=[0, W * mpp, H * mpp, 0])
    walls = np.isin(grid, [1, 2, 3, 5, 6, 7])
    ov = np.zeros((H, W, 4)); ov[walls] = (0, 0, 0, 0.55)
    ax.imshow(ov, extent=[0, W * mpp, H * mpp, 0])
    ax.set_title(f"Predicted RSRP (2D O2I) — {row['site_id']} · PCI {pci} · {freq:.0f} MHz · "
                 f"arrival {row['bearing_deg']}°"
                 + (f"  (measured mean {meas_mean:.1f} dBm)" if meas_mean is not None else ""))
    ax.set_xlabel("meters"); ax.set_ylabel("meters")
    fig.colorbar(im, ax=ax, label="predicted RSRP (dBm)", shrink=0.85)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pci", type=int, default=396)
    ap.add_argument("--band", type=int, default=13)
    ap.add_argument("--network", default="lte")
    ap.add_argument("--carrier", default=None)
    ap.add_argument("--eirp-dbm", type=float, default=62.0)
    ap.add_argument("--gr-db", type=float, default=0.0)
    ap.add_argument("--o2i-db", type=float, default=0.0)
    ap.add_argument("--wall-sat-db", type=float, default=60.0)
    ap.add_argument("--bw-mhz", type=float, default=10.0)
    ap.add_argument("--scs-khz", type=float, default=15.0)
    args = ap.parse_args()
    predict(pci=args.pci, band=args.band, network=args.network, carrier=args.carrier,
            eirp_dbm=args.eirp_dbm, gr_db=args.gr_db, o2i_db=args.o2i_db,
            wall_sat_db=args.wall_sat_db, bw_mhz=args.bw_mhz, scs_khz=args.scs_khz)


if __name__ == "__main__":
    main()
