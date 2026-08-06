#!/usr/bin/env python3
"""
fw_bs_catalog.py — real base-station transmitters for the outdoor surrogate.

Uses the georeferenced NoMa scene (city_georef) to place each REAL live donor
from base_stations.csv on its building via lonlat_to_vox, as a directional
panel/sector source. Unlike random-street Tx, real cells sit ON buildings and
are panel-shaped → low Tx variation → easier to learn (the user's insight).

Directivity: the panel boresight is the catalog `bearing_deg` (site→coverage
area) — the real fitted azimuths live in free-text notes, so bearing is the
clean proxy for "which way the sector faces." The FDTD source gets a small rigid
BACKPLANE behind it (opposite boresight) so radiation is forward — a real panel
= radiator + back reflector. The 3GPP horizontal pattern A(φ)=min(12(φ/65)²,25)
(RadiationPattern.cs) is carried as an input channel so the net learns the beam
direction.
"""
from __future__ import annotations

import csv
import numpy as np

import _bootstrap as B
import city_georef as CG
import dataset_3d as D3
import fw_dataset as FD
import perf_v3
from bands_v3 import get, C0
from fullwave2d import FullWaveScene
from plane_extract import Plane

import sys
sys.path.insert(0, str(B.SIM3D))
import voxelize_city as VC  # noqa: E402

HPBW_DEG = 65.0          # 3GPP macro sector azimuth HPBW
AM_DB = 25.0             # front-to-back (max off-boresight attenuation)
CATALOG = B._PHYS / "Cross_Validation" / "base_station_catalog" / "base_stations.csv"

# Map catalog band → a SIM V3 real band label (cellular LTE + NR).
BAND_TO_LABEL = {
    "617": "LTE_B71_617", "619": "LTE_B71_617", "627": "LTE_B71_617",
    "731.5": "LTE_B13_751", "739": "LTE_B13_751", "751": "LTE_B13_751",
    "763": "LTE_B13_751", "884.55": "LTE_B13_751",
    "1970": "LTE_B2_1960", "1960": "LTE_B2_1960", "2145": "LTE_B2_1960",
    "2510.55": "NR_n41_2506", "2506": "NR_n41_2506", "2355": "NR_n41_2506",
}


def panel_atten_db(bearing_to_cell_deg, boresight_deg):
    """3GPP horizontal sector attenuation (dB ≥ 0) off boresight."""
    d = ((np.asarray(bearing_to_cell_deg, float) - boresight_deg + 180.0) % 360.0) - 180.0
    return np.minimum(12.0 * (d / HPBW_DEG) ** 2, AM_DB)


def _bearing_to_grid(bearing_deg):
    """Compass bearing (deg from north, CW) → unit grid vector (dx, dz).
    This render's X = -easting (runs west), Z = northing."""
    th = np.radians(bearing_deg)
    return np.array([-np.sin(th), np.cos(th)])          # (dX, dZ)


def load_stations(band_labels=None):
    """Georeferenced grid + manifest + a list of real Tx dicts (one per sector)."""
    grid, man = CG.load_georef_city()
    NX, NZ = grid.shape[0], grid.shape[2]
    want = set(band_labels) if band_labels else None
    out = []
    for r in csv.DictReader(CATALOG.open()):
        if r["status"] != "live" or not r["lat"] or not r["freq_mhz"]:
            continue
        label = BAND_TO_LABEL.get(r["freq_mhz"].rstrip("0").rstrip("."))
        label = label or BAND_TO_LABEL.get(str(int(float(r["freq_mhz"]))))
        if label is None or (want and label not in want):
            continue
        vx, vz = VC.lonlat_to_vox(man, float(r["lon"]), float(r["lat"]))
        if not (0 <= vx < NX and 0 <= vz < NZ):
            continue
        out.append(dict(site=r["site_id"], vx=float(vx), vz=float(vz),
                        band=label, f_mhz=get(label).f_mhz,
                        boresight=float(r["bearing_deg"]) if r["bearing_deg"] else 0.0,
                        pci=r["pci"], rsrp_mean=(float(r["rsrp_mean"]) if r["rsrp_mean"] else None),
                        distance_m=float(r["distance_m"]) if r["distance_m"] else None,
                        agl=float(r["antenna_agl_m"]) if r["antenna_agl_m"] else 11.0))
    return grid, man, out


def _place_source(fine, tx_fine, boresight_deg, zoom):
    """Snap the source to air and add a rigid backplane behind it (opposite
    boresight) so the FDTD beam radiates forward (panel = radiator + reflector)."""
    H, W = fine.shape
    ix, iz = int(np.clip(tx_fine[0], 2, H - 3)), int(np.clip(tx_fine[1], 2, W - 3))
    if fine[ix, iz] != 0:                                # snap out of a building
        air = np.argwhere(fine == 0)
        j = np.argmin(np.abs(air[:, 0] - ix) + np.abs(air[:, 1] - iz))
        ix, iz = int(air[j, 0]), int(air[j, 1])
    d = _bearing_to_grid(boresight_deg)
    back = np.array([ix, iz]) - np.round(d * 2).astype(int)   # 2 cells behind
    perp = np.array([-d[1], d[0]])
    for t in np.arange(-6, 7):                            # short rigid backplane
        bx, bz = np.round(back + perp * t).astype(int)
        if 0 <= bx < H and 0 <= bz < W and (abs(bx - ix) > 1 or abs(bz - iz) > 1):
            fine[bx, bz] = 3
    return (ix, iz)


def band_region_m(f_mhz):
    """Band-adaptive coverage region (metres): bigger for low bands (longer range),
    local for high bands. `max_cells` still caps the actual grid per GPU, so this
    is the *target* extent — set max_cells high on an A100 to realize it."""
    if f_mhz < 900:
        return 400.0
    if f_mhz < 2000:
        return 250.0
    if f_mhz < 3000:
        return 180.0
    return 140.0


def bs_region(grid, man, st, npw=8.0, region_m=None, max_cells=2_500_000):
    """Extract + resample the region around a station (classes at lambda/N + the
    facade source/backplane placement), WITHOUT the FDTD — for surrogate-only
    inference/coverage. `bs_field` = this + a solve."""
    from scipy import ndimage
    cs = float(man["cell_size_m"]); NX, NZ = grid.shape[0], grid.shape[2]
    plane = grid[:, 2, :]
    band = get(st["band"]); h = band.cell_size_m(npw); zoom = cs / h
    if region_m is None:                                 # band-adaptive default
        region_m = band_region_m(band.f_mhz)
    half = int(region_m / cs / 2)
    while (2 * half * zoom) ** 2 > max_cells and half > 10:
        half = int(half * 0.85)
    x0, x1 = max(0, int(st["vx"]) - half), min(NX, int(st["vx"]) + half)
    z0, z1 = max(0, int(st["vz"]) - half), min(NZ, int(st["vz"]) + half)
    fine = ndimage.zoom(plane[x0:x1, z0:z1], zoom, order=0).astype(np.int8)
    txf = [(st["vx"] - x0) * zoom, (st["vz"] - z0) * zoom]
    tx = _place_source(fine, txf, st["boresight"], zoom)
    return Plane(fine, h, (fine.shape[0] * h, fine.shape[1] * h), tx, 2.0, npw,
                 dict(scene="outdoor-bs", site=st["site"], boresight=st["boresight"]))


def bs_field(grid, man, st, npw=8.0, region_m=None, crossings=2.5, max_cells=2_500_000):
    """FDTD complex field for one real station (facade source + backplane)."""
    p = bs_region(grid, man, st, npw=npw, region_m=region_m, max_cells=max_cells)
    U = FD._run_field(p.classes, p.h_m, p.tx_idx, get(st["band"]).f_mhz, crossings)
    return p, U


# 10-channel input: the 9 dataset_3d channels + a panel-gain (direction) channel.
IN_CH_BS = len(D3.INPUT_CHANNELS) + 1


def featurize_bs(classes, tx_local, d_m, freq_feat, boresight_deg):
    x9 = FD._featurize(classes, tx_local, d_m, freq_feat)
    H, W = classes.shape
    ii, jj = np.mgrid[0:H, 0:W]
    bearing = np.degrees(np.arctan2(-(ii - tx_local[0]), (jj - tx_local[1])))  # to-cell bearing
    gain = 10.0 ** (-panel_atten_db(bearing, boresight_deg) / 20.0)            # 0..1
    return np.concatenate([x9, gain[None].astype(np.float32)], 0)


def _dedupe_by_site(stations):
    """One representative sector per (site, band) to avoid redundant FDTD fields."""
    seen, out = set(), []
    for s in stations:
        key = (s["site"], s["band"])
        if key not in seen:
            seen.add(key); out.append(s)
    return out


def generate_bs(band_labels, holdout_site="BS7_forte_hall", boxes_per_station=48,
                box=128, npw=8.0, region_m=None, max_cells=1_600_000,
                out_dir=None, seed=1):
    """Dataset from REAL stations: train on known sites, hold out one for test."""
    import json
    from pathlib import Path
    rng = np.random.default_rng(seed)
    out = Path(out_dir) if out_dir else B._PHYS.parent / "fw_data_bs"
    out.mkdir(parents=True, exist_ok=True)
    norm = D3.load_norm(json.loads(B.MANIFEST.read_text()))
    grid, man, sts = load_stations(band_labels)
    train = _dedupe_by_site([s for s in sts if s["site"] != holdout_site])
    print(f"train sites: {[s['site'] for s in train]}  (holdout {holdout_site})")
    sid = 0
    for st in train:
        band = get(st["band"]); k = 2 * np.pi / band.wavelength_m
        ff = norm.freq_feature(band.f_mhz)
        p, U = bs_field(grid, man, st, npw=npw, region_m=region_m, max_cells=max_cells)
        H, W = U.shape; h = p.h_m; txf = np.array(p.tx_idx, float)
        ref = np.percentile(np.abs(U), 99.0) or 1.0
        xs, ys = [], []
        for _ in range(boxes_per_station):
            x0 = int(rng.integers(0, max(1, H - box))); y0 = int(rng.integers(0, max(1, W - box)))
            b = min(box, H, W)
            ii, jj = np.mgrid[x0:x0 + b, y0:y0 + b]
            d_m = np.hypot(ii - txf[0], jj - txf[1]) * h
            Ut = U[x0:x0 + b, y0:y0 + b] * np.exp(1j * k * d_m)
            ys.append((np.stack([Ut.real, Ut.imag]) / ref).astype(np.float32))
            xs.append(featurize_bs(p.classes[x0:x0 + b, y0:y0 + b],
                                   (txf[0] - x0, txf[1] - y0), d_m, ff, st["boresight"]))
        np.savez_compressed(out / f"shard_{sid:03d}.npz", x=np.stack(xs), y=np.stack(ys),
                            site=st["site"], band=st["band"], h_m=h, ref=ref,
                            boresight=st["boresight"])
        print(f"  shard {sid:03d} {st['site']} {st['band']} boxes={len(xs)}")
        sid += 1
    (out / "dataset_meta.json").write_text(json.dumps(dict(
        spec="fw-bs-v1", in_channels=IN_CH_BS, holdout=holdout_site,
        bands=list(band_labels), box=box, region_m=region_m), indent=2))
    print(f"wrote {sid} shards -> {out}")
    return out


def _tiled_predict_bs(model, classes, tx_ij, h, f_mhz, boresight, box=128, stride=96):
    import json
    import torch
    from fw_unet2d import pick_device
    dev = pick_device()
    norm = D3.load_norm(json.loads(B.MANIFEST.read_text()))
    ff = norm.freq_feature(f_mhz); k = 2 * np.pi / (C0 / (f_mhz * 1e6))
    H, W = classes.shape
    win = (np.hanning(box)[:, None] * np.hanning(box)[None, :] + 1e-3).astype(np.float32)
    acc = np.zeros((2, H, W)); wsum = np.zeros((H, W))
    xs = sorted(set(list(range(0, max(1, H - box + 1), stride)) + [max(0, H - box)]))
    ys = sorted(set(list(range(0, max(1, W - box + 1), stride)) + [max(0, W - box)]))
    for x0 in xs:
        for y0 in ys:
            b = min(box, H, W)
            ii, jj = np.mgrid[x0:x0 + b, y0:y0 + b]
            d_m = np.hypot(ii - tx_ij[0], jj - tx_ij[1]) * h
            xin = featurize_bs(classes[x0:x0 + b, y0:y0 + b],
                               (tx_ij[0] - x0, tx_ij[1] - y0), d_m, ff, boresight)
            with torch.no_grad():
                pr = model(torch.from_numpy(xin[None]).to(dev)).cpu().numpy()[0]
            acc[:, x0:x0 + b, y0:y0 + b] += pr * win[:b, :b]
            wsum[x0:x0 + b, y0:y0 + b] += win[:b, :b]
    Ut = (acc[0] + 1j * acc[1]) / np.maximum(wsum, 1e-9)
    ii, jj = np.mgrid[0:H, 0:W]
    return Ut * np.exp(-1j * k * np.hypot(ii - tx_ij[0], jj - tx_ij[1]) * h)


def validate_bs(model, band_labels, holdout_site="BS7_forte_hall", region_m=None,
                max_cells=1_600_000):
    from pathlib import Path
    from scipy.stats import spearmanr
    import fw_infer
    grid, man, sts = load_stations(band_labels)
    cand = _dedupe_by_site([s for s in sts if s["site"] == holdout_site])
    if not cand:
        print(f"holdout {holdout_site} not in bands {band_labels}"); return None
    st = cand[0]
    p, Ut = bs_field(grid, man, st, npw=8, region_m=region_m, max_cells=max_cells)
    Up = _tiled_predict_bs(model, p.classes, p.tx_idx, p.h_m, st["f_mhz"], st["boresight"])
    air = (p.classes == 0); at, ap = np.abs(Ut), np.abs(Up)
    et = fw_infer._db(at, np.percentile(at[air][at[air] > 0], 99))
    ep = fw_infer._db(ap, np.percentile(ap[air][ap[air] > 0], 99))
    m = air & (et > -60)
    rmse = float(np.sqrt(np.mean((et[m] - ep[m]) ** 2))); rho = spearmanr(et[m], ep[m])[0]
    a, b = Up[m], Ut[m]
    coh = float(np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"BS g2 @ {holdout_site} {st['band']} (held-out real station): "
          f"RMSE {rmse:.2f} dB  Spearman {rho:+.3f}  coherence {coh:.3f}")
    fw_infer._plot(Path("out") / f"fw_validate_bs_{holdout_site}", et, ep, m, p,
                   get(st["band"]), rmse, rho, coh)
    return dict(rmse_db=rmse, spearman=float(rho), coherence=coh)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", nargs="+", default=["LTE_B13_751"])
    ap.add_argument("--holdout", default="BS7_forte_hall")
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--ckpt", default="fw_bs.pt")
    ap.add_argument("--out", default="fw_data_bs")
    a = ap.parse_args()
    if a.gen:
        generate_bs(a.bands, holdout_site=a.holdout, out_dir=a.out)
    if a.validate:
        from fw_unet2d import load_model
        validate_bs(load_model(a.ckpt), a.bands, holdout_site=a.holdout)


if __name__ == "__main__":
    main()


