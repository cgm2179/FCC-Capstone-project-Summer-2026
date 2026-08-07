#!/usr/bin/env python3
"""
fw_bs_catalog.py — real live base stations as directional 3-D sources (outdoor).

SIM V2 3-D port of `2D/SIM V3/fw_bs_catalog.py`: place each live donor from
`base_stations.csv` on the georeferenced city grid (`city_georef` + voxelize_city),
as a directional source (3-D conductive backplane via `antenna_patterns_3D`), and
solve its near field with the JAX FDTD. Directivity is encoded by the material grid
(the backplane is class 3), so the training input contract stays 9-ch — a
`generate_bs3d` mirrors `fw_dataset3d` for outdoor shards.

  python fw_bs_catalog.py --list
  python fw_bs_catalog.py --site BS7_forte_hall --npw 6
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

import bootstrap as B
import antenna_patterns_3D as AP3
import city_georef as CG
import fullwave3d as F3
from bands_v3 import BANDS_V3, get

sys.path.insert(0, str(B.SIM3D))
import voxelize_city as VC  # noqa: E402

CATALOG = B.STEP2.parent.parent / "Cross_Validation" / "base_station_catalog" / "base_stations.csv"


def _nearest_band(f_mhz):
    return min(BANDS_V3, key=lambda b: abs(b.f_mhz - float(f_mhz))).label


def load_stations(city_dir=None, live_only=True):
    """Return (grid, georef-manifest, [station dicts]) — each placed at a city voxel."""
    grid, man = CG.load_georef_city(city_dir)
    NY = grid.shape[1]
    stations = []
    for r in csv.DictReader(CATALOG.open()):
        if live_only and r.get("status") != "live":
            continue
        if not r.get("lat") or not r.get("lon"):
            continue
        vx, vz = VC.lonlat_to_vox(man, float(r["lon"]), float(r["lat"]))
        f_mhz = float(r.get("freq_mhz") or r.get("f_mhz") or 2506.0)
        agl = float(r.get("agl_m") or r.get("agl") or 10.0)
        stations.append(dict(
            site=r.get("site_id", "?"), vx=int(vx), vz=int(vz),
            vy=int(np.clip(agl / float(man["cell_size_m"]), 1, NY - 2)),
            f_mhz=f_mhz, band=_nearest_band(f_mhz),
            boresight=float(r.get("bearing_deg") or 0.0),
            kind=(r.get("antenna_type") or "panel").lower(),
            pci=r.get("pci"), name=r.get("name", "")))
    return grid, man, stations


def bs_field3d(grid, man, st, npw=6.0, region_m=40.0, crossings=1.5,
               max_cells=3_000_000, dtype="float32"):
    """Crop a region around the station, place its directional backplane, solve."""
    from scipy import ndimage
    cs = float(man["cell_size_m"]); band = get(st["band"])
    h = band.cell_size_m(npw); zoom = cs / h
    r = int(region_m / cs / 2)
    x0, x1 = max(0, st["vx"] - r), min(grid.shape[0], st["vx"] + r)
    z0, z1 = max(0, st["vz"] - r), min(grid.shape[2], st["vz"] + r)
    sub = grid[x0:x1, :, z0:z1]
    while np.prod(np.array(sub.shape) * zoom) > max_cells:
        zoom *= 0.9
    fine = ndimage.zoom(sub, zoom, order=0).astype(np.int8)
    tx = tuple(int(np.clip(v, 1, fine.shape[i] - 2)) for i, v in enumerate(
        ((st["vx"] - x0) * zoom, st["vy"] * zoom, (st["vz"] - z0) * zoom)))
    if AP3.is_directional(st["kind"]):
        bore = AP3.boresight_vec_from_angles(st["boresight"], 0.0)
        fine, _ = AP3.place_backplane_3d(fine, tx, bore, kind=st["kind"],
                                         offset_cells=2, radius_cells=max(4, int(1.0 / h)))
    res = F3.solve3d(fine, h, band.f_mhz, tx, crossings=crossings,
                     extract_phasor=True, dtype=dtype)
    return fine, res["phasor"], tx, h, res


def generate_bs3d(band_labels=None, holdout_site="BS7_forte_hall", boxes_per=32,
                  box=24, npw=6.0, region_m=40.0, max_cells=3_000_000, out_dir=None,
                  seed=1, city_dir=None):
    """Outdoor training shards — same 9-ch/2-ch contract + phase-reduction as
    fw_dataset3d, one shard per station (held-out site excluded)."""
    import json
    from pathlib import Path
    from fw_dataset3d import featurize3d
    import dataset_3d as D3
    rng = np.random.default_rng(seed)
    grid, man, stations = load_stations(city_dir)
    norm = D3.load_norm(json.loads(B.MANIFEST.read_text()))
    out = Path(out_dir) if out_dir else Path(__file__).resolve().parent / "fw_data_bs3d"
    out.mkdir(parents=True, exist_ok=True)
    sid = 0
    for st in stations:
        if st["site"] == holdout_site:
            continue
        if band_labels and st["band"] not in band_labels:
            continue
        fine, U, tx, h, _ = bs_field3d(grid, man, st, npw=npw, region_m=region_m,
                                       max_cells=max_cells)
        Dx, Dy, Dz = U.shape
        if min(Dx, Dy, Dz) < box:
            print(f"  [skip] {st['site']}: fine {U.shape} < box {box}"); continue
        k = 2 * np.pi / get(st["band"]).wavelength_m
        ff = norm.freq_feature(st["f_mhz"]); ref = np.percentile(np.abs(U), 99.0) or 1.0
        txf = np.array(tx, float); xs, ys = [], []
        for _ in range(boxes_per):
            x0 = int(rng.integers(0, Dx - box + 1)); y0 = int(rng.integers(0, Dy - box + 1))
            z0 = int(rng.integers(0, Dz - box + 1))
            sl = (slice(x0, x0 + box), slice(y0, y0 + box), slice(z0, z0 + box))
            ii, jj, kk = np.mgrid[x0:x0 + box, y0:y0 + box, z0:z0 + box]
            d_m = np.sqrt((ii - txf[0]) ** 2 + (jj - txf[1]) ** 2 + (kk - txf[2]) ** 2) * h
            Ut = U[sl] * np.exp(1j * k * d_m)
            ys.append((np.stack([Ut.real, Ut.imag]) / ref).astype(np.float32))
            xs.append(featurize3d(fine[sl], (txf[0] - x0, txf[1] - y0, txf[2] - z0), d_m, ff))
        np.savez_compressed(out / f"shard_{sid:03d}.npz", x=np.stack(xs), y=np.stack(ys),
                            h_m=h, ref=ref, band=st["band"], site=st["site"])
        print(f"  shard {sid:03d} {st['site']} ({st['kind']}) boxes={len(xs)}")
        sid += 1
    print(f"wrote {sid} outdoor shards -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--site", default=None)
    ap.add_argument("--npw", type=float, default=6.0)
    a = ap.parse_args()
    grid, man, stations = load_stations()
    if a.list or not a.site:
        print(f"{len(stations)} live stations on grid {grid.shape}:")
        for st in stations:
            print(f"  {st['site']:16s} vox=({st['vx']},{st['vy']},{st['vz']}) "
                  f"{st['band']:12s} {st['kind']:10s} bore={st['boresight']:.0f}")
        return
    st = next(s for s in stations if s["site"] == a.site)
    fine, U, tx, h, res = bs_field3d(grid, man, st, npw=a.npw)
    print(f"{st['site']} field {fine.shape} [{res['backend']}] {res['dt_run']:.1f}s "
          f"|U|max={np.abs(U).max():.3e}")


if __name__ == "__main__":
    main()
