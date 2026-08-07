#!/usr/bin/env python3
"""
fw_dataset3d.py — Phase B: 3-D FDTD boxes for the pure-JAX 3-D field U-Net.

SIM V2 port of `2D/SIM V3/fw_dataset3d.py`. The FDTD teacher runs on the JAX/XLA
hot path (`simulate_jax`). Fixes the exploration-flagged issues:
  * per-BAND output subdir (no more `sid` overwrite across bands),
  * enforce a UNIFORM cube box (skip a Tx whose fine volume is thinner than `box`,
    so `ShardDS` can concatenate all shards),
  * optional PHYSICAL directivity: a fraction of Tx get a class-3 backplane behind
    them (`antenna_patterns_3D.place_backplane_3d`) → directional teacher fields;
    directivity is captured for free by the 6 material one-hot channels (the
    backplane is class 3), so the input contract stays 9-ch.

Shard schema (spec "fw-unet3d-v1"), one shard per Tx, in out/<band>/ :
    x=(N,9,B,B,B) f32, y=(N,2,B,B,B) f32, h_m (), ref (), band ()

  python fw_dataset3d.py --band LTE_B71_617 --n-tx 4 --box 24 --npw 8 --directional-frac 0.4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import bootstrap as B
import dataset_3d as D3
import antenna_patterns_3D as AP3
from bands_v3 import get
from fullwave2d import FullWaveScene
from fw_fdtd_jax import simulate_jax

HERE = Path(__file__).resolve().parent
C0 = 299_792_458.0


# channel contracts: 9-ch (class one-hot) or 11-ch (+ continuous effective medium)
INPUT_CHANNELS_V2 = tuple(f"material_onehot_{i}" for i in range(6)) + \
    ("tx_blob", "freq_feat", "log_distance", "epsr_eff", "sigma_eff")   # 11
EPSR_NORM = 6.0        # (εr - 1) / EPSR_NORM  → ~[0,1] over air(1)…marble(7)
SIGMA_LOG = 7.0        # log10(1+σ) / SIGMA_LOG → ~[0,1] over air(0)…metal(1e7)


def featurize3d(classes, tx, d_m, ff, ncls=6, epsr=None, sigma=None):
    """Input tensor: 6 material one-hot + Tx Gaussian blob + freq feat + log-dist (9-ch),
    and — when `epsr`/`sigma` are given — two normalized continuous effective-medium
    channels (11-ch, the conformal-voxelizer contract). 9-ch stays backward-compatible."""
    D = classes.shape
    nch = 9 if epsr is None else 11
    x = np.zeros((nch,) + D, np.float32)
    for c in range(ncls):
        x[c] = (classes == c)
    ii, jj, kk = np.mgrid[0:D[0], 0:D[1], 0:D[2]]
    sig = D3.TX_SIGMA_CELLS
    x[ncls] = np.exp(-(((ii - tx[0]) ** 2 + (jj - tx[1]) ** 2 + (kk - tx[2]) ** 2)
                       / (2.0 * sig * sig)))
    x[ncls + 1] = ff
    x[ncls + 2] = np.log10(np.maximum(d_m, 1.0)) / D3.LOGDIST_DIVISOR
    if epsr is not None:
        x[ncls + 3] = np.clip((np.asarray(epsr) - 1.0) / EPSR_NORM, 0.0, 1.5)
        x[ncls + 4] = np.clip(np.log10(1.0 + np.maximum(np.asarray(sigma), 0.0)) / SIGMA_LOG, 0.0, 1.5)
    return x


def subvol_field(band, tx_m, npw, region_m, crossings, max_cells,
                 directional=False, boresight_az=None, kind="panel", dtype="float32"):
    """One 3-D JAX FDTD over an indoor sub-volume → (fine classes, U, tx, h).
    If `directional`, drop a class-3 backplane behind the source first."""
    grid = np.load(B.MATERIAL_GRID)
    cs = float(json.loads(B.MANIFEST.read_text())["cell_size_m"])
    h = band.cell_size_m(npw); zoom = cs / h
    ix, iz = int(tx_m[0] / cs), int(tx_m[1] / cs)
    r = int(region_m / cs / 2)
    x0, x1 = max(0, ix - r), min(grid.shape[0], ix + r)
    z0, z1 = max(0, iz - r), min(grid.shape[2], iz + r)
    sub = grid[x0:x1, :, z0:z1]
    while (sub.shape[0] * zoom) * (sub.shape[1] * zoom) * (sub.shape[2] * zoom) > max_cells:
        zoom *= 0.9
    fine = ndimage.zoom(sub, zoom, order=0).astype(np.int8)
    tx = tuple(int(np.clip(v, 1, fine.shape[i] - 2)) for i, v in enumerate(
        ((ix - x0) * zoom, (grid.shape[1] // 2) * zoom, (iz - z0) * zoom)))
    if directional:
        az = float(np.random.default_rng().uniform(0, 360)) if boresight_az is None else boresight_az
        bore = AP3.boresight_vec_from_angles(az, 0.0)
        fine, _ = AP3.place_backplane_3d(fine, tx, bore, kind=kind, offset_cells=2,
                                         radius_cells=max(4, int(0.8 / h)))
    sim = FullWaveScene(fine, h, band.f_mhz, tx, source="cw")
    steps = int(round(crossings * max(fine.shape) * h / C0 / sim.dt))
    res = simulate_jax(sim, steps, warmup_frac=0.6, extract_phasor=True, dtype=dtype)
    if not res["finite"]:
        raise FloatingPointError("3-D field blew up")
    return fine, res["phasor"], tx, h


def generate3d(band_label="LTE_B71_617", n_tx=2, boxes_per=20, box=24, npw=8.0,
               region_m=7.0, crossings=2.0, max_cells=2_000_000, out_dir=None,
               seed=1, directional_frac=0.0):
    rng = np.random.default_rng(seed)
    out = Path(out_dir) if out_dir else HERE / "fw_data3d" / band_label   # per-band dir
    out.mkdir(parents=True, exist_ok=True)
    norm = D3.load_norm(json.loads(B.MANIFEST.read_text()))
    band = get(band_label); k = 2 * np.pi / band.wavelength_m
    ff = norm.freq_feature(band.f_mhz)
    cs = float(json.loads(B.MANIFEST.read_text())["cell_size_m"])
    txc = np.argwhere(np.load(B.VALID_TX_MASK).any(axis=1))
    sid = 0
    for _ in range(n_tx):
        tc = txc[rng.integers(len(txc))]
        directional = rng.random() < directional_frac
        fine, U, tx, h = subvol_field(band, (tc[0] * cs, tc[1] * cs), npw, region_m,
                                      crossings, max_cells, directional=directional)
        Dx, Dy, Dz = U.shape
        if min(Dx, Dy, Dz) < box:
            print(f"  [skip] tx {tuple(tc)}: fine {U.shape} thinner than box={box} "
                  f"(raise npw or region_m)")
            continue
        txf = np.array(tx, float)
        ref = np.percentile(np.abs(U), 99.0) or 1.0
        xs, ys = [], []
        for _ in range(boxes_per):
            x0 = int(rng.integers(0, Dx - box + 1)); y0 = int(rng.integers(0, Dy - box + 1))
            z0 = int(rng.integers(0, Dz - box + 1))
            sl = (slice(x0, x0 + box), slice(y0, y0 + box), slice(z0, z0 + box))
            ii, jj, kk = np.mgrid[x0:x0 + box, y0:y0 + box, z0:z0 + box]
            d_m = np.sqrt((ii - txf[0]) ** 2 + (jj - txf[1]) ** 2 + (kk - txf[2]) ** 2) * h
            Ut = U[sl] * np.exp(1j * k * d_m)                       # phase-reduce Ũ
            ys.append((np.stack([Ut.real, Ut.imag]) / ref).astype(np.float32))
            xs.append(featurize3d(fine[sl], (txf[0] - x0, txf[1] - y0, txf[2] - z0), d_m, ff))
        np.savez_compressed(out / f"shard_{sid:03d}.npz", x=np.stack(xs), y=np.stack(ys),
                            h_m=h, ref=ref, band=band.label,
                            directional=bool(directional))
        print(f"  shard {sid:03d} boxes={len(xs)} x{xs[0].shape} "
              f"{'[directional]' if directional else '[omni]'}")
        sid += 1
    (out / "dataset_meta.json").write_text(json.dumps(dict(
        spec="fw-unet3d-v1", box=box, bands=[band_label], n_shards=sid,
        in_channels=list(D3.INPUT_CHANNELS), out_channels=["field_re", "field_im"],
        phase_reduced=True, npw=npw, directional_frac=directional_frac), indent=2))
    print(f"wrote {sid} shards -> {out}")
    return out


def subvol_field_em(band, tx_m, npw=8.0, region_m=7.0, crossings=2.0, super=3, dtype="float32",
                    baked=None, base_cell=0.3):
    """CONFORMAL effective-medium sub-volume solve around tx (metres, X/Z). Two modes:
      * `baked=(epsr,sigma,metal,class)` (whole-floor grids @ base_cell) → crop + zoom
        to λ/npw (order-1 for continuous). The Colab path — no 344 MB mesh needed.
      * else → re-mesh + bake the region from the OBJ (voxelize_conformal).
    Returns (material_grid, epsr, sigma, U, tx, h)."""
    import fullwave3d as F3
    h = band.cell_size_m(npw)
    if baked is not None:
        epsr_b, sigma_b, metal_b, cls_b = baked
        NX, NY, NZ = epsr_b.shape
        ix, iz = int(tx_m[0] / base_cell), int(tx_m[1] / base_cell)
        r = int(region_m / base_cell / 2)

        def win(g, fill):                         # FIXED-size window (pad edges w/ air) so
            out = np.full((2 * r, NY, 2 * r), fill, g.dtype)   # every crop shares one shape
            sx0, sx1 = max(0, ix - r), min(NX, ix + r)          # → JAX compiles once, not per Tx
            sz0, sz1 = max(0, iz - r), min(NZ, iz + r)
            out[sx0 - (ix - r):sx1 - (ix - r), :, sz0 - (iz - r):sz1 - (iz - r)] = g[sx0:sx1, :, sz0:sz1]
            return out
        zoom = base_cell / h
        # mode='nearest' + εr≥1 clamp: default zoom extrapolates edges → 0, which drives
        # c=C0/√εr huge → tiny dt → runaway step count. εr must stay ≥ 1 (n ≥ 1, c ≤ C0).
        epsr = np.maximum(ndimage.zoom(win(epsr_b, 1.0), zoom, order=1, mode="nearest"), 1.0)
        sigma = np.maximum(ndimage.zoom(win(sigma_b, 0.0), zoom, order=1, mode="nearest"), 0.0)
        mf = np.clip(ndimage.zoom(win(metal_b, 0.0), zoom, order=1, mode="nearest"), 0.0, 1.0)
        cg = ndimage.zoom(win(cls_b, 0).astype(np.float32), zoom, order=0, mode="nearest").astype(np.int8)
    else:
        import voxelize_conformal as VC
        man = json.loads(B.MANIFEST.read_text()); ceil_m = float(man["ceiling_height_m"])
        crop = (tx_m[0] - region_m, tx_m[0] + region_m, 0.0, ceil_m,
                tx_m[1] - region_m, tx_m[1] + region_m)
        o = VC.bake(band.label, crop_m=crop, cell_m=h, super=super)
        epsr, sigma, mf, cg = o["epsr_eff"], o["sigma_eff"], o["metal_frac"], o["material_grid"]
    tx = tuple(int(s // 2) for s in epsr.shape)
    if epsr[tx] >= 1.05:                                       # snap to nearest air
        air = np.argwhere(epsr < 1.05)
        tx = tuple(int(v) for v in air[np.abs(air - np.array(tx)).sum(1).argmin()])
    res = F3.solve3d_em(epsr, sigma, mf, h, band.f_mhz, tx, crossings=crossings,
                        extract_phasor=True, dtype=dtype)
    if not res["finite"]:
        raise FloatingPointError("EM field blew up")
    return cg, epsr, sigma, res["phasor"], tx, h


def generate3d_em(band_label="LTE_B71_617", n_tx=2, boxes_per=20, box=24, npw=8.0,
                  region_m=7.0, crossings=2.0, super=3, out_dir=None, seed=1, grids_dir=None):
    """Phase B with the CONFORMAL effective medium: 11-channel input (adds εr_eff/σ_eff)
    + continuous-medium FDTD teacher. One shard per Tx, spec fw-unet3d-v2.
    `grids_dir` = whole-floor baked grids (Colab path, no mesh); else re-mesh per Tx."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir) if out_dir else HERE / "fw_data3d_em" / band_label
    out.mkdir(parents=True, exist_ok=True)
    norm = D3.load_norm(json.loads(B.MANIFEST.read_text()))
    band = get(band_label); k = 2 * np.pi / band.wavelength_m
    ff = norm.freq_feature(band.f_mhz)
    cs = float(json.loads(B.MANIFEST.read_text())["cell_size_m"])
    baked = None
    if grids_dir:
        gd = Path(grids_dir)
        baked = (np.load(gd / "epsr_eff.npy"), np.load(gd / "sigma_eff.npy"),
                 np.load(gd / "metal_frac.npy"), np.load(gd / "material_grid.npy"))
    txc = np.argwhere(np.load(B.VALID_TX_MASK).any(axis=1))
    sid = 0
    for _ in range(n_tx):
        tc = txc[rng.integers(len(txc))]
        cg, epsr, sigma, U, tx, h = subvol_field_em(
            band, (tc[0] * cs, tc[1] * cs), npw=npw, region_m=region_m,
            crossings=crossings, super=super, baked=baked, base_cell=cs)
        Dx, Dy, Dz = U.shape
        if min(Dx, Dy, Dz) < box:
            print(f"  [skip] tx {tuple(tc)}: fine {U.shape} < box {box}"); continue
        txf = np.array(tx, float); ref = np.percentile(np.abs(U), 99.0) or 1.0
        xs, ys = [], []
        for _ in range(boxes_per):
            x0 = int(rng.integers(0, Dx - box + 1)); y0 = int(rng.integers(0, Dy - box + 1))
            z0 = int(rng.integers(0, Dz - box + 1))
            sl = (slice(x0, x0 + box), slice(y0, y0 + box), slice(z0, z0 + box))
            ii, jj, kk = np.mgrid[x0:x0 + box, y0:y0 + box, z0:z0 + box]
            d_m = np.sqrt((ii - txf[0]) ** 2 + (jj - txf[1]) ** 2 + (kk - txf[2]) ** 2) * h
            Ut = U[sl] * np.exp(1j * k * d_m)
            ys.append((np.stack([Ut.real, Ut.imag]) / ref).astype(np.float32))
            xs.append(featurize3d(cg[sl], (txf[0] - x0, txf[1] - y0, txf[2] - z0), d_m, ff,
                                  epsr=epsr[sl], sigma=sigma[sl]))
        np.savez_compressed(out / f"shard_{sid:03d}.npz", x=np.stack(xs), y=np.stack(ys),
                            h_m=h, ref=ref, band=band.label)
        print(f"  shard {sid:03d} boxes={len(xs)} x{xs[0].shape} [conformal 11-ch]")
        sid += 1
    (out / "dataset_meta.json").write_text(json.dumps(dict(
        spec="fw-unet3d-v2", box=box, bands=[band_label], n_shards=sid,
        in_channels=list(INPUT_CHANNELS_V2), out_channels=["field_re", "field_im"],
        phase_reduced=True, npw=npw, conformal=True), indent=2))
    print(f"wrote {sid} conformal shards -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--n-tx", type=int, default=2)
    ap.add_argument("--boxes-per", type=int, default=20)
    ap.add_argument("--box", type=int, default=24)
    ap.add_argument("--npw", type=float, default=8.0)
    ap.add_argument("--region-m", type=float, default=7.0)
    ap.add_argument("--max-cells", type=int, default=2_000_000)
    ap.add_argument("--directional-frac", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    generate3d(a.band, n_tx=a.n_tx, boxes_per=a.boxes_per, box=a.box, npw=a.npw,
               region_m=a.region_m, max_cells=a.max_cells, out_dir=a.out,
               directional_frac=a.directional_frac)


if __name__ == "__main__":
    main()
