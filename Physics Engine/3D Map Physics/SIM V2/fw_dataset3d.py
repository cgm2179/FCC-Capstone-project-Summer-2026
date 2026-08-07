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


# channel contracts: 9 (one-hot) → 11 (+ effective medium) → 15 (+ SDF + surface normals)
INPUT_CHANNELS_V2 = tuple(f"material_onehot_{i}" for i in range(6)) + \
    ("tx_blob", "freq_feat", "log_distance", "epsr_eff", "sigma_eff")   # 11
INPUT_CHANNELS_V3 = INPUT_CHANNELS_V2 + ("sdf", "normal_x", "normal_y", "normal_z")  # 15
INPUT_CHANNELS_V4 = INPUT_CHANNELS_V3 + ("antenna_jx", "antenna_jy", "antenna_jz")   # 18
EPSR_NORM = 6.0        # (εr - 1) / EPSR_NORM  → ~[0,1] over air(1)…marble(7)
SIGMA_LOG = 7.0        # log10(1+σ) / SIGMA_LOG → ~[0,1] over air(0)…metal(1e7)
SDF_NORM = 0.5         # SDF / SDF_NORM (m) → ~[-1,1] within ½ m of a surface


def featurize3d(classes, tx, d_m, ff, ncls=6, epsr=None, sigma=None, sdf=None, normals=None,
                antenna=None):
    """Input tensor: 6 material one-hot + Tx blob + freq + log-dist (9-ch); + normalized
    effective medium εr/σ when given (11-ch); + signed-distance field and surface normals
    when given (15-ch — the conformal contract that corrects sub-voxel walls + staircasing);
    + the antenna current map |Jx|,|Jy|,|Jz| when given (18-ch — the explicit oriented source).
    Fewer args stay backward-compatible (9 / 11 / 15 ch)."""
    D = classes.shape
    nch = 9 + (2 if epsr is not None else 0) + (4 if sdf is not None else 0) \
        + (3 if antenna is not None else 0)
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
    if sdf is not None:
        x[ncls + 5] = np.clip(np.asarray(sdf) / SDF_NORM, -1.0, 1.0)   # sub-voxel boundary
        x[ncls + 6:ncls + 9] = np.asarray(normals)                    # nx, ny, nz (air-ward)
    if antenna is not None:                                           # |Jx|,|Jy|,|Jz| oriented source
        assert epsr is not None and sdf is not None, "antenna channels require the 15-ch base (V4 ⊃ V3)"
        x[ncls + 9:ncls + 12] = np.abs(np.asarray(antenna))
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


def subvol_field_em(band, tx_m, npw=10.0, region_m=7.0, crossings=2.0, super="auto", dtype="float32",
                    baked=None, base_cell=0.3, az_deg=None, tilt_deg=0.0, kind="panel"):
    """CONFORMAL effective-medium sub-volume solve around tx (metres, X/Z). Two modes:
      * `baked=(epsr,sigma,metal,class)` (whole-floor grids @ base_cell) → crop + zoom
        to λ/npw (order-1 for continuous). The Colab path — no 344 MB mesh needed.
      * else → re-mesh + bake the region from the OBJ at λ/npw with adaptive `super`
        (full CAD detail at the solve resolution — the high-fidelity path).
    When `az_deg` is given, the antenna pattern (`kind`, boresight az/tilt) is imposed on the
    solved field → a directional teacher matching the antenna J-map input channels.
    Returns (material_grid, epsr, sigma, sdf, normals, U, tx, h)."""
    import fullwave3d as F3
    h = band.cell_size_m(npw)
    if baked is not None:
        epsr_b, sigma_b, metal_b, cls_b, sdf_b = baked[:5]
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
        sdf = ndimage.zoom(win(sdf_b, 1.0), zoom, order=1, mode="nearest").astype(np.float32)  # 1.0=outside
    else:
        import voxelize_conformal as VC
        man = json.loads(B.MANIFEST.read_text()); ceil_m = float(man["ceiling_height_m"])
        crop = (tx_m[0] - region_m, tx_m[0] + region_m, 0.0, ceil_m,
                tx_m[1] - region_m, tx_m[1] + region_m)
        o = VC.bake(band.label, crop_m=crop, cell_m=h, super=super)
        epsr, sigma, mf, cg, sdf = o["epsr_eff"], o["sigma_eff"], o["metal_frac"], o["material_grid"], o["sdf"]
    tx = tuple(int(s // 2) for s in epsr.shape)
    if epsr[tx] >= 1.05:                                       # snap to nearest air
        air = np.argwhere(epsr < 1.05)
        tx = tuple(int(v) for v in air[np.abs(air - np.array(tx)).sum(1).argmin()])
    res = F3.solve3d_em(epsr, sigma, mf, h, band.f_mhz, tx, crossings=crossings,
                        extract_phasor=True, dtype=dtype)
    if not res["finite"]:
        raise FloatingPointError("EM field blew up")
    U = res["phasor"]
    if az_deg is not None:                                      # impose the antenna directivity
        U = AP3.apply_directivity(U, tx, kind, boresight_az=float(az_deg), tilt_deg=float(tilt_deg))
    gx, gy, gz = np.gradient(sdf.astype(np.float64), h)         # surface normals from the SDF
    gn = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-9
    normals = np.stack([gx / gn, gy / gn, gz / gn]).astype(np.float32)
    return cg, epsr, sigma, sdf, normals, U, tx, h


def _stratified_origins(rng, dims, box, n):
    """`n` box origins spread across x/y/z (per-axis Latin-hypercube + independent shuffle) so the
    crops vary along all three axes rather than clustering — the '300 positions, vary each of x/y/z'."""
    his = [max(1, d - box + 1) for d in dims]
    out = np.zeros((n, 3), int)
    for a in range(3):
        edges = np.linspace(0, his[a], n + 1)
        pos = edges[:-1] + rng.random(n) * (edges[1:] - edges[:-1])
        rng.shuffle(pos)
        out[:, a] = np.clip(pos.astype(int), 0, his[a] - 1)
    return out


def generate3d_em(band_label="LTE_B71_617", n_tx=2, boxes_per=20, box=24, npw=10.0,
                  region_m=7.0, crossings=2.0, super="auto", out_dir=None, seed=1, grids_dir=None,
                  directional_frac=0.5, antenna=True):
    """Phase B with the CONFORMAL effective medium: 18-channel input (6 material one-hot + Tx blob
    + freq + log-dist + εr/σ + SDF + normals + antenna |Jx|,|Jy|,|Jz|) and a continuous-medium FDTD
    teacher with per-Tx directivity. One shard per Tx, spec fw-unet3d-v4 (v3 when antenna=False).
    `grids_dir` = whole-floor baked grids (no-OBJ/quick path, εr/σ upsampled from 0.3 m); else per-Tx
    MESH bake @ λ/npw with adaptive super (full CAD detail at the solve resolution)."""
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
                 np.load(gd / "metal_frac.npy"), np.load(gd / "material_grid.npy"),
                 np.load(gd / "sdf.npy"))
    txc = np.argwhere(np.load(B.VALID_TX_MASK).any(axis=1))
    tx_idx = rng.integers(len(txc), size=n_tx)             # deterministic Tx picks (resume-stable)
    for sid in range(n_tx):
        shard_path = out / f"shard_{sid:03d}.npz"
        if shard_path.exists():                            # PER-Tx resume: a Tx already on disk is skipped
            print(f"  [skip] shard {sid:03d} already done"); continue
        trng = np.random.default_rng([seed, sid])          # independent per-Tx stream → skip-invariant
        tc = txc[tx_idx[sid]]
        directional = antenna and (trng.random() < directional_frac)
        if directional:                                    # random panel boresight (az, downtilt)
            az = float(trng.uniform(0, 360)); tilt = float(trng.uniform(-15, 5)); kind = "panel"
            bore = AP3.boresight_vec_from_angles(az, tilt)
        else:                                              # ~isotropic source (no directivity mask)
            az = None; tilt = 0.0; kind = "iso"; bore = np.ones(3) / np.sqrt(3.0)
        cg, epsr, sigma, sdf, normals, U, tx, h = subvol_field_em(
            band, (tc[0] * cs, tc[1] * cs), npw=npw, region_m=region_m, crossings=crossings,
            super=super, baked=baked, base_cell=cs, az_deg=az, tilt_deg=tilt, kind=kind)
        Dx, Dy, Dz = U.shape
        if min(Dx, Dy, Dz) < box:
            print(f"  [skip] tx {tuple(tc)}: fine {U.shape} < box {box}"); continue
        txf = np.array(tx, float); ref = np.percentile(np.abs(U), 99.0) or 1.0
        Jfull = None
        if antenna:                                        # oriented current map over the sub-volume
            ii, jj, kk = np.mgrid[0:Dx, 0:Dy, 0:Dz]
            s2 = 2.0 * D3.TX_SIGMA_CELLS ** 2
            blob = np.exp(-(((ii - txf[0]) ** 2 + (jj - txf[1]) ** 2 + (kk - txf[2]) ** 2) / s2))
            Jfull = (np.abs(bore)[:, None, None, None] * blob[None]).astype(np.float32)  # (3,Dx,Dy,Dz)
        xs, ys = [], []
        for x0, y0, z0 in _stratified_origins(trng, (Dx, Dy, Dz), box, boxes_per):
            x0, y0, z0 = int(x0), int(y0), int(z0)
            sl = (slice(x0, x0 + box), slice(y0, y0 + box), slice(z0, z0 + box))
            ii, jj, kk = np.mgrid[x0:x0 + box, y0:y0 + box, z0:z0 + box]
            d_m = np.sqrt((ii - txf[0]) ** 2 + (jj - txf[1]) ** 2 + (kk - txf[2]) ** 2) * h
            Ut = U[sl] * np.exp(1j * k * d_m)
            ys.append((np.stack([Ut.real, Ut.imag]) / ref).astype(np.float32))
            xs.append(featurize3d(cg[sl], (txf[0] - x0, txf[1] - y0, txf[2] - z0), d_m, ff,
                                  epsr=epsr[sl], sigma=sigma[sl], sdf=sdf[sl],
                                  normals=normals[(slice(None),) + sl],
                                  antenna=(Jfull[(slice(None),) + sl] if Jfull is not None else None)))
        tmp = out / f".shard_{sid:03d}.tmp.npz"            # atomic: a timeout mid-write can't leave a half shard
        np.savez_compressed(tmp, x=np.stack(xs), y=np.stack(ys),
                            h_m=h, ref=ref, band=band.label, directional=bool(directional))
        tmp.replace(shard_path)
        print(f"  shard {sid:03d} boxes={len(xs)} x{xs[0].shape} "
              f"[{'directional' if directional else 'iso'} {xs[0].shape[0]}-ch]")
    n_done = len(list(out.glob("shard_*.npz")))            # done-marker only after every Tx is on disk
    spec = "fw-unet3d-v4" if antenna else "fw-unet3d-v3"
    chans = INPUT_CHANNELS_V4 if antenna else INPUT_CHANNELS_V3
    (out / "dataset_meta.json").write_text(json.dumps(dict(
        spec=spec, box=box, bands=[band_label], n_shards=n_done,
        in_channels=list(chans), out_channels=["field_re", "field_im"],
        phase_reduced=True, npw=npw, conformal=True, antenna=antenna,
        directional_frac=directional_frac), indent=2))
    print(f"wrote/verified {n_done} conformal shards -> {out}")
    return out


# band-aware (region_m, npw) — λ/10 everywhere; region shrinks with f so cells/solve stay ≲ ~60 M
# (the 4.58 m ceiling dominates the high bands). super is adaptive (auto_super, ~5 mm floor).
BAND_PLAN = {
    "LTE_B71_617": (6.0, 10.0), "LTE_B13_751": (6.0, 10.0),
    "LTE_B2_1960": (4.0, 10.0), "NR_n41_2506": (3.5, 10.0), "WiFi_2G4": (3.5, 10.0),
    "NR_n77_3700": (2.5, 10.0), "WiFi_5G": (2.0, 10.0), "WiFi_6E": (2.0, 10.0),
}


def generate_bands(bands, out_root, conformal_root, geom_dir=None, n_tx=15, boxes_per=32,
                   box=24, crossings=2.0, seed=1, plan=None, mesh_bake=False, super="auto",
                   antenna=True, directional_frac=0.5):
    """Resumable multiband Phase B. For each band: build the conformal medium, solve `n_tx`
    transmitters, write 18-channel shards — SKIPPING any band whose dataset_meta.json already
    exists (the done-marker), so a re-run after a Colab timeout resumes where it stopped.

    Two fidelity modes:
      * `mesh_bake=True`  — per-Tx MESH bake @ λ/10 with adaptive super (full CAD detail at the
        solve resolution). Needs the OBJ on the node. The high-fidelity A100 path.
      * `mesh_bake=False` — derive per-band εr/σ once from the frozen geometry (`geom_dir`/frac.npy,
        no OBJ) @ 0.3 m and upsample per Tx. The quick / OBJ-free path."""
    import voxelize_conformal as VC
    plan = plan or BAND_PLAN
    out_root, conformal_root = Path(out_root), Path(conformal_root)
    geom_dir = Path(geom_dir) if geom_dir else None
    have_geom = geom_dir is not None and (geom_dir / "frac.npy").exists()
    for band in bands:
        bd = out_root / band
        if (bd / "dataset_meta.json").exists():
            print(f"[skip] {band}: shards already generated"); continue
        reg, npw = plan.get(band, (4.0, 10.0))
        gd = conformal_root / band
        grids_dir = None
        if not mesh_bake:                                  # OBJ-free: compose/bake 0.3 m grids once
            grids_dir = str(gd)
            if not (gd / "sdf.npy").exists():
                if have_geom:
                    print(f"[grid] {band}: compose from frozen geometry {geom_dir.name} (no OBJ)")
                    VC.compose_band(str(geom_dir), band, out_dir=str(gd))
                else:
                    print(f"[bake] {band}: voxelize from OBJ @ 0.3 m …")
                    VC.bake(band, cell_m=0.3, super=3, out_dir=str(gd))
        else:
            print(f"[mesh] {band}: per-Tx mesh bake @ λ/{npw:.0f} + adaptive super (full detail)")
        print(f"[gen ] {band}: n_tx={n_tx} boxes={boxes_per} region={reg}m npw=λ/{npw:.0f}")
        generate3d_em(band, n_tx=n_tx, boxes_per=boxes_per, box=box, npw=npw,
                      region_m=reg, crossings=crossings, out_dir=str(bd),
                      grids_dir=grids_dir, seed=seed, super=super, antenna=antenna,
                      directional_frac=directional_frac)
        print(f"[done] {band}")
    return out_root


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
