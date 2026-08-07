#!/usr/bin/env python3
"""
voxelize_conformal.py — conformal effective-medium voxelizer for the CAD OBJ.

Instead of the crude 6-class surface grid, each cell gets **subcell material
volume fractions** (from a supersampled rasterization of the real polygons), then
a volume-averaged **effective permittivity εr_eff and conductivity σ_eff** for a
band. This de-staircases the physics (standard conformal/subcell FDTD) and lets
every object — walls, ceiling, floor, furniture, glass, metal — be represented by
its true material. Materials resolve through `cad_materials` → ITU P.2040 εr/σ.

Indoor note: we do NOT exterior-flood-fill (that would solidify whole rooms). v1 =
supersampled surface rasterization + a light binary_closing (fills thin
walls/furniture shells up to the closing radius), then block-average → fractions.

Outputs (scene registration, same shape as material_grid.npy):
  epsr_eff.npy, sigma_eff.npy (float32) · metal_frac.npy (float32) ·
  material_grid.npy int8 (nearest 6-class, keeps masks/eikonal/one-hot working)

  python voxelize_conformal.py --band LTE_B71_617 --super 3            # whole floor
  python voxelize_conformal.py --band LTE_B71_617 --crop 14 40 0 6 4 30 --super 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import bootstrap as B
import cad_materials as CM
from bands_v3 import get

HERE = Path(__file__).resolve().parent
OBJ = "/Users/cameronmickle/Downloads/3ff8432c-980b-41b6-9045-c09dbe1d74cc copy.obj"
MTL = "/Users/cameronmickle/Documents/Indoor_Walk_Test_7-7/Sandbox_Version_3D_Simulation_v1.obj/3ff8432c-980b-41b6-9045-c09dbe1d74cc.mtl"


def parse_obj_mat(path):
    """verts (V,3), tris (T,3) int, tri material NAME index (T,), material names list."""
    verts, tris, tmat, names, n2i, cur = [], [], [], [], {}, -1
    for ln in open(path):
        if ln[:2] == "v ":
            verts.append(ln[2:].split()[:3])
        elif ln[:7] == "usemtl ":
            nm = ln[7:].strip()
            if nm not in n2i:
                n2i[nm] = len(names); names.append(nm)
            cur = n2i[nm]
        elif ln[:2] == "f ":
            idx = [int(x.split("/")[0]) - 1 for x in ln.split()[1:]]
            for i in range(1, len(idx) - 1):
                tris.append((idx[0], idx[i], idx[i + 1])); tmat.append(cur)
    return (np.asarray(verts, np.float32), np.asarray(tris, np.int64),
            np.asarray(tmat, np.int32), names)


_PARSE_CACHE = {}


def _parsed(path):
    """Cached OBJ parse — repeated bakes (per-Tx data-gen) reuse the 344 MB read."""
    if path not in _PARSE_CACHE:
        _PARSE_CACHE[path] = parse_obj_mat(path)
    return _PARSE_CACHE[path]


def _rasterize(Vs, F, Fk, dims, priority):
    """Surface-rasterize triangles into a subcell material-index grid (-1 air).
    Vs = verts in SUBCELL coords; Fk = per-tri compact-material index; higher
    `priority` overwrites. Adaptive barycentric sampling, bucketed + vectorized."""
    M = np.full(dims, -1, np.int32)
    tv = Vs[F]                                                  # (T,3,3)
    e = np.maximum.reduce([np.linalg.norm(tv[:, 0] - tv[:, 1], axis=1),
                           np.linalg.norm(tv[:, 1] - tv[:, 2], axis=1),
                           np.linalg.norm(tv[:, 2] - tv[:, 0], axis=1)])
    ns = np.clip(np.ceil(e).astype(int) + 1, 2, 96)
    dims_a = np.asarray(dims)
    for n in np.unique(ns):
        sel = np.where(ns == n)[0]
        sel = sel[np.argsort(priority[Fk[sel]], kind="stable")]  # high priority written last
        u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        keep = (u + v) <= 1.0
        u, v = u[keep], v[keep]; w = 1.0 - u - v                # (P,)
        p = (tv[sel, 0][:, None] * w[None, :, None] +
             tv[sel, 1][:, None] * u[None, :, None] +
             tv[sel, 2][:, None] * v[None, :, None])            # (S,P,3)
        idx = np.floor(p).astype(np.int64).reshape(-1, 3)
        mats = np.repeat(Fk[sel], len(w))
        ok = ((idx >= 0) & (idx < dims_a)).all(1)
        idx, mats = idx[ok], mats[ok]
        M[idx[:, 0], idx[:, 1], idx[:, 2]] = mats
    return M


def bake(band_label="LTE_B71_617", crop_m=None, cell_m=None, super=3, out_dir=None):
    band = get(band_label)
    man = json.loads(B.MANIFEST.read_text())
    mpu = float(man["m_per_unit"]); scene_cell = float(man["cell_size_m"])
    cell_m = float(cell_m or scene_cell)
    V, F, TM, names = _parsed(OBJ)

    # --- material table: distinct P.2040 keys present + εr/σ at band ---
    props = [CM.props(nm, band.f_mhz) for nm in names]
    keys = sorted({p["key"] for p in props if p})                # compact key list
    kidx = {k: i for i, k in enumerate(keys)}
    name_k = np.array([kidx[p["key"]] if p else -1 for p in props], np.int32)  # name→compact idx (-1 excl)
    eps_k = np.array([CM.eps_sigma(k, band.f_mhz)[0] for k in keys], np.complex128)
    sig_k = np.array([CM.eps_sigma(k, band.f_mhz)[1] for k in keys], np.float64)
    is_metal_k = np.array([k == "metal" for k in keys])
    # priority: air lowest; metal/concrete high (overwrite soft materials at a shared cell)
    priority = np.array([{"metal": 6, "concrete": 5, "marble": 5, "glass": 4,
                          "plasterboard": 2, "ceiling_board": 2, "wood": 1,
                          "carpet": 1}.get(k, 3) for k in keys])
    Fk = name_k[TM]                                              # per-tri compact idx
    good = Fk >= 0
    F, Fk = F[good], Fk[good]

    # --- scene-frame voxel coords; optional crop (metres) ---
    lo = V.min(0)                                               # == origin_units
    Vm = (V - lo) * mpu                                         # metres, (X, Y_up, Z)
    if crop_m is not None:
        x0, x1, y0, y1, z0, z1 = crop_m
        org = np.array([x0, y0, z0]); ext = np.array([x1 - x0, y1 - y0, z1 - z0])
    else:
        org = np.zeros(3); ext = Vm.max(0)
    dims = np.maximum(np.ceil(ext / cell_m).astype(int), 1)
    sdims = tuple((dims * super).tolist())
    Vs = (Vm - org) / (cell_m / super)                         # SUBCELL coords

    # --- rasterize surfaces + light closing (fills thin solids, not rooms) ---
    Msub = _rasterize(Vs, F, Fk, sdims, priority)
    occ = Msub >= 0
    solid = ndimage.binary_closing(occ, iterations=max(1, super - 1))
    newly = solid & ~occ
    if newly.any():                                            # fill closed gaps w/ nearest material
        ind = ndimage.distance_transform_edt(~occ, return_distances=False, return_indices=True)
        Msub[newly] = Msub[tuple(ind[:, newly])]

    # --- block-average subcells → per-material fraction at cell_m ---
    K = len(keys)
    frac = np.zeros((K + 1,) + tuple(dims), np.float32)          # [0..K-1]=materials, K=air
    lbl = np.where(Msub >= 0, Msub, K)                          # air = index K
    for m in range(K + 1):
        block = (lbl == m).astype(np.float32).reshape(
            dims[0], super, dims[1], super, dims[2], super)
        frac[m] = block.mean(axis=(1, 3, 5))

    # --- effective medium ---
    fair = frac[K]
    epsr_eff = (fair * 1.0 + (frac[:K] * eps_k[:, None, None, None].real).sum(0)
                ).astype(np.float32)
    epsr_im = (frac[:K] * eps_k[:, None, None, None].imag).sum(0).astype(np.float32)
    sigma_eff = (frac[:K] * sig_k[:, None, None, None]).sum(0).astype(np.float32)
    metal_frac = frac[list(np.where(is_metal_k)[0])].sum(0).astype(np.float32) if is_metal_k.any() \
        else np.zeros(tuple(dims), np.float32)
    # nearest 6-class (dominant non-air material's class) for the compat grid
    key_cls = np.array([{"metal": 3, "concrete": 2, "marble": 2, "glass": 5,
                         "plasterboard": 1, "ceiling_board": 1, "wood": 4,
                         "carpet": 2}.get(k, 1) for k in keys])
    dom = frac[:K].argmax(0)
    material_grid = np.where(fair > 0.5, 0, key_cls[dom]).astype(np.int8)

    out = dict(epsr_eff=epsr_eff, epsr_im=epsr_im, sigma_eff=sigma_eff,
               metal_frac=metal_frac, material_grid=material_grid, dims=tuple(dims),
               cell_m=cell_m, keys=keys, band=band.label)
    if out_dir:
        od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
        for k in ("epsr_eff", "epsr_im", "sigma_eff", "metal_frac"):
            np.save(od / f"{k}.npy", out[k])
        np.save(od / "material_grid.npy", material_grid)
        (od / "conformal_meta.json").write_text(json.dumps(dict(
            band=band.label, f_mhz=band.f_mhz, cell_m=cell_m, super=super,
            dims=list(map(int, dims)), keys=keys, crop_m=crop_m), indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--cell-m", type=float, default=None)
    ap.add_argument("--super", type=int, default=3)
    ap.add_argument("--crop", type=float, nargs=6, default=None,
                    metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import time
    t0 = time.time()
    o = bake(a.band, crop_m=a.crop, cell_m=a.cell_m, super=a.super, out_dir=a.out)
    e = o["epsr_eff"]; fair_solid = 100 * (e > 1.05).mean()
    print(f"conformal bake {o['dims']} @ {o['cell_m']*100:.1f}cm  super={a.super}  "
          f"materials={o['keys']}  {time.time()-t0:.1f}s")
    print(f"  εr_eff [{e.min():.2f}, {e.max():.2f}]  σ_eff [{o['sigma_eff'].min():.3g}, "
          f"{o['sigma_eff'].max():.3g}]  metal_frac max={o['metal_frac'].max():.2f}  "
          f"non-air cells {fair_solid:.1f}%")


if __name__ == "__main__":
    main()
