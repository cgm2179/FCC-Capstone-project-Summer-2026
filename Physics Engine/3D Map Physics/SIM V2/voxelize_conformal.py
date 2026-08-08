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

# --- optional CuPy backend: the supersampled closing + Euclidean-distance-transform (the SDF and
# gap-fill) dominate the bake at super=8 and are pure scipy/CPU. On a CUDA GPU (A100) CuPy runs them
# on-device — pairing with the JAX/XLA FDTD (simulate_jax) so the whole data-gen is GPU. NumPy/scipy
# fallback otherwise (this Mac has no NVIDIA GPU), so the code path is identical, just slower.
try:
    import cupy as _cp
    import cupyx.scipy.ndimage as _cndi
    _HAS_CUPY = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _cp, _cndi, _HAS_CUPY = None, None, False
USE_GPU = _HAS_CUPY                    # auto-on when CuPy + a CUDA GPU are present


def _backend():
    """(array_module, ndimage_module) — CuPy/cupyx on a CUDA GPU, else NumPy/scipy."""
    return (_cp, _cndi) if (USE_GPU and _HAS_CUPY) else (np, ndimage)


def _asnp(a):
    """Bring an array back to host NumPy (no-op for NumPy; .get() for CuPy)."""
    return a.get() if (_cp is not None and isinstance(a, _cp.ndarray)) else np.asarray(a)


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


MIN_STRUCT_VERTS = 20        # connected solids smaller than this = sub-wavelength clutter
_CLUTTER_CACHE = {}

ADAPTIVE_SUBCELL_M = 0.005   # ~5 mm geometry-sampling floor (the finest useful wave cell, 5.5 GHz λ/10)
MAX_SUBVOX = 3.0e8           # cap on prod(dims)·super³ — the supersampled bake (rasterize + EDT, esp.
#                             CuPy's PBA distance-transform) must fit GPU/host RAM (sized for A100-40)


def auto_super(cell_m, subcell_m=ADAPTIVE_SUBCELL_M, lo=2, hi=8):
    """super=8 where the wave cell is coarse (low bands), stepped down at high bands so the
    subcell rasterization stays ~`subcell_m` (≈5 mm) — a constant geometry-detail floor that
    keeps the bake tractable at λ/10 (super=8 at 5.5 GHz would be ~10¹¹ subvoxels)."""
    return int(np.clip(round(float(cell_m) / subcell_m), lo, hi))


def _clutter_tri(V, F, min_verts=MIN_STRUCT_VERTS):
    """Two-tier split: per-triangle bool, True if the triangle's connected solid is smaller
    than `min_verts` verts (sub-wavelength clutter — chair wheels, handles, cables). Those
    are homogenized to a bulk lossy medium rather than resolved as their real material."""
    key = (len(V), len(F), int(min_verts))
    if key not in _CLUTTER_CACHE:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]])
        _, lab = connected_components(
            coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V), len(V))),
            directed=False)
        _CLUTTER_CACHE[key] = np.bincount(lab)[lab[F[:, 0]]] < min_verts
    return _CLUTTER_CACHE[key]


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


def _effective_from_frac(frac, keys, f_mhz):
    """Volume-average per-cell material fractions → effective (εr', εr'', σ, metal_frac,
    6-class grid) at a frequency. This is the ONLY frequency-dependent part of the bake:
    `frac` and `keys` are pure geometry, so re-calling this at another band's frequency
    gives that band's medium with NO OBJ / re-voxelization (see `compose_band`)."""
    frac = np.asarray(frac, np.float32)
    K = len(keys)
    eps_k = np.array([CM.eps_sigma(k, f_mhz)[0] for k in keys], np.complex128)
    sig_k = np.array([CM.eps_sigma(k, f_mhz)[1] for k in keys], np.float64)
    is_metal_k = np.array([k == "metal" for k in keys])
    fair = frac[K]
    epsr_eff = (fair * 1.0 + (frac[:K] * eps_k[:, None, None, None].real).sum(0)).astype(np.float32)
    epsr_im = (frac[:K] * eps_k[:, None, None, None].imag).sum(0).astype(np.float32)
    sigma_eff = (frac[:K] * sig_k[:, None, None, None]).sum(0).astype(np.float32)
    metal_frac = (frac[list(np.where(is_metal_k)[0])].sum(0).astype(np.float32)
                  if is_metal_k.any() else np.zeros(frac.shape[1:], np.float32))
    # nearest 6-class (dominant non-air material's class) for the compat grid
    key_cls = np.array([{"metal": 3, "concrete": 2, "marble": 2, "glass": 5,
                         "plasterboard": 1, "ceiling_board": 1, "wood": 4,
                         "carpet": 2}.get(k, 1) for k in keys])
    dom = frac[:K].argmax(0)
    material_grid = np.where(fair > 0.5, 0, key_cls[dom]).astype(np.int8)
    return dict(epsr_eff=epsr_eff, epsr_im=epsr_im, sigma_eff=sigma_eff,
                metal_frac=metal_frac, material_grid=material_grid)


def _save_grids(out_dir, out):
    od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
    for k in ("epsr_eff", "epsr_im", "sigma_eff", "metal_frac", "sdf"):
        np.save(od / f"{k}.npy", out[k])
    np.save(od / "material_grid.npy", out["material_grid"])
    np.save(od / "normals.npy", out["normals"])
    return od


def compose_band(geom_dir, band_label, out_dir=None):
    """Derive a band's effective-medium grids from a FROZEN geometry bake — reads the
    frequency-independent `frac.npy` + `sdf`/`normals` and recomputes εr/σ at the band
    frequency. Milliseconds, needs NO OBJ. This is how the extra bands are built on Colab
    from the one committed geometry (εr flat-ish with f, σ = c·f^d rescales per band)."""
    geom_dir = Path(geom_dir)
    frac_p = geom_dir / "frac.npy"
    if not frac_p.exists():
        raise FileNotFoundError(f"{frac_p} missing — bake once with out_dir to freeze geometry")
    meta = json.loads((geom_dir / "conformal_meta.json").read_text())
    keys, cell_m = meta["keys"], meta["cell_m"]
    frac = np.load(frac_p).astype(np.float32)
    band = get(band_label)
    out = dict(_effective_from_frac(frac, keys, band.f_mhz),
               sdf=np.load(geom_dir / "sdf.npy"), normals=np.load(geom_dir / "normals.npy"),
               dims=tuple(np.load(geom_dir / "sdf.npy").shape), cell_m=cell_m,
               keys=keys, band=band.label)
    if out_dir:
        od = _save_grids(out_dir, out)
        (od / "conformal_meta.json").write_text(json.dumps(dict(
            band=band.label, f_mhz=band.f_mhz, cell_m=cell_m, super=meta.get("super"),
            dims=list(map(int, out["epsr_eff"].shape)), keys=keys,
            derived_from=geom_dir.name, channels=meta.get("channels")), indent=2))
    return out


def bake(band_label="LTE_B71_617", crop_m=None, cell_m=None, super=3, out_dir=None):
    band = get(band_label)
    man = json.loads(B.MANIFEST.read_text())
    mpu = float(man["m_per_unit"]); scene_cell = float(man["cell_size_m"])
    cell_m = float(cell_m or scene_cell)
    if super == "auto" or super is None:
        super = auto_super(cell_m)                # ~5 mm subcell floor; 8 at low bands, 2–3 at high
    super = int(super)
    V, F, TM, names = _parsed(OBJ)

    # --- material table: distinct P.2040 keys present + a homogenized 'clutter' key ---
    props = [CM.props(nm, band.f_mhz) for nm in names]
    present = sorted({p["key"] for p in props if p})
    keys = present + (["clutter"] if "clutter" not in present else [])
    kidx = {k: i for i, k in enumerate(keys)}
    name_k = np.array([kidx[p["key"]] if p else -1 for p in props], np.int32)  # name→idx (-1 excl)
    # priority: air lowest; metal/concrete high (overwrite soft materials at a shared cell)
    priority = np.array([{"metal": 6, "concrete": 5, "marble": 5, "glass": 4,
                          "plasterboard": 2, "ceiling_board": 2, "wood": 1,
                          "carpet": 1, "clutter": 1}.get(k, 3) for k in keys])
    # --- two-tier: sub-wavelength solids → homogenized clutter (bulk lossy medium) ---
    tri_small = _clutter_tri(V, F)
    Fk = name_k[TM]                                             # per-tri compact idx
    good = Fk >= 0
    F, Fk, tri_small = F[good], Fk[good], tri_small[good]
    Fk[tri_small] = kidx["clutter"]                            # clutter → σ_clutter, not real material
    print(f"  two-tier: {100*tri_small.mean():.0f}% of faces = sub-wavelength clutter "
          f"(<{MIN_STRUCT_VERTS} verts) → homogenized")

    # --- scene-frame voxel coords; optional crop (metres) ---
    lo = V.min(0)                                               # == origin_units
    Vm = (V - lo) * mpu                                         # metres, (X, Y_up, Z)
    if crop_m is not None:
        x0, x1, y0, y1, z0, z1 = crop_m
        org = np.array([x0, y0, z0]); ext = np.array([x1 - x0, y1 - y0, z1 - z0])
    else:
        org = np.zeros(3); ext = Vm.max(0)
    dims = np.maximum(np.ceil(ext / cell_m).astype(int), 1)
    # memory cap: rasterize + distance-transform allocate ~ prod(dims)·super³ int32 (CuPy's PBA EDT
    # especially) — reduce super, then coarsen the cell, so a large crop degrades gracefully instead of
    # OOMing. bake returns the actual cell_m/super it used; the caller solves at that resolution.
    while super > 2 and int(np.prod(dims)) * super ** 3 > MAX_SUBVOX:
        super -= 1
    while int(np.prod(dims)) * super ** 3 > MAX_SUBVOX:
        cell_m *= 1.25; dims = np.maximum(np.ceil(ext / cell_m).astype(int), 1)
    sdims = tuple((dims * super).tolist())
    Vs = (Vm - org) / (cell_m / super)                         # SUBCELL coords

    # --- rasterize (CPU triangle scatter) then closing + EDT on the backend (GPU on A100) ---
    xp, xndi = _backend()
    Msub = xp.asarray(_rasterize(Vs, F, Fk, sdims, priority))   # to device when CuPy is active
    occ = Msub >= 0
    # brute_force=True: cupyx only implements the brute-force iteration path (scipy gives the identical
    # result either way), so this is required for iterations>1 on the GPU and harmless on CPU.
    solid = xndi.binary_closing(occ, iterations=max(1, super - 1), brute_force=True)
    newly = solid & ~occ
    if bool(newly.any()):                                      # fill closed gaps w/ nearest material
        ind = xndi.distance_transform_edt(~occ, return_distances=False, return_indices=True)
        Msub[newly] = Msub[tuple(ind[:, newly])]

    # --- block-average subcells → per-material fraction at cell_m ---
    K = len(keys)
    frac = xp.zeros((K + 1,) + tuple(dims), xp.float32)          # [0..K-1]=materials, K=air
    lbl = xp.where(Msub >= 0, Msub, K)                          # air = index K
    for m in range(K + 1):
        block = (lbl == m).astype(xp.float32).reshape(
            dims[0], super, dims[1], super, dims[2], super)
        frac[m] = block.mean(axis=(1, 3, 5))
    frac = _asnp(frac)                                          # host for the effective medium

    # --- effective medium (the ONLY frequency-dependent step; see _effective_from_frac) ---
    eff = _effective_from_frac(frac, keys, band.f_mhz)
    epsr_eff, epsr_im = eff["epsr_eff"], eff["epsr_im"]
    sigma_eff, metal_frac, material_grid = eff["sigma_eff"], eff["metal_frac"], eff["material_grid"]

    # --- signed distance field (sub-voxel via the super-grid) + surface normals ---
    # SDF preserves exact wall boundaries below cell size; normals give the CNN the surface
    # orientation to correct staircasing diffraction. Both are surrogate-input channels only
    # (NOT fed to the FDTD — the physics already gets subcell-averaged εr/σ).
    d_out = xndi.distance_transform_edt(~solid)                # on the backend (GPU on A100)
    d_in = xndi.distance_transform_edt(solid)
    sdf_sub = (d_out - d_in) * (cell_m / super)                # signed metres (+ outside solid)
    sdf = _asnp(sdf_sub.reshape(dims[0], super, dims[1], super, dims[2], super)
                .mean((1, 3, 5))).astype(np.float32)
    gx, gy, gz = np.gradient(sdf.astype(np.float64), cell_m)
    gn = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-9
    normals = np.stack([gx / gn, gy / gn, gz / gn]).astype(np.float32)   # (3,X,Y,Z), air-ward

    out = dict(epsr_eff=epsr_eff, epsr_im=epsr_im, sigma_eff=sigma_eff, metal_frac=metal_frac,
               material_grid=material_grid, sdf=sdf, normals=normals, frac=frac, dims=tuple(dims),
               cell_m=cell_m, super=super, keys=keys, band=band.label)
    if out_dir:
        od = _save_grids(out_dir, out)
        # frozen geometry: frac (freq-independent) lets compose_band derive any other band w/o the OBJ
        np.save(od / "frac.npy", frac.astype(np.float16))
        (od / "conformal_meta.json").write_text(json.dumps(dict(
            band=band.label, f_mhz=band.f_mhz, cell_m=cell_m, super=super,
            dims=list(map(int, dims)), keys=keys, crop_m=crop_m,
            min_struct_verts=MIN_STRUCT_VERTS,
            channels="epsr_eff,sigma_eff,metal_frac,sdf,normals(3)"), indent=2))
    if USE_GPU and _cp is not None:
        _cp.get_default_memory_pool().free_all_blocks()        # release sub-grid GPU mem before the JAX solve
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--cell-m", type=float, default=None)
    ap.add_argument("--super", default=3, type=lambda s: s if s == "auto" else int(s),
                    help='geometry supersampling per cell, or "auto" (~5 mm floor)')
    ap.add_argument("--crop", type=float, nargs=6, default=None,
                    metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import time
    t0 = time.time()
    o = bake(a.band, crop_m=a.crop, cell_m=a.cell_m, super=a.super, out_dir=a.out)
    e = o["epsr_eff"]; fair_solid = 100 * (e > 1.05).mean()
    print(f"conformal bake {o['dims']} @ {o['cell_m']*100:.1f}cm  super={o['super']}  "
          f"materials={o['keys']}  {time.time()-t0:.1f}s")
    print(f"  εr_eff [{e.min():.2f}, {e.max():.2f}]  σ_eff [{o['sigma_eff'].min():.3g}, "
          f"{o['sigma_eff'].max():.3g}]  metal_frac max={o['metal_frac'].max():.2f}  "
          f"non-air cells {fair_solid:.1f}%")


if __name__ == "__main__":
    main()
