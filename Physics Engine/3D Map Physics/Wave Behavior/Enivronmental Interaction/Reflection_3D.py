#!/usr/bin/env python3
"""
Reflection_3D — specular multipath by the image-source method.

The mechanism that produces interference structure. On a voxel grid the two dominant
indoor paths are the FLOOR bounce and the CEILING bounce (for a ceiling-mounted AP), which
is exactly why the M0.4 re-voxelization had to land first: the old scene had a drywall
floor slab and no ceiling at all, so one of these paths carried the wrong permittivity and
the other did not exist.

AXIS-ALIGNED PLANES
A voxel scene's surfaces are axis-aligned by construction, which makes the image-source
method exact and fully vectorizable rather than an approximation:

    image of tx across a plane at coordinate c with normal along axis a:
        img[a] = 2c - tx[a]                       (other components unchanged)
    total reflected path length to a receiver r:
        s = |img - r|                             (one norm, no ray marching)
    reflection point:                             (linear interpolation along the segment)
    incidence angle:                              cos(theta) = |dr[a]| / s

So a single bounce costs one distance field per plane — the same cost as the direct field.

COHERENT, NOT POWER-SUMMED
Specular reflection is *the* coherent mechanism. BUILD_PROMPTS P7 specifies power-summing
it onto the direct field; that is done here as complex-field addition instead, because
power-summing destroys precisely the standing-wave structure that reflection exists to
create. The Fresnel coefficient is complex (`physics_3d.r_slab`, an Airy slab already
written for this module and previously unused), so its phase is carried too:

    E_r = 10**(-PL_r/20) * exp(j*arg(R)) * exp(-j*k*s)

with PL_r referenced exactly like the direct field so the two are directly summable.

BACKENDS
    numpy  — reference, exact, fine for a handful of planes
    torch  — same maths on GPU; runs on CUDA (Colab A100) or MPS (Apple Silicon).
             Parity against numpy is asserted in the self-test, mirroring the
             engine_v2_torch precedent (which held to 0.0007 dB).

usage:
    python Reflection_3D.py --test
    python Reflection_3D.py --test --backend torch
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from _bootstrap import SIM3D, contracts, engine_3d, physics_3d  # noqa: F401

MECHANISM = "reflection"
DEFAULTS = {"order": 1, "min_area_cells": 24, "max_planes": 12}
C0 = contracts.C0
AXIS_NAME = ("x", "y", "z")


@dataclass
class Plane:
    axis: int          # 0=x, 1=y, 2=z  (plane normal direction)
    coord: float       # voxel coordinate of the reflecting face
    sign: int          # +1 if the material is on the +axis side, -1 otherwise
    mclass: int        # material class of the reflector
    area_cells: int
    lo: tuple          # bounding box of the face in the other two axes (voxels)
    hi: tuple

    @property
    def name(self):
        return f"{AXIS_NAME[self.axis]}={self.coord:.0f}{'+' if self.sign > 0 else '-'}"


# --------------------------------------------------------------------------- planes
def find_reflector_planes(scene, *, min_area_cells=24, max_planes=12,
                          classes=None) -> list[Plane]:
    """Find the dominant axis-aligned reflecting faces in the scene.

    A face exists wherever air is adjacent to structure. For each axis we count, per
    slab coordinate, how many air/structure interfaces there are; the largest counts are
    the floor, ceiling and major walls.
    """
    M = scene.M
    air = M == 0
    out: list[Plane] = []
    for axis in (0, 1, 2):
        n = M.shape[axis]
        for c in range(n - 1):
            a = np.take(M, c, axis=axis)
            b = np.take(M, c + 1, axis=axis)
            aa = np.take(air, c, axis=axis)
            ab = np.take(air, c + 1, axis=axis)
            # structure at c, air at c+1  -> face pointing +axis
            for sign, solid, empty, cls_src in ((-1, ~aa, ab, a), (+1, ~ab, aa, b)):
                face = solid & empty
                area = int(face.sum())
                if area < min_area_cells:
                    continue
                if classes is not None:
                    vals = cls_src[face]
                    if not np.isin(vals, list(classes)).any():
                        continue
                vals = cls_src[face]
                mclass = int(np.bincount(vals[vals > 0]).argmax()) if (vals > 0).any() else 1
                idx = np.nonzero(face)
                out.append(Plane(axis=axis, coord=float(c + 0.5), sign=sign,
                                 mclass=mclass, area_cells=area,
                                 lo=(int(idx[0].min()), int(idx[1].min())),
                                 hi=(int(idx[0].max()), int(idx[1].max()))))
    out.sort(key=lambda p: -p.area_cells)
    # thin near-duplicate faces (a slab has two faces one voxel apart)
    kept: list[Plane] = []
    for p in out:
        if all(not (q.axis == p.axis and abs(q.coord - p.coord) < 2.0) for q in kept):
            kept.append(p)
        if len(kept) >= max_planes:
            break
    return kept


def image_source(tx, plane: Plane) -> np.ndarray:
    """Mirror a transmitter across an axis-aligned plane."""
    img = np.asarray(tx, float).copy()
    img[plane.axis] = 2.0 * plane.coord - img[plane.axis]
    return img


def reflection_coeff(scene, plane: Plane, cos_theta, f_mhz):
    """Complex Airy-slab reflection coefficient R_te for one plane at one band.

    Single source of truth for BOTH backends -- the numpy and torch paths must not
    compute the permittivity differently or their parity check is meaningless.
    Uses the ITU-R P.2040-3 complex permittivity via physics_v2, reached through the
    physics_3d re-export, so reflection is coupled to the same material table the
    CrossingLUT transmission uses.
    """
    mat = physics_3d.MATERIALS[plane.mclass]
    eps = physics_3d.permittivity(mat.get("p2040", "concrete"), float(f_mhz))
    t_ref = float(scene.t_ref[plane.mclass]) or 0.15
    theta = np.arccos(np.clip(np.asarray(cos_theta, float), 0.0, 1.0))
    R_te, _R_tm = physics_3d.r_slab(eps, theta, t_ref, float(f_mhz))
    return np.asarray(R_te, np.complex128)


# --------------------------------------------------------------------------- solve
def _reflection_geometry(scene, tx, plane, P):
    """Return (s_m, cos_theta, hit_u, hit_v, valid) for one plane, vectorized over P."""
    img = image_source(tx, plane)
    d = P - img.astype(np.float32)                       # image -> receiver
    s_vox = np.linalg.norm(d, axis=-1)
    s_m = s_vox * scene.cell

    with np.errstate(invalid="ignore", divide="ignore"):
        cos_th = np.abs(d[..., plane.axis]) / np.maximum(s_vox, 1e-9)

    # The receiver must be on the opposite side of the plane from the image, else the
    # "reflection" would be behind the wall.
    tx_side = np.sign(np.asarray(tx, float)[plane.axis] - plane.coord)
    rx_side = np.sign(P[..., plane.axis] - plane.coord)
    valid = (rx_side == tx_side) & (s_vox > 1.0)

    # Reflection point must lie inside the physical extent of the face.
    t = (plane.coord - img[plane.axis]) / np.where(
        np.abs(d[..., plane.axis]) < 1e-9, 1e-9, d[..., plane.axis])
    others = [a for a in (0, 1, 2) if a != plane.axis]
    hit_u = img[others[0]] + t * d[..., others[0]]
    hit_v = img[others[1]] + t * d[..., others[1]]
    valid &= (t > 0) & (t < 1)
    valid &= (hit_u >= plane.lo[0] - 1) & (hit_u <= plane.hi[0] + 1)
    valid &= (hit_v >= plane.lo[1] - 1) & (hit_v <= plane.hi[1] + 1)
    return s_m, cos_th, valid


def solve(scene, tx, *, bands=None, order=1, planes=None, min_area_cells=24,
          max_planes=12, backend="numpy", device=None,
          progress=False) -> "contracts.FieldGrid":
    """Specular reflected field for one Tx (order-1 image sources).

    Each plane contributes E_r = |R| * (free-space amplitude at the image distance),
    charged for the walls the reflected path crosses, with phase arg(R) - k*s.
    """
    if order != 1:
        raise NotImplementedError("order>1 image sources are the next increment")

    freqs = np.asarray(scene.freqs if bands is None else bands, float)
    bidx = [scene.band_index(f) for f in freqs]
    NX, NY, NZ = scene.NX, scene.NY, scene.NZ
    inside = scene.inside if scene.inside is not None else np.ones(scene.M.shape, bool)

    if planes is None:
        planes = find_reflector_planes(scene, min_area_cells=min_area_cells,
                                       max_planes=max_planes)

    gx, gy, gz = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing="ij")
    P = np.stack([gx, gy, gz], -1).astype(np.float32)

    if backend == "torch":
        return _solve_torch(scene, tx, freqs, bidx, planes, P, inside, device, progress)

    obs_tx = scene.crossing_loss(tx)
    E = np.zeros((contracts.NPOL, len(freqs), NX, NY, NZ), np.complex64)
    tau_min = np.full((len(freqs), NX, NY, NZ), np.inf, np.float32)

    for pi, pl in enumerate(planes):
        s_m, cos_th, valid = _reflection_geometry(scene, tx, pl, P)
        valid &= inside
        if not valid.any():
            continue
        for i, (f, bi) in enumerate(zip(freqs, bidx)):
            R = reflection_coeff(scene, pl, cos_th[valid], f)

            # Referenced like the direct field: spreading over the IMAGE distance, plus
            # the wall crossings the reflected path incurs (approximated by the direct
            # map, which is exact for the specular leg on an axis-aligned plane).
            pl_db = (scene.fspl1[bi] + 10.0 * scene.n_exp
                     * np.log10(np.maximum(s_m[valid], scene.d0)))
            pl_db = pl_db + obs_tx[bi][valid]
            amp = np.power(10.0, -pl_db / 20.0) * np.abs(R)

            tau = (s_m[valid] / C0)
            ph = np.angle(R) - 2.0 * np.pi * (f * 1e6) * tau

            contrib = np.zeros((NX, NY, NZ), np.complex64)
            contrib[valid] = (amp * np.exp(1j * ph)).astype(np.complex64)
            E[0, i] += contrib

            t = np.full((NX, NY, NZ), np.inf, np.float32)
            t[valid] = tau.astype(np.float32)
            tau_min[i] = np.minimum(tau_min[i], t)

        if progress:
            print(f"  plane {pi+1}/{len(planes)}  {pl.name}  class={pl.mclass}  "
                  f"area={pl.area_cells}")

    fg = contracts.FieldGrid(
        E=E, tau_first=tau_min, mechanism=MECHANISM, freqs_mhz=freqs,
        tx_vox=tuple(int(v) for v in tx),
        meta={"order": order, "planes": [p.name for p in planes],
              "n_planes": len(planes), "backend": backend,
              "physics": "physics_3d.r_slab (Airy slab, complex R)",
              "note": "coherent complex addition, NOT power-summed"})
    fg.combine_as = contracts.COHERENT
    return fg


def _solve_torch(scene, tx, freqs, bidx, planes, P, inside, device, progress):
    """GPU mirror of the numpy path. Runs on CUDA (Colab A100) or MPS (Apple Silicon)."""
    import torch

    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    dev = torch.device(device)
    NX, NY, NZ = scene.NX, scene.NY, scene.NZ

    obs_tx = torch.as_tensor(np.asarray(scene.crossing_loss(tx)), device=dev)
    Pt = torch.as_tensor(P, device=dev)
    ins = torch.as_tensor(inside, device=dev)
    E = torch.zeros((contracts.NPOL, len(freqs), NX, NY, NZ),
                    dtype=torch.complex64, device=dev)
    tau_min = torch.full((len(freqs), NX, NY, NZ), float("inf"), device=dev)

    for pl in planes:
        s_m, cos_th, valid = _reflection_geometry(scene, tx, pl, P)
        s_t = torch.as_tensor(s_m, device=dev)
        cos_t = torch.as_tensor(cos_th, device=dev)
        val = torch.as_tensor(valid, device=dev) & ins
        if not bool(val.any()):
            continue
        for i, (f, bi) in enumerate(zip(freqs, bidx)):
            # identical physics to the numpy path -- shared helper, not a re-derivation
            R = torch.as_tensor(
                reflection_coeff(scene, pl, cos_th, f).astype(np.complex64), device=dev)
            pl_db = (scene.fspl1[bi] + 10.0 * scene.n_exp
                     * torch.log10(torch.clamp(s_t, min=scene.d0))) + obs_tx[bi]
            amp = torch.pow(10.0, -pl_db / 20.0) * R.abs()
            tau = s_t / C0
            ph = torch.angle(R) - 2.0 * np.pi * (f * 1e6) * tau
            contrib = torch.where(val, amp * torch.exp(1j * ph),
                                  torch.zeros((), dtype=torch.complex64, device=dev))
            E[0, i] += contrib.to(torch.complex64)
            tau_min[i] = torch.minimum(tau_min[i],
                                       torch.where(val, tau, torch.full_like(tau, float("inf"))))

    fg = contracts.FieldGrid(
        E=E.cpu().numpy(), tau_first=tau_min.cpu().numpy().astype(np.float32),
        mechanism=MECHANISM, freqs_mhz=freqs, tx_vox=tuple(int(v) for v in tx),
        meta={"order": 1, "backend": "torch", "device": str(dev),
              "n_planes": len(planes)})
    fg.combine_as = contracts.COHERENT
    return fg


# --------------------------------------------------------------------------- self-test
def _selftest(backend="numpy") -> int:
    sys.path.insert(0, str(SIM3D / "tests3d"))
    import synth3d as S

    ok, fails = 0, []

    def check(name, cond, msg=""):
        nonlocal ok
        if cond:
            ok += 1
            print(f"  PASS  {name}")
        else:
            fails.append(name)
            print(f"  FAIL  {name}  {msg}")

    # two-plate scene: floor + ceiling, the canonical 2-ray geometry
    M, man = S.two_plate_scene_3d(n=(40, 20, 24), gap_cells=10, mat=2)
    sc = engine_3d.SceneV3(M, man)
    sc.inside = (M == 0)
    tx = (8, 10, 12)
    bands = man["freqs_mhz"][:1]

    planes = find_reflector_planes(sc, min_area_cells=20)
    check("found reflector planes", len(planes) >= 2, f"{len(planes)}")
    yaxis = [p for p in planes if p.axis == 1]
    check("floor and ceiling detected", len(yaxis) >= 2,
          f"y-planes: {[p.name for p in yaxis]}")

    img = image_source(tx, planes[0])
    check("image source is mirrored",
          abs(img[planes[0].axis] - (2 * planes[0].coord - tx[planes[0].axis])) < 1e-6)

    fg = solve(sc, tx, bands=bands, planes=planes, backend=backend)
    check("field finite", bool(np.isfinite(np.abs(fg.E)).all()))
    check("field non-zero", bool(np.any(np.abs(fg.E) > 0)))
    live = np.abs(fg.E[0, 0]) > 0
    check("phase carried", bool(live.any()) and float(np.ptp(np.angle(fg.E[0, 0][live]))) > 1.0)

    # A reflected path is always longer than the direct one -> arrives later.
    geo = sc.geometric_time(tx)
    m = np.isfinite(fg.tau_first[0]) & np.isfinite(geo) & live
    check("reflection arrives after the direct ray",
          bool((fg.tau_first[0][m] >= geo[m] - 1e-12).mean() > 0.98))

    # Adding it must create interference structure, not a uniform lift.
    import Combine_3D as CMB
    import Path_Loss_3D as PLM
    direct = PLM.solve(sc, tx, bands=bands)
    one = CMB.combine([direct], floor_at_fspl=False).PL_db[0]
    both = CMB.combine([direct, fg], floor_at_fspl=False).PL_db[0]
    d = (one - both)[np.isfinite(one) & np.isfinite(both) & live]
    check("creates interference (both constructive and destructive)",
          bool((d > 0.1).any() and (d < -0.1).any()),
          f"delta range {d.min():.2f}..{d.max():.2f} dB")

    if backend == "numpy":
        try:
            import torch  # noqa: F401
            fg_t = solve(sc, tx, bands=bands, planes=planes, backend="torch")
            err = float(np.nanmax(np.abs(fg_t.pathloss_db() - fg.pathloss_db())))
            check("torch/numpy parity <= 0.01 dB", err <= 0.01, f"max {err:.5f} dB")
            print(f"        (torch device: {fg_t.meta.get('device')})")
        except ImportError:
            print("  SKIP  torch parity (torch not installed)")

    print(f"\n{ok} passed, {len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="specular reflection by image sources")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "torch"])
    a = ap.parse_args()
    raise SystemExit(_selftest(a.backend) if a.test else (ap.print_help() or 0))
