#!/usr/bin/env python3
"""
Scattering_3D — diffuse scattering by the effective-roughness model (Degli-Esposti).

The only fundamentally INCOHERENT mechanism, and the one that supplies the delay-spread
tail. Diffuse contributions arrive from many surface patches with uncorrelated phase, so
summing them as complex fields would be physically wrong: this module fills `p_incoh`
and leaves `E` at zero, and `Combine_3D` adds it as power.

ENERGY IS SPLIT, NOT INVENTED
A scattering coefficient S partitions the power an interface re-radiates:

    specular keeps   sqrt(1 - S^2) * R      (handed to Reflection_3D)
    diffuse  takes   S^2 * |R|^2            (radiated into a cos^alpha_R lobe)

so the model conserves energy BY CONSTRUCTION rather than adding a fudge term. That is
what makes it testable, and the self-test asserts exactly that: as S is swept, the power
the specular path loses equals the power the diffuse path gains.

WHY IT MATTERS
Without a diffuse component the power-delay profile has no tail, so RMS delay spread comes
out far too small and the channel looks more coherent than any measurement. The scanner's
`Ref Signal - Delay Spread` column (5,951 unused samples) is the observable that tests
this, and it tests the COMBINER rather than the path-loss model.

COST — this is the mechanism that actually wants the A100
Every lit surface patch radiates to every receiver: O(n_patches * n_rx). At full scene
resolution that is ~2e4 patches x 5.9e5 voxels x bands. Mitigations, all exposed as
arguments: patch decimation (`patch_cells`), receiver decimation with trilinear upsample
(`rx_downsample`), and a torch backend (CUDA on Colab, MPS on Apple Silicon).

usage:
    python Scattering_3D.py --test
    python Scattering_3D.py --test --backend torch
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy import ndimage

from _bootstrap import SIM3D, contracts, engine_3d, physics_3d  # noqa: F401

MECHANISM = "scattering"
DEFAULTS = {"S": 0.3, "alpha_R": 4, "patch_cells": 2, "rx_downsample": 2}
C0 = contracts.C0


# --------------------------------------------------------------------------- surfaces
def lit_surface_patches(scene, tx, *, patch_cells=2, max_patches=20000):
    """Surface voxels that face the transmitter, decimated into patches.

    Returns (pos (n,3) float32 voxels, normal (n,3) float32, mclass (n,) int8,
             area_m2 (n,) float32).
    """
    M = scene.M
    air = M == 0
    solid = ~air
    # a surface voxel is structure with at least one air neighbour
    surf = solid & ~ndimage.binary_erosion(solid, np.ones((3, 3, 3), bool))
    if patch_cells > 1:                      # decimate on a regular lattice
        keep = np.zeros_like(surf)
        keep[::patch_cells, ::patch_cells, ::patch_cells] = True
        surf = surf & keep
    idx = np.nonzero(surf)
    if len(idx[0]) == 0:
        return (np.zeros((0, 3), np.float32),) * 2 + (np.zeros(0, np.int8),
                                                      np.zeros(0, np.float32))
    pos = np.stack(idx, -1).astype(np.float32)

    nx, ny, nz = physics_3d.wall_normals_3d(M)
    nrm = np.stack([nx[idx], ny[idx], nz[idx]], -1).astype(np.float32)
    ln = np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm = np.divide(nrm, np.maximum(ln, 1e-6), dtype=np.float32)

    # keep only patches facing the transmitter
    to_tx = np.asarray(tx, np.float32) - pos
    to_tx /= np.maximum(np.linalg.norm(to_tx, axis=-1, keepdims=True), 1e-6)
    facing = np.einsum("ij,ij->i", nrm, to_tx) > 0.1
    pos, nrm = pos[facing], nrm[facing]
    mcl = M[idx][facing]

    if len(pos) > max_patches:               # farthest-first thinning would be O(n^2);
        sel = np.linspace(0, len(pos) - 1, max_patches).astype(int)   # stride instead
        pos, nrm, mcl = pos[sel], nrm[sel], mcl[sel]

    area = np.full(len(pos), (scene.cell * patch_cells) ** 2, np.float32)
    return pos, nrm, mcl.astype(np.int8), area


# --------------------------------------------------------------------------- model
def roughness_split(S):
    """(specular_amplitude_factor, diffuse_power_fraction) for scattering coefficient S.

    Degli-Esposti: the specular field is scaled by sqrt(1-S^2) and the removed power
    S^2*|R|^2 goes into the diffuse lobe. Energy is conserved identically.
    """
    S = float(np.clip(S, 0.0, 1.0))
    return float(np.sqrt(1.0 - S * S)), S * S


def _lobe(cos_s, alpha_R):
    """Normalized cos^alpha_R directive lobe about the specular direction."""
    c = np.clip(cos_s, 0.0, 1.0)
    norm = (alpha_R + 1.0) / (2.0 * np.pi)      # integrates to 1 over the hemisphere
    return norm * np.power(c, alpha_R)


def solve(scene, tx, *, bands=None, S=0.3, alpha_R=4, patch_cells=2,
          rx_downsample=2, max_patches=20000, backend="numpy", device=None,
          progress=False) -> "contracts.FieldGrid":
    """Diffuse scattered power for one Tx. Fills `p_incoh`; `E` stays zero."""
    freqs = np.asarray(scene.freqs if bands is None else bands, float)
    bidx = [scene.band_index(f) for f in freqs]
    NX, NY, NZ = scene.NX, scene.NY, scene.NZ
    inside = scene.inside if scene.inside is not None else np.ones(scene.M.shape, bool)
    _spec_amp, diffuse_frac = roughness_split(S)

    pos, nrm, mcl, area = lit_surface_patches(scene, tx, patch_cells=patch_cells,
                                              max_patches=max_patches)
    p_incoh = np.zeros((len(freqs), NX, NY, NZ), np.float32)
    tau_first = np.full((len(freqs), NX, NY, NZ), np.inf, np.float32)
    if len(pos) == 0 or diffuse_frac <= 0:
        fg = contracts.FieldGrid(
            E=np.zeros((contracts.NPOL, len(freqs), NX, NY, NZ), np.complex64),
            tau_first=tau_first, mechanism=MECHANISM, freqs_mhz=freqs,
            tx_vox=tuple(int(v) for v in tx), p_incoh=p_incoh,
            meta={"S": S, "n_patches": int(len(pos)), "note": "no diffuse power"})
        fg.combine_as = contracts.INCOHERENT
        return fg

    d = max(1, int(rx_downsample))
    gx, gy, gz = np.meshgrid(np.arange(0, NX, d), np.arange(0, NY, d),
                             np.arange(0, NZ, d), indexing="ij")
    R = np.stack([gx, gy, gz], -1).astype(np.float32)          # coarse receiver lattice
    cshape = R.shape[:3]

    obs_tx = scene.crossing_loss(tx)
    txv = np.asarray(tx, np.float32)

    if backend == "torch":
        acc, tau_c = _accumulate_torch(scene, pos, nrm, area, txv, R, freqs, bidx, obs_tx,
                                       alpha_R, diffuse_frac, device, progress)
    else:
        acc, tau_c = _accumulate_numpy(scene, pos, nrm, area, txv, R, freqs, bidx, obs_tx,
                                       alpha_R, diffuse_frac, progress)

    # upsample the coarse diffuse field back to the full grid
    for i in range(len(freqs)):
        if d == 1:
            p_incoh[i] = acc[i]
        else:
            zoom = (NX / cshape[0], NY / cshape[1], NZ / cshape[2])
            p_incoh[i] = ndimage.zoom(acc[i], zoom, order=1)[:NX, :NY, :NZ]
    p_incoh *= inside[None].astype(np.float32)

    # First diffuse arrival: shortest tx->patch->rx two-leg path (band-independent, so one
    # coarse solve broadcast across bands). This is what puts the diffuse halo LAST in the
    # mechanism time-lapse -- and it is causal by the triangle inequality, since
    # |tx-p| + |p-rx| >= |tx-rx| for every patch p.
    tau_up = _upsample_tau(tau_c, (NX, NY, NZ), d)
    tau_first[:] = np.where(inside, tau_up, np.inf)[None]

    fg = contracts.FieldGrid(
        E=np.zeros((contracts.NPOL, len(freqs), NX, NY, NZ), np.complex64),
        tau_first=tau_first, mechanism=MECHANISM, freqs_mhz=freqs,
        tx_vox=tuple(int(v) for v in tx), p_incoh=p_incoh,
        meta={"S": S, "alpha_R": alpha_R, "n_patches": int(len(pos)),
              "patch_cells": patch_cells, "rx_downsample": d, "backend": backend,
              "model": "effective roughness / directive (Degli-Esposti)",
              "note": "INCOHERENT: fills p_incoh, E is zero by design",
              "tau": "min over patches of (|tx-p| + |p-rx|)/c, band-independent"})
    fg.combine_as = contracts.INCOHERENT
    return fg


def _upsample_tau(tau_c, shape, d):
    """Coarse arrival-time lattice -> full grid, preserving the unreachable sentinel.

    inf cannot go through an interpolator, so unreachable cells ride a large finite
    sentinel and are restored to inf afterwards. Arrival time is smooth in space (unlike
    power, which spans decades), so linear interpolation is the right choice here.
    """
    NX, NY, NZ = shape
    if d == 1 and tau_c.shape == shape:
        return tau_c.astype(np.float32)
    finite = np.isfinite(tau_c)
    big = float(tau_c[finite].max() * 4.0) if finite.any() else 1.0
    filled = np.where(finite, tau_c, big).astype(np.float32)
    zoom = (NX / tau_c.shape[0], NY / tau_c.shape[1], NZ / tau_c.shape[2])
    up = ndimage.zoom(filled, zoom, order=1)[:NX, :NY, :NZ]
    return np.where(up < big * 0.5, up, np.inf).astype(np.float32)


def _accumulate_numpy(scene, pos, nrm, area, txv, R, freqs, bidx, obs_tx,
                      alpha_R, diffuse_frac, progress):
    """-> (power (nf, *coarse) float64, tau_first (*coarse) float32 seconds)."""
    cell = scene.cell
    out = np.zeros((len(freqs),) + R.shape[:3], np.float64)
    tau_min = np.full(R.shape[:3], np.inf, np.float32)
    Rf = R.reshape(-1, 3)
    for pi in range(len(pos)):
        p, n = pos[pi], nrm[pi]
        r_in = txv - p
        s_in = np.linalg.norm(r_in) * cell
        if s_in < cell:
            continue
        u_in = r_in / max(np.linalg.norm(r_in), 1e-6)
        cos_i = float(np.dot(n, u_in))
        if cos_i <= 0:
            continue
        # specular direction about the surface normal
        spec = 2.0 * cos_i * n - u_in

        r_out = Rf - p
        s_out = np.linalg.norm(r_out, axis=-1) * cell
        live = s_out > cell
        u_out = r_out / np.maximum(np.linalg.norm(r_out, axis=-1, keepdims=True), 1e-6)
        cos_s = u_out @ spec
        lobe = _lobe(cos_s, alpha_R)
        # patches only radiate into their own half-space
        lobe *= (u_out @ n > 0)

        # two-leg delay, counted only where this patch actually radiates
        radiates = live & (lobe > 0)
        if radiates.any():
            t_p = ((s_in + s_out) / C0).astype(np.float32)
            np.minimum(tau_min, np.where(radiates, t_p, np.inf).reshape(R.shape[:3]),
                       out=tau_min)

        for i, bi in enumerate(bidx):
            lam = C0 / (freqs[i] * 1e6)
            # incident power density at the patch (free-space spreading + wall crossings)
            pl_in = scene.fspl1[bi] + 10.0 * scene.n_exp * np.log10(max(s_in, scene.d0))
            pl_in += float(obs_tx[bi][int(p[0]), int(p[1]), int(p[2])])
            p_in = 10.0 ** (-pl_in / 10.0) * area[pi] * cos_i
            # diffuse re-radiation, 1/s^2 spreading outward
            with np.errstate(divide="ignore", invalid="ignore"):
                contrib = diffuse_frac * p_in * lobe / np.maximum(s_out ** 2, cell ** 2)
            contrib = np.where(live, contrib, 0.0)
            out[i] += contrib.reshape(R.shape[:3])
        if progress and pi % 500 == 0:
            print(f"  patch {pi}/{len(pos)}")
    return out, tau_min


def _accumulate_torch(scene, pos, nrm, area, txv, R, freqs, bidx, obs_tx,
                      alpha_R, diffuse_frac, device, progress):
    """-> (power (nf, *coarse) float64, tau_first (*coarse) float32 seconds)."""
    import torch
    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    dev = torch.device(device)
    # MPS has no float64; CUDA and CPU do. Diffuse power spans many decades, so use the
    # widest dtype the device supports rather than silently degrading everywhere.
    acc_dt = torch.float32 if dev.type == "mps" else torch.float64
    cell = scene.cell
    Rf = torch.as_tensor(R.reshape(-1, 3), device=dev)
    P = torch.as_tensor(pos, device=dev)
    N = torch.as_tensor(nrm, device=dev)
    A = torch.as_tensor(area, device=dev)
    tx_t = torch.as_tensor(txv, device=dev)
    out = torch.zeros((len(freqs), Rf.shape[0]), dtype=acc_dt, device=dev)
    tau_min = torch.full((Rf.shape[0],), float("inf"), dtype=torch.float32, device=dev)

    r_in = tx_t[None] - P
    s_in = torch.linalg.norm(r_in, dim=-1) * cell
    u_in = r_in / torch.clamp(torch.linalg.norm(r_in, dim=-1, keepdim=True), min=1e-6)
    cos_i = (N * u_in).sum(-1)
    ok = (cos_i > 0) & (s_in > cell)
    spec = 2.0 * cos_i[:, None] * N - u_in

    obs_t = [torch.as_tensor(np.asarray(obs_tx[bi]), device=dev) for bi in bidx]
    sel = torch.nonzero(ok).flatten()
    if sel.numel() == 0:
        return (out.reshape((len(freqs),) + R.shape[:3]).cpu().numpy(),
                tau_min.reshape(R.shape[:3]).cpu().numpy())

    # Per-patch incident power, all patches and bands at once (n_patch, n_band).
    ixyz = P[sel].long()
    p_in = torch.empty((sel.numel(), len(freqs)), dtype=acc_dt, device=dev)
    for i, bi in enumerate(bidx):
        pl_in = (scene.fspl1[bi] + 10.0 * scene.n_exp
                 * torch.log10(torch.clamp(s_in[sel], min=scene.d0))
                 + obs_t[i][ixyz[:, 0], ixyz[:, 1], ixyz[:, 2]])
        p_in[:, i] = (torch.pow(10.0, -pl_in / 10.0) * A[sel] * cos_i[sel]).to(acc_dt)

    # CHUNKED over patches rather than looped: the earlier per-patch Python loop issued
    # ~200k tiny kernels per Tx and left the A100 at 0% utilisation -- launch-bound, not
    # compute-bound. Chunking turns it into a handful of large batched reductions.
    Pc, Nc, Sc = P[sel], N[sel], spec[sel]
    s_in_c = s_in[sel]
    n_rx = Rf.shape[0]
    chunk = max(1, min(int(sel.numel()), max(1, (1 << 24) // max(n_rx, 1))))
    norm_lobe = (alpha_R + 1.0) / (2.0 * np.pi)
    for k in range(0, sel.numel(), chunk):
        pc, nc, sc_ = Pc[k:k + chunk], Nc[k:k + chunk], Sc[k:k + chunk]
        r_out = Rf[None, :, :] - pc[:, None, :]                       # (c, n_rx, 3)
        dist = torch.linalg.norm(r_out, dim=-1)
        s_out = dist * cell
        u_out = r_out / torch.clamp(dist, min=1e-6).unsqueeze(-1)
        cos_s = torch.einsum("crk,ck->cr", u_out, sc_)
        lobe = norm_lobe * torch.clamp(cos_s, min=0) ** alpha_R
        lobe = lobe * (torch.einsum("crk,ck->cr", u_out, nc) > 0)
        radiates = (s_out > cell) & (lobe > 0)
        geo = (lobe / torch.clamp(s_out ** 2, min=cell ** 2)).to(acc_dt)
        geo = torch.where(s_out > cell, geo, torch.zeros((), dtype=acc_dt, device=dev))
        # (n_band, c) @ (c, n_rx) -> (n_band, n_rx): one matmul per chunk
        out += diffuse_frac * (p_in[k:k + chunk].T @ geo)
        # earliest two-leg delay, same chunk, one more batched reduction (not a per-patch
        # loop -- that is what left the A100 launch-bound before)
        t_chunk = (s_in_c[k:k + chunk, None] + s_out) / C0
        t_chunk = torch.where(radiates, t_chunk.float(),
                              torch.full((), float("inf"), device=dev))
        tau_min = torch.minimum(tau_min, t_chunk.amin(dim=0))
        if progress:
            print(f"  patches {min(k+chunk, sel.numel())}/{sel.numel()}", flush=True)
    return (out.reshape((len(freqs),) + R.shape[:3]).cpu().numpy(),
            tau_min.reshape(R.shape[:3]).cpu().numpy())


# --------------------------------------------------------------------------- self-test
def _selftest(backend="numpy") -> int:
    sys.path.insert(0, str(SIM3D / "tests3d"))
    import synth3d as S_

    ok, fails = 0, []

    def check(name, cond, msg=""):
        nonlocal ok
        if cond:
            ok += 1
            print(f"  PASS  {name}")
        else:
            fails.append(name)
            print(f"  FAIL  {name}  {msg}")

    M, man = S_.rough_wall_scene_3d(n=(32, 16, 24), mat=2, sigma_cells=1.0)
    sc = engine_3d.SceneV3(M, man)
    sc.inside = (M == 0)
    tx = (6, 8, 12)
    bands = man["freqs_mhz"][:1]

    pos, nrm, mcl, area = lit_surface_patches(sc, tx, patch_cells=2)
    check("found lit patches", len(pos) > 0, f"{len(pos)}")
    check("normals are unit", bool(np.allclose(np.linalg.norm(nrm, axis=-1), 1.0, atol=0.05))
          if len(nrm) else False)

    fg = solve(sc, tx, bands=bands, S=0.4, patch_cells=2, rx_downsample=2,
               backend=backend)
    check("is declared INCOHERENT", fg.combine_as == contracts.INCOHERENT)
    check("E is zero by design", not np.any(np.abs(fg.E)))
    check("p_incoh populated", fg.p_incoh is not None and float(fg.p_incoh.max()) > 0)
    check("p_incoh finite and non-negative",
          bool(np.isfinite(fg.p_incoh).all() and (fg.p_incoh >= 0).all()))

    # ---- diffuse arrival time (what places the halo LAST in the time-lapse) ----
    tau = fg.tau_first[0]
    live = np.isfinite(tau)
    check("tau_first populated", bool(live.any()),
          "all inf -- the diffuse halo would never appear in the time-lapse")
    gx, gy, gz = np.meshgrid(*[np.arange(s) for s in M.shape], indexing="ij")
    d_direct = np.sqrt((gx - tx[0]) ** 2 + (gy - tx[1]) ** 2
                       + (gz - tx[2]) ** 2) * sc.cell
    tau_direct = d_direct / C0
    # two legs via a patch can never beat the straight line (triangle inequality); allow
    # one coarse-lattice cell of interpolation slack
    slack = (sc.cell * 2.0) / C0
    check("diffuse arrival is causal",
          bool((tau[live] >= tau_direct[live] - slack).all()),
          f"min margin {float(np.min(tau[live] - tau_direct[live])) * 1e9:.2f} ns")
    check("diffuse arrives after the direct path",
          float(np.median(tau[live] - tau_direct[live])) > 0)

    # S = 0 must produce exactly no diffuse power
    fg0 = solve(sc, tx, bands=bands, S=0.0, patch_cells=2, rx_downsample=2,
                backend=backend)
    check("S=0 gives zero diffuse power", float(np.max(fg0.p_incoh)) == 0.0)

    # energy split: diffuse power must scale as S^2, and specular loss must match it
    a, b = roughness_split(0.0), roughness_split(1.0)
    check("S=0 keeps all specular", abs(a[0] - 1.0) < 1e-9 and a[1] == 0.0)
    check("S=1 moves all power to diffuse", abs(b[0]) < 1e-9 and abs(b[1] - 1.0) < 1e-9)
    for S_val in (0.2, 0.5, 0.8):
        amp, frac = roughness_split(S_val)
        check(f"energy conserved at S={S_val}", abs(amp ** 2 + frac - 1.0) < 1e-9,
              f"{amp**2 + frac:.6f}")

    # diffuse power should grow with S^2
    p2 = float(solve(sc, tx, bands=bands, S=0.2, patch_cells=2, rx_downsample=2,
                     backend=backend).p_incoh.sum())
    p4 = float(solve(sc, tx, bands=bands, S=0.4, patch_cells=2, rx_downsample=2,
                     backend=backend).p_incoh.sum())
    check("diffuse power scales as S^2", abs(p4 / max(p2, 1e-30) - 4.0) < 0.05,
          f"ratio {p4/max(p2,1e-30):.3f} (expect 4.0)")

    # combining must add it as POWER, not field
    import Combine_3D as CMB
    import Path_Loss_3D as PLM
    direct = PLM.solve(sc, tx, bands=bands)
    cf = CMB.combine([direct, fg], floor_at_fspl=False)
    check("combiner treats it as incoherent",
          cf.meta["incoherent"] == [MECHANISM], str(cf.meta["incoherent"]))
    check("adding diffuse never increases path loss",
          bool((cf.PL_db <= CMB.combine([direct], floor_at_fspl=False).PL_db + 1e-6).all()))

    print(f"\n{ok} passed, {len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="diffuse scattering (effective roughness)")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "torch"])
    a = ap.parse_args()
    raise SystemExit(_selftest(a.backend) if a.test else (ap.print_help() or 0))
