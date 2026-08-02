#!/usr/bin/env python3
"""
engine_3d.py — SceneV3: full-physics 3-D RF path loss + eikonal time (SIM V1 3D).

Two co-registered grids per transmitter, computed in true (x, y, z):

  PL(x,y,z)  direct through-wall loss = Motley-Keenan spreading (n=2)
             + Σ per-crossing slab loss from the CrossingLUT — real
             angle+thickness Fresnel/Airy transmission AND Im(q) bulk
             absorption (physics_v2 Blocks A/B). Diffraction/reflection/
             scattering are added by later-stage modules (power-summed).
  T(x,y,z)   first-arrival time from ONE 3-D eikonal fast-march
             (|∇T| = 1/speed, scikit-fmm) on the slowness field s = Re(√ε_r)/c
             with opaque classes masked — so refraction (Snell bending, in-wall
             lag) and diffraction (corner wraparound) come out of the single
             solve (Fermat). This is a real physics field, not d/c.

The EM kernels are reused verbatim from SIM/physics_v2.py via physics_3d.py;
this module is the 3-D geometry (ray-march, normals, run-length thickness) that
was 2-D in engine_v2.py.
"""
import json
from pathlib import Path

import numpy as np
import skfmm

import physics_3d as P3

HERE = Path(__file__).resolve().parent
STEP_CELL = 0.5          # ray sample spacing (voxels): catches 1-voxel walls
K_QUANTUM = 32           # quantize per-ray sample count → few unique-K groups
MEM_CAP = 3_000_000      # cap (K × chunk) samples held at once


def sat_obs(x, T, S):
    """Saturating obstruction stand-in for the summed direct wall loss (dB).

    Loss grows linearly up to T, then asymptotes to T + S:
        x <= T  ->  x
        x >  T  ->  T + S * (1 - exp(-(x - T) / S))

    Ported verbatim from the 2-D browser model (SIM/web/simulator_tab.js:31). The flat
    per-crossing dB stack in `crossing_loss` has no upper bound, so a ray that clips many
    voxels of thick structure (straight through the core) runs away to 300-500 dB — which
    made the diffuse channel appear to dominate 94.6 % of interior voxels. A real signal
    stops attenuating once it is buried; the diffracted/scattered fields, not an unbounded
    transmission stack, are what fill deep shadows. Applied to the crossing sum ONLY, never
    to free space (sat_obs(0) == 0, so vacuum/FSPL is untouched)."""
    return np.where(x <= T, x, T + S * (1.0 - np.exp(-(x - T) / S)))


class SceneV3:
    def __init__(self, M, manifest, materials=P3.MATERIALS, barrier_classes=(3,)):
        self.M = np.ascontiguousarray(M.astype(np.int8))
        self.NX, self.NY, self.NZ = self.M.shape
        self.cell = float(manifest["cell_size_m"])
        self.freqs = np.asarray(manifest["freqs_mhz"], float)
        self.materials = materials
        self.barrier_classes = tuple(barrier_classes)
        phys = manifest["physics"]
        self.n_exp = float(phys["n_exp"])
        self.d0 = float(phys["d0_m"])
        self.fspl1 = 20.0 * np.log10(self.freqs) + float(phys["fspl_const_db"])
        # Saturating obstruction cap for the DIRECT path (see sat_obs). Params come from
        # physics.fallback_obstruction_db; the engine now uses them (they were previously
        # browser-only). Off only if the manifest sets physics.use_satobs = false.
        _obs = phys.get("fallback_obstruction_db", {})
        self.use_satobs = bool(phys.get("use_satobs", True))
        self.obs_T = float(_obs.get("obstruction_linear_db", 40.0))
        self.obs_S = float(_obs.get("obstruction_sat_extra_db", 50.0))

        # geometry + material machinery (all reused / ported)
        self.lut = P3.CrossingLUT(materials, self.freqs)
        self.nx, self.ny, self.nz = P3.wall_normals_3d(self.M)
        self.w_ref = P3.measure_nominal_widths_3d(self.M, len(materials), self.cell)
        self.t_ref = np.array([m.get("t_ref_m", 0.0) for m in materials], np.float32)
        self.per_metre = np.array([bool(m.get("per_metre")) for m in materials])

        gx, gy, gz = np.meshgrid(np.arange(self.NX), np.arange(self.NY),
                                 np.arange(self.NZ), indexing="ij")
        self.P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1).astype(np.float32)
        self.inside = None

    def band_index(self, f_mhz):
        return int(np.argmin(np.abs(self.freqs - f_mhz)))

    # -- direct through-wall loss: 3-D port of SceneV2.obstruction_maps -------
    def crossing_loss(self, tx):
        """Summed per-crossing wall loss (dB, excluding free space) for every
        voxel and every band → (nf, NX, NY, NZ). One geometric pass, all bands
        via the CrossingLUT (angle+thickness Fresnel/Airy + Im(q) absorption)."""
        M = self.M
        cell = self.cell
        tx = np.asarray(tx, np.float32)
        shape = np.array(M.shape)
        N = self.P.shape[0]
        nf = len(self.freqs)
        d_cells = np.linalg.norm(self.P - tx, axis=1)
        out = np.zeros((nf, N), np.float32)

        need = np.ceil(d_cells / STEP_CELL).astype(np.int64) + 2
        kq = ((np.maximum(need, K_QUANTUM) + K_QUANTUM - 1) // K_QUANTUM) * K_QUANTUM

        for K in np.unique(kq):
            K = int(K)
            idx = np.nonzero(kq == K)[0]
            t = np.linspace(0.0, 1.0, K, dtype=np.float32)          # (K,)
            step = max(1, MEM_CAP // K)
            for s0 in range(0, idx.size, step):
                sel = idx[s0:s0 + step]
                ends = self.P[sel]                                   # (m,3)
                # sample coords: rows = rays, cols = K samples  (m, K)
                xi = np.clip(np.rint(tx[0] + (ends[:, 0] - tx[0])[:, None] * t[None, :]),
                             0, shape[0] - 1).astype(np.int32)
                yi = np.clip(np.rint(tx[1] + (ends[:, 1] - tx[1])[:, None] * t[None, :]),
                             0, shape[1] - 1).astype(np.int32)
                zi = np.clip(np.rint(tx[2] + (ends[:, 2] - tx[2])[:, None] * t[None, :]),
                             0, shape[2] - 1).astype(np.int32)
                mats = M[xi, yi, zi]                                 # (m, K)
                seg_m = (d_cells[sel] / max(K - 1, 1) * cell).astype(np.float32)  # (m,)

                # crossings = maximal runs of one material along the ray (axis 1)
                same = mats[:, 1:] == mats[:, :-1]
                starts = np.empty_like(mats, bool); ends_b = np.empty_like(mats, bool)
                starts[:, 0] = True; starts[:, 1:] = ~same
                ends_b[:, -1] = True; ends_b[:, :-1] = ~same
                sr, sc = np.nonzero(starts)                          # ray idx, sample idx
                er, ec = np.nonzero(ends_b)
                m_run = mats[sr, sc]
                keep = m_run > 0
                if not keep.any():
                    continue
                sr, sc, ec, m_run = sr[keep], sc[keep], ec[keep], m_run[keep]
                run_cells = (ec - sc + 1).astype(np.float32)
                run_len_m = run_cells * seg_m[sr]

                # incidence angle, averaged over the two run faces (reciprocal)
                rx = ends[sr, 0] - tx[0]; ry = ends[sr, 1] - tx[1]; rz = ends[sr, 2] - tx[2]
                rn = np.maximum(np.sqrt(rx * rx + ry * ry + rz * rz), 1e-6)
                cos_acc = np.zeros(sr.size, np.float32)
                n_ok = np.zeros(sr.size, np.float32)
                for cc in (sc, ec):
                    ax, ay, az = xi[sr, cc], yi[sr, cc], zi[sr, cc]
                    wx, wy, wz = self.nx[ax, ay, az], self.ny[ax, ay, az], self.nz[ax, ay, az]
                    ok = (wx * wx + wy * wy + wz * wz) > 0.25
                    cos_acc += np.where(ok, np.abs((rx * wx + ry * wy + rz * wz) / rn), 0.0)
                    n_ok += ok
                cos_th = np.where(n_ok > 0, cos_acc / np.maximum(n_ok, 1), 1.0)
                cos_th = np.clip(cos_th, 0.06, 1.0).astype(np.float32)

                # thickness: along-ray run projected onto the normal, then scaled
                # by class construction thickness / measured raster width
                t_raster = run_len_m * cos_th
                scale = self.t_ref[m_run] / np.maximum(self.w_ref[m_run], 1e-6)
                thick = (t_raster * scale).astype(np.float32)

                per = self.lut.loss_all_bands(m_run, cos_th, thick)   # (nf, ncross)
                is_pm = self.per_metre[m_run]
                if is_pm.any():
                    per = np.where(is_pm[None, :], per * run_len_m[None, :], per)
                for fi in range(nf):
                    out[fi, sel] += np.bincount(sr, weights=per[fi], minlength=sel.size)
        return out.reshape(nf, self.NX, self.NY, self.NZ)

    def pathloss_maps(self, tx):
        """PL(nf, NX, NY, NZ) = free-space spreading + (saturated) direct crossing loss."""
        tx = np.asarray(tx, np.float32)
        d_m = np.maximum(np.linalg.norm(self.P - tx, axis=1) * self.cell, self.d0)
        fs = (self.fspl1[:, None] + 10.0 * self.n_exp * np.log10(d_m)[None, :]).astype(np.float32)
        fs = fs.reshape(len(self.freqs), self.NX, self.NY, self.NZ)
        cl = self.crossing_loss(tx)
        if self.use_satobs:                       # cap the direct transmission runaway
            cl = sat_obs(cl, self.obs_T, self.obs_S).astype(np.float32)
        return fs + cl

    def pathloss_map(self, tx, f_mhz):
        return self.pathloss_maps(tx)[self.band_index(f_mhz)]

    # -- eikonal arrival time (the real T) -----------------------------------
    def arrival_time(self, tx, f_mhz):
        """First-arrival time T(NX,NY,NZ) in seconds from a 3-D eikonal solve.
        Refraction (bending/lag) and diffraction (wraparound) emerge from the
        single fast-march; opaque classes are masked so the front routes around
        them. Returns +inf inside barriers / unreachable cells."""
        speed, barrier = P3.speed_field(self.M, f_mhz, self.materials, self.barrier_classes)
        phi = np.ones(self.M.shape)
        ijk = tuple(int(round(v)) for v in tx)
        phi[ijk] = -1.0                                        # zero contour at Tx
        phi_ma = np.ma.MaskedArray(phi, mask=barrier)
        T = skfmm.travel_time(phi_ma, speed, dx=self.cell)
        return np.ma.filled(T, np.inf).astype(np.float32)

    def geometric_time(self, tx):
        """Straight-line d/c (seconds) — the lower bound the eikonal must meet."""
        d_m = np.linalg.norm(self.P - np.asarray(tx, np.float32), axis=1) * self.cell
        return (d_m / P3.C_LIGHT).reshape(self.M.shape).astype(np.float32)


def load_scene(sim_dir=HERE, barrier_classes=(3,)):
    manifest = json.loads((Path(sim_dir) / "manifest_3d.json").read_text())
    M = np.load(Path(sim_dir) / "material_grid.npy")
    scene = SceneV3(M, manifest, barrier_classes=barrier_classes)
    try:
        scene.inside = np.load(Path(sim_dir) / "inside_mask.npy")
    except FileNotFoundError:
        pass
    return scene, manifest


if __name__ == "__main__":
    # smoke test: PL + eikonal T for a ceiling Tx at the interior centroid
    scene, man = load_scene()
    inside = scene.inside
    tx = np.argwhere(inside).mean(0)
    tx[1] = inside.shape[1] * 0.85
    ijk = np.argwhere(inside)
    tx = ijk[np.argmin(((ijk - tx) ** 2).sum(1))].astype(float)

    PL = scene.pathloss_map(tx, 3500.0)
    T = scene.arrival_time(tx, 3500.0)
    Tgeo = scene.geometric_time(tx)
    pin = PL[inside]
    reach = np.isfinite(T) & inside
    print(f"tx={tx.astype(int).tolist()}  PL(interior) min/mean/max = "
          f"{pin.min():.1f}/{pin.mean():.1f}/{pin.max():.1f} dB")
    print(f"T reachable interior voxels: {reach.sum()}/{inside.sum()}  "
          f"T max = {T[reach].max()*1e9:.1f} ns")
    slack = (T[reach] - Tgeo[reach])
    tol = 3.0 * scene.cell / P3.C_LIGHT          # spec tol_T ≈ 3 cells of travel time
    print(f"T - d/c: min={slack.min()*1e9:.3f} ns, mean={slack.mean()*1e9:.2f} ns "
          f"(≥ straight-line within grid tol {tol*1e9:.2f} ns; a sub-cell source "
          f"offset is expected)")
    assert np.isfinite(PL).all(), "non-finite PL"
    assert slack.min() > -tol, "eikonal beat straight-line beyond grid tolerance — bug"
    print("NOTE: direct-only PL — deep shadows read high (e.g. straight through the "
          "core) until diffraction (Stage 2) power-fills them.")
    print("engine_3d (full physics) smoke test OK")
