#!/usr/bin/env python3
"""
fullwave3d.py — the SIM V2 3-D full-wave solve entry point.

One thin wrapper over the dimension-agnostic `FullWaveScene` engine that defaults
to the JAX/XLA hot path (`fw_fdtd_jax.simulate_jax`) and returns the steady-state
complex field U(x,y,z). The engine and medium builder are IMPORTED from SIM V3 /
SIM V1 3D (via `bootstrap`) — not reimplemented. This is what `run_wave3d`,
`fw_field3d`, `fw_dataset3d`, and `hybrid_field_3d` call to get a 3-D solve.

  from fullwave3d import solve3d, solve_band
  res = solve_band("LTE_B71_617", npw=8.0)          # extract scene + solve
  U = res["phasor"]                                  # complex (nx,ny,nz)
"""
from __future__ import annotations

import argparse

import numpy as np

import bootstrap  # noqa: F401  (side effect: sys.path → engine)
import fullwave2d as FW
from fullwave2d import FullWaveScene
from fw_fdtd_jax import simulate_jax
from Time_Domain_Physics import WaveSim

C0 = 299_792_458.0
EPS0 = 8.854_187_8128e-12


def steps_for(extent_m, dt, crossings=1.5) -> int:
    """Number of timesteps for `crossings` domain crossings at timestep dt."""
    return int(round(crossings * max(extent_m) / C0 / dt))


def solve3d(classes, h_m, f_mhz, tx, source="cw", crossings=1.5, warmup_frac=0.5,
            dtype="float32", extract_phasor=True, with_loss=True, backend="jax",
            steps=None, **scene_kw):
    """Solve the 3-D scalar full-wave field on a fine material-class grid.

    backend: "jax" (default, XLA — CPU/GPU) or "numpy" (the reference engine).
    Returns dict(phasor U, rms, steps, dt, cfl, dt_run, mcps, backend, sim).
    """
    sim = FullWaveScene(classes, h_m, f_mhz, tx, source=source,
                        with_loss=with_loss, **scene_kw)
    extent = tuple(s * h_m for s in classes.shape)
    n_steps = int(steps) if steps else steps_for(extent, sim.dt, crossings)

    if backend == "jax":
        res = simulate_jax(sim, n_steps, warmup_frac=warmup_frac,
                           extract_phasor=extract_phasor, dtype=dtype)
    else:
        res = FW.simulate(sim, n_steps, warmup_frac=warmup_frac,
                          extract_phasor=extract_phasor)
        res.setdefault("mcps", classes.size * n_steps / max(res["dt_run"], 1e-9) / 1e6)
        res.setdefault("backend", "numpy")

    res.update(steps=n_steps, dt=sim.dt, cfl=sim.cfl(), h_m=h_m,
               extent_m=extent, sim=sim)
    return res


def solve_band(band_label="LTE_B71_617", npw=8.0, crop_m=None, tx_m=None,
               max_cells=8_000_000, **kw):
    """Extract the scene for a band (solid_extract) and solve it in one call."""
    from bands_v3 import get
    from solid_extract import extract_solid
    s = extract_solid(get(band_label), n_per_wavelength=npw, crop_m=crop_m,
                      tx_m=tx_m, max_cells=max_cells)
    res = solve3d(s.classes, s.h_m, get(band_label).f_mhz, s.tx_idx, **kw)
    res["solid"] = s
    return res


class EffectiveMediumScene(WaveSim):
    """Continuous effective-medium FDTD scene: builds c(x) and γ(x) from per-cell
    εr_eff/σ_eff (the conformal voxelizer) instead of an integer class grid. Reuses
    FullWaveScene.step verbatim — only the medium construction differs."""
    step = FullWaveScene.step
    cfl = FullWaveScene.cfl

    @classmethod
    def from_effective_medium(cls, epsr_eff, sigma_eff, metal_frac, h_m, f_mhz,
                              source_xy, source="cw", metal_threshold=0.5,
                              sponge_width=None):
        eps = np.asarray(epsr_eff, float)
        c = C0 / np.maximum(np.sqrt(np.maximum(eps, 1e-6)), 1e-3)   # c = C0/Re(√εr)
        rigid = np.asarray(metal_frac, float) > metal_threshold      # metal → reflector
        if sponge_width is None:
            sponge_width = max(20, int(0.06 * min(eps.shape)))
        self = cls(c, h_m, source_xy, float(f_mhz) * 1e6, source=source,
                   rigid=rigid, sponge_width=sponge_width)           # WaveSim sets dt/cdt2/…
        gamma = np.asarray(sigma_eff, float) / (2.0 * EPS0 * np.maximum(eps, 1e-6))
        gamma[rigid] = 0.0                        # metal → rigid reflector, no bulk loss (u=0)
        gamma = np.minimum(gamma, 0.5 / max(self.dt, 1e-30))         # cap a≤0.5 (stability)
        self.a = (gamma * self.dt).astype(np.float64)                # γ·dt (needs self.dt)
        self._inv1pa = 1.0 / (1.0 + self.a)
        self._1ma = 1.0 - self.a
        self.f_mhz = float(f_mhz); self.classes = None
        return self


def solve3d_em(epsr_eff, sigma_eff, metal_frac, h_m, f_mhz, tx, source="cw",
               crossings=1.5, warmup_frac=0.5, dtype="float32", extract_phasor=True,
               steps=None):
    """Solve the continuous effective-medium 3-D field on the JAX teacher."""
    sim = EffectiveMediumScene.from_effective_medium(epsr_eff, sigma_eff, metal_frac,
                                                     h_m, f_mhz, tx, source=source)
    extent = tuple(s * h_m for s in epsr_eff.shape)
    n_steps = int(steps) if steps else steps_for(extent, sim.dt, crossings)
    res = simulate_jax(sim, n_steps, warmup_frac=warmup_frac,
                       extract_phasor=extract_phasor, dtype=dtype)
    res.update(steps=n_steps, dt=sim.dt, cfl=sim.cfl(), h_m=h_m, extent_m=extent, sim=sim)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--npw", type=float, default=6.0)
    ap.add_argument("--crossings", type=float, default=1.5)
    ap.add_argument("--backend", default="jax", choices=["jax", "numpy"])
    ap.add_argument("--max-cells", type=int, default=6_000_000)
    a = ap.parse_args()
    res = solve_band(a.band, npw=a.npw, crossings=a.crossings, backend=a.backend,
                     max_cells=a.max_cells)
    s = res["solid"]
    U = res["phasor"]
    print(f"solve {s.classes.shape} ({s.classes.size/1e6:.2f} M) x {res['steps']} steps  "
          f"[{res['backend']}] {res['dt_run']:.1f}s {res['mcps']:.0f} Mcps  "
          f"cfl={res['cfl']:.3f}  finite={res['finite']}")
    print(f"|U| max={np.abs(U).max():.3e}  extent={tuple(round(e,1) for e in res['extent_m'])} m")


if __name__ == "__main__":
    main()
