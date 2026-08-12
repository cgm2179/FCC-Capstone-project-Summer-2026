#!/usr/bin/env python3
"""
Time_Domain_Physics.py — time integrator + sources + run loop for the
full-wave sandbox (Wave Behavior / Wave Generation).

Steps the scalar wave equation with a leapfrog (Stormer-Verlet) update:

    u^{n+1} = 2 u^n - u^{n-1} + (c*dt)^2 * laplacian(u^n) + dt^2 * S(t)

Pairs with Spatial_Physics.py, which supplies the operator and the medium c(x).

  * courant_dt()   CFL-stable timestep  dt = safety * h / (c_max * sqrt(ndim))
  * ricker() / cw()  broadband impulse / continuous-wave point source
  * sponge()       cosine-ramped absorbing boundary layer (crude ABC)
  * WaveSim        holds u_prev / u_curr, .step(), .run()

======================= RESOLUTION BUDGET (read first) =======================
Full-wave FDTD needs cells h <= lambda/10 AND dt <= h/(c*sqrt(dim)).
You cannot have all three of {real building size, real RF frequency,
full-wave} at once -- pick two:

  f = 3.5 GHz -> lambda = 8.6 cm -> h <= 8.6 mm
    * a 256^3 grid at 8.6 mm spans only ~2.2 m   (one room corner)
    * a 77 m floor at 8.6 mm needs ~9000^3 ~ 7e11 cells   (infeasible)

Three practical modes -- SAME code, different inputs:
  A. Scaled-frequency : low effective f so a coarse grid resolves the wave over
                        the whole model (a "ripple tank" on the floor plan).
                        Generalizes to any model; intuition, not calibrated RF.
  B. Small-domain     : crop to a room/doorway at h <= lambda/10 for real f.
  C. 2-D slice        : one horizontal plane -> N^2 not N^3 -> finer h.

Validate in 2-D first (--demo), THEN lift the identical code to the 3-D grid.
=============================================================================

usage:
  python Time_Domain_Physics.py --demo            # 2-D point source + slit
  python Time_Domain_Physics.py --demo --n 600 --steps 1200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import Spatial_Physics as SP

HERE = Path(__file__).resolve().parent


def courant_dt(c_max, h, ndim, safety=0.5):
    """CFL-stable timestep. safety must be < 1/sqrt(ndim); 0.5 is comfortable."""
    return safety * h / (c_max * np.sqrt(ndim))


def ricker(t, f0):
    """Ricker (Mexican-hat) broadband pulse, delayed to start near zero."""
    a = (np.pi * f0 * (t - 1.0 / f0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def cw(t, f0):
    """continuous sine source, sin(2*pi*f0*t)."""
    return np.sin(2.0 * np.pi * f0 * t)


def sponge(shape, width, absorb=0.08):
    """multiplicative absorbing layer: 1.0 in the interior, easing to (1-absorb)
    at every face over `width` cells. Crude ABC -- upgrade to Mur/PML for
    quantitative work. Also mops up the np.roll edge wrap from the Laplacian."""
    damp = np.ones(shape, dtype=np.float64)
    for ax in range(len(shape)):
        n = shape[ax]
        w = min(width, n // 2)
        if w <= 0:
            continue
        edge = np.linspace(0.0, 1.0, w)           # 0 at the boundary, 1 inside
        atten = 1.0 - absorb * (1.0 - edge) ** 2  # (1-absorb) -> 1.0
        prof = np.ones(n)
        prof[:w] = atten
        prof[-w:] = atten[::-1]
        bshape = [1] * len(shape)
        bshape[ax] = n
        damp *= prof.reshape(bshape)
    return damp


class WaveSim:
    """Leapfrog scalar-wave field. Dimension-agnostic: pass a 2-D or 3-D c."""

    def __init__(self, c, h, source_xy, f0, source="ricker",
                 rigid=None, sponge_width=20, safety=0.5):
        self.c = np.ascontiguousarray(c.astype(np.float64))
        self.ndim = self.c.ndim
        self.h = float(h)
        self.inv_h2 = 1.0 / self.h ** 2
        self.dt = courant_dt(self.c.max(), self.h, self.ndim, safety)
        self.f0 = float(f0)
        self.source = source
        self.src_idx = tuple(int(round(v)) for v in source_xy)
        self.rigid = rigid if rigid is not None else np.zeros(self.c.shape, bool)
        self.damp = sponge(self.c.shape, sponge_width)
        self.cdt2 = (self.c * self.dt) ** 2
        self.u = np.zeros(self.c.shape)       # u^n
        self.u_prev = np.zeros(self.c.shape)  # u^{n-1}
        self.n = 0

    def step(self):
        t = self.n * self.dt
        lap = SP.laplacian(self.u, self.inv_h2)
        u_next = 2.0 * self.u - self.u_prev + self.cdt2 * lap
        # soft point source (amplitude arbitrary -- normalize when rendering)
        s = ricker(t, self.f0) if self.source == "ricker" else cw(t, self.f0)
        u_next[self.src_idx] += s
        u_next[self.rigid] = 0.0            # perfect reflectors
        u_next *= self.damp                 # absorbing boundary
        self.u_prev = self.u * self.damp
        self.u = u_next
        self.n += 1
        return self.u

    def run(self, steps, record_every=0):
        frames = []
        for _ in range(steps):
            self.step()
            if record_every and self.n % record_every == 0:
                frames.append(self.u.copy())
        if not np.isfinite(self.u).all():
            raise FloatingPointError("field blew up -- CFL/safety too high")
        return frames


def _demo(n, steps):
    """2-D validation: point source, a rigid barrier with a slit -> you should
    see spreading, reflection off the barrier, and diffraction through the slit,
    with the edges quietly absorbing."""
    M = np.zeros((n, n), dtype=np.int8)
    bar = n // 2
    M[bar - 1:bar + 1, :] = 3                      # rigid barrier (class 3)
    slit = slice(n // 2 - n // 40, n // 2 + n // 40)
    M[bar - 1:bar + 1, slit] = 0                   # open a slit
    c = np.ones((n, n))                            # normalized medium (mode A)
    rigid = SP.rigid_mask(M)
    sim = WaveSim(c, h=1.0, source_xy=(n // 4, n // 2), f0=0.05, rigid=rigid)
    print(f"grid {c.shape}  dt={sim.dt:.3f}  CFL={c.max()*sim.dt/sim.h:.3f}  "
          f"lambda~{1/0.05:.0f} cells")
    frames = sim.run(steps, record_every=max(1, steps // 4))
    np.save(HERE / "demo_frames.npy", np.array(frames))
    print(f"ok -- {len(frames)} frames -> demo_frames.npy")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f = frames[-1]
        m = np.abs(f).max() or 1.0
        plt.figure(figsize=(5, 5))
        plt.imshow(f.T, cmap="RdBu", vmin=-m, vmax=m, origin="lower")
        plt.contour(rigid.T, colors="k", linewidths=0.4)
        plt.title("scalar-wave FDTD: reflection + slit diffraction")
        plt.tight_layout()
        plt.savefig(HERE / "demo_wave.png", dpi=120)
        print("ok -- demo_wave.png")
    except Exception as e:  # matplotlib optional
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="run the 2-D demo")
    ap.add_argument("--n", type=int, default=400, help="grid cells per side")
    ap.add_argument("--steps", type=int, default=700)
    args = ap.parse_args()
    if args.demo:
        _demo(args.n, args.steps)
    else:
        ap.print_help()
