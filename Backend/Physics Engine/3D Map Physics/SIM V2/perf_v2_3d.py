#!/usr/bin/env python3
"""
perf_v2_3d.py — runtime estimate + progress feed for SIM V2 3-D solves.

Full-wave FDTD cost is deterministic: cells × steps cell-updates. A short
calibration burst gives a Mcell-updates/s rate, and the up-front estimate is just
`cells·steps/rate`. Ports `2D/SIM V3/perf_v3.py` and adds a **JAX** calibrator
(`calibrate_throughput_jax`) — the SIM V2 teacher runs on `simulate_jax`, whose
device throughput (CPU/GPU) is what should drive ETAs, not the NumPy stencil.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import bootstrap  # noqa: F401  (side effect: sys.path)
import Spatial_Physics as SP


def calibrate_throughput(cal_side=1024, n_iters=10) -> float:
    """Mcell-updates/s for the NumPy leapfrog stencil (2-D dummy grid)."""
    rng = np.random.default_rng(0)
    u = rng.standard_normal((cal_side, cal_side))
    up = u.copy()
    c2 = np.full((cal_side, cal_side), 0.25)
    SP.laplacian(u, 1.0)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        lap = SP.laplacian(u, 1.0)
        un = 2.0 * u - up + c2 * lap
        up, u = u, un
    dt = time.perf_counter() - t0
    return (cal_side * cal_side * n_iters) / dt / 1e6


def calibrate_throughput_jax(shape=(96, 40, 96), steps=200, dtype="float32") -> dict:
    """Warm Mcell-updates/s for the JAX FDTD stepper on this device (the real SIM V2
    teacher throughput). Builds a tiny synthetic scene and times simulate_jax warm."""
    from fullwave2d import FullWaveScene
    from fw_fdtd_jax import simulate_jax
    from bands_v3 import get
    band = get("LTE_B71_617")
    h = band.wavelength_m / 8.0
    M = np.zeros(shape, np.int8)
    M[shape[0] // 3, :, :] = 2  # a wall so loss/medium are exercised
    tx = tuple(s // 2 for s in shape)
    sim = FullWaveScene(M, h, band.f_mhz, tx, source="cw")
    res = simulate_jax(sim, steps, dtype=dtype, repeats=3)
    return dict(mcps=res["mcps"], backend=res["backend"], dt_run=res["dt_run"],
                dt_cold=res["dt_cold"])


def estimate_seconds(cells, steps, mcps) -> float:
    return cells * steps / (max(mcps, 1e-6) * 1e6)


def fmt_eta(s) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


class ProgressWriter:
    """Writes {fraction, eta_s, mcells_per_s, step, steps, stage, done} JSON that a
    static frontend polls. Same contract as perf_v3.ProgressWriter."""

    def __init__(self, path, steps, cells, stage="fdtd3d", every=None):
        self.path = Path(path)
        self.steps = int(steps)
        self.cells = int(cells)
        self.stage = stage
        self.every = every or max(1, self.steps // 200)
        self.t0 = time.perf_counter()
        self._write(0.0, None, 0)

    def update(self, k):
        if (k % self.every) and k != self.steps - 1:
            return
        frac = (k + 1) / self.steps
        el = time.perf_counter() - self.t0
        eta = el * (1 - frac) / max(frac, 1e-9)
        mcps = (self.cells * (k + 1)) / max(el, 1e-9) / 1e6
        self._write(frac, eta, k + 1, mcps)

    def finish(self):
        el = time.perf_counter() - self.t0
        mcps = (self.cells * self.steps) / max(el, 1e-9) / 1e6
        self._write(1.0, 0.0, self.steps, mcps, done=True)

    def _write(self, frac, eta, step, mcps=None, done=False):
        self.path.write_text(json.dumps(dict(
            fraction=round(float(frac), 4),
            eta_s=None if eta is None else round(float(eta), 1),
            mcells_per_s=None if mcps is None else round(float(mcps), 1),
            step=int(step), steps=self.steps, stage=self.stage, done=bool(done))))


if __name__ == "__main__":
    print(f"NumPy stencil: {calibrate_throughput():.0f} Mcell-updates/s")
    j = calibrate_throughput_jax()
    print(f"JAX [{j['backend']}]: {j['mcps']:.0f} Mcell-updates/s "
          f"(compile {j['dt_cold']-j['dt_run']:.2f}s)")
