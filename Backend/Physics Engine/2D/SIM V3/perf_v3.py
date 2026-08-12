#!/usr/bin/env python3
"""
perf_v3.py — CPU runtime estimate + progress feed for SIM V3 runs.

Full-wave FDTD cost is deterministic: cells x steps cell-updates. So a short
calibration burst on THIS CPU gives a Mcell-updates/s rate, and the up-front
estimate is just (cells x steps / rate). During the run we emit a `progress.json`
(fraction / ETA / rate) that the static frontend polls to drive the existing
`#simProgress` bar — no backend needed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
import Spatial_Physics as SP


def calibrate_throughput(cal_side=1024, n_iters=10) -> float:
    """Mcell-updates/s for the leapfrog stencil on this CPU (dummy grid, so it
    does not disturb any real sim). Uses the SAME np.roll Laplacian the engine
    runs, at a size large enough to be memory-bound like a real grid, so the
    rate is representative (small cache-resident grids over-report)."""
    rng = np.random.default_rng(0)
    u = rng.standard_normal((cal_side, cal_side))
    up = u.copy()
    c2 = np.full((cal_side, cal_side), 0.25)
    SP.laplacian(u, 1.0)                      # warm caches / JIT-free but fair
    t0 = time.perf_counter()
    for _ in range(n_iters):
        lap = SP.laplacian(u, 1.0)
        un = 2.0 * u - up + c2 * lap
        up, u = u, un
    dt = time.perf_counter() - t0
    return (cal_side * cal_side * n_iters) / dt / 1e6


def estimate_seconds(cells, steps, mcps) -> float:
    return cells * steps / (max(mcps, 1e-6) * 1e6)


def fmt_eta(s) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


class ProgressWriter:
    """Writes {fraction, eta_s, mcells_per_s, step, steps, stage, done} to a JSON
    file every `every` steps. Cheap enough to call each step (throttled inside)."""

    def __init__(self, path, steps, cells, stage="fdtd", every=None):
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
    r = calibrate_throughput()
    print(f"this CPU: {r:.0f} Mcell-updates/s")
    for cells, steps in [(1.3e6, 13700), (2.5e6, 9200), (0.47e6, 3000)]:
        print(f"  {cells/1e6:.2f} M cells x {steps} steps -> "
              f"~{fmt_eta(estimate_seconds(cells, steps, r))}")
