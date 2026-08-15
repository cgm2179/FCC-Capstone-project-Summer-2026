#!/usr/bin/env python3
"""
fw_solver_v2_3d.py — pick the 3-D full-wave solver: FDTD (ground truth) or the
pure-JAX U-Net surrogate (fast), behind one interface.

3-D analog of `2D/SIM V3/fw_solver.py`. Both return the SAME thing — a complex
phasor U over the whole volume — so callers render them identically (envelope |U|,
animation Re{U·e^{jωt}}).

  resolve_solver("auto") -> surrogate if a checkpoint exists, else FDTD
  resolve_solver("fdtd") -> FDTD always ;  resolve_solver("jax") -> surrogate or raise
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # engine root, for bootstrap
import bootstrap  # noqa: F401
import perf_v2_3d as perf

HERE = Path(__file__).resolve().parent.parent
SOLVER_CHOICES = ("auto", "fdtd", "jax")
DEFAULT_BOX, DEFAULT_STRIDE = 24, 18


class SolverUnavailable(RuntimeError):
    pass


def surrogate_ckpt() -> Path:
    return HERE / "fw_unet3d.pt"


def surrogate_available() -> tuple[bool, str]:
    if not surrogate_ckpt().exists():
        return False, f"checkpoint not found: {surrogate_ckpt().name}"
    try:
        import torch  # noqa: F401
    except Exception as e:
        return False, f"torch not importable ({e})"
    return True, "ok"


def resolve_solver(requested: str) -> tuple[str, str]:
    requested = (requested or "auto").lower()
    if requested not in SOLVER_CHOICES:
        raise ValueError(f"solver must be one of {SOLVER_CHOICES}, got {requested!r}")
    if requested == "fdtd":
        return "fdtd", ""
    if requested == "jax":
        ok, why = surrogate_available()
        if not ok:
            raise SolverUnavailable(f"--solver jax requested but surrogate unavailable ({why}). "
                                    f"Train it (fw_unet3d) or use --solver fdtd.")
        return "jax", ""
    ok, why = surrogate_available()
    return ("jax", "auto → JAX surrogate (fast)") if ok else \
           ("fdtd", f"auto → surrogate unavailable ({why}); using FDTD.")


def solve(classes, h, f_mhz, tx, solver="auto", crossings=1.5, box=DEFAULT_BOX,
          stride=DEFAULT_STRIDE, **kw):
    """Return complex U(x,y,z) from the chosen solver."""
    which, _ = resolve_solver(solver)
    if which == "fdtd":
        import fullwave3d as F3
        return F3.solve3d(classes, h, f_mhz, tx, crossings=crossings,
                          extract_phasor=True, **kw)["phasor"]
    from fw_unet3d import load_model3d, tiled_predict3d
    model = load_model3d(str(surrogate_ckpt()))
    return tiled_predict3d(model, classes, tx, h, f_mhz, box=box, stride=stride)


def estimate_grid_seconds(solver, classes, *, steps=None, mcps=None):
    """(seconds, detail) up-front cost. FDTD = cells×steps/mcps; surrogate ≈ tiles×per-tile."""
    cells = int(classes.size)
    if solver == "fdtd":
        if steps is None:
            raise ValueError("FDTD estimate needs `steps`")
        rate = mcps or perf.calibrate_throughput_jax()["mcps"]
        return perf.estimate_seconds(cells, steps, rate), \
            f"{cells/1e6:.2f} M cells × {steps} steps @ {rate:.0f} Mcps (FDTD-JAX)"
    return None, "surrogate ~ ms/tile (run fw_infer_v1_3d to time)"


def synth_frames(U, f_mhz, record_frames=60):
    """CW animation frames from a phasor: u(x,t)=Re{U·e^{jωt}} over one period."""
    frames, times = [], []
    T = 1.0 / (float(f_mhz) * 1e6)
    for i in range(int(record_frames)):
        ph = np.exp(1j * 2.0 * np.pi * i / record_frames)
        frames.append((U * ph).real.astype(np.float16)); times.append(i / record_frames * T)
    return frames, times, (np.abs(U) / np.sqrt(2.0)).astype(np.float64)


if __name__ == "__main__":
    for req in SOLVER_CHOICES:
        try:
            print(f"resolve_solver({req!r:6}) ->", resolve_solver(req))
        except SolverUnavailable as e:
            print(f"resolve_solver({req!r:6}) -> unavailable: {e}")
