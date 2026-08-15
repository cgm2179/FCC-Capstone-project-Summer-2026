#!/usr/bin/env python3
"""
fw_solver.py — pick the indoor 2-D full-wave solver: FDTD (ground truth) or the
ONNX U-Net surrogate (fast), behind one interface.

Two ways to produce the steady-state complex field U(x) on a floor plane:

  * "fdtd" — the real leapfrog full-wave engine (fullwave2d.FullWaveScene). Exact,
    but O(cells x steps): seconds→minutes on a CPU. Always available.
  * "onnx" — fw_unet2d.onnx run under onnxruntime, tiled over the grid exactly
    like fw_infer.tiled_predict but with NO torch dependency. Milliseconds. Only
    available when the exported model + onnxruntime are both present.

Selection rule (matches the product requirement):
  resolve_solver("auto")  -> onnx if available, else FDTD  (default; never fails)
  resolve_solver("fdtd")  -> FDTD                          (always)
  resolve_solver("onnx")  -> onnx, or raise SolverUnavailable if it is not there
                             (an explicit request must NOT silently fall back)

Both solvers return the SAME thing — a complex phasor U over the whole grid — so
the caller renders them identically (envelope |U|, animation Re{U e^{jwt}}).

Before either grid is generated the caller should show the up-front cost with
`estimate_grid_seconds` + `format_time_warning` ("TIME TO CALCULATE GRID ~..."),
because the FDTD grid in particular can take minutes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # engine root, for _bootstrap
import _bootstrap as B          # noqa: F401  (side effect: sys.path)
import dataset_3d as D3
import perf_v3
from fw_dataset import _featurize

HERE = Path(__file__).resolve().parent.parent
C0 = 299_792_458.0

SOLVER_CHOICES = ("auto", "fdtd", "onnx")

# ONNX box must be a multiple of this (the U-Net pools 4x by 2 = 16).
SIZE_MULTIPLE = 16
DEFAULT_BOX = 128
DEFAULT_STRIDE = 96


class SolverUnavailable(RuntimeError):
    """Raised when the ONNX solver is explicitly requested but cannot be used."""


# --------------------------------------------------------------------------- onnx availability
def onnx_paths():
    """(model, contract) paths for the exported surrogate (SIM V1 3D/web/)."""
    web = B.SIM3D / "web"
    return web / "fw_unet2d.onnx", web / "fw_unet2d.json"


def onnx_available() -> tuple[bool, str]:
    """(ok, reason). ok only if the model file AND onnxruntime are both present."""
    model, _ = onnx_paths()
    if not model.exists():
        return False, f"model file not found: {model}"
    try:
        import onnxruntime  # noqa: F401
    except Exception as e:                                   # pragma: no cover
        return False, f"onnxruntime not importable ({e})"
    return True, "ok"


_SESSION = None


def load_session():
    """Cached onnxruntime session, or raise SolverUnavailable with a clear reason."""
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    ok, why = onnx_available()
    if not ok:
        raise SolverUnavailable(
            f"ONNX solver unavailable — {why}. Export it with `python fw_export.py "
            f"--ckpt assets/fw_unet2d.pt`, or use --solver fdtd."
        )
    import onnxruntime as ort
    model, _ = onnx_paths()
    _SESSION = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    return _SESSION


def trained_bands() -> list[str]:
    """Band labels the surrogate was actually trained on (from the contract)."""
    _, contract = onnx_paths()
    try:
        return list(json.loads(contract.read_text()).get("bands_mhz") or [])
    except Exception:
        return []


def resolve_solver(requested: str) -> tuple[str, str]:
    """Map {auto,fdtd,onnx} -> a concrete solver, honouring the selection rule.

    Returns (solver, notice). `notice` is a human line describing what happened
    (why a fallback occurred, etc.) or "".
    Raises SolverUnavailable if `onnx` is explicitly requested but not available.
    """
    requested = (requested or "auto").lower()
    if requested not in SOLVER_CHOICES:
        raise ValueError(f"solver must be one of {SOLVER_CHOICES}, got {requested!r}")

    if requested == "fdtd":
        return "fdtd", ""

    if requested == "onnx":
        ok, why = onnx_available()
        if not ok:
            # explicit opt-in must fail loudly, never fall back silently
            raise SolverUnavailable(
                f"--solver onnx requested but the ONNX surrogate is unavailable "
                f"({why}). Export it (python fw_export.py) or use --solver fdtd."
            )
        return "onnx", ""

    # auto: prefer the fast surrogate, fall back to FDTD with a notice
    ok, why = onnx_available()
    if ok:
        return "onnx", "auto → ONNX surrogate (fast); pass --solver fdtd to force full-wave."
    return "fdtd", f"auto → ONNX unavailable ({why}); defaulting to the FDTD solver."


# --------------------------------------------------------------------------- onnx inference
def _hann2d(box: int) -> np.ndarray:
    return (np.hanning(box)[:, None] * np.hanning(box)[None, :] + 1e-3).astype(np.float32)


def _tile_starts(n: int, box: int, stride: int) -> list[int]:
    if n <= box:
        return [0]
    xs = list(range(0, n - box + 1, stride)) + [n - box]
    return sorted(set(xs))


def num_tiles(shape, box=DEFAULT_BOX, stride=DEFAULT_STRIDE) -> int:
    H, W = shape
    return len(_tile_starts(H, box, stride)) * len(_tile_starts(W, box, stride))


def _next_mult(n: int, m: int = SIZE_MULTIPLE) -> int:
    return int(np.ceil(n / m) * m)


def _predict_tile(session, in_name, out_name, cls, tx_local, h, ff) -> np.ndarray:
    """Run one tile through ONNX, padding to a multiple of 16 and cropping back."""
    b0, b1 = cls.shape
    p0, p1 = _next_mult(b0), _next_mult(b1)
    if (p0, p1) != (b0, b1):
        pad = np.zeros((p0, p1), cls.dtype)          # pad with air (class 0)
        pad[:b0, :b1] = cls
        cls = pad
    ii, jj = np.mgrid[0:cls.shape[0], 0:cls.shape[1]]
    d_m = np.hypot(ii - tx_local[0], jj - tx_local[1]) * h
    xin = _featurize(cls, tx_local, d_m, ff)[None].astype(np.float32)   # (1,9,p0,p1)
    y = session.run([out_name], {in_name: xin})[0][0]                   # (2,p0,p1)
    return y[:, :b0, :b1]


def onnx_tiled_predict(classes, tx_ij, h, f_mhz, box=DEFAULT_BOX,
                       stride=DEFAULT_STRIDE, session=None):
    """Predict the complex field U(x) over `classes` by Hann-feathered tiling.

    Mirrors fw_infer.tiled_predict (blend on the smooth phase-reduced field, then
    reapply the single global free-space ramp so the stitched field stays
    phase-coherent) but runs under onnxruntime instead of torch.
    """
    session = session or load_session()
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    norm = D3.load_norm(json.loads(B.MANIFEST.read_text()))
    ff = norm.freq_feature(f_mhz)
    k = 2.0 * np.pi / (C0 / (f_mhz * 1e6))
    H, W = classes.shape
    win = _hann2d(box)

    acc = np.zeros((2, H, W), np.float64)
    wsum = np.zeros((H, W), np.float64)
    for x0 in _tile_starts(H, box, stride):
        for y0 in _tile_starts(W, box, stride):
            b0 = min(box, H - x0)
            b1 = min(box, W - y0)
            cls = classes[x0:x0 + b0, y0:y0 + b1]
            p = _predict_tile(session, in_name, out_name, cls,
                              (tx_ij[0] - x0, tx_ij[1] - y0), h, ff)     # (2,b0,b1)
            w = win[:b0, :b1]
            acc[:, x0:x0 + b0, y0:y0 + b1] += p * w
            wsum[x0:x0 + b0, y0:y0 + b1] += w
    Utilde = (acc[0] + 1j * acc[1]) / np.maximum(wsum, 1e-9)
    ii, jj = np.mgrid[0:H, 0:W]
    d_global = np.hypot(ii - tx_ij[0], jj - tx_ij[1]) * h
    return Utilde * np.exp(-1j * k * d_global)


# --------------------------------------------------------------------------- time estimate
def _calibrate_onnx_per_tile(session, box=DEFAULT_BOX, n=3) -> float:
    """Seconds for one `box`x`box` tile inference on this machine (warm run)."""
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    x = np.zeros((1, len(D3.INPUT_CHANNELS), box, box), np.float32)
    session.run([out_name], {in_name: x})                   # warm up
    t0 = time.perf_counter()
    for _ in range(n):
        session.run([out_name], {in_name: x})
    return (time.perf_counter() - t0) / n


def estimate_grid_seconds(solver, classes, *, steps=None, box=DEFAULT_BOX,
                          stride=DEFAULT_STRIDE, session=None):
    """(seconds, detail) up-front cost of generating the field grid.

    FDTD cost is cells x steps cell-updates at a CPU-calibrated rate; ONNX cost is
    n_tiles x per-tile inference time, both measured on THIS machine.
    """
    cells = int(classes.size)
    if solver == "fdtd":
        if steps is None:
            raise ValueError("FDTD estimate needs `steps`")
        mcps = perf_v3.calibrate_throughput()
        secs = perf_v3.estimate_seconds(cells, steps, mcps)
        return secs, (f"{cells/1e6:.2f} M cells × {steps} steps "
                      f"@ {mcps:.0f} Mcell-updates/s (FDTD)")
    # onnx
    session = session or load_session()
    nt = num_tiles(classes.shape, box, stride)
    per = _calibrate_onnx_per_tile(session, box)
    return nt * per, f"{nt} tile(s) × {per*1e3:.0f} ms/tile (ONNX, onnxruntime)"


def format_time_warning(seconds, solver, detail) -> str:
    """The 'TIME TO CALCULATE GRID' warning line shown before generation."""
    return (f"⏱  TIME TO CALCULATE GRID  ~{perf_v3.fmt_eta(seconds)}   "
            f"[solver: {solver.upper()}]\n    {detail}")


# --------------------------------------------------------------------------- phasor -> outputs
def synth_frames(U, f_mhz, record_frames=80):
    """CW steady-state animation from a phasor: u(x,t) = Re{U e^{jwt}} over one
    period. Returns (frames[list float16], times[list s], rms) — the same shape
    the FDTD path produces, so the renderer is solver-agnostic.

    For a CW field u = Re{U e^{jwt}}, |U| is the amplitude, so the RMS envelope is
    |U|/sqrt(2). That is exactly what the FDTD run accumulates, keeping the two
    solvers on one scale.
    """
    nf = max(1, int(record_frames))
    T = 1.0 / (float(f_mhz) * 1e6)
    frames, times = [], []
    for i in range(nf):
        ph = np.exp(1j * 2.0 * np.pi * i / nf)
        frames.append((U * ph).real.astype(np.float16))
        times.append(i / nf * T)
    rms = (np.abs(U) / np.sqrt(2.0)).astype(np.float64)
    return frames, times, rms


if __name__ == "__main__":
    ok, why = onnx_available()
    print(f"onnx available: {ok} ({why})")
    print(f"trained bands : {trained_bands()}")
    for req in SOLVER_CHOICES:
        try:
            s, note = resolve_solver(req)
            print(f"  resolve_solver({req!r:6}) -> {s:5}  {note}")
        except SolverUnavailable as e:
            print(f"  resolve_solver({req!r:6}) -> SolverUnavailable: {e}")
