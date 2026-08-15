#!/usr/bin/env python3
"""
run_wave2d.py — SIM V3 driver: real-frequency full-wave on the floor plane.

Pipeline:  plane_extract (slice + resample)  ->  FullWaveScene (real medium +
loss)  ->  time-step  ->  {animation GIF, frame stack .npz, steady-state
field-strength (RMS envelope) map, diagnostics}.

Examples:
  # cheap whole-floor 4G low-band
  python run_wave2d.py --band LTE_B71_617
  # fast smoke test (coarse, few steps) -- proves the pipeline in ~seconds
  python run_wave2d.py --band LTE_B71_617 --quick
  # real-frequency 5G, cropped to a wing (whole floor is ~42 M cells at N=10)
  python run_wave2d.py --band NR_n77_3700 --crop 20 55 8 32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # engine root, for _bootstrap
import _bootstrap  # noqa: F401
import fullwave2d as FW
import fw_solver as SV
import perf_v3
from bands_v3 import get, estimate
from plane_extract import extract_plane
from fullwave2d import FullWaveScene

HERE = Path(__file__).resolve().parent.parent
C0 = 299_792_458.0


def _auto_steps(extent_m, dt, crossings):
    t_cross = max(extent_m) / C0
    return int(round(crossings * t_cross / dt))


def main():
    ap = argparse.ArgumentParser(description="SIM V3 full-wave 2-D floor plane")
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--n-per-wavelength", type=float, default=10.0)
    ap.add_argument("--height-m", type=float, default=None)
    ap.add_argument("--height-frac", type=float, default=0.5)
    ap.add_argument("--crop", type=float, nargs=4, default=None,
                    metavar=("X0", "X1", "Z0", "Z1"), help="metres")
    ap.add_argument("--tx", type=float, nargs=2, default=None, metavar=("X", "Z"))
    ap.add_argument("--source", choices=["cw", "ricker"], default="cw")
    ap.add_argument("--no-loss", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--crossings", type=float, default=2.0,
                    help="domain crossings to simulate when --steps is unset")
    ap.add_argument("--record-frames", type=int, default=80)
    ap.add_argument("--warmup-frac", type=float, default=0.5,
                    help="fraction of the run to skip before the RMS envelope")
    ap.add_argument("--max-cells", type=int, default=6_000_000)
    ap.add_argument("--quick", action="store_true",
                    help="coarse + short: pipeline smoke test in seconds")
    ap.add_argument("--solver", choices=SV.SOLVER_CHOICES, default="auto",
                    help="auto: ONNX surrogate if available else FDTD (default); "
                         "fdtd: force the full-wave engine; onnx: force the "
                         "surrogate (errors if it is not available)")
    ap.add_argument("--onnx-box", type=int, default=SV.DEFAULT_BOX,
                    help="ONNX tile size (multiple of 16)")
    ap.add_argument("--onnx-stride", type=int, default=SV.DEFAULT_STRIDE,
                    help="ONNX tile stride (overlap = box - stride)")
    ap.add_argument("--warn-threshold-s", type=float, default=20.0,
                    help="prompt to confirm when the grid estimate exceeds this")
    ap.add_argument("--yes", action="store_true",
                    help="skip the pre-generation time-estimate confirmation")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.quick:
        args.n_per_wavelength = min(args.n_per_wavelength, 6.0)
        args.max_cells = min(args.max_cells, 1_200_000)
        args.crossings = min(args.crossings, 1.2)

    band = get(args.band)
    out = Path(args.out) if args.out else HERE / "out" / band.label
    out.mkdir(parents=True, exist_ok=True)

    # ---- build the plane -------------------------------------------------
    bud = estimate(band, (78.6, 39.6), args.n_per_wavelength)
    print(f"band {band.label}  f={band.f_mhz:.0f} MHz  lambda={band.wavelength_m*100:.1f} cm"
          f"  budget(full floor)={bud['total_cells']/1e6:.1f} M -> {bud['verdict']}")
    p = extract_plane(band, height_m=args.height_m, height_frac=args.height_frac,
                      n_per_wavelength=args.n_per_wavelength, crop_m=args.crop,
                      tx_m=tuple(args.tx) if args.tx else None,
                      max_cells=args.max_cells)
    print(f"plane {p.classes.shape}  h={p.h_m*100:.2f} cm  "
          f"extent={p.extent_m[0]:.1f}x{p.extent_m[1]:.1f} m  tx={p.tx_idx}  "
          f"height={p.height_m:.2f} m  N_eff={p.n_eff:.1f}")

    # ---- pick the solver: FDTD ground truth vs ONNX surrogate ------------
    try:
        solver, notice = SV.resolve_solver(args.solver)
    except SV.SolverUnavailable as e:
        raise SystemExit(f"error: {e}")
    if notice:
        print(notice)

    if solver == "fdtd":
        sim = FullWaveScene(p.classes, p.h_m, band.f_mhz, p.tx_idx,
                            source=args.source, with_loss=not args.no_loss)
        steps = args.steps or _auto_steps(p.extent_m, sim.dt, args.crossings)
        if args.quick:
            steps = min(steps, 3000)
        sim_dt, cfl = sim.dt, sim.cfl()
        print(f"dt={sim_dt:.3e} s  CFL={cfl:.3f} (limit {1/np.sqrt(2):.3f})  "
              f"steps={steps}  loss={'off' if args.no_loss else 'on'}  "
              f"cells={p.classes.size/1e6:.2f} M")
        est, detail = SV.estimate_grid_seconds("fdtd", p.classes, steps=steps)
    else:  # onnx surrogate
        steps = 0
        sim_dt = 1.0 / (band.f_mhz * 1e6) / max(1, args.record_frames)  # frame dt
        cfl = float("nan")
        est, detail = SV.estimate_grid_seconds(
            "onnx", p.classes, box=args.onnx_box, stride=args.onnx_stride)
        tb = SV.trained_bands()
        if tb and band.label not in tb:
            print(f"note: ONNX surrogate trained on {tb}; {band.label} is "
                  f"extrapolation (freq feature interpolates; envelope may drift).")

    # ---- TIME TO CALCULATE GRID warning (before generating anything) ------
    print(SV.format_time_warning(est, solver, detail))
    if not args.yes and est > args.warn_threshold_s and sys.stdin.isatty():
        try:
            if input("    proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                raise SystemExit("aborted before grid generation.")
        except EOFError:
            pass

    # ---- generate the field grid -----------------------------------------
    if solver == "fdtd":
        prog = perf_v3.ProgressWriter(out / "progress.json", steps, p.classes.size)
        res = FW.simulate(sim, steps, warmup_frac=args.warmup_frac,
                          record_frames=args.record_frames, progress=prog,
                          show_tqdm=True, desc=band.label)
        frames, times, rms = res["frames"], res["times"], res["rms"]
        dt_run, finite = res["dt_run"], res["finite"]
        print(f"ran {steps} steps in {dt_run:.1f} s "
              f"({steps*p.classes.size/1e6/max(dt_run,1e-9):.0f} Mcell-updates/s)  "
              f"finite={finite}")
        if not finite:
            raise FloatingPointError("field blew up — CFL/safety too high")
    else:  # onnx surrogate: one tiled inference -> steady-state phasor U
        import time as _time
        t0 = _time.time()
        U = SV.onnx_tiled_predict(p.classes, p.tx_idx, p.h_m, band.f_mhz,
                                  box=args.onnx_box, stride=args.onnx_stride)
        dt_run = _time.time() - t0
        frames, times, rms = SV.synth_frames(U, band.f_mhz, args.record_frames)
        finite = bool(np.isfinite(U).all())
        print(f"ONNX solve {dt_run*1e3:.0f} ms  grid {p.classes.shape}  "
              f"finite={finite}")
        if not finite:
            raise FloatingPointError("ONNX field non-finite")
    ref = np.percentile(rms[rms > 0], 99.5) if np.any(rms > 0) else 1.0
    field_db = 20.0 * np.log10(np.maximum(rms, ref * 1e-6) / ref)  # 0 dB = strong

    np.savez_compressed(out / "wave_frames.npz",
                        frames=np.array(frames), times=np.array(times),
                        h_m=p.h_m, extent_m=np.array(p.extent_m),
                        tx=np.array(p.tx_idx), dt=sim_dt, classes=p.classes)
    np.save(out / "field_rms.npy", rms.astype(np.float32))
    np.save(out / "field_db.npy", field_db.astype(np.float32))
    meta = dict(band=band.label, f_mhz=band.f_mhz, h_m=p.h_m, solver=solver,
                extent_m=list(p.extent_m), grid=list(p.classes.shape),
                tx_idx=list(p.tx_idx), height_m=p.height_m, n_eff=p.n_eff,
                dt=sim_dt, cfl=cfl, steps=steps, loss=not args.no_loss,
                run_seconds=dt_run, finite=finite, plane_meta=p.meta)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))

    _render(out, frames, times, field_db, p, band, do_gif=not args.no_gif)
    print(f"outputs -> {out}")


def _render(out, frames, times, field_db, plane, band, do_gif=True):
    """Field animation (GIF) + a still + the RMS field-strength map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    solids = (plane.classes > 0)
    ex = [0, plane.extent_m[0], 0, plane.extent_m[1]]  # x (X) horiz, y (Z) vert
    late = np.abs(frames[len(frames) // 2].astype(np.float32))
    m = float(np.percentile(late, 99.0)) or 1.0

    # arcsinh (signed-log) compression: the near-source amplitude is orders of
    # magnitude above the far-field ripple, so a linear scale hides the wave.
    # comp() keeps the sign, compresses the peak, and lifts the far field.
    eps = max(m / 300.0, 1e-12)
    vlim = float(np.arcsinh(m / eps))

    def comp(a):
        return np.arcsinh(a.astype(np.float32) / eps)

    # still of the last field
    fig, axf = plt.subplots(figsize=(9, 4.6))
    axf.imshow(comp(frames[-1]).T, cmap="RdBu", vmin=-vlim, vmax=vlim,
               origin="lower", extent=ex, aspect="equal")
    axf.contour(solids.T, levels=[0.5], colors="k", linewidths=0.35, extent=ex)
    axf.plot(plane.tx_idx[0] * plane.h_m, plane.tx_idx[1] * plane.h_m,
             "k*", ms=9)
    axf.set_title(f"SIM V3 full-wave  {band.label}  ({band.f_mhz:.0f} MHz)  "
                  f"— last field")
    axf.set_xlabel("x (m)"); axf.set_ylabel("z (m)")
    fig.tight_layout(); fig.savefig(out / "field_last.png", dpi=120); plt.close(fig)

    # RMS field-strength (envelope) map
    fig, axe = plt.subplots(figsize=(9, 4.6))
    im = axe.imshow(field_db.T, cmap="viridis", vmin=-40, vmax=0,
                    origin="lower", extent=ex, aspect="equal")
    axe.contour(solids.T, levels=[0.5], colors="w", linewidths=0.3, extent=ex)
    axe.plot(plane.tx_idx[0] * plane.h_m, plane.tx_idx[1] * plane.h_m, "r*", ms=9)
    axe.set_title(f"SIM V3 RMS field strength (dB rel.)  {band.label}")
    axe.set_xlabel("x (m)"); axe.set_ylabel("z (m)")
    fig.colorbar(im, ax=axe, shrink=0.8, label="dB")
    fig.tight_layout(); fig.savefig(out / "field_rms.png", dpi=120); plt.close(fig)

    if not do_gif:
        return
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
        fig, ax = plt.subplots(figsize=(9, 4.6))
        img = ax.imshow(comp(frames[0]).T, cmap="RdBu",
                        vmin=-vlim, vmax=vlim, origin="lower", extent=ex, aspect="equal")
        ax.contour(solids.T, levels=[0.5], colors="k", linewidths=0.35, extent=ex)
        ax.plot(plane.tx_idx[0] * plane.h_m, plane.tx_idx[1] * plane.h_m, "k*", ms=8)
        ttl = ax.set_title("")
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")

        def _upd(i):
            img.set_data(comp(frames[i]).T)
            ttl.set_text(f"{band.label}  t = {times[i]*1e9:6.1f} ns")
            return img, ttl

        anim = FuncAnimation(fig, _upd, frames=len(frames), blit=False)
        anim.save(out / "wave.gif", writer=PillowWriter(fps=15))
        plt.close(fig)
        print("  wrote wave.gif")
    except Exception as e:
        print(f"  (gif skipped: {e})")


if __name__ == "__main__":
    main()
