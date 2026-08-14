#!/usr/bin/env python3
"""
fw_fdtd_jax.py — JAX/XLA-accelerated stepper for the SIM V3 scalar full-wave FDTD.

The high-leverage accelerator for M2 data-gen. The FDTD teacher is the pipeline's
bottleneck (~hours per whole-floor solve on NumPy/CPU); every training sample is one
solve. This module JIT-compiles the *time loop* with XLA so the entire run fuses into
one compiled kernel that runs on CPU / CUDA / TPU from the SAME code (JAX picks the
backend). CuPy was the alternative but it is NVIDIA-only — useless on this Apple Mac
and not device-portable; JAX mirrors NumPy (np.roll → jnp.roll) and, like PyTorch, is
device-agnostic + differentiable, so it also lines up with the surrogate stack.

PHYSICS PARITY BY CONSTRUCTION. We do NOT reimplement the medium. We reuse
`fullwave2d.FullWaveScene` to build c(x), the per-cell loss a=γ·dt, the sponge `damp`,
the rigid mask, and dt on the host (NumPy) exactly as the ground-truth engine does,
then only move the leapfrog stepping to JAX. The update is identical to
FullWaveScene.step():

    u^{n+1} = [ 2u^n - (1-a) u^{n-1} + (c dt)^2 ∇²u ] / (1+a)

and reduces to the lossless sandbox update when a = 0. The phasor DFT window and RMS
envelope accumulation match fw_solver/fullwave2d.simulate() step-for-step, so
`simulate_jax` is a drop-in for `fullwave2d.simulate` returning the same U(x).

  # as a library (dimension-agnostic: 2-D or 3-D FullWaveScene)
  from fullwave2d import FullWaveScene
  from fw_fdtd_jax import simulate_jax
  sim = FullWaveScene(classes, h_m, f_mhz, tx, source="cw")
  res = simulate_jax(sim, steps, extract_phasor=True)   # res["phasor"] = U

  # benchmark vs the NumPy engine on this machine (prints Mcell-updates/s + speedup)
  python fw_fdtd_jax.py --nx 100 --ny 40 --nz 100 --steps 600
  python fw_fdtd_jax.py --2d --nx 800 --ny 800 --steps 1500
"""
from __future__ import annotations

try:  # SIM V2: put the shared SIM V3 engine on sys.path (harmless if absent)
    import bootstrap  # noqa: F401
except Exception:
    pass

import argparse
import time

import numpy as np

C0 = 299_792_458.0


# --------------------------------------------------------------------------- jax setup
def _import_jax(x64: bool):
    """Import jax with the requested precision. x64 must be set before jnp use."""
    import jax
    if x64:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import lax
    return jax, jnp, lax


def _laplacian(jnp, u, inv_h2):
    """n-D 2nd-order Laplacian via jnp.roll — mirrors Spatial_Physics.laplacian."""
    lap = jnp.zeros_like(u)
    nd = u.ndim
    for ax in range(nd):
        lap = lap + jnp.roll(u, 1, axis=ax) + jnp.roll(u, -1, axis=ax)
    lap = lap - 2.0 * nd * u
    return lap * inv_h2


# --------------------------------------------------------------------------- core run
def simulate_jax(sim, steps, warmup_frac=0.5, extract_phasor=False,
                 phasor_periods=2, dtype="float32", return_field=False, repeats=0):
    """Step a (NumPy-built) FullWaveScene `sim` for `steps` under XLA.

    Returns dict(rms, dt_run, dt_cold, mcps, finite, backend[, phasor][, field]) —
    matching fullwave2d.simulate()'s keys for the pieces the surrogate pipeline
    consumes. `dtype="float64"` (x64) is the parity path vs the NumPy engine;
    "float32" is the fast GPU-friendly path used for data-gen. `repeats>0` also times
    warm (post-compile) runs and reports the fastest as dt_run — the representative
    number for data-gen, where the one-time XLA compile amortizes over many solves.
    """
    x64 = (dtype == "float64")
    jax, jnp, lax = _import_jax(x64=x64)
    npf = np.float64 if x64 else np.float32

    # --- host-side constants straight off the ground-truth scene (identical physics) ---
    inv_h2 = float(sim.inv_h2)
    cdt2 = jnp.asarray(sim.cdt2, npf)
    inv1pa = jnp.asarray(getattr(sim, "_inv1pa", np.ones_like(sim.c)), npf)
    one_ma = jnp.asarray(getattr(sim, "_1ma", np.ones_like(sim.c)), npf)
    damp = jnp.asarray(sim.damp, npf)
    rigid = jnp.asarray(sim.rigid, bool)
    dt = float(sim.dt)
    f0 = float(sim.f0)
    omega = 2.0 * np.pi * f0
    src_idx = tuple(int(i) for i in sim.src_idx)
    is_cw = (getattr(sim, "source", "cw") != "ricker")
    cdtype = jnp.complex128 if x64 else jnp.complex64

    warmup = int(warmup_frac * steps)
    period_steps = max(1, int(round((1.0 / f0) / dt)))
    win_start = int(max(warmup, steps - phasor_periods * period_steps)) if extract_phasor else steps + 1

    shape = tuple(int(s) for s in sim.c.shape)
    zeros = jnp.zeros(shape, npf)

    def _source(n):
        t = n.astype(npf) * dt
        if is_cw:
            return jnp.sin(2.0 * jnp.pi * f0 * t)
        a = (jnp.pi * f0 * (t - 1.0 / f0)) ** 2
        return (1.0 - 2.0 * a) * jnp.exp(-a)

    def body(n, carry):
        u, u_prev, sum_u2, n_acc, pacc, n_win = carry
        lap = _laplacian(jnp, u, inv_h2)
        u_next = (2.0 * u - one_ma * u_prev + cdt2 * lap) * inv1pa
        u_next = u_next.at[src_idx].add(_source(n).astype(npf))
        u_next = jnp.where(rigid, jnp.asarray(0.0, npf), u_next)
        u_next = u_next * damp
        u_prev_new = u * damp

        after_warm = (n >= warmup).astype(npf)
        sum_u2 = sum_u2 + after_warm * (u_next * u_next)
        n_acc = n_acc + (n >= warmup).astype(jnp.int32)

        in_win = (n >= win_start).astype(npf)
        t1 = (n + 1).astype(npf) * dt
        phase = jnp.exp((-1j * omega * t1).astype(cdtype))
        pacc = pacc + (in_win * u_next).astype(cdtype) * phase
        n_win = n_win + (n >= win_start).astype(jnp.int32)
        return (u_next, u_prev_new, sum_u2, n_acc, pacc, n_win)

    carry0 = (zeros, zeros, zeros, jnp.asarray(0, jnp.int32),
              jnp.zeros(shape, cdtype), jnp.asarray(0, jnp.int32))

    @jax.jit
    def run():
        u, u_prev, sum_u2, n_acc, pacc, n_win = lax.fori_loop(0, steps, body, carry0)
        rms = jnp.sqrt(sum_u2 / jnp.maximum(n_acc, 1).astype(npf))
        U = (2.0 / jnp.maximum(n_win, 1).astype(npf)) * pacc
        return u, rms, U, n_acc, n_win

    def _once():
        u, rms, U, n_acc, n_win = run()
        u.block_until_ready(); rms.block_until_ready(); U.block_until_ready()
        return u, rms, U

    t0 = time.perf_counter()
    u, rms, U = _once()                      # cold: includes one-time XLA compile
    dt_cold = time.perf_counter() - t0
    dt_run = dt_cold
    for _ in range(repeats):                 # warm: steady-state stepping cost
        t0 = time.perf_counter()
        u, rms, U = _once()
        dt_run = min(dt_run, time.perf_counter() - t0)

    rms = np.asarray(rms)
    out = dict(rms=rms, dt_run=dt_run, dt_cold=dt_cold,
               finite=bool(np.isfinite(rms).all()), backend=jax.default_backend(),
               mcps=(np.prod(shape) * steps) / max(dt_run, 1e-12) / 1e6)
    if extract_phasor:
        out["phasor"] = np.asarray(U)
    if return_field:
        out["field"] = np.asarray(u)
    return out


# --------------------------------------------------------------------------- benchmark
def _synthetic_scene(shape, f_mhz, npw, twod=False):
    """A reproducible material scene exercising loss + rigid + refraction + sponge:
    drywall (1), concrete (2), a rigid barrier with a slit (3), furniture (4)."""
    from fullwave2d import FullWaveScene
    from bands_v3 import get
    band = get("LTE_B71_617"); band_f = f_mhz or band.f_mhz
    h = (C0 / (band_f * 1e6)) / npw
    if twod:
        nx, nz = shape
        M = np.zeros((nx, nz), np.int8)
        M[nx // 3: nx // 3 + 3, :] = 2                      # concrete wall
        M[2 * nx // 3: 2 * nx // 3 + 2, :] = 3              # rigid barrier
        M[2 * nx // 3: 2 * nx // 3 + 2, nz // 2 - 3: nz // 2 + 3] = 0   # slit
        M[nx // 2 - 6: nx // 2 + 6, nz // 4 - 6: nz // 4 + 6] = 4       # furniture
        tx = (nx // 6, nz // 2)
    else:
        nx, ny, nz = shape
        M = np.zeros((nx, ny, nz), np.int8)
        M[nx // 3: nx // 3 + 3, :, :] = 2
        M[2 * nx // 3: 2 * nx // 3 + 2, :, :] = 3
        M[2 * nx // 3: 2 * nx // 3 + 2, :, nz // 2 - 3: nz // 2 + 3] = 0
        M[nx // 2 - 5: nx // 2 + 5, :, nz // 4 - 5: nz // 4 + 5] = 4
        tx = (nx // 6, ny // 2, nz // 2)
    sim = FullWaveScene(M, h, band_f, tx, source="cw")
    return sim, band_f, h


def benchmark(shape, steps, twod=False, f_mhz=None, npw=8.0, phasor=True):
    """Run the NumPy engine and the JAX engine on the same scene; report parity +
    throughput. NumPy is float64; JAX is timed at float32 (fast) and float64 (parity)."""
    import fullwave2d as FW

    sim_np, band_f, h = _synthetic_scene(shape, f_mhz, npw, twod)
    cells = int(np.prod(sim_np.c.shape))
    print(f"scene {sim_np.c.shape}  ({cells/1e6:.3f} M cells)  h={h*100:.2f} cm  "
          f"f={band_f:.0f} MHz  dt={sim_np.dt:.3e}s  cfl={sim_np.cfl():.3f}  steps={steps}")
    print("-" * 78)

    # ---- NumPy ground truth (float64) ----
    t0 = time.perf_counter()
    res_np = FW.simulate(sim_np, steps, warmup_frac=0.5, extract_phasor=phasor,
                         show_tqdm=False)
    t_np = time.perf_counter() - t0
    mcps_np = cells * steps / t_np / 1e6
    print(f"NumPy  (float64)   {t_np:8.3f} s   {mcps_np:7.1f} Mcell-updates/s   "
          f"finite={res_np['finite']}")

    def _rebuild():
        s, _, _ = _synthetic_scene(shape, f_mhz, npw, twod)
        return s

    # ---- JAX float64 (parity path) ----
    r64 = simulate_jax(_rebuild(), steps, warmup_frac=0.5, extract_phasor=phasor,
                       dtype="float64", repeats=3)
    mcps64 = cells * steps / r64["dt_run"] / 1e6
    print(f"JAX    (float64)   {r64['dt_run']:8.3f} s   {mcps64:7.1f} Mcell-updates/s   "
          f"[{r64['backend']}]  speedup ×{t_np / r64['dt_run']:.2f}   "
          f"(compile {r64['dt_cold'] - r64['dt_run']:.2f}s, one-time)")

    # ---- JAX float32 (fast data-gen path) ----
    r32 = simulate_jax(_rebuild(), steps, warmup_frac=0.5, extract_phasor=phasor,
                       dtype="float32", repeats=3)
    mcps32 = cells * steps / r32["dt_run"] / 1e6
    print(f"JAX    (float32)   {r32['dt_run']:8.3f} s   {mcps32:7.1f} Mcell-updates/s   "
          f"[{r32['backend']}]  speedup ×{t_np / r32['dt_run']:.2f}   "
          f"(compile {r32['dt_cold'] - r32['dt_run']:.2f}s, one-time)")

    # ---- parity ----
    print("-" * 78)
    rms_np = res_np["rms"]
    def _rel(a, b):
        d = np.abs(a - b); s = np.abs(b).max() + 1e-30
        return d.max() / s, float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    e64, c64 = _rel(r64["rms"], rms_np)
    e32, c32 = _rel(r32["rms"], rms_np)
    print(f"RMS parity vs NumPy:  float64  max-rel-err={e64:.2e}  corr={c64:.6f}")
    print(f"                      float32  max-rel-err={e32:.2e}  corr={c32:.6f}")
    if phasor:
        def _coh(a, b):
            a, b = a.ravel(), b.ravel()
            return float(np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        print(f"phasor U coherence:   float64={_coh(r64['phasor'], res_np['phasor']):.6f}  "
              f"float32={_coh(r32['phasor'], res_np['phasor']):.6f}")
    return dict(t_np=t_np, mcps_np=mcps_np, mcps_jax64=mcps64, mcps_jax32=mcps32,
                speedup32=t_np / r32["dt_run"], corr32=c32)


def main():
    ap = argparse.ArgumentParser(description="JAX-accelerated FDTD benchmark vs NumPy")
    ap.add_argument("--nx", type=int, default=100)
    ap.add_argument("--ny", type=int, default=40)
    ap.add_argument("--nz", type=int, default=100)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--2d", dest="twod", action="store_true", help="2-D floor-plane bench")
    ap.add_argument("--f-mhz", type=float, default=None)
    ap.add_argument("--npw", type=float, default=8.0, help="cells per wavelength")
    a = ap.parse_args()
    shape = (a.nx, a.nz) if a.twod else (a.nx, a.ny, a.nz)
    benchmark(shape, a.steps, twod=a.twod, f_mhz=a.f_mhz, npw=a.npw)


if __name__ == "__main__":
    main()
