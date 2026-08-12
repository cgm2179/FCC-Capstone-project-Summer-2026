#!/usr/bin/env python3
"""
fullwave2d.py — SIM V3 full-wave engine: FullWaveScene.

This is a THIN subclass of the sandbox WaveSim (Wave Behavior/Wave Generation/
Time_Domain_Physics.py). It changes exactly two things vs the toy demo:

  1. the medium is the REAL, frequency-dependent ITU speed field
     physics_3d.speed_field(classes, f_mhz)  -- not the crude hard-coded EPS_R;
  2. it adds a per-cell ABSORPTION term (the one genuinely new equation), so
     lossy materials attenuate the wave instead of only reflecting it.

Loss model. A conductive medium turns the wave equation into the damped form

    d^2u/dt^2 + 2*gamma(x) du/dt = c(x)^2 * laplacian(u)

with a damping rate gamma = pi * f * tan(delta) built from the imaginary part of
the ITU-R P.2040 complex permittivity (physics_v2.permittivity). The leapfrog
update becomes

    u^{n+1} = [ 2u^n - (1 - a) u^{n-1} + (c dt)^2 lap ] / (1 + a),   a = gamma*dt

which reduces EXACTLY to the sandbox update when a = 0 (air / lossless), so the
lossless path is unchanged and the demo still validates.

Name is FullWaveScene, deliberately NOT SceneV3 -- that class is the existing
3-D eikonal production engine (engine_3d.py); reusing the name would collide.
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (side effect: sys.path)
import physics_3d as P3
import physics_v2 as PV
import Spatial_Physics as SP
import Time_Domain_Physics as TD
from Time_Domain_Physics import WaveSim


def damping_by_class(materials, f_mhz, barrier_classes=(3,)) -> np.ndarray:
    """Per-class damping rate gamma [1/s] = pi f tan(delta) from Im(eps_r).

    Air / vacuum and the rigid barrier classes get 0 (barriers are perfect
    reflectors held at u = 0, so their bulk loss never applies). `per_metre`
    dilute classes (open-plan furniture clutter) are scaled by fill_fraction so
    they don't absorb like solid wood.
    """
    ids = [m["id"] for m in materials]
    g = np.zeros(max(ids) + 1, np.float64)
    for m in materials:
        p = m.get("p2040")
        if p in (None, "vacuum") or m["id"] in barrier_classes:
            continue
        eps = np.atleast_1d(PV.permittivity(p, f_mhz))
        er = float(np.real(eps)[0])
        ei = float(abs(np.imag(eps)[0]))
        tand = ei / max(er, 1e-6)
        gamma = np.pi * (f_mhz * 1e6) * tand
        if m.get("per_metre") and m.get("fill_fraction"):
            gamma *= float(m["fill_fraction"])
        g[m["id"]] = gamma
    return g


class FullWaveScene(WaveSim):
    """Leapfrog scalar-wave field on the real floor plane, with material loss."""

    def __init__(self, classes, h_m, f_mhz, source_xy, source="cw",
                 with_loss=True, materials=None, barrier_classes=(3,),
                 sponge_width=None, safety=0.5):
        materials = materials if materials is not None else PV.MATERIALS7
        # real ITU medium + rigid mask, straight from the production adapter
        speed, rigid = P3.speed_field(
            classes, f_mhz, materials=materials, barrier_classes=barrier_classes)
        # cw()/ricker() take the REAL frequency in Hz because c is real m/s and
        # dt (courant_dt) is real seconds.
        f0 = float(f_mhz) * 1e6
        if sponge_width is None:
            sponge_width = max(20, int(0.06 * min(classes.shape)))
        super().__init__(speed, h_m, source_xy, f0, source=source,
                         rigid=rigid, sponge_width=sponge_width, safety=safety)

        self.f_mhz = float(f_mhz)
        self.classes = classes
        self.with_loss = bool(with_loss)
        if with_loss:
            g = damping_by_class(materials, f_mhz, barrier_classes)
            self.a = (g[classes] * self.dt).astype(np.float64)
        else:
            self.a = np.zeros_like(self.c)
        self._inv1pa = 1.0 / (1.0 + self.a)
        self._1ma = 1.0 - self.a

    def step(self):
        t = self.n * self.dt
        lap = SP.laplacian(self.u, self.inv_h2)
        # damped leapfrog; identical to the sandbox when a == 0
        u_next = (2.0 * self.u - self._1ma * self.u_prev
                  + self.cdt2 * lap) * self._inv1pa
        s = TD.ricker(t, self.f0) if self.source == "ricker" else TD.cw(t, self.f0)
        u_next[self.src_idx] += s          # soft point source (air cell, a=0)
        u_next[self.rigid] = 0.0           # perfect reflectors
        u_next *= self.damp                # absorbing boundary (sponge)
        self.u_prev = self.u * self.damp
        self.u = u_next
        self.n += 1
        return self.u

    def cfl(self) -> float:
        """Courant number actually in use (must stay < 1/sqrt(ndim))."""
        return float(self.c.max() * self.dt / self.h)


def simulate(sim, steps, warmup_frac=0.5, record_frames=0, progress=None,
             show_tqdm=False, desc="fdtd", extract_phasor=False, phasor_periods=2):
    """Step `sim` for `steps`, recording ~`record_frames` frames and the
    post-warmup RMS envelope. Shared by run_wave2d.py and hybrid_field.py.

    progress        optional perf_v3.ProgressWriter (updated each step, throttled).
    show_tqdm       wrap the loop in a tqdm bar.
    extract_phasor  also accumulate the steady-state complex phasor U(x) via an
                    on-the-fly single-frequency DFT over the LAST `phasor_periods`
                    CW periods: U = (2/N) Σ u(x,t) e^{-jωt}. For a CW field
                    u = Re{U e^{jωt}}, |U| is the amplitude so |U|/√2 = the RMS
                    envelope (a built-in sanity check). This is the ML training
                    target (real/imag). Returns U in `phasor`.
    Returns dict(frames, times, rms, dt_run, finite, n_acc[, phasor]).
    """
    import time as _time

    rec_every = max(1, steps // max(1, record_frames)) if record_frames else 0
    warmup = int(warmup_frac * steps)
    frames, times = [], []
    sum_u2 = np.zeros(sim.c.shape, np.float64)
    n_acc = 0

    # phasor DFT window = the last `phasor_periods` full CW periods
    acc = None
    n_win = 0
    if extract_phasor:
        omega = 2.0 * np.pi * sim.f0
        period_steps = max(1, int(round((1.0 / sim.f0) / sim.dt)))
        win_start = max(warmup, steps - phasor_periods * period_steps)
        acc = np.zeros(sim.c.shape, np.complex128)

    it = range(steps)
    if show_tqdm:
        try:
            from tqdm import tqdm
            it = tqdm(it, desc=desc, unit="step")
        except Exception:
            pass

    t0 = _time.time()
    for k in it:
        u = sim.step()
        if k >= warmup:
            sum_u2 += u * u
            n_acc += 1
        if extract_phasor and k >= win_start:
            t = (k + 1) * sim.dt                      # u is the field at (k+1)dt
            acc += u * np.exp(-1j * omega * t)
            n_win += 1
        if rec_every and (k % rec_every == 0):
            frames.append(u.astype(np.float16).copy())
            times.append(k * sim.dt)
        if progress is not None:
            progress.update(k)
    dt_run = _time.time() - t0
    if progress is not None:
        progress.finish()
    rms = np.sqrt(sum_u2 / max(n_acc, 1))
    out = dict(frames=frames, times=times, rms=rms, dt_run=dt_run,
               finite=bool(np.isfinite(sim.u).all()), n_acc=n_acc)
    if extract_phasor:
        out["phasor"] = (2.0 / max(n_win, 1)) * acc
    return out
