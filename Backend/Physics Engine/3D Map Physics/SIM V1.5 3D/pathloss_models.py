#!/usr/bin/env python3
"""
pathloss_models.py — the selectable Layer-1 spreading models (analytic PL(d), dB).

These give the distance / frequency dependence of path loss; the physics engine
(engine.py, reusing SIM V1 3D's SceneV3) adds the geometry — the per-crossing material
loss. All return the *mean* PL in dB (no shadow term; shadow enters as the fade margin,
link_budget.py). Every model is vectorized over `d_m`.

    close_in            PL(d0) + 10 n log10(d/d0)                      (indoor default)
    floating_intercept  alpha + 10 beta log10(d) + 10 gamma log10(fGHz)  (ABG, calibration)
    dual_slope          n near d0..dbp, ple_far past dbp               (outdoor default)
"""
from __future__ import annotations

import numpy as np

from pathloss_config import PathLossConfig


def close_in(d_m, f_mhz: float, cfg: PathLossConfig):
    """Close-in reference model. PLE `n`, reference `PL(d0)` at `d0`."""
    d = np.maximum(np.asarray(d_m, np.float64), cfg.d0_m)
    return (cfg.pl_ref_db(f_mhz) + 10.0 * cfg.ple_n * np.log10(d / cfg.d0_m)).astype(np.float32)


def floating_intercept(d_m, f_mhz: float, cfg: PathLossConfig):
    """Alpha-Beta-Gamma model. Free intercept `alpha`, slope `beta`, freq term `gamma`."""
    d = np.maximum(np.asarray(d_m, np.float64), cfg.d0_m)
    return (cfg.fi_alpha_db + 10.0 * cfg.fi_beta * np.log10(d)
            + 10.0 * cfg.fi_gamma * np.log10(max(f_mhz, 1e-9) / 1000.0)).astype(np.float32)


def dual_slope(d_m, f_mhz: float, cfg: PathLossConfig):
    """Dual-slope with a breakpoint. Slope `ple_n` up to `dbp`, `ple_far` beyond it.

    Continuous at dbp: the far segment starts from the near PL evaluated at dbp, so
    there is no discontinuity where the exponent changes.
    """
    d = np.maximum(np.asarray(d_m, np.float64), cfg.d0_m)
    dbp = max(cfg.breakpoint(f_mhz), cfg.d0_m)
    ref = cfg.pl_ref_db(f_mhz)
    near = ref + 10.0 * cfg.ple_n * np.log10(d / cfg.d0_m)
    pl_at_bp = ref + 10.0 * cfg.ple_n * np.log10(dbp / cfg.d0_m)
    far = pl_at_bp + 10.0 * cfg.ple_far * np.log10(np.maximum(d, dbp) / dbp)
    return np.where(d <= dbp, near, far).astype(np.float32)


MODELS = {
    "close_in": close_in,
    "floating_intercept": floating_intercept,
    "dual_slope": dual_slope,
}

# Which model a mode falls back to when cfg.model is left unset / unknown.
MODE_DEFAULT = {"indoor": "close_in", "outdoor": "dual_slope"}


def select_model(cfg: PathLossConfig):
    """Resolve the spreading model: explicit `cfg.model`, else the mode default."""
    name = cfg.model if cfg.model in MODELS else MODE_DEFAULT.get(cfg.mode, "close_in")
    return MODELS[name]


def pathloss_db(d_m, f_mhz: float, cfg: PathLossConfig):
    """Mean spreading path loss (dB) for the configured/mode-selected model."""
    return select_model(cfg)(d_m, f_mhz, cfg)
