#!/usr/bin/env python3
"""
link_budget.py — Layer 2: PL(dB) -> received power / RSRP / coverage.

This is applied *after* the path-loss field (from engine.py or, at deploy time, the
Triplanar UNet's cached PL) so it is a cheap per-query add with nothing to learn:

    P_rx  = EIRP - Lc - PL
    RSRP  = P_rx - M            with fade margin  M = z_sigma * sigma
    covered  <=>  RSRP >= threshold

Consolidates the link-budget terms that were scattered across the 2-D optimizer
(phase_e_optimizer.py: EIRP = Pt+Gt, margin = z*sigma) and Construct_Reciever_3D.py
(RSRP ~= EIRP - PL) into the one parameterized config.
"""
from __future__ import annotations

import numpy as np

from pathloss_config import PathLossConfig


def received_power_dbm(pl_db, cfg: PathLossConfig):
    """P_rx = EIRP - Lc - PL   (no fade margin)."""
    return cfg.eirp_dbm - cfg.cable_loss_db - np.asarray(pl_db, np.float32)


def rsrp_dbm(pl_db, cfg: PathLossConfig, *, apply_margin: bool = True):
    """RSRP = EIRP - Lc - PL - M, with M = z_sigma * sigma the fade margin.

    `apply_margin=False` gives the un-margined median RSRP (50% reliability).
    """
    p = received_power_dbm(pl_db, cfg)
    return p - (cfg.fade_margin_db if apply_margin else 0.0)


def coverage(pl_db, cfg: PathLossConfig, *, threshold_dbm: float = -85.0, mask=None) -> float:
    """Fraction of (masked) cells whose margin-adjusted RSRP clears the threshold."""
    ok = rsrp_dbm(pl_db, cfg) >= threshold_dbm
    if mask is not None:
        ok = ok[np.asarray(mask, bool)]
    return float(ok.mean()) if ok.size else 0.0


def link_budget_summary(cfg: PathLossConfig) -> dict:
    """The Layer-2 terms, for logging / the surrogate contract sidecar."""
    return dict(
        eirp_dbm=cfg.eirp_dbm,
        tx_power_dbm=cfg.tx_power_dbm,
        tx_gain_dbi=cfg.tx_gain_dbi,
        cable_loss_db=cfg.cable_loss_db,
        shadow_sigma_db=cfg.shadow_sigma_db,
        reliability=cfg.reliability,
        z_sigma=cfg.z_sigma,
        fade_margin_db=round(cfg.fade_margin_db, 3),
    )
