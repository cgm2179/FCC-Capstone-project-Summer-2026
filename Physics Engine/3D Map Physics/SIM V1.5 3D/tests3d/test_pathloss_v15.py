#!/usr/bin/env python3
"""Tests for the parameterized default path-loss engine (SIM V1.5 config/models/budget)."""
import numpy as np
import pytest

import link_budget as LB
import pathloss_config as C
import pathloss_models as PM
from pathloss_config import PathLossConfig


# ------------------------------------------------------------- config: all 9 parameters
def test_all_nine_parameters_present():
    cfg = PathLossConfig.indoor()
    # Layer 1: n, sigma, PL(d0), (alpha,beta,gamma), dbp, d0
    for a in ("ple_n", "shadow_sigma_db", "fi_alpha_db", "fi_beta", "fi_gamma", "d0_m"):
        assert hasattr(cfg, a)
    assert cfg.pl_ref_db(3500.0) > 0          # PL(d0), close-in reference
    assert cfg.breakpoint(3500.0) > 0         # dbp
    # Layer 2: EIRP=Pt+Gt, Lc, M=z*sigma
    assert cfg.eirp_dbm == cfg.tx_power_dbm + cfg.tx_gain_dbi
    assert hasattr(cfg, "cable_loss_db")
    assert cfg.fade_margin_db == cfg.z_sigma * cfg.shadow_sigma_db


def test_reliability_z_factors_match_2d_optimizer():
    assert (C.RELIABILITY_Z["80"], C.RELIABILITY_Z["90"], C.RELIABILITY_Z["95"]) == (0.84, 1.28, 1.65)
    assert PathLossConfig.indoor(reliability="95").fade_margin_db == pytest.approx(1.65 * 8.0)


def test_mode_presets_and_model_selection():
    assert PathLossConfig.indoor().mode == "indoor"
    assert PathLossConfig.outdoor().mode == "outdoor"
    assert PM.select_model(PathLossConfig.indoor()) is PM.close_in       # indoor -> close-in
    assert PM.select_model(PathLossConfig.outdoor()) is PM.dual_slope    # outdoor -> dual-slope
    assert PathLossConfig.for_mode("outdoor").model == "dual_slope"


def test_config_validation_and_json_roundtrip():
    with pytest.raises(ValueError):
        PathLossConfig(mode="space")
    with pytest.raises(ValueError):
        PathLossConfig(model="nope")
    with pytest.raises(ValueError):
        PathLossConfig(reliability="77")
    cfg = PathLossConfig.outdoor(ple_n=2.2, cable_loss_db=1.5, breakpoint_m=60.0)
    assert PathLossConfig.from_dict(cfg.to_dict()) == cfg


# ------------------------------------------------------------- models
def test_close_in_formula_and_monotonic():
    cfg = PathLossConfig.indoor(ple_n=3.0)
    f = 3500.0
    pl = PM.close_in(np.array([1.0, 10.0, 100.0]), f, cfg)
    ref = cfg.pl_ref_db(f)
    assert float(pl[0]) == pytest.approx(ref, abs=1e-2)          # at d0
    assert float(pl[1]) == pytest.approx(ref + 30.0, abs=1e-2)   # x10 -> +10*n
    assert float(pl[2]) == pytest.approx(ref + 60.0, abs=1e-2)   # x100 -> +20*n
    assert np.all(np.diff(pl) > 0)


def test_floating_intercept_formula():
    cfg = PathLossConfig.indoor(model="floating_intercept", fi_alpha_db=30.0, fi_beta=2.0, fi_gamma=2.0)
    pl = PM.floating_intercept(np.array([10.0]), 1000.0, cfg)    # f=1 GHz -> gamma term = 0
    assert float(pl[0]) == pytest.approx(30.0 + 20.0, abs=1e-2)


def test_dual_slope_continuous_and_steeper_past_breakpoint():
    cfg = PathLossConfig.outdoor(ple_n=2.0, ple_far=4.0, breakpoint_m=50.0)
    f = 3500.0
    dbp = cfg.breakpoint(f)
    assert dbp == 50.0
    lo = float(PM.dual_slope(np.array([dbp - 1e-6]), f, cfg)[0])
    hi = float(PM.dual_slope(np.array([dbp + 1e-6]), f, cfg)[0])
    assert abs(lo - hi) < 1e-2                                    # continuous at dbp
    seg = PM.dual_slope(np.array([dbp, dbp * 10.0]), f, cfg)      # a decade beyond dbp
    assert float(seg[1] - seg[0]) == pytest.approx(10 * 4.0, abs=1e-2)   # steep far slope


def test_pathloss_db_dispatches_by_mode():
    assert np.allclose(PM.pathloss_db(np.array([10.0]), 3500.0, PathLossConfig.indoor()),
                       PM.close_in(np.array([10.0]), 3500.0, PathLossConfig.indoor()))
    out = PathLossConfig.outdoor(breakpoint_m=50.0)
    assert np.allclose(PM.pathloss_db(np.array([200.0]), 3500.0, out),
                       PM.dual_slope(np.array([200.0]), 3500.0, out))


# ------------------------------------------------------------- link budget
def test_rsrp_and_margin():
    cfg = PathLossConfig.indoor(tx_power_dbm=20.0, tx_gain_dbi=3.0, cable_loss_db=2.0,
                                shadow_sigma_db=8.0, reliability="90")
    pl = np.array([100.0])
    assert float(LB.received_power_dbm(pl, cfg)[0]) == pytest.approx(23.0 - 2.0 - 100.0)
    assert float(LB.rsrp_dbm(pl, cfg)[0]) == pytest.approx(23.0 - 2.0 - 100.0 - 1.28 * 8.0)
    assert float(LB.rsrp_dbm(pl, cfg, apply_margin=False)[0]) == pytest.approx(23.0 - 2.0 - 100.0)


def test_coverage_fraction():
    cfg = PathLossConfig.indoor(tx_power_dbm=30.0, tx_gain_dbi=0.0, cable_loss_db=0.0,
                                shadow_sigma_db=0.0)                 # zero margin
    pl = np.array([100.0, 130.0])                                    # RSRP = -70, -100
    assert LB.coverage(pl, cfg, threshold_dbm=-85.0) == pytest.approx(0.5)


if __name__ == "__main__":
    import sys
    raise SystemExit(pytest.main([__file__, "-q"]))
