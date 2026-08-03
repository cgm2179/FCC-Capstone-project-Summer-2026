"""M5 scaffolding: Construct_Reciever_3D + osm_building_height (offline)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
RX = SIM.parent / "Object and Tranmission" / "Reciever Objects"
sys.path.insert(0, str(RX))
sys.path.insert(0, str(SIM))

from Construct_Reciever_3D import BandSpec, metrics_from_pl, rsrp_from_pl  # noqa: E402
from osm_building_height import (  # noqa: E402
    FALLBACK_HEIGHT_M,
    height_from_tags,
    resolve_tx_height_m,
)


def test_metrics_from_pl_monotonic():
    pl = np.array([60.0, 80.0, 100.0, 120.0])
    m = metrics_from_pl(pl, eirp_dbm=20.0, band=BandSpec.nr(20.0))
    assert np.all(np.diff(m["rsrp"]) < 0)
    assert np.all(m["rsrp"] < 0)
    assert m["n_rb"] >= 1
    assert rsrp_from_pl(80.0, eirp_dbm=20.0) < -50


def test_osm_height_from_tags():
    h, src = height_from_tags({"height": "12"})
    assert h == 12.0 and src == "osm:height"
    h, src = height_from_tags({"building:levels": "3"})
    assert abs(h - 10.5) < 1e-6 and "levels" in src


def test_osm_resolve_offline_fallback():
    r = resolve_tx_height_m(use_network=False)
    assert abs(r["height_m"] - FALLBACK_HEIGHT_M) < 1e-6
    assert r["source"].startswith("fallback")
