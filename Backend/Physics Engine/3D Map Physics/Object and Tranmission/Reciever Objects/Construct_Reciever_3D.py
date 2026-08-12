#!/usr/bin/env python3
"""
Construct_Reciever_3D.py — PL (dB) → cellular quality metrics (M5).

Mirrors the 2-D STEP_2/multiband_metrics.py bookkeeping, but takes a path-loss
field (from SceneV3, a cached volume, or the DL surrogate) instead of re-solving
Motley-Keenan. The browser Tier-1 display already uses the same identity:

    RSRP_dBm ≈ EIRP_RE − PL_dB

where EIRP_RE is the EIRP referred to one resource element.

usage:
  from Construct_Reciever_3D import metrics_from_pl, BandSpec
  m = metrics_from_pl(pl_db, eirp_dbm=20.0, band=BandSpec.nr(bw_mhz=20))
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOISE_FIGURE_DB = 7.0
KTB_1HZ_DBM = -174.0


@dataclass(frozen=True)
class BandSpec:
    """Resource-block geometry for one carrier."""
    protocol: str          # "LTE" | "NR"
    bw_mhz: float
    scs_khz: float         # 15 LTE, 30 FR1 typical

    @property
    def n_rb(self) -> int:
        # usable fraction ≈ 0.9 of the channel for RBs
        return max(1, int(self.bw_mhz * 1000 * 0.9 / (12 * self.scs_khz)))

    @classmethod
    def lte(cls, bw_mhz: float = 20.0) -> "BandSpec":
        return cls("LTE", float(bw_mhz), 15.0)

    @classmethod
    def nr(cls, bw_mhz: float = 20.0, scs_khz: float = 30.0) -> "BandSpec":
        return cls("NR", float(bw_mhz), float(scs_khz))

    @classmethod
    def from_freq_mhz(cls, f_mhz: float) -> "BandSpec":
        """Rough default from carrier frequency (NR FR1 above 1 GHz)."""
        f = float(f_mhz)
        if f < 1000:
            return cls.lte(10.0) if f < 700 else cls.lte(20.0)
        if f < 3000:
            return cls.nr(20.0 if f < 2500 else 60.0, 30.0)
        return cls.nr(100.0, 30.0)


def noise_re_dbm(band: BandSpec, noise_figure_db: float = NOISE_FIGURE_DB) -> float:
    return KTB_1HZ_DBM + 10.0 * np.log10(band.scs_khz * 1e3) + float(noise_figure_db)


def eirp_re_dbm(eirp_dbm: float, band: BandSpec) -> float:
    """Total EIRP referred to one RE (12 subcarriers × 1 symbol)."""
    return float(eirp_dbm) - 10.0 * np.log10(12 * band.n_rb)


def metrics_from_pl(pl_db, *, eirp_dbm: float = 20.0, band: BandSpec | None = None,
                    noise_figure_db: float = NOISE_FIGURE_DB,
                    interferer_pl_db=None) -> dict:
    """Convert a PL field (dB) into RSRP / RSSI / RSRQ / SINR.

    Parameters
    ----------
    pl_db : array-like
        Path loss of the *serving* link (dB). Same shape is returned.
    eirp_dbm : float
        Total radiated power + antenna gain of the serving Tx.
    band : BandSpec
        Defaults to NR 20 MHz / 30 kHz SCS.
    interferer_pl_db : array-like or None
        Optional co-channel interferer PL; if None, SINR is noise-limited.

    Returns
    -------
    dict with numpy arrays: rsrp, rssi, rsrq, sinr (and scalars n_rb, noise_re_dbm).
    """
    band = band or BandSpec.nr(20.0)
    pl = np.asarray(pl_db, np.float64)
    eirp_re = eirp_re_dbm(eirp_dbm, band)
    n_re = noise_re_dbm(band, noise_figure_db)

    rsrp = eirp_re - pl                                          # dBm per RE
    serv_lin = 10.0 ** (rsrp / 10.0)
    noise_lin = 10.0 ** (n_re / 10.0)

    if interferer_pl_db is None:
        interf_lin = 0.0
    else:
        interf_lin = 10.0 ** ((eirp_re - np.asarray(interferer_pl_db, np.float64)) / 10.0)

    # Wideband RSSI ≈ 12*N_RB REs of (serving + interferer + noise)
    rssi = 10.0 * np.log10(12 * band.n_rb) + 10.0 * np.log10(
        serv_lin + interf_lin + noise_lin)
    rsrq = 10.0 * np.log10(band.n_rb) + rsrp - rssi
    sinr = rsrp - 10.0 * np.log10(interf_lin + noise_lin)

    return {
        "rsrp": rsrp.astype(np.float32),
        "rssi": np.asarray(rssi, np.float32),
        "rsrq": np.asarray(rsrq, np.float32),
        "sinr": np.asarray(sinr, np.float32),
        "n_rb": int(band.n_rb),
        "noise_re_dbm": float(n_re),
        "eirp_re_dbm": float(eirp_re),
        "band": band,
    }


def rsrp_from_pl(pl_db, eirp_dbm: float = 20.0, band: BandSpec | None = None) -> np.ndarray:
    """Scalar/array convenience: RSRP only (matches the browser Tier-1 identity)."""
    return metrics_from_pl(pl_db, eirp_dbm=eirp_dbm, band=band)["rsrp"]


if __name__ == "__main__":
    # Smoke: free-space-ish PL 80 dB @ 20 dBm EIRP → RSRP around -60 to -80 dBm
    pl = np.array([60.0, 80.0, 100.0, 120.0])
    m = metrics_from_pl(pl, eirp_dbm=20.0, band=BandSpec.nr(20.0))
    print(f"n_rb={m['n_rb']}  noise_RE={m['noise_re_dbm']:.1f} dBm  "
          f"EIRP_RE={m['eirp_re_dbm']:.1f} dBm")
    for i, p in enumerate(pl):
        print(f"  PL {p:5.0f} → RSRP {m['rsrp'][i]:7.1f}  RSRQ {m['rsrq'][i]:6.1f}  "
              f"SINR {m['sinr'][i]:6.1f}  RSSI {m['rssi'][i]:7.1f}")
    assert np.all(m["rsrp"] < 0) and np.all(m["rsrq"] < 0)
    print("Construct_Reciever_3D smoke OK")
