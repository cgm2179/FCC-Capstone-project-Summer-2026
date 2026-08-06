#!/usr/bin/env python3
"""
bands_v3.py — real RF bands for the SIM V3 full-wave engine + the FDTD cell-size
budget that follows from each one.

The headline SIM V3 mode is REAL frequency (4G / 5G / WLAN), full-wave, on a 2-D
horizontal floor plane. Unlike the eikonal engine (frequency only rescales loss),
a full-wave solve must RESOLVE the wavelength: cells h <= lambda / N with N ~ 10.
That single constraint decides whether a band is cheap (whole floor) or needs a
crop -- see estimate().

Cellular set mirrors the PCTEL G-flex walk carried in SIM V1 3D/manifest_3d.json
(band_catalog); Wi-Fi 2.4/5/6E are the WLAN bands the user asked for.
"""
from __future__ import annotations

from dataclasses import dataclass

C0 = 299_792_458.0  # m/s


@dataclass(frozen=True)
class Band:
    label: str
    f_mhz: float
    family: str      # "4G" | "5G" | "WLAN"
    note: str = ""

    @property
    def wavelength_m(self) -> float:
        return C0 / (self.f_mhz * 1e6)

    def cell_size_m(self, n_per_wavelength: float = 10.0) -> float:
        """FDTD cell h = lambda / N. N ~ 10-15 keeps numerical dispersion low."""
        return self.wavelength_m / float(n_per_wavelength)


# Real bands. Cellular from the T-Mobile/Verizon walk; WLAN = Wi-Fi 4/5/6E.
BANDS_V3 = [
    Band("LTE_B71_617",  617.0,  "4G",   "n71 low-band; cheapest full-wave (lambda ~0.49 m)"),
    Band("LTE_B13_751",  751.0,  "4G",   "Verizon Band 13 (cross-validation anchor)"),
    Band("LTE_B2_1960",  1960.0, "4G",   "PCS mid-band"),
    Band("NR_n41_2506",  2506.0, "5G",   "n41 2.5 GHz TDD"),
    Band("NR_n77_3700",  3700.0, "5G",   "n77 3.7 GHz C-band; needs a crop at N=10"),
    Band("WiFi_2G4",     2442.0, "WLAN", "Wi-Fi 4/6 2.4 GHz"),
    Band("WiFi_5G",      5500.0, "WLAN", "Wi-Fi 5 U-NII"),
    Band("WiFi_6E",      6125.0, "WLAN", "Wi-Fi 6E 6 GHz"),
]

_BY_LABEL = {b.label: b for b in BANDS_V3}


def get(label: str) -> Band:
    if label not in _BY_LABEL:
        raise KeyError(
            f"unknown band '{label}'. Available: {', '.join(_BY_LABEL)}"
        )
    return _BY_LABEL[label]


def estimate(band: Band, extent_m, n_per_wavelength: float = 10.0):
    """Grid size the band implies over a domain of `extent_m` = (Lx, Lz) metres.

    Returns a dict: cell size, grid cells per axis, total cells, and a rough
    'whole floor / crop advised' verdict. This is the 'pick two of three'
    budget made concrete.
    """
    lx, lz = float(extent_m[0]), float(extent_m[1])
    h = band.cell_size_m(n_per_wavelength)
    nx, nz = int(round(lx / h)), int(round(lz / h))
    total = nx * nz
    return dict(
        band=band.label, f_mhz=band.f_mhz, wavelength_m=band.wavelength_m,
        h_m=h, nx=nx, nz=nz, total_cells=total,
        whole_floor_ok=total <= 4_000_000,
        verdict=("whole floor" if total <= 4_000_000 else "crop advised"),
    )


if __name__ == "__main__":
    # quick budget table over the shipped 7th-floor extent (~78.6 x 39.6 m)
    for b in BANDS_V3:
        e = estimate(b, (78.6, 39.6))
        print(f"{b.label:14s} f={b.f_mhz:6.0f} MHz  lambda={b.wavelength_m*100:5.1f} cm"
              f"  h={e['h_m']*100:5.2f} cm  grid={e['nx']:5d}x{e['nz']:4d}"
              f"  {e['total_cells']/1e6:6.2f} M  -> {e['verdict']}")
