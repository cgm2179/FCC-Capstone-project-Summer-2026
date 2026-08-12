#!/usr/bin/env python3
"""
nearfield.py — near/far field zones + the close-in path-loss correction.

Standard FSPL (1/d^2) fails within ~1 wavelength of the transmitter: it drops
toward 0 dB and then negative (unphysical "gain") as d -> 0. Near a source the
field is reactive/radiating with much steeper falloff and interference, so path
loss cannot be read off FSPL there. In the hybrid engine the true near field is
handled by the FDTD solve; this module supplies (a) the zone boundaries used to
size the FDTD near-box, and (b) a close-in FLOOR that keeps the analytic
far-field model physical in the crossfade band, per Schantz's rule of thumb
(matched near-field coupling ~ -6 dB path gain, i.e. ~6 dB loss, roughly flat).

Refs (user-supplied): en.wikipedia.org/wiki/Near_and_far_field; Schantz near-field
path-loss laws.
"""
from __future__ import annotations

import numpy as np

C0 = 299_792_458.0


def wavelength_m(f_mhz) -> float:
    return C0 / (float(f_mhz) * 1e6)


def zones(f_mhz, D_m=0.15) -> dict:
    """Field-zone boundaries for frequency f and antenna max dimension D.

      reactive     d < lambda/2pi            (stored energy, ~1/d^3)
      radiating    lambda/2pi < d < 2D^2/lam (Fresnel, interference)
      far          d > 2D^2/lambda           (Fraunhofer, ~1/d^2)

    D defaults to 0.15 m (a small WLAN AP); for such an antenna the Fraunhofer
    distance is only cm, so indoors the region that actually NEEDS full-wave is
    the dense-multipath zone (several metres), not the antenna near-field — the
    hybrid near-radius is a cost choice, with these as physical annotations.
    """
    lam = wavelength_m(f_mhz)
    return dict(wavelength_m=lam, reactive_m=lam / (2 * np.pi),
                fraunhofer_m=2.0 * float(D_m) ** 2 / lam)


def fspl_db(d_m, f_mhz):
    """Free-space path loss (dB). Matches manifest fspl_const_db = -27.55."""
    d = np.maximum(np.asarray(d_m, float), 1e-3)
    return 20.0 * np.log10(d) + 20.0 * np.log10(float(f_mhz)) - 27.55


def apply_close_in(PL, d_m, f_mhz, flat_loss_db=6.0):
    """Keep an analytic path-loss map physical within ~1 wavelength.

    Within d < lambda, hold the loss at least at its 1-wavelength FSPL value
    (stops the FSPL singularity), and floor the whole map at `flat_loss_db`
    (the ~-6 dB matched near-field coupling). Leaves d > lambda untouched.
    """
    lam = wavelength_m(f_mhz)
    d = np.asarray(d_m, float)
    pl_lambda = float(fspl_db(lam, f_mhz))
    out = np.where(d < lam, np.maximum(PL, pl_lambda), PL)
    return np.maximum(out, flat_loss_db)


# --- Schantz relative near-field power laws (informational; FDTD covers the
#     true near field in the hybrid). Return relative power vs distance. --------
def schantz_relative_power(d_m, f_mhz, mode="e"):
    """Relative received power vs d using Schantz's near-field expansions in
    x = k d (k = 2pi/lambda). E-dominant: 1/x^2 - 1/x^4 + 1/x^6;
    H-dominant: 1/x^2 + 1/x^4. Far field (x>>1) -> 1/x^2 (inverse square)."""
    lam = wavelength_m(f_mhz)
    x = np.maximum(2.0 * np.pi * np.asarray(d_m, float) / lam, 1e-6)
    if mode == "h":
        return 1.0 / x**2 + 1.0 / x**4
    return 1.0 / x**2 - 1.0 / x**4 + 1.0 / x**6


if __name__ == "__main__":
    for f in (617.0, 2442.0, 3700.0, 5500.0):
        z = zones(f)
        print(f"{f:6.0f} MHz  lambda={z['wavelength_m']*100:5.1f} cm  "
              f"reactive<{z['reactive_m']*100:4.1f} cm  "
              f"far>{z['fraunhofer_m']*100:5.1f} cm (D=0.15 m)")
