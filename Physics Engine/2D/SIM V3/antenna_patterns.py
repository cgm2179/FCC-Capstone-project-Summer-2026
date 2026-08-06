#!/usr/bin/env python3
"""
antenna_patterns.py — per-family azimuth radiation patterns for the outdoor surrogate.

Generalizes `fw_bs_catalog.panel_atten_db` (a single 3GPP sector) to the antenna
families the outdoor catalog carries per PCI (`base_stations.csv` `antenna_type`),
so a **dish** beams a narrow pencil, a **yagi/lpda** a forward lobe, a
**patch/panel/slot** a ~65–70° sector, a **horn** a medium flare, and a
**loop/fractal** (and the rod family: omni/monopole/dipole/…) radiate ~omni in
azimuth.

The functional form is the 3GPP horizontal pattern A(φ)=min(12·(φ/HPBW)², Am) per
family — one forward main lobe plus a uniform back floor at Am dB. Near-omni
families use a huge HPBW + tiny Am (≈flat). `pattern_gain` is the 0..1 linear
amplitude the surrogate carries as its directivity input channel; `is_directional`
tells the FDTD whether to add the panel backplane (skipped for ~omni radiators, so
the ground-truth field matches the flat directivity channel).

Defaults are azimuth HPBW / front-to-back typical of each family; override per call
with `hpbw_deg` / `am_db`. Mesh geometry defaults live in
`export_antenna_catalog_3d.CATALOG_DEFAULTS`; these are the RF-pattern counterparts.
"""
from __future__ import annotations

import numpy as np

# (azimuth HPBW °, front-to-back Am dB) per family.
PATTERNS = {
    # pencil / high-gain
    "dish":         (10.0, 30.0),
    "phased_array": (20.0, 25.0),
    "horn":         (40.0, 25.0),
    # forward lobe (endfire arrays)
    "yagi":         (50.0, 20.0),
    "quad":         (55.0, 18.0),
    "lpda":         (60.0, 18.0),
    # sector / hemisphere (flat plates)
    "panel":        (65.0, 25.0),   # == fw_bs_catalog HPBW_DEG / AM_DB (backward-compatible)
    "patch":        (65.0, 20.0),
    "slot":         (70.0, 18.0),
    "pifa":         (110.0, 6.0),
    "inverted_f":   (120.0, 5.0),
    # near-omni in azimuth
    "loop":         (150.0, 3.0),
    "fractal":      (160.0, 2.0),
    "helical":      (140.0, 4.0),
    "metamaterial": (160.0, 3.0),
    "omni":         (360.0, 0.0),
    "monopole":     (360.0, 0.0),
    "whip":         (360.0, 0.0),
    "handheld":     (360.0, 0.0),
    "dipole":       (360.0, 0.0),
    "discone":      (360.0, 0.0),
    "biconical":    (360.0, 0.0),
}
DEFAULT = (65.0, 25.0)              # unknown kind → panel-like sector
DIRECTIONAL_AM_DB = 6.0            # ≥ this front-to-back ⇒ add the FDTD backplane


def params(kind):
    """(hpbw_deg, am_db) for a family; unknown → panel-like DEFAULT."""
    return PATTERNS.get((kind or "panel").lower(), DEFAULT)


def is_directional(kind):
    """True if the family has a real forward lobe (⇒ gets a backplane in the FDTD).
    Near-omni radiators (loop/dipole/omni/…) → False ⇒ radiate omni."""
    return params(kind)[1] >= DIRECTIONAL_AM_DB


def pattern_atten_db(kind, bearing_to_cell_deg, boresight_deg, hpbw_deg=None, am_db=None):
    """Azimuth off-boresight attenuation (dB ≥ 0) for one antenna family.
    Same 3GPP form as fw_bs_catalog.panel_atten_db; 'panel' is byte-identical."""
    hp, am = params(kind)
    if hpbw_deg is not None:
        hp = hpbw_deg
    if am_db is not None:
        am = am_db
    d = ((np.asarray(bearing_to_cell_deg, float) - boresight_deg + 180.0) % 360.0) - 180.0
    return np.minimum(12.0 * (d / hp) ** 2, am)


def pattern_gain(kind, bearing_to_cell_deg, boresight_deg, **kw):
    """0..1 linear amplitude — the surrogate's directivity input channel."""
    return 10.0 ** (-pattern_atten_db(kind, bearing_to_cell_deg, boresight_deg, **kw) / 20.0)


def _selftest():
    """Sanity: dish ≪ panel HPBW; loop ≈ flat; panel matches the 3GPP macro."""
    bear = np.linspace(-180, 180, 361)
    ok, fails = 0, []

    def chk(name, cond):
        nonlocal ok
        (fails.append(name) if not cond else None)
        ok += cond
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    def hpbw_of(kind):                 # measured -3 dB full width
        a = pattern_atten_db(kind, bear, 0.0)
        return 2 * bear[bear >= 0][np.argmax(a[bear >= 0] >= 3.0)]

    chk("panel == 3GPP macro (65°, 25 dB)",
        abs(pattern_atten_db("panel", 32.5, 0.0) - 3.0) < 0.05 and
        abs(pattern_atten_db("panel", 180.0, 0.0) - 25.0) < 1e-9)
    chk("dish beam ≪ panel beam", hpbw_of("dish") < 0.3 * hpbw_of("panel"))
    chk("loop ≈ near-omni (≤3 dB ripple)", float(np.max(pattern_atten_db("loop", bear, 0.0))) <= 3.0 + 1e-6)
    chk("omni is flat (0 dB everywhere)", float(np.max(pattern_atten_db("omni", bear, 0.0))) < 1e-9)
    chk("dish/horn/yagi directional; loop/omni not",
        all(map(is_directional, ["dish", "horn", "yagi", "panel"])) and
        not any(map(is_directional, ["loop", "fractal", "omni", "dipole"])))
    print(f"\n{ok} passed, {len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
