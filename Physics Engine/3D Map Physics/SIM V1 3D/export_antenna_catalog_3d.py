#!/usr/bin/env python3
"""
export_antenna_catalog_3d.py — antenna type picker catalog for the SIM V1 3D
placement sandbox (web/antenna_sandbox.html).

Builds one instance of every ANTENNA_TYPES kind (Antenna_Type_3D.py, in the
sibling "Object and Tranmission/Transmitter Objects" folder) at a reasonable
default parameterization, attaches its physics body (Antenna_Physics_3D.py),
and writes web/antenna_catalog_3d.js as window.SIM3D_ANTENNA_CATALOG — mirrors
export_web3.py's `window.X = {...}` pattern so it loads via <script src> on
file://, with no Python/server needed at browser runtime. The browser only
needs to translate/rotate/re-mass an entry when the user tweaks a parameter
in the UI; it does not need to re-run this geometry code.

usage: python "SIM V1 3D/export_antenna_catalog_3d.py"
"""
import json
import sys
from pathlib import Path

import math

HERE = Path(__file__).resolve().parent
TRANSMITTER_OBJECTS = (HERE.parent / "Object and Tranmission" / "Transmitter Objects").resolve()
if str(TRANSMITTER_OBJECTS) not in sys.path:
    sys.path.insert(0, str(TRANSMITTER_OBJECTS))
# antenna_patterns.py (SIM V3) is the single source of truth for the azimuth
# radiation pattern (HPBW / front-to-back) — the surrogate builds its directivity
# channel from it, and the catalog carries the same numbers so the UI beam matches.
SIM_V3 = (HERE.parent.parent / "2D" / "SIM V3").resolve()
if str(SIM_V3) not in sys.path:
    sys.path.insert(0, str(SIM_V3))

from Antenna_Type_3D import ANTENNA_TYPES, build_antenna  # noqa: E402
from Antenna_Physics_3D import physics_for  # noqa: E402
import antenna_patterns as AP  # noqa: E402

LABELS = {
    "omni": "Omnidirectional Rod",
    "monopole": "Monopole / Vertical Whip",
    "whip": "Whip (Long Monopole)",
    "handheld": "Handheld Device Whip",
    "dipole": "Dipole (Vertical / Horizontal)",
    "inverted_f": "Inverted-F",
    "pifa": "PIFA (Planar Inverted-F)",
    "panel": "Panel / Sector",
    "patch": "Patch (Microstrip)",
    "yagi": "Yagi-Uda",
    "quad": "Quad (Cubical Quad)",
    "lpda": "Log-Periodic Dipole Array (LPDA)",
    "phased_array": "Phased Array",
    "discone": "Discone",
    "dish": "Parabolic Dish",
    "horn": "Horn (Conical)",
    "loop": "Small Loop",
    "helical": "Helical",
    "slot": "Slot",
    "biconical": "Biconical",
    "fractal": "Fractal (Koch Monopole)",
    "metamaterial": "Metamaterial (SRR Array)",
}

# Per-kind render color so visually-similar shapes (e.g. the thin-rod family:
# omni/monopole/whip/handheld/dipole, or the flat-plate family: panel/patch/
# slot) are still distinguishable at a glance in the sandbox, not just by
# their (structurally different, but small/far-away) mesh silhouette.
COLORS = {
    "omni": 0xffb020,
    "monopole": 0xff6b35,
    "whip": 0xf7931e,
    "handheld": 0xffd166,
    "dipole": 0xef476f,
    "inverted_f": 0x06d6a0,
    "pifa": 0x118ab2,
    "panel": 0x4361ee,
    "patch": 0x3a86ff,
    "yagi": 0x2ec4b6,
    "quad": 0x80ed99,
    "lpda": 0x57cc99,
    "phased_array": 0x7209b7,
    "discone": 0xf72585,
    "dish": 0xb5179e,
    "horn": 0x9d4edd,
    "loop": 0xffd60a,
    "helical": 0xfb8500,
    "slot": 0x8ecae6,
    "biconical": 0xe63946,
    "fractal": 0x06ffa5,
    "metamaterial": 0x00b4d8,
}

# Reasonable default params per catalog entry -- the exact set the user can
# then edit in the sandbox UI (per-kind, since each antenna's create() takes
# different physical dimensions).
CATALOG_DEFAULTS = {
    "omni": dict(length_m=0.9, radius_m=0.02),
    "monopole": dict(length_m=0.3, radius_m=0.005, ground_plane_radius_m=0.15),
    "whip": dict(length_m=1.2, radius_m=0.004),
    "handheld": dict(length_m=0.08, radius_m=0.002),
    "dipole": dict(total_length_m=0.5, radius_m=0.004),
    "inverted_f": dict(ground_width_m=0.1, ground_depth_m=0.05, arm_length_m=0.04, height_m=0.008, planar=False),
    "pifa": dict(ground_width_m=0.1, ground_depth_m=0.05, arm_length_m=0.04, height_m=0.008, planar=True),
    "panel": dict(width_m=0.3, height_m=0.9, depth_m=0.08),
    "patch": dict(patch_width_m=0.05, patch_height_m=0.05),
    "yagi": dict(boom_length_m=1.2, element_lengths_m=[0.5, 0.48, 0.44], element_spacings_m=[0.0, 0.3, 0.6]),
    "quad": dict(boom_length_m=1.0, element_perimeters_m=[2.0, 1.9], element_spacings_m=[0.0, 0.4]),
    "lpda": dict(longest_element_m=0.6, n_elements=6, tau=0.85, sigma=0.06),
    "phased_array": dict(rows=4, cols=4, element_spacing_x_m=0.06, element_spacing_y_m=0.06),
    "discone": dict(cone_base_radius_m=0.15, cone_length_m=0.4, disc_radius_m=0.06),
    "dish": dict(diameter_m=0.6, focal_length_m=0.25),
    "horn": dict(shape="conical", length_m=0.2, throat_radius_m=0.02, aperture_radius_m=0.12),
    "loop": dict(diameter_m=0.2),
    "helical": dict(diameter_m=0.1, pitch_m=0.08, turns=8, ground_plane_radius_m=0.1),
    "slot": dict(plate_width_m=0.2, plate_height_m=0.1, slot_width_m=0.1, slot_height_m=0.01),
    "biconical": dict(cone_length_m=0.2, cone_base_radius_m=0.08),
    "fractal": dict(total_length_m=0.3, iterations=3, mode="monopole", ground_plane_radius_m=0.1),
    "metamaterial": dict(rows=3, cols=3, cell_size_m=0.02),
}


# ---------------------------------------------------------------- 2D footprints
# A distinctive TOP-DOWN silhouette per family (boresight = +X), in metres, for
# the antenna/layout view. Polylines are [[x,y],…]; `outline` is the body,
# `detail` extra strokes (elements/feed/slot). Paired with `pattern` (the azimuth
# beam from antenna_patterns) the UI can draw both the physical antenna and its lobe.
def _circle(r, cx=0.0, cy=0.0, n=44):
    return [[cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)]
            for i in range(n + 1)]


def _rect(x0, x1, y0, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def footprint_2d(kind, p):
    """Return {outline, detail:[…], boresight:bool} — a family-distinctive top-down
    outline (boresight along +X). Falls back to a small rod-circle for the omni
    rod family and anything without a bespoke shape."""
    d = dict(outline=[], detail=[], boresight=AP.is_directional(kind))
    if kind == "panel":                                    # back plate + radiator face
        w, dep = p.get("width_m", 0.3), p.get("depth_m", 0.08)
        d["outline"] = _rect(-dep, 0.0, -w / 2, w / 2)
        d["detail"] = [[[0.0, -w / 2], [0.0, w / 2]]]      # radiating face at x=0
    elif kind == "patch":
        w, h = p.get("patch_width_m", 0.05), p.get("patch_height_m", 0.05)
        d["outline"] = _rect(-h, 0.0, -w / 2, w / 2)
    elif kind == "slot":
        w, h = p.get("plate_width_m", 0.2), p.get("plate_height_m", 0.1)
        sw, sh = p.get("slot_width_m", 0.1), p.get("slot_height_m", 0.01)
        d["outline"] = _rect(-h, 0.0, -w / 2, w / 2)
        d["detail"] = [_rect(-h / 2 - sh / 2, -h / 2 + sh / 2, -sw / 2, sw / 2)]  # slot cutout
    elif kind in ("yagi", "quad"):                          # boom + perpendicular elements
        boom = p.get("boom_length_m", 1.2)
        els = p.get("element_lengths_m") or [0.5, 0.48, 0.44]
        sp = p.get("element_spacings_m") or [0.0, 0.3, 0.6]
        d["outline"] = [[0.0, 0.0], [boom, 0.0]]           # boom along +X
        d["detail"] = [[[x, -L / 2], [x, L / 2]] for x, L in zip(sp, els)]  # elements
    elif kind == "lpda":                                    # boom + graduated elements
        L0, n, tau = p.get("longest_element_m", 0.6), p.get("n_elements", 6), p.get("tau", 0.85)
        sig = p.get("sigma", 0.06)
        xs, els, x = [], [], 0.0
        for i in range(n):
            Li = L0 * (tau ** i)
            xs.append(x); els.append(Li); x += 2 * sig * Li / (1 - tau) * (1 - tau)  # spacing∝element
        boom = xs[-1] + 0.05
        d["outline"] = [[0.0, 0.0], [boom, 0.0]]
        d["detail"] = [[[x, -L / 2], [x, L / 2]] for x, L in zip(xs, els)]
    elif kind == "dish":                                    # parabolic arc + aperture + feed
        D, f = p.get("diameter_m", 0.6), p.get("focal_length_m", 0.25)
        ys = [(-D / 2) + D * i / 40 for i in range(41)]
        arc = [[-(y * y) / (4 * f), y] for y in ys]         # concave, opening toward +X
        d["outline"] = arc
        d["detail"] = [[[arc[0][0], -D / 2], [arc[-1][0], D / 2]], _circle(0.015, f, 0.0, 12)]  # rim chord + feed
    elif kind == "horn":                                    # throat -> aperture trapezoid
        L = p.get("length_m", 0.2); tr = p.get("throat_radius_m", 0.02); ar = p.get("aperture_radius_m", 0.12)
        d["outline"] = [[0.0, -tr], [L, -ar], [L, ar], [0.0, tr], [0.0, -tr]]
    elif kind == "loop":
        d["outline"] = _circle(p.get("diameter_m", 0.2) / 2)
    elif kind == "fractal":                                 # small Koch-ish open curve
        seg = p.get("total_length_m", 0.3) / 4
        d["outline"] = [[0, 0], [seg, 0], [1.5 * seg, seg * 0.6], [2 * seg, 0],
                        [3 * seg, 0], [3.5 * seg, -seg * 0.6], [4 * seg, 0]]
        d["boresight"] = False
    elif kind == "phased_array":                            # element grid
        rows, cols = p.get("rows", 4), p.get("cols", 4)
        dx, dy = p.get("element_spacing_x_m", 0.06), p.get("element_spacing_y_m", 0.06)
        d["outline"] = _rect(-0.01, 0.01, -(rows - 1) * dy / 2 - dy / 2, (rows - 1) * dy / 2 + dy / 2)
        d["detail"] = [_circle(dx * 0.18, 0.0, (j - (rows - 1) / 2) * dy, 10) for j in range(rows)]
    else:                                                   # omni rod family: top-down = a dot/ring
        d["outline"] = _circle(0.05)
        d["boresight"] = False
    return d


def build_catalog():
    missing = set(ANTENNA_TYPES) - set(CATALOG_DEFAULTS)
    assert not missing, f"CATALOG_DEFAULTS missing entries for: {missing}"

    entries = {}
    for kind, defaults in CATALOG_DEFAULTS.items():
        antenna = build_antenna(kind, **defaults)
        physics = physics_for(antenna)
        d = antenna.to_web_dict()
        hpbw, am = AP.params(kind)
        entries[kind] = dict(
            kind=kind,
            label=LABELS.get(kind, kind),
            color=COLORS.get(kind, 0xffb020),
            default_params=defaults,
            vertices=d["vertices"],
            faces=d["faces"],
            physics=physics.to_web_dict(),
            footprint2d=footprint_2d(kind, defaults),
            pattern=dict(hpbw_deg=hpbw, am_db=am, directional=AP.is_directional(kind)),
        )
    return entries


def main():
    entries = build_catalog()
    out = dict(kinds=list(entries.keys()), entries=entries)
    (HERE / "web").mkdir(exist_ok=True)
    p = HERE / "web" / "antenna_catalog_3d.js"
    p.write_text("window.SIM3D_ANTENNA_CATALOG = " + json.dumps(out) + ";\n")
    print(f"wrote {p} ({len(entries)} antenna kinds, {p.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
