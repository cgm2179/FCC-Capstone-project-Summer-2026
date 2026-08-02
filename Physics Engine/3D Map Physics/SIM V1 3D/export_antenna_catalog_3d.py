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

HERE = Path(__file__).resolve().parent
TRANSMITTER_OBJECTS = (HERE.parent / "Object and Tranmission" / "Transmitter Objects").resolve()
if str(TRANSMITTER_OBJECTS) not in sys.path:
    sys.path.insert(0, str(TRANSMITTER_OBJECTS))

from Antenna_Type_3D import ANTENNA_TYPES, build_antenna  # noqa: E402
from Antenna_Physics_3D import physics_for  # noqa: E402

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


def build_catalog():
    missing = set(ANTENNA_TYPES) - set(CATALOG_DEFAULTS)
    assert not missing, f"CATALOG_DEFAULTS missing entries for: {missing}"

    entries = {}
    for kind, defaults in CATALOG_DEFAULTS.items():
        antenna = build_antenna(kind, **defaults)
        physics = physics_for(antenna)
        d = antenna.to_web_dict()
        entries[kind] = dict(
            kind=kind,
            label=LABELS.get(kind, kind),
            color=COLORS.get(kind, 0xffb020),
            default_params=defaults,
            vertices=d["vertices"],
            faces=d["faces"],
            physics=physics.to_web_dict(),
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
