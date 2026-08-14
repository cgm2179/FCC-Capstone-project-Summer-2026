#!/usr/bin/env python3
"""
cad_materials.py — map the CAD OBJ's material names → ITU-R P.2040 εr/σ.

The 7th-floor CAD carries 253 `.mtl` materials (wood flooring, carpet, glass,
aluminum, columns, appliances, painted colours…). The full-wave engine needs a
complex permittivity εr and conductivity σ per material, from `physics_v2.P2040`.
This module resolves each CAD material NAME → a P.2040 key (by keyword rules) →
(εr, σ) at a frequency, plus a nearest 6-class id for the compat `material_grid`.

This is the one judgment-call layer: `--dump` writes `cad_material_map.json`
(all names → P.2040 + εr/σ) for review before it's wired into the voxelizer.

  python cad_materials.py --mtl <file.mtl> --band LTE_B71_617 --dump
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import bootstrap  # noqa: F401  (side effect: sys.path → physics_v2)
import physics_v2 as PV

HERE = Path(__file__).resolve().parent

# P.2040 has no carpet; add a thin low-loss floor-covering (εr≈1.2, small σ).
# `clutter` = homogenized sub-wavelength office clutter (chair wheels, handles, cables…):
# a bulk lossy medium (σ≈0.03 S/m) per the two-tier rule, not resolved as its real material.
LOCAL = {"carpet": dict(a=1.2, b=0.0, c=0.002, d=1.0),
         "clutter": dict(a=1.2, b=0.0, c=0.03, d=0.0)}

# nearest compat class ids (manifest 6-class scheme):
# 0 air · 1 drywall · 2 concrete/floor · 3 barrier(core/metal) · 4 furniture/wood · 5 glass
RULES = [
    (r"lisanne|ty_|sophie|people|person|skin|human|man\b|woman",  None,   None),  # people (transient)
    (r"glass|window|glazing",                              "glass",        5),
    (r"alum|alu|inox|stainless|steel|metal|chrome|brass|copper|iron|corrogate|corrugate|"
     r"fencing|diamond_mesh|charcoal_appliance|appliance", "metal",        3),
    (r"formica|laminate|vinyl",                            "wood",         4),  # plastic laminate ≈ wood
    (r"concrete|colonne|basi|column|blacktop|slab|cement|structural|aggregate",
                                                           "concrete",     2),
    (r"marble|granite|stone|ceramic|tile|pavers",          "marble",       2),
    (r"wood|walnut|oak|ply|chipboard|hardwood|floorboard|bukva|birch|beech|bamboo|cherry|"
     r"veneer|mdf|desk|table",                             "wood",         4),
    (r"carpet|loop|plush|rug",                             "carpet",       2),
    (r"gyp|drywall|plaster|gypsum|partition|cubicle",      "plasterboard", 1),
    (r"ceiling|acoustic",                                  "ceiling_board", 1),
    (r"cardboard|paper|fabric|textile|leather|foam|cushion|seat|chair|sofa",
                                                           "wood",         4),  # soft clutter ≈ wood
    (r".*",                                                "plasterboard", 1),  # generic default (painted drywall)
]


def resolve(name: str):
    """(p2040_key or None, class_id or None). First matching rule wins."""
    low = (name or "").lower()
    for pat, key, cls in RULES:
        if re.search(pat, low):
            return key, cls
    return "plasterboard", 1


def _coef(key):
    return PV.P2040[key] if key in PV.P2040 else LOCAL[key]


def eps_sigma(key, f_mhz):
    """(complex εr, σ [S/m]) for a P.2040/LOCAL key at frequency."""
    p = _coef(key)
    f_ghz = float(f_mhz) / 1000.0
    eps_re = p["a"] * f_ghz ** p["b"]
    sigma = p["c"] * f_ghz ** p["d"]
    eps_im = sigma / (2.0 * np.pi * (f_ghz * 1e9) * PV.EPS0)
    return complex(eps_re, -eps_im), float(sigma)


def props(name: str, f_mhz: float):
    """dict(key, cls, eps_r, sigma, is_metal) for a CAD material, or None if excluded."""
    key, cls = resolve(name)
    if key is None:
        return None
    eps, sigma = eps_sigma(key, f_mhz)
    return dict(name=name, key=key, cls=cls, eps_r=eps, sigma=sigma, is_metal=(key == "metal"))


def class_of(name: str) -> int:
    """Nearest 6-class id (for the compat material_grid.npy). Excluded → 0 (air)."""
    cls = resolve(name)[1]
    return 0 if cls is None else cls


def resolve_table(names, f_mhz):
    """{name: props} for a list of CAD material names at a band frequency."""
    return {n: props(n, f_mhz) for n in names}


def parse_mtl_names(mtl_path):
    return [ln.split()[1] for ln in open(mtl_path) if ln.startswith("newmtl ")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtl", required=True)
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--dump", action="store_true", help="write cad_material_map.json + print")
    a = ap.parse_args()
    from bands_v3 import get
    f = get(a.band).f_mhz
    names = parse_mtl_names(a.mtl)
    tbl = resolve_table(names, f)

    # summary by P.2040 key
    from collections import Counter
    by_key = Counter((p["key"] if p else "EXCLUDED") for p in tbl.values())
    print(f"{len(names)} CAD materials @ {a.band} ({f:.0f} MHz) → P.2040:")
    for k, n in by_key.most_common():
        if k == "EXCLUDED":
            print(f"  {k:14s} {n:4d}"); continue
        eps, sig = eps_sigma(k, f)
        print(f"  {k:14s} {n:4d}   εr'={eps.real:5.2f}  σ={sig:.3g} S/m")

    if a.dump:
        out = {n: (None if p is None else dict(
            p2040=p["key"], cls=p["cls"], eps_re=round(p["eps_r"].real, 3),
            eps_im=round(p["eps_r"].imag, 4), sigma=p["sigma"], is_metal=p["is_metal"]))
            for n, p in tbl.items()}
        (HERE.parent / "assets" / "cad_material_map.json").write_text(json.dumps(
            {"band": a.band, "f_mhz": f, "materials": out}, indent=2))
        print(f"wrote {HERE.parent / 'assets' / 'cad_material_map.json'} ({len(out)} materials)")


if __name__ == "__main__":
    main()
