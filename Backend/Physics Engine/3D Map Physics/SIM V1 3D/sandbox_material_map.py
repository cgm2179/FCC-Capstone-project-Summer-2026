#!/usr/bin/env python3
"""
Material classification for Sandbox_Version_3D_Simulation_1.obj (M0.4 re-voxelization).

The current voxel scene collapses the building into one class because voxelize.py's
obj_material_map ends with a `".*" -> class 1` catch-all: concrete is 0.08% of voxels,
furniture 0.02%, drywall 23%. Since material fidelity (not solver quality) is the binding
error term for RF, this module replaces that guess with rules derived from the model's
actual 253 `newmtl` names.

The Sandbox model is the SAME 7th floor as the current grid -- same coordinate frame
(origin_units [-61.4, -8.25, -1875.1] vs Sandbox mins [-45.2, -8.2, -1882.6]), spanning
77.8 x 4.6 x 39.1 m against the current 78.6 x 3.3 x 35.4 m -- but with 2,720,316
vertices instead of 5,554, and it is furnished.

Classes (manifest_3d.json):
    0 air · 1 drywall_partition · 2 concrete_masonry · 3 core_service_area (barrier)
    4 furniture_clutter · 5 exterior_glass

Rules are ordered; FIRST match wins. Run `python sandbox_material_map.py` to print a
coverage report against the real .mtl and confirm nothing important lands in the fallback.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

AIR, DRYWALL, CONCRETE, CORE, FURNITURE, GLASS = range(6)

CLASS_NAMES = {AIR: "air", DRYWALL: "drywall_partition", CONCRETE: "concrete_masonry",
               CORE: "core_service_area", FURNITURE: "furniture_clutter",
               GLASS: "exterior_glass"}

# --------------------------------------------------------------------------- exclusions
# People, personal effects and desk props are NOT building materials. They are transient
# clutter that happened to be in the CAD scene; baking them into a static RF grid would
# model one moment's furniture arrangement as permanent structure.
EXCLUDE = [
    r"^Lisanne",           # a posed human figure (skin, hair, boots, brush, watch, necklace)
    r"^Ty_",               # a second human figure
    r"^Sophie",            # a third
    r"keyboard", r"blackmouse", r"mouse\d*$",
    r"starbucks", r"^PEPE$", r"^google$",
    r"^CP_[AB]$",          # coffee cups
    r"Tides_Pool",         # decorative artwork
]

# --------------------------------------------------------------------------- rules
# (regex, class, note). First match wins, so specific patterns precede generic ones.
RULES: list[tuple[str, int, str]] = [
    # ---- glass / glazing -> class 5 (exterior_glass) -----------------------------
    (r"low[_ ]?e[_ ]?glass",                 GLASS,     "low-emissivity glazing (resistive sheet, ~20 dB)"),
    (r"glass[_ ]?door|door.*glass",          GLASS,     "interior glass door"),
    (r"translucent[_ ]?glass|gray[_ ]?glass", GLASS,    "tinted / translucent glazing"),
    (r"translucent[_ ]?block|glass[_ ]?block", GLASS,   "glass block partition"),
    (r"glass|glazing|window",                GLASS,     "generic glazing"),

    # ---- metal -> class 3 (core/barrier): high sigma, effectively opaque ---------
    (r"stainless|inox|steel|aluminum|alu\d|aluminium", CORE, "structural / appliance metal"),
    (r"metal[_ ]?(corrogated|corrugated|brushed|steel|shiny)", CORE, "sheet metal"),
    (r"brushed[_ ]?metal|polished[_ ]?alu|platinum|chrome",    CORE, "metal finish"),
    (r"^HMI[_ ]?(Polished|Brushed)",         CORE,      "HMI metal finish (dominant face count)"),
    (r"fencing|diamond[_ ]?mesh|wire[_ ]?mesh", CORE,   "metal mesh (screens RF)"),
    (r"metal|appliance",                     CORE,      "residual metal / appliance"),

    # ---- concrete / masonry / structural core -> class 2 -------------------------
    (r"concrete|aggregate|masonry|colonne|basi_",  CONCRETE, "structural concrete"),
    (r"stone|granite|marble|terrazzo|paver",       CONCRETE, "stone / masonry finish"),
    (r"blacktop|asphalt",                          CONCRETE, "exterior hardscape"),

    # ---- furniture / clutter -> class 4 (Beer-Lambert per-metre, dilute) ---------
    (r"cubicle|workstation|partition[_ ]?panel",   FURNITURE, "office cubicle system"),
    (r"leather|vinyl|fabric|upholster|textile",    FURNITURE, "seating"),
    (r"wood|veneer|bamboo|walnut|oak|laminate|formica|melamine", FURNITURE, "casework / worksurface"),
    (r"carpet|rug",                                FURNITURE, "floor covering (thin, low loss)"),
    (r"cardboard|paper|book|plant|foliage",        FURNITURE, "loose clutter"),
    (r"ceramic|porcelain",                         FURNITURE, "fixtures"),
    (r"^HMI__",                                    FURNITURE, "HMI furniture (vinyl/black trim)"),

    # ---- painted partition surfaces -> class 1 (drywall) -------------------------
    # RAL 9016 is 'traffic white', the standard interior paint code; the pale named
    # colours are wall/ceiling paint rather than objects.
    (r"ral[_ ]?9016|whitesmoke|white[_ ]?smoke", DRYWALL, "painted partition (RAL 9016)"),
    (r"^_?(beige|ivory|charcoal|tomato)_?\d*$",  DRYWALL, "painted partition"),
    (r"gypsum|drywall|plaster|sheetrock",        DRYWALL, "explicit partition"),
    (r"paint|gloss.*gray|darkgray|_black_\d*$",  DRYWALL, "painted surface"),
    (r"bukva|pages|natur_eg",                    FURNITURE, "casework / books"),
]

# Untextured/auto-named surfaces the exporter emitted (Color_XXX, FrontColor, _auto_N,
# Material_N, M_0137_*, bare _N). These are overwhelmingly wall and ceiling planes in a
# SketchUp export, so they fall back to drywall -- the same assumption the old catch-all
# made, but now it is the LAST resort rather than the FIRST, and it is reported.
FALLBACK = DRYWALL
FALLBACK_PATTERNS = [r"^color", r"^frontcolor", r"^_?auto", r"^material_?\d*$",
                     r"^M_\d+", r"^_\d+$", r"^_?color_"]

_EXCLUDE_RE = [re.compile(p, re.I) for p in EXCLUDE]
_RULES_RE = [(re.compile(p, re.I), c, n) for p, c, n in RULES]
_FALLBACK_RE = [re.compile(p, re.I) for p in FALLBACK_PATTERNS]


def classify(name: str) -> tuple[int | None, str]:
    """Map an OBJ material name to a class id. Returns (class_id | None, reason).

    None means 'exclude this geometry entirely' (people, props).
    """
    for rx in _EXCLUDE_RE:
        if rx.search(name):
            return None, "excluded (person/prop, not building material)"
    for rx, cls, note in _RULES_RE:
        if rx.search(name):
            return cls, note
    for rx in _FALLBACK_RE:
        if rx.search(name):
            return FALLBACK, "fallback: untextured surface -> partition"
    return FALLBACK, "fallback: unmatched"


def rules_for_manifest() -> list[dict]:
    """Emit the ordered rule list in manifest_3d.json's obj_material_map form."""
    out = [{"pattern": p, "class": -1, "note": "EXCLUDE " + n.split("(")[0].strip()}
           for p, n in ((p, "person/prop") for p in EXCLUDE)]
    out += [{"pattern": p, "class": c, "note": n} for p, c, n in RULES]
    out += [{"pattern": p, "class": FALLBACK, "note": "fallback: untextured surface"}
            for p in FALLBACK_PATTERNS]
    out.append({"pattern": ".*", "class": FALLBACK, "note": "final fallback"})
    return out


# --------------------------------------------------------------------------- report
def report(mtl_path: Path, obj_path: Path | None = None) -> int:
    names = [ln.split(None, 1)[1].strip()
             for ln in mtl_path.read_text(errors="replace").splitlines()
             if ln.startswith("newmtl ")]
    faces = Counter()
    if obj_path and obj_path.exists():
        cur = None
        with obj_path.open(errors="replace") as fh:
            for ln in fh:
                if ln.startswith("usemtl "):
                    cur = ln.split(None, 1)[1].strip()
                elif ln.startswith("f ") and cur:
                    faces[cur] += 1

    by_class: dict[object, list[str]] = {}
    face_by_class: Counter = Counter()
    for n in names:
        cls, _ = classify(n)
        by_class.setdefault(cls, []).append(n)
        face_by_class[cls] += faces.get(n, 0)

    tot_f = sum(faces.values()) or 1
    print(f"materials: {len(names)}   faces: {sum(faces.values()):,}\n")
    print(f"{'class':<22} {'#mats':>6} {'#faces':>12} {'face %':>8}")
    print("-" * 52)
    for cls in sorted(by_class, key=lambda c: (c is None, c if c is not None else -1)):
        label = "EXCLUDED" if cls is None else f"{cls} {CLASS_NAMES[cls]}"
        print(f"{label:<22} {len(by_class[cls]):>6} {face_by_class[cls]:>12,} "
              f"{100*face_by_class[cls]/tot_f:>7.2f}%")

    fb = [n for n in names if classify(n)[1].startswith("fallback: unmatched")]
    print(f"\nunmatched -> fallback: {len(fb)}")
    for n in sorted(fb)[:15]:
        print(f"   {n}  ({faces.get(n,0):,} faces)")
    if len(fb) > 15:
        print(f"   ... and {len(fb)-15} more")
    return 0


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3] / "Sandbox_Version_3D_Simulation_1.obj"
    mtls = sorted(root.glob("*.mtl")) if root.exists() else []
    if not mtls:
        print(f"no .mtl under {root}", file=sys.stderr)
        raise SystemExit(1)
    objs = sorted(root.glob("*.obj"))
    raise SystemExit(report(mtls[0], objs[0] if objs else None))
