#!/usr/bin/env python3
"""
bootstrap.py — sys.path shim + canonical scene paths for the SIM V2 3-D stack.

SIM V2 does NOT reimplement the physics. It REUSES, in place:
  * the full-wave engine + JAX hot path   (2D/SIM V3/: fullwave2d, fw_fdtd_jax, bands_v3,
                                           nearfield, antenna_patterns)
  * the base scalar-wave sandbox          (3D Map Physics/Wave Behavior/Wave Generation/:
                                           Time_Domain_Physics, Spatial_Physics)
  * the 3-D geometry + analytic engine     (SIM V1 3D/: physics_3d, engine_3d, dataset_3d,
                                           voxelize_city, manifest_3d.json, material_grid.npy)
  * the validated EM core                  (2D/SIM/physics_v2.py)
  * outdoor Motley-Keenan                  (2D/STEP_2/)
  * 3-D antenna geometry                   (3D Map Physics/Object and Tranmission/
                                           Transmitter Objects/Antenna_Type_3D.py)

Those live in folders with spaces and import each other by BARE name, so the only
clean way to reach them is to prepend their dirs to sys.path. Import this FIRST in
every SIM V2 module, then `import fullwave2d`, `import physics_3d`, etc. resolve.

This is the SIM-V2 superset of `2D/SIM V3/_bootstrap.py` — it additionally puts
SIM V3 itself and the 3-D antenna-geometry dir on the path.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent            # .../3D Map Physics/SIM V2
_PHYS = _HERE.parent.parent                        # .../Physics Engine

SIMV3    = _PHYS / "2D" / "SIM V3"                  # engine + JAX hot path + bands
WAVE_GEN = _PHYS / "3D Map Physics" / "Wave Behavior" / "Wave Generation"
SIM3D    = _PHYS / "3D Map Physics" / "SIM V1 3D"   # 3-D geometry + analytic + scene
SIM2D    = _PHYS / "2D" / "SIM"                     # physics_v2 EM core
STEP2    = _PHYS / "2D" / "STEP_2"                  # 2-D Motley-Keenan (outdoor far field)
ANTENNA_3D = _PHYS / "3D Map Physics" / "Object and Tranmission" / "Transmitter Objects"

# SIM V2's OWN dir must win over SIM V3 for same-named modules (fw_dataset3d,
# fw_field3d, fw_field3d_floor, city_georef, fw_bs_catalog, fw_export): prepend it.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
# engine dirs are FALLBACK (appended) so the local SIM V2 ports take precedence;
# engine-internal bare imports (fullwave2d→bands_v3, _bootstrap, physics_3d, …) still resolve.
for _d in (SIMV3, WAVE_GEN, SIM3D, SIM2D, STEP2, ANTENNA_3D):
    _s = str(_d)
    if _d.is_dir() and _s not in sys.path:
        sys.path.append(_s)

# canonical indoor scene (SIM V1 3D/voxelize.py output)
SCENE_DIR     = SIM3D
MATERIAL_GRID = SIM3D / "material_grid.npy"
VALID_TX_MASK = SIM3D / "valid_tx_mask.npy"
MANIFEST      = SIM3D / "manifest_3d.json"
REGISTRATION  = SIM3D / "registration_3d.json"

# outdoor city scene (voxelize_city.py output; large, gitignored)
CITY_DIR = SIM3D / "city" / "NoMa_DC_buildings"


def require(path: Path) -> Path:
    """Fail loud and actionable if a reused engine / scene file is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"SIM V2 could not find {path}. Expected the shared engine/scene from "
            f"SIM V3 / SIM V1 3D. Check the repo layout or re-run voxelize.py."
        )
    return Path(path)


if __name__ == "__main__":
    print("SIM V2 bootstrap — paths on sys.path:")
    for d in (SIMV3, WAVE_GEN, SIM3D, SIM2D, STEP2, ANTENNA_3D):
        print(f"  [{'ok' if d.is_dir() else 'MISSING'}] {d}")
    print("scene:", "ok" if MANIFEST.exists() else "MISSING", MANIFEST)
