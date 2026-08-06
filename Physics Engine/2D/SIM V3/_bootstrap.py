#!/usr/bin/env python3
"""
_bootstrap.py — put the reused engines on sys.path for the SIM V3 full-wave stack.

SIM V3 does NOT reimplement physics. It reuses, verbatim:
  * the scalar-wave FDTD sandbox  (Wave Behavior/Wave Generation/)
        Time_Domain_Physics.py  -> WaveSim, courant_dt, ricker, cw, sponge
        Spatial_Physics.py      -> laplacian, rigid_mask
  * the production 3-D geometry adapter  (SIM V1 3D/physics_3d.py)
        speed_field(grid, f_mhz) -> ITU-calibrated c(x) + barrier mask
  * the validated EM core  (2D/SIM/physics_v2.py)
        permittivity(material, f_mhz) -> complex eps_r  (for the loss term)

Those live in folders with spaces in the name and are imported by bare module
name (Time_Domain_Physics imports `Spatial_Physics` directly), so the only clean
way to reach them is to prepend their directories to sys.path. Import this module
first, then `import physics_3d`, `import Time_Domain_Physics`, etc.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent            # .../Physics Engine/2D/SIM V3
_PHYS = _HERE.parent.parent                        # .../Physics Engine

WAVE_GEN = _PHYS / "3D Map Physics" / "Wave Behavior" / "Wave Generation"
SIM3D = _PHYS / "3D Map Physics" / "SIM V1 3D"
SIM2D = _PHYS / "2D" / "SIM"
STEP2 = _PHYS / "2D" / "STEP_2"          # 2-D Motley-Keenan (outdoor far field)

# most-specific first so a bare `import physics_3d` resolves to the production one
for _d in (WAVE_GEN, SIM3D, SIM2D, STEP2):
    _s = str(_d)
    if _d.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)

# canonical on-disk scene the indoor engine already ships (voxelize.py output)
SCENE_DIR = SIM3D
MATERIAL_GRID = SIM3D / "material_grid.npy"
VALID_TX_MASK = SIM3D / "valid_tx_mask.npy"
MANIFEST = SIM3D / "manifest_3d.json"

# outdoor city scene (voxelize_city.py output; ~350 MB, gitignored)
CITY_DIR = SIM3D / "city" / "NoMa_DC_buildings"


def require(path: Path) -> Path:
    """Fail loud and actionable if a reused engine / scene file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"SIM V3 could not find {path}. Expected the indoor scene from "
            f"SIM V1 3D/voxelize.py. Re-run voxelize.py or check the repo layout."
        )
    return path
