"""
Import bootstrap for the mechanism modules.

The mechanism modules live under `Wave Behavior/Enivronmental Interaction/` but depend on
`SIM V1 3D/` (engine_3d, physics_3d, contracts) which in turn path-searches for the 2D
`physics_v2`. This reproduces that search once so each module can simply:

    from _bootstrap import engine_3d, physics_3d, contracts
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WAVE = HERE.parent                                   # .../Wave Behavior
PHYS3D = WAVE.parent                                 # .../3D Map Physics
ROOT = next((p for p in (PHYS3D, *PHYS3D.parents) if (p / "Data").is_dir()), PHYS3D.parent)

SIM3D = PHYS3D / "SIM V1 3D"
SIM2D = ROOT / "Physics Engine" / "2D" / "SIM"

for _p in (SIM3D, SIM2D, HERE, WAVE / "Wave Generation"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import contracts            # noqa: E402
import engine_3d           # noqa: E402
import physics_3d          # noqa: E402

__all__ = ["contracts", "engine_3d", "physics_3d", "SIM3D", "SIM2D", "ROOT"]
