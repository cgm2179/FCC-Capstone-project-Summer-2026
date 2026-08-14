#!/usr/bin/env python3
"""
_bootstrap.py — put SIM V1.5's role subfolders + SIM V1 3D on sys.path.

SIM V1.5 is a thin parameterized layer over SIM V1 3D's SceneV3. Its four
modules were bucketed into physics/ (engine, pathloss_models) and helpers/
(link_budget, pathloss_config); this re-exposes them by bare name so the
existing intra-folder imports keep working, and adds the sibling SIM V1 3D
folder (+ its buckets) so `from engine_3d import load_scene` resolves.

Import this FIRST in engine.py, before the `import link_budget` etc. lines.
Compat symlinks at the folder root keep external, manual-sys.path consumers
(e.g. Cross_Validation/pathloss_3d/predict_o2i_3d.py) working unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent              # .../3D Map Physics/SIM V1.5 3D
_SIM3D = _HERE.parent / "SIM V1 3D"                   # sibling engine (engine_3d, …)


def _add_tree(d: Path) -> None:
    """Add d and its immediate code subdirs (physics/ helpers/ …) to sys.path."""
    if not d.is_dir():
        return
    subs = [d] + [c for c in sorted(d.iterdir())
                  if c.is_dir() and not c.name.startswith((".", "_"))
                  and any(f.suffix == ".py" for f in c.iterdir())]
    for x in subs:
        s = str(x)
        if s not in sys.path:
            sys.path.insert(0, s)


for _d in (_SIM3D, _HERE):
    _add_tree(_d)
