#!/usr/bin/env python3
"""
Construct_Transmitter_3D.py — places a built antenna (Antenna_Type_3D.py) plus
its physics body (Antenna_Physics_3D.py) into the scene as one Transmitter3D
object, and turns a JSON-style list of transmitter specs into the render-ready
payload the SIM V1 3D web sandbox loads (see antenna_sandbox.html).

World-space convention (matches manifest_3d.json: "axes": ["X","Y_up","Z"]):
    position_m = (voxel_i, voxel_j, voxel_k) * cell_size_m
i.e. meters measured from the material_grid.npy corner (0,0,0), Y up. That is
the same corner export_collision_3d.py's merged building-shell boxes use, so a
transmitter placed at position_m sits in the same coordinate frame as the
walls/floor it must not phase through.

A transmitter spec is a flat dict:
    {"id": "tx1", "kind": "panel", "width_m": 0.3, "height_m": 0.9,
     "position_m": [3.0, 2.5, 4.0], "azimuth_deg": 90.0}
"kind"/"id"/"position_m" are pulled out here; everything else is forwarded to
build_antenna() (i.e. it must match that antenna kind's create() signature —
azimuth_deg/tilt_deg/roll_deg included).
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from Antenna_Type_3D import ANTENNA_TYPES, AntennaMesh, build_antenna
from Antenna_Physics_3D import PhysicsBody, physics_for

HERE = Path(__file__).resolve().parent


@dataclass
class Transmitter3D:
    id: str
    antenna: AntennaMesh
    physics: PhysicsBody

    def to_web_dict(self):
        d = self.antenna.to_web_dict()
        d["id"] = self.id
        d["physics"] = self.physics.to_web_dict()
        return d


def build_transmitter(spec: dict, default_id=None) -> Transmitter3D:
    """Build one Transmitter3D from a flat spec dict. "position_m" (preferred)
    or "position" is forwarded to build_antenna() as `position`; every other
    key besides "id"/"kind" is forwarded as-is (antenna-specific params +
    azimuth_deg/tilt_deg/roll_deg)."""
    if "kind" not in spec:
        raise ValueError(f"transmitter spec missing 'kind'; choices: {sorted(ANTENNA_TYPES)}")
    kwargs = {k: v for k, v in spec.items() if k not in ("id", "kind", "position_m", "position")}
    position = spec.get("position_m", spec.get("position", (0.0, 0.0, 0.0)))
    antenna = build_antenna(spec["kind"], position=position, **kwargs)
    physics = physics_for(antenna)
    tx_id = spec.get("id", default_id or f"{spec['kind']}_{id(antenna):x}")
    return Transmitter3D(id=tx_id, antenna=antenna, physics=physics)


def build_transmitters(specs) -> list:
    """Build a whole site's worth of transmitters from a list of spec dicts."""
    return [build_transmitter(spec, default_id=f"tx{i}") for i, spec in enumerate(specs)]


def write_transmitters_json(specs, out_path):
    """Build every spec and write the render-ready list as plain JSON (for
    Python-side consumers, e.g. feeding placed-Tx positions into run_one_calc.py
    /engine_3d.py). The browser sandbox instead gets a `window.X = [...]`-wrapped
    copy from export_antenna_catalog_3d.py, which loads over file:// without a
    server."""
    txs = build_transmitters(specs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([t.to_web_dict() for t in txs], indent=2))
    return txs


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("specs_json", nargs="?", help="JSON file with a list of transmitter specs")
    ap.add_argument("-o", "--out", default=str(HERE / "transmitters_3d.json"))
    args = ap.parse_args()

    if args.specs_json:
        specs = json.loads(Path(args.specs_json).read_text())
    else:
        # a small default scene: one of each of a handful of common indoor Tx types
        specs = [
            dict(id="ceiling_ap", kind="omni", length_m=0.15, radius_m=0.01,
                 position_m=[10.0, 2.7, 15.0]),
            dict(id="wall_panel", kind="panel", width_m=0.3, height_m=0.9,
                 position_m=[1.0, 2.0, 5.0], azimuth_deg=90.0, tilt_deg=6.0),
            dict(id="hallway_patch", kind="patch", patch_width_m=0.05, patch_height_m=0.05,
                 position_m=[20.0, 2.5, 8.0]),
        ]

    txs = write_transmitters_json(specs, args.out)
    for t in txs:
        bore = t.antenna.boresight_world()
        print(f"{t.id:14s} kind={t.antenna.kind:10s} mass_kg={t.physics.mass_kg:6.2f} "
              f"pos_m={np.round(t.antenna.position, 2).tolist()} boresight={np.round(bore, 2).tolist()}")
    print(f"wrote {args.out} ({len(txs)} transmitter(s))")


if __name__ == "__main__":
    _cli()
