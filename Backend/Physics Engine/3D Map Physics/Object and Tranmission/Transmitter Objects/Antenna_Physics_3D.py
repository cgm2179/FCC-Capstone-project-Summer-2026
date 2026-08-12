#!/usr/bin/env python3
"""
Antenna_Physics_3D.py — rigid-body physics properties for the antenna
geometry built by Antenna_Type_3D.py (SIM V1 3D, Object and Transmission stage).

The antenna meshes there are thin shells/wireframes (a rod, a wire loop, a
dish surface, ...), so mass can't be read off mesh volume the way it could
for a solid CAD part — a helix's triangle-mesh volume is ~0. Instead each
antenna kind gets a reference mass at a reference characteristic size
(BASE_PHYSICS below, drawn from typical real hardware weights), scaled by
(actual_size / reference_size)**3 as the antenna is parameterized larger or
smaller, and clamped to a sane range.

The collision shape is a single axis-aligned box in the antenna's LOCAL
frame (half-extents + center offset from the mesh's own AABB) — cheap
enough for a browser physics engine (cannon-es/Rapier) to test against a
few thousand static building-shell boxes (see export_collision_3d.py) in
real time, and good enough to stop a placed antenna from phasing through
walls/floors/furniture. A per-triangle mesh collider would be more exact
but is unnecessary for "does it rest on the shelf instead of falling
through it."

physics_for(antenna_mesh) is the entry point Construct_Transmitter_3D.py
uses to attach a PhysicsBody to a built AntennaMesh.
"""
from dataclasses import dataclass

import numpy as np

MIN_MASS_KG = 0.02
MAX_MASS_KG = 60.0
MIN_HALF_EXTENT_M = 0.01   # keeps flat/thin-wire antennas from getting a degenerate zero-thickness box

# kind -> (reference mass at reference size, reference characteristic size = the
# antenna's largest local-frame AABB dimension at that mass). Masses are rough
# real-hardware ballparks, not manufacturer datasheets.
BASE_PHYSICS = {
    "omni":         dict(base_mass_kg=0.30, ref_dim_m=0.9),
    "monopole":     dict(base_mass_kg=0.15, ref_dim_m=0.3),
    "whip":         dict(base_mass_kg=0.15, ref_dim_m=0.3),
    "handheld":     dict(base_mass_kg=0.05, ref_dim_m=0.15),
    "dipole":       dict(base_mass_kg=0.20, ref_dim_m=0.5),
    "inverted_f":   dict(base_mass_kg=0.05, ref_dim_m=0.05),
    "pifa":         dict(base_mass_kg=0.05, ref_dim_m=0.05),
    "panel":        dict(base_mass_kg=3.50, ref_dim_m=0.9),
    "patch":        dict(base_mass_kg=0.10, ref_dim_m=0.05),
    "yagi":         dict(base_mass_kg=1.20, ref_dim_m=1.2),
    "quad":         dict(base_mass_kg=1.00, ref_dim_m=1.0),
    "lpda":         dict(base_mass_kg=2.00, ref_dim_m=1.5),
    "phased_array": dict(base_mass_kg=5.00, ref_dim_m=0.5),
    "discone":      dict(base_mass_kg=1.50, ref_dim_m=0.5),
    "dish":         dict(base_mass_kg=4.00, ref_dim_m=0.6),
    "horn":         dict(base_mass_kg=1.50, ref_dim_m=0.3),
    "loop":         dict(base_mass_kg=0.30, ref_dim_m=0.3),
    "helical":      dict(base_mass_kg=0.80, ref_dim_m=0.6),
    "slot":         dict(base_mass_kg=0.50, ref_dim_m=0.2),
    "biconical":    dict(base_mass_kg=1.00, ref_dim_m=0.4),
    "fractal":      dict(base_mass_kg=0.10, ref_dim_m=0.3),
    "metamaterial": dict(base_mass_kg=0.40, ref_dim_m=0.2),
}
_DEFAULT_PHYSICS = dict(base_mass_kg=0.5, ref_dim_m=0.5)


@dataclass
class PhysicsBody:
    """A single-box rigid body for a placed antenna: mass in kg plus a local-
    frame axis-aligned collider (half_extents_m about center_offset_m, i.e.
    before the antenna's own world position/rotation is applied — a physics
    engine attaches this box to the same rigid body driving the render mesh,
    so it rotates/translates with it automatically)."""
    mass_kg: float
    half_extents_m: np.ndarray      # (3,) float32, local frame
    center_offset_m: np.ndarray     # (3,) float32, local frame
    restitution: float = 0.15
    friction: float = 0.6

    def to_web_dict(self):
        return dict(
            mass_kg=round(float(self.mass_kg), 4),
            collider="box",
            half_extents_m=np.asarray(self.half_extents_m, np.float32).round(5).tolist(),
            center_offset_m=np.asarray(self.center_offset_m, np.float32).round(5).tolist(),
            restitution=self.restitution,
            friction=self.friction,
        )


def physics_for(antenna_mesh, restitution=0.15, friction=0.6):
    """Derive a PhysicsBody from a built AntennaMesh (Antenna_Type_3D.py):
    mass scaled off the kind's reference mass/size, collider box sized to
    the mesh's own local-frame bounding box (so it hugs whatever the user
    actually parameterized, not just the reference size)."""
    defaults = BASE_PHYSICS.get(antenna_mesh.kind, _DEFAULT_PHYSICS)
    base_mass_kg, ref_dim_m = defaults["base_mass_kg"], defaults["ref_dim_m"]

    v = antenna_mesh.vertices
    lo, hi = v.min(axis=0), v.max(axis=0)
    span = hi - lo
    half_extents = np.maximum(span / 2.0, MIN_HALF_EXTENT_M).astype(np.float32)
    center_offset = ((hi + lo) / 2.0).astype(np.float32)

    max_dim = float(span.max())
    ratio = (max_dim / ref_dim_m) if ref_dim_m > 0 else 1.0
    mass_kg = float(np.clip(base_mass_kg * ratio ** 3, MIN_MASS_KG, MAX_MASS_KG))

    return PhysicsBody(mass_kg=mass_kg, half_extents_m=half_extents, center_offset_m=center_offset,
                        restitution=restitution, friction=friction)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from Antenna_Type_3D import ANTENNA_TYPES, OmniAntenna, YagiAntenna, ParabolicAntenna

    for name, ant in [
        ("omni", OmniAntenna.create(length_m=0.9, radius_m=0.02)),
        ("yagi", YagiAntenna.create(boom_length_m=1.2, element_lengths_m=[0.5, 0.48, 0.44],
                                     element_spacings_m=[0.0, 0.3, 0.6])),
        ("dish", ParabolicAntenna.create(diameter_m=0.6, focal_length_m=0.25)),
    ]:
        body = physics_for(ant)
        assert body.mass_kg > 0 and np.isfinite(body.mass_kg)
        assert (body.half_extents_m > 0).all()
        print(f"{name:6s} mass_kg={body.mass_kg:7.3f} half_extents_m={np.round(body.half_extents_m, 3).tolist()} "
              f"center_offset_m={np.round(body.center_offset_m, 3).tolist()}")

    missing = set(ANTENNA_TYPES) - set(BASE_PHYSICS)
    assert not missing, f"BASE_PHYSICS missing entries for: {missing}"
    print("Antenna_Physics_3D smoke test OK")
