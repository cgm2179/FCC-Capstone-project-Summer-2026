#!/usr/bin/env python3
"""
Antenna_Type_3D.py — parametric 3-D polygon geometry for transmitter antenna
structures (SIM V1 3D, Object and Transmission stage).

Each antenna type is a small dataclass built from user-supplied physical
dimensions (length, radius, boom/element layout, dish diameter, ...) plus a
world placement (position + azimuth/tilt/roll). build() returns a triangle
mesh (vertices, faces) in the antenna's LOCAL frame:

    +Y = up (rod / mast axis for omnidirectional wire types)  } matches manifest_3d.json
    +Z = boresight (main radiation axis for directional types)  } "axes": ["X","Y_up","Z"]
    +X = right (panel width / element axis)

world_vertices() applies R = Ry(azimuth) @ Rx(tilt) @ Rz(roll) then translates
by position, so the mesh is render-ready (e.g. a THREE.js BufferGeometry via
to_web_dict()) and boresight_world() gives the pointing vector for later RF
pattern modules (Reflection_3D.py, Diffraction_3D.py, ...) to consume.

Antenna families covered (ANTENNA_TYPES keys in parens):
    Monopole / whip / handheld device antenna    ("monopole", "whip", "handheld")
    Center-fed dipole (vertical or horizontal via tilt_deg/roll_deg)  ("dipole")
    Inverted-F / PIFA                            ("inverted_f", "pifa")
    Panel / sector                               ("panel")
    Microstrip patch                             ("patch")
    Yagi-Uda                                     ("yagi")
    Quad (cubical quad / loop Yagi)               ("quad")
    Log-periodic dipole array (LPDA)             ("lpda")
    Phased array                                 ("phased_array")
    Discone                                      ("discone")
    Parabolic dish                               ("dish")
    Horn (pyramidal or conical)                  ("horn")
    Small loop                                   ("loop")
    Helical (axial mode)                         ("helical")
    Slot                                         ("slot")
    Biconical                                    ("biconical")
    Fractal (Koch monopole/dipole)                ("fractal")
    Metamaterial (split-ring-resonator array)    ("metamaterial")

This module only builds geometry — it does not place antennas in the scene
or attach RF patterns; that is Construct_Transmitter_3D.py's job, which can
use ANTENNA_TYPES / build_antenna() below to build from a JSON transmitter
spec the same way physics_3d.py consumes manifest_3d.json.
"""
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# --- local-frame primitive builders -----------------------------------------

def _cylinder(radius, length, segments=16, axis=1):
    """Capped cylinder centered at the local origin, spanning +/- length/2
    along local `axis` (0=X, 1=Y, 2=Z); the other two axes form the circle."""
    if radius <= 0 or length <= 0 or segments < 3:
        raise ValueError(f"bad cylinder params: radius={radius} length={length} segments={segments}")
    a2, a3 = [i for i in range(3) if i != axis]
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    c, s = np.cos(theta), np.sin(theta)
    half = length / 2.0
    n = segments

    verts = np.zeros((2 * n + 2, 3), np.float32)
    verts[0:n, axis] = -half
    verts[0:n, a2] = radius * c
    verts[0:n, a3] = radius * s
    verts[n:2 * n, axis] = half
    verts[n:2 * n, a2] = radius * c
    verts[n:2 * n, a3] = radius * s
    bottom, top = 2 * n, 2 * n + 1
    verts[bottom, axis] = -half
    verts[top, axis] = half

    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
        faces.append([bottom, j, i])
        faces.append([top, n + i, n + j])
    return verts, np.array(faces, np.int32)


def _box(width, height, depth):
    """Axis-aligned box centered at the local origin: X=width, Y=height, Z=depth."""
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError(f"bad box params: width={width} height={height} depth={depth}")
    hx, hy, hz = width / 2.0, height / 2.0, depth / 2.0
    verts = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ], np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],   # -Z
        [4, 6, 5], [4, 7, 6],   # +Z
        [0, 4, 5], [0, 5, 1],   # -Y
        [3, 2, 6], [3, 6, 7],   # +Y
        [0, 3, 7], [0, 7, 4],   # -X
        [1, 5, 6], [1, 6, 2],   # +X
    ], np.int32)
    return verts, faces


def _paraboloid(diameter, focal_length, radial_segments=24, ring_segments=12):
    """Reflector surface z = r^2/(4f) for r in [0, diameter/2], apex at the
    local origin, rim at +Z; the concave (reflecting) side faces +Z, i.e. the
    focal point / feed at (0,0,focal_length) sits on the boresight axis."""
    if diameter <= 0 or focal_length <= 0 or radial_segments < 3 or ring_segments < 1:
        raise ValueError(f"bad dish params: diameter={diameter} focal_length={focal_length}")
    R = diameter / 2.0
    theta = np.linspace(0.0, 2.0 * np.pi, radial_segments, endpoint=False)
    c, s = np.cos(theta), np.sin(theta)

    apex = np.zeros((1, 3), np.float32)
    rings = []
    for k in range(1, ring_segments + 1):
        r = R * k / ring_segments
        z = (r * r) / (4.0 * focal_length)
        rings.append(np.stack([r * c, r * s, np.full_like(c, z)], axis=1))
    verts = np.concatenate([apex] + rings, axis=0).astype(np.float32)

    faces = []
    for i in range(radial_segments):
        j = (i + 1) % radial_segments
        faces.append([0, 1 + i, 1 + j])
    for ri in range(ring_segments - 1):
        base, nbase = 1 + ri * radial_segments, 1 + (ri + 1) * radial_segments
        for i in range(radial_segments):
            j = (i + 1) % radial_segments
            a, b, c2, d2 = base + i, base + j, nbase + j, nbase + i
            faces.append([a, b, c2])
            faces.append([a, c2, d2])
    return verts, np.array(faces, np.int32)


def _frustum(radius0, radius1, length, segments=24, axis=2, cap_start=False, cap_end=False):
    """Conical frustum: circle of `radius0` at -length/2, `radius1` at +length/2
    along local `axis`. Degenerates to a cone when either radius is 0. Side
    wall only by default (hollow, e.g. a horn flare); cap_start/cap_end add a
    fan cap at that end (e.g. the wide rim of a discone's cone)."""
    if length <= 0 or segments < 3 or radius0 < 0 or radius1 < 0 or (radius0 == 0 and radius1 == 0):
        raise ValueError(f"bad frustum params: radius0={radius0} radius1={radius1} length={length}")
    a2, a3 = [i for i in range(3) if i != axis]
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    c, s = np.cos(theta), np.sin(theta)
    half = length / 2.0
    n = segments

    ring0 = np.zeros((n, 3), np.float32)
    ring0[:, axis] = -half
    ring0[:, a2] = radius0 * c
    ring0[:, a3] = radius0 * s
    ring1 = np.zeros((n, 3), np.float32)
    ring1[:, axis] = half
    ring1[:, a2] = radius1 * c
    ring1[:, a3] = radius1 * s

    verts = [ring0, ring1]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])

    if cap_start:
        idx = sum(v.shape[0] for v in verts)
        center = np.zeros((1, 3), np.float32)
        center[0, axis] = -half
        verts.append(center)
        for i in range(n):
            j = (i + 1) % n
            faces.append([idx, j, i])
    if cap_end:
        idx = sum(v.shape[0] for v in verts)
        center = np.zeros((1, 3), np.float32)
        center[0, axis] = half
        verts.append(center)
        for i in range(n):
            j = (i + 1) % n
            faces.append([idx, n + i, n + j])

    return np.concatenate(verts, axis=0), np.array(faces, np.int32)


def _rect_frustum(w0, h0, w1, h1, length):
    """Open rectangular frustum along local Z: w0*h0 cross-section (throat) at
    -length/2, w1*h1 (aperture) at +length/2 — pyramidal horn flare, side
    walls only, boresight (+Z) points out of the aperture."""
    if length <= 0 or w0 <= 0 or h0 <= 0 or w1 <= 0 or h1 <= 0:
        raise ValueError(f"bad pyramidal-frustum params: w0={w0} h0={h0} w1={w1} h1={h1} length={length}")
    half = length / 2.0
    hx0, hy0, hx1, hy1 = w0 / 2.0, h0 / 2.0, w1 / 2.0, h1 / 2.0
    verts = np.array([
        [-hx0, -hy0, -half], [hx0, -hy0, -half], [hx0, hy0, -half], [-hx0, hy0, -half],
        [-hx1, -hy1, half], [hx1, -hy1, half], [hx1, hy1, half], [-hx1, hy1, half],
    ], np.float32)
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        faces.append([i, j, 4 + j])
        faces.append([i, 4 + j, 4 + i])
    return verts, np.array(faces, np.int32)


def _disc(radius, segments=24, axis=1):
    """Flat double-sided circular disc lying in the plane perpendicular to
    `axis`, centered at the local origin (ground planes, discone plate)."""
    if radius <= 0 or segments < 3:
        raise ValueError(f"bad disc params: radius={radius} segments={segments}")
    a2, a3 = [i for i in range(3) if i != axis]
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    c, s = np.cos(theta), np.sin(theta)
    n = segments
    verts = np.zeros((n + 1, 3), np.float32)
    verts[0:n, a2] = radius * c
    verts[0:n, a3] = radius * s
    center = n
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([center, i, j])
        faces.append([center, j, i])  # opposite winding so both faces render
    return verts, np.array(faces, np.int32)


def _torus(major_radius, minor_radius, major_segments=24, minor_segments=8, axis=2):
    """Ring/loop primitive (torus) lying in the plane perpendicular to
    `axis` — a small loop antenna, or a wound-coil ground-plane ring."""
    if major_radius <= 0 or minor_radius <= 0 or major_segments < 3 or minor_segments < 3:
        raise ValueError(f"bad torus params: major_radius={major_radius} minor_radius={minor_radius}")
    a2, a3 = [i for i in range(3) if i != axis]
    phi = np.linspace(0.0, 2.0 * np.pi, major_segments, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, minor_segments, endpoint=False)
    PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
    radial = major_radius + minor_radius * np.cos(THETA)

    verts = np.zeros((major_segments * minor_segments, 3), np.float32)
    verts[:, a2] = (radial * np.cos(PHI)).ravel()
    verts[:, a3] = (radial * np.sin(PHI)).ravel()
    verts[:, axis] = (minor_radius * np.sin(THETA)).ravel()

    faces = []
    m = minor_segments
    for i in range(major_segments):
        i2 = (i + 1) % major_segments
        for j in range(minor_segments):
            j2 = (j + 1) % minor_segments
            a, b, c2, d2 = i * m + j, i2 * m + j, i2 * m + j2, i * m + j2
            faces.append([a, b, c2])
            faces.append([a, c2, d2])
    return verts, np.array(faces, np.int32)


def _cylinder_between(p0, p1, radius, segments=8):
    """Open-ended cylindrical tube from point p0 to p1 in any orientation —
    building block for swept-wire antenna elements."""
    p0 = np.asarray(p0, np.float32)
    p1 = np.asarray(p1, np.float32)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        raise ValueError("zero-length wire segment")
    axis_dir = d / length
    ref = np.array([0.0, 0.0, 1.0], np.float32) if abs(axis_dir[2]) < 0.9 else np.array([1.0, 0.0, 0.0], np.float32)
    u = np.cross(axis_dir, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis_dir, u)

    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    c, s = np.cos(theta), np.sin(theta)
    offsets = radius * (np.outer(c, u) + np.outer(s, v))
    ring0, ring1 = p0 + offsets, p1 + offsets
    verts = np.concatenate([ring0, ring1], axis=0).astype(np.float32)

    n = segments
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    return verts, np.array(faces, np.int32)


def _wire_path(points, radius, segments=8, closed=False):
    """Tube swept through an ordered list of 3-D points — bent-wire antenna
    elements (Inverted-F arm, helix, fractal wire, quad/SRR loop)."""
    pts = [np.asarray(p, np.float32) for p in points]
    if len(pts) < 2:
        raise ValueError("a wire path needs at least 2 points")
    segs = [_cylinder_between(pts[i], pts[i + 1], radius, segments) for i in range(len(pts) - 1)]
    if closed:
        segs.append(_cylinder_between(pts[-1], pts[0], radius, segments))
    return _combine(*segs)


def _slotted_plate(width, height, thickness, slot_width, slot_height):
    """Rectangular plate (local X=width, Y=height, Z=thickness) with a
    rectangular slot cut from its center, built as four framing bars
    (top/bottom/left/right) rather than a boolean subtraction."""
    bar_h = (height - slot_height) / 2.0
    bar_w = (width - slot_width) / 2.0
    if bar_h <= 0 or bar_w <= 0:
        raise ValueError("slot dimensions must be smaller than the plate")
    bars = [
        _translate(_box(width, bar_h, thickness), dy=(slot_height / 2.0 + bar_h / 2.0)),
        _translate(_box(width, bar_h, thickness), dy=-(slot_height / 2.0 + bar_h / 2.0)),
        _translate(_box(bar_w, slot_height, thickness), dx=(slot_width / 2.0 + bar_w / 2.0)),
        _translate(_box(bar_w, slot_height, thickness), dx=-(slot_width / 2.0 + bar_w / 2.0)),
    ]
    return _combine(*bars)


def _koch_points_2d(p0, p1, iterations):
    """Recursive 2-D Koch-curve subdivision of segment p0->p1 (4 sub-segments
    per iteration, symmetric 60 deg bump); returns an ordered list of (u, w)
    tuples for the caller to embed into whichever 3-D plane it needs."""
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    if iterations <= 0:
        return [tuple(p0), tuple(p1)]
    d = p1 - p0
    a = p0 + d / 3.0
    b = p0 + 2.0 * d / 3.0
    angle = np.radians(60.0)
    u, w = d[0] / 3.0, d[1] / 3.0
    ru = u * np.cos(angle) - w * np.sin(angle)
    rw = u * np.sin(angle) + w * np.cos(angle)
    peak = a + np.array([ru, rw])
    left = _koch_points_2d(p0, a, iterations - 1)
    mid1 = _koch_points_2d(a, peak, iterations - 1)
    mid2 = _koch_points_2d(peak, b, iterations - 1)
    right = _koch_points_2d(b, p1, iterations - 1)
    return left[:-1] + mid1[:-1] + mid2[:-1] + right


def _combine(*meshes):
    verts, faces, offset = [], [], 0
    for v, f in meshes:
        verts.append(v)
        faces.append(f + offset)
        offset += v.shape[0]
    return np.concatenate(verts, axis=0), np.concatenate(faces, axis=0)


def _translate(mesh, dx=0.0, dy=0.0, dz=0.0):
    v, f = mesh
    return v + np.array([dx, dy, dz], np.float32), f


def _rotation_yxz(azimuth_deg, tilt_deg, roll_deg):
    """R = Ry(azimuth) @ Rx(tilt) @ Rz(roll): roll about the local boresight
    (polarization slant) first, then downtilt about local X, then azimuth
    (compass heading) about Y_up — matches manifest_3d.json's Y-up axes."""
    az, ti, ro = np.radians([azimuth_deg, tilt_deg, roll_deg])
    cz, sz = np.cos(ro), np.sin(ro)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float32)
    cx, sx = np.cos(ti), np.sin(ti)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float32)
    cy, sy = np.cos(az), np.sin(az)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    return (Ry @ Rx @ Rz).astype(np.float32)


# --- shared antenna geometry container --------------------------------------

@dataclass
class AntennaMesh:
    kind: str
    vertices: np.ndarray            # (N,3) float32, local frame
    faces: np.ndarray               # (M,3) int32 triangle indices
    params: dict
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    azimuth_deg: float = 0.0
    tilt_deg: float = 0.0
    roll_deg: float = 0.0

    def rotation_matrix(self):
        return _rotation_yxz(self.azimuth_deg, self.tilt_deg, self.roll_deg)

    def world_vertices(self):
        R = self.rotation_matrix()
        return self.vertices @ R.T + np.asarray(self.position, np.float32)

    def boresight_world(self):
        """World-space unit vector the antenna's main lobe points along (+Z local)."""
        return self.rotation_matrix() @ np.array([0.0, 0.0, 1.0], np.float32)

    def to_web_dict(self):
        """JSON-serializable render payload (a THREE.js BufferGeometry maps
        straight onto vertices/faces): mirrors the manifest-style dicts used
        elsewhere in SIM V1 3D (e.g. export_web3.py)."""
        return dict(
            kind=self.kind,
            params=self.params,
            position=np.asarray(self.position, np.float32).round(5).tolist(),
            azimuth_deg=self.azimuth_deg,
            tilt_deg=self.tilt_deg,
            roll_deg=self.roll_deg,
            vertices=self.world_vertices().round(5).tolist(),
            faces=self.faces.tolist(),
        )


def _place(kind, mesh, params, position, azimuth_deg, tilt_deg, roll_deg):
    v, f = mesh
    return AntennaMesh(
        kind=kind, vertices=v, faces=f, params=params,
        position=np.asarray(position, np.float32),
        azimuth_deg=float(azimuth_deg), tilt_deg=float(tilt_deg), roll_deg=float(roll_deg),
    )


# --- per-type antenna geometry -----------------------------------------------

class OmniAntenna:
    """Omnidirectional rod (indoor ceiling/wall whip). Mounted vertically by
    default (rod axis = local +Y), so azimuth leaves the (rotationally
    symmetric) mesh unchanged — matches real omni behavior."""

    @staticmethod
    def create(length_m, radius_m=0.02, segments=16, position=(0.0, 0.0, 0.0),
               azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        mesh = _cylinder(radius_m, length_m, segments, axis=1)
        params = dict(length_m=length_m, radius_m=radius_m, segments=segments)
        return _place("omni", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


class MonopoleAntenna:
    """Vertical monopole / whip: a single grounded element (local +Y, base at
    the ground plane, tip at +length_m), fed against an optional ground-plane
    disc. Also used for the common handheld-device whip antenna."""

    @staticmethod
    def create(length_m, radius_m=0.003, ground_plane_radius_m=None, segments=16,
               position=(0.0, 0.0, 0.0), azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        rod = _translate(_cylinder(radius_m, length_m, segments, axis=1), dy=length_m / 2.0)
        parts = [rod]
        if ground_plane_radius_m:
            parts.append(_disc(ground_plane_radius_m, axis=1))
        mesh = _combine(*parts)
        params = dict(length_m=length_m, radius_m=radius_m,
                      ground_plane_radius_m=ground_plane_radius_m, segments=segments)
        return _place("monopole", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


class DipoleAntenna:
    """Center-fed dipole: two colinear rod arms along local +/-Y separated by
    a small feed gap. tilt_deg=0 gives a vertically-polarized dipole
    ("Vertical"), tilt_deg=90 gives a horizontally-polarized one ("Horizontal")."""

    @staticmethod
    def create(total_length_m, radius_m=0.004, feed_gap_m=0.004, segments=12,
               position=(0.0, 0.0, 0.0), azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        if not (0.0 < feed_gap_m < total_length_m):
            raise ValueError(f"feed_gap_m must be in (0, total_length_m); got {feed_gap_m}")
        arm_len = (total_length_m - feed_gap_m) / 2.0
        top = _translate(_cylinder(radius_m, arm_len, segments, axis=1), dy=feed_gap_m / 2.0 + arm_len / 2.0)
        bottom = _translate(_cylinder(radius_m, arm_len, segments, axis=1), dy=-(feed_gap_m / 2.0 + arm_len / 2.0))
        mesh = _combine(top, bottom)
        params = dict(total_length_m=total_length_m, radius_m=radius_m, feed_gap_m=feed_gap_m)
        return _place("dipole", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


class InvertedFAntenna:
    """Inverted-F antenna over a ground plane: a vertical shorting post, a
    horizontal radiating arm, and a feed pin, boresight (+Z) broadside off
    the ground plane. planar=True builds the planar (PIFA) variant — a
    raised metal plate instead of a thin wire arm — over the same layout."""

    @staticmethod
    def create(ground_width_m, ground_depth_m, arm_length_m, height_m, arm_width_m=0.01,
               feed_offset_m=None, wire_radius_m=0.0015, ground_thickness_m=0.001, planar=True,
               position=(0.0, 0.0, 0.0), azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        if feed_offset_m is None:
            feed_offset_m = arm_length_m * 0.15
        short_x = -ground_width_m / 2.0
        ground = _translate(_box(ground_width_m, ground_depth_m, ground_thickness_m),
                             dz=-height_m - ground_thickness_m / 2.0)
        parts = [ground]
        if planar:
            plate = _translate(_box(arm_length_m, arm_width_m, wire_radius_m * 2.0), dx=short_x + arm_length_m / 2.0)
            wall = _translate(_box(wire_radius_m * 2.0, arm_width_m, height_m), dx=short_x, dz=-height_m / 2.0)
            parts += [plate, wall]
        else:
            arm = _wire_path([(short_x, 0.0, 0.0), (short_x + arm_length_m, 0.0, 0.0)], wire_radius_m, segments=8)
            post = _wire_path([(short_x, 0.0, -height_m), (short_x, 0.0, 0.0)], wire_radius_m, segments=8)
            parts += [arm, post]
        feed = _wire_path([(short_x + feed_offset_m, 0.0, -height_m), (short_x + feed_offset_m, 0.0, 0.0)],
                           wire_radius_m, segments=8)
        parts.append(feed)
        mesh = _combine(*parts)
        params = dict(ground_width_m=ground_width_m, ground_depth_m=ground_depth_m, arm_length_m=arm_length_m,
                      height_m=height_m, arm_width_m=arm_width_m, feed_offset_m=feed_offset_m, planar=planar)
        return _place("inverted_f", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


class PanelAntenna:
    """Flat rectangular sector/panel antenna. Boresight (+Z) points out of
    the panel face; width=X, height=Y, depth=Z (radome thickness)."""

    @staticmethod
    def create(width_m, height_m, depth_m=0.08, position=(0.0, 0.0, 0.0),
               azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        mesh = _box(width_m, height_m, depth_m)
        params = dict(width_m=width_m, height_m=height_m, depth_m=depth_m)
        return _place("panel", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


class PatchAntenna:
    """Microstrip patch: metal patch over a dielectric substrate over a
    ground plane, stacked along local +Z (boresight is broadside off the
    patch face)."""

    @staticmethod
    def create(patch_width_m, patch_height_m, substrate_thickness_m=0.0016,
               ground_width_m=None, ground_height_m=None, patch_thickness_m=0.00003,
               position=(0.0, 0.0, 0.0), azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        gw = ground_width_m or patch_width_m * 1.5
        gh = ground_height_m or patch_height_m * 1.5
        ground = _translate(_box(gw, gh, 0.001), dz=-(substrate_thickness_m / 2.0 + 0.0005))
        substrate = _box(patch_width_m * 1.05, patch_height_m * 1.05, substrate_thickness_m)
        patch = _translate(_box(patch_width_m, patch_height_m, patch_thickness_m),
                            dz=substrate_thickness_m / 2.0 + patch_thickness_m / 2.0)
        mesh = _combine(ground, substrate, patch)
        params = dict(patch_width_m=patch_width_m, patch_height_m=patch_height_m,
                      substrate_thickness_m=substrate_thickness_m, ground_width_m=gw, ground_height_m=gh)
        return _place("patch", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


class YagiAntenna:
    """Yagi-Uda: boom along the boresight (+Z), driven/director/reflector
    elements as perpendicular rods (local +X) at given spacings along the boom
    (measured from the reflector end, 0 <= spacing <= boom_length_m)."""

    @staticmethod
    def create(boom_length_m, element_lengths_m: Sequence[float], element_spacings_m: Sequence[float],
               boom_radius_m=0.01, element_radius_m=0.005, element_segments=8,
               position=(0.0, 0.0, 0.0), azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        if len(element_lengths_m) != len(element_spacings_m):
            raise ValueError("element_lengths_m and element_spacings_m must be the same length")
        if len(element_lengths_m) == 0:
            raise ValueError("a Yagi needs at least one element")
        for s in element_spacings_m:
            if not (0.0 <= s <= boom_length_m):
                raise ValueError(f"element spacing {s} outside boom_length_m [0, {boom_length_m}]")

        parts = [_cylinder(boom_radius_m, boom_length_m, 12, axis=2)]
        half_boom = boom_length_m / 2.0
        for length, spacing in zip(element_lengths_m, element_spacings_m):
            elem = _cylinder(element_radius_m, length, element_segments, axis=0)
            parts.append(_translate(elem, dz=spacing - half_boom))
        mesh = _combine(*parts)
        params = dict(boom_length_m=boom_length_m, element_lengths_m=list(element_lengths_m),
                      element_spacings_m=list(element_spacings_m), boom_radius_m=boom_radius_m,
                      element_radius_m=element_radius_m)
        return _place("yagi", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


class QuadAntenna:
    """Quad (cubical quad / loop-Yagi) array: boom along local +Z with square
    wire-loop elements (driven + optional reflector/directors) perpendicular
    to the boom at given spacings, each loop sized to element_perimeters_m/4
    per side."""

    @staticmethod
    def create(boom_length_m, element_perimeters_m: Sequence[float], element_spacings_m: Sequence[float],
               boom_radius_m=0.01, wire_radius_m=0.004,
               position=(0.0, 0.0, 0.0), azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        if len(element_perimeters_m) != len(element_spacings_m):
            raise ValueError("element_perimeters_m and element_spacings_m must be the same length")
        if len(element_perimeters_m) == 0:
            raise ValueError("a quad array needs at least one loop element")
        for sp in element_spacings_m:
            if not (0.0 <= sp <= boom_length_m):
                raise ValueError(f"element spacing {sp} outside boom_length_m [0, {boom_length_m}]")

        half_boom = boom_length_m / 2.0
        parts = [_cylinder(boom_radius_m, boom_length_m, 12, axis=2)]
        for perim, spacing in zip(element_perimeters_m, element_spacings_m):
            half = (perim / 4.0) / 2.0
            loop_pts = [(-half, -half, 0.0), (half, -half, 0.0), (half, half, 0.0), (-half, half, 0.0)]
            loop = _wire_path(loop_pts, wire_radius_m, segments=6, closed=True)
            parts.append(_translate(loop, dz=spacing - half_boom))
        mesh = _combine(*parts)
        params = dict(boom_length_m=boom_length_m, element_perimeters_m=list(element_perimeters_m),
                      element_spacings_m=list(element_spacings_m), boom_radius_m=boom_radius_m,
                      wire_radius_m=wire_radius_m)
        return _place("quad", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


class LPDAAntenna:
    """Log-periodic dipole array: two parallel boom rails along local +Z
    (feed alternates rail-to-rail, element to element) carrying dipole rods
    along local X. Pass explicit element_lengths_m/element_spacings_m, or
    longest_element_m/n_elements (auto-generated via scaling factor tau and
    spacing factor sigma — a simplified geometric generator, not a full
    log-periodic RF synthesis)."""

    @staticmethod
    def create(boom_length_m=None, element_lengths_m: Sequence[float] = None,
               element_spacings_m: Sequence[float] = None, longest_element_m=None, n_elements=None,
               tau=0.9, sigma=0.05, rail_gap_m=0.02, boom_radius_m=0.006, element_radius_m=0.003,
               element_segments=8, position=(0.0, 0.0, 0.0), azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        if element_lengths_m is None or element_spacings_m is None:
            if longest_element_m is None or n_elements is None:
                raise ValueError("provide either element_lengths_m/element_spacings_m, "
                                 "or longest_element_m/n_elements (with tau/sigma)")
            if not (0.0 < tau < 1.0):
                raise ValueError(f"tau must be in (0,1), got {tau}")
            lengths = [longest_element_m * (tau ** i) for i in range(n_elements)]
            spacings = [0.0]
            for i in range(1, n_elements):
                spacings.append(spacings[-1] + sigma * (lengths[i - 1] + lengths[i]))
            element_lengths_m, element_spacings_m = lengths, spacings
        if len(element_lengths_m) != len(element_spacings_m):
            raise ValueError("element_lengths_m and element_spacings_m must be the same length")
        if len(element_lengths_m) == 0:
            raise ValueError("an LPDA needs at least one element")

        boom_length_m = boom_length_m or (max(element_spacings_m) * 1.05)
        for sp in element_spacings_m:
            if not (0.0 <= sp <= boom_length_m):
                raise ValueError(f"element spacing {sp} outside boom_length_m [0, {boom_length_m}]")

        half_boom = boom_length_m / 2.0
        rail_a = _translate(_cylinder(boom_radius_m, boom_length_m, 10, axis=2), dy=rail_gap_m / 2.0)
        rail_b = _translate(_cylinder(boom_radius_m, boom_length_m, 10, axis=2), dy=-rail_gap_m / 2.0)
        parts = [rail_a, rail_b]
        for i, (length, spacing) in enumerate(zip(element_lengths_m, element_spacings_m)):
            elem = _cylinder(element_radius_m, length, element_segments, axis=0)
            dy = rail_gap_m / 2.0 if i % 2 == 0 else -rail_gap_m / 2.0
            parts.append(_translate(elem, dy=dy, dz=spacing - half_boom))
        mesh = _combine(*parts)
        params = dict(boom_length_m=boom_length_m, element_lengths_m=list(element_lengths_m),
                      element_spacings_m=list(element_spacings_m), tau=tau, sigma=sigma, rail_gap_m=rail_gap_m)
        return _place("lpda", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


class PhasedArrayAntenna:
    """Planar phased array: a rows x cols grid of patch-like radiating
    elements over a common backing panel, spaced element_spacing_x_m /
    element_spacing_y_m apart; boresight (+Z) is broadside off the array face."""

    @staticmethod
    def create(rows, cols, element_spacing_x_m, element_spacing_y_m,
               element_width_m=None, element_height_m=None, element_depth_m=0.01, backing_depth_m=0.02,
               position=(0.0, 0.0, 0.0), azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        if rows < 1 or cols < 1:
            raise ValueError(f"rows/cols must be >= 1, got rows={rows} cols={cols}")
        ew = element_width_m or element_spacing_x_m * 0.6
        eh = element_height_m or element_spacing_y_m * 0.6
        panel_w = cols * element_spacing_x_m
        panel_h = rows * element_spacing_y_m
        backing = _translate(_box(panel_w, panel_h, backing_depth_m), dz=-backing_depth_m / 2.0 - element_depth_m / 2.0)
        parts = [backing]
        for r in range(rows):
            y = (r - (rows - 1) / 2.0) * element_spacing_y_m
            for c in range(cols):
                x = (c - (cols - 1) / 2.0) * element_spacing_x_m
                parts.append(_translate(_box(ew, eh, element_depth_m), dx=x, dy=y))
        mesh = _combine(*parts)
        params = dict(rows=rows, cols=cols, element_spacing_x_m=element_spacing_x_m,
                      element_spacing_y_m=element_spacing_y_m, element_width_m=ew, element_height_m=eh,
                      element_depth_m=element_depth_m, backing_depth_m=backing_depth_m)
        return _place("phased_array", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


class DisconeAntenna:
    """Discone: a flat disc above the apex of a downward-flaring cone, fed
    across the small gap between them; wideband, omnidirectional around the
    local +Y (mast) axis."""

    @staticmethod
    def create(cone_base_radius_m, cone_length_m, disc_radius_m, feed_gap_m=0.02, segments=24,
               position=(0.0, 0.0, 0.0), azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        cone = _frustum(cone_base_radius_m, 0.0, cone_length_m, segments, axis=1, cap_start=True)
        cone = _translate(cone, dy=-feed_gap_m / 2.0 - cone_length_m / 2.0)
        disc = _translate(_disc(disc_radius_m, segments, axis=1), dy=feed_gap_m / 2.0)
        mesh = _combine(cone, disc)
        params = dict(cone_base_radius_m=cone_base_radius_m, cone_length_m=cone_length_m,
                      disc_radius_m=disc_radius_m, feed_gap_m=feed_gap_m)
        return _place("discone", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


class ParabolicAntenna:
    """Parabolic (dish) reflector; boresight (+Z) is the focal axis. A thin
    feed strut runs from the vertex to the focal point for visual reference."""

    @staticmethod
    def create(diameter_m, focal_length_m, feed_radius_m=0.01, radial_segments=24, ring_segments=12,
               position=(0.0, 0.0, 0.0), azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        dish = _paraboloid(diameter_m, focal_length_m, radial_segments, ring_segments)
        feed = _translate(_cylinder(feed_radius_m, focal_length_m, 8, axis=2), dz=focal_length_m / 2.0)
        mesh = _combine(dish, feed)
        params = dict(diameter_m=diameter_m, focal_length_m=focal_length_m,
                      radial_segments=radial_segments, ring_segments=ring_segments)
        return _place("dish", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


class HornAntenna:
    """Waveguide horn — pyramidal (rectangular) or conical (circular) flare —
    throat at local -Z, aperture (mouth) at local +Z = boresight."""

    @staticmethod
    def create(shape, length_m, aperture_width_m=None, aperture_height_m=None,
               throat_width_m=None, throat_height_m=None, aperture_radius_m=None, throat_radius_m=None,
               segments=24, position=(0.0, 0.0, 0.0), azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        if shape == "pyramidal":
            if None in (aperture_width_m, aperture_height_m, throat_width_m, throat_height_m):
                raise ValueError("pyramidal horn needs throat_width_m/throat_height_m/"
                                 "aperture_width_m/aperture_height_m")
            mesh = _rect_frustum(throat_width_m, throat_height_m, aperture_width_m, aperture_height_m, length_m)
            params = dict(shape=shape, length_m=length_m, throat_width_m=throat_width_m,
                          throat_height_m=throat_height_m, aperture_width_m=aperture_width_m,
                          aperture_height_m=aperture_height_m)
        elif shape == "conical":
            if None in (aperture_radius_m, throat_radius_m):
                raise ValueError("conical horn needs throat_radius_m/aperture_radius_m")
            mesh = _frustum(throat_radius_m, aperture_radius_m, length_m, segments, axis=2)
            params = dict(shape=shape, length_m=length_m, throat_radius_m=throat_radius_m,
                          aperture_radius_m=aperture_radius_m)
        else:
            raise ValueError(f"shape must be 'pyramidal' or 'conical', got {shape!r}")
        return _place("horn", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


class LoopAntenna:
    """Small circular loop antenna; the loop lies in the local X-Y plane so
    its boresight (the loop's magnetic-dipole axis) is local +Z."""

    @staticmethod
    def create(diameter_m, wire_radius_m=0.003, major_segments=24, minor_segments=8,
               position=(0.0, 0.0, 0.0), azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        mesh = _torus(diameter_m / 2.0, wire_radius_m, major_segments, minor_segments, axis=2)
        params = dict(diameter_m=diameter_m, wire_radius_m=wire_radius_m)
        return _place("loop", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


class HelicalAntenna:
    """Axial-mode helical antenna: a wire helix wound around local +Z (the
    boresight/axial direction), optionally mounted on a ground-plane disc."""

    @staticmethod
    def create(diameter_m, pitch_m, turns, wire_radius_m=0.003, ground_plane_radius_m=None,
               segments_per_turn=16, position=(0.0, 0.0, 0.0), azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        if turns <= 0 or pitch_m <= 0 or diameter_m <= 0:
            raise ValueError(f"bad helix params: diameter_m={diameter_m} pitch_m={pitch_m} turns={turns}")
        n = max(int(round(turns * segments_per_turn)), 2)
        t = np.linspace(0.0, turns, n + 1)
        R = diameter_m / 2.0
        height = pitch_m * turns
        pts = np.stack([R * np.cos(2.0 * np.pi * t), R * np.sin(2.0 * np.pi * t), pitch_m * t - height / 2.0], axis=1)
        parts = [_wire_path(pts, wire_radius_m, segments=6)]
        if ground_plane_radius_m:
            parts.append(_translate(_disc(ground_plane_radius_m, axis=2), dz=-height / 2.0))
        mesh = _combine(*parts)
        params = dict(diameter_m=diameter_m, pitch_m=pitch_m, turns=turns, wire_radius_m=wire_radius_m,
                      ground_plane_radius_m=ground_plane_radius_m)
        return _place("helical", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


class SlotAntenna:
    """Slot antenna: a conductive plate (local X-Y, facing +Z) with a
    rectangular slot cut out of it, built as four surrounding frame bars."""

    @staticmethod
    def create(plate_width_m, plate_height_m, slot_width_m, slot_height_m, thickness_m=0.002,
               position=(0.0, 0.0, 0.0), azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        mesh = _slotted_plate(plate_width_m, plate_height_m, thickness_m, slot_width_m, slot_height_m)
        params = dict(plate_width_m=plate_width_m, plate_height_m=plate_height_m,
                      slot_width_m=slot_width_m, slot_height_m=slot_height_m, thickness_m=thickness_m)
        return _place("slot", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


class BiconicalAntenna:
    """Biconical dipole: two cones joined apex-to-apex at the feed point,
    flaring outward along local +/-Y; wideband, omnidirectional pattern."""

    @staticmethod
    def create(cone_length_m, cone_base_radius_m, feed_gap_m=0.01, segments=24,
               position=(0.0, 0.0, 0.0), azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        top = _frustum(0.0, cone_base_radius_m, cone_length_m, segments, axis=1, cap_end=True)
        top = _translate(top, dy=feed_gap_m / 2.0 + cone_length_m / 2.0)
        bottom = _frustum(cone_base_radius_m, 0.0, cone_length_m, segments, axis=1, cap_start=True)
        bottom = _translate(bottom, dy=-(feed_gap_m / 2.0 + cone_length_m / 2.0))
        mesh = _combine(top, bottom)
        params = dict(cone_length_m=cone_length_m, cone_base_radius_m=cone_base_radius_m, feed_gap_m=feed_gap_m)
        return _place("biconical", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


class FractalAntenna:
    """Koch-curve fractal wire monopole/dipole — a common miniaturized-antenna
    construction where each straight arm is replaced by a self-similar
    zig-zag (iterations = fractal order); more iterations pack more wire
    length into the same overall span. Arm runs along local +Y (bump into
    local X), matching the vertical Monopole/Dipole convention."""

    @staticmethod
    def create(total_length_m, iterations=3, wire_radius_m=0.002, mode="monopole",
               ground_plane_radius_m=None, feed_gap_m=0.003,
               position=(0.0, 0.0, 0.0), azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        if mode not in ("monopole", "dipole"):
            raise ValueError(f"mode must be 'monopole' or 'dipole', got {mode!r}")
        if iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {iterations}")
        parts = []
        if mode == "monopole":
            arm2d = _koch_points_2d((0.0, 0.0), (0.0, total_length_m), iterations)
            parts.append(_wire_path([(u, w, 0.0) for u, w in arm2d], wire_radius_m, segments=6))
            if ground_plane_radius_m:
                parts.append(_disc(ground_plane_radius_m, axis=1))
        else:
            half = total_length_m / 2.0
            top2d = _koch_points_2d((0.0, feed_gap_m / 2.0), (0.0, half), iterations)
            bot2d = _koch_points_2d((0.0, -feed_gap_m / 2.0), (0.0, -half), iterations)
            parts.append(_wire_path([(u, w, 0.0) for u, w in top2d], wire_radius_m, segments=6))
            parts.append(_wire_path([(u, w, 0.0) for u, w in bot2d], wire_radius_m, segments=6))
        mesh = _combine(*parts)
        params = dict(total_length_m=total_length_m, iterations=iterations, wire_radius_m=wire_radius_m,
                      mode=mode, ground_plane_radius_m=ground_plane_radius_m, feed_gap_m=feed_gap_m)
        return _place("fractal", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


class MetamaterialAntenna:
    """Metamaterial (split-ring-resonator) unit-cell array over a backing
    substrate: a rows x cols grid of square wire loops, each with a small
    gap (the 'split'), spaced cell_size_m apart; boresight (+Z) is broadside
    off the array face."""

    @staticmethod
    def create(rows, cols, cell_size_m, ring_size_m=None, gap_m=None, wire_radius_m=0.0008,
               backing_depth_m=0.002, position=(0.0, 0.0, 0.0), azimuth_deg=0.0, tilt_deg=0.0, roll_deg=0.0):
        if rows < 1 or cols < 1:
            raise ValueError(f"rows/cols must be >= 1, got rows={rows} cols={cols}")
        ring_size_m = ring_size_m or cell_size_m * 0.7
        gap_m = gap_m or ring_size_m * 0.15
        half = ring_size_m / 2.0
        panel_w, panel_h = cols * cell_size_m, rows * cell_size_m
        backing = _translate(_box(panel_w, panel_h, backing_depth_m), dz=-backing_depth_m / 2.0)
        parts = [backing]
        loop_pts = [
            (-half + gap_m / 2.0, -half, 0.0), (half, -half, 0.0), (half, half, 0.0),
            (-half, half, 0.0), (-half, -half + gap_m / 2.0, 0.0),
        ]
        for r in range(rows):
            y = (r - (rows - 1) / 2.0) * cell_size_m
            for c in range(cols):
                x = (c - (cols - 1) / 2.0) * cell_size_m
                ring = _wire_path(loop_pts, wire_radius_m, segments=6, closed=False)
                parts.append(_translate(ring, dx=x, dy=y))
        mesh = _combine(*parts)
        params = dict(rows=rows, cols=cols, cell_size_m=cell_size_m, ring_size_m=ring_size_m,
                      gap_m=gap_m, wire_radius_m=wire_radius_m)
        return _place("metamaterial", mesh, params, position, azimuth_deg, tilt_deg, roll_deg)


ANTENNA_TYPES = {
    "omni": OmniAntenna,
    "monopole": MonopoleAntenna,
    "whip": MonopoleAntenna,
    "handheld": MonopoleAntenna,
    "dipole": DipoleAntenna,
    "inverted_f": InvertedFAntenna,
    "pifa": InvertedFAntenna,
    "panel": PanelAntenna,
    "patch": PatchAntenna,
    "yagi": YagiAntenna,
    "quad": QuadAntenna,
    "lpda": LPDAAntenna,
    "phased_array": PhasedArrayAntenna,
    "discone": DisconeAntenna,
    "dish": ParabolicAntenna,
    "horn": HornAntenna,
    "loop": LoopAntenna,
    "helical": HelicalAntenna,
    "slot": SlotAntenna,
    "biconical": BiconicalAntenna,
    "fractal": FractalAntenna,
    "metamaterial": MetamaterialAntenna,
}


def build_antenna(kind, **kwargs):
    """Factory for a transmitter spec dict, e.g. {"kind": "panel", "width_m": ...}
    — the JSON-driven entry point Construct_Transmitter_3D.py is expected to use.
    See ANTENNA_TYPES for the full list of supported kind strings."""
    try:
        cls = ANTENNA_TYPES[kind]
    except KeyError:
        raise ValueError(f"unknown antenna kind {kind!r}; choices: {sorted(ANTENNA_TYPES)}") from None
    return cls.create(**kwargs)


if __name__ == "__main__":
    # smoke test: build one of every antenna type, sanity-check the mesh + placement math
    examples = {
        "omni": OmniAntenna.create(length_m=0.9, radius_m=0.02, position=(1.0, 1.5, 2.0), azimuth_deg=40.0),
        "monopole": MonopoleAntenna.create(length_m=0.3, ground_plane_radius_m=0.15, position=(2.0, 0.0, 0.0)),
        "dipole": DipoleAntenna.create(total_length_m=0.5, tilt_deg=90.0, position=(2.5, 1.0, 0.0)),
        "inverted_f": InvertedFAntenna.create(ground_width_m=0.1, ground_depth_m=0.05, arm_length_m=0.04,
                                               height_m=0.008, planar=True, position=(0.0, 1.0, 0.0)),
        "panel": PanelAntenna.create(width_m=0.3, height_m=0.9, depth_m=0.08,
                                      position=(0.0, 2.5, 0.0), azimuth_deg=90.0, tilt_deg=6.0),
        "patch": PatchAntenna.create(patch_width_m=0.05, patch_height_m=0.05, position=(1.0, 0.5, 0.0)),
        "yagi": YagiAntenna.create(boom_length_m=1.2, element_lengths_m=[0.5, 0.48, 0.44],
                                    element_spacings_m=[0.0, 0.3, 0.6], position=(-2.0, 1.0, 0.0)),
        "quad": QuadAntenna.create(boom_length_m=1.0, element_perimeters_m=[2.0, 1.9],
                                    element_spacings_m=[0.0, 0.4], position=(-2.0, 2.0, 0.0)),
        "lpda": LPDAAntenna.create(longest_element_m=0.6, n_elements=6, tau=0.85, sigma=0.06,
                                    position=(-1.0, 1.5, -1.0)),
        "phased_array": PhasedArrayAntenna.create(rows=4, cols=4, element_spacing_x_m=0.06,
                                                    element_spacing_y_m=0.06, position=(0.0, 1.0, 1.0)),
        "discone": DisconeAntenna.create(cone_base_radius_m=0.15, cone_length_m=0.4, disc_radius_m=0.06,
                                          position=(1.5, 0.0, 1.0)),
        "dish": ParabolicAntenna.create(diameter_m=0.6, focal_length_m=0.25, position=(0.0, 3.0, -1.0), tilt_deg=-10.0),
        "horn_conical": HornAntenna.create(shape="conical", length_m=0.2, throat_radius_m=0.02,
                                            aperture_radius_m=0.12, position=(2.0, 2.0, 0.0)),
        "horn_pyramidal": HornAntenna.create(shape="pyramidal", length_m=0.2, throat_width_m=0.03,
                                              throat_height_m=0.02, aperture_width_m=0.15,
                                              aperture_height_m=0.1, position=(2.5, 2.0, 0.0)),
        "loop": LoopAntenna.create(diameter_m=0.2, position=(-0.5, 0.5, 0.0)),
        "helical": HelicalAntenna.create(diameter_m=0.1, pitch_m=0.08, turns=8, ground_plane_radius_m=0.1,
                                          position=(-0.5, 1.0, 1.0)),
        "slot": SlotAntenna.create(plate_width_m=0.2, plate_height_m=0.1, slot_width_m=0.1, slot_height_m=0.01,
                                    position=(0.0, 0.0, -1.0)),
        "biconical": BiconicalAntenna.create(cone_length_m=0.2, cone_base_radius_m=0.08, position=(3.0, 1.0, 0.0)),
        "fractal_monopole": FractalAntenna.create(total_length_m=0.3, iterations=3, mode="monopole",
                                                    ground_plane_radius_m=0.1, position=(3.0, 0.0, -1.0)),
        "fractal_dipole": FractalAntenna.create(total_length_m=0.3, iterations=2, mode="dipole",
                                                  position=(3.5, 0.0, -1.0)),
        "metamaterial": MetamaterialAntenna.create(rows=3, cols=3, cell_size_m=0.02, position=(-1.0, 0.0, 0.0)),
    }
    for name, ant in examples.items():
        wv = ant.world_vertices()
        assert np.isfinite(wv).all(), f"{name}: non-finite vertices"
        assert ant.faces.max() < ant.vertices.shape[0], f"{name}: face index out of range"
        bore = ant.boresight_world()
        assert abs(np.linalg.norm(bore) - 1.0) < 1e-4, f"{name}: boresight not unit length"
        lo, hi = wv.min(0), wv.max(0)
        print(f"{name:18s} kind={ant.kind:12s} verts={ant.vertices.shape[0]:5d} faces={ant.faces.shape[0]:5d} "
              f"bbox=({lo[0]:.2f},{lo[1]:.2f},{lo[2]:.2f})..({hi[0]:.2f},{hi[1]:.2f},{hi[2]:.2f})")

    d = examples["panel"].to_web_dict()
    assert set(d) == {"kind", "params", "position", "azimuth_deg", "tilt_deg", "roll_deg", "vertices", "faces"}
    assert build_antenna("dish", diameter_m=0.6, focal_length_m=0.25).kind == "dish"
    assert build_antenna("pifa", ground_width_m=0.1, ground_depth_m=0.05, arm_length_m=0.04, height_m=0.008).kind == "inverted_f"
    assert set(ANTENNA_TYPES) >= {
        "omni", "monopole", "whip", "handheld", "dipole", "inverted_f", "pifa", "panel", "patch",
        "yagi", "quad", "lpda", "phased_array", "discone", "dish", "horn", "loop", "helical",
        "slot", "biconical", "fractal", "metamaterial",
    }
    print("Antenna_Type_3D smoke test OK")
