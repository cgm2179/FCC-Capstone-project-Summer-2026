#!/usr/bin/env python3
"""
modes_3d.py — the four propagation modes (M3).

One registry, four scenes, one solve path. Everything downstream (exporters, the browser,
the validation stage) asks this module "what am I simulating?" instead of hardcoding the
indoor floor, so adding a mode never means forking the engine.

    1. vacuum   all air. PL == FSPL exactly. This is the V0 invariant gate made
                user-visible: if the engine cannot reproduce free space it cannot be
                trusted anywhere else, so the mode exists to be checked, not admired.
    2. indoor   the voxelized 7th floor, Tx inside. The production case.
    3. o2i      outdoor-to-indoor. The same indoor scene, but the source is a distant
                macro cell: a plane wave lights the exterior envelope, every illuminated
                facade voxel re-radiates inward through the CrossingLUT, and the facade
                itself is charged the 3GPP O2I penetration loss.
    4. outdoor  the NoMa DC city grid from voxelize_city.py — buildings are one barrier
                class, streets are air. Ground reflection and terrain diffraction are not
                new physics here: they are Reflection_3D and Diffraction_3D on a scene
                whose "walls" happen to be buildings.

WHY A REGISTRY AND NOT FOUR SCRIPTS
Modes differ in three things only — which grid, where the source is, and whether the
source is a point or a plane wave. Everything after that (mechanisms, combiner, export
format, browser) is identical. Encoding that as data keeps the difference honest and
stops mode-specific behaviour leaking into the physics modules.

THE O2I GEOMETRY IS DERIVED, NOT GUESSED
`bs_preset.bearing_deg = 135` in the 2D manifest was explicitly a placeholder for demos.
The real transmitter is known — Forte Hall rooftop, 38.901550, -77.011420 — so this module
computes the range and arrival bearing from the floor plan's own georeference
(`STEP_1/floorplan_meta.json`) instead of carrying a magic number. `forte_hall_geometry()`
returns 415.9 m and an arrival bearing of 237.0 deg, and the test asserts the derivation
rather than the constant.

usage:
    python modes_3d.py --list
    python modes_3d.py --test
    python modes_3d.py --describe o2i
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MECH_DIR = HERE.parent / "Wave Behavior" / "Enivronmental Interaction"
ROOT = HERE.parent.parent.parent
for _p in (HERE, MECH_DIR, ROOT / "Physics Engine" / "2D" / "SIM"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import engine_3d  # noqa: E402

C0 = 299_792_458.0
R_EARTH = 6_378_137.0

# 3GPP TR 38.901 building-penetration classes. Two numbers, not one: a modern
# low-E / metallized facade is ~13 dB worse than old plain glass, and this floor's
# exterior is `Glass_Basic_01`, so which one is used changes the answer materially.
O2I_DB = {"low_loss": 15.0, "high_loss": 28.0}

EXTERIOR_CLASS = 5           # exterior_glass — the envelope the plane wave lights
FLOORPLAN_META = ROOT / "Physics Engine" / "2D" / "STEP_1" / "floorplan_meta.json"
CITY_DIR_FULL = HERE / "city" / "NoMa_DC_buildings"
CITY_DIR_DEMO = HERE / "city_demo" / "NoMa_DC_tile"

# The known transmitter (user-supplied): Forte Hall rooftop.
FORTE_HALL_LONLAT = (-77.011420, 38.901550)


# --------------------------------------------------------------------------- geometry
def _enu(lon, lat, lon0, lat0):
    """Equirectangular local ENU metres about (lon0, lat0). Good to <0.1% at 400 m."""
    x = np.radians(lon - lon0) * R_EARTH * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R_EARTH
    return float(x), float(y)


RX_FLOOR_AGL_M = 21.0   # FCC HQ 7th-floor receiver plane (~7 storeys); see sites_registry


def base_station_geometry(lon: float, lat: float, meta_path: Path | None = None,
                          *, elev_m: float | None = None) -> dict:
    """Range + arrival bearing from the floor-plan centre to ANY base station (lon, lat).

    Derived from the floor plan's own QGIS-fitted georeference, so it moves if the
    registration is refit and never silently disagrees with the 2D pipeline. Returns
    distance_m, arrival_bearing_deg (compass direction the wave comes FROM — the
    convention `facade_sources_3d` takes), and the propagation direction. When `elev_m`
    (antenna height AGL) is given, also returns the vertical elevation angle vs the
    receiver floor (RX_FLOOR_AGL_M) — near 0 for Forte Hall (~20 m) at 415 m.
    """
    p = Path(meta_path or FLOORPLAN_META)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} — the O2I geometry is derived from the floor plan georeference; "
            f"without it the bearing would be a guess, which is what this replaces")
    meta = json.loads(p.read_text())
    lon0, lat0 = meta["origin_lonlat"]

    # building centre: raster centre pushed through the local-metre affine
    A = np.asarray(meta["affine_px_to_local_m"], float)
    H, W = meta["grid_shape"]
    cx, cy = W / 2.0, H / 2.0
    bx = A[0, 0] * cx + A[0, 1] * cy + A[0, 2]
    by = A[1, 0] * cx + A[1, 1] * cy + A[1, 2]

    tx_e, tx_n = _enu(lon, lat, lon0, lat0)
    dE, dN = tx_e - bx, tx_n - by
    dist = float(np.hypot(dE, dN))
    arrival = float(np.degrees(np.arctan2(dE, dN)) % 360.0)
    out = {
        "tx_lonlat": [float(lon), float(lat)],
        "origin_lonlat": [lon0, lat0],
        "building_centre_local_m": [round(bx, 2), round(by, 2)],
        "tx_local_m": [round(tx_e, 1), round(tx_n, 1)],
        "distance_m": round(dist, 1),
        "arrival_bearing_deg": round(arrival, 1),
        "propagation_bearing_deg": round((arrival + 180.0) % 360.0, 1),
        "derivation": "local ENU about floorplan_meta.origin_lonlat; building centre "
                      "from affine_px_to_local_m at the raster centre",
        "source": str(p),
    }
    if elev_m is not None:
        out["elev_m"] = float(elev_m)
        out["elevation_angle_deg"] = round(
            float(np.degrees(np.arctan2(elev_m - RX_FLOOR_AGL_M, max(dist, 1e-6)))), 2)
    return out


def forte_hall_geometry(meta_path: Path | None = None) -> dict:
    """The known Forte Hall macro cell — a thin wrapper over base_station_geometry."""
    return base_station_geometry(*FORTE_HALL_LONLAT, meta_path=meta_path)


def city_base_station_vox(manifest, lon, lat, elev_m, *, scene=None, search_radius_m=30.0):
    """Voxel Tx for a donor base station in a georeferenced OUTDOOR city scene.

    The outdoor sibling of `base_station_geometry`: horizontal position from the scene's
    `voxelize_city.lonlat_to_vox` (EPSG:3857 anchor), vertical from the antenna height AGL.
    The lat/lon pins (Apple Maps) can sit a few metres off the main tower, so snap horizontally
    to the TALLEST building column within `search_radius_m` (the real rooftop near the pin) and
    place the antenna one voxel above that roof (rooftop mount, in air). Returns (ix, iy, iz)."""
    import voxelize_city as VC
    vx, vz = VC.lonlat_to_vox(manifest, lon, lat)
    NX, NY, NZ = manifest["grid_shape"]
    cell = float(manifest["cell_size_m"])
    ix = int(np.clip(round(vx), 0, NX - 1)); iz = int(np.clip(round(vz), 0, NZ - 1))
    iy = int(np.clip(round(float(elev_m) / cell), 1, NY - 1))
    if scene is not None:
        M = scene.M
        r = max(1, int(round(search_radius_m / cell)))
        x0, x1 = max(0, ix - r), min(NX, ix + r + 1)
        z0, z1 = max(0, iz - r), min(NZ, iz + r + 1)
        sub = M[x0:x1, :, z0:z1] > 0                                  # (wx, NY, wz)
        ys = np.arange(NY)[None, :, None]
        top = np.where(sub, ys, -1).max(axis=1)                       # (wx, wz) roof height per column
        has = top >= 0
        if has.any():
            # Snap to the building column whose ROOF HEIGHT best matches the known antenna height
            # (with a light nearer-is-better tiebreak). The Apple-Maps pin can land on a low edge
            # structure or next to a taller neighbour, so height-match beats both "nearest" and
            # "tallest": Forte -> its ~20 m tower, not a 3 m ledge or a 57 m neighbour.
            gx, gz = np.meshgrid(np.arange(x0, x1), np.arange(z0, z1), indexing="ij")
            dist = np.sqrt((gx - ix) ** 2 + (gz - iz) ** 2) * cell
            score = np.where(has, np.abs(top * cell - float(elev_m)) + 0.05 * dist, 1e18)
            jx, jz = np.unravel_index(int(np.argmin(score)), score.shape)
            ix, iz = x0 + int(jx), z0 + int(jz)
            iy = int(top[jx, jz]) + 1                                  # just above the roof (air)
        iy = min(iy, NY - 1)
        while iy < NY - 1 and M[ix, iy, iz] > 0:                       # guarantee an air voxel
            iy += 1
    return (ix, iy, iz)


def fspl_db(distance_m: float, f_mhz: float) -> float:
    return 20.0 * np.log10(max(distance_m, 1e-6)) + 20.0 * np.log10(f_mhz) - 27.55


# --------------------------------------------------------------------------- facade
def facade_sources_3d(scene, bearing_deg, *, exterior_class=EXTERIOR_CLASS,
                      max_sources=None, y_band=None):
    """Illuminated exterior-envelope voxels for a plane wave arriving FROM `bearing_deg`.

    Compass convention, matching the 2D `phase_a.facade_sources`: 0 = north = -Z,
    90 = east = +X. The outward normal at each envelope voxel is estimated from the air
    voxels around it (weighted by 1/distance), which is robust to the ragged voxel
    boundary in a way a per-face normal is not.

    `max_sources` evenly thins the illuminated set (see `bs_field_3d` — each surviving
    source costs a full `crossing_loss` solve, so this is the cost knob).

    Returns (pos (n,3) float32 voxel coords, proj (n,) float32 metres along the arrival
    direction — larger = nearer the base station, used to phase the wavefront).
    """
    M = scene.M
    NX, NY, NZ = M.shape
    th = np.radians(float(bearing_deg))
    # building -> BS unit vector in the horizontal plane (X east, Z north-negative)
    u_from = np.array([np.sin(th), 0.0, -np.cos(th)], np.float32)

    env = (M == exterior_class)
    if y_band is not None:
        band = np.zeros(NY, bool)
        band[y_band[0]:y_band[1]] = True
        env &= band[None, :, None]
    if not env.any():
        return np.zeros((0, 3), np.float32), np.zeros(0, np.float32)

    # Outward normal from air neighbours in a 2-voxel shell, computed for the WHOLE grid
    # at once. The per-voxel Python version was ~573k iterations on this scene; shifting
    # the air mask 74 times is the same arithmetic as a handful of array passes.
    air = (M == 0).astype(np.float32)
    ap = np.pad(air, 2, mode="constant")
    nrm = np.zeros((3,) + M.shape, np.float32)
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-1, 0, 1):
            for dz in (-2, -1, 0, 1, 2):
                if dx == dy == dz == 0:
                    continue
                w = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
                sh = ap[2 + dx:2 + dx + NX, 2 + dy:2 + dy + NY, 2 + dz:2 + dz + NZ]
                nrm[0] += (dx * w) * sh
                nrm[1] += (dy * w) * sh
                nrm[2] += (dz * w) * sh

    ln = np.linalg.norm(nrm, axis=0)
    lit = env & (ln >= 0.5)                            # ln < 0.5 => interior glass
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_bs = ((nrm[0] * u_from[0] + nrm[1] * u_from[1] + nrm[2] * u_from[2])
                  / np.maximum(ln, 1e-6))
    lit &= cos_bs > 0.15                               # this face sees the base station
    idx = np.nonzero(lit)
    if len(idx[0]) == 0:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.float32)

    pos = np.stack(idx, -1).astype(np.float32)
    proj = (pos @ u_from) * scene.cell

    if max_sources and len(pos) > max_sources:
        # Spread the survivors along the arrival direction rather than taking an array
        # stride: index order is X-major, so a stride would over-sample one end of the
        # facade and leave the other dark.
        order = np.argsort(proj)
        keep = order[np.linspace(0, len(order) - 1, int(max_sources)).astype(int)]
        pos, proj = pos[keep], proj[keep]
    return pos, proj


DEFAULT_O2I_SOURCES = 48
O2I_CACHE = HERE / "cache"


def outdoor_scene_dir() -> Path:
    if (CITY_DIR_FULL / "material_grid.npy").exists():
        return CITY_DIR_FULL
    if (CITY_DIR_DEMO / "material_grid.npy").exists():
        return CITY_DIR_DEMO
    return CITY_DIR_FULL


def _o2i_cache_key(scene, bearing_deg, o2i_db, freqs, n_src, exterior_class):
    import hashlib
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(scene.M).tobytes())
    h.update(repr((round(float(bearing_deg), 3), float(o2i_db), [float(f) for f in freqs],
                   int(n_src), int(exterior_class), float(scene.cell))).encode())
    return h.hexdigest()[:16]


def bs_field_3d(scene, bearing_deg, *, bands=None, o2i_db=O2I_DB["low_loss"],
                max_sources=DEFAULT_O2I_SOURCES, distance_m=None, p_ref_dbm=None,
                y_band=None, exterior_class=EXTERIOR_CLASS, cache=True,
                cache_dir=None, progress=False):
    """Outdoor-to-indoor field: a distant macro cell lighting the envelope.

    Each illuminated facade voxel re-radiates inward through the engine's own
    `crossing_loss`, charged the 3GPP O2I penetration loss once. Contributions add in
    LINEAR POWER, not coherently: the facade sources are a discretization of one wavefront,
    not physically distinct scatterers, so summing them as fields would manufacture an
    interference pattern that is an artifact of how finely the envelope was sampled.
    Averaging over sources keeps the result facade-referenced rather than source-count
    scaled — the same choice `phase_a.bs_maps` makes in 2D.

    COST, AND WHY IT IS CACHED
    Every source is one `crossing_loss` solve (~2.9 s on the production scene), so the
    naive "one source per illuminated voxel" is 645 solves — half an hour for one map.
    But the O2I source geometry is FIXED for a deployment: Forte Hall does not move. So
    the whole field is a one-time computation, cached on disk and keyed by the scene grid
    plus the bearing/loss/band set. First call ~2 min at the default 48 sources, then free.
    This is the same reasoning as the diffraction relay cache.

    Returns a `contracts.FieldGrid` (INCOHERENT) so it drops into `Combine_3D.combine`
    beside the mechanism contributions.
    """
    import contracts

    freqs = np.asarray(scene.freqs if bands is None else bands, float)
    bidx = [scene.band_index(f) for f in freqs]
    NX, NY, NZ = scene.M.shape

    pos, proj = facade_sources_3d(scene, bearing_deg, max_sources=max_sources,
                                  exterior_class=exterior_class, y_band=y_band)
    if len(pos) == 0:
        raise ValueError(
            f"no illuminated facade at arrival bearing {bearing_deg} deg "
            f"(exterior class {exterior_class} covers "
            f"{int((scene.M == exterior_class).sum())} voxels)")

    cdir = Path(cache_dir or O2I_CACHE)
    key = _o2i_cache_key(scene, bearing_deg, o2i_db, freqs, len(pos), exterior_class)
    cpath = cdir / f"o2i_{key}.npz"
    if cache and cpath.exists():
        z = np.load(cpath)
        p_lin, tau = z["p_lin"], z["tau"]
        if progress:
            print(f"  o2i cache hit ({cpath.name})")
    else:
        p_lin = np.zeros((len(freqs), NX, NY, NZ), np.float64)
        tau = np.full((len(freqs), NX, NY, NZ), np.inf, np.float32)

        # The facade voxel nearest the base station is lit first; the rest lag by the
        # projected offset along the arrival direction. That is what tilts the indoor
        # wavefront to match the angle of arrival instead of blooming from one point.
        t0 = (proj.max() - proj) / C0

        gx, gy, gz = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ),
                                 indexing="ij")
        P = np.stack([gx, gy, gz], -1).astype(np.float32)

        for si, (s, ts) in enumerate(zip(pos, t0)):
            anchor = tuple(int(v) for v in s)
            obs = scene.crossing_loss(anchor)                  # (nf_scene, NX,NY,NZ) dB
            d_m = np.linalg.norm(P - s, axis=-1) * scene.cell
            d_clamped = np.maximum(d_m, scene.d0)
            for i, bi in enumerate(bidx):
                pl = (scene.fspl1[bi] + 10.0 * scene.n_exp * np.log10(d_clamped)
                      + obs[bi] + float(o2i_db))
                p_lin[i] += np.power(10.0, -pl.astype(np.float64) / 10.0)
                # unclamped for time: the d0 log clamp is a level guard, not a delay
                np.minimum(tau[i], (ts + d_m / C0).astype(np.float32), out=tau[i])
            if progress:
                print(f"  facade source {si + 1}/{len(pos)}", flush=True)

        p_lin /= float(len(pos))                               # facade-referenced
        if cache:
            cdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cpath, p_lin=p_lin, tau=tau)

    fg = contracts.FieldGrid(
        E=np.zeros((contracts.NPOL, len(freqs), NX, NY, NZ), np.complex64),
        tau_first=np.asarray(tau, np.float32), mechanism="o2i_facade", freqs_mhz=freqs,
        tx_vox=tuple(int(v) for v in pos[int(np.argmax(proj))]),
        p_incoh=np.asarray(p_lin, np.float32),
        meta={"arrival_bearing_deg": float(bearing_deg), "o2i_db": float(o2i_db),
              "n_facade_sources": int(len(pos)), "max_sources": int(max_sources or 0),
              "distance_m": distance_m, "p_ref_dbm": p_ref_dbm, "cache": cpath.name,
              "model": "3GPP TR 38.901 O2I + per-source multiwall, power-summed",
              "note": "INCOHERENT by construction: facade sources discretize ONE "
                      "wavefront, so coherent summing would be a sampling artifact"})
    fg.combine_as = contracts.INCOHERENT
    return fg


# ------------------------------------------------------ explicit directional O2I march (Commit 2)
# A single PARALLEL wavefront marched along the arrival direction — the 3-D lift of the 2-D
# `o2i_indoor_loss`, and an EXPLICIT directional path loss (only the wall-march term varies in
# space, so it alone sets ρ; the outdoor spreading + O2I penetration are a calibrated level).
#
# NOTE ON WHERE THIS IS USED: the INDOOR O2I cross-validation does NOT march the Sandbox voxel
# scene — a full bearing sweep showed that grid caps ρ at ≈0.22 for any direction (crude/generic
# geometry), while the fine floor-plan grid reaches 0.64. So the indoor O2I predictor
# (Cross_Validation/pathloss_3d/predict_o2i_3d.py) marches the ACCURATE floor-plan grid in the px
# frame instead. These voxel-scene marchers remain the right tool for the OUTDOOR city voxel scene
# (Commit 4), where there is no finer 2-D grid to fall back to.
REG3D_PATH = HERE / "registration_3d.json"


def arrival_dir_vox(bearing_deg, elevation_angle_deg=0.0, *, meta_path=None, reg_path=None):
    """Unit arrival direction (toward the donor) in VOXEL axes (X,Y_up,Z).

    Derived through the SAME georeference chain the rest of the pipeline uses, NOT the
    `facade_sources_3d` "+X=east" convention (which the registration shows is mirrored —
    voxel +X actually runs west). compass bearing → ENU → floor-plan px (inverse
    `affine_px_to_local_m`) → voxel (the diagonal `affine_px_to_vox` sign/scale). The
    vertical component comes from the elevation angle (≈0 for Forte Hall)."""
    meta = json.loads(Path(meta_path or FLOORPLAN_META).read_text())
    Linv = np.linalg.inv(np.asarray(meta["affine_px_to_local_m"], float)[:, :2])
    br = np.radians(float(bearing_deg))
    dpx = Linv @ np.array([np.sin(br), np.cos(br)])          # px-space dir toward the donor
    reg = json.loads(Path(reg_path or REG3D_PATH).read_text())
    aff = reg.get("affine_px_to_vox")
    if aff is None:                                          # legacy scale-only fallback
        aX = eZ = reg["px_to_vox_scale"]["x"]
    else:
        aX, eZ = aff[0][0], aff[1][1]                        # diagonal: px_x→X, px_y→Z
    ux, uz = aX * dpx[0], eZ * dpx[1]
    h = float(np.hypot(ux, uz)) or 1.0
    ux, uz = ux / h, uz / h
    phi = np.radians(float(elevation_angle_deg))
    u = np.array([np.cos(phi) * ux, np.sin(phi), np.cos(phi) * uz], np.float64)
    n = float(np.linalg.norm(u)) or 1.0
    return (u / n).astype(np.float32)


def material_loss_arrays(manifest, n=256):
    """(loss_db, loss_per_m) indexed by material class id, from the manifest materials table.

    The engine's `crossing_loss` uses the CrossingLUT (Fresnel/Airy); this explicit march
    uses the FLAT per-crossing `loss_db` + per-metre `loss_per_m_db` table — exactly the
    Motley-Keenan model the validated 2-D pipeline uses."""
    loss_db = np.zeros(n, np.float32)
    loss_per_m = np.zeros(n, np.float32)
    for m in manifest.get("materials", []):
        i = int(m["id"])
        loss_db[i] = float(m.get("loss_db", 0.0))
        loss_per_m[i] = float(m.get("loss_per_m_db", 0.0))
    return loss_db, loss_per_m


def o2i_wall_loss_3d(scene, u_from, *, loss_db, loss_per_m, step_cells=0.5,
                     wall_sat_db=60.0, mask=None, chunk=8000):
    """Indoor obstruction loss (dB) for a plane wave arriving along `u_from` (voxel axes).

    3-D lift of the 2-D `o2i_indoor_loss`: from every (masked) voxel, march ONE parallel ray
    toward the donor, count contiguous wall-material runs once (Motley-Keenan) + per-metre
    bulk clutter, saturate at `wall_sat_db`. Voxels outside `mask` are +inf (masked later)."""
    M = scene.M
    NX, NY, NZ = M.shape
    cell = float(scene.cell)
    u = np.asarray(u_from, np.float32); u = u / (np.linalg.norm(u) or 1.0)
    K = int(np.ceil(float(np.linalg.norm([NX, NY, NZ])) / step_cells)) + 2
    s = np.arange(K, dtype=np.float32) * step_cells
    seg_m = step_cells * cell
    wall_ids = [c for c in range(len(loss_db)) if loss_db[c] > 0]
    bulk_ids = [c for c in range(len(loss_per_m)) if loss_per_m[c] > 0]

    ins = scene.inside if mask is None else mask
    ins = np.ones(M.shape, bool) if ins is None else np.asarray(ins, bool)
    V = np.argwhere(ins).astype(np.float32)                  # (Nvox,3) voxels we solve
    out = np.full(V.shape[0], np.inf, np.float32)
    for c0 in range(0, V.shape[0], chunk):
        vv = V[c0:c0 + chunk]
        sx = vv[:, 0:1] + s[None, :] * u[0]
        sy = vv[:, 1:2] + s[None, :] * u[1]
        sz = vv[:, 2:3] + s[None, :] * u[2]
        valid = (sx >= 0) & (sx < NX) & (sy >= 0) & (sy < NY) & (sz >= 0) & (sz < NZ)
        xi = np.clip(sx.round(), 0, NX - 1).astype(np.int32)
        yi = np.clip(sy.round(), 0, NY - 1).astype(np.int32)
        zi = np.clip(sz.round(), 0, NZ - 1).astype(np.int32)
        mats = np.where(valid, M[xi, yi, zi], -1)            # -1 = off-grid sentinel (int8-safe)
        extra = np.zeros(mats.shape[0], np.float32)
        for c in wall_ids:
            hit = mats == c
            runs = hit[:, 0].astype(np.int32) + (hit[:, 1:] & ~hit[:, :-1]).sum(1, dtype=np.int32)
            extra += loss_db[c] * runs
        for c in bulk_ids:
            extra += loss_per_m[c] * (mats == c).sum(1) * seg_m
        out[c0:c0 + chunk] = wall_sat_db * -np.expm1(-extra / wall_sat_db)
    full = np.zeros(M.shape, np.float32)     # unsolved voxels: 0 wall loss (free space, finite)
    full[ins] = out
    return full


def o2i_pl_3d(scene, manifest, bearing_deg, freq_mhz, *, elevation_angle_deg=0.0,
              level_db=0.0, ple_n=None, d0_m=None, distance_m=None,
              step_cells=0.5, wall_sat_db=60.0, mask=None):
    """Explicit directional O2I path loss PL(NX,NY,NZ) in dB.

        PL(v) = outdoor_leg + O2I_penetration + indoor_walls(v)

    `indoor_walls(v)` (the only spatially-varying term → the ρ driver) is the parallel-ray
    Motley-Keenan march. The outdoor spreading + penetration collapse into the single
    additive `level_db` the caller passes (the calibration fits it), so ρ is set by the
    march alone. When `level_db` is 0 and `distance_m` is given, a plain log-distance outdoor
    leg is added for a physically-sensible absolute level."""
    u = arrival_dir_vox(bearing_deg, elevation_angle_deg)
    loss_db, loss_per_m = material_loss_arrays(manifest)
    walls = o2i_wall_loss_3d(scene, u, loss_db=loss_db, loss_per_m=loss_per_m,
                             step_cells=step_cells, wall_sat_db=wall_sat_db, mask=mask)
    lvl = float(level_db)
    if lvl == 0.0 and distance_m is not None:
        n_exp = float(scene.n_exp if ple_n is None else ple_n)
        d0 = float(scene.d0 if d0_m is None else d0_m)
        bi = scene.band_index(freq_mhz)
        lvl = float(scene.fspl1[bi] + 10.0 * n_exp * np.log10(max(distance_m, d0)))
    return (walls + lvl).astype(np.float32), u


# --------------------------------------------------------------------------- registry
@dataclass
class Mode:
    key: str
    label: str
    summary: str
    source: str                 # 'point' | 'plane_wave' | 'none'
    scene_dir: Path | None
    barrier_classes: tuple
    mechanisms: tuple
    notes: str = ""
    extra: dict = field(default_factory=dict)


MODES: dict[str, Mode] = {
    "vacuum": Mode(
        key="vacuum", label="Vacuum / free space",
        summary="All air. PL must equal FSPL exactly — the invariant gate, visible.",
        source="point", scene_dir=None, barrier_classes=(),
        mechanisms=("path_loss",),
        notes="No materials, so only spreading applies; every other mechanism would be "
              "identically zero. Running the other modules here is not wrong, just empty.",
    ),
    "indoor": Mode(
        key="indoor", label="Indoor",
        summary="The voxelized 7th floor with the transmitter inside. The production case.",
        source="point", scene_dir=HERE, barrier_classes=(3,),
        mechanisms=("path_loss", "reflection", "diffraction", "scattering"),
    ),
    "o2i": Mode(
        key="o2i", label="Indoor + Outdoor (O2I)",
        summary="Distant macro cell lighting the facade; energy penetrates and spreads inside.",
        source="plane_wave", scene_dir=HERE, barrier_classes=(3,),
        mechanisms=("o2i_facade", "reflection", "diffraction", "scattering"),
        notes="Source geometry is derived from the Forte Hall coordinates and the floor "
              "plan georeference, not from the placeholder 135 deg bearing.",
    ),
    "outdoor": Mode(
        key="outdoor", label="Outdoor",
        summary="City block: buildings are one barrier class, streets are air.",
        source="point", scene_dir=outdoor_scene_dir(), barrier_classes=(3,),
        mechanisms=("path_loss", "reflection", "diffraction"),
        notes="Ground reflection and terrain diffraction are not new physics — they are "
              "Reflection_3D and Diffraction_3D on a scene whose walls are buildings. "
              "Diffuse scattering is omitted by default: at 1 m cells the facade patch "
              "count explodes and the effective-roughness model is calibrated for indoor "
              "surfaces, so enabling it here would be extrapolation.",
    ),
}

DEFAULT_MODE = "indoor"


def list_modes() -> list[str]:
    return list(MODES)


def get_mode(key: str) -> Mode:
    if key not in MODES:
        raise KeyError(f"unknown mode {key!r}; have {list(MODES)}")
    if key == "outdoor":
        MODES[key].scene_dir = outdoor_scene_dir()
    return MODES[key]


def describe(key: str) -> dict:
    """Everything a caller (or the browser) needs to explain a mode to the user."""
    m = get_mode(key)
    d = {"key": m.key, "label": m.label, "summary": m.summary, "source": m.source,
         "mechanisms": list(m.mechanisms), "barrier_classes": list(m.barrier_classes),
         "notes": m.notes, "available": True, "unavailable_reason": None}
    if m.key == "o2i":
        try:
            d["geometry"] = forte_hall_geometry()
            d["o2i_db"] = O2I_DB
        except FileNotFoundError as e:
            d["available"], d["unavailable_reason"] = False, str(e)
    if m.scene_dir is not None and not (m.scene_dir / "material_grid.npy").exists():
        d["available"] = False
        d["unavailable_reason"] = (
            f"{m.scene_dir}/material_grid.npy not built"
            + (" — run voxelize_city.py" if m.key == "outdoor" else " — run voxelize.py"))
    return d


def build_scene(key: str, *, n=None, bands=None):
    """(scene, manifest) for a mode. Vacuum is synthesized; the rest are loaded.

    Vacuum defaults to the PRODUCTION grid shape, not a small cube: the point of the mode
    is to be compared against the indoor result in the same viewport and at the same
    scale, and a 48-cube would not line up with anything.
    """
    m = get_mode(key)
    if m.key == "vacuum":
        man = json.loads((HERE / "manifest_3d.json").read_text())
        n = tuple(int(v) for v in (n or man["grid_shape"]))
        freqs = [float(f) for f in (bands or man["freqs_mhz"])]
        vman = dict(man)
        vman["grid_shape"] = list(n)
        vman["freqs_mhz"] = freqs
        sc = engine_3d.SceneV3(np.zeros(n, np.int8), vman, barrier_classes=())
        sc.inside = np.ones(n, bool)
        return sc, vman
    if not (m.scene_dir / "material_grid.npy").exists():
        raise FileNotFoundError(describe(key)["unavailable_reason"])
    sc, man = engine_3d.load_scene(m.scene_dir, barrier_classes=m.barrier_classes)
    return sc, man


def source_for(key: str, scene, *, tx=None):
    """Where the energy comes from, per mode.

    Point modes return a Tx voxel; the plane-wave mode returns the arrival bearing and
    range instead, because there is no Tx voxel inside the grid to return.
    """
    m = get_mode(key)
    if m.source == "plane_wave":
        g = forte_hall_geometry()
        return {"kind": "plane_wave", "arrival_bearing_deg": g["arrival_bearing_deg"],
                "distance_m": g["distance_m"], "geometry": g}
    if tx is not None:
        return {"kind": "point", "tx_vox": tuple(int(v) for v in tx)}
    vt = m.scene_dir / "valid_tx_mask.npy" if m.scene_dir else None
    if vt and vt.exists():
        valid = np.load(vt)
        ijk = np.argwhere(valid)
        target = np.array([0.5 * valid.shape[0], 0.85 * valid.shape[1],
                           0.5 * valid.shape[2]])
        pick = ijk[np.argmin(((ijk - target) ** 2).sum(1))]
        return {"kind": "point", "tx_vox": tuple(int(v) for v in pick)}
    c = tuple(int(s // 2) for s in scene.M.shape)
    return {"kind": "point", "tx_vox": c}


def solve_mode(key: str, scene, *, tx=None, bands=None, mechanisms=None,
               backend="numpy", relay=None, edges=None, o2i_class="low_loss",
               max_sources=DEFAULT_O2I_SOURCES, progress=False):
    """Run a mode's mechanism stack. Returns (contribs, source_info).

    The only mode-specific branch is the SOURCE: `o2i` replaces the direct field with the
    facade plane wave. Every other mechanism is called exactly as the indoor path calls it,
    which is the point of the registry.
    """
    m = get_mode(key)
    mechs = list(mechanisms or m.mechanisms)
    src = source_for(key, scene, tx=tx)
    contribs = []

    if "o2i_facade" in mechs:
        g = src["geometry"]
        fg = bs_field_3d(scene, g["arrival_bearing_deg"], bands=bands,
                         o2i_db=O2I_DB[o2i_class], max_sources=max_sources,
                         distance_m=g["distance_m"], progress=progress)
        contribs.append(fg)
        # Secondary mechanisms need a point to expand from. Use the facade voxel nearest
        # the base station, so reflection and diffraction inherit the true angle of
        # arrival instead of radiating from an arbitrary interior point.
        tx_eff = fg.tx_vox
    else:
        tx_eff = src["tx_vox"]

    if "path_loss" in mechs:
        import Path_Loss_3D as PLM
        contribs.append(PLM.solve(scene, tx_eff, bands=bands))
    if "reflection" in mechs and tx_eff is not None:
        import Reflection_3D as RFL
        contribs.append(RFL.solve(scene, tx_eff, bands=bands, backend=backend))
    if "diffraction" in mechs and tx_eff is not None:
        import Diffraction_3D as DIF
        contribs.append(DIF.solve(scene, tx_eff, bands=bands, edges=edges,
                                  n_edges=len(edges) if edges else 16, relay=relay))
    if "scattering" in mechs and tx_eff is not None:
        import Scattering_3D as SCT
        contribs.append(SCT.solve(scene, tx_eff, bands=bands, backend=backend,
                                  patch_cells=3, rx_downsample=2))
    if not contribs:
        raise ValueError(f"mode {key!r} selected no mechanisms")
    src["tx_effective_vox"] = tx_eff
    return contribs, src


# --------------------------------------------------------------------------- self-test
def _selftest() -> int:
    ok, fails = 0, []

    def check(name, cond, msg=""):
        nonlocal ok
        if cond:
            ok += 1
            print(f"  PASS  {name}")
        else:
            fails.append(name)
            print(f"  FAIL  {name}  {msg}")

    check("four modes registered", list_modes() == ["vacuum", "indoor", "o2i", "outdoor"],
          str(list_modes()))

    # ---- the vacuum gate: PL must BE free space ---------------------------
    import Path_Loss_3D as PLM
    sc, man = build_scene("vacuum", n=(28, 28, 28), bands=[2442.0, 3500.0])
    tx = (14, 14, 14)
    fg = PLM.solve(sc, tx, bands=[2442.0, 3500.0])
    fs = PLM.free_space_field(sc, tx, bands=[2442.0, 3500.0])
    err = float(np.nanmax(np.abs(fg.pathloss_db() - fs.pathloss_db())))
    check("vacuum PL == FSPL", err < 1e-3, f"max err {err:.6f} dB")

    d = np.linalg.norm(sc.P - np.asarray(tx, np.float32), axis=1).reshape(sc.M.shape)
    d_m = np.maximum(d * sc.cell, sc.d0)
    ana = 20 * np.log10(d_m) + 20 * np.log10(2442.0) - 27.55
    err2 = float(np.nanmax(np.abs(fg.pathloss_db()[0] - ana)))
    check("vacuum matches closed-form 20log10(d)+20log10(f)-27.55", err2 < 0.05,
          f"max err {err2:.4f} dB")

    # Causality is checked against the UNCLAMPED distance: d_m carries the d0 = 1 m log
    # clamp (CS2), which near the source claims a 3.3 ns lower bound where the true
    # travel time is ~0. The slack is the fast-marching discretization (skfmm seeds the
    # zero contour on the source cell, so times start about half a cell early).
    tau = fg.tau_first[0]
    d_true = (d * sc.cell) / C0
    slack = 1.5 * sc.cell / C0
    worst = float(np.min(tau - d_true))
    check("vacuum causal", worst >= -slack,
          f"arrives {-worst * 1e9:.2f} ns early, past the {slack * 1e9:.2f} ns grid tolerance")

    # ---- O2I geometry is derived, not the 135 deg placeholder -------------
    try:
        g = forte_hall_geometry()
        check("Forte Hall range ~415 m", abs(g["distance_m"] - 415.9) < 2.0,
              f"{g['distance_m']} m")
        check("arrival bearing ~237 deg", abs(g["arrival_bearing_deg"] - 237.0) < 1.0,
              f"{g['arrival_bearing_deg']} deg")
        check("bearing is NOT the 135 deg placeholder",
              abs(g["arrival_bearing_deg"] - 135.0) > 50.0)
        f0 = fspl_db(g["distance_m"], 619.0)
        f1 = fspl_db(g["distance_m"], 3710.0)
        check("outdoor-leg FSPL spans 80.7-96.2 dB",
              abs(f0 - 80.7) < 0.5 and abs(f1 - 96.2) < 0.5, f"{f0:.1f}..{f1:.1f} dB")
    except FileNotFoundError as e:
        check("Forte Hall geometry", False, str(e))

    check("3GPP O2I classes present",
          O2I_DB["low_loss"] == 15.0 and O2I_DB["high_loss"] == 28.0)

    # ---- describe() must never raise, for any mode ------------------------
    for k in list_modes():
        try:
            dsc = describe(k)
            check(f"describe({k})", dsc["key"] == k and "summary" in dsc)
        except Exception as e:                                   # noqa: BLE001
            check(f"describe({k})", False, repr(e))

    print(f"\n{ok} passed, {len(fails)} failed")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--list", action="store_true",
                    help="list the modes and availability (the default action)")
    ap.add_argument("--describe", metavar="MODE", help="print one mode as JSON")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args(argv)

    if a.test:
        return _selftest()
    if a.describe:
        print(json.dumps(describe(a.describe), indent=2))
        return 0
    # Listing is the fallthrough, not a gated branch: README documents a bare
    # `modes_3d.py` as the "what modes are there" command, so --list names the default
    # rather than switching it on.
    for k in list_modes():
        d = describe(k)
        mark = "ok " if d["available"] else "NA "
        print(f"  [{mark}] {k:8s} {d['label']:26s} {d['summary']}")
        if not d["available"]:
            print(f"            -> {d['unavailable_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
