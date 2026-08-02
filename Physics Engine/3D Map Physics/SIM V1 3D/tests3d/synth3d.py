"""
Synthetic 3-D scenes with ANALYTIC ground truth.

Every mechanism is tested against these first (where the right answer is known in
closed form) and against the real voxel scene second (regression only). These are the
3-D analogues of `engine_v2.synth_slab_scene` / `synth_floor_scene` in the 2D engine.

Each builder returns (M, manifest) ready for `engine_3d.SceneV3(M, manifest)`.

Material classes (manifest_3d.json):
    0 air · 1 drywall_partition · 2 concrete_masonry
    3 core_service_area (barrier) · 4 furniture_clutter · 5 exterior_glass
"""
from __future__ import annotations

import numpy as np

DEFAULT_FREQS = (2442.0, 3500.0, 5500.0, 6125.0)
C0 = 299_792_458.0


def manifest_for(cell_m: float = 0.3, freqs=DEFAULT_FREQS, **physics) -> dict:
    """Minimal manifest carrying exactly the keys SceneV3.__init__ reads."""
    phys = {"n_exp": 2.0, "d0_m": 1.0, "fspl_const_db": -27.56}
    phys.update(physics)
    return {
        "cell_size_m": float(cell_m),
        "freqs_mhz": [float(f) for f in freqs],
        "physics": phys,
        "grid_shape": None,          # filled by the builders
        "synthetic": True,
    }


def _finish(M: np.ndarray, cell_m: float, freqs) -> tuple[np.ndarray, dict]:
    man = manifest_for(cell_m, freqs)
    man["grid_shape"] = list(M.shape)
    return np.ascontiguousarray(M.astype(np.int8)), man


# --------------------------------------------------------------------------- scenes
def free_space_scene(n=(48, 48, 48), cell_m=0.3, freqs=DEFAULT_FREQS):
    """All air. PL must equal FSPL exactly - the vacuum mode and the strictest gate."""
    return _finish(np.zeros(n, np.int8), cell_m, freqs)


def slab_scene_3d(n=(64, 32, 32), mat=2, x0=28, x1=31, cell_m=0.3, freqs=DEFAULT_FREQS):
    """One planar slab normal to +X. The analytic Fresnel / Airy transmission case."""
    M = np.zeros(n, np.int8)
    M[x0:x1, :, :] = mat
    return _finish(M, cell_m, freqs)


def two_plate_scene_3d(n=(64, 24, 48), gap_cells=16, mat=2, cell_m=0.3, freqs=DEFAULT_FREQS):
    """Floor + ceiling slabs. The 2-ray ground-reflection and waveguiding case.

    This is the geometry the REAL scene is currently missing (no ceiling slab), so it is
    also the reference for what a correct indoor scene should reproduce.
    """
    M = np.zeros(n, np.int8)
    y0 = (n[1] - gap_cells) // 2
    M[:, :y0, :] = mat                      # floor
    M[:, y0 + gap_cells:, :] = mat          # ceiling
    return _finish(M, cell_m, freqs)


def corner_scene_3d(n=(64, 24, 64), mat=2, cell_m=0.3, freqs=DEFAULT_FREQS):
    """A 90-degree wedge (half-plane pair). The UTD diffraction case.

    Wall occupies x < xc for z > zc, leaving a shadow region a straight ray cannot reach.
    """
    M = np.zeros(n, np.int8)
    xc, zc = n[0] // 2, n[2] // 2
    M[xc:xc + 2, :, zc:] = mat
    return _finish(M, cell_m, freqs)


def corridor_scene_3d(n=(96, 16, 32), width_cells=7, mat=2, cell_m=0.3, freqs=DEFAULT_FREQS):
    """A long corridor. Waveguiding should give an effective exponent n_eff < 2."""
    M = np.full(n, mat, np.int8)
    z0 = (n[2] - width_cells) // 2
    M[:, 1:n[1] - 1, z0:z0 + width_cells] = 0        # hollow the corridor
    return _finish(M, cell_m, freqs)


def rough_wall_scene_3d(n=(64, 24, 48), mat=2, sigma_cells=1.0, seed=0,
                        cell_m=0.3, freqs=DEFAULT_FREQS):
    """A wall with a randomized surface profile. The diffuse-scattering case."""
    rng = np.random.default_rng(seed)
    M = np.zeros(n, np.int8)
    xc = n[0] // 2
    jitter = rng.normal(0.0, sigma_cells, size=(n[1], n[2])).round().astype(int)
    for j in range(n[1]):
        for k in range(n[2]):
            x = int(np.clip(xc + jitter[j, k], 1, n[0] - 3))
            M[x:x + 2, j, k] = mat
    return _finish(M, cell_m, freqs)


def pec_box_scene_3d(n=(40, 40, 40), mat=3, cell_m=0.3, freqs=DEFAULT_FREQS):
    """A sealed shell of barrier material. Energy-conservation case: nothing escapes."""
    M = np.zeros(n, np.int8)
    M[0, :, :] = M[-1, :, :] = mat
    M[:, 0, :] = M[:, -1, :] = mat
    M[:, :, 0] = M[:, :, -1] = mat
    return _finish(M, cell_m, freqs)


# --------------------------------------------------------------------------- helpers
def fspl_db(d_m, f_mhz, n_exp=2.0, d0_m=1.0, fspl_const_db=-27.56):
    """The engine's own spreading term: fspl_1m(f) + 10*n*log10(max(d, d0))."""
    d = np.maximum(np.asarray(d_m, float), d0_m)
    return 20.0 * np.log10(f_mhz) + fspl_const_db + 10.0 * n_exp * np.log10(d)


def distance_grid(shape, tx, cell_m):
    """Euclidean distance in metres from `tx` (voxel coords) to every voxel."""
    gx, gy, gz = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    return np.sqrt((gx - tx[0]) ** 2 + (gy - tx[1]) ** 2 + (gz - tx[2]) ** 2) * cell_m


def centre(shape):
    return tuple(int(s // 2) for s in shape)
