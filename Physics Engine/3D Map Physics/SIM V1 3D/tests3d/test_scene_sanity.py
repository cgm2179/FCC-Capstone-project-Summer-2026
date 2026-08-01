"""
Scene fidelity — the binding error term.

The physics solvers are only as good as the material grid they read. These tests pin the
properties a physically usable indoor scene must have.

STATUS: PASSING as hard gates (STRICT_SCENE=True) since the M0.4 re-voxelization.

Why these exist: for a ceiling-mounted AP the floor bounce and the ceiling bounce are the
two dominant specular paths, so a scene with no ceiling, or a floor slab carrying the
wrong permittivity, produces a confidently wrong answer no matter how good the solver is.

The defect these were written against (old grid, 262x11x118, from the sparse 18-material
mesh with a `".*" -> class 1` catch-all):
    y=10 was 100% air                  -> NO CEILING SLAB
    floor slab was drywall (eps'~2.9)  -> should be concrete (~5.24)
    concrete 0.084%, furniture 0.021%, drywall 23.3%
    valid_tx_mask identical to inside_mask (no wall clearance)

After re-voxelizing from Sandbox_Version_3D_Simulation_1.obj (253 named materials,
5.08M triangles, 542k people/prop triangles excluded) the grid is 262x17x132 with:
    room band y=3..8, floor slabs y=0,2 (100% concrete), ceiling y=9 (87.5% solid)
    concrete 11.33%  furniture 2.17%  drywall 11.53%  glass 1.32%
    valid_tx 2,510 of 69,432 interior voxels (2-voxel wall clearance)
"""
from __future__ import annotations

import numpy as np
import pytest

STRICT_SCENE = True         # M0.4 re-voxelization landed - these are now hard gates
_defect = pytest.mark.xfail(not STRICT_SCENE, reason="M0.4 re-voxelization pending",
                            strict=False)

AIR, DRYWALL, CONCRETE, CORE, FURNITURE, GLASS = range(6)


# --------------------------------------------------------------------------- structural
def test_grid_is_wellformed(grid, manifest):
    assert grid.ndim == 3
    assert list(grid.shape) == list(manifest["grid_shape"])
    assert grid.min() >= 0 and grid.max() < len(manifest["materials"])
    assert np.isfinite(manifest["cell_size_m"]) and manifest["cell_size_m"] > 0


def test_inside_mask_is_consistent(grid, inside_mask):
    assert inside_mask.shape == grid.shape
    assert inside_mask.any(), "inside_mask is empty"
    # interior cells should be predominantly air, not solid
    air_frac = float((grid[inside_mask] == AIR).mean())
    assert air_frac > 0.5, f"only {100*air_frac:.1f}% of 'inside' voxels are air"


def test_scene_has_matter(grid):
    solid = float((grid != AIR).mean())
    assert solid > 0.02, f"scene is {100*(1-solid):.1f}% air - no structure to propagate through"


# --------------------------------------------------------------------------- fidelity
def room_band(grid, thresh=0.5):
    """(room_lo, room_hi, floor_slabs, ceiling_y) - the room is the largest vertical gap
    between dense slabs. Mirrors voxelize.interior_air so tests and voxelizer agree."""
    prof = (grid != AIR).mean(axis=(0, 2))
    slabs = np.nonzero(prof > thresh)[0]
    if slabs.size < 2:
        return 1, grid.shape[1] - 1, slabs.tolist(), None
    k = int(np.argmax(np.diff(slabs)))
    return int(slabs[k]) + 1, int(slabs[k + 1]) - 1, slabs[:k + 1].tolist(), int(slabs[k + 1])


@_defect
def test_ceiling_slab_exists(grid):
    """A ceiling slab must cap the room, else the ceiling bounce cannot be modelled.

    Checked against the ROOM BAND, not the top grid layer: a detailed mesh may model a
    plenum/roof above the ceiling, so the topmost layer is legitimately air.
    """
    _lo, hi, _floor, ceil_y = room_band(grid)
    assert ceil_y is not None, (
        "no ceiling slab found above the room band - ceiling reflection is a dominant "
        "indoor path and cannot be modelled.")
    solid = float((grid[:, ceil_y, :] != AIR).mean())
    assert solid > 0.5, f"ceiling layer y={ceil_y} is only {100*solid:.1f}% solid"
    assert ceil_y > hi, "ceiling must sit above the room band"


@_defect
def test_floor_slab_is_concrete(grid):
    """A floor slab modelled as drywall has the wrong permittivity for the floor bounce."""
    lo, _hi, _floor, _c = room_band(grid)
    slab = grid[:, :lo, :]
    solid = slab[slab != AIR]
    assert solid.size > 0, "no floor slab at all"
    frac_concrete = float((solid == CONCRETE).mean())
    assert frac_concrete > 0.5, (
        f"floor slab (y<{lo}) is only {100*frac_concrete:.1f}% concrete "
        f"(dominant class = {np.bincount(solid).argmax()}); "
        "eps' 2.9 vs 5.24 gives the wrong floor-bounce reflection coefficient.")


@_defect
def test_material_classes_are_represented(grid):
    """The catch-all regex must not collapse the building into one class."""
    tot = grid.size
    frac = {c: float((grid == c).mean()) for c in range(6)}
    assert frac[CONCRETE] > 0.03, f"concrete only {100*frac[CONCRETE]:.3f}% (expect >3%)"
    assert frac[FURNITURE] > 0.02, f"furniture only {100*frac[FURNITURE]:.3f}% (expect >2%)"
    assert frac[DRYWALL] < 0.18, (
        f"drywall {100*frac[DRYWALL]:.1f}% - the '.*' catch-all is still swallowing structure")


@_defect
def test_valid_tx_mask_is_a_subset(grid):
    """valid_tx should be a clearance-carved subset of inside, not identical to it."""
    from conftest import SIM3D
    import numpy as _np
    vt = _np.load(SIM3D / "valid_tx_mask.npy")
    ins = _np.load(SIM3D / "inside_mask.npy")
    assert vt.sum() <= ins.sum()
    assert vt.sum() < ins.sum(), (
        "valid_tx_mask is identical to inside_mask - no wall clearance was applied, "
        "so transmitters can be placed inside walls.")


# --------------------------------------------------------------------------- report
def test_report_layer_census(grid, capsys):
    """Always-green: prints the census so regressions are visible in CI output."""
    names = ["air", "drywall", "concrete", "core", "furniture", "glass"]
    lines = ["", "per-Y-layer class census:",
             "  y | " + " ".join(f"{n:>8s}" for n in names)]
    for y in range(grid.shape[1]):
        lines.append(f" {y:2d} | " + " ".join(
            f"{int((grid[:, y, :] == c).sum()):8d}" for c in range(6)))
    lines.append("totals: " + "  ".join(
        f"{names[c]}={100*float((grid == c).mean()):.3f}%" for c in range(6)))
    with capsys.disabled():
        print("\n".join(lines))
