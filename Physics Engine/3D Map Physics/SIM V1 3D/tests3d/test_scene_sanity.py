"""
Scene fidelity — the binding error term.

The physics solvers are only as good as the material grid they read. These tests pin the
properties a physically usable indoor scene must have.

STATUS: the `scene_defect` tests are expected to FAIL against the current grid and are
marked xfail. They document the M0.4 re-voxelization acceptance criteria. Once
`voxelize.py` is re-run from Sandbox_Version_3D_Simulation_1.obj (253 named materials,
sealed ceiling, concrete floor slab), flip STRICT_SCENE to True and they become hard gates.

Measured defect in the current grid (262x11x118):
    y=10 is 100% air            -> NO CEILING SLAB
    floor slab (y=1,2) is class 1 drywall (eps'~2.9), should be class 2 concrete (~5.24)
    concrete   284 voxels (0.08%)
    furniture   73 voxels (0.02%)
    drywall  79220 voxels (23.3%)   <- the ".*" -> class 1 catch-all swallowed the structure

Why this gates every reflection result: for a ceiling-mounted AP the floor bounce and the
ceiling bounce are the two dominant specular paths. One currently has the wrong
permittivity; the other does not exist.
"""
from __future__ import annotations

import numpy as np
import pytest

STRICT_SCENE = False        # flip to True after M0.4 re-voxelization
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
@_defect
def test_ceiling_slab_exists(grid):
    """The topmost layer must contain structure, else there is no ceiling bounce."""
    top = grid[:, -1, :]
    solid = float((top != AIR).mean())
    assert solid > 0.5, (
        f"top layer is {100*(1-solid):.1f}% air - NO CEILING SLAB. "
        "Ceiling reflection is a dominant indoor path and cannot be modelled.")


@_defect
def test_floor_slab_is_concrete(grid):
    """A floor slab modelled as drywall has the wrong permittivity for the floor bounce."""
    floor = grid[:, 1, :]
    solid = floor[floor != AIR]
    assert solid.size > 0, "no floor slab at all"
    frac_concrete = float((solid == CONCRETE).mean())
    assert frac_concrete > 0.5, (
        f"floor slab is only {100*frac_concrete:.1f}% concrete "
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
