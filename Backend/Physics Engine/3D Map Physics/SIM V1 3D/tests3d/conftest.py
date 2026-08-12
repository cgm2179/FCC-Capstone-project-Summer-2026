"""
Shared fixtures and import bootstrap for the 3D simulation test harness.

The 3D pipeline is a flat-import directory (`import engine_3d`, `import physics_3d`),
and `physics_3d` itself path-searches for the 2D `physics_v2`. This module reproduces
that search once so every test file can just `import engine_3d`.

Run:  python TESTS3D/run_all.py --fast      (or)  pytest "SIM V1 3D/tests3d" -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# --------------------------------------------------------------------------- paths
SIM3D = Path(__file__).resolve().parent.parent          # .../SIM V1 3D
ROOT = SIM3D.parent.parent.parent                        # repo root
MECH = SIM3D.parent / "Wave Behavior" / "Enivronmental Interaction"
WAVEGEN = SIM3D.parent / "Wave Behavior" / "Wave Generation"
SIM2D = ROOT / "Physics Engine" / "2D" / "SIM"
TX_OBJ = SIM3D.parent / "Object and Tranmission" / "Transmitter Objects"


def _bootstrap_paths() -> None:
    for p in (SIM3D, SIM2D, MECH, WAVEGEN, TX_OBJ, Path(__file__).resolve().parent):
        s = str(p)
        if p.exists() and s not in sys.path:
            sys.path.insert(0, s)


_bootstrap_paths()


# --------------------------------------------------------------------------- marks
def pytest_configure(config):
    for m, desc in [
        ("slow", "takes more than ~10 s"),
        ("gpu", "requires CUDA / Colab"),
        ("needs_scanner_data", "requires the PCTEL walk-test CSVs"),
        ("golden", "compares against a committed baseline JSON"),
        ("scene_defect", "asserts a scene-fidelity property; xfail until M0.4 re-voxelization"),
    ]:
        config.addinivalue_line("markers", f"{m}: {desc}")


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def sim3d_dir() -> Path:
    return SIM3D


@pytest.fixture(scope="session")
def manifest() -> dict:
    return json.loads((SIM3D / "manifest_3d.json").read_text())


@pytest.fixture(scope="session")
def scene():
    """The real voxel scene (SceneV3). Skips if the grid has not been generated."""
    import engine_3d

    if not (SIM3D / "material_grid.npy").exists():
        pytest.skip("material_grid.npy not generated - run voxelize.py")
    sc, _man = engine_3d.load_scene(SIM3D)
    return sc


@pytest.fixture(scope="session")
def grid(manifest) -> np.ndarray:
    p = SIM3D / "material_grid.npy"
    if not p.exists():
        pytest.skip("material_grid.npy not generated")
    return np.load(p)


@pytest.fixture(scope="session")
def inside_mask() -> np.ndarray:
    p = SIM3D / "inside_mask.npy"
    if not p.exists():
        pytest.skip("inside_mask.npy not generated")
    return np.load(p)


@pytest.fixture(scope="session")
def registration() -> dict:
    """px -> local metres -> voxel transform. Written by M0.5."""
    p = SIM3D / "registration_3d.json"
    if not p.exists():
        pytest.skip("registration_3d.json not fitted yet (M0.5)")
    return json.loads(p.read_text())


@pytest.fixture
def rng() -> np.random.Generator:
    """Always seeded - every test in this suite is deterministic."""
    return np.random.default_rng(0)
