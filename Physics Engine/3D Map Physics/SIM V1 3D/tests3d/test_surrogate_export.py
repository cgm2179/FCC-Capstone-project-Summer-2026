from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import dataset_3d as D
import plane_surrogate_3d as P

HERE = Path(__file__).resolve().parent
SIM3D = HERE.parent
WEB = SIM3D / "web"


def test_exported_contract_has_browser_required_keys():
    p = WEB / "pl_unet3d.json"
    assert p.exists(), "run SIM3D/export_surrogate_smoke.py to ship the sidecar"
    c = json.loads(p.read_text())

    for key in ("spec_version", "input", "output", "grid_shape", "scene_sha"):
        assert key in c
    assert c["spec_version"] == D.SPEC_VERSION
    assert len(c["input"]["channels"]) == 9
    assert c["input"]["channels"] == list(D.INPUT_CHANNELS)
    assert c["input"]["name"] == "x"
    assert c["output"]["name"] == "y"
    assert c["output"]["channels"] == list(D.OUTPUT_CHANNELS)
    assert c["metrics"]["smoke"] is True
    assert c["metrics"]["rmse_pl_db"] is None
    assert c["trained"] is False


def test_exported_onnx_uses_canonical_io_names():
    p = WEB / "pl_unet3d.onnx"
    if not p.exists():
        pytest.skip("smoke ONNX has not been exported")
    try:
        import onnx
    except ImportError:
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("neither onnx nor onnxruntime is installed")
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        assert sess.get_inputs()[0].name == "x"
        assert sess.get_outputs()[0].name == "y"
    else:
        model = onnx.load(str(p))
        assert model.graph.input[0].name == "x"
        assert model.graph.output[0].name == "y"


def test_dataset_build_input_volume_matches_lower_level_helpers(manifest, grid):
    norm = D.load_norm(manifest)
    tx = (60, 5, 63)
    x = D.build_input_volume(grid, tx, 3500.0, norm, manifest["cell_size_m"])
    ref = D.make_input(D.static_channels(grid), tx, norm.freq_feature(3500.0),
                       D.voxel_coords(grid.shape), manifest["cell_size_m"])
    assert x.shape == (9,) + grid.shape
    assert x.dtype == np.float32
    assert np.allclose(x, ref)


def test_plane_surrogate_extract_compose_roundtrips_volume():
    v = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
    planes = P.extract_planes(v)
    assert planes["xy"].shape == (5, 4, 3)
    assert planes["yz"].shape == (4, 3, 5)
    assert planes["zx"].shape == (3, 5, 4)
    got = P.compose_planes(planes, v.shape)
    assert np.allclose(got, v)


def test_plane_surrogate_single_axis_leaves_non_plane_contract_clear():
    v = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    planes = P.extract_planes(v, P.PlaneSpec(axes=("xy",)))
    got = P.compose_planes(planes, v.shape, P.PlaneSpec(axes=("xy",)))
    assert np.allclose(got, v)
