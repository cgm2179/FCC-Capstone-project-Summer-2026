#!/usr/bin/env python3
"""
fw_export.py — export the 3-D full-wave field surrogate to ONNX + a browser sidecar.

3-D analog of `2D/SIM V3/fw_export.py`. Exports the PyTorch `UNet3DField`
(`fw_unet3d.pt`) — layout `N,C,X,Y,Z`, `cout=2` (field re/im). Per the plan, the
model is TRAINED in pure JAX (`unet3d_jax`) but EXPORTED through PyTorch, since the
weights are portable both ways (bit-exact parity) and torch.onnx.export is mature.

Dynamic spatial axes let the browser tile any size; X,Z must be multiples of 8
(three (2,1,2) pools), Y may be thin.

  python fw_export.py --ckpt fw_unet3d.pt
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import bootstrap as B
import dataset_3d as D3

HERE = Path(__file__).resolve().parent
WEB = B.SIM3D / "web"


def export_onnx(model, out_path, dummy_xyz=(24, 16, 24), opset=17):
    import torch
    model.eval()
    dummy = torch.zeros(1, len(D3.INPUT_CHANNELS), *dummy_xyz)
    kw = dict(input_names=["x"], output_names=["y"], opset_version=opset,
              dynamic_axes={"x": {0: "n", 2: "x", 3: "y", 4: "z"},
                            "y": {0: "n", 2: "x", 3: "y", 4: "z"}})
    try:
        torch.onnx.export(model, dummy, str(out_path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, dummy, str(out_path), **kw)


def ort_check(onnx_path, xyz=(24, 16, 24)):
    try:
        import onnxruntime as ort
    except Exception:
        return {"onnxruntime": "skipped (not installed)"}
    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.zeros((1, len(D3.INPUT_CHANNELS), *xyz), np.float32)
    y = s.run(["y"], {"x": x})[0]
    return {"onnxruntime": "ok", "out_shape": list(y.shape)}


def parity_check(model, onnx_path, xyz=(24, 16, 24)):
    """Max abs diff Torch vs ONNXRuntime on a random input (target < 0.2 dB-equiv)."""
    try:
        import onnxruntime as ort
        import torch
    except Exception as e:
        return {"parity": f"skipped ({e})"}
    x = np.random.default_rng(0).standard_normal((1, len(D3.INPUT_CHANNELS), *xyz)).astype(np.float32)
    with torch.no_grad():
        yt = model(torch.from_numpy(x)).numpy()
    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    yo = s.run(["y"], {"x": x})[0]
    return {"parity_max_abs": float(np.abs(yt - yo).max())}


def build_contract(band=None, box=None, check=None):
    norm = D3.load_norm(json.loads(B.MANIFEST.read_text()))
    return {
        "spec_version": "fw-unet3d-v1",
        "model_file": "fw_unet3d.onnx",
        "predicts": "complex full-wave field (3-D), phase-reduced",
        "input": {"name": "x", "layout": "N,C,X,Y,Z", "dtype": "float32",
                  "channels": list(D3.INPUT_CHANNELS), "material_classes": 6,
                  "tx_blob_sigma_cells": D3.TX_SIGMA_CELLS,
                  "logdist_divisor": D3.LOGDIST_DIVISOR,
                  "freq_log_lo_mhz": norm.freq_log_lo_mhz,
                  "freq_log_hi_mhz": norm.freq_log_hi_mhz,
                  "size_multiple_xz": 8, "y_axis": "thin, free"},
        "output": {"name": "y", "layout": "N,C,X,Y,Z",
                   "channels": ["field_re", "field_im"], "phase_reduced": True,
                   "reconstruct": "U=(re+j·im)·ref·exp(-j·k·d), d=|voxel-tx|·h, k=2π/λ",
                   "normalized_by": "per-field 99th-percentile |U|"},
        "band": band, "box_train": box,
        "architecture": "UNet3DField(base, cin=9, cout=2)  [no sigmoid]",
        "trained": True, "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "smoke_check": check,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="fw_unet3d.pt")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    from fw_unet3d import load_model3d
    WEB.mkdir(parents=True, exist_ok=True)
    model = load_model3d(str(HERE / a.ckpt), dev="cpu")
    onnx_path = WEB / "fw_unet3d.onnx"
    export_onnx(model, onnx_path, opset=a.opset)
    check = {**ort_check(onnx_path), **parity_check(model, onnx_path)}
    (WEB / "fw_unet3d.json").write_text(json.dumps(build_contract(check=check), indent=2))
    print(f"exported {onnx_path}  {check}")
    print(f"contract  {WEB / 'fw_unet3d.json'}")


if __name__ == "__main__":
    main()
