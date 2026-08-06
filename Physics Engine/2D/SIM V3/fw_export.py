#!/usr/bin/env python3
"""
fw_export.py — export the full-wave surrogate to ONNX + a browser contract sidecar.

Mirrors SIM V1 3D/export_surrogate_smoke.py, but for the 2-D complex-field model
(cout=2). Dynamic spatial axes let the browser run it on tiles of any size (H,W
must be multiples of 16). The sidecar (fw_unet2d.json) tells the browser how to
build the 9-channel input, denormalize (re,im), and reconstruct U = Ũ·e^{-jkd}.

  python fw_export.py --ckpt fw_unet2d.pt
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import _bootstrap as B
import dataset_3d as D3
from fw_unet2d import load_model

HERE = Path(__file__).resolve().parent
WEB = B.SIM3D / "web"


def export_onnx(model, out_path, dummy_hw=(128, 128), opset=17):
    model.eval()
    dummy = torch.zeros(1, len(D3.INPUT_CHANNELS), *dummy_hw)
    kw = dict(input_names=["x"], output_names=["y"], opset_version=opset,
              dynamic_axes={"x": {0: "n", 2: "h", 3: "w"},
                            "y": {0: "n", 2: "h", 3: "w"}})
    # dynamo=False → legacy TorchScript exporter (no onnxscript dep), matching
    # export_surrogate_smoke.py; fall back if the kwarg is unsupported.
    try:
        torch.onnx.export(model, dummy, str(out_path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, dummy, str(out_path), **kw)


def ort_check(onnx_path, hw=(128, 128)):
    try:
        import onnxruntime as ort
    except Exception:
        return {"onnxruntime": "skipped (not installed)"}
    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.zeros((1, len(D3.INPUT_CHANNELS), *hw), np.float32)
    y = s.run(["y"], {"x": x})[0]
    return {"onnxruntime": "ok", "out_shape": list(y.shape)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="fw_unet2d.pt")
    ap.add_argument("--data-meta", default="fw_data_617/dataset_meta.json")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    WEB.mkdir(parents=True, exist_ok=True)
    model = load_model(HERE / args.ckpt, dev="cpu")
    onnx_path = WEB / "fw_unet2d.onnx"
    export_onnx(model, onnx_path, opset=args.opset)
    check = ort_check(onnx_path)

    manifest = json.loads(B.MANIFEST.read_text())
    norm = D3.load_norm(manifest)
    dmeta = {}
    dm_path = HERE / args.data_meta
    if dm_path.exists():
        dmeta = json.loads(dm_path.read_text())

    contract = {
        "spec_version": "fw-unet2d-v1",
        "model_file": "fw_unet2d.onnx",
        "predicts": "complex full-wave field, phase-reduced",
        "input": {
            "name": "x", "layout": "N,C,H,W", "dtype": "float32",
            "channels": list(D3.INPUT_CHANNELS),
            "material_classes": 6,
            "tx_blob_sigma_cells": D3.TX_SIGMA_CELLS,
            "logdist_divisor": D3.LOGDIST_DIVISOR,
            "freq_log_lo_mhz": norm.freq_log_lo_mhz,
            "freq_log_hi_mhz": norm.freq_log_hi_mhz,
            "size_multiple": 16,
        },
        "output": {
            "name": "y", "layout": "N,C,H,W",
            "channels": ["field_re", "field_im"],
            "phase_reduced": True,
            "reconstruct": "U(x) = (re + j*im) * exp(-j*k*d),  k=2*pi/lambda,  "
                           "d = |x - tx| metres;  envelope |U|, animation Re{U e^{jwt}}",
            "normalized_by": "per-field 99th-percentile |U| (relative; re-anchor to "
                             "FSPL for absolute dB)",
        },
        "bands_mhz": dmeta.get("bands"),
        "box_train": dmeta.get("box"),
        "architecture": "UNet2D(base, cin=9, cout=2)",
        "trained": True,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "smoke_check": check,
    }
    (WEB / "fw_unet2d.json").write_text(json.dumps(contract, indent=2))
    print(f"exported {onnx_path}  {check}")
    print(f"contract  {WEB / 'fw_unet2d.json'}")


if __name__ == "__main__":
    main()
