#!/usr/bin/env python3
"""
fw_export.py — export the full-wave surrogate to ONNX + a browser contract sidecar.

Mirrors SIM V1 3D/export_surrogate_smoke.py, but for the 2-D complex-field model
(cout=2). Dynamic spatial axes let the browser run it on tiles of any size (H,W
must be multiples of 16). The sidecar (fw_unet2d.json) tells the browser how to
build the 9-channel input, denormalize (re,im), and reconstruct U = Ũ·e^{-jkd}.

  python fw_export.py --ckpt assets/fw_unet2d.pt
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # engine root, for _bootstrap
import _bootstrap as B
import dataset_3d as D3
from fw_unet2d import load_model

HERE = Path(__file__).resolve().parent.parent
WEB = B.SIM3D / "web"


def export_onnx(model, out_path, dummy_hw=(128, 128), opset=17):
    model.eval()
    # channels from the MODEL, not a constant: indoor = 9 (dataset_3d), outdoor
    # base-station = 10 (+ the directivity channel).
    cin = model.enc[0][0].in_channels
    dummy = torch.zeros(1, cin, *dummy_hw)
    kw = dict(input_names=["x"], output_names=["y"], opset_version=opset,
              dynamic_axes={"x": {0: "n", 2: "h", 3: "w"},
                            "y": {0: "n", 2: "h", 3: "w"}})
    # dynamo=False → legacy TorchScript exporter (no onnxscript dep), matching
    # export_surrogate_smoke.py; fall back if the kwarg is unsupported.
    try:
        torch.onnx.export(model, dummy, str(out_path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, dummy, str(out_path), **kw)


def ort_check(onnx_path, cin=None, hw=(128, 128)):
    try:
        import onnxruntime as ort
    except Exception:
        return {"onnxruntime": "skipped (not installed)"}
    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if cin is None:                                     # read C from the graph (9 or 10)
        c = s.get_inputs()[0].shape[1]
        cin = c if isinstance(c, int) else len(D3.INPUT_CHANNELS)
    x = np.zeros((1, cin, *hw), np.float32)
    y = s.run(["y"], {"x": x})[0]
    return {"onnxruntime": "ok", "in_channels": cin, "out_shape": list(y.shape)}


def _bands_to_mhz(bands):
    """Band LABELS -> MHz via bands_v3, so the browser can match a picked frequency
    numerically without parsing labels (WiFi labels carry no MHz). Unknowns skipped."""
    out = []
    try:
        import bands_v3
        for lb in (bands or []):
            try:
                out.append(round(bands_v3.get(lb).f_mhz))
            except Exception:
                pass
    except Exception:
        pass
    return out


def _directivity_spec():
    """Contract block for the 10th (base-station) input channel: the per-family
    azimuth directivity gain the browser must synthesize (JS port of
    antenna_patterns.pattern_gain). Panel defaults; a dish/yagi/… override via kind."""
    import antenna_patterns as AP
    hp, am = AP.params("panel")
    return {
        "channel": len(D3.INPUT_CHANNELS),         # index 9, appended after the 9 dataset_3d channels
        "name": "directivity",
        "default_kind": "panel",
        "hpbw_deg": hp, "am_db": am,
        "families": {k: {"hpbw_deg": v[0], "am_db": v[1]} for k, v in AP.PATTERNS.items()},
        "formula": "bearing = atan2(-(i - tx_i), (j - tx_j)) deg;  "
                   "d = wrap180(bearing - boresight_deg);  "
                   "atten_db = min(12*(d/hpbw_deg)^2, am_db);  gain = 10^(-atten_db/20)  (0..1)",
    }


def build_contract(bands=None, box=None, check=None, cin=9,
                   model_file="fw_unet2d.onnx", directivity=False):
    """The fw_unet2d.json / fw_bs.json sidecar. `bands` = the training-band LABELS
    (the notebook's BANDS) — the ground truth for what the model saw, so ALWAYS pass
    it for a multiband model. `trained_mhz` is derived so the front-ends can match by
    MHz. `directivity=True` (cin=10, base-station model) appends the directivity
    channel spec so the browser builds the 10th channel from the Tx boresight."""
    norm = D3.load_norm(json.loads(B.MANIFEST.read_text()))
    channels = list(D3.INPUT_CHANNELS)
    inp = {
        "name": "x", "layout": "N,C,H,W", "dtype": "float32",
        "channels": channels + (["directivity"] if directivity else []),
        "in_channels": cin,
        "material_classes": 6,
        "tx_blob_sigma_cells": D3.TX_SIGMA_CELLS,
        "logdist_divisor": D3.LOGDIST_DIVISOR,
        "freq_log_lo_mhz": norm.freq_log_lo_mhz,
        "freq_log_hi_mhz": norm.freq_log_hi_mhz,
        "size_multiple": 16,
    }
    if directivity:
        inp["directivity"] = _directivity_spec()
    return {
        "spec_version": "fw-unet2d-v1",
        "model_file": model_file,
        "predicts": "complex full-wave field, phase-reduced",
        "input": inp,
        "output": {
            "name": "y", "layout": "N,C,H,W",
            "channels": ["field_re", "field_im"],
            "phase_reduced": True,
            "reconstruct": "U(x) = (re + j*im) * exp(-j*k*d),  k=2*pi/lambda,  "
                           "d = |x - tx| metres;  envelope |U|, animation Re{U e^{jwt}}",
            "normalized_by": "per-field 99th-percentile |U| (relative; re-anchor to "
                             "FSPL for absolute dB)",
        },
        "bands_mhz": bands,
        "trained_mhz": _bands_to_mhz(bands),
        "box_train": box,
        "architecture": f"UNet2D(base, cin={cin}, cout=2)",
        "trained": True,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "smoke_check": check,
    }


def write_contract(bands=None, box=None, check=None, path=None, cin=9,
                   model_file="fw_unet2d.onnx", directivity=False):
    """Write the sidecar next to the ONNX. Call this WHENEVER you export the ONNX —
    `export_onnx()` alone does not, which is exactly how `bands_mhz` went stale (a
    multiband model shipped with a single-band contract)."""
    path = path or (WEB / (Path(model_file).stem + ".json"))
    path.write_text(json.dumps(
        build_contract(bands, box, check, cin=cin, model_file=model_file,
                       directivity=directivity), indent=2))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["indoor", "bs"], default="indoor",
                    help="indoor = 9-ch floor-plan model (fw_unet2d); bs = 10-ch "
                         "outdoor base-station model (fw_bs, + directivity channel).")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint; default fw_unet2d.pt (indoor) / fw_bs.pt (bs).")
    ap.add_argument("--data-meta", default=None,
                    help="dataset_meta.json for bands/box; default per --model.")
    ap.add_argument("--bands", nargs="+", default=None,
                    help="training band LABELS for the contract, e.g. --bands "
                         "LTE_B71_617 LTE_B13_751 NR_n41_2506 ...  Pass this for a "
                         "MULTIBAND model so the sidecar isn't written single-band; "
                         "overrides the bands read from --data-meta.")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    is_bs = args.model == "bs"
    ckpt = args.ckpt or ("assets/fw_bs.pt" if is_bs else "assets/fw_unet2d.pt")
    dmeta_path = args.data_meta or (
        "fw_data_bs/dataset_meta.json" if is_bs else "fw_data_617/dataset_meta.json")
    onnx_name = "fw_bs.onnx" if is_bs else "fw_unet2d.onnx"

    WEB.mkdir(parents=True, exist_ok=True)
    model = load_model(HERE / ckpt, dev="cpu")
    cin = model.enc[0][0].in_channels
    onnx_path = WEB / onnx_name
    export_onnx(model, onnx_path, opset=args.opset)
    check = ort_check(onnx_path, cin=cin)

    dmeta = {}
    dm_path = HERE / dmeta_path
    if dm_path.exists():
        dmeta = json.loads(dm_path.read_text())
    bands = args.bands if args.bands else dmeta.get("bands")
    path = write_contract(bands=bands, box=dmeta.get("box") or 128, check=check,
                          cin=cin, model_file=onnx_name, directivity=(cin >= 10))
    print(f"exported {onnx_path}  {check}")
    print(f"contract  {path}")


if __name__ == "__main__":
    main()
