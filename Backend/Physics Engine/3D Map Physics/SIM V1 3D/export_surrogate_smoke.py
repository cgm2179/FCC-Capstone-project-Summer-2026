#!/usr/bin/env python3
"""Export an untrained M4 smoke surrogate for browser integration tests.

This is not a trained propagation model and must not be evaluated as one. It
ships the exact ONNX/JSON interface the browser needs so the DL tier can be
exercised end-to-end while the full physics dataset remains gated by Pre-M4.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import dataset_3d as D
from models_3d import UNet3D

HERE = Path(__file__).resolve().parent
WEB = HERE / "web"


def export_onnx(model, out_path: Path, shape: tuple[int, int, int], opset: int) -> None:
    """Write dynamic-spatial ONNX with canonical `x`/`y` names."""
    import torch

    dummy = torch.zeros(1, len(D.INPUT_CHANNELS), *shape, dtype=torch.float32)
    kwargs = dict(
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={
            "x": {2: "nx", 3: "ny", 4: "nz"},
            "y": {2: "nx", 3: "ny", 4: "nz"},
        },
        opset_version=opset,
    )
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        try:
            torch.onnx.export(model, dummy, str(out_path), dynamo=False, **kwargs)
        except TypeError:
            torch.onnx.export(model, dummy, str(out_path), **kwargs)


def maybe_check_ort(onnx_path: Path, shape: tuple[int, int, int]) -> dict:
    """Run a best-effort ONNX Runtime smoke pass if the package is installed."""
    try:
        import onnxruntime as ort
    except ImportError:
        return {"onnxruntime": "skipped (not installed)"}

    rng = np.random.default_rng(0)
    x = rng.random((1, len(D.INPUT_CHANNELS), *shape), dtype=np.float32)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    y = sess.run(["y"], {"x": x})[0]
    ok = y.shape == (1, len(D.OUTPUT_CHANNELS), *shape) and np.isfinite(y).all()
    return {
        "onnxruntime": "ok" if ok else "failed",
        "ort_input": sess.get_inputs()[0].name,
        "ort_output": sess.get_outputs()[0].name,
        "ort_shape": list(y.shape),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=int, default=4,
                    help="UNet base width; 4 keeps smoke export light for CI")
    ap.add_argument("--dummy-shape", default="32,8,32",
                    help="NX,NY,NZ used for tracing; exported axes remain dynamic")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-ort-check", action="store_true",
                    help="skip optional onnxruntime smoke inference")
    args = ap.parse_args(argv)

    import torch

    shape = tuple(int(v) for v in args.dummy_shape.split(","))
    if len(shape) != 3 or min(shape) < 8:
        raise SystemExit("--dummy-shape must be NX,NY,NZ with each dimension >= 8")

    torch.manual_seed(0)
    np.random.seed(0)

    manifest = json.loads((HERE / "manifest_3d.json").read_text())
    material_grid = np.load(HERE / "material_grid.npy")
    model = UNet3D(cin=len(D.INPUT_CHANNELS), cout=len(D.OUTPUT_CHANNELS), base=args.base)
    n_params = sum(p.numel() for p in model.parameters())

    onnx_path = WEB / "pl_unet3d.onnx"
    json_path = WEB / "pl_unet3d.json"
    export_onnx(model, onnx_path, shape, args.opset)

    metrics = {
        "smoke": True,
        "rmse_pl_db": None,
        "rmse_tau_ns": None,
        "note": "Untrained random weights; validates ONNX/browser plumbing only.",
    }
    extra = {
        "trained": False,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "architecture": "UNet3D",
        "base": int(args.base),
        "params": int(n_params),
        "dummy_shape": list(shape),
        "dynamic_spatial_axes": True,
        "torch": torch.__version__,
    }
    contract = D.surrogate_contract(
        manifest,
        material_grid,
        bands=manifest.get("freqs_mhz", []),
        mechanisms=["path_loss"],
        mode="indoor",
        metrics=metrics,
        extra=extra,
    )
    if not args.no_ort_check:
        contract.setdefault("smoke_check", {}).update(maybe_check_ort(onnx_path, shape))
    D.write_surrogate_contract(json_path, contract)

    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KiB)")
    print(f"wrote {json_path} ({json_path.stat().st_size / 1024:.1f} KiB)")
    print("smoke model is untrained; do not use for RMSE claims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
