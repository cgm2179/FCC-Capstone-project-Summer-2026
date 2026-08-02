#!/usr/bin/env python3
"""
export_web3.py — browser assets for a SIM V1 3D simulator tab.

Mirrors SIM/export_web_assets.py. Writes web/sim_assets_3d.js defining
window.SIM3D_ASSETS with the manifest, the material grid, and the interior mask
(base64, so it loads via <script src> even on file://). The trained volume model
(pl_unet3d.onnx from Phase C) ships alongside and is fetched by onnxruntime-web.

Browser deploy path (documented; the viewer wiring is the follow-on render task):
  1. onnxruntime-web (wasm EP — the WebGPU EP does not yet cover Conv3d) loads
     pl_unet3d.onnx.
  2. Build the 9-channel input for the placed Tx + band (same featurization as
     phase_c3_train_colab.ipynb) → one forward pass → normalized PL volume.
  3. Denormalize: PL_dB = out * norm.pl_range_db + norm.pl_min_db.
  4. Upload as a THREE.Data3DTexture (nx × ny × nz, R32F/R16F) and ray-march it
     in the WebGPU viewer (the volumetric-heatmap render).

usage: python "SIM V1 3D/export_web3.py"
"""
import base64
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def main():
    man = json.loads((HERE / "manifest_3d.json").read_text())
    M = np.load(HERE / "material_grid.npy")
    inside = np.load(HERE / "inside_mask.npy")

    out = dict(
        manifest_3d=man,
        grid_shape=man["grid_shape"],
        axes=man["axes"],
        grid_b64=b64(M.astype(np.int8)),            # material class per voxel
        inside_b64=b64(inside.astype(np.uint8)),    # interior mask (display/loss)
    )
    (HERE / "web").mkdir(exist_ok=True)
    p = HERE / "web" / "sim_assets_3d.js"
    p.write_text("window.SIM3D_ASSETS = " + json.dumps(out) + ";\n")
    print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")
    print("next: train Phase C → pl_unet3d.onnx into web/, then wire the viewer.")


if __name__ == "__main__":
    main()
