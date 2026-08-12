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

usage:
  python "SIM V1 3D/export_web3.py"
  python "SIM V1 3D/export_web3.py" --sim-dir city_demo/NoMa_DC_tile \
      --out web/sim_assets_outdoor_3d.js --global SIM3D_ASSETS_OUTDOOR
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", default=str(HERE),
                    help="directory containing manifest_3d/material_grid/inside_mask")
    ap.add_argument("--out", default=str(HERE / "web" / "sim_assets_3d.js"),
                    help="output JS file")
    ap.add_argument("--global", dest="global_name", default="SIM3D_ASSETS",
                    help="window global to assign")
    args = ap.parse_args()

    sim_dir = Path(args.sim_dir).expanduser().resolve()
    man = json.loads((sim_dir / "manifest_3d.json").read_text())
    M = np.load(sim_dir / "material_grid.npy")
    inside = np.load(sim_dir / "inside_mask.npy")

    out = dict(
        manifest_3d=man,
        grid_shape=man["grid_shape"],
        axes=man["axes"],
        grid_b64=b64(M.astype(np.int8)),            # material class per voxel
        inside_b64=b64(inside.astype(np.uint8)),    # interior mask (display/loss)
    )
    p = Path(args.out).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"window.{args.global_name} = " + json.dumps(out) + ";\n")
    print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")
    print("next: train Phase C → pl_unet3d.onnx into web/, then wire the viewer.")


if __name__ == "__main__":
    main()
