# SIM V1 3D — Volumetric RF Path-Loss Pipeline

A **true 3-D** path-loss pipeline: it computes loss *through the real geometry of
the OBJ* → `PL(x, y, z)`, then trains an ML surrogate so the volume can be
redrawn in one forward pass. This is the volumetric fork the 2-D skill spec
documents but defers (`(x,y,z)` grid, §8) — distinct from:

- **`SIM/`** — the deployed **2-D** pipeline (`PL(x,y)` on the floor grid), and
- **`viewer3d.js buildTerrain()`** — which renders a 2-D field as a 3-D *terrain*
  surface ("loss = height"). That is a render trick, not 3-D physics.

Here the physics domain itself is a voxel volume. Design mirrors `SIM/` so the
two read as siblings (see [MODEL_CARD_3D.md](MODEL_CARD_3D.md) for the physics
scope and fidelity ladder).

## Pipeline

| Stage | File | 2-D analog | Runs |
|---|---|---|---|
| A · voxelize | `voxelize.py` | `phase_a.py` | local |
| — engine | `engine_3d.py` (`SceneV3`) | `engine_v2.py` | local / Colab |
| — one calc | `run_one_calc.py` | — | local |
| B · dataset | `phase_b3_dataset.ipynb` | `phase_b_dataset.py` | Colab |
| C · train | `phase_c3_train_colab.ipynb` | `phase_c_train_colab_v3.ipynb` | Colab |
| export | `export_web3.py` | `export_web_assets.py` | local |
| config | `manifest_3d.json` | `manifest.json` | — |

## Quickstart (local)

```bash
pip install -r "SIM V1 3D/requirements_3d.txt"

# A — OBJ -> 3-D material grid + masks (writes *.npy, updates manifest_3d.json)
python "SIM V1 3D/voxelize.py"

# one path-loss calculation + preview slices (the experiment)
python "SIM V1 3D/run_one_calc.py"            # -> preview/pl_one_calc.png
python "SIM V1 3D/run_one_calc.py" --freq 2442
```

`run_one_calc.py` prints physics checks (loss ≥ free space everywhere, loss rises
with distance) and writes `preview/pl_volume.npy` + `preview/pl_one_calc.png`.

## Colab (dataset → train → export)

Upload the `SIM V1 3D/` folder (with the `*.npy` grids, `manifest_3d.json`, and
`engine_3d.py`) to Drive, then, in order:

```text
# B — open phase_b3_dataset.ipynb (CPU runtime is fine), set ROOT.
#     Leave SMOKE=True for an 8-position check, then set SMOKE=False and Run All.
#     Writes resumable dataset/shard_*.npz — re-run the generate cell to resume.

# C — open phase_c3_train_colab.ipynb (GPU runtime), set ROOT, Run All.
#     Trains the 3-D UNet, reports test RMSE/MAE in dB vs FSPL baseline,
#     exports web/pl_unet3d.onnx.
```

```bash
# export browser assets (grid + masks + manifest -> sim_assets_3d.js)
python "SIM V1 3D/export_web3.py"
```

## Status

**Built & verified locally**

- `voxelize.py` — OBJ → `262×11×118` grid @ 0.30 m; QGIS-calibrated scale
  (ceiling ≈ 2.85 m); interior mask (45 % of volume).
- `engine_3d.py` — 3-D Motley-Keenan, full field in ~1 s; formula matches the
  deployed 2-D web physics.
- `run_one_calc.py` — one calc; PL 43→158 dB, excess-over-free-space ≥ 0,
  corr(PL, log-dist) > 0.6. See `preview/pl_one_calc.png`.
- `export_web3.py` — `web/sim_assets_3d.js` (0.9 MB).

**Scaffolded (run on Colab)**

- `phase_b3_dataset.ipynb` — generation logic smoke-tested locally; run the
  notebook on Colab (a CPU runtime is fine).
- `phase_c3_train_colab.ipynb` — 3-D UNet + ONNX export; **not executed locally**
  (no GPU/torch here) — expect to iterate.

**Deferred** — see [MODEL_CARD_3D.md](MODEL_CARD_3D.md): 3-D UTD diffraction,
eikonal arrival time, shadow field, GPU dataset engine (`engine_3d_torch`), and
wiring the volume into `viewer3d.js` as the volumetric heatmap.
