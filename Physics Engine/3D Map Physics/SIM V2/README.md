# SIM V2 3D — JAX-first full-wave pipeline (Phases A–D)

JAX end-to-end 3-D full-wave RF. **FDTD is the offline teacher** that generates training data; the
**pure-JAX 3-D U-Net is the runtime student**. Indoor first; outdoor is surrogate/analytic (full-wave
is infeasible over a city). The 3-D shape the wave takes is set by the solid geometry AND the
transmitter antenna geometry (directional lobes emerge from a conductive backplane in the voxel grid).

## Design principle
JAX only the **compute hot paths** (`fw_fdtd_jax`, `unet3d_jax`, `unet3d_train_jax`). Everything else —
the dimension-agnostic engine, ITU materials (`physics_v2`/`physics_3d`), bands, georef, antenna
patterns, and the analytic 3-D `SceneV3` — is one-time NumPy/Python config, **imported in place** via
`bootstrap.py` (never rewritten). Precision: **fp32 physics / fp16 data / bf16 training**.

## Modules (all under this folder)
| Group | Modules |
|---|---|
| foundations | `bootstrap` · `bands_v1_for_3D` · `perf_v2_3d` · `indoor_georef` · `city_georef` |
| geometry + JAX core | `solid_extract` · `fullwave3d` · `fw_fdtd_jax` · `unet3d_jax` |
| directivity | `antenna_patterns_3D` (az×el + 3-D backplane; ~38× front/back lobe) |
| field runners | `nearfield_3d` · `far_field_3d` · `fw_field3d` · `fw_field3d_floor` · `hybrid_field_3d` |
| solver/infer/export | `fw_solver_v2_3d` · `fw_infer_v1_3d` · `fw_export` (torch→ONNX) |
| drivers | `run_wave3d` · `run_indoor3d` · `run_outdoor3d` · `fw_bs_catalog` |
| data + train | `fw_dataset3d` (Phase B) · `unet3d_train_jax` (Phase C/D) |

## Phases
- **A — physics (done):** `fw_fdtd_jax` (JAX/XLA FDTD, 4–11× vs NumPy on CPU, parity to machine-ε).
  `Indoor/Phase_A_test_jax_fdtd.ipynb` runs one solve on the 7th-floor model (Colab-ready; ships a
  voxel cache).
- **B — data-gen:** `fw_dataset3d.generate3d(band, n_tx, boxes_per, npw, directional_frac)` → per-band
  fp16 shards `x=(N,9,X,Y,Z), y=(N,2,X,Y,Z)` (`spec:"fw-unet3d-v1"`). Run at scale on a Colab GPU.
- **C — train (pure JAX):** `unet3d_train_jax.train(data_dir, epochs, base)` — GroupNorm + optax Adam,
  saves `unet3d_jax.npz`. (`unet3d_jax` also holds a BatchNorm variant, bit-exact vs the PyTorch ref.)
- **D — eval + export:** `unet3d_train_jax.validate(...)` gates tiled prediction vs a fresh 3-D FDTD
  (envelope-dB RMSE / Spearman / coherence). Export to ONNX via `fw_export` (weights → PyTorch
  `UNet3DField` → `torch.onnx.export`).

## Quick start
```bash
python fullwave3d.py --band LTE_B71_617 --npw 6          # one 3-D full-wave solve (JAX)
python run_wave3d.py --antenna panel --azimuth 90 --directivity backplane   # antenna-shaped lobes
python fw_dataset3d.py --band LTE_B71_617 --n-tx 12 --npw 8 --directional-frac 0.4   # Phase B
python unet3d_train_jax.py --data fw_data3d/LTE_B71_617 --epochs 60          # Phase C
python unet3d_train_jax.py --validate --band LTE_B71_617                     # Phase D
```

Reuse-in-place sources (via `bootstrap`): `2D/SIM V3` (engine, bands, nearfield, antenna_patterns),
`SIM V1 3D` (physics_3d, engine_3d/SceneV3, dataset_3d, voxelize_city, scene grid), `2D/SIM`
(physics_v2), `Object and Tranmission/Transmitter Objects` (Antenna_Type_3D geometry).
