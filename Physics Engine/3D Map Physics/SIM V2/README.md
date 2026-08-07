# SIM V2 3D — JAX full-wave pipeline (Phases A–D)

JAX end-to-end 3-D full-wave RF. **FDTD is the offline teacher** that generates training data; the
**3-D U-Net is the runtime student**. Indoor first; outdoor is surrogate/analytic (full-wave is
infeasible over a city).

## Components
- **Phase A — physics (JAX):** [`../../2D/SIM V3/fw_fdtd_jax.py`](../../2D/SIM%20V3/fw_fdtd_jax.py) —
  JAX/XLA port of the SIM V3 scalar full-wave engine. Reuses `FullWaveScene` to build the ITU medium,
  moves only the leapfrog time loop to JAX. **4–11× vs NumPy on CPU**, ~10–50× more on a GPU, parity
  to machine-ε (`python fw_fdtd_jax.py --nx 100 --ny 40 --nz 100` to benchmark).
- **Phase C — model (pure JAX):** [`../../2D/SIM V3/unet3d_jax.py`](../../2D/SIM%20V3/unet3d_jax.py) —
  hand-written 3-D U-Net (`lax.conv_general_dilated`, no Flax/Equinox). **Bit-exact vs the PyTorch
  reference** (`python unet3d_jax.py --parity`). Export to ONNX via PyTorch (weights are portable).
- **`Indoor/Phase_A_test_jax_fdtd.ipynb`** — one full-wave FDTD solve on the 7th-floor model. Ships a
  voxel cache (`phaseA_classes.npz`) so it runs **without** the 344 MB OBJ.

## Colab
1. Open `Indoor/Phase_A_test_jax_fdtd.ipynb` in Colab.
2. **Runtime → Change runtime type → GPU** (JAX auto-uses the GPU backend).
3. Run all — the bootstrap cell clones the repo; the solve runs on the shipped voxel cache.
   (Private repo? Use a tokenized clone URL — see the note in the bootstrap cell.)

## Precision policy
fp32 for physics stepping (fp16/fp8/fp4 are unfit for the leapfrog — accumulation + field dynamic
range), fp16 for data storage, bf16 for U-Net training.

## Phases
A physics (done) · B data-gen notebook (JAX on GPU → fp16 shards) · C U-Net training (optax, bf16) ·
D eval (dB RMSE / Spearman / coherence vs held-out FDTD) + viz (mplot3d for field data;
PyVista/Plotly/three.js for meshes).
