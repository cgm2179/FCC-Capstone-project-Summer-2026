# Physics Engine — layout & navigation

Two RF-propagation stacks, each a sequence of versions. Everything you **run** sits at a
version folder's root; **library code and data** are sorted into role subfolders.

## Version lineage

- **2D (floor-plane):** `2D/SIM` (v1) → `2D/SIM V2` → `2D/SIM V2.2 (1)` → `2D/SIM V3`
  - `2D/SIM` (v1) also holds the earliest pipeline stages, folded in:
    `step1_geometry/` (was `2D/STEP_1`, floor-plan raster → model) and
    `step2_pathloss/` (was `2D/STEP_2`, Motley-Keenan heatmap).
- **3D (voxel):** `3D Map Physics/SIM V1 3D` → `.../SIM V1.5 3D` → `.../SIM V2`
- `Cross_Validation/` — predicted-vs-measured RSRP; left as-is.

## Folder convention (every version folder)

| Location | Holds |
|---|---|
| **(root)** | the bootstrap shim + the drivers you invoke (`run_*`, `phase_*`, `export_*`, `voxelize*`, `fetch_*`, `fw_solver/infer/export`, dataset generators) |
| **`physics/`** | EM cores, engines, FDTD/full-wave, near/far-field, hybrid, extractors |
| **`surrogate/`** | U-Net / JAX models, training, featurization (the "unet-specific" code) |
| **`helpers/`** | bands, antenna patterns, georef, catalogs, perf, config, contracts (getters/config) |
| **`assets/`** | `.npy/.npz/.pt/.json/.png` data (scene inputs, weights, outputs) |
| **`docs/`**, **`notebooks/`** | `.md/.pdf` reference · `.ipynb` |

## Role map (bucketed library modules)

| Folder | `physics/` | `surrogate/` | `helpers/` |
|---|---|---|---|
| `2D/SIM` | engine_v2, engine_v2_torch, physics_v2 | — | — |
| `2D/SIM V2.2 (1)` | engine_v2, engine_v2_torch, physics_v2 | — | — |
| `2D/SIM V3` | fullwave2d, nearfield, plane_extract, hybrid_field, hybrid_city, fw_field3d, fw_field3d_floor | fw_unet2d, fw_unet3d | bands_v3, antenna_patterns, city_georef, fw_bs_catalog, perf_v3 |
| `SIM V1 3D` | engine_3d, physics_3d | dataset_3d, models_3d, plane_surrogate_3d | contracts |
| `SIM V1.5 3D` | engine, pathloss_models | — | link_budget, pathloss_config |
| `SIM V2` (3D) | fullwave3d, fw_fdtd_jax, far_field_3d, nearfield_3d, hybrid_field_3d, solid_extract, fw_field3d, fw_field3d_floor | unet3d_jax, unet3d_train_jax | bands_v1_for_3D, antenna_patterns_3D, cad_materials, city_georef, indoor_georef, fw_bs_catalog, perf_v2_3d |

`2D/SIM V2` has no engine of its own (it reuses v1's) — only `assets/notebooks/docs`.

## How imports survive the layout

Each folder's bootstrap (`_bootstrap.py` / `bootstrap.py`) adds its own role subfolders — and
those of every engine it reuses — to `sys.path` via an `_add_tree()` helper. So the bare-name
imports the engine uses everywhere (`import fullwave2d`, `import bands_v3`, `import physics_3d`)
still resolve after the modules were bucketed one level deeper.

**Compat symlinks** (e.g. `SIM/physics_v2.py → physics/physics_v2.py`, `2D/STEP_1 → SIM/step1_geometry`,
and the six `SIM V1 3D` lib symlinks) are the shims for code / JSON / notebooks that reference a
file by *filesystem path* rather than by import. **They are load-bearing — do not delete them.**

## Deliberately kept flat (at a folder root)

Some files are **not** bucketed, on purpose:

- **Backend-invoked:** `SIM V1 3D/voxelize.py` and `export_web3.py` — `Backend/server/pipeline.py`
  runs them by absolute path.
- **Root-coupled 3D libs:** `SIM V1 3D/modes_3d.py`, `cache_index.py`, `voxelize_city.py` — they
  compute multi-level parent paths and/or hash sibling files by path; moving them would break
  scene/cache/registration loading.
- **The canonical 3D scene** (`SIM V1 3D/material_grid.npy`, `inside_mask.npy`, `manifest_3d.json`,
  `valid_tx_mask.npy`, `registration_3d.json`) stays at the folder root, which the bootstraps,
  `tests3d/`, cross-validation, and the backend all treat as `SCENE_DIR`.
