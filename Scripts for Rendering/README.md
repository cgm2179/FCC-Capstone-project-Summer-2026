# Scripts for Rendering/

Home for **new** standalone render / model-export scripts.

The 3D CAD model the viewer loads is currently built by scripts that live in
`../STEP_1/` (left in place — the pipeline references them):

| Script | Builds |
|---|---|
| `../STEP_1/build_model.py` | raster → 3D model (`STEP_1/model_3d/`) |
| `../STEP_1/build_model_vector.py` | vector → 3D model (`STEP_1/model_3d_vector/`) |
| `../STEP_1/rasterize_floorplan.py` | floor-plan raster + material grid |

The viewer (`../Frontend/js/viewer3d.js`) loads the exported `.obj` folders from
the repo root (`Indoor 7th floor v2*.obj/`); those stay put — it references them
by URL.

**Convention:** new one-off render / export tooling goes here; leave the STEP_1
builders where the pipeline expects them.
