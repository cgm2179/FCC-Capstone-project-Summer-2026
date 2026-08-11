# Scripts + Scripts for Rendering/

Archived home for **standalone render / model-export** tooling notes.

Active bake / export scripts used by the live dashboard stay at repo-root
`scripts/` (`bake_outdoor_*.py`, `bake_noma_*.py`, `prepare_indoor_basemap.py`, …).

The 3D CAD model the viewer loads is built by scripts in
`Physics Engine/2D/STEP_1/` (left in place — the pipeline references them):

| Script | Builds |
|---|---|
| `Physics Engine/2D/STEP_1/build_model.py` | raster → 3D model |
| `Physics Engine/2D/STEP_1/build_model_vector.py` | vector → 3D model |
| `Physics Engine/2D/STEP_1/rasterize_floorplan.py` | floor-plan raster + material grid |

**Convention:** one-off / superseded render tooling notes go here; leave STEP_1
builders and root `scripts/` where the pipeline expects them.
