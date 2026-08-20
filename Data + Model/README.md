# Data + Model

The datasets, 3D models, and project manifests the dashboard boots from. A root symlink
`Data → Data + Model` keeps every hard-coded `Data/…` URL (in the manifests, `Frontend_Data_Display.html`,
and the SIM studios) resolving — so files here are referenced as `Data/<file>`, **not** by this folder's
real name. Don't move files into subfolders without updating those references.

## Walk / RF payloads
| File | Role | Loaded by (project) |
|---|---|---|
| `records_data.js` | Indoor walk — position-averaged records (the signal map). | `projects/indoor_2d_default.js`, `indoor_3d_default.js` |
| `timeseries_data.js` | Indoor walk — the time-ordered series (playback). | indoor 2D / 3D |
| `outdoor_walk.js` | NoMa outdoor walk — records. | `projects/outdoor_2d_default.js`, `outdoor_3d_default.js` |
| `outdoor_timeseries.js` | NoMa outdoor walk — time series (Time Elapse replay). | outdoor 2D / 3D |
| `base_stations.js` | Surveyed base-station catalog (PCI ↔ operator). | all modes |

## Georef / basemap
| File | Role | Notes |
|---|---|---|
| `floorplan_image.js` | 7th-floor plan raster (base64). | indoor modes |
| `floorplan_meta.js` | Its georeference (affines + GCPs). | indoor modes |
| `basemap_image.js` | OSM under-layer raster. | *git-ignored, generated* by `scripts/prepare_indoor_basemap.py` |
| `pl2d_crossval.js` | 2D outdoor-to-indoor path-loss overlay. | *git-ignored, generated* |

## City grids (outdoor studio)
| File | Role |
|---|---|
| `noma_city2d.js` | Penetrable NoMa city grid (baked by `scripts/bake_noma_city2d.py`). |
| `noma_clutter.js` | NoMa clutter layer. |

## `models/` — CAD
| Entry | Role | Loaded by |
|---|---|---|
| `7th_floor_full.glb`, `7th_floor_3ff8432c.glb` | Indoor 7th-floor GLB meshes. | `Frontend/Map Coverage/viewer3d.js`, the SIM 3D studios |
| `Indoor 7th floor v2 First Render.obj/` | Multi-file OBJ bundle (`6afecb6b….obj` + `.mtl` + textures). | `viewer3d.js` model picker |
| `Indoor 7th floor v2.obj-HTML Test 1/`, `…Test 2/` | Alternate OBJ bundles for the model picker. | `viewer3d.js` |
| `NoMa_DC/` | Outdoor NoMa city render (`Actual_Outdoor_Sim…obj`) + building OBJs. | `projects/outdoor_3d_default.js`, the outdoor 3D studio |
| `convert_obj_to_glb.mjs` | Helper: OBJ → GLB conversion (kept for re-generating the GLBs). | — |

> The odd `…obj-HTML Test 1/2` and `…First Render.obj/` **directory** names are load-bearing — they are
> the exact URL-encoded paths `viewer3d.js` fetches. Rename only alongside a `viewer3d.js` update.

## `projects/` — the project/dataset layer
`registry.js` + one `*_default.js` manifest per mode (indoor/outdoor · 2D/3D). Each manifest **names** the
data files above that its workspace boots from; `Frontend/landing/project_loader.js` injects them. See
[`projects/README.md`](projects/README.md) for the manifest schema.

## Regenerating vs. shipping
The committed `Data/*.js` above are what the built-in **Default** projects load, so a fresh clone runs the
dashboard out of the box with no build step. **Rebuilding** them (via `scripts/bake_*.py` /
`prepare_indoor_basemap.py`) needs the raw scanner exports in `ARCHIVE/raw_walk_data/` and the source
walk-test folder (`FCC_Walk_Outdoor_Indoor_Full/`), which are **not shipped** — regeneration is a maintainer
task, not required to use the app. Source floor-plan ground truth: `Essentials + HTML/`.
