# Repo structure — Frontend / Backend

The browser code and the server/engine code are now split under **`Frontend/`** and **`Backend/`**.
The physical folders moved there; **root compat symlinks** preserve every existing path (so the
hard-coded `<script>`/iframe URLs, the manifests, and `Backend/server/pipeline.py`'s script paths
all keep working unchanged — the same convention the repo already used for `Data → Data + Model`
and `SIM → Physics Engine/2D/SIM`). Browse `Frontend/` or `Backend/` for the real files; the old
names still resolve.

## Frontend/ — the browser app
| Path | What it is |
|---|---|
| `Frontend_Data_Display.html` | entry page (stays at repo root; served at `/`) |
| `Frontend/landing/` | screen router (`landing.js`), project layer loader (`project_loader.js`), styles |
| `Frontend/Map Coverage/` | the dashboard (`dashboard.js`), 3D viewer (`viewer3d.js`), georef, outdoor OSM view |
| `Frontend/Statistics/` | histograms + antenna-pattern sub-tab (`spectrum.js`) |
| `Frontend/Simulation/` | the SIM V3 studios (`fw_studio*.html`), solvers (`fw_solve2d.js`, `fw_pathloss2d.js`), AI/ML gate (`aiml_gate.js`) |
| `Frontend/Time Elapse/` | outdoor walk replay (`outdoor_timelapse.html`) |
| `Data + Model/` (via `Data →`) | the datasets + built-in project manifests (`Data/projects/`) — data, kept at root |

Root symlinks kept for compat: `Map Coverage`, `Statistics`, `Simulation`, `Time Elapse` → their
`Frontend/…` homes.

## Backend/ — server + physics/engine
| Path | What it is |
|---|---|
| `Backend/server/` | the Flask project backend (`app.py`, `pipeline.py`, `validate.py`, `workspace_store.py`) |
| `Backend/Physics Engine/` | the 2D/3D RF engines (STEP rasterizers, SIM V1/V2/V3, voxelizers, cross-validation) |
| `Backend/scripts/` | the bake / rasterize scripts (`bake_outdoor_*`, `prepare_indoor_basemap`, `make_fake_walktest`, …) |
| `ARCHIVE/` | superseded/early code (`EARLY Analysis/build_all_bands_data.py` is still used by the pipeline) — kept at root |
| `SIM →`, `SIM3D →`, `TESTS3D →` | symlinks into `Backend/Physics Engine/…` (resolve through the `Physics Engine` root symlink) |

Root symlinks kept for compat: `Physics Engine`, `scripts` → their `Backend/…` homes.

## Why symlinks instead of rewriting references
Rewriting every reference to the canonical `Frontend/…` / `Backend/…` paths would break the **deep
relative** references inside the moved files (e.g. `Frontend/Simulation/fw_studio2d.html` loads
`../Physics Engine/…`, which only resolves as `/Physics Engine/…` through the root symlink). Keeping
the root symlinks means all URLs and script paths resolve exactly as before, with zero churn.
Verified: the default workspace + the SIM studio both boot; the backend `--self-test` /
`--fake-project` pass.
