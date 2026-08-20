# Directory guide — what every folder holds

The **Indoor/Outdoor Walk-Test Studio**: a browser dashboard for exploring measured 4G/5G/WLAN
walk-test data (map, statistics, time-elapse) plus an RF-propagation simulation sandbox, backed by
a small Flask server that turns raw uploads into "projects."

The tree is split into **`Frontend/`** (everything the browser runs) and **`Backend/`** (the server
+ the physics/engine code). The physical folders live under those two; **root compat symlinks**
keep every old path working (the repo's existing convention — `Data → Data + Model`,
`SIM → Physics Engine/2D/SIM`). See [`REPO_STRUCTURE.md`](REPO_STRUCTURE.md) for *why* it's split
this way; this file lists *what's in each folder*.

---

## Top level

| Entry | Kind | What it is |
|---|---|---|
| `Frontend_Data_Display.html` | file | **The app entry page** (served at `/`). Loads the project layer, then the dashboard shell (Map Coverage / Statistics / Time Elapse / Simulation tabs). |
| `Frontend/` | dir | All browser code (below). |
| `Backend/` | dir | Server + physics/engine code (below). |
| `Data + Model/` | dir | The datasets + 3D models + project manifests (below). Reached as `Data/` too. |
| `Essentials + HTML/` | dir | Source ground-truth assets — the floor-plan PNG + MapInfo `.TAB` georeference the indoor pipeline rasterizes. |
| `ARCHIVE/` | dir | Superseded / early code kept for reference (below). |
| `docs/` | dir | Project docs, plans, roadmaps + the AI/ML guide (below). |
| `misc/` | dir | Ancillary bits corralled out of the root: `misc/unity/` (the vendored ~32 MB Unity WebGL build the Simulation tab embeds) and `misc/.githooks/` (the commit-message hook). |
| `Unity_RF_Simulator/`* | dir | The (separate) Unity C# **source** project. *nested repo, git-ignored — not part of a clone. |
| `.claude/` | dir | Editor/agent config (e.g. `launch.json` dev-server presets). Kept at root — tooling reads it there. |
| `.github/` | dir | `CODEOWNERS` (+ any CI workflows). Kept at root — GitHub only honors it there. |
| `REPO_STRUCTURE.md`, `DIRECTORY.md` | file | This split's rationale, and this guide. |
| **Root symlinks** | link | `Map Coverage`, `Statistics`, `Simulation`, `Time Elapse` → `Frontend/…`; `Physics Engine`, `scripts` → `Backend/…`; `Data → Data + Model`; `SIM`/`SIM3D`/`TESTS3D` → into `Backend/Physics Engine/…`; `AGENTS.md`/`QUICKSTART.md` → `docs/…`. |

---

## `Frontend/` — the browser app

| Folder | Contents |
|---|---|
| `Frontend/landing/` | The front-of-app **screen router** (`landing.js`: chooser → Import → Preprocess → Workspace) and the **project/dataset loader** (`project_loader.js`) + `landing.css`. |
| `Frontend/Map Coverage/` | The **dashboard** (`dashboard.js` + `dashboard.css`) — the signal map, filters, playback; the **3D CAD viewer** (`viewer3d.js`), the floor-plan georeference (`georef.js`), and the outdoor OSM views (`outdoor_view.html`, `osm_buildings_view.html`). |
| `Frontend/Statistics/` | The **Statistics tab** — histograms + the antenna-radiation-pattern sub-tab (`spectrum.js` + `spectrum.css`). |
| `Frontend/Time Elapse/` | The outdoor **walk replay** (`outdoor_timelapse.html`). |
| `Frontend/Simulation/` | The **SIM V3 Full-Wave studios**: `fw_studio2d.html`, `fw_studio2d_outdoor.html`, `fw_studio3d.html`, `fw_studio3d_outdoor.html`. Solver engines — `fw_solve2d.js` (ONNX + FDTD), `fw_solve2d_bs.js`, `fw_solve3d.js`, and the analytic `fw_pathloss2d.js` (FSPL / Motley-Keenan / Eikonal / Ray-trace); the AI/ML gate (`aiml_gate.js`); FDTD workers + progress (`fw_fdtd3d_worker.js`, `sim_worker.js`, `fdtd_progress.js`); browser voxelizer/CAD helpers; `simulator.css`. |
| `Frontend/indoor-outdoor/` | Shared indoor/outdoor UI bits. |

The `Data/*.js` datasets, `Map Coverage/dashboard.js`, etc. are loaded by URL, which resolves
through the root symlinks — so nothing here hard-codes `Frontend/…`.

---

## `Backend/` — server + physics/engine

| Folder | Contents |
|---|---|
| `Backend/server/` | The **Flask project backend**. `app.py` (routes + static serving + `--self-test`/`--stress-test`/`--fake-project`), `validate.py` (per-mode required-info + checks), `pipeline.py` (shells the bake/rasterize scripts to build a project), `workspace_store.py` (versioned SAVE on the `workspace-store` branch), `requirements.txt`, `README.md`. `projects/` (git-ignored) is the runtime store of created projects. |
| `Backend/Physics Engine/` | The **RF engines**. `2D/` — STEP rasterizers (`STEP_1/`, `STEP_2/`) + `SIM`, `SIM V2`, `SIM V2.2`, `SIM V3` (full-wave FDTD + surrogate training). `3D Map Physics/SIM V1 3D/` — the 3D voxelizer, engine, web assets (`web/*.onnx`, `sim_assets_3d.js`), datasets, tests. `Cross_Validation/` — predicted-vs-measured RSRP (base-station catalog, 2D/3D O2I). |
| `Backend/scripts/` | The **bake / rasterize scripts** the pipeline orchestrates: `bake_outdoor_walk.py`, `bake_outdoor_timeseries.py`, `bake_noma_city2d.py`, `prepare_indoor_basemap.py`, `enrich_base_stations.py`, plus the synthetic-project generator `make_fake_walktest.py`. |
| `Backend/early_analysis/` | Early data-analysis notebooks/scripts (legacy). The active data-build script `build_all_bands_data.py` lives under `ARCHIVE/EARLY Analysis/` and is called by `pipeline.py`. |

---

## `Data + Model/` — datasets, models, manifests (reached as `Data/`)

| Item | Contents |
|---|---|
| `records_data.js`, `timeseries_data.js` | the indoor walk — position-averaged records + the time-ordered series. |
| `floorplan_image.js`, `floorplan_meta.js` | the floor-plan raster (base64) + its georeference (affines + GCPs). |
| `base_stations.js` | the surveyed base-station catalog (PCI ↔ operator). |
| `outdoor_walk.js`, `outdoor_timeseries.js`, `noma_city2d.js`, `noma_clutter.js` | the outdoor NoMa walk + the penetrable city grid for the outdoor studio. |
| `basemap_image.js`, `pl2d_crossval.js` | *(git-ignored, generated)* OSM under-layer + 2D O2I overlay. |
| `projects/` | the **project manifests** — `registry.js` + `indoor_2d_default.js` / `outdoor_2d_default.js` / `indoor_3d_default.js` / `outdoor_3d_default.js`. Each names the data files a workspace boots from (the "data" half of the data-vs-features split). See `Data/projects/README.md`. |
| `models/` | the 3D CAD models — the 7th-floor OBJ/GLB and the `NoMa_DC/` city render. |

---

## `docs/`, `ARCHIVE/`, and the rest

- `docs/` — `QUICKSTART.md`, `AGENTS.md`, plans/roadmaps (`3D_RF_Simulator_PLAN_v2.md`,
  `UI_ROADMAP.md`, `BUILD_PROMPTS.md`, `UNITY_MAPCOV_PLAN.md`), and **`docs/ai-ml/`** — how to
  enable the AI/ML surrogate for a new project + the roadmap (ADE-FDTD, Monte-Carlo/VMC, conformal FDTD).
- `ARCHIVE/` — `EARLY Analysis/` (incl. the still-used `build_all_bands_data.py`), `MATLAB/`,
  `One-offs/`, `Scripts + Scripts for Rendering/`, `raw_walk_data/` (the scanner CSVs the pipeline
  reads), `superseded/`.
- `Essentials + HTML/` — the source `7th_Floor…V2.2.png` + `.TAB` (the indoor pipeline's inputs).
- Each `Frontend/*` and `Backend/*` folder also has its own `README.md` with finer detail.

## Running it
```bash
pip install -r Backend/server/requirements.txt
python Backend/server/app.py          # → http://127.0.0.1:8000
```
The built-in **Default** projects also work under a plain static server; only creating/saving
projects needs the Flask backend.
