# AGENTS.md

## Cursor Cloud specific instructions

This repo is an **RF (radio frequency) propagation simulator**: a static browser app backed
by an offline Python physics engine. There is **no running backend service** — the app is
static HTML/JS/CSS, and the precomputed physics volumes are committed to the repo, so the
simulator works fully offline. Python is only needed for running tests or re-solving new
transmitter positions.

See `QUICKSTART.md` for the canonical run/serve/test commands and the full simulator flow;
this section only captures the non-obvious, durable gotchas for future cloud agents.

### Layout
- `Frontend_Data_Display.html` — the app shell / router (open this in a browser).
- `Frontend/landing/` — Indoor/Outdoor · 2D/3D workspace router.
- `Map Coverage/`, `Statistics/`, `Time Elapse/`, `Simulation/` — dashboard tab modules.
- `Data + Model/` — walk payloads + CAD (compat symlink `Data/` → this folder).
- `SIM3D/` → `Physics Engine/3D Map Physics/SIM V1 3D/` (symlink) — the Python engine.
- `TESTS3D/` → the engine's `tests3d/` (symlink) — pytest harness.
- `docs/` — plans and roadmaps (`AGENTS.md` / `QUICKSTART.md` also symlinked at repo root).
- `Backend/` is a staging area only; it has no runtime service.

### Python environment
- A prebuilt virtualenv lives at `/workspace/.venv` (created/refreshed by the startup update
  script from the two committed requirements files). Activate it before running Python:
  `source .venv/bin/activate`.
- `scikit-fmm` (the 3D eikonal solver) has **no Linux wheel** and is compiled from source on
  install. The VM image already carries the needed system packages (a C++ compiler and the
  Python dev headers). NOTE: the system default `c++` is clang and fails to link `-lstdc++`;
  the environment installs deps with `CXX=g++` set so the build uses g++ instead. If you ever
  reinstall Python deps by hand and hit `Compiler c++ cannot compile programs`, prefix the
  pip command with `CC=gcc CXX=g++`.

### Running the app (frontend)
- Serve the **repo root** over HTTP and open the shell — never `file://`, or `fetch()` of the
  precomputed volumes is blocked and the sim silently drops to a wrong-looking analytic
  fallback. From `/workspace`: `python3 -m http.server 8777`, then open
  `http://localhost:8777/Frontend_Data_Display.html`.
- First load fetches three.js, cannon-es, and Plotly from CDNs, so it needs internet once.

### Known-good end-to-end flow (and its gotchas)
- Flow: **Indoor → 3D mode → Continue → Continue (Import) → Enter workspace (Preprocess) →
  Simulation tab**. In the right rail set **Mode = Vacuum / free space**, set **Visualize →
  Property = Propagation & coverage**, then click **Static 3D Simulation**. A coverage heatmap
  renders over the floor and the status line reads `Static field · cached volume —
  full-physics solve (SceneV3 eikonal/Fresnel) @ 2412 MHz …`. That "cached volume /
  full-physics solve" text is the signal the real engine ran (not the analytic fallback).
- Gotcha: in this headless VM, WebGPU has no adapter and WebGL falls back to software
  rendering. The devtools console shows benign warnings (`WebGL is not available`,
  `DMNX init failed on webgpu — no available adapters`) plus a 404 for `pl_unet3d.onnx`
  (no ML surrogate is trained yet — the cached/analytic tiers are the intended floor). The
  model still renders and the simulation still completes.
- Gotcha: clicking directly on the 3D model surface to *place* a transmitter can crash the
  tab (raycasting). It is not needed for the demo above — `Static 3D Simulation` uses cached
  transmitter positions automatically. Cached Tx positions are listed in `QUICKSTART.md`.
- `Outdoor` mode reports itself unavailable until the city voxel grid is built
  (`voxelize_city.py`); the `cache/` and `city/` dirs are gitignored, regenerable artifacts.

### Tests
- Activate the venv, then `python3 TESTS3D/run_all.py --full` (currently 112 passed, 1
  skipped). The 3D engine has no lint config; correctness is enforced by this suite.
