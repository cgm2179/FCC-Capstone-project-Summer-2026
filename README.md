# Indoor / Outdoor Walk-Test Studio

A browser dashboard for exploring measured **4G / 5G / Wi-Fi** walk-test data — a signal-coverage
map, statistics, and a time-elapse replay — paired with an **RF-propagation simulation** sandbox
(full-wave FDTD + physics-based path loss, indoor and outdoor, 2D and 3D). A small Flask backend
turns raw walk-test uploads into reusable "projects."

The built-in **Default** datasets are committed, so a fresh clone runs the whole dashboard with no
build step; Python is only needed to create/save new projects, re-solve fields, or run the tests.

---

## Quick start

### Just browse the dashboard (no Python)
```bash
python3 -m http.server 8777
```
Then open **http://localhost:8777/Frontend_Data_Display.html**.

> Serve it over HTTP as above — opening the `.html` directly as a `file://` page makes the browser
> block `fetch()`, so the datasets silently fail to load.

### Full backend (create + save projects)
```bash
pip install -r Backend/server/requirements.txt
python Backend/server/app.py          # → http://127.0.0.1:8000
```

For a deeper run/serve/solve walkthrough of the 3D engine, see **[docs/QUICKSTART.md](docs/QUICKSTART.md)**.

---

## Repository layout

The tree is split into **`Frontend/`** (everything the browser runs) and **`Backend/`** (the Flask
server + the physics/engine code). **Load-bearing root symlinks** (`Data/`, `Physics Engine/`, `SIM3D/`,
`TESTS3D/`, the tab folders, …) keep every hard-coded URL and script path resolving, so browse
`Frontend/`/`Backend/` for the real files while the old names still work.

| Path | What it is |
|---|---|
| `Frontend_Data_Display.html` | The app entry page (served at `/`). |
| `Frontend/` | All browser code — the dashboard tabs and the SIM studios. |
| `Backend/` | The Flask project backend + the 2D/3D RF engines. |
| `Data + Model/` | Datasets, CAD models, and project manifests (reached as `Data/`). |
| `docs/` | Quickstart, plans/roadmaps, and the AI/ML guide. |
| `misc/` | Ancillary bits kept out of the root: the vendored Unity WebGL build + git hooks. |

- **[DIRECTORY.md](DIRECTORY.md)** — a folder-by-folder guide to *what's in each folder*.
- **[REPO_STRUCTURE.md](REPO_STRUCTURE.md)** — *why* it's split `Frontend/` + `Backend/` and how the symlinks work.

---

## Running the tests

The 3D engine ships an acceptance suite (its helpers are also imported at runtime by the
Wave-Behavior effect modules):
```bash
python3 TESTS3D/run_all.py --full
```

---

## Cloning on Windows (important)

This repo relies on **symlinks** (`Data/`, `Physics Engine/`, `TESTS3D/`, and the tab folders). Git
and macOS/Linux handle these automatically. On **Windows** they need symlink support turned on, or the
app breaks in confusing ways (links become plain text files):

```bash
git config --global core.symlinks true      # then clone; also enable Developer Mode
```

Cloning under **WSL**, macOS, or Linux avoids the issue entirely. If you receive this project as a
**zip** rather than a clone, prefer one exported with symlinks dereferenced into real file copies —
a naïvely-zipped copy will carry broken links.

---

## What ships vs. what needs regenerating

A fresh clone includes everything needed to run the browser dashboard and simulator offline (scene
grids, cached volumes, the committed Default datasets, and the browser ONNX surrogates). Large,
regenerable artifacts (multi-GB dataset shards, the full outdoor city grid, trained checkpoints) and
the raw walk-test source folders are **omitted on purpose** — rebuilding the baked data is a
maintainer task, not required to use the app. See the detailed list in
[docs/QUICKSTART.md](docs/QUICKSTART.md).

---

*Sole author/contributor: **cgm2179** (see `.github/CODEOWNERS`).*
