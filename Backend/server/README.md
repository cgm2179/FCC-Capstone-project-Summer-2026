# Backend/server — project backend

The "functional backend" behind the Import step. It stores **projects** (named datasets), turns
raw uploads (CSV folders, floor plan + TAB, 3D models) into a project by shelling out to the
**existing** repo scripts (no physics is re‑implemented here), and versions SAVEd workspaces on a
dedicated git branch. It serves the static site **and** the `/api` on one origin (same‑origin).

## Run

```bash
pip install -r Backend/server/requirements.txt   # first time (flask + the pipeline's deps)
python Backend/server/app.py                      # -> http://127.0.0.1:8000
```

Open **http://127.0.0.1:8000**. The built‑in **Default** projects also work under plain Live
Preview (static fallback), but **creating / saving** projects needs this server.

```bash
python Backend/server/app.py --self-test     # imports + default data resolve
python Backend/server/app.py --stress-test   # a create must NOT touch the committed default data
```

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | liveness + which projects exist |
| GET | `/api/projects?environment=&dim=` | list manifests (built‑in + created) |
| GET | `/api/projects/<id>` | one manifest (data srcs resolved to URLs) |
| GET | `/api/data/<id>/<role>` | serve a project's data file |
| POST | `/api/validate` | multipart uploads → required‑info + checks report |
| POST | `/api/projects` | multipart uploads → validate → pipeline → store project |
| DELETE | `/api/projects/<id>` | remove a created project (built‑ins protected) |
| POST | `/api/projects/<id>/save` | commit a new **version** to the `workspace-store` branch |
| GET | `/api/projects/<id>/versions` | list a project's saved versions |
| POST | `/api/projects/<id>/restore` | restore an older version (`{sha}`) |
| POST | `/api/workspace-store/publish` | push the `workspace-store` branch to origin |

## Create pipelines (`pipeline.py`) — shells the existing scripts

- **Indoor · 2D** — `CSV folder + PNG + .TAB` → parse GCPs → `ARCHIVE/EARLY Analysis/
  build_all_bands_data.py` (records/timeseries) → base64 PNG (floorplan) → px→lon/lat affine
  (meta) → reuse the base‑station catalog. Reproduces the default dataset byte‑for‑byte.
- **Outdoor · 2D** — CSV folder → `build_all_bands_data.py` → `scripts/bake_outdoor_walk.py` +
  `bake_outdoor_timeseries.py` (PCI→operator via the catalog) → `outdoor_walk.js` +
  `outdoor_timeseries.js`.
- **3D (indoor/outdoor)** — `voxelize.py` on the uploaded model honoring the **elevation** option
  (none | floor + ceiling) → `export_web3.py` → `sim_assets_3d.js`; the model is stored for the
  viewer. (The deep 3D viewer/sim + outdoor‑iframe rendering of project data is the follow‑on
  Simulation‑UI PR; indoor · 2D renders fully today.)

Every created manifest carries **`ai_ml: false`** (surrogates are valid only for the default
data — see `docs/ai-ml`). `validate.py` reports required‑vs‑optional inputs + checks per mode.

## Versioned SAVE (`workspace_store.py`)

Created projects are git‑ignored on the code branches. A dedicated **orphan branch
`workspace-store`** (never main) holds them, checked out via a **git worktree** at
`.workspace-store/` (git‑ignored) so your working tree is never touched. Each SAVE is a commit =
a version (newest on top); `restore` checks out an older one; on startup the server materializes
the branch into the store. Publishing that branch to GitHub is the explicit `.../publish` step.

## Layout & notes

- `app.py` routes + built‑in defaults (mirror of `Data/projects/registry.js`) + on‑disk store.
- `validate.py`, `pipeline.py`, `workspace_store.py`, `.env.example` (optional keys — **no key is
  required**; the OSM/Carto basemap is free).
- `projects/<id>/` — a created project (`manifest.json` + `data/` + `inputs/`). **git‑ignored.**
- On main, `Data` is a **symlink** → `Data + Model`; manifests reference `Data/*.js` and resolve
  on a fresh clone. Uploaded CSVs keep their original basename (band + "Blind Scan" filter are
  decoded from the filename).
