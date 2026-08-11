# ARCHIVE/

Provenance only — not on the active iteration path. If it's under here, it's
a one-off, superseded, or historical artifact. Active app code stays at the
repo root (`Frontend/`, `Map Coverage/`, `Simulation/`, `scripts/`, …).

| Path | What it is |
|---|---|
| **`EARLY Analysis/`** | First-generation scripts/notebooks: CSV concatenation, histogram R, methods notebook, dashboard builder. |
| **`Scripts + Scripts for Rendering/`** | Notes for archived / one-off render & model-export tooling. Active bake scripts stay in repo-root `scripts/`. |
| **`raw_walk_data/`** | Original scanner exports: `CSV/`, `DTR/`, concatenated CSVs, misc. |
| **`MATLAB/`** | `.mat` data bundle + MATLAB ports. |
| **`superseded/`** | Replaced pipelines: `WEB/`, `STEP_3/`, `STEP_4/`. |
| **`One-offs/`** | Everything else that used to clutter the repo root — see [One-offs/README.md](One-offs/README.md). |

Active consumers that still read ARCHIVE:

- `ARCHIVE/MATLAB/export_to_matlab.py` → `raw_walk_data/Concat_Indoor_Walk_Test_from_csv.csv`
- `Physics Engine/Cross_Validation/base_station_catalog/build_catalog.py` → `EARLY Analysis/build_all_bands_data.py`
