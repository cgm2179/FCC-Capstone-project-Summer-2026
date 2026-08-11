# ARCHIVE/

Provenance only — not on the active iteration path. Layout:

| Path | What it is |
|---|---|
| **`EARLY Analysis/`** | First-generation scripts/notebooks: CSV concatenation, histogram R script, methods notebook, dashboard builder. (Was `Backend/early_analysis/`.) |
| **`Scripts + Scripts for Rendering/`** | Home for archived / one-off render & model-export tooling notes. Active bake scripts stay in repo-root `scripts/`. |
| **`raw_walk_data/`** | Original scanner exports: `CSV/`, `DTR/`, `Device032409007/`, concatenated CSVs, misc. |
| **`MATLAB/`** | `.mat` data bundle + MATLAB ports. Re-run `export_to_matlab.py` if STEP_1/2 outputs change. |
| **`superseded/WEB/`** | Standalone coverage app — replaced by the Simulator tab. |
| **`superseded/STEP_4/`** | First surrogate (8-cell notebook) — replaced by `SIM/` Phase C. |
| **`superseded/STEP_3/`** | Ray-tracing/FDTD upgrade notes — folded into the SIM physics ladder. |

Active consumers that still read ARCHIVE:

- `ARCHIVE/MATLAB/export_to_matlab.py` → `raw_walk_data/Concat_Indoor_Walk_Test_from_csv.csv`
- `Physics Engine/Cross_Validation/base_station_catalog/build_catalog.py` → `EARLY Analysis/build_all_bands_data.py`
