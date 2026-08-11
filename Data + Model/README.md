# Data + Model

Browser payloads and CAD models for the dashboard (was `Data/`).

| Area | Contents |
|------|----------|
| Walk / RF JS | `records_data.js`, `timeseries_data.js`, `outdoor_walk.js`, `outdoor_timeseries.js`, … |
| Georef / basemap | `floorplan_*.js`, `basemap_image.js`, `base_stations.js` |
| City rasters | `noma_city2d.js`, `noma_clutter.js` |
| CAD | `models/` (indoor GLB/OBJ, NoMa outdoor OBJ) |

Raw scanner exports live in `ARCHIVE/raw_walk_data/` (not here).
Source floor-plan ground truth: `Essentials + HTML/floorplan/`.

A root symlink `Data → Data + Model` keeps older scripts and docs working.
Regenerate baked JS with `scripts/bake_*.py` / `prepare_indoor_basemap.py`.
