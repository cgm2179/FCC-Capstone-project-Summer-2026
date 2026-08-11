# Data/

Frontend data + new / derived datasets. **Never edit the contents of a data file
that something references by path** — move only, and update the reference.

Lives here:

| Data | File | Read by |
|---|---|---|
| Measurement points | `records_data.js` (`const records`) | `Frontend/2d-3d/dashboard.js` |
| Floor-plan raster (base64) | `floorplan_image.js` (`const floorPlanImage`) | `Frontend/2d-3d/dashboard.js` |
| Time series | `timeseries_data.js` (`timeseriesRecords`) | dashboard.js / spectrum.js |
| 3D CAD models | `models/Indoor 7th floor v2*/…obj` | `Frontend/2d-3d/viewer3d.js` |

Stays elsewhere (referenced by the backend at those paths — don't move):

| Data | Location | Read by |
|---|---|---|
| Source ground truth | `../Essentials + HTML/floorplan/7th_Floor_2nd_Indoor_Walk_Test_V2.2.{png,TAB,aux.xml,csv}` | SIM / STEP scripts |
| Raw scanner exports | `../ARCHIVE/raw_walk_data/` | MATLAB exporter / early analysis |
| Material grid + meta | `../STEP_1/*.npy`, `../STEP_1/*.json` | SIM physics |
| Training shards | `../SIM/dataset/` (fetch via `make dataset-fetch`) | Phase C training |
| Physics constants (source of truth) | `../SIM/manifest.json` | Python + JS (Rule R6) |

**Convention:** new exports / derived tables go here as `Data/<name>/`.
