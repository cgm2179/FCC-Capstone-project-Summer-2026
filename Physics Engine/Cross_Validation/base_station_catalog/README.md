# Base-station catalog (PR #24)

Ground-truth control points for the outdoor donor cell sites observed from the FCC HQ
7th-floor indoor walk (7/7). Each **antenna sector (= one PCI)** becomes a row with its
3D position, identification, and the measured RF stats attributed to it. This is the
transmitter definition for the path-loss prediction (PR #26/#27) and the join key for
cross-validation.

> Naming: this repo already uses "GCP" for the floor-plan *ground control points*
> (`Data/floorplan_meta.js`). These antenna control points live here as
> `base_station_catalog` to avoid the collision.

## Run

```bash
cd "Physics Engine/Cross_Validation/base_station_catalog"
python3 build_catalog.py            # full: queries OSM building heights once, then caches
python3 build_catalog.py --no-network   # offline: manual/surveyed/cached heights only
python3 build_catalog.py --test         # geometry + registry self-test
```

`--no-network` needs no internet and reproduces the measured stats exactly. The default
run fetches building heights from OSM/Overpass (the automated elevation source) and
writes `osm_heights_cache.json` so subsequent runs are offline. (If your Python can't
verify TLS to Overpass — "CERTIFICATE_VERIFY_FAILED" — run
`/Applications/Python*/Install\ Certificates.command` once, or use `--no-network`.)

## Files

| File | What |
|---|---|
| `sites_registry.json` | Hand-authored site/antenna facts (lat/lon, PCIs, bands, panel azimuth hypotheses). **Edit this** to add/fix sites. |
| `build_catalog.py` | Joins the registry to the measured scan; emits the CSVs below. |
| `base_stations.csv` | Combined catalog — one row per (site, carrier, band, PCI). |
| `bs_<slug>.csv` | The same rows split per site. |
| `unregistered_measured_cells.csv` | Measured (network,band,PCI) cells **not** in the registry (transparency). |
| `osm_heights_cache.json` | Cached Overpass results (generated). |

Reuses `Backend/early_analysis/build_all_bands_data.py` (scan parsing + georeference)
and `Physics Engine/3D Map Physics/SIM V1 3D/osm_building_height.py` (height resolver,
extended here with caching + a manual-override / placeholder guard).

## Columns

Position/height: `lat, lon, building_height_m, antenna_agl_m, elev_source,
elev_confidence`. Identity: `carrier, network, band, channel, freq_mhz, pci, antenna_id,
antenna_type, azimuth_deg, azimuth_source, azimuth_confidence, cellidentity, enb_gnb_id,
sector_id`. Measured stats: `n_points, rsrp_{min,max,mean,std}, rsrq_mean, sinr_mean,
rssi_mean`. Geometry: `distance_m, bearing_deg` (site → floor-plan centre, same local-ENU
convention as `modes_3d.forte_hall_geometry`). Bookkeeping: `status`
(`live`/`na_tv`/`na_bankrupt`), `carrier_match` (`exact` = MCC/MNC decoded to this
carrier; `pci_only` = PCI matched but carrier undecoded; `conflict` = PCI reused by a
different decoded carrier; `none` = declared but unobserved), `notes`.

## Anchor — Forte Hall (Verizon)

Highest certainty. The catalog reproduces the dashboard readouts exactly and
independently recovers the site identity:

- **LTE Band 13 (751 MHz):** PCI 396 → 72 pts, mean −106.7 dBm; PCI 397 → 50 pts, mean
  −112.0; PCI 398 → 2 pts. Decoded **eNB 107258**, sectors 1/2 (matches CellMapper).
- Same PCIs carry Verizon **5G NR** (n26/n5/n77/n78/…) with much larger point counts —
  available for later multi-band validation.
- Geometry 415.4 m / arrival 237° to the floor plan; antenna height 20 m (manual — OSM at
  this coordinate returns the adjacent glass "Forte Hall" building, not the brick tower
  the panels sit on).
- 3 **panel** sectors; azimuths are hypotheses (396 strongest indoors → hypothesized to
  face the FCC HQ, ~57°). Resolved from the 7/24 outdoor walk in the PR #26/#27
  calibration.

Donors (BS2/4/5/6) have approximate locations → shape-only validation. BS1 (TV) and BS3
(bankrupt Starry) are flagged and excluded.
