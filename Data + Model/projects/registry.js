/* registry.js — the built-in project registry + the shared "default" data list.
 *
 * A PROJECT is a named dataset the workspace can boot from. Each project's manifest
 * (Data/projects/<id>.js, or later the Flask server) NAMES its data files; the loader
 * (Frontend/landing/project_loader.js) injects the active project's data, then the
 * dashboard. This is the "data" half of the data-vs-features split — the dashboard
 * code (Frontend/2d-3d/*) is the "features" half and reads whatever project loaded.
 *
 * See Data/projects/README.md for the manifest schema + how to add a dataset/project.
 */
window.PROJECTS = window.PROJECTS || {};

/* The indoor walk dataset the dashboard has always booted from — previously seven
 * hardcoded <script> tags in Frontend_Data_Display.html <head>. dashboard.js reads
 * `records`, `timeseriesRecords`, `floorPlanImage` at init, so these load before it,
 * in EVERY mode: outdoor and 3D layer their extra data on top (iframes, SIM3D assets,
 * the lazy OUTDOOR_WALK), exactly as before — they don't replace this base. The two
 * `optional` files are git-ignored (generated locally); the dashboard already guards
 * window.INDOOR_BASEMAP / window.PL2D_CROSSVAL, so a 404 there is non-fatal. */
window.PROJECT_DATA_INDOOR_BASE = [
  { role: 'records',      src: 'Data/records_data.js' },     // const records
  { role: 'timeseries',   src: 'Data/timeseries_data.js' },  // const timeseriesRecords / timeseriesStartTime
  { role: 'floorplan',    src: 'Data/floorplan_image.js' },  // const floorPlanImage
  { role: 'meta',         src: 'Data/floorplan_meta.js' },   // window.FLOORPLAN_META
  { role: 'baseStations', src: 'Data/base_stations.js' },    // window.BASE_STATIONS
  { role: 'outdoorWalk',  src: 'Data/outdoor_walk.js', optional: true },   // window.OUTDOOR_WALK (Indoor 3D OSM ground; main eager-loads it)
  { role: 'basemap',      src: 'Data/basemap_image.js', optional: true },  // window.INDOOR_BASEMAP (git-ignored)
  { role: 'crossval',     src: 'Data/pl2d_crossval.js', optional: true },  // window.PL2D_CROSSVAL (git-ignored)
];

/* Lightweight index of the built-in defaults, one per mode. The full manifests live in
 * the sibling *_default.js files (loaded alongside this one) as window.PROJECTS[id]. */
window.PROJECT_REGISTRY = [
  { id: 'indoor-2d-default',  label: 'Default (Indoor · 2D)',  environment: 'indoor',  dim: '2d' },
  { id: 'outdoor-2d-default', label: 'Default (Outdoor · 2D)', environment: 'outdoor', dim: '2d' },
  { id: 'indoor-3d-default',  label: 'Default (Indoor · 3D)',  environment: 'indoor',  dim: '3d' },
  { id: 'outdoor-3d-default', label: 'Default (Outdoor · 3D)', environment: 'outdoor', dim: '3d' },
];
