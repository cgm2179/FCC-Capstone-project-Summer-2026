# Data/projects — the project/dataset layer

This is the **data** half of a data-vs-features split. A **project** is a named dataset the
workspace can boot from; the dashboard code (`Map Coverage/* + Statistics/*`) is the **features** half and
reads whatever project loaded. Swap the project, keep the dashboard — so new datasets (any
2D/3D · indoor/outdoor) and new features can be added independently.

## Files here

- `registry.js` — `window.PROJECT_REGISTRY` (index of the built-in defaults) + the shared
  `window.PROJECT_DATA_INDOOR_BASE` list (the indoor walk dataset the dashboard has always
  booted from). **Keep in sync with `Backend/server/app.py` `BUILTINS`.**
- `indoor_2d_default.js`, `outdoor_2d_default.js`, `indoor_3d_default.js`,
  `outdoor_3d_default.js` — the four built-in **"Default"** manifests, each
  `window.PROJECTS[id] = {...}`. These are also the **static fallback** when the backend isn't
  running (plain Live Preview still shows the Default workspace, read-only).

## Manifest schema

```js
window.PROJECTS['my-id'] = {
  id: 'my-id',
  label: 'My project',
  environment: 'indoor' | 'outdoor',
  dim: '2d' | '3d',
  builtin: true,                 // false for backend-created projects
  data: [                        // injected in order, BEFORE the dashboard boots
    { role: 'records',  src: 'Data/records_data.js' },          // required, bare const
    { role: 'basemap',  src: 'Data/basemap_image.js', optional: true },  // 404 → warn, continue
    // ...
  ],
  // optional metadata (documented / used by the 3D + outdoor paths):
  iframes: { map: '…', time: '…', sim2d: '…', sim3d: '…' },
  model: { src: 'Data/models/…', label: '…' },
  elevation: { mode: 'none' | 'floor', floor_z_m: 0, ceiling_height_m: 4.58 },
};
```

Roles the indoor dashboard reads (see `registry.js` for the exact files):
`records` (`const records`), `timeseries` (`const timeseriesRecords`/`timeseriesStartTime`),
`floorplan` (`const floorPlanImage`), `meta` (`window.FLOORPLAN_META`), `baseStations`
(`window.BASE_STATIONS`), and the optional `basemap`/`crossval`.

## How loading works

`Frontend/landing/project_loader.js` injects each `data[].src` in order (awaiting each — the
dashboard reads bare `const`s at init), then injects `dashboard.js` + `spectrum.js` once.
`getProjects(env,dim)` / `resolveDefaultProject(env,dim)` feed the Import dropdown;
`refreshProjects()` merges backend + built-in projects.

## Add a dataset / project

- **From raw inputs (recommended):** run the backend and use Import → **New project…** (upload a
  CSV folder + floor plan + `.TAB`, or a 3D model + elevation). See `Backend/server/README.md`.
- **By hand (a committed built-in):** add a `window.PROJECTS[id] = {...}` file here, list it in
  `registry.js` + `Backend/server/app.py` `BUILTINS`, and add its `<script>` tag in
  `Frontend_Data_Display.html`.

Because features read the **active** project's globals, a new workspace feature is dataset-
agnostic by construction — write it against `records` / `window.FLOORPLAN_META` /
`window.BASE_STATIONS`, and it works for every project.
