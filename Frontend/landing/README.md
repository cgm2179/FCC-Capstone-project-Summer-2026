# Frontend/landing/

The **app screen router** — the entry flow the app opens on.

| File | Role |
|---|---|
| `landing.js` | Screen router + chooser / import / preprocess logic (**classic** script) |
| `landing.css` | Styles for the step screens (reuses the dashboard design tokens) |

Flow: **mode chooser** (Indoor \| Outdoor · 2D \| 3D) → *(3D only)* **import**
(3D model + `.TAB`) → **preprocess** (voxel-grid summary + measurement CSV) →
**workspace** (the existing sidebar + tabbed viewer in [../2d-3d/](../2d-3d/)).
The 2D path skips import/preprocess and goes straight to the workspace.

`landing.js` publishes the single source of truth the viewers read:

- `window.appMode = { environment: 'indoor'|'outdoor', dim: '2d'|'3d' }`
- `window.appImport = { modelFiles, tabFile, csvFile }`

It drives the 3D CAD viewer through the `window.__viewer3d` hook exposed by
[../2d-3d/viewer3d.js](../2d-3d/viewer3d.js) (load model files, validate/import a
CSV) and matches the workspace to the chosen mode (Map-Coverage 2D/3D, and the
Simulation tab's 2D vs 3D panel). The screens live in
[../../Frontend_Data_Display.html](../../Frontend_Data_Display.html) as
`#screenChooser` / `#screenImport` / `#screenPreprocess` / `#screenWorkspace`.

**Deferred:** Outdoor mode (currently disabled — it lives in the separate
`Outdoor_Walk_Test_7-24` project, see [../indoor-outdoor/](../indoor-outdoor/))
and live in-browser voxelization of a newly imported model (that is `voxelize.py`).
