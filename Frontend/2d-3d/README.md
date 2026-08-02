# Frontend/2d-3d/

The **Map-Coverage viewer** — 2D and 3D rendering of the walk test.

| File | Role | Script type |
|---|---|---|
| `dashboard.js` | 2D Plotly map + Histogram + Time-Elapse tabs + tab controller | **classic** |
| `viewer3d.js` | 3D CAD viewer (three.js) + RF coverage overlay | **ES module** |
| `spectrum.js` | Local Spectrum & ARFCN sidebar | **classic** |
| `dashboard.css`, `spectrum.css` | styles | — |

**Contract — don't break:** `dashboard.js` / `spectrum.js` must stay classic
scripts so their top-level functions land on `window`; `viewer3d.js` is a module
that reads them via `window.filtered` / `window.idwGrid` /
`window.filteredTimeseries`. Script types and load order are set in
[../../Frontend_Data_Display.html](../../Frontend_Data_Display.html).

`dashboard.js` currently also owns the Histogram and Time-Elapse tabs; as those
grow they can split into their own feature folder later. The 3D models it loads
live in [../../Data/models/](../../Data/models/).
