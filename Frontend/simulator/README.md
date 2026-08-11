# Frontend/simulator/

The **Simulation** tab UI — a per-mode tab with a 2D and a 3D variant.

| File | Role | Script type |
|---|---|---|
| `simulation3d.js` | 3D Simulation shell — construct/place Tx & Rx, waveform, static field + animated wavefront | **ES module** |
| `fw_studio2d.html` | **SIM V3 full-wave (2D)** live studio — ONNX surrogate / FDTD field on the floor plan (indoor) | standalone page |
| `fw_studio2d_outdoor.html` | **SIM V3 outdoor full-wave (2D)** — base-station coverage over the real NoMa OSM city | standalone page |
| `fw_solve2d.js` | indoor full-wave solvers: `onnxTiledPredict` (`fw_unet2d.onnx`) + `fdtdSolve` (leapfrog) + time estimates, colormaps | **ES module** |
| `fw_solve2d_bs.js` | outdoor solvers: `buildRegion` + `onnxTiledPredictBs` (10-ch `fw_bs.onnx`, +directivity) + `fdtdSolvePenetrable` (per-class ε/loss) | **ES module** |
| `simulator.css` | Styles for both the 2D simulator and the 3D construction rail / viewport | — |

**2D variant** (`#sim2dPanel`) is the received-power / path-loss tool. Its
**engine** is generated into `SIM/web/` by the backend (`make assets` /
`make model`) and loaded by the page as `SIM/web/sim_assets.js` +
`SIM/web/simulator_tab.js` (+ `pl_unet.onnx`).

**3D variant** (`#sim3dPanel`, `simulation3d.js`) is a light-themed integration
of the standalone antenna sandbox: three.js (**WebGPURenderer**, matching
`viewer3d.js`) + cannon-es physics. It reuses the SIM V1 3D browser assets loaded
from `SIM3D/web/` (`sim_assets_3d.js`, `collision_3d.js`, `antenna_catalog_3d.js`
→ `window.SIM3D_ASSETS / _COLLISION / _ANTENNA_CATALOG`). It lazy-inits when the
Simulation tab is opened in 3D mode and exposes `window.__sim3d` (with debug
handles).

## SIM V3 full-wave (2D) — `fw_studio2d.html`

A standalone 2-D full-wave studio: place a Tx, pick a band, and generate the
complex field `U(x,y)` on the real 7th-floor plan (`SIM/web/sim_assets.js`). Two
solvers behind one switch (`fw_solve2d.js`):

- **ONNX** — `fw_unet2d.onnx` (`SIM V1 3D/web/`) under **onnxruntime-web**, tiled
  + Hann-feathered exactly like the Python `fw_solver.onnx_tiled_predict`.
- **FDTD** — a minimal in-browser scalar leapfrog (rigid class-3 barriers, sponge
  edges, CW source, phasor DFT). Approximate stand-in; the ground-truth FDTD is
  the Python engine (`Physics Engine/2D/SIM V3/run_wave2d.py --solver fdtd`).

Selection mirrors the Python contract: **Auto** = ONNX if available else FDTD;
**ONNX** raises (no silent fallback) if the model can't load; a **"TIME TO
CALCULATE GRID"** estimate + confirm precedes generation. Renders `|U|` dB
(viridis) or the animated wavefront `Re{U e^{jwt}}`, with the manifest material
palette coloured onto the map (legend ↔ pixels) and a real-world scale bar.

It embeds in the dashboard's Simulation tab via the **"Full-Wave (SIM V3)"**
toggle in `Frontend_Data_Display.html` (`#sim2dPanel`), which lazy-loads this page
in an iframe — distinct from the legacy path-loss 2D tool below it.

## SIM V3 outdoor full-wave (2D) — `fw_studio2d_outdoor.html`

The outdoor sibling: pick a real **base station** on the NoMa DC map, choose a
band + antenna, and generate the full-wave **coverage** over the real OSM city
(penetrable **concrete cores + glass façades + foliage**, terrain-following). The
map is **north-up / east-right** with a lon-lat readout, metre scale bar, material
legend, and compass. Same solver switch + **"TIME TO CALCULATE GRID"** confirm,
same colour scales (Viridis / Rainbow / Red-White-Blue) and adjustable field range
(default `0 / −40 dB` relative), same heatmap / wavefront views.

Two solvers behind the switch (`fw_solve2d_bs.js`):

- **ONNX** — `fw_bs.onnx` (the **10-channel** base-station surrogate: the 9
  `dataset_3d` channels + a **directivity** channel synthesized from the Tx
  boresight & antenna family, a JS port of `fw_bs_catalog.featurize_bs` /
  `antenna_patterns.pattern_gain`). A **region** is extracted around the Tx and
  resampled to the surrogate's fine grid (`cell = λ/npw`, mirror of
  `fw_bs_catalog.bs_region`), then tiled + Hann-feathered like the indoor path.
- **penetrable FDTD** — a damped-leapfrog scalar solve whose per-cell **speed
  `c/n`** and **damping `γ = gPerF·f`** come from the baked material palette
  (`window.NOMA_CITY_2D.physics`, computed from `physics_3d.speed_field` /
  `fullwave2d.damping_by_class`), so the fallback matches the *penetrable* physics
  the surrogate trained on — mirror of `fullwave2d.FullWaveScene`.

Data: `Data/noma_city2d.js` (`window.NOMA_CITY_2D`) — the baked penetrable NoMa
grid + stations, produced by `scripts/bake_noma_city2d.py` from the OSM grid
(`SIM V1 3D/city/NoMa_DC_osm`). The model + contract live in `SIM V1 3D/web/`
(`fw_bs.onnx` + `fw_bs.json`); **`fw_bs.onnx` ships with the repo** (~30 MB,
allow-listed like `fw_unet2d.onnx`). Re-export after retraining with
`fw_export.py --model bs` (see `EXPORT_ONNX.md`). Without the weights,
**Auto → FDTD** and **ONNX → a clear error** (no silent fallback).

It embeds in the dashboard via the **"Outdoor Full-Wave (SIM V3)"** toggle in
`Frontend_Data_Display.html` (`#sim2dPanel`), revealed by `Frontend/landing/landing.js`
whenever the workspace is entered in **outdoor + 2D** mode. The legacy path-loss
"Simulate 2D PL" city view (`Frontend/osm3d/outdoor_view.html`) is unchanged.

**Auto = which engine?** In *this* new studio, **Auto = SIM V3** (ONNX → FDTD),
clearly labelled. The dashboard's other outdoor/3D sims (`outdoor_view.html`,
`simulation3d.js`) still use the pre-SIM-V3 **legacy path-loss** engine (`marchPL`).

**WebGPU note:** the page renders on the three.js WebGPU backend, which does *not*
size `THREE.Points` or draw `wireframe`. Volumetric visuals here use
`InstancedMesh` (the static field is instanced cubes; the wavefront is a
translucent sphere) — reuse that pattern for new 3D overlays.

**Deferred (later prompts):** real geometry-aware wave propagation (reflection /
refraction / diffraction / absorption / scattering) from `engine_3d.py` via
`export_web3.py`; the RF-physics visualize modes; GPU offload; ML surrogate
measurement (`pl_unet3d.onnx` + onnxruntime-web). The static field / animated
wavefront are honest FSPL / kinematic stand-ins until then.
