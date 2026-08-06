# Frontend/simulator/

The **Simulation** tab UI — a per-mode tab with a 2D and a 3D variant.

| File | Role | Script type |
|---|---|---|
| `simulation3d.js` | 3D Simulation shell — construct/place Tx & Rx, waveform, static field + animated wavefront | **ES module** |
| `fw_studio2d.html` | **SIM V3 full-wave (2D)** live studio — ONNX surrogate / FDTD field on the floor plan | standalone page |
| `fw_solve2d.js` | full-wave solvers: `onnxTiledPredict` (`fw_unet2d.onnx` via onnxruntime-web) + `fdtdSolve` (in-browser leapfrog) + time estimates, colormaps | **ES module** |
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

**WebGPU note:** the page renders on the three.js WebGPU backend, which does *not*
size `THREE.Points` or draw `wireframe`. Volumetric visuals here use
`InstancedMesh` (the static field is instanced cubes; the wavefront is a
translucent sphere) — reuse that pattern for new 3D overlays.

**Deferred (later prompts):** real geometry-aware wave propagation (reflection /
refraction / diffraction / absorption / scattering) from `engine_3d.py` via
`export_web3.py`; the RF-physics visualize modes; GPU offload; ML surrogate
measurement (`pl_unet3d.onnx` + onnxruntime-web). The static field / animated
wavefront are honest FSPL / kinematic stand-ins until then.
