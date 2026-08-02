# Quickstart — run the 3D RF simulator

Clone, serve, open. **No Python needed to browse the simulator** — it is static files plus
a web server. Python is only for re-solving volumes or running the test suite.

## 1. Clone

```bash
git clone https://github.com/cgm2179/indoor-walk-test.git
cd indoor-walk-test
```

~254 MB, which includes the precomputed volume cache so the physics works offline.

Active development happens on `sim3d-m0-physics-harness`; `main` is a merged checkpoint.
For the newest work:

```bash
git clone -b sim3d-m0-physics-harness https://github.com/cgm2179/indoor-walk-test.git
```

## 2. Serve over HTTP — not `file://`

This is the step that bites. Opening `Frontend_Data_Display.html` by double-clicking gives
you a `file://` page, and browsers block `fetch()` there. Every precomputed volume fails to
load and the simulator silently drops to its analytic fallback — the physics looks wrong
but nothing errors. The status line will tell you if this happens.

```bash
python3 -m http.server 8777
```

Then open **http://localhost:8777/Frontend_Data_Display.html**

First load needs internet: three.js and cannon-es come from jsDelivr/unpkg, Plotly from
its own CDN.

## 3. Get to the simulator

The app is a router, so there are a few clicks before the 3D view:

**Indoor** → **3D mode** → *Continue* → *Continue* (Import) → *Continue* (Preprocess) →
**Simulation** tab.

Then in the right rail:

1. **Propagation mode** — vacuum / indoor / O2I / outdoor
2. **Transmitter** → *Place on model*, then click a surface in the viewport
3. **Visualize** → *Propagation & coverage*

## 4. What you should see

| Where | Expected |
|---|---|
| Status line | `Static field · cached volume — full-physics solve (SceneV3 eikonal/Fresnel)` |
| Mode = **Vacuum** + coverage | `INVARIANT GATE PASSED · max │Δ│ ≈ 0.04 dB · causality violations: 0` — the engine reproducing closed-form free space, checked in the browser |
| Visualize → **Mechanism time-lapse** | Four fronts on one clock: Direct 81 ns → Reflected 81 → Diffuse 81 → Diffracted 94 |
| Visualize → **Specular reflection / multipath** | Contribution share in dB against the total |

If the status says `analytic (in-browser)` instead of `cached volume`, you are either on
`file://` (see step 2) or the transmitter is more than 20 voxels from a cached solve — the
status line says which, and names the nearest cached transmitter.

Cached transmitters, if you want to place near one: indoor `66,5,54` (all four mechanisms),
`174,5,42`, `60,5,63`, `217,5,110`; vacuum `131,6,65`; O2I `2,3,115`.

## 5. Optional — Python, tests, re-solving

```bash
pip install -r "Physics Engine/3D Map Physics/SIM V1 3D/requirements_3d.txt"

python3 TESTS3D/run_all.py --full                      # 113 tests
python3 "SIM V1 3D/cache_index.py" --stats             # what is cached
python3 "SIM V1 3D/cache_index.py" --verify            # are the .bin files intact
python3 "SIM V1 3D/modes_3d.py" --list                 # the four modes + availability
```

Solve a new transmitter position (~27 s for all four mechanisms):

```bash
python3 "SIM V1 3D/export_pl_volume.py" --tx 100 5 60 \
  --mechanisms path_loss,reflection,diffraction,scattering --mech-bands 2442,3500,5500
```

The first diffraction run builds a relay cache (~74 s, once); every later solve reuses it.

## What a fresh clone does *not* include

None of these are breakage — they are regenerable artifacts kept out of git deliberately.

- **`SIM V1 3D/cache/`** — the diffraction relay cache (~376 MB) and the O2I facade field.
  Rebuilt automatically on the first solve that needs them (~74 s and ~2 min respectively).
  Not needed to browse the cached volumes.
- **`SIM V1 3D/city/`** — the outdoor city grid (~350 MB). **Outdoor mode reports itself
  unavailable** until you rebuild it. The source mesh *is* in the repo:
  ```bash
  python3 "SIM V1 3D/voxelize_city.py" --mode 2.5d --cell 1
  ```
- **`pl_unet3d.onnx`** — no 3D surrogate is trained yet (that is M4). The browser's
  resolution order is *cached volume → DL surrogate → analytic*; with no model the
  surrogate tier reports its own absence and falls through. The analytic mirror is the
  guaranteed-correct floor, so the simulator works without a surrogate by design.
- **`FCC_Walk_Outdoor_Indoor_Full/`, `Sandbox_Version_3D_Simulation_v1.obj/`** — local
  working data, far past GitHub's 100 MB per-file limit.

## Where things live

| | |
|---|---|
| `Frontend_Data_Display.html` | the app shell |
| `Frontend/simulator/simulation3d.js` | 3D simulator UI, volume cache, mechanism views |
| `SIM3D/` → `Physics Engine/3D Map Physics/SIM V1 3D/` | the engine (symlink) |
| `SIM3D/modes_3d.py` | the four propagation modes |
| `SIM3D/cache_index.py` | content-addressed volume cache |
| `SIM3D/PLAN_3D_SIM.md` | milestone status |
| `SIM3D/MODEL_CARD_3D.md` | physics scope, and what is deliberately not modelled |
| `TESTS3D/run_all.py` | test harness (symlink) |
