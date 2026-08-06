# SIM V3 — full-wave (FDTD) 2-D floor-plane engine

A **time-domain, full-wave** complement to the production eikonal engine
(`SIM V1 3D/engine_3d.py`, `SceneV3`). Where `SceneV3` solves the *geometric-optics
limit* (steady-state path loss `PL` + eikonal arrival time `T` — two static grids),
SIM V3 **steps the actual scalar field `u(x, t)`** so reflection, refraction,
diffraction and interference *emerge* from the medium instead of being added as
analytic terms. The output is a real wave animation plus a co-registered
field-strength map.

This engine does **not** reimplement physics. It reuses, verbatim:

| Piece | Source |
|---|---|
| leapfrog stepper, CFL, sources, sponge ABC | `3D Map Physics/Wave Behavior/Wave Generation/Time_Domain_Physics.py` (`WaveSim`) |
| n-D Laplacian, rigid mask | `.../Wave Generation/Spatial_Physics.py` |
| ITU-calibrated, freq-dependent speed field `c(x)` + barrier mask | `SIM V1 3D/physics_3d.py::speed_field` |
| complex permittivity `eps_r(f)` (for the loss term) | `2D/SIM/physics_v2.py::permittivity` (ITU-R P.2040) |
| floor geometry (metre-registered voxels) | `SIM V1 3D/voxelize.py` output (`material_grid.npy`) |

## The one new equation — material loss

A conductive medium makes the wave equation *damped*:

```
d^2u/dt^2 + 2 gamma(x) du/dt = c(x)^2 laplacian(u)
```

with damping rate `gamma = pi f tan(delta)` built from the imaginary part of the
ITU complex permittivity. The leapfrog update becomes

```
u^{n+1} = [ 2u^n - (1 - a) u^{n-1} + (c dt)^2 lap ] / (1 + a),   a = gamma dt
```

which reduces **exactly** to the sandbox update when `a = 0` (air / lossless), so
the lossless path — and the sandbox `--demo` — is unchanged.

## Modules

- `bands_v3.py`      — real 4G / 5G / WLAN bands + the `h = lambda/N` cell budget.
- `plane_extract.py` — slice the horizontal floor plane from the 3-D voxels,
  nearest-resample 0.30 m classes up to the FDTD cell, place the Tx.
- `fullwave2d.py`    — `FullWaveScene(WaveSim)`: real medium + loss term.
- `run_wave2d.py`    — driver: run → GIF + frame stack + RMS envelope + diagnostics.
- `validate_vs_eikonal.py` — Spearman(SceneV3 `PL`, FDTD loss) on the same plane.

> Naming: the class is `FullWaveScene`, **not** `SceneV3` — that name is the
> existing 3-D eikonal engine and reusing it would collide.

## Usage

```bash
python bands_v3.py                                  # cell-size budget per band
python run_wave2d.py --band LTE_B71_617             # cheap whole-floor 4G low-band
python run_wave2d.py --band LTE_B71_617 --quick     # pipeline smoke test (~20 s)
python run_wave2d.py --band NR_n77_3700 --crop 34 47 14 26   # real 3.7 GHz, cropped
python validate_vs_eikonal.py --band LTE_B71_617 --out out/LTE_B71_617
```

**Resolution budget (pick two of three).** Full-wave needs `h <= lambda/10`, so a
band's grid grows as `f^2`. Over the shipped 78.6 x 39.6 m floor: 617 MHz is
~1.3 M cells (whole floor, ~2 min); 3.7 GHz is ~47 M cells (crop required). The
`--max-cells` guard auto-coarsens `N` and warns rather than allocating tens of
millions. Cellular low-bands are cheap; Wi-Fi / C-band need a crop or the GPU.

## Outputs (per run, in `out/<band>/`)

- `wave.gif` / `field_last.png` — the animated field (arcsinh-compressed so
  far-field ripples show, not just the near-source blob).
- `field_rms.npy` / `field_db.png` — steady-state RMS field-strength (envelope),
  co-registered like the eikonal `PL`.
- `wave_frames.npz` — float16 frame stack (for the C# / browser replay).
- `run_meta.json`, `validation_vs_eikonal.{json,png}`.

## Validation

Full-wave and eikonal model different limits, so they are never dB-identical, but
they should **agree in shape** where geometry dominates and **diverge** exactly in
the deep-shadow / diffraction regime the ray engine is weakest at. On the 7th-floor
plane at 617 MHz: **Spearman rho = +0.79** between `SceneV3` path loss and the FDTD
loss (`-strength`) — two independent engines (ray vs full-wave) confirming each
other's coverage pattern.

## Phase 2 — C# GPU 3-D animation (planned)

Port the identical leapfrog stencil to the GPU in `Unity_RF_Simulator/`, rendering
the live plane inside the 3-D scene via the existing slice/volume renderers
(`Viz/CoverageDemo.BuildSlice`, `Viz/VolumetricReplay`), with a "Full-wave (FDTD)"
toggle on `Viz/Bridge.cs` ↔ `Frontend/simulator/unity_bridge.js`.

> **WebGL caveat:** Unity WebGL2 has no compute shaders. Web-safe path is a
> **fragment-shader ping-pong** (`Graphics.Blit` between two `RenderTexture`s —
> proven WebGL2 GPGPU); reserve true compute shaders for desktop / the experimental
> Unity 6 WebGPU backend. The `wave_frames.npz` above also supports a
> pre-baked replay with no in-browser solve.

## Caveats

- **Scalar wave, not full Maxwell** — no polarization; a single field. Fine for
  visualization; a Yee-grid E/H FDTD is the future fidelity upgrade.
- **Envelope is relative dB**, not calibrated absolute path loss (the CW source
  amplitude is arbitrary) — the bridge is shape/rank, not absolute level.
- **Residual radial streaks** in the RMS map are transient CW; lengthen the run
  (`--crossings`, `--warmup-frac`) or extract a proper frequency-domain
  steady-state for a cleaner envelope.

---

# 2D Physics Engine (Part 2)

SIM V3 is also the basis for a general 2-D engine: a progress bar + CPU estimate,
a fast **hybrid** solver (full-wave near the Tx, material-aware analytics far),
and outdoor cellular coverage.

## Runtime estimate + progress (`perf_v3.py`)
`calibrate_throughput()` times the leapfrog stencil on THIS CPU; every driver
prints an up-front `est ~X min` and shows a `tqdm` bar. Each run writes
`out/<run>/progress.json` (`fraction/eta_s/mcells_per_s/stage/done`).
**Frontend:** `Frontend/simulator/fdtd_progress.js` polls that file and drives the
`#simProgress` bar (append `?fdtd_progress=<url>` to the simulator, or open
`fdtd_progress_demo.html?url=<progress.json>`). Static-server friendly — no backend.

## Hybrid near-FDTD / far-analytic (`hybrid_field.py`)
Full-wave FDTD is accurate everywhere but scales with area, so it runs only on a
near-box around the Tx; the rest of the floor is filled by the material-aware
`SceneV3` path loss (attenuation/absorption/refraction/diffraction), level-anchored
at the handoff radius and crossfaded (seamless). `nearfield.py` supplies the field
zones (`lambda/2pi`, `2D^2/lambda`) and the −6 dB close-in FSPL fix (Schantz).

```bash
python hybrid_field.py --band WiFi_2G4                 # indoor WLAN, whole floor fast
python hybrid_field.py --band WiFi_2G4 --near-radius-m 8 --quick
```
Output: `hybrid_coverage.png` (far | near | hybrid), `hybrid_pl.npy`, near-zone GIF.

## Outdoor 2-D cellular (`run_outdoor2d.py`)
Slices the NoMa city plane (buildings = barrier class 3) and runs the 2-D
Motley-Keenan ray-march (`STEP_2/motley_keenan.compute_pathloss`) for citywide
coverage — the "old physics" (per-crossing building loss + saturating obstruction).
Citywide full-wave is infeasible, so FDTD is an optional local patch (`--fdtd-patch`).

```bash
python run_outdoor2d.py                                 # 800 m coverage @ 2506 MHz
python run_outdoor2d.py --tx 1400 900 --crop-m 1000 --eirp-dbm 46
```
Output: `outdoor_coverage.png` (buildings grey), `outdoor_prx.npy`.

## Part 2 caveats
- Hybrid is a **cost optimization** (FDTD is accurate everywhere); the anchor is a
  relative-dB calibration, not an absolute level.
- Outdoor is **analytic-dominant**; straight-ray Motley-Keenan gives pessimistic
  radial shadows (no diffraction reroute) — documented in that module.
- Frontend shows Python-CPU progress via **file-polling**; no in-browser FDTD.

---

# ML full-wave surrogate (Part 3)

Real-frequency full-wave over a whole scene is a compute wall (indoor floor: min→21 h
per band; whole city: infeasible). The escape is a **learned surrogate**: run FDTD on
finite **boxes** for training data, train a **U-Net** to map (geometry+source+band) →
the full-wave field, then infer any box in **ms** and **tile** the scene. Reuses the
repo's surrogate scaffold (built for eikonal, untrained) — the change is the target.

Predicts the **complex field** (re/im). Phase is made learnable/tileable by removing
the free-space ramp: the net learns `Ũ = U·e^{+jkd}`; inference rebuilds `U = Ũ·e^{-jkd}`.

| Stage | File | Reuses |
|---|---|---|
| Complex phasor from FDTD (on-the-fly DFT) | `fullwave2d.simulate(extract_phasor=True)` | — |
| Data-gen: big FDTD per Tx → crop boxes → (9-ch input, re/im target) | `fw_dataset.py` | `dataset_3d` featurization, `plane_extract`, `FullWaveScene` |
| 2-D U-Net (cin=9, cout=2) + train (MPS/CUDA) | `fw_unet2d.py` | `phase_c` UNet arch |
| ONNX export + browser contract | `fw_export.py` → `web/fw_unet2d.{onnx,json}` | `export_surrogate_smoke` pattern |
| Tiled inference + **g2 gate** (tiled floor vs fresh FDTD) | `fw_infer.py` | `plane_extract`, `FullWaveScene`, `scipy` |

```bash
python fw_dataset.py --bands LTE_B71_617 --n-tx 6 --boxes-per-field 80 --box 128 --out fw_data_617
python fw_unet2d.py  --data fw_data_617 --epochs 80 --base 32     # MPS/CUDA
python fw_export.py  --ckpt fw_unet2d.pt --data-meta fw_data_617/dataset_meta.json
python fw_infer.py   --ckpt fw_unet2d.pt --band LTE_B71_617 --validate   # g2
```
Full scale (all 8 bands, thousands of boxes, GPU): run `fw_dataset.py` per band, train
with `--base 64` on Colab (`fw_unet2d.py` picks CUDA), then export + validate.

**Colab notebooks** (thin wrappers around the modules above):
- `Indoor V3 Sim/` — `Phase_B_Dataset`, `Phase_C_Train_UNet`, `Phase_D_Validate` (.ipynb)
- `Outdoor V3 Sim/` — same three, on NoMa city region-crops (buildings = barrier)

Speed: once trained, a whole-floor map is **~0.5 s (617 MHz) → ~1 min (6 GHz)** vs FDTD's
**2 min → ~21 h** — ~240× to ~1300× per map, and interactive instead of batch.

## Part 3 caveats
- **Complex/phase is research-grade**; the ramp-factoring is load-bearing.
- A surrogate **approximates** full-wave, bounded by the training distribution — **g2**
  (tiled prediction vs a fresh FDTD solve on a held-out Tx) is the honesty gate.
- **Data-gen is the real cost** (many FDTD boxes, offline); training needs a GPU at scale.
- Cross-box multipath phase can leave residual seams; feather-blend on `Ũ` mitigates.
