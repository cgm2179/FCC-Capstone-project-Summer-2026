# Build Prompts — 3D RF Simulation Pipeline (20 pasteable prompts)

A sequenced set of copy-pasteable prompts that carry the browser-based 3D RF simulation
forward. Each says *what to code, which files, how it wires to the frontend, and whether
it must run on a faster processor (Colab GPU / WebGPU)*. Paste one **P#** body at a time,
in order.

## Current state (why these start where they do)

The frontend restructure (mode chooser → import → preprocess → workspace, plus the 3D
Simulation construction shell) is **already built and functional**:
- Screen router + `window.appMode`/`window.appImport`: `Frontend/landing/landing.js`.
- Root symlinks `SIM → Physics Engine/2D/SIM`, `SIM3D → Physics Engine/3D Map Physics/SIM V1 3D`.
- Importmap: `three@0.185.1` (WebGPU build) + `cannon-es@0.20.0` in `Frontend_Data_Display.html`.
- 3D Simulation shell: `Frontend/simulator/simulation3d.js` — place/collide Tx+Rx from
  `window.SIM3D_ANTENNA_CATALOG`, WiFi band select, Static/Animated buttons, `#viz3dMode`
  Visualize dropdown, `window.__sim3d` debug handle.

But the physics is **honest placeholder**: `runStaticField()` (`simulation3d.js:233`) is a
free-space-path-loss point cloud; `runAnimWave()` (`:278`) is a kinematic expanding sphere.
Meanwhile the **backend is ahead of the browser**: `SIM3D/engine_3d.py` (`SceneV3`) already
computes real through-wall loss (Fresnel/Airy `CrossingLUT`) **and** a true 3-D eikonal
arrival time (`skfmm`) — but nothing exports that volume to the browser, and six physics
modules are **0-byte stubs** (`Physics Engine/3D Map Physics/Wave Behavior/Enivronmental
Interaction/{Absorption,Diffraction,Path_Loss,Reflection,Refraction,Scattering}_3D.py`), as
is `Reciever Objects/Construct_Reciever_3D.py`.

These 20 prompts fill that gap along the roadmap: *vacuum → indoor realistic → outdoor →
ground-truth*, plus GPU offload and ML.

## Header legend

- **Compute** — `Local CPU` (your machine), `Colab GPU` (offload heavy solve/train), or
  `WebGPU` (browser GPU).
- **Skill** — when an `anthropic-skills:engine-v1-3d-*` physics spec governs the work,
  invoke it first (`/engine-v1-3d-…`).

## Facts every prompt can rely on (verified)

- Frontend 3D module: `Frontend/simulator/simulation3d.js`; DOM ids `#sim3dViewport`,
  `#sim3dStatus`, `#tx3dType/#rx3dType`, `#wf3dBand`, `#run3dStatic/#run3dAnim`,
  `#anim3dPlay/#anim3dScrub`, `#viz3dMode`; globals `window.__sim3d`,
  `window.SIM3D_ASSETS/_COLLISION/_ANTENNA_CATALOG`, `window.appMode`.
- Grid: **262 × 11 × 118 voxels @ 0.30 m**, extent `[78.6, 3.3, 35.4] m`, Y-up.
  Materials (`manifest_3d.materials`, `loss_db`): air 0 · drywall 4 · concrete 15 · core 20 ·
  furniture 0.3 dB/m · glass 3. Bands `[2442, 3500, 5500, 6125] MHz`. Norm `pl_min 40, pl_range 130`.
- Backend 3D engine: `SIM3D/engine_3d.py` → `SceneV3(M, manifest)`, `.pathloss_maps(tx)` →
  `(nf,262,11,118)`, `.arrival_time(tx,f)` (eikonal T), `.band_index(f)`; EM kernels in
  `SIM3D/physics_3d.py` ← `SIM/physics_v2.py`.
- Browser-asset exporters: `SIM3D/export_web3.py`, `export_collision_3d.py`,
  `export_antenna_catalog_3d.py`.
- ML: `SIM3D/phase_b3_dataset.ipynb` (dataset; only 3 shards committed),
  `SIM3D/phase_c3_train_colab.ipynb` (`UNet3D`, input `(1,9,262,11,118)`, exports
  `SIM3D/web/pl_unet3d.onnx`, opset 17, wasm EP). `onnxruntime-web@1.19.2` already in the page.
  2-D precedent: `SIM/web/simulator_tab.js` `pathlossOnnx()` + `pl_unet.onnx`.
- Ground truth: `ARCHIVE/raw_walk_data/` (215 scanner CSVs + `Concat_Indoor_Walk_Test_from_csv.csv`,
  182 cols) — `Latitude/Longitude`, `Ref Signal - Received Power`=RSRP, `…Received Quality`=RSRQ,
  `…CINR`=SINR, `Carrier RSSI Antenna Port n`=RSSI, **no elevation**. Georef:
  `Essentials + HTML/floorplan/7th_Floor_2nd_Indoor_Walk_Test_V2.2.TAB` + `…_PseudoMercator.csv`
  (3 GCPs, Web-Mercator).

---

# PHASE A — Client-side visualizations & cleanup (no Colab; quick wins)

## P1 · Foundation fixes: reconcile docs↔code, pin deps, 3D Makefile, WebGPU render bugs
**Compute:** Local CPU

> Before adding physics, make the existing 3-D pipeline trustworthy and reproducible.
> 1. **Reconcile the docs/code gap.** `SIM3D/manifest_3d.json` (`physics` block) and
>    `SIM3D/MODEL_CARD_3D.md` still describe the *shipped v1* as flat-dB Motley-Keenan with
>    geometric `T = d/c`, but `SIM3D/engine_3d.py` (`SceneV3`) already computes per-crossing
>    **Fresnel/Airy** loss via `physics_3d.CrossingLUT` and a real 3-D **eikonal** `T` via
>    `skfmm`. Update the model card + a `physics.model` version string to describe what the
>    code actually does; note the flat-`loss_db` table is now only a fallback/label source.
> 2. **Pin the missing dependency.** `SIM3D/engine_3d.py` imports `skfmm` (scikit-fmm) but
>    `SIM3D/requirements_3d.txt` lists only numpy/scipy/matplotlib. Add `scikit-fmm`.
> 3. **Add 3-D Makefile targets.** `Essentials + HTML/Makefile` has 2-D targets only
>    (`prepare/test/dataset/assets/model`). Add a `3d` group mirroring them: `voxelize`
>    (→ `SIM3D/voxelize.py`), `calc-3d` (→ `run_one_calc.py`), `assets-3d`
>    (→ `export_web3.py export_collision_3d.py export_antenna_catalog_3d.py`), all writing
>    into `SIM3D/web/…`. Keep it runnable from repo root.
> 4. **Fix the WebGPU render stand-ins in the 2-D viewer.** `Frontend/2d-3d/viewer3d.js`
>    uses `THREE.Points`+`PointsMaterial` (`buildRFPoints` ~761, `buildPath` ~793) and a
>    `wireframe=true` toggle (`applyWireframe` ~528). On the `three@0.185.1` **WebGPU**
>    backend these are mis-sized/no-ops (documented in `Frontend/simulator/README.md`).
>    Replace the point clouds with small `InstancedMesh` cubes (as `simulation3d.js` already
>    does) and replace the wireframe toggle with an `EdgesGeometry`/`LineSegments` overlay.
>
> **Done when:** `make voxelize calc-3d assets-3d` regenerates `SIM3D/web/*`; the model card
> matches the code; and the Map-Coverage 3-D "Floating dots" + Wireframe toggle render
> correctly on WebGPU with zero console warnings.

## P2 · Analytic geometry-aware static field (replace the FSPL point cloud)
**Compute:** Local CPU (browser JS) · **Skill:** `/engine-v1-3d-simulation-map-path-loss`

> Replace the free-space stand-in in `runStaticField()` (`simulation3d.js:233–271`) with an
> **in-browser geometry-aware path-loss volume** so the Static button already respects walls
> before the heavy solver lands. Port the exact 2-D crossing math from
> `SIM/web/simulator_tab.js` `pathlossPhysics()` (`:35`, the JS mirror of the engine) up to
> 3-D: march a ray from the Tx voxel to each sample voxel through `window.SIM3D_ASSETS`
> (decode `grid_b64` int8 material grid, `grid_shape 262×11×118`), accumulate per-material
> `loss_db` for each wall crossing + Beer-Lambert furniture (`loss_per_m_db`), add
> Motley-Keenan spreading `fspl1m(f) + 10·n·log10(d)` (`manifest_3d.physics`: `n_exp 2`,
> `fspl_const_db`, `freq_loss_mult`). Render the resulting `PL(x,y,z)` as the existing
> coloured `InstancedMesh` cube field (keep the `ramp()` viridis + the "drop faint far-field"
> cull), now shadowed by walls. Add a small dB colour-legend under `#sim3dStatus`.
>
> **Files:** `Frontend/simulator/simulation3d.js` (rewrite `runStaticField`, add
> `decodeGrid()` + `marchPL(tx, sample, freq)`); styles in `Frontend/simulator/simulator.css`.
> **Wiring:** reads `window.SIM3D_ASSETS.manifest_3d` + `grid_b64`; keeps `#wf3dBand`,
> `#viz3dMode="Propagation-coverage"`, `window.__sim3d.field`.
> **Reuse:** `pathlossPhysics()` crossing/step logic (`simulator_tab.js:35–61`).
> **Done when:** placing a Tx behind a concrete wall shows a clear shadow vs. line-of-sight;
> status labels it "analytic Motley-Keenan (in-browser)".

## P3 · Radiation-pattern visualization (3-D antenna lobes)
**Compute:** Local CPU (browser JS)

> Implement the **Radiation pattern** option of `#viz3dMode` — "View 3D lobes showing antenna
> gain and direction." For the selected/placed Tx antenna `kind`, build an analytic
> gain-pattern mesh (a deformed sphere whose radius = normalized gain `G(θ,φ)`) anchored at
> the Tx and oriented to its boresight, coloured by gain. Start with closed-form patterns
> keyed off catalog kind: isotropic (omni/dipole → `sin` torus), directional
> (patch/panel/horn/dish/yagi → cos^n main lobe + back lobe), phased_array/lpda placeholder.
> Pull orientation/`default_params` from `window.SIM3D_ANTENNA_CATALOG.entries[kind]`.
>
> **Files:** `Frontend/simulator/simulation3d.js` (new `runRadiationPattern()` +
> `gainMesh(kind, params)`); wire into the `#viz3dMode` change handler (`:319`).
> **Wiring:** `window.SIM3D_ANTENNA_CATALOG`; renders at each placed-Tx position; expose as
> `window.__sim3d.lobe`.
> **Reuse:** `buildAntennaMesh()` pattern (`:171`).
> **Backend hook (later):** real patterns live in `Object and Tranmission/Transmitter
> Objects/Antenna_Physics_3D.py` — a follow-up can export a per-kind gain table via
> `export_antenna_catalog_3d.py`.
> **Done when:** switching Visualize→Radiation pattern draws a correctly-oriented lobe;
> changing `#tx3dType` changes the lobe shape.

## P4 · Field-distribution visualization (oscillating E/H vectors)
**Compute:** Local CPU (browser JS), WebGPU-ready

> Implement the **Field distribution** mode — "oscillating electric and magnetic field
> vectors move through space." On a coarse 3-D lattice inside the building interior (use
> `SIM3D_ASSETS.inside_b64` mask), draw an `InstancedMesh` of arrow glyphs whose direction =
> local propagation/E-vector from the Tx and whose length/colour = field amplitude
> ∝ `10^(−PL/20)` (reuse P2's `marchPL`). Animate the phase `cos(k·r − ωt)` in
> `setAnimationLoop` so vectors oscillate; k from `#wf3dBand`. Keep instance count bounded
> (≤ ~4–6k arrows) for CPU; structure the update so it can later move to a WebGPU compute
> pass (P13).
>
> **Files:** `Frontend/simulator/simulation3d.js` (`runFieldVectors()`, phase update in the
> animation loop `:139`).
> **Wiring:** `#viz3dMode="Field distribution"`, `#wf3dBand`; `window.__sim3d.vectors`.
> **Done when:** vectors point away from the Tx, shorten with distance/behind walls, and
> visibly oscillate; toggling Visualize off disposes them.

## P5 · Interference & multipath visualization (complex phasor sum)
**Compute:** Local CPU (browser JS), WebGPU-ready

> Implement **Interference-multipath** — "constructive and destructive wave overlap." When
> ≥2 transmitters are placed (or one Tx + a first specular image across the nearest wall),
> sum complex phasors per voxel: `Σ_i A_i · e^{j(k·d_i + φ_i)}`, with `A_i = 10^(−PL_i/20)`
> from P2's `marchPL` and `d_i` the path length. Colour the `InstancedMesh` field by `|Σ|`
> on a diverging ramp (bright = constructive, dark = destructive nulls). Add a
> phase-animation toggle reusing P4's clock. Include a single image-source reflection off the
> nearest wall as the "multipath" seed (full specular tree is P7).
>
> **Files:** `Frontend/simulator/simulation3d.js` (`runInterference()`, `phasorSum()`).
> **Wiring:** `#viz3dMode="Interference-multipath"`; uses all `placed` Tx; the **Interference
> physics** checkbox in the Receiver rail gates whether Rx readout (P14) includes it;
> `window.__sim3d.interference`.
> **Done when:** two Tx at the same band show a standing-wave fringe pattern that shifts when
> one Tx is moved.

---

# PHASE B — Real physics solvers (heavy → Colab GPU) + browser export

## P6 · Export the real PL(x,y,z) + T(x,y,z) volume to the browser
**Compute:** Local CPU (single calc) / Colab GPU (full band set) · **Skill:** `/engine-v1-3d-simulation-map-path-loss`

> The real solver already exists (`SIM3D/engine_3d.py` `SceneV3` → `pathloss_maps(tx)`,
> `arrival_time(tx,f)`), but the browser never sees it. Build the bridge:
> 1. Extend `SIM3D/export_web3.py` to serialize a computed **PL volume** (and eikonal
>    **T volume**) for a chosen Tx into a compact binary — e.g.
>    `SIM3D/web/pl_volume_<txid>.bin` (float16, `262×11×118×nbands`) + a JSON sidecar (tx
>    voxel, bands, norm). Keep the geometry export (`sim_assets_3d.js`) unchanged.
> 2. In `simulation3d.js`, add a loader that, when a precomputed volume exists for the placed
>    Tx, renders **it** instead of the P2 analytic field (fetch + decode into the same
>    `InstancedMesh` cube field / P15 heatmap). Fall back to P2 when absent. Label the source
>    honestly in `#sim3dStatus` ("full-physics solve" vs "analytic").
> 3. A full multi-Tx sweep is heavy → generate volumes on **Colab** with a small batch script
>    (reuse `SIM3D/run_one_calc.py`, which already writes `preview/pl_volume.npy`,
>    `t_volume.npy`).
>
> **Files:** `SIM3D/export_web3.py` (+ `export_pl_volume.py` or a flag), `SIM3D/run_one_calc.py`
> (batch mode), `Frontend/simulator/simulation3d.js` (volume loader + render path).
> **Wiring:** new `window.SIM3D_VOLUME` (or fetch) consumed by `runStaticField`.
> **Done when:** Static on a Tx with a precomputed volume shows the eikonal-shaped field
> (curved shadows, in-wall lag) distinct from the analytic P2 result.

## P7 · Reflection map (`Reflection_3D.py`)
**Compute:** Colab GPU · **Skill:** `/engine-v1-3d-simulation-map-reflection`

> Fill the 0-byte stub `…/Enivronmental Interaction/Reflection_3D.py`. Implement specular
> reflection (image-source method up to N bounces off the voxel wall set) producing a
> **reflected-power contribution volume** co-registered with `SceneV3`'s direct PL. Reuse
> Fresnel reflection coefficients in `SIM/physics_v2.py` (`fresnel_coeffs`) and wall normals
> `physics_3d.wall_normals_3d`. Power-sum the reflected paths into the direct field (linear
> power), as the engine docstring anticipates. Export via P6 so the browser shows it as a
> Visualize→**Reflection** mode (currently a disabled "coming soon" option in `#viz3dMode`).
>
> **Files:** `Reflection_3D.py` (new impl), hook in `engine_3d.py` to accept an additive
> contribution, `export_web3.py` for the extra channel; enable Reflection in `simulation3d.js`.
> **Compute note:** N-bounce image sources over 340k voxels × bands is the first true
> **GPU-offload** solver — prototype in NumPy, then port the ray batch to torch on Colab
> (mirror `SIM/engine_v2_torch.py`, the existing CUDA precedent).
> **Done when:** a glossy corridor shows a brighter reflected lobe along the wall vs.
> direct-only.

## P8 · Refraction map (`Refraction_3D.py`)
**Compute:** Colab GPU · **Skill:** `/engine-v1-3d-refraction-map-reflection`

> Fill `…/Enivronmental Interaction/Refraction_3D.py`. Model Snell bending + in-slab
> transmission delay through dielectrics (drywall/glass), producing a refraction correction
> to both PL and arrival-time. The eikonal `T` in `SceneV3.arrival_time` **already captures
> refraction implicitly** (slowness `Re(√εr)/c`) — this module makes the **ray-bending
> contribution explicit and visualizable** and reconciles transmitted-ray geometry with the
> crossing-loss LUT. Reuse `physics_v2.permittivity` (ITU-R P.2040) +
> `slab_transmission_coherent`.
>
> **Files:** `Refraction_3D.py`; export channel via P6; enable Visualize→**Refraction**.
> **Done when:** rays through glass show measurable bend/lag vs. straight-line, matching the
> eikonal front.

## P9 · Diffraction map (`Diffraction_3D.py`)
**Compute:** Colab GPU · **Skill:** `/engine-v1-3d-simulation-map-diffraction`

> Fill `…/Enivronmental Interaction/Diffraction_3D.py`: UTD/knife-edge diffraction around
> wall edges and door frames so signal wraps into geometric shadow (adds the
> **diffraction-loss field**; the eikonal already wraps corners). This is the 3-D analogue of
> the working 2-D `SIM/engine_v2.py` (`find_diffracting_edges`, `diffracted_maps`) — port that
> edge-finding to the 3-D voxel grid and reuse `physics_v2.utd_coefficient` /
> `utd_pathloss_db` / `knife_edge_j_db` verbatim. Power-sum into the field.
>
> **Files:** `Diffraction_3D.py` (port from `engine_v2.py` + `physics_v2.py`); export via P6;
> enable Visualize→**Diffraction**.
> **Compute note:** edge enumeration × receiver voxels is heavy → Colab/torch.
> **Done when:** the deep shadow behind a corner shows a finite diffracted field that decays
> with shadow depth, not a hard zero.

## P10 · Absorption & attenuation map (`Absorption_3D.py`)
**Compute:** Colab GPU · **Skill:** `/engine-v1-3d-absorption-and-attenuation-map`

> Fill `…/Enivronmental Interaction/Absorption_3D.py`: bulk material absorption (Beer-Lambert
> along the ray using `Im(εr)`/`loss_per_m_db`, e.g. furniture clutter 0.3 dB/m) as an
> explicit, separable attenuation field distinct from the interface (Fresnel) loss already in
> the `CrossingLUT`. Produce an absorbed-energy volume for Visualize→**Absorption**.
>
> **Files:** `Absorption_3D.py`; export via P6; enable Visualize→**Absorption**.
> **Reuse:** `physics_v2` imaginary-permittivity path; the `per_metre` material flags in
> `engine_3d.py`.
> **Done when:** a furniture-dense zone shows steady per-metre dimming separate from wall
> steps.

## P11 · Scattering map (`Scattering_3D.py`)
**Compute:** Colab GPU · **Skill:** `/engine-v1-3d-simulation-map-scattering`

> Fill `…/Enivronmental Interaction/Scattering_3D.py`: diffuse scattering off rough surfaces
> (effective-roughness / directive-scattering model) redistributing a fraction of incident
> power into non-specular directions — the last of the six RF effects. Produce a
> scattered-power volume for Visualize→**Scattering**, power-summed with the rest.
>
> **Files:** `Scattering_3D.py`; export via P6; enable Visualize→**Scattering**.
> **Compute note:** heaviest solver (every lit surface → many directions) → definitely
> Colab/torch; consider a coarse angular basis.
> **Done when:** rough concrete surfaces show a soft scattered halo; a "scattering strength"
> slider changes its intensity.

---

# PHASE C — Animated wave + GPU offload

## P12 · Time-domain animated wave (replace the kinematic sphere)
**Compute:** Colab GPU (precompute) → WebGPU (playback)

> Replace `runAnimWave()`/`advanceWave()` (`simulation3d.js:278–303`, a kinematic expanding
> sphere) with a **real time-domain wavefront**. Wire in the existing scalar-wave FDTD
> sandbox: `Physics Engine/3D Map Physics/Wave Behavior/Wave Generation/Spatial_Physics.py`
> (`laplacian`, `speed_field`, `courant_dt`) + `Time_Domain_Physics.py` (leapfrog) +
> `Complied_Wave_Behavior.ipynb`. Run the FDTD on the voxel `speed_field` (from
> `material_grid.npy`) on **Colab**, export a downsampled sequence of pressure/intensity
> snapshots (`SIM3D/web/wave_frames_<txid>.bin`, e.g. 60–120 frames), and in the browser play
> them through `#anim3dPlay`/`#anim3dScrub` as a time-varying `InstancedMesh` field. This is
> your "Animated Wave 3D Simulation … in Space and Time."
>
> **Files:** Colab exporter (new `SIM3D/export_wave_frames.py` driving the Wave-Generation
> code), `Frontend/simulator/simulation3d.js` (frame loader + scrub playback in the animation
> loop).
> **Wiring:** `#anim3dControls/#anim3dPlay/#anim3dScrub`, `window.__sim3d.wave`.
> **Compute note:** FDTD respects the "pick two of {resolution, domain, speed}" limit — keep
> the domain the 262×11×118 grid, resolution coarse, precompute offline.
> **Done when:** Animated Wave plays a wavefront that **reflects and diffracts** off walls
> (not a perfect sphere) and scrubs frame-accurately.

## P13 · WebGPU compute offload (interactive in-browser solve)
**Compute:** WebGPU (browser GPU)

> Move per-voxel field math off the CPU so Static/animated views update at interactive rates
> without a Colab round-trip. Using the `three@0.185.1` **WebGPU** backend already in use
> (`renderer = new THREE.WebGPURenderer`, `simulation3d.js:97`), implement the P2 ray-march
> path-loss and the P4/P5 phasor sums as **TSL/WebGPU compute shaders** writing a storage
> texture that feeds the instanced field / P15 heatmap. Provide a capability check + CPU
> fallback (WebGL2) matching `viewer3d.js`.
>
> **Files:** `Frontend/simulator/simulation3d.js` (compute pipeline), new
> `Frontend/simulator/field_compute.js` (TSL kernels).
> **Wiring:** consumes `SIM3D_ASSETS` grid textures; outputs the same `InstancedMesh`/heatmap
> the CPU path used, so P2/P4/P5 switch to GPU transparently.
> **Note:** this is the "offload from my CPU" item for the *interactive* case; the *heavy
> solver* case stays on Colab (P6–P11).
> **Done when:** re-solving Static after moving the Tx updates in <100 ms on a WebGPU browser;
> falls back cleanly where WebGPU is unavailable.

---

# PHASE D — Measurement, coverage heatmap, ML surrogate

## P14 · Receiver measurement layer — RSRP / RSSI (then SINR / RSRQ)
**Compute:** Local CPU (browser JS + `Construct_Reciever_3D.py`)

> Turn placed Receivers into real measurement points. Implement the 0-byte stub
> `Object and Tranmission/Reciever Objects/Construct_Reciever_3D.py` (mirror
> `Construct_Transmitter_3D.py` `Transmitter3D`: an `Rx3D` with `position_m`, antenna kind,
> gain), and in `simulation3d.js` compute, at each placed Rx voxel, from the field volume:
> - **RSRP** = per-RE received power = `EIRP + G_rx − PL(rx)`;
> - **RSSI** = total in-band power (RSRP + interference + noise), gated by the **Interference
>   physics** checkbox (sum other Tx via P5 phasors when checked);
> - then **SINR** and **RSRQ** as follow-on (need noise floor + resource-block model).
> Show a per-Rx readout (dBm) in the Receiver rail, updating live as the Tx/field changes.
>
> **Files:** `Construct_Reciever_3D.py`; `Frontend/simulator/simulation3d.js` (`measureAtRx()`
> + readout DOM); reuse the 2-D convention `P_rx = EIRP − PL` from `simulator_tab.js`.
> **Wiring:** `#rx3dList` entries gain a dBm readout; Interference checkbox in the Rx rail;
> `window.__sim3d.measurements`.
> **Done when:** moving an Rx nearer the Tx raises RSRP; enabling Interference with a 2nd Tx
> lowers SINR.

## P15 · Volumetric coverage heatmap (RSRP/RSSI over lat/long/elevation)
**Compute:** WebGPU (raymarch) / CPU fallback

> Render the field volume as a **true volumetric 3-D heat map** over the model — "coverage on
> lat, long, and elevation" — instead of discrete cubes. Implement a raymarched volume
> (WebGPU 3-D storage texture from the PL/measurement volume) or a stacked
> transparent-slice fallback, with an opacity transfer function and a dBm colour-legend keyed
> to the selected metric (RSRP/RSSI). Register it to the building extent so it overlays the
> voxel shell.
>
> **Files:** `Frontend/simulator/simulation3d.js` + new `Frontend/simulator/volume_render.js`.
> **Wiring:** metric select (reuse the Map-Coverage `#metric` pattern) + `#viz3dMode`
> "Propagation-coverage"; consumes P6/P14 volumes; `window.__sim3d.heatmap`.
> **Done when:** a semi-transparent coloured coverage cloud fills the building, brightest at
> the Tx, legend in dBm.

## P16 · 3-D ML surrogate — train `pl_unet3d.onnx`, run in browser
**Compute:** Colab GPU (dataset + train) → wasm (browser inference)

> Stand up the deep-learning path so PL(x,y,z) is predicted in ~ms instead of solved. The
> scaffolding exists: `SIM3D/phase_b3_dataset.ipynb` (only `shard_000..002` committed —
> generate the full Tx×band set with `SceneV3`) and `SIM3D/phase_c3_train_colab.ipynb`
> (`UNet3D`, base 16, anisotropic `MaxPool3d((2,1,2))`, input `(1, 9, 262, 11, 118)` =
> 6 material one-hot + Tx Gaussian blob + freq feature + log-distance; exports
> `SIM3D/web/pl_unet3d.onnx`, opset 17). Then wire browser inference in `simulation3d.js`
> mirroring the working 2-D `simulator_tab.js` `pathlossOnnx()`
> (`ort.InferenceSession.create(..., {executionProviders:[['wasm']]})` — WebGPU EP does
> **not** cover `Conv3d`). `onnxruntime-web@1.19.2` is already loaded in the page.
>
> **Files:** run the two notebooks on Colab (GPU) → `pl_unet3d.onnx`;
> `Frontend/simulator/simulation3d.js` (ONNX session + 9-channel tensor builder + denorm
> `y·130 + 40`, physics fallback on `file://`).
> **Wiring:** a "Surrogate (ML)" vs "Physics" engine toggle in the Run rail (mirror 2-D
> `#simEngine`); `window.__sim3d.onnx`.
> **Why ML:** measurement is "very nonlinear, spatial and time-wise" — the U-Net learns the
> spatial nonlinearity; the temporal extension is P16b.
> **Done when:** the ML toggle produces a PL volume visually matching the solver within a few
> dB, far faster; falls back to physics when the model is absent.

*(P16b — deferred stretch: extend the surrogate to spatio-temporal (predict the P12 wave
sequence) with a 3-D+time or ConvLSTM net; Colab GPU. Marked hook.)*

---

# PHASE E — Voxelization, cellular, outdoor, ground truth

## P17 · Live in-browser voxelization (import → preprocess for arbitrary models)
**Compute:** Local CPU (browser) — heavy models may fall back to Colab `voxelize.py`

> Today Preprocess only shows the **precomputed** grid; a user-imported model isn't voxelized.
> Port `SIM3D/voxelize.py` (NumPy OBJ parse + triangle raster + `scipy.ndimage` fill →
> `material_grid` / `inside_mask` / collision boxes) into the browser so an imported
> `.obj/.glb/.dae` (via the existing `viewer3d.js` loaders + `window.__viewer3d.loadFiles`)
> becomes a live voxel grid + material table + collision set, regenerating
> `window.SIM3D_COLLISION/_ASSETS` in memory. Add a material-tagging UI (paint faces →
> material class) on the Preprocess screen. Keep the precomputed grid as the default/fast
> path; offer "voxelize this model" on demand.
>
> **Files:** new `Frontend/landing/voxelize_web.js` (or a Web Worker), `Frontend/landing/landing.js`
> (Preprocess wiring), `viewer3d.js` (expose loaded geometry).
> **Wiring:** rebuilds `window.SIM3D_COLLISION.boxes`, `window.SIM3D_ASSETS.grid_b64`; feeds
> the whole 3-D Simulation tab.
> **Done when:** importing a fresh OBJ and clicking "Voxelize" produces a placeable,
> collidable building the Simulation tab can use — no Python step.

## P18 · Cellular band presets (LTE/NR) in Waveform generation
**Compute:** Local CPU (frontend + manifest)

> Extend Waveform generation beyond WiFi. Add a cellular band group to the `WIFI_BANDS` select
> source (`simulation3d.js:29`, rename to `BAND_PRESETS` with `Indoor WiFi` / `Cellular`
> optgroups) and the 2-D `#simBand`, matching the scanner's actual bands so simulation and
> ground truth (P20) share a band axis: e.g. **LTE B71 617 MHz, B2 1960 MHz, NR n41 2506 MHz,
> n77 3750 MHz** (from `STEP_2/multiband/walk_trace.csv`), plus engine bands
> `[2442, 3500, 5500, 6125]`. Ensure `manifest_3d.freqs_mhz` / `freq_loss_mult` cover the new
> centres (extend the solver band list + re-export).
>
> **Files:** `Frontend/simulator/simulation3d.js` (`BAND_PRESETS` + optgroups),
> `SIM3D/manifest_3d.json` (+ re-run `voxelize.py`/`export_web3.py`), the 2-D band list.
> **Wiring:** `#wf3dBand`, `#simBand`.
> **Done when:** the Waveform dropdown offers WiFi **and** cellular centres; solving at a
> cellular band produces a plausible (lower-loss at 617 MHz) field.

## P19 · Outdoor mode — activate the deferred path
**Compute:** Local CPU + Colab GPU (larger domains)

> The chooser shows **Outdoor** disabled ("separate project — coming soon",
> `Frontend/indoor-outdoor/`). Unify it: enable the Outdoor pill in `landing.js`, bring in the
> separate `Outdoor_Walk_Test_7-24` project's model/domain, and route the 3-D Simulation tab
> to an outdoor voxel/terrain grid. Reuse the 2-D outdoor machinery already present —
> `SIM/phase_a.py` `bs_maps()`/`facade_sources()` (plane-wave base stations, O2I loss) and the
> precomputed `SIM_ASSETS.bs` 8-bearing maps — as the outdoor Tx model.
>
> **Files:** `Frontend/landing/landing.js` (enable Outdoor + branch), `Frontend/indoor-outdoor/`
> (mode logic), outdoor manifest/grid, engine outdoor entry.
> **Wiring:** `window.appMode.environment==='outdoor'` gates the tab variants.
> **Compute note:** outdoor domains are large → Colab for full solves; browser for the shell.
> **Done when:** choosing Outdoor→3D loads an outdoor scene and places a base-station Tx with a
> coverage field.

## P20 · Ground-truth comparison vs. the walk-test scanner
**Compute:** Local CPU (align/metrics) + browser overlay

> Close the loop: compare simulated coverage against the real scanner data — your "3D
> environment heat map that matches up with the scanner." Load
> `ARCHIVE/raw_walk_data/Concat_Indoor_Walk_Test_from_csv.csv` (or the 215 per-band CSVs) —
> columns `Latitude/Longitude`, `Ref Signal - Received Power`(RSRP), `…Received Quality`(RSRQ),
> `…CINR`(SINR), `Carrier RSSI Antenna Port n`(RSSI) — and georeference to the floor via
> `Essentials + HTML/floorplan/7th_Floor_2nd_Indoor_Walk_Test_V2.2.TAB` + `…_PseudoMercator.csv`
> (3 GCPs, Web-Mercator: `(lon,lat) ↔ pixel/mercator`). Sample the simulated volume along the
> walk path (single 7th-floor elevation — the scan has **no altitude column**) and report
> error metrics (ME/RMSE/correlation per band) + a side-by-side measured-vs-simulated overlay.
> Reuse the existing `rfCsv` importer (`viewer3d.js` `parseRFCsv`, exposed via
> `window.__viewer3d.importCsvText`).
>
> **Files:** new `Frontend/2d-3d/ground_truth.js` (CSV load + GCP affine + metrics), a
> comparison panel (new tab or Statistics sub-view); reuse `parseRFCsv`.
> **Wiring:** overlays onto the Map-Coverage/heatmap; `window.appMode`-aware.
> **Compute note:** parsing 215 CSVs is IO-heavy — consider a one-time concat/precompute
> (there's already `Concat_Indoor_Walk_Test_from_csv.csv`).
> **Done when:** the app shows measured RSRP vs. simulated RSRP along the 7th-floor walk with
> an RMSE number per band — the calibration target for "fix indoor to be realistic".

---

## Verification (per prompt + end-to-end)

1. **Serve from repo root:** `python3 -m http.server 8432`, open
   `http://localhost:8432/Frontend_Data_Display.html` — console shows **zero 404s**.
2. **Per-prompt "Done when"** is the acceptance check — a concrete browser behavior or a
   regenerated asset.
3. **Regression after each frontend prompt:** Map-Coverage 2-D heatmap renders, 2-D Simulation
   solves, tab switching works, 3-D place/collide still runs (`window.__sim3d.placed`).
4. **Backend solvers:** validate each new `_3D.py` with a `run_one_calc`-style single-Tx check
   (assert PL ≥ FSPL, energy conservation vs. vacuum) before wiring to the browser.
5. **Colab artifacts** (`pl_unet3d.onnx`, `*_volume.bin`, `wave_frames.bin`) are git-ignored —
   regenerate from the committed shards/notebooks; don't expect them in the tree.

```bash
python3 -m http.server 8432
```
