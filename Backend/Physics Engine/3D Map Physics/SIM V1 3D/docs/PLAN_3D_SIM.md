# 3D RF Propagation Simulator — Plan & Roadmap

Physics-first 3D RF simulator: the EM mechanisms from *Introduction to RF Propagation* (Seybold),
rendered in the browser with a time-lapse of the propagation mechanism, accelerated by a
deep-learning surrogate trained on physics-generated data, and cross-validated against PCTEL
G-flex scanner measurements.

**Status legend:** ✅ done · 🚧 in progress · ⬜ planned · 🔬 research branch · ⛔ blocked

**Where this stands:** M0–M3 are shipped and green. M4 is *scaffolded but deliberately gated* —
the dataset notebook refuses to generate against the current scene, and the reason is a physics
bug, not a missing feature. That gate is Pre-M4, and it is the next thing to build.

```
$ python3 TESTS3D/run_all.py --full
129 passed in 25.30s
PASS

$ python3 TESTS3D/run_all.py --fast --selftest
mechanism self-tests:
  Path_Loss_3D       OK
  Reflection_3D      OK
  Refraction_3D      SKIP (not implemented yet)
  Diffraction_3D     OK
  Absorption_3D      SKIP (not implemented yet)
  Scattering_3D      OK
  Combine_3D         OK
  dataset_3d         OK
PASS
```

---

## 0. Design principles

1. **Physics first, ML second.** The deterministic engine is the source of truth and the permanent
   fallback. The surrogate is an *accelerator*, never a dependency — if it is missing the simulator
   still runs correctly, just slower.
2. **Reuse over rebuild.** `physics_v2.py` (2D) already implements ITU-R P.2040-3 materials, Fresnel
   coefficients, coherent/incoherent slab transmission and full Kouyoumjian–Pathak UTD. `SceneV3`
   already solves 3D path loss and a true eikonal arrival time. We wrap these; we do not re-derive them.
3. **Complex fields, physics-exact combining.** Each mechanism emits a complex field. The combiner
   is *not* learned — coherent sum for correlated/specular paths, incoherent for the diffuse tail.
   Summing powers instead of fields destroys the interference structure that makes a map read as
   measured rather than rendered.
4. **Offload + cache.** Heavy solves run on external GPU; results are cached in the browser's volume
   format so interaction stays real-time, visually and analytically.
5. **Honesty about limits.** Where the data cannot support a claim, the report says so (see §9).
6. **A gate you can raise is not a gate.** Every acceptance check in this plan is a test or an
   assertion in the code path it guards, not a line in a document.

---

## 1. The pipeline, end to end

```
                    Sandbox_Version_3D_Simulation_1.obj  (253 named materials)
                                      │
                       voxelize.py ───┤ sandbox_material_map.py  (regex → 6 classes,
                       --seal-ceiling │                           ~40 people/props excluded)
                       --floor-slab   │
                                      ▼
                material_grid.npy · inside_mask.npy · valid_tx_mask.npy · manifest_3d.json
                                      │                     262 × 17 × 132 @ 0.30 m
                                      ▼
        ┌──────────────────── engine_3d.SceneV3 ────────────────────┐
        │  crossing_loss()  → amplitude   (physics_v2 CrossingLUT)  │
        │  arrival_time()   → phase       (3-D eikonal, skfmm)      │
        └───────────────────────────┬───────────────────────────────┘
                                    │  wrapped by, never bypassed
        ┌───────────────┬───────────┼───────────────┬────────────────┐
        ▼               ▼           ▼               ▼                ▼
  Path_Loss_3D    Reflection_3D  Diffraction_3D  Scattering_3D   (Refraction_3D)
  coherent E      coherent E     coherent E      p_incoh         (Absorption_3D)
        └───────────────┴───────────┼───────────────┘                 ⬜ not built
                                    ▼
                    Combine_3D.combine(bandwidth_hz=20e6)
                    coherent Σ E · incoherent Σ p · T=min(τ) · FSPL floor
                                    │
                    ┌───────────────┼────────────────────┐
                    ▼               ▼                    ▼
          to_legacy_volumes()   dataset_3d shards   validate_scanner_3d.py ⬜
                    │            (M4 surrogate)        (M5, vs G-flex)
                    ▼                    │
    precompute_volumes.py ──▶ cache_index.py ──▶ web/volumes/*.bin + index.json
      (batch, A100)            content-addressed        float16, browser format
                                LRU, --verify                   │
                                                                ▼
                                            Frontend/simulator/simulation3d.js
                                    cached volume → DL surrogate → analytic fallback
```

The surrogate branch, which M4 fills:

```
phase_b3_dataset.ipynb ──▶ dataset/shard_*_{pl,tau}.npy + _meta.npz   (Colab CPU, ~1 h)
        │  preflight gate: refuses to run while the target saturates
        ▼
phase_c3_train_colab.ipynb ──▶ web/pl_unet3d.onnx + web/pl_unet3d.json  (Colab GPU, ~2 h)
        │                       checkpoints/ckpt_pl_unet3d.pt (resumable)
        ▼
   simulation3d.js tier 2   ⬜ session.run() not wired yet — see Pre-M4 §P5
```

Both notebooks and the browser import their tensor layout from **one** module,
`dataset_3d.py`, and the trainer copies it into `pl_unet3d.json`. A channel-order
disagreement between trainer and browser does not raise — it renders a confident,
wrong field — so the layout is data, asserted in `tests3d/test_dataset_3d.py`.

---

## 2. Directory map — what is built

`SIM3D/` is a repo-root symlink to `Physics Engine/3D Map Physics/SIM V1 3D/`;
`TESTS3D/` points at its `tests3d/`.

### `SIM V1 3D/` — the engine

| File | Bytes | Status | What it is |
|---|---:|:--:|---|
| `engine_3d.py` | 10.7 k | ✅ | `SceneV3`: path loss + eikonal arrival time. Everything wraps this. |
| `physics_3d.py` | 5.1 k | ✅ | 3-D geometry adapter over `2D/SIM/physics_v2.py`; `r_slab`, `speed_field`, `wall_normals_3d` |
| `contracts.py` | 8.0 k | ✅ | `FieldGrid` / `CombinedField` / `PathSet` — the complex-field spine |
| `modes_3d.py` | 26.6 k | ✅ | mode registry, `forte_hall_geometry()`, `facade_sources_3d`, `bs_field_3d` |
| `cache_index.py` | 25.4 k | ✅ | content-addressed volume cache, LRU, `--stats` / `--verify` / `--gc` |
| `precompute_volumes.py` | 21.9 k | ✅ | batch Tx sweep, mechanism channels, float16 export |
| `export_pl_volume.py` | 8.0 k | ✅ | single-Tx front door |
| `voxelize.py` | 14.3 k | ✅ | OBJ → voxels, `--seal-ceiling`, `--floor-slab-class` |
| `sandbox_material_map.py` | 9.2 k | ✅ | 253 OBJ material names → 6 classes, with exclusions |
| `voxelize_city.py` | 18.5 k | ✅ | outdoor city grid (2.5D/3D) — code shipped, **grid not built** |
| `export_web3.py` · `export_collision_3d.py` · `export_antenna_catalog_3d.py` | 12.1 k | ✅ | browser asset exporters |
| `export_registration.py` | 7.5 k | ✅ | px → local m → voxel fit |
| `run_one_calc.py` | 6.4 k | ✅ | one-shot solve + preview PNG |
| **`dataset_3d.py`** | **19.9 k** | **✅ new** | **the one definition of the surrogate input tensor** |
| **`phase_b3_dataset.ipynb`** | — | **✅ new** | **dataset generation, gated, resumable** |
| **`phase_c3_train_colab.ipynb`** | — | **✅ new** | **training, resumable, ONNX + contract export** |
| `validate_scanner_3d.py` | — | ⬜ | M5 — does not exist yet |
| `PLAN_3D_SIM.md` · `MODEL_CARD_3D.md` · `README.md` | — | ✅ | docs |

### `Wave Behavior/Enivronmental Interaction/` — the mechanisms

*(directory name is misspelled in the repo; paths depend on it)*

| Module | Bytes | Status | Emits | Backends |
|---|---:|:--:|---|---|
| `Path_Loss_3D.py` | 7.1 k | ✅ | coherent **E** | numpy |
| `Reflection_3D.py` | 16.4 k | ✅ | coherent **E** | numpy, torch |
| `Diffraction_3D.py` | 18.1 k | ✅ | coherent **E** | numpy |
| `Scattering_3D.py` | 19.8 k | ✅ | **p_incoh** | numpy, torch |
| `Combine_3D.py` | 11.2 k | ✅ | the combiner (not a mechanism) | numpy |
| `Refraction_3D.py` | **0** | ⬜ | — | — |
| `Absorption_3D.py` | **0** | ⬜ | — | — |

Both empty modules describe effects the engine *already applies* — in-wall slowdown lives in the
eikonal speed field, `Im(q)` bulk absorption lives inside the `CrossingLUT`. What is missing is
exposing them as separately viewable channels, which is why their two `#viz3dMode` options stay
disabled and honestly labelled rather than showing a plausible-looking placeholder.

### `Object and Tranmission/`

| Module | Bytes | Status |
|---|---:|:--:|
| `Antenna_Type_3D.py` | 49.2 k | ✅ 22 antenna kinds, parametric meshes |
| `Antenna_Physics_3D.py` | 6.3 k | ✅ collision bodies for placement |
| `Construct_Transmitter_3D.py` | 4.6 k | ✅ |
| `Reciever Objects/Construct_Reciever_3D.py` | **0** | ⬜ **blocks M5** — no PL → RSRP/RSRQ/SINR/RSSI |

Antenna geometry is render/collision only; no radiation pattern feeds the mechanism solvers yet.

### Frontend

| File | Status | Note |
|---|:--:|---|
| `Frontend_Data_Display.html` | ✅ | app shell, importmap, four workspace tabs |
| `Frontend/simulator/simulation3d.js` | 🚧 | volume cache, mechanism views, time-lapse, mode selector all live; **surrogate tier loads but never infers** |
| `Frontend/2d-3d/viewer3d.js` | 🚧 | hardcodes `FLOOR_W/FLOOR_H = 1150/515` and rubber-sheets geographic CSVs |
| `Frontend/2d-3d/georef.js` | ⬜ | does not exist — the missing px↔lonlat↔voxel bridge |
| `Frontend/2d-3d/dashboard.js` · `landing.js` · `spectrum.js` | ✅ | measured-walk dashboard and router |

### Artifacts not in git (regenerable, and correctly excluded)

| Path | Size | Rebuild |
|---|---|---|
| `SIM V1 3D/cache/` | ~376 MB | automatic on the first diffraction / O2I solve |
| `SIM V1 3D/city/` | ~350 MB | `voxelize_city.py --mode 2.5d --cell 1` — **outdoor mode reports itself unavailable until then** |
| `web/pl_unet3d.onnx` | — | M4 (this plan) |
| `dataset/shard_*` | ~8 GB | `phase_b3_dataset.ipynb` |

Cache today: 6 transmitters, 118.8 MB, `--verify` clean. Five of the six predate content
addressing and are served but never treated as hits (Pre-M4 §P9).

---

## 3. The six mechanisms

| Mechanism | Module | Emits | Physics source | Compute |
|---|---|---|---|---|
| Spreading / transmission | `Path_Loss_3D` | coherent **E** | `SceneV3` + `CrossingLUT` | local |
| Reflection | `Reflection_3D` | coherent **E** | `r_slab` (complex Airy), image sources | local (~3 s) |
| Diffraction | `Diffraction_3D` | coherent **E** | `utd_coefficient` (complex D) | local / GPU |
| Scattering | `Scattering_3D` | **p_incoh** | effective-roughness (Degli-Esposti) | GPU |
| Refraction | ⬜ | coherent **E** | `speed_field`, `slab_transmission_coherent` | local |
| Absorption | ⬜ | multiplier + absorbed-power density | `Im(q)`, `per_metre` | local |

> **Compute note (measured, not estimated).** Order-1 image sources on axis-aligned planes cost
> one distance field per plane — 8 planes over 262×17×132×2 bands takes **3.0 s on CPU**
> (MPS 4.4 s; transfer overhead dominates at this size). The A100 is still wanted for
> **Scattering_3D** (~20k lit patches × angular basis), the **Tx dataset sweep**, and order-2+
> reflections.

**Emergent, not modules:** multipath and fading (coherent sum over space), delay spread and
dispersion (that sum resolved in time), waveguiding (high-order reflection in confined geometry).
Doppler is dynamics; depolarization is a cross-cutting property of reflection and scattering.

### Where amplitude and phase come from

> **Amplitude ← `CrossingLUT`** (calibrated, decohered, correct magnitude).
> **Phase ← the eikonal `T` that `SceneV3.arrival_time` already solves** — it is the excess path
> length including in-wall slowdown and Fermat corner-wrap.

`E = 10**(-PL/20) · exp(-j·2πf·τ)`, so `−10·log10|E|² == PL_dB` exactly. Complex fields with **no
new solver**, and no Fabry-Pérot phase we cannot trust at 0.3 m raster thickness.

---

## 4. The four propagation modes

| # | Mode | Scene | Source | Mechanisms | State |
|---|---|---|---|---|:--:|
| 1 | **Vacuum** | all air, production shape | point | path loss | ✅ |
| 2 | **Indoor** | voxelized 7th floor | point | all four | ✅ |
| 3 | **O2I** | same floor | plane wave on the facade | facade + 3 | ✅ |
| 4 | **Outdoor** | `city/NoMa_DC_buildings` | point | path loss, reflection, diffraction | 🚧 grid not built |

Mode 3 is driven by a **known transmitter**: Forte Hall rooftop (38.901550, −77.011420) →
**415.9 m at arrival bearing 237.0°**, derived from the floor plan's own QGIS georeference rather
than the manifest's placeholder `bearing_deg = 135`. Outdoor-leg FSPL 80.7 dB (619 MHz) to
96.2 dB (3710 MHz). The tests assert the *derivation*, not the constant.

---

## 5. Milestones

### M0 — Unblock, re-voxelize, register ✅ COMPLETE

- ✅ Restore `Antenna_Type_3D.py` (was truncated to 17 lines; 22 antenna types recovered).
- ✅ This document.
- ✅ `tests3d/` harness — analytic scenes + physics invariants (§7).
- ✅ Re-voxelize from the 253-material mesh: seal the ceiling, reclass the floor slab to concrete,
  exclude ~40 people/prop materials.
- ✅ Fit `registration_3d.json`; widen `norm.freq_log` to `[600, 6200]` and add the cellular bands.

> **Why M0 gated everything.** Layer census of the ORIGINAL grid: **y=10 was 100 % air → no ceiling
> slab**; the floor slab was `drywall_partition` (ε′≈2.9) not concrete (ε′≈5.24); **concrete = 284
> voxels (0.08 %)**, **furniture = 73 (0.02 %)**. For a ceiling-mounted AP the floor and ceiling
> bounces are the two dominant specular paths — one had the wrong permittivity, the other did not
> exist. Six perfect solvers on that grid would have produced a confidently wrong answer.
>
> **After M0.4** (262×17×132): concrete **11.33 %**, furniture **2.17 %**, drywall 11.53 %, floor
> slab 100 % concrete, ceiling at y=9. All four scene gates are hard-passing tests.

### M1 — Complex-field spine ✅ COMPLETE

`contracts.py` · `Path_Loss_3D.py` · `Combine_3D.py` · `_bootstrap.py`.

**Gate PASSED:** `to_legacy_volumes(combine([path_loss])) == SceneV3.pathloss_maps(tx)` to
< 0.01 dB on free-space, slab, two-plate **and** the production scene — so the complex
architecture is a strict superset of the working engine, and the browser and dataset stages
needed no changes.

Verified: `−10log10 Σ|E|² == PL` · phase rotates with range · **coherent doubling = 6.02 dB vs
incoherent = 3.01 dB** · bandwidth averaging shrinks fringe dynamic range for two sources and is a
no-op for one · FSPL floor · a single path has exactly zero delay spread.

Two numerical fixes this found: power must accumulate in **float64** (deep-shadow voxels reach
540 dB, so |E|² ~1e-54 underflows float32 to zero and reports +inf), and the eikonal returns inf
inside masked barriers where the `CrossingLUT` still reports a finite level — so amplitude must
not be gated on eikonal reachability.

### M2 — Mechanisms + the demo 🚧

- ✅ **`Diffraction_3D.py`** — UTD wedge diffraction on the voxel grid; slice-wise edge finding with
  the wedge parameter measured from geometry (n = air-arc/π), Keller-cone angle β₀, **D kept complex**
  so the combiner can interfere. Per-edge relay cache: 7.5× faster per Tx, bit-exact.
- ✅ **`Reflection_3D.py`** — specular multipath by image sources. A voxel scene's faces are
  axis-aligned by construction, so mirroring and path length are exact and fully vectorized.
  Complex Airy-slab R added **coherently**, not power-summed. numpy + torch, parity ≤ 0.01 dB.
- ✅ **`Scattering_3D.py`** — diffuse scattering, effective-roughness / directive model. The only
  INCOHERENT mechanism. Energy is *split* not invented — specular keeps `sqrt(1−S²)·R`, diffuse
  takes `S²|R|²` — so conservation is structural and testable.
- ✅ **Mechanism channels + the browser demo** — `m_<mech>_<txid>.bin` (dB) and `tau_<mech>_<txid>.bin`
  (ns) beside the total volume; `loadMechanism` / `chanAt` / `runMechanismField` /
  `runMechanismTimeLapse`. Per-mechanism colour is **contribution share in dB**, not percent — a
  mechanism carrying 0.1 % of the power still has structure worth seeing, and a linear 0–100 % ramp
  collapsed it to 18 voxels out of 588 k.
- ⬜ `Refraction_3D.py`, `Absorption_3D.py` — both cheap and local. Until they land their two
  `viz3dMode` options stay disabled and labelled "module not built".
- ⬜ `Frontend/2d-3d/georef.js` — still missing (Pre-M4 §P8).

**Two arrival-time clocks, on purpose.** `path_loss` reports the eikonal τ (charged for in-wall
slowdown, ~+6 ns median); reflection, diffraction and scattering report vacuum path length. Mixing
them put the direct front LAST in 99 % of interior voxels, so `path_loss` also exports
`tau_geom_*.bin` (d/c) and the time-lapse runs every layer on that one convention. The **Wavefront
sweep** view keeps the true eikonal arrival. Charging the other three for in-wall slowdown is the
real fix and is a physics change, not an export change (Pre-M4 §P6).

### M3 — Four modes + offload/cache ✅ (outdoor grid pending)

- ✅ **`modes_3d.py`** — four modes, one solve path: modes differ only in which grid they build and
  where the source is, so that difference is data, not four forked scripts.
- ✅ **Vacuum is the invariant gate made visible.** The browser re-checks the exported volume against
  `20log₁₀(d) + 20log₁₀(f) − 27.55` and prints the worst deviation, so the gate is something a user
  can *see* rather than a line in a test log.
- ✅ **O2I geometry is derived, not guessed** — `forte_hall_geometry()`.
- ✅ **`facade_sources_3d` / `bs_field_3d`** — normals estimated for the whole grid at once (the
  per-voxel Python version was ~573k iterations; the vectorized one is 0.15 s). Opposite bearings
  light **disjoint** facades — the property that makes O2I directional at all.
- ✅ **`cache_index.py`** — key is `H(scene_sha, mode, tx_vox, bands, mechanisms, mech_bands,
  engine_ver)`, where `engine_ver` hashes the **physics source itself**, so editing `Reflection_3D.py`
  invalidates every affected volume with nobody remembering to bump a number.
- ✅ **The resume skip was silently wrong.** It asked "does `pl_volume_<tid>.bin` exist?", so changing
  `--mechanisms` left a stale volume in place and reported success. Now content-addressed.
- ✅ **Three-tier resolution, named on screen**; **neighbour prefetch** of the 3 nearest cached Tx.
- 🚧 Outdoor mode reports itself unavailable until `voxelize_city.py` is run.

**LRU evicts whole transmitters, never individual files.** Half a volume is worse than none: the
browser would fetch a 404 mid-render instead of falling through to the analytic tier.

**Two deliberate departures from the indoor path in O2I**, both because the transmitter is 416 m
outside the grid: the FSPL floor is disabled (flooring against an in-grid facade voxel would clamp
the map to a fiction), and facade contributions are summed as **power, not field** — they discretize
one wavefront, so a coherent sum would manufacture an interference pattern that is an artifact of
the sampling stride.

---

### Pre-M4 — Fix what would be baked into the surrogate ⬜ **NEXT**

M4 is scaffolded and smoke-tested, and it is **gated on purpose**. A surrogate is a compression of
whatever you train it on, so every one of these defects would be learned, exported to ONNX, and
served to the browser as physics. The two marked ⛔ are hard blockers; the rest are correctness and
hygiene that should ride along.

#### ⛔ P1 — Saturating obstruction model in `SceneV3.crossing_loss`

The target is `clip(PL, 40, 170) dB`. Measured on the production scene, averaged over 4 random Tx:

| Band | Interior voxels at the 170 dB ceiling | Median PL | Max PL |
|---:|---:|---:|---:|
| 619 MHz | **44.4 %** | 159 dB | 753 dB |
| 2442 MHz | **72.6 %** | 245 dB | — |
| 3500 MHz | **79.1 %** | 287 dB | 1313 dB |
| 6125 MHz | **87.1 %** | 387 dB | 1791 dB |

Most of the volume is a constant. A network trained on that learns a plateau and still reports a
flattering RMSE, because most of the error budget is spent on voxels where every model is right by
construction.

The cause is known and already solved in 2D. `engine_v2.effective_obstruction`
(`obs_eff = ceiling · tanh(solidity · obs / ceiling)`, solidity 0.35, ceiling 55 dB) exists because
straight-ray tracing over-counts wall runs — about 9 per cell on this floor plate. `VALIDATION_v2.2.md`
records the same failure in 2D (174–273 dB raw, 60–75 % clipped) and the same fix bringing medians to
77–131 dB with **0 % clip**. The 3-D port never took it.

*Do:* port `effective_obstruction` into `crossing_loss` with `obs_solidity` / `obs_ceiling_db` in
`manifest_3d.json`; re-fit against the 2-D calibrated values; re-solve the cached volumes.
*Done when:* `clip_report` worst-band clipped fraction < 35 % and `tests3d/test_dataset_3d.py::
test_clip_report_flags_the_saturated_target` is deliberately inverted.
*Guard today:* `phase_b3_dataset.ipynb` runs `dataset_3d.clip_report()` and **raises** before
generating a single sample.

#### ⛔ P2 — Wall loss on the diffuse outbound leg

`Scattering_3D._accumulate_*` charges `obs_tx` on the **inbound** (Tx→patch) leg only, so diffuse
power reaches the receiver through structure unattenuated. At `tx_66-5-54` / 2442 MHz the direct
field's median interior PL is **302 dB** while the diffuse channel's is **103 dB** — diffuse beats
direct in **94.6 %** of interior voxels. Combined with P1 (which inflates the direct path) the two
bugs partly mask each other in the total, and separating them is why the mechanism channels exist.

*Do:* charge `crossing_loss` on the patch→Rx leg too, or justify a cheaper approximation in the
model card. *Done when:* diffuse wins in a minority of interior voxels, and `Scattering_3D --test`
grows a case asserting it.
*Consequence for M4:* until P1 and P2 land, the surrogate may only be trained on **`path_loss`
alone** (which is what the shipped notebooks do). Training on the combined four-mechanism field —
the thing the browser actually caches — is Stage 2.

#### P3 — Stale dataset splits ✅ FIXED in this pass

`dataset/splits.json` claimed 1,500 transmitter positions sampled from `valid_tx_mask`. Only
**14 of 1,500** landed on a currently-valid voxel: the file predated the M0.4 re-voxelization and was
never regenerated. Worse, it was never generatable — `valid_tx_mask` holds 2,510 voxels and
min-spacing sampling saturates at **370** positions (173 at 3-voxel spacing).

The fix is to sample `inside_mask` (69,432 voxels → **1,592** positions at 3-voxel spacing).
`valid_tx_mask` exists to stop the placement UI dropping a physical AP inside a wall; a *training*
transmitter has no such constraint, and honouring it threw away 96 % of the domain. `splits.json`
now carries `scene_sha`, and both notebooks refuse to run against a mismatch.

#### P4 — `Construct_Reciever_3D.py` is 0 bytes

Blocks M5 entirely: there is no path from PL to RSRP/RSRQ/SINR/RSSI, so the scanner data cannot be
compared to anything. Cheap to write, and it should land before M5 rather than inside it.

#### P5 — The browser never runs the surrogate

`simulation3d.js:518` loads `pl_unet3d.json` and creates an `ort.InferenceSession`, and there is no
`session.run()` anywhere in the file. Tier 2 is a real slot with no occupant. M4 produces the model
and the contract; this wires the input builder (a JS mirror of `dataset_3d.dynamic_channels`), the
inference call, and the denormalization back to dB and ns.

*Done when:* with a trained model present the status line reads `DL surrogate` and a Tx more than
20 voxels from any cached solve renders from the network instead of the analytic mirror.

#### P6 — In-wall slowdown on non-direct legs

Reflection, diffraction and scattering time their paths at vacuum `c`. The `tau_geom` export makes
the time-lapse honest, but the physics fix is to charge those legs the eikonal lag. Until then the
mechanism time-lapse is a path-length comparison, not an arrival comparison, and the model card
says so.

#### P7 — Documentation drift ✅ FIXED in this pass

`MODEL_CARD_3D.md` still described the pre-M0.4 grid (`262 × 11 × 118`) and the four WiFi bands,
against an actual `262 × 17 × 132` and ten bands. Corrected. `BUILD_PROMPTS.md` carries the same
stale numbers and is a historical document — left alone, but do not treat it as current.

#### P8 — `georef.js` and the hardcoded floor frame

`viewer3d.js:621` hardcodes `FLOOR_W/FLOOR_H = 1150/515` and an inches-based elevation constant the
manifest explicitly supersedes, and it min-max rubber-sheets geographic CSVs onto the model bounds.
That is fine for a picture and fatal for validation: M5 needs measured lon/lat to land on the right
voxel. `registration_3d.json` and `floorplan_meta.json` already hold the transforms; what is missing
is the ~80 lines of `lonlatToPx` / `pxToLocalM` / `localMToVox` / `voxToWorld` that consume them.

#### P9 — Unkeyed cache entries

Five of six cached volumes predate content addressing (`key=None`). They are still served to the
browser, but a re-solve will not treat them as hits, so they will silently double until LRU evicts
them. `cache_index.py --prune-stale` plus a re-solve clears it.

#### P10 — Small correctness items

- `modes_3d.main()` has `if a.list or True:` — the `--list` flag does nothing and bare invocation
  always lists.
- `Diffraction_3D.select_edges(edges, tx, ...)` never uses `tx`; farthest-point sampling runs over
  already-picked edges rather than from the transmitter.
- `precompute_volumes.DEFAULT_MECHANISMS` omits `scattering` while `modes_3d`'s indoor stack
  includes it, so the two entry points disagree about what "indoor" means by default.
- No `test_mech_scattering.py`; scattering is covered only by its own selftest and the export tests.

---

### M4 — Dataset + DL surrogate ⬜ *(Colab; scaffolded, gated on Pre-M4 P1/P2)*

The deliverables are built, smoke-tested end to end, and waiting on the gate. Nothing here needs to
be designed; it needs to be **run**, on a Colab runtime, by a human with an hour or two.

#### What ships

**`dataset_3d.py`** — the single definition of the input tensor, imported by both notebooks and
mirrored into the browser contract.

| | |
|---|---|
| **Input** | 9 channels `(N, C, X, Y, Z)` float32: 6 material one-hot + Tx Gaussian blob (σ = 2 cells) + frequency feature + log₁₀(distance)/3 |
| **Output** | **2 channels**: `pl_norm = (PL−40)/130` and `tau_norm = τ_ns/400`, both clipped to [0,1] |
| **Sampling** | `inside_mask`, 3-voxel min spacing, split **by position** stratified over X–Z octants |
| **Shards** | `shard_NNN_pl.npy` + `shard_NNN_tau.npy` + `shard_NNN_meta.npz` |

Two output channels, not one, because the browser's wavefront sweep and mechanism time-lapse read a
`T` volume — a PL-only surrogate would accelerate the coverage view and nothing else. Carrying τ
also turns the causality constraint into a real loss term instead of an aspiration.

**`phase_b3_dataset.ipynb`** *(Colab CPU, ~1 h for 1,000 positions)*
1. **Preflight gate** — `clip_report()` measures target saturation and **raises** above 35 %. This is
   the Pre-M4 P1 blocker enforced in code.
2. Reuses `splits.json` after checking `scene_sha`; refuses on mismatch.
3. Solves `pathloss_maps` (one geometric pass, all bands) + one eikonal per position.
   τ is stored **per position, not per band** — P.2040 permittivity moves a few percent across
   600–6200 MHz, so a τ per band would be the same array six times.
4. Writes through `.part` names and renames last, so a disconnected runtime never leaves a partial
   shard that a resumed run would trust.
5. Reads one sample back **through the memmap path Phase C uses** and asserts it reproduces a fresh
   solve to < 0.2 dB (measured: **0.03 dB**, the fp16 quantization floor).

**`phase_c3_train_colab.ipynb`** *(Colab GPU, ~2 h for 30 epochs)*

- **3-level UNet3D**, anisotropic pooling `(2,1,2)` — the vertical axis is 17 voxels for one 2.85 m
  storey, so halving it three times would leave two. Skips are size-matched with `interpolate`
  because 262 and 132 do not divide evenly. Sigmoid head, since both targets are already [0,1].
  `base=24` → ~3.3 M parameters.
- **Memmap, never load.** The shards are ~8 GB and Colab has 12.7 GB. Volumes are plain `.npy` so
  `mmap_mode='r'` reaches them; the fp16→fp32 cast happens in `__getitem__`. *(The old notebook cast
  at load and needed 8.2 GB before training started.)*
- **Only the 3 dynamic channels move through the DataLoader.** The material one-hot is constant, so
  shipping it per sample would push 14 MB of redundancy per item; it is concatenated once per batch
  on the GPU.
- **Physics-constraint losses from day one**, all computed from channels the network already sees:

  | Constraint | Form | Weight |
  |---|---|---|
  | FSPL floor | `relu(fspl_norm − pl)` | 0.10 |
  | Causality | `relu(d/c − τ)` | 0.10 |
  | Band ordering | `relu(PL_lo − PL_hi)` at one Tx | 0.05 |
  | Reciprocity | `abs(PL_p(q) − PL_q(p))` for two Tx in a batch | 0.05 |

  A **paired batch sampler** draws `BATCH_POS` positions and gives each the *same* two bands, so
  every comparison differs in exactly one variable. **Energy shells are deliberately absent**: the
  invariant needs a closed surface, and on a bounded floor plate every shell past a few metres is
  clipped by the facade, so the residual would measure the truncation. It stays a `tests3d` check on
  the engine, where the synthetic scenes are unbounded.
- **Survivable on Colab.** Full-state checkpoints to Drive every 2 epochs
  (`{net, opt, scaler, epoch, best, history, gstep}`), resume by default, AMP (bf16 on A100), early
  stop, and a `MAX_HOURS` budget that checkpoints and exits cleanly before Colab kills the runtime.
  The LR schedule is a **pure function of the global step**, not a stateful scheduler: a resumed
  epoch replays steps, and `OneCycleLR` raises the moment the replayed total passes `total_steps`.
  *(Found by running the resume path twice, not by reasoning about it.)*
- **Two export gates.** ONNX parity vs PyTorch ≤ **0.1 dB** worst-case, and a size check against the
  parameter count. The 2-D gate hardcoded `> 50 MB`, which is a fact about that model rather than a
  property of a healthy export; the check that generalizes is whether the weights reached the file.
- Writes `web/pl_unet3d.onnx` **and** `web/pl_unet3d.json` from
  `dataset_3d.surrogate_contract()` — the same module that built the training inputs. The browser
  reads the channel order from the trainer instead of guessing it, which is exactly why
  `simulation3d.js` refuses to load a model without the sidecar.

#### Verified locally before handoff

Both notebooks were executed end to end at smoke scale on CPU (16 positions × 6 bands): dataset
generation → training → evaluation → ONNX export at **0.0000 dB parity** → contract written. The
resume path was exercised by interrupting on the time budget and re-running. Four real bugs were
found and fixed this way — `np.save` appending `.npy` to temp names, the paired sampler dropping a
short split, `OneCycleLR` overrunning on resume, and the default ONNX exporter needing `onnxscript`
on torch ≥ 2.9.

#### Targets and honesty

**RMSE ≤ 5 dB** against the simulator, reported beside FSPL and log-distance baselines — a surrogate
number without them is not interpretable. The 2-D run achieved 4.68 dB against an FSPL baseline of
72.9 dB, and the gap is the claim. This is **sim-vs-sim** and is not the validation number; M5
measures sim-vs-measurement, where the bar is 8 dB.

#### Staged, because of Pre-M4

| Stage | Target | Gate |
|---|---|---|
| **1** *(shipped)* | `path_loss` + eikonal τ | needs P1 only |
| **2** | combined four-mechanism field — what the browser actually caches | needs P1 **and** P2, or the surrogate faithfully reproduces diffuse-beats-direct in 94.6 % of voxels |
| 🔬 **3** | per-mechanism complex surrogates predicting `(log\|E\|, cos φ, sin φ)` | off the critical path |

Stage 3 decouples smooth magnitude from high-wavenumber phase and confines the spectral-bias problem
to two channels; FNO-vs-UNet ablation on diffraction only; residual head; differentiable calibration
(`set_material_scale`, mirroring `engine_v2_torch.py:249`).

---

### M5 — Validation against the scanner data ⬜

`validate_scanner_3d.py` + `Construct_Reciever_3D.py` (Pre-M4 §P4) + `georef.js` (§P8). All three
are prerequisites, not parts.

- **Known-Tx O2I case (the strong test):** Forte Hall → building, 415.9 m, arrival 237.0°. Resolve
  the antenna height from OSM/OpenBuildingMap for 38.90155/−77.01142 via a small
  `osm_building_height.py`; fall back to ~3 storeys ≈ 10–12 m. With Tx position **and** indoor
  geometry known, the outdoor leg is computable and per-wall behaviour becomes testable.
- **The truth sets.** `records_data.js` / `timeseries_data.js` hold **2,625 NR Top-N samples** over a
  164-minute walk on bands n2/n25/n41/n71 only, because the ETL filters `*nr Top N Signal*.CSV`.
  All 71 LTE files and both n77/n78 blind scans are excluded — n78 is the one band overlapping the
  sim's original WiFi list, so re-exporting to include LTE and n78 is what makes validation possible
  at scale.
- Aggregate ≥ 3 samples/cell (median) to suppress fast fading; per-donor fixed effects for the
  unknown donors; **`Ref Signal - Delay Spread` (5,951 unused samples)** validates the *combiner*,
  not just path loss.

---

### M6 — Stress testing: frontend and processing limits ⬜

Every number in this plan was measured on one scene, one browser and one machine. M6 finds where
each of them stops being true, and writes the ceilings down so the next person does not discover
them in a demo. Deliverables: `SIM V1 3D/bench3d.py`, `tests3d/test_stress.py` (marked `slow`), a
browser harness driven through `window.__sim3d`, and `STRESS_REPORT.md` with the measured curves.

#### M6.1 — Solver scaling

Sweep grid size, band count, mechanism stack and Tx count; record wall time, peak RSS and the
memory high-water inside `crossing_loss` (whose `MEM_CAP = 3e6` sample cap is a constant nobody has
re-derived since the grid grew from 262×11×118 to 262×17×132).

| Question | Why it bites |
|---|---|
| Time and memory vs grid voxels | outdoor city grids are ~350 MB before any solve |
| Cost vs band count | `crossing_loss` does one geometric pass for all bands — confirm it is really flat |
| Diffraction relay-cache build | ~74 s once, ~376 MB on disk; what happens at 2× edges |
| Scattering patch count | O(patches × Rx); `max_patches=20000` is a guess |
| O2I facade sources | 48 sources ≈ 2 min; the cost is linear and the default was chosen by eye |
| `MEM_CAP` under a bigger grid | the failure mode is an OOM kill, not a slow solve |

*Gates:* no configuration in the shipped modes exceeds 16 GB RSS; every solve either completes or
fails with a diagnosis naming the knob.

#### M6.2 — Cache and disk pressure

`cache_index.py` owns eviction and integrity, and neither has been tested near its limits.

- LRU eviction under a budget deliberately smaller than the working set — assert **whole
  transmitters** leave, never individual files (a half volume 404s mid-render instead of falling
  through to analytic).
- Truncated and corrupted `.bin` files: does `--verify` catch every one by arithmetic?
- Concurrent writers. `cache_index` is documented as single-writer; prove that two `precompute`
  processes cannot interleave into a broken `index.json`, or make it fail loudly.
- `index.json` growth at 1,000+ transmitters, and the browser's parse time for it.
- Cold start with a partially-synced Drive/`web/volumes` directory.

*Gates:* `--verify` clean after every eviction; no partial transmitter ever reachable from
`index.json`; catalog parse < 100 ms at 1,000 entries.

#### M6.3 — Browser rendering limits

The 3-D view draws instanced cubes, one per rendered voxel, on a WebGPU backend.

- Frame time vs instance count; find the knee and set the far-field cull from it rather than from
  the current hand-tuned threshold.
- Volume cache memory ceiling in the tab: 6 transmitters is 118.8 MB of float16 today; measure the
  point at which the LRU budget must engage and confirm it does before the tab is killed.
- Mechanism time-lapse with all four layers live — four instanced meshes on one clock.
- WebGPU vs WASM fallback, and integrated-GPU / low-VRAM machines.
- Rapid Tx scrubbing: does neighbour prefetch help or thrash?
- Long-session stability — leaked meshes and buffers across dozens of mode and visualization
  switches.
- Failure paths, all of which currently exist and none of which are tested: `file://`, a 404 volume,
  a truncated `.bin`, a mode with nothing cached, a surrogate whose contract does not match the
  scene.

*Gates:* ≥ 30 fps at the default visualization on a mid-range laptop GPU; tab RSS bounded across a
30-minute session; every failure path produces a status line naming the cause and the fix, and never
a silently wrong picture.

#### M6.4 — Surrogate inference limits *(after Pre-M4 §P5)*

- onnxruntime-web session memory and first-inference latency, WebGPU vs WASM.
- Inference latency vs the ~27 s full solve and vs the cached-volume fetch — the surrogate is only
  worth loading if it sits between them.
- Numerical agreement browser-vs-Python on the same input, which is the last place a channel-order
  mistake can hide.

*Gates:* inference < 2 s on the reference machine; browser-vs-Python worst deviation ≤ 0.5 dB.

#### M6.5 — Data-scale limits

- The dashboard at 10× the current 2,625 samples (the raw concat has 23,017 rows across 182 columns).
- Plotly scattergl and the IDW gradient at that size.
- `records_data.js` / `timeseries_data.js` as ~1 MB of parsed-at-load JS — the point at which they
  should become fetched JSON.

---

## 6. Data flow (arrays)

```
.obj ──voxelize.py──▶ material_grid.npy ──┐
                                          ├──▶ SceneV3 ──▶ per-mechanism FieldGrid (complex E, τ)
       manifest_3d.json ──────────────────┘                        │
                                                                   ▼
                                                       Combine_3D.combine()   ◀── bandwidth_hz
                                                                   │             (8 sub-carriers;
                                          ┌────────────────────────┤              a single tone shows
                                          │                        │              nulls no real Rx sees)
                            to_legacy_volumes()            to_cir() 🔬
                                          │
                    ┌─────────────────────┼──────────────────────┐
                    ▼                     ▼                      ▼
        web/volumes/*.bin        dataset/shard_*        validate_scanner_3d.py ⬜
         (browser cache)          (surrogate training)     (vs G-flex data)
```

---

## 7. Test harness — `tests3d/`

```
tests3d/
├── conftest.py              path bootstrap, seeded fixtures
├── synth3d.py               free space · slab · two-plate · corner · corridor · rough wall
├── invariants.py            reciprocity · energy · causality · FSPL floor · monotone LOS ·
│                            band ordering · |E|² == PL · shadow-boundary continuity
├── test_scene_sanity.py     ceiling sealed · floor concrete · class proportions
├── test_engine_3d_core.py   FSPL exact in vacuum · reciprocity · T ≥ d/c
├── test_mech_*.py           reflection · diffraction  (scattering: Pre-M4 §P10)
├── test_combiner.py         the superset proof
├── test_modes_3d.py         registry · vacuum gate · O2I derivation · facade directionality
├── test_cache_index.py      key stability · put/verify/gc/migrate
├── test_export_channels.py  browser index keys · float16 layout · both clocks
├── test_dataset_3d.py       surrogate featurization contract · sampling budget · shard memmap
└── run_all.py               --fast | --full | --gpu | --selftest
```

Each mechanism also keeps a dependency-free self-test: `python Reflection_3D.py --test`, and so does
`dataset_3d.py`.

---

## 8. Validation gates

| Gate | Criterion | State |
|---|---|:--:|
| **V0** physics self-consistency | invariants green · energy closes < 1 % · reciprocity < 0.5 dB · causality exact · FSPL floor violated < 0.1 % | ✅ |
| **M1** superset proof | combine(path-loss only) == `SceneV3.pathloss_maps` < 0.01 dB | ✅ |
| **Pre-M4** target sanity | worst-band clipped fraction < 35 % | ⛔ 87 % |
| **M4** surrogate | held-out RMSE ≤ 5 dB vs the simulator · ONNX parity ≤ 0.1 dB | ⬜ |
| **V1** spatial structure | Spearman ρ(sim PL, −RSRP) ≥ 0.6 on held-out cells | ⬜ |
| **V2** anchored level | held-out RMSE ≤ **8 dB** after per-donor constants | ⬜ |
| **V3** delay structure | simulated RMS delay spread within 2× of measured | ⬜ |
| **V4** per-material calibration | Forte Hall known-Tx case only | ⬜ |
| **M6** stress | see §5 M6 sub-gates | ⬜ |

> **Why 8 dB and not 3 dB.** Indoor measurement scatter with an unknown or partially-known
> transmitter is 6–10 dB. The 2-D surrogate's 4.68 dB RMSE is *sim-vs-sim* and is not comparable to
> a sim-vs-measurement number. Claiming better would be claiming to beat the experiment's own noise
> floor.

---

## 9. Known limits

1. **WiFi 5.5 / 6.1 GHz cannot be validated** — no measurements exist at those bands. This is why
   cellular bands were added early: they are what make validation possible.
2. **Donors other than Forte Hall are at unknown sites** → shape-only (per-donor fixed-effect)
   validation; absolute level and per-wall values are not identifiable from them.
3. **Full-wave 3D ground truth is impossible at this scale** — at 3.5 GHz, λ = 8.6 cm ⇒ h ≤ 8.6 mm,
   so a 256³ grid spans ~2.2 m. The residual head is limited to 2D slices and scaled-frequency crops;
   2D→3D transfer is unproven and labelled as such.
4. **Scene material fidelity is the binding constraint**, not solver quality — hence M0 first, and
   hence Pre-M4 before M4.
5. **The surrogate can only ever be as good as the field it is trained on.** Stage 1 learns the
   direct field, which is not what the browser caches. That is a deliberate scope limit, not an
   oversight, and the model card says so.

---

## 10. References

- Seybold, *Introduction to RF Propagation* — the physics; per-chapter notes and verified equation
  implementations live in [`RF Propagation Theory/`](../../../RF%20Propagation%20Theory/).
- **ITU-R P.2040-3** — building material electrical properties (εr, σ vs frequency).
- **ITU-R P.1238** — indoor path loss coefficients N, floor-penetration Lf, delay spread.
- **3GPP TR 38.901** — O2I building-penetration classes (15 / 28 dB).
- Degli-Esposti et al. — effective-roughness diffuse scattering.
- NVIDIA **Sionna RT** — differentiable ray tracing (reference for the differentiable-core branch).
- **RadioUNet** / FNO / DeepONet / PINNs — surrogate prior art to benchmark against.
