# SIM V1 3D — Model Card

**What it is.** A volumetric indoor RF path-loss model for the 7th floor:
`PL(x, y, z)` computed through a voxelized version of the OBJ, plus a UNet
surrogate that learns `(Tx, frequency) → PL` volume over the **fixed** geometry.
This is the 2-D skill spec's deferred *volumetric* fork, scoped pragmatically for
a first version.

## Domain & inputs (`manifest_3d.json`)

| | |
|---|---|
| Grid | `262 × 17 × 132` voxels (X, Y-up, Z), cubic **0.30 m** — 587,928 total (post-M0.4 re-voxelization) |
| Scale | `m_per_unit ≈ 0.018377`, from registering the OBJ X-span to the floor-plan width (1150 px × 0.0679 m/px, QGIS GCP calibration in `SIM/manifest.json`) |
| Interior | façade-footprint fill; room band iy 3–8; **69,432 voxels** (11.8 % of the grid), `valid_tx` 2,510 |
| Materials | 6 classes — air 72.73 %, drywall 11.53 %, concrete 11.33 %, core 0.93 %, furniture 2.17 %, glass 1.32 % |
| Bands | 10: WiFi 2442 / 5500 / 6125 and the measured cellular set 619 / 627 / 1935 / 2510 / 2600 / 3500 / 3710 MHz |
| Norm | PL → `(PL − 40)/130` clamped to [0,1]; `freq_log` window `[600, 6200]` MHz |

## Physics (`engine_3d.py` `SceneV3`)

Two co-registered volumes per transmitter, in true (x, y, z).

**Path loss** — along the Tx→voxel ray, summed per wall crossing:

```
PL = fspl_1m(f) + 10·n·log10(max(d, d0))              # n = 2 (Motley-Keenan spread), d0 = 1 m
   + Σ CrossingLUT( material, incidence θ, thickness ) # per-crossing Fresnel/Airy
                                                       # transmission + Im(q) absorption
```

The per-crossing term is the **validated `physics_v2` EM kernel** (`CrossingLUT` —
angle- and thickness-dependent Fresnel/Airy slab transmission plus bulk absorption),
reused verbatim through `physics_3d.py`; it is **not** the flat per-class dB table.
Incidence angle comes from voxel wall normals; thickness from the along-ray run
length scaled to construction thickness.

**Arrival time** — a real 3-D **eikonal** solve (`skfmm.travel_time`, `|∇T| = 1/speed`)
on the slowness field `Re(√ε_r)/c` with opaque classes masked. Refraction (bending,
in-wall lag) and diffraction (corner wraparound) fall out of the single fast-march via
Fermat — **not** geometric `T = d/c` (that survives only as `geometric_time()`, the
lower-bound sanity check).

The flat `materials[].loss_db` table and `physics.fallback_obstruction_db` in
`manifest_3d.json` are the **browser display / in-browser analytic fallback** model
(`Frontend/simulator/simulation3d.js`, `landing.js` preprocess), not this engine.

## Material mapping (OBJ → class)

Tunable in `manifest_3d.json` (`obj_material_map`). Owner should sanity-check:

| OBJ material | → class | note |
|---|---|---|
| `Glass_Basic_01` | exterior_glass (3 dB) | façade + interior glass |
| `FrontColor` | drywall (4 dB) | **bulk** incl. the floor slab (iy 1–2) & partitions |
| `Steel_Brushed_Stainless` | core (20 dB) | stand-in for metal (no 30 dB class in the 6-class scheme) |
| `Blacktop / Formica` | concrete (15 dB) | hard surfaces |
| `Ty_*` (incl. `Ty_Skin`) | furniture (0.3 dB/m) | furniture / people clutter |

## Fidelity ladder

Effect numbering follows the v1 spatial catalog (S1–S12) used across this project.
Two of the mechanisms below carry **no** S-number on purpose: that catalog describes a
single-first-arrival model and lists specular multipath and the diffuse tail as v2 forks.
This engine implements them anyway — which is what makes the browser's mechanism
time-lapse possible at all ("reflected fronts as separate sweeps" is literally the fork).

**Shipped (this engine):**

- **Geometric spreading (S1) + multi-wall log-distance (S2)** — `Path_Loss_3D` wrapping
  `SceneV3.pathloss_maps`: `fspl_1m(f) + 10·n·log₁₀(d)` with the Motley-Keenan `n`, `d₀`
  conventions.
- **Per-material Fresnel/slab (S3–S7)** — angle+thickness `CrossingLUT`, replacing
  the old flat per-class dB.
- **3-D UTD diffraction (S8)** — `Diffraction_3D`, Kouyoumjian-Pathak `D` kept **complex**
  so the combiner can interfere; slice-wise edge finding with the wedge parameter measured
  from geometry. A per-edge relay cache makes the second leg Tx-independent.
- **Clutter / Beer-Lambert (S10)** — `per_metre` material classes charged by path length
  inside `crossing_loss`.
- **Specular reflection & multipath** *(no S-number — v2 fork)* — `Reflection_3D`,
  order-1 image sources with complex Airy-slab `R`, summed **coherently**.
- **Diffuse scattering** *(no S-number — v2 fork)* — `Scattering_3D`, effective-roughness
  / directive model. The only INCOHERENT mechanism; supplies the delay-spread tail.
- **Eikonal arrival time (T3c)** — 3-D fast-march routes the front around barriers
  and bends/lags it through dielectrics, instead of punching straight through.
- **Physics-exact combining** — `Combine_3D` sums coherent `E` and incoherent `p_incoh`
  separately, with a bandwidth-averaging knob so single-tone nulls no receiver sees are
  not rendered as coverage holes.

**Floor FAF (S9) does not apply here.** It is a 2-D stand-in for slabs the model cannot
see; in a 3-D voxel scene the floor and ceiling are real crossings already charged by
S3–S7. Keeping FAF as well would double-count.

**Still deferred:**

- **Refraction & absorption as their own modules** — `Refraction_3D.py`,
  `Absorption_3D.py` are still 0-byte. Both effects are *present* inside the current
  stack (in-wall slowdown via the eikonal speed field; `Im(q)` bulk absorption inside the
  `CrossingLUT`); what is missing is exposing them as separate viewable channels.
- **Order-2+ image sources** — `Reflection_3D.solve` raises on `order > 1`. CS8's bounce
  budget is therefore `N_max = 1`.
- **Multi-arrival PDP (T6)** — `contracts.PathSet` exists but nothing populates it; the
  export is first-arrival per mechanism.
- **Correlated shadow field (S11)** — spatially-correlated large-scale fading.
- **In-wall slowdown on non-direct legs** — reflection, diffraction and scattering time
  their paths at vacuum `c`; only the direct field is charged the eikonal lag. See
  `precompute_volumes.write_mechanism_channels` ("TWO CLOCKS").
- **Wall loss on the diffuse outbound leg** — `Scattering_3D` charges `obs_tx` on the
  inbound leg only, so diffuse power passes through structure unattenuated. On the
  production scene this makes the diffuse channel dominate coverage (median interior PL
  103 dB vs the direct field's 302 dB; diffuse wins in 94.6% of voxels).
- **Saturating obstruction model** — `SceneV3.crossing_loss` sums every wall crossing along
  a straight ray, and straight-ray tracing over-counts (~9 wall runs per cell on this floor
  plate). The 2-D engine corrects this with `engine_v2.effective_obstruction`
  (`ceiling·tanh(solidity·obs/ceiling)`, solidity 0.35, ceiling 55 dB); the 3-D port does
  not. Direct path loss consequently reaches **1,791 dB**, and **79 % of interior voxels at
  3500 MHz (87 % at 6125 MHz)** exceed the 170 dB normalization ceiling. This is the binding
  defect for the surrogate — see `PLAN_3D_SIM.md` Pre-M4 §P1.
- **Floor/ceiling material split** — `FrontColor` lumps the floor slab with walls
  as drywall; splitting horizontal (concrete slab) from vertical (partition)
  faces by normal would be more accurate for near-vertical rays.
- **Generalization** — v1 is single-building, fixed geometry (= "fixed boundary
  conditions"); the surrogate conditions on Tx+freq only.

## Known limitations

- **Ray "spokes."** Discrete per-voxel ray sampling produces faint radial streaks
  (same as the 2-D engine). Physical (wall shadows), and the surrogate smooths
  them; not a bug.
- **Interior mask is a heuristic** (façade-footprint fill); it drives display and
  Tx sampling only, not the PL values (PL is computed for every voxel).
- **No trained surrogate yet.** `phase_b3_dataset.ipynb` and `phase_c3_train_colab.ipynb`
  are built and smoke-tested end to end (dataset → train → ONNX at 0.0000 dB parity), but
  no full run has happened, so `web/pl_unet3d.onnx` does not exist and the browser's
  surrogate tier reports its own absence. Treat the ≤ 5 dB RMSE target as unmet until the
  first real run. Phase B additionally **refuses to generate** while the saturating-obstruction
  defect above is unfixed; that gate is intentional.
- **Vertical axis.** One ~2.85 m storey is ~6 room voxels tall at 0.30 m (room band iy 3–8),
  so vertical detail is coarse — honest for a single floor. Lower `--cell` for more.

## Validation (`run_one_calc.py`, 3500 MHz, interior-centroid ceiling Tx)

- PL over interior: **43.3 / 146.1 / 540.0 dB** (min/median/max). The high tail is
  the uncapped Fresnel/Airy loss straight through the metal core — physical, and
  power-filled once diffraction (deferred) is added.
- Excess over free space ≥ 0 everywhere (walls only add loss). ✓
- Eikonal `T`: max **144 ns**, and `T − d/c ≥ 0` within grid tolerance (the front
  never beats the straight line). ✓
- `corr(PL, log-distance) = 0.57` (loss rises with range). ✓
- Visual: radial gradient, wall shadowing, glass-façade leakage, vertical
  variation (`preview/pl_one_calc.png`).

## Provenance

Source OBJ: `Data/models/Indoor 7th floor v2 First Render.obj/…`. Scale
calibration inherited from `SIM/manifest.json`. Physics constants and the
6-class material scheme mirror the deployed 2-D model.
