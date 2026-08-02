# 3D RF Propagation Simulator — Plan & Roadmap

Physics-first 3D RF simulator: six EM mechanisms from *Introduction to RF Propagation* (Seybold),
rendered in the browser with a time-lapse of the propagation mechanism, accelerated by a deep-learning
surrogate trained on physics-generated data, and cross-validated against PCTEL G-flex scanner data.

**Status legend:** ✅ done · 🚧 in progress · ⬜ planned · 🔬 research branch · ⛔ blocked

---

## 0. Design principles

1. **Physics first, ML second.** The deterministic engine is the source of truth and the permanent
   fallback. The surrogate is an *accelerator*, never a dependency — if it is missing or unavailable,
   the simulator still runs correctly, just slower.
2. **Reuse over rebuild.** `physics_v2.py` (2D) already implements ITU-R P.2040-3 materials, Fresnel
   coefficients, coherent/incoherent slab transmission and full Kouyoumjian–Pathak UTD. `SceneV3`
   already solves 3D path loss and a true eikonal arrival time. We wrap these; we do not re-derive them.
3. **Complex fields, physics-exact combining.** Each mechanism emits a complex field (amplitude +
   phase). The combiner is *not* learned — coherent sum for correlated/specular paths, incoherent
   sum for the diffuse tail. Summing powers instead of fields destroys the interference structure
   that makes a map read as measured rather than rendered.
4. **Offload + cache.** Heavy solves run on external GPU; results are cached in the browser's volume
   format so interaction stays real-time, visually and analytically.
5. **Honesty about limits.** Where the data cannot support a claim, the report says so (see §7).

---

## 1. The six mechanisms

These are the complete set of first-order EM–matter interactions; everything else is emergent.

| Mechanism | Module | Emits | Physics source | Compute |
|---|---|---|---|---|
| Spreading / transmission | `Path_Loss_3D.py` | coherent **E** | `SceneV3` + `CrossingLUT` | local |
| Reflection | `Reflection_3D.py` | coherent **E** | `r_slab` (complex Airy), image sources | local (~3 s) |
| Refraction | `Refraction_3D.py` | coherent **E** | `speed_field`, eikonal, `slab_transmission_coherent` | local |
| Diffraction | `Diffraction_3D.py` | coherent **E** | `utd_coefficient` (complex D) | local / GPU |
| Scattering | `Scattering_3D.py` | **`p_incoh`** | effective-roughness (Degli-Esposti) | GPU |
| Absorption | `Absorption_3D.py` | multiplier + absorbed-power density | `Im(q)`, `per_metre` | local |

> **Compute note (measured, not estimated).** Order-1 image sources on axis-aligned planes cost
> one distance field per plane — 8 planes over 262×17×132×2 bands takes **3.0 s on CPU** (MPS 4.4 s;
> transfer overhead dominates at this size). The earlier "first true GPU solver" estimate assumed
> general plane clustering with order-2 bounces. The A100 is still wanted for **Scattering_3D**
> (~20k lit patches × angular basis), the **1,500-Tx dataset sweep**, and order-2+ reflections.

**Emergent, not modules:** multipath & fading (coherent sum over space), delay spread & dispersion
(that sum resolved in time), waveguiding/ducting (high-order reflection in confined geometry — watch
the bounce-order cap or corridors read as unrealistically lossy). Doppler is dynamics; depolarization
is a cross-cutting property of reflection and scattering.

### Where amplitude and phase come from

> **Amplitude ← `CrossingLUT`** (calibrated, decohered, correct magnitude).
> **Phase ← the eikonal `T` that `SceneV3.arrival_time` already solves** — it is the excess path
> length including in-wall slowdown and Fermat corner-wrap.

`E = 10**(-PL/20) · exp(-j·2πf·τ)`, so `−10·log10|E|² == PL_dB` exactly. This gives complex fields
with **no new solver** and avoids Fabry-Pérot phase we cannot trust at 0.3 m raster thickness.

---

## 2. The four propagation modes

| # | Mode | Scene | Purpose |
|---|---|---|---|
| 1 | **Vacuum / free space** | `free_space_scene` | Physics self-test made visible: FSPL exact, energy closes, reciprocity, causality |
| 2 | **Indoor** | fixed voxel scene, Tx inside | The primary use case (WiFi + cellular) |
| 3 | **Indoor + Outdoor (O2I)** | façade sources | Signals entering from outside — the real scanner scenario |
| 4 | **Outdoor** | `voxelize_city.py` | Street-canyon / city scale |

Mode 3 is driven by a **known transmitter**: Forte Hall rooftop (38.901550, −77.011420) →
**415 m at bearing 57°, arriving at the building from 237°**. FSPL 80.8–96.2 dB across the measured
cellular bands. *(The manifest's `bs_preset.bearing_deg = 135` was a guess and is superseded.)*

---

## 3. Milestones

### M0 — Unblock, re-voxelize, register  ✅ COMPLETE
- ✅ **M0.1** Restore `Antenna_Type_3D.py` (was truncated to 17 lines; 22 antenna types recovered,
  3 importers unblocked, original authorship header preserved).
- ✅ **M0.2** This document.
- ✅ **M0.3** `tests3d/` harness — analytic scenes + physics invariants (see §5).
- ✅ **M0.4** Re-voxelize from `Sandbox_Version_3D_Simulation_1.obj` (**253 named materials** vs the
  current 18): seal the ceiling, reclass the floor slab to concrete, exclude ~40 people/prop materials.
- ✅ **M0.5** Fit `registration_3d.json` (px → local m → voxel); widen `norm.freq_log` to
  `[600, 6200]` and add the measured cellular bands.

> **Why M0 gated everything (now resolved).** Layer census of the ORIGINAL `material_grid.npy`:
> **y=10 is 100 % air → no ceiling slab**; the floor slab is `drywall_partition` (ε′≈2.9) not
> concrete (ε′≈5.24); **concrete = 284 voxels (0.08 %)**, **furniture = 73 (0.02 %)**, drywall = 23 %.
> For a ceiling-mounted AP the floor and ceiling bounces are the two dominant specular paths — one has
> the wrong permittivity, the other does not exist. Six perfect solvers on this grid would produce a
> confidently wrong answer. **Material fidelity, not solver quality, is the binding error term.**
>
> **After M0.4** (262×17×132 from the 253-material mesh): concrete **11.33 %**, furniture
> **2.17 %**, drywall 11.53 %, floor slab 100 % concrete, ceiling at y=9, `valid_tx` 2,510 of
> 69,432 interior voxels. All four scene gates are now hard-passing tests.

### M1 — Complex-field spine  ✅ COMPLETE
`contracts.py` (`FieldGrid`, `CombinedField`, `PathSet`) · `Path_Loss_3D.py` (first of the six
0-byte stubs filled) · `Combine_3D.py` · `_bootstrap.py`.

**Gate PASSED:** `to_legacy_volumes(combine([path_loss])) == SceneV3.pathloss_maps(tx)` to
< 0.01 dB on free-space, slab, two-plate AND the production scene — so the complex architecture
is a strict superset of the working engine, and the browser/dataset stages need no changes.

Verified behaviours: `−10log10 Σ|E|² == PL` · phase carried (rotates with range) ·
**coherent doubling = 6.02 dB vs incoherent = 3.01 dB** (the distinction the design rests on) ·
bandwidth averaging is a no-op for one path but shrinks fringe dynamic range for two sources ·
FSPL floor · single path has exactly zero delay spread.

Two numerical fixes this found: power must accumulate in **float64** (deep-shadow voxels reach
540 dB, so |E|² ~1e-54 underflows float32 to zero and reports +inf), and the eikonal returns inf
inside masked barriers where the CrossingLUT still reports a finite level — so amplitude must not
be gated on eikonal reachability; phase falls back to d/c there.

### M2 — Mechanisms + the demo  🚧
- ✅ **`Diffraction_3D.py`** — UTD wedge diffraction on the voxel grid. Physics reused
  verbatim from the 2D Kouyoumjian-Pathak implementation; new in 3-D is slice-wise edge
  finding with the wedge parameter measured from geometry (n = air-arc/π), the Keller-cone
  angle β₀, and keeping **D complex** so the combiner can interfere.
- ✅ **`Reflection_3D.py`** — specular multipath by image sources. A voxel scene's faces are
  axis-aligned by construction, so mirroring and path length are exact and fully vectorized:
  one distance field per plane. Complex Airy-slab R (`physics_3d.r_slab`) added COHERENTLY,
  not power-summed as P7 specifies — power-summing destroys the standing-wave structure
  reflection exists to create. numpy + torch backends (CUDA/MPS), parity ≤ 0.01 dB.
  **Runs in ~3 s on the full scene — no GPU needed** (see compute note below).
- ✅ **`Scattering_3D.py`** — diffuse scattering, effective-roughness / directive model
  (Degli-Esposti). The only fundamentally INCOHERENT mechanism: fills `p_incoh`, leaves `E`
  zero, and the combiner adds it as power. Energy is *split* not invented — specular keeps
  `sqrt(1−S²)·R`, diffuse takes `S²|R|²` — so conservation is structural and testable.
  Supplies the delay-spread tail that the scanner's unused `Ref Signal - Delay Spread`
  column (5,951 samples) can validate. numpy + torch backends. **This is the mechanism that
  genuinely wants the A100** (O(n_patches × n_rx)).
- ✅ **mechanism channels + the browser demo** — `export_pl_volume.py --mechanisms` and
  `precompute_volumes.py --mech-channels` write `m_<mech>_<txid>.bin` (dB) and
  `tau_<mech>_<txid>.bin` (ns) beside the total volume; `simulation3d.js` gained
  `loadMechanism`/`chanAt`/`runMechanismField`/`runMechanismTimeLapse`; the four implemented
  mechanisms are live in `viz3dMode` plus a **mechanism time-lapse** entry.
  Per-mechanism colour is the **contribution share in dB** against the total, not percent —
  a mechanism carrying 0.1 % of the power still has structure worth seeing, and a linear
  0–100 % ramp collapsed it to 18 voxels out of 588 k on the production scene.
- ⬜ `Refraction_3D.py`, `Absorption_3D.py` (both cheap, local) — until these land their two
  `viz3dMode` options stay disabled and honestly labelled "module not built".

The time-lapse reuses the sweep's clock and scrub wiring; `sweepState` now holds a *list* of
layers (one instanced mesh per mechanism) instead of a single mesh, so the eikonal sweep is
just the one-layer case. No new render code, no second animation path.

**Two arrival-time clocks, on purpose.** `path_loss` reports the eikonal τ (charged for
in-wall slowdown, ~+6 ns median on this scene); reflection, diffraction and scattering report
vacuum path length. Mixing them put the direct front LAST in 99 % of interior voxels. So
`path_loss` also exports `tau_geom_*.bin` (d/c) and the time-lapse runs every layer on that
one convention — an honest path-length comparison in which direct ≤ reflected ≤ diffracted
holds by construction. The **Wavefront sweep** view keeps the true eikonal arrival.
Charging the other three mechanisms for in-wall slowdown is the real fix and is a physics
change, not an export change.

**Two findings the mechanism views surfaced** (both pre-existing, neither introduced here):
1. *Diffuse scattering dominates the coverage map.* At tx_66-5-54 / 2442 MHz the direct field's
   median interior PL is **302 dB** while the diffuse channel's is **103 dB** — diffuse beats
   direct in **94.6 %** of interior voxels. The outbound patch→Rx leg is charged **no** wall
   loss (`Scattering_3D._accumulate_*` applies `obs_tx` to the inbound leg only), so diffuse
   power leaks through structure unattenuated. The direct path meanwhile has no saturating
   obstruction model (the 2D engine's `satObs`), so it runs away to 300–500 dB.
2. *The browser assets were a re-voxelization behind.* `sim_assets_3d.js` / `collision_3d.js`
   still described the pre-M0.4 262×11×118 grid while the volumes were 262×17×132. Both are
   regenerated; `insideMaskFor()` now refuses to index one grid with the other's strides.

### M3 — Four modes + offload/cache  🚧
- ✅ **`modes_3d.py`** — the mode registry. Four modes, one solve path: modes differ only
  in which grid they build and where the source is, so that difference is data, not four
  forked scripts.
  | mode | scene | source | mechanisms |
  |---|---|---|---|
  | `vacuum` | all air, production grid shape | point | path loss only |
  | `indoor` | the voxelized 7th floor | point | all four |
  | `o2i` | the same floor | plane wave on the facade | facade + reflection/diffraction/scattering |
  | `outdoor` | `city/NoMa_DC_buildings` | point | path loss, reflection, diffraction |
- ✅ **Vacuum is the invariant gate made visible.** The engine must reproduce
  `20log₁₀(d) + 20log₁₀(f) − 27.55` exactly; the browser re-checks the exported volume
  against the closed form and prints the worst deviation, so the gate is something the
  user can *see* rather than a line in a test log.
- ✅ **O2I geometry is derived, not guessed.** `forte_hall_geometry()` computes
  **415.9 m at arrival bearing 237.0°** from the floor plan's own QGIS georeference plus
  the known rooftop coordinates — replacing `bs_preset.bearing_deg = 135`, which its own
  comment called a demo placeholder. Outdoor-leg FSPL 80.7 dB (619 MHz) → 96.2 dB
  (3710 MHz). The tests assert the *derivation*, not the constant.
- ✅ **`facade_sources_3d` / `bs_field_3d`** — 3-D port of `phase_a.facade_sources`/
  `bs_maps`. Normals are estimated for the whole grid at once (the per-voxel Python
  version was ~573k iterations; the vectorized one is 0.15 s). Opposite bearings light
  **disjoint** facades — the property that makes O2I directional at all.
- ✅ **Mode selector in the browser**, with the volume catalog filtered by mode. A volume
  solved in one mode is not interchangeable with another's, so the filter is correctness,
  not tidiness.
- ⬜ `cache_index.py` (content-addressed LRU) · browser prefetch of neighbouring Tx.

**Why the O2I field is cached.** Every facade source costs one `crossing_loss` solve
(~2.9 s), so one source per lit voxel is 645 solves — half an hour for one map. But the
O2I source geometry is *fixed*: Forte Hall does not move. So the field is a one-time
computation keyed by scene+bearing+loss+bands, ~2 min at the default 48 sources and free
afterwards. Same reasoning as the diffraction relay cache.

**Two deliberate departures from the indoor path**, both because the transmitter is
416 m outside the grid: the FSPL floor is disabled in O2I (flooring against an in-grid
facade voxel would clamp the map to a fiction), and the facade contributions are summed
as **power, not field** — they discretize one wavefront, so a coherent sum would
manufacture an interference pattern that is an artifact of the sampling stride.

Browser resolution order: **cached volume → DL surrogate → analytic fallback.**

### M4 — Dataset + DL surrogate  ⬜ *(Colab A100)*
Finish the dataset against the **fixed** scene and widened bands; rewrite the trainer with the proven
resumability pattern (Drive checkpoints every 2 epochs, full state, seed pinning, AMP, ONNX parity
gate ≤ 0.1 dB). Physics-constraint losses from day one: FSPL floor, band ordering, causality
`τ ≥ d/c`, reciprocity, energy shells. **Target RMSE ≤ 5 dB.**

🔬 Then, off the critical path: per-mechanism complex surrogates predicting `(log|E|, cos φ, sin φ)`
— decoupling smooth magnitude from high-wavenumber phase; FNO-vs-UNet ablation on diffraction only;
residual head; differentiable calibration.

### M5 — Validation  ⬜
`validate_scanner_3d.py` + `Construct_Reciever_3D.py` (RSRP/RSRQ/SINR/RSSI).

---

## 4. Data flow

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
        web/volumes/*.bin        dataset/shard_*.npz     validate_scanner_3d.py
         (browser cache)          (surrogate training)      (vs G-flex data)
```

---

## 5. Test harness — `tests3d/`

```
tests3d/
├── conftest.py              path bootstrap, seeded fixtures
├── synth3d.py               free space · slab · two-plate · corner · corridor · rough wall · PEC box
├── invariants.py            reciprocity · energy · causality · FSPL floor · monotone LOS ·
│                            band ordering · |E|² == PL · shadow-boundary continuity
├── test_scene_sanity.py     ceiling sealed · floor concrete · class proportions
├── test_geometry_registration.py   px→voxel roundtrip · footprint IoU ≥ 0.85
├── test_engine_3d_core.py   FSPL exact in vacuum · reciprocity · T ≥ d/c
├── test_mech_*.py           one per mechanism, analytic first then regression
├── test_combiner.py         the superset proof
├── test_emergent.py         multipath · delay spread · K-factor · waveguiding
└── run_all.py               --fast | --full | --gpu
```

Each mechanism also keeps a dependency-free self-test: `python Reflection_3D.py --test`
(the `physics_v2.py --test` convention).

---

## 6. Validation gates

| Gate | Criterion |
|---|---|
| **V0** physics self-consistency | invariants green · energy closes < 1 % · reciprocity < 0.5 dB · causality exact · FSPL floor violated < 0.1 % |
| **V1** spatial structure | Spearman ρ(sim PL, −RSRP) ≥ 0.6 on held-out cells |
| **V2** anchored level | held-out RMSE ≤ **8 dB** after per-donor constants |
| **V3** delay structure | simulated RMS delay spread within 2× of measured `Ref Signal - Delay Spread` (5,951 unused samples) |
| **V4** per-material calibration | Forte Hall known-Tx case only |

> **Why 8 dB and not 3 dB.** Indoor measurement scatter with an unknown or partially-known
> transmitter is 6–10 dB. The 2D surrogate's 4.68 dB RMSE is *sim-vs-sim* and is not comparable to a
> sim-vs-measurement number. Claiming better would be claiming to beat the experiment's own noise floor.

---

## 7. Known limits

1. **WiFi 5.5 / 6.1 GHz cannot be validated** — no measurements exist at those bands. This is why
   cellular bands are added early: they are what make validation possible.
2. **Donors other than Forte Hall are at unknown sites** → shape-only (per-donor fixed-effect)
   validation; absolute level and per-wall values are not identifiable from them.
3. **Full-wave 3D ground truth is impossible at this scale** — at 3.5 GHz, λ = 8.6 cm ⇒ h ≤ 8.6 mm,
   so a 256³ grid spans only ~2.2 m. The residual head is limited to 2D slices and scaled-frequency
   crops; 2D→3D transfer is unproven and labelled as such.
4. **Scene material fidelity is the binding constraint**, not solver quality — hence M0 first.

---

## 8. References

- Seybold, *Introduction to RF Propagation* — the physics; per-chapter notes and verified equation
  implementations live in [`RF Propagation Theory/`](../../../RF%20Propagation%20Theory/).
- **ITU-R P.2040-3** — building material electrical properties (εr, σ vs frequency).
- **ITU-R P.1238** — indoor path loss coefficients N, floor-penetration Lf, delay spread.
- NVIDIA **Sionna RT** — differentiable ray tracing (reference for the differentiable-core branch).
- **RadioUNet** / FNO / DeepONet / PINNs — surrogate prior art to benchmark against.
