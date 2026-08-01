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
| Reflection | `Reflection_3D.py` | coherent **E** | `r_slab`, `fresnel_coeffs`, image sources | GPU |
| Refraction | `Refraction_3D.py` | coherent **E** | `speed_field`, eikonal, `slab_transmission_coherent` | local |
| Diffraction | `Diffraction_3D.py` | coherent **E** | `utd_coefficient` (complex D) | local / GPU |
| Scattering | `Scattering_3D.py` | **`p_incoh`** | effective-roughness (Degli-Esposti) | GPU |
| Absorption | `Absorption_3D.py` | multiplier + absorbed-power density | `Im(q)`, `per_metre` | local |

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

### M0 — Unblock, re-voxelize, register  🚧
- ✅ **M0.1** Restore `Antenna_Type_3D.py` (was truncated to 17 lines; 22 antenna types recovered,
  3 importers unblocked, original authorship header preserved).
- 🚧 **M0.2** This document.
- ⬜ **M0.3** `tests3d/` harness — analytic scenes + physics invariants (see §5).
- ⬜ **M0.4** Re-voxelize from `Sandbox_Version_3D_Simulation_1.obj` (**253 named materials** vs the
  current 18): seal the ceiling, reclass the floor slab to concrete, exclude ~40 people/prop materials.
- ⬜ **M0.5** Fit `registration_3d.json` (px → local m → voxel); widen `norm.freq_log` to
  `[600, 6200]` and add the measured cellular bands.

> **Why M0 gates everything.** Layer census of the current `material_grid.npy`:
> **y=10 is 100 % air → no ceiling slab**; the floor slab is `drywall_partition` (ε′≈2.9) not
> concrete (ε′≈5.24); **concrete = 284 voxels (0.08 %)**, **furniture = 73 (0.02 %)**, drywall = 23 %.
> For a ceiling-mounted AP the floor and ceiling bounces are the two dominant specular paths — one has
> the wrong permittivity, the other does not exist. Six perfect solvers on this grid would produce a
> confidently wrong answer. **Material fidelity, not solver quality, is the binding error term.**

### M1 — Complex-field spine  ⬜
`contracts.py` (`FieldGrid`, `CombinedField`) · `Path_Loss_3D.py` · `Combine_3D.py`.
**Gate:** `to_legacy_volumes(combine([pathloss])) == SceneV3.pathloss_maps(tx)` to < 0.01 dB —
proving the new architecture is a strict superset of the working engine, with zero exporter or
frontend changes required.

### M2 — Mechanisms + the demo  ⬜
`Diffraction_3D` → `Reflection_3D` → mechanism channels in `export_pl_volume.py --mechanisms` →
enable the six disabled `viz3dMode` options → **mechanism time-lapse**.

The time-lapse reuses `runWavefrontSweep`/`setSweepTau` unchanged: feed each mechanism's `tau_first`
as another channel, one instanced mesh per mechanism. The reflected front genuinely arrives after the
direct, the diffracted after that, the diffuse halo last — so playing them together *is* the
mechanism time-lapse, with no new render code.

### M3 — Four modes + offload/cache  ⬜
Remaining mechanisms (`Refraction`, `Absorption`, `Scattering`) · mode selector ·
`cache/precompute_volumes.py` + `cache_index.py`.
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
