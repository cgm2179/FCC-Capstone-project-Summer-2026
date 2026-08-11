# Plan v2 — 3D RF Propagation Simulator (physics → surrogate → validated)
*Updated 2026-08-03. Supersedes v1; status verified against the repo, not commit messages.*

## Goal (unchanged)
A 3D RF propagation simulator that implements the six EM mechanisms from the Seybold
textbook, renders in the browser with a time-lapse of the propagation mechanism, is
accelerated by a deep-learning surrogate trained on physics-generated data, and is
cross-validated against real PCTEL G-flex scanner measurements.

## Status at a glance

| Milestone | Status | What's left |
|---|---|---|
| **M0** unblock · re-voxelize · register | ✅ **complete** | 2 documented test files never written (minor) |
| **M1** complex-field spine | ✅ **complete** | — |
| **M2** six mechanisms + demo | ✅ **complete** | optional in-wall τ clock; six mechanisms + satObs + georef shipped (PR #10) |
| **M3** four modes + offload/cache | 🚧 **demo outdoor done** | full-city @ 1 m volumes still A100-scale; demo tile `129×33×129` + `tx_67-20-66` cached |
| **M4** dataset + DL surrogate | 🚧 **browser path live** | smoke ONNX+JSON + inference wired; full Colab train still ⬜ (Pre-M4 clip gate cleared) |
| **M5** validation vs scanner | ⬜ **absent** | `Construct_Reciever_3D` (0 B) + `validate_scanner_3d` (missing) — **the goal** |
| **UI** usability layer (new) | ✅ **Tier 1** | Tiers 2–3 (link profile, result layers, multi-Tx, walk-vs-measured) |

## What changed since v1 (the old "verified findings", now resolved)
1. ✅ **The voxel scene is fixed.** Re-voxelized from the 253-material mesh → `262×17×132`,
   concrete **11.33%**, furniture **2.17%**, sealed ceiling, concrete floor. (Was 0.08% /
   0.02% / no ceiling.) Scene-sanity gates pass.
2. ✅ **The known Tx / O2I geometry is wired.** `forte_hall_geometry()` derives **415.9 m,
   arrival 237.0°** from the QGIS georeference + rooftop coords — the guessed `bearing=135`
   is gone.
3. ✅ **Cellular bands + widened norm are in.** `freqs_mhz` includes 619/627/1935/2510/2600/
   3710; `norm.freq_log = [600, 6200]`. The registered indoor `Data/records_data.js` is
   **already NR FR1** (bands 2/25/41/71, with rsrp/rsrq/cinr/rssi + a delay-spread column),
   so v1's "re-export to add cellular" item is largely **already satisfied** for the indoor
   truth set. The outdoor truth set (`FCC_Walk_Outdoor_Indoor_Full/`) exists with n7x names.
4. ✅ **`Antenna_Type_3D.py` restored** (961 lines); importers unblocked.
5. ✅ **M4 trainer plumbing is rewritten.** `phase_c3_train_colab.ipynb` now carries
   resumability / AMP / seed pinning / ONNX parity, and `ShardDS` keeps fp16 shards in RAM
   until `__getitem__`. The remaining M4 blocker is physics label quality, not trainer shape.

---

## Remaining work (what needs to be done)

### M2 — finish the mechanisms + one correctness fix  ✅
**Done (PR #10):** six mechanisms (`Diffraction_3D`, `Reflection_3D`, `Scattering_3D`,
`Refraction_3D`, `Absorption_3D` + path loss); diffuse-dominates fix (outbound wall loss +
`satObs` on the direct path); `georef.js`; mech-channel volume export + browser views.

**Optional remaining:**
1. ⬜ **In-wall-slowdown clock.** Reflection/diffraction/scattering still report vacuum path
   length, not eikonal τ; time-lapse works around it with `tau_geom`.

### M3 — finish outdoor  🚧
**Done:** mode registry + vacuum/O2I; three-tier cache; **demo outdoor tile**
(`city_demo/NoMa_DC_tile`, `129×33×129` @ 2 m) with browser assets, one cached Tx
(`tx_67-20-66` @ 2412 MHz), frontend outdoor scene swap, crop flags on `voxelize_city.py`,
and `choose_tx`/`export_pl_volume` reading the mode's own `valid_tx_mask`.

**To do:**
1. ⬜ **Full-city outdoor volumes (A100).** Rebuild `city/NoMa_DC_buildings/`
   (`2768×74×1776` @ 1 m — gitignored, regenerable) and run `precompute_volumes.py --mode outdoor`
   for a multi-Tx sweep. Demo tile unblocks the UI path without that RAM bill.

### M4 — dataset + DL surrogate  🚧  *(Colab A100, long unattended; SPEED is the binding concern)*
> **Speed strategy (user direction): train on xy/yz/zx 2-D PLANES, not full 3-D.** The 2-D
> surrogate already hit 4.68 dB (tractable); decomposing the volume into the three axis-aligned
> slice stacks keeps every network 2-D and composes to 3-D — "speed up solving exponentially,"
> and it sidesteps the 256³ full-wave wall. This is also what makes **imported / arbitrary
> scenes** fast: a new scene is a cache miss (it can't reuse the fixed-grid cached volumes or a
> scene-locked surrogate), so today an imported model runs on the slow analytic tier; plane
> surrogates generalize across geometry without a per-scene precompute.

1. ✅ **Trainer rewritten:** `phase_c3_train_colab.ipynb` has checkpoint/resume, AMP,
   deterministic seeds, early stop/cosine LR, physics-constraint losses, and an ONNX parity
   gate. Do not rewrite it for scaffold work.
2. ✅ **ShardDS fp32 bug fixed:** shards stay fp16/memmap-friendly and cast at sample time.
3. 🚧 **Smoke ONNX + sidecar shipped:** `SIM3D/export_surrogate_smoke.py` writes
   `SIM3D/web/pl_unet3d.onnx` + `pl_unet3d.json` with untrained weights and `"smoke": true`.
   This validates the browser contract only; it makes **no** ≤5 dB RMSE claim.
4. 🚧 **Browser inference wired:** `simulation3d.js` can build the 9-channel input, run
   ONNX Runtime (WASM first), cache the denormalized PL/tau volume, and label the tier
   `DL surrogate` when used.
5. ⬜ **Full dataset + Colab train:** Pre-M4 clip gate is **cleared** (satObs → worst clip
   ≪ 35%). Generate shards on Colab A100 via `phase_b3_dataset.ipynb`, train with
   `phase_c3_train_colab.ipynb`, replace the smoke ONNX with a real ≤5 dB model.
6. 🔬 **Plane surrogate scaffolded:** `plane_surrogate_3d.py` documents xy/yz/zx 2-D slice
   training and compose-back-to-3-D as the speed strategy for arbitrary scenes.
7. 🔬 **Research branch (later):** per-mechanism complex surrogates `(log|E|, cos φ, sin φ)`;
   FNO-vs-UNet on diffraction only; residual head + differentiable calibration.

### M5 — validation against the scanner data  ⬜  *(the goal)*
> M2 scattering/satObs fix has landed — validation against the physics map is now meaningful.

1. ⬜ **`Construct_Reciever_3D.py`** (currently **0 bytes**): turn PL → RSRP/RSRQ/SINR/RSSI.
2. ⬜ **`validate_scanner_3d.py`** (absent): compute the gates, write `validation_report.json`
   + PNGs.
3. ⬜ **`osm_building_height.py`** (absent): resolve the Forte Hall antenna height (OSM/OBM
   at 38.90155/−77.01142; fall back ~10–12 m). *(A city manifest already carries
   `ceiling_height_m = 72.061` as one anchor.)*
4. ⬜ **Aggregate & anchor:** ≥3 samples/cell (median) to suppress fast fading; per-donor
   fixed effects for unknown donors; use the scanner's `Ref Signal - Delay Spread` (5,951
   samples) to validate the **combiner**, not just path loss.
5. **Gates:** V0 invariants green · V1 Spearman ρ ≥ 0.6 · V2 held-out RMSE ≤ **8 dB** after
   per-donor constants · V3 RMS delay spread within 2× of measured · V4 per-material
   calibration on the Forte Hall known-Tx case only. *(8 dB, not 3: sim-vs-measurement with a
   partially-known Tx has a 6–10 dB floor; the 2D 4.68 dB is sim-vs-sim.)*

### UI — usability layer (new workstream, WinProp/WRAP-informed)
**Done (Tier 1, PR #7 → main):** prediction-plane height slider, received-power view
(EIRP→RSRP, fixed dBm scale, good/marginal/dead classes), click-a-cell point probe
(PL/RSRP/τ + dominant mechanism), persistent context strip. Frontend-only; reuses
`plAt`/`chanAt`/`runVizMode`.

**To do:**
1. ⬜ **Slice 1A/1B (deferred):** unify Run⇄Visualize into one verb; expose the engine/tier
   selector (`cached / surrogate / analytic`) with a timing readout.
2. ⬜ **Tier 2:** Tx→Rx **link profile** (distance-vs-PL, wall crossings, first-Fresnel
   clearance, LOS/NLOS); more Visualize layers — LOS/NLOS mask, **RMS delay spread** (from
   `tau_rms`, already in the contract), best-server.
3. ⬜ **Tier 3:** multi-Tx **best-server / SIR**; **virtual walk-test playback + measured
   PCTEL overlay** — this *is* the M5 cross-validation made interactive, so build it with M5.

---

## Recommended sequence (guidance)
1. **M2 scattering fix** (correctness gate for everything downstream) →
2. **M5** `Construct_Reciever_3D` + `validate_scanner_3d` (the goal; needs no surrogate) →
3. **M2** Refraction + Absorption + `georef.js` (completes the six mechanisms) →
4. **M3** outdoor city volumes →
5. **M4** dataset + surrogate (accelerator, off the critical path) →
6. **UI** Tiers 2–3 (Tier 3 walk-vs-measured pairs naturally with M5).

## Critical files
- `SIM V1 3D/engine_3d.py` — `SceneV3` (`crossing_loss` = amplitude, `arrival_time` = phase).
- `…/Wave Behavior/Enivronmental Interaction/` — the mechanism modules (2 still 0-byte).
- `SIM V1 3D/modes_3d.py`, `cache_index.py`, `precompute_volumes.py`, `voxelize_city.py`.
- `SIM V1 3D/phase_b3_dataset.ipynb`, `phase_c3_train_colab.ipynb` (rewrite target).
- `SIM V1 3D/validate_scanner_3d.py` (create), `…/Reciever Objects/Construct_Reciever_3D.py`
  (0 bytes → fill).
- `Frontend/simulator/simulation3d.js` (UI); `Frontend/2d-3d/georef.js` (create).
- Truth data: `Data/records_data.js` (already NR FR1), `FCC_Walk_Outdoor_Indoor_Full/`.

## Known limits (unchanged)
WiFi 5.5/6.1 GHz can't be validated (no measurements). Non–Forte-Hall donors are
shape-only. Full-wave 3D ground truth is infeasible at scale (256³ ≈ 2.2 m at 3.5 GHz) —
the residual head is limited to 2D slices / scaled-frequency crops; 2D→3D transfer is
unproven.

---
*Companion docs: `PLAN_3D_SIM.md` (in-repo milestone log, engine-side detail) · the original
v1 plan (`making 3D RF Propagation Simulator (physics → surrogate → validated).md`).*
