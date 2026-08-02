# Plan v2 — 3D RF Propagation Simulator (physics → surrogate → validated)
*Updated 2026-08-02. Supersedes v1; status verified against the repo, not commit messages.*

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
| **M2** six mechanisms + demo | 🚧 **3 of 6** | Refraction + Absorption (0-byte stubs); **fix diffuse-dominates bug**; `georef.js` |
| **M3** four modes + offload/cache | 🚧 **indoor/O2I done** | **outdoor has a grid but no cached city volumes** |
| **M4** dataset + DL surrogate | ⬜ **~10%** | 27/30 shards; rewrite trainer; export `.onnx` + `.json` sidecar |
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
5. ⚠️ **`phase_c3_train_colab.ipynb` still lacks** resumability / AMP / seed pinning / the
   ONNX parity-gate — the fp32-at-load bug family is **not yet addressed** (M4 below).

---

## Remaining work (what needs to be done)

### M2 — finish the mechanisms + one correctness fix  🚧
**Done:** `Diffraction_3D` (UTD, 405 L), `Reflection_3D` (image sources, 372 L),
`Scattering_3D` (Degli-Esposti, 419 L); `export_pl_volume.py --mechanisms` +
`precompute_volumes.py --mech-channels` write `m_<mech>_*.bin` / `tau_<mech>_*.bin`;
browser `runMechanismField` / `runMechanismTimeLapse` live for the four implemented channels.

**To do:**
1. 🐛 **Fix the diffuse-scattering-dominates artifact (highest value).** At `tx_66-5-54` /
   2442 MHz the diffuse channel (median 103 dB) beats the direct field (median **302 dB**)
   in **94.6% of interior voxels**. Two root causes, both in existing code:
   - `Scattering_3D._accumulate_*` charges wall loss on the **inbound** patch leg only
     (`obs_tx`), so diffuse power leaks through structure unattenuated on the outbound leg.
   - The direct path has **no saturating obstruction model** (the 2D engine's `satObs`), so
     deep-shadow PL runs away to 300–500 dB. Port `satObs` into the 3D direct path.
   *Until this is fixed the coverage map — and any dataset or validation built on it — is
   dominated by an artifact.* This is now the binding correctness term (v1's "material
   fidelity is binding" → resolved; "physics correctness" takes its place).
2. ⬜ **`Refraction_3D.py`** (0 bytes → implement). Reuse `speed_field`, `arrival_time`,
   `slab_transmission_coherent`; emits coherent **E**. Cheap/local.
3. ⬜ **`Absorption_3D.py`** (0 bytes → implement). Reuse `Im(q)` from `electrical_thickness`,
   `per_metre`/`fill_fraction`; emits a multiplier + **absorbed-power-density** viz artifact.
   Removing these two 0-byte stubs is also what un-disables their two `viz3dMode` options.
4. ⬜ **In-wall-slowdown clock (physics, optional but honest).** Reflection/diffraction/
   scattering currently report vacuum path length, not the eikonal τ; the time-lapse works
   around it with `tau_geom`. The real fix is charging those three for in-wall slowdown.
5. ⬜ **`Frontend/2d-3d/georef.js`** (absent → create): `lonlatToPx` / `pxToLocalM` /
   `localMToVox` / `voxToWorld` from `registration_3d.json` + `floorplan_meta.json`.
   `viewer3d.js` still hardcodes `FLOOR_W/FLOOR_H` and rubber-sheets geographic CSVs — this
   blocks registering the scanner walk onto the model, which M5 needs.

### M3 — finish outdoor  🚧
**Done:** `modes_3d.py` (vacuum/indoor/o2i/outdoor registry), vacuum invariant gate,
O2I facade sources, `cache_index.py` (content-addressed on a physics-source hash, LRU,
`--verify`), three-tier `cached → surrogate → analytic`, neighbour prefetch. `index.json`
carries 6 cached Tx (indoor ×4, o2i ×1, vacuum ×1).

**To do:**
1. ⬜ **Cache outdoor city volumes.** The grid exists (`SIM V1 3D/city/NoMa_DC_buildings/`,
   `2768×74×1776` @ 1 m, from the complete `voxelize_city.py`) but **no `web/volumes/` entry
   is a city Tx** — outdoor mode has geometry and renders nothing. Run `precompute_volumes.py`
   on the city grid (A100; large) to populate the outdoor tier. Physics is ground reflection
   + terrain diffraction, both already covered by the M2 modules.

### M4 — dataset + DL surrogate  ⬜  *(Colab A100, long unattended; SPEED is the binding concern)*
> **Speed strategy (user direction): train on xy/yz/zx 2-D PLANES, not full 3-D.** The 2-D
> surrogate already hit 4.68 dB (tractable); decomposing the volume into the three axis-aligned
> slice stacks keeps every network 2-D and composes to 3-D — "speed up solving exponentially,"
> and it sidesteps the 256³ full-wave wall. This is also what makes **imported / arbitrary
> scenes** fast: a new scene is a cache miss (it can't reuse the fixed-grid cached volumes or a
> scene-locked surrogate), so today an imported model runs on the slow analytic tier; plane
> surrogates generalize across geometry without a per-scene precompute.

1. ⬜ **Finish the dataset:** 3 of 30 shards exist (`SIM V1 3D/dataset/shard_00{0,1,2}.npz`,
   `splits.json` = 1198/151/151). Generate the remaining **27** against the **fixed** scene
   and widened bands — *after the M2 scattering fix*, or the labels bake in the artifact.
   Fix the `ShardDS.__init__` fp32-at-load bug (keep fp16 in RAM, cast in `__getitem__`).
2. ⬜ **Rewrite `phase_c3_train_colab.ipynb`** — it is the OLD version (has an `onnx`/`parity`
   mention but **no** checkpoint/resume, AMP autocast/GradScaler, seed pinning, or a hard
   `assert worst ≤ 0.1 dB` gate). Port the proven 2D v3 pattern: Drive checkpoints every 2
   epochs (full `{net,opt,sched,scaler,epoch,best}`), cosine LR, early stop, ONNX parity gate.
   Physics-constraint losses from day one (FSPL floor, band ordering, causality τ≥d/c,
   reciprocity, energy shells). **Target RMSE ≤ 5 dB.**
3. ⬜ **Ship `pl_unet3d.onnx` + write `pl_unet3d.json`** to `SIM3D/web/`. The browser's
   surrogate tier stays dark until BOTH exist (it refuses to guess the input contract);
   the analytic mirror keeps everything working meanwhile, by design.
4. 🔬 **Research branch (later):** per-mechanism complex surrogates `(log|E|, cos φ, sin φ)`;
   FNO-vs-UNet on diffraction only; residual head + differentiable calibration.

### M5 — validation against the scanner data  ⬜  *(the goal)*
> Depends on the M2 scattering fix — validating an artifact-dominated map is not meaningful.

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
