# Unity Map-Coverage plan — Outdoor fix → OSM align → Indoor Unity → Wi‑Fi Tx

## STATUS — all four implemented; awaiting ONE Unity rebuild + verify/tune
Verified here as far as the environment allows (JS message-flow, syntax, C# API/brace cross-check).
Unity render + the two blind assumptions must be verified/tuned by you.

**Files changed**
- *Unity C# (needs rebuild):* `Assets/Scripts/Viz/OutdoorNoMaView.cs` (Step 2 basemap centre/raise + nudge + visibility), `CoverageDemo.cs` (Step 4 Wi‑Fi default + Tx marker; Step 3 hooks), `IndoorBaseStations.cs` (NEW, Step 3), `Bridge.cs` (route indoor + basemap + floor heading), `Assets/Editor/WebGLBuilder.cs` (add IndoorBaseStations to the Coverage scene).
- *Frontend JS (live now, no rebuild):* `Frontend/osm3d/outdoor_view.html` (scene phase machine + Model=indoor → indoor Unity scene), `Frontend/simulator/unity_bridge.js` (Wi‑Fi‑only bands, OSM reveal/align controls, indoor floor donor + heading controls).

**Rebuild (your one R)**
1. In `Unity_RF_Simulator/Unity_RF_Simulator/`, run `WebGLBuilder.BuildWebGL` (Unity 6, headless or Editor menu).
2. Copy `Build/WebGL` → `indoor-walk-test/misc/unity/Build/WebGL` (the vendored build the dashboard serves).
3. Hard-reload the dashboard.

**Verify + tune (report the tuned numbers → I bake them as C# defaults)**
- *Step 1 outdoor render:* Map Coverage → Unity 3D (C#) → NoMa buildings render, no black/reload.
- *Step 2 OSM:* Simulation tab, outdoor scene → tick **OSM map**, set **Align X/Z/Rot** until streets sit under the buildings (heatmap off). → give me X, Z, Rot.
- *Step 3 indoor:* Map Coverage → Model **Indoor (FCC 7th floor)** + Unity  (or Simulation tab → Scene **Indoor floor**) → the 7th‑floor model + base‑station posts + Wi‑Fi Tx + floor coverage. Set **Floor°** until the posts point at the real bearings. → give me the heading.
- *Step 4 Wi‑Fi:* indoor band selector shows only 2442 / 5500 / 6125 MHz; the Tx gizmo is labelled "Tx · Wi‑Fi … MHz".

**Design (Step 3/4):** the floor coverage is a **separate in-floor Wi‑Fi AP** — `CoverageDemo`, Wi‑Fi band, interior Tx (visible "Tx · Wi‑Fi" gizmo). The georeferenced fixed base stations are **cellular donor CONTEXT** — posts at their true bearing/distance with a modelled BS→floor path‑loss readout (FSPL + clutter + O2I through the unrendered clutter, at the donor's own cellular band); they do **not** drive the floor coverage. **Anchor = FCC HQ, 45 L St NE (lat 38.9034138, lon −77.009815 → tile vox 302.33/160.01)** — used for the indoor floor georeference and the outdoor FCC marker/Rx (the base_stations `fcc` entry is the walk centroid, ~210 m off). The in-floor Wi‑Fi Tx is a **bright vertical beam at floor centre** — zoom in with the 3D view to inspect. **Floor heading still defaults to 0 and needs your tuning.**

---


Sequenced plan for the four Unity Map-Coverage items. Repos:
- **UI / vendored build:** `indoor-walk-test` (this repo) — `Frontend/…`, `misc/unity/Build/WebGL`, `scripts/*.py`
- **Unity C# source:** `indoor-outdoor-walk-test-with-Unity-Engine` → nested at `Unity_RF_Simulator/Unity_RF_Simulator/` (own git)

## Working constraint (read first)
The in-app preview browser **cannot render Unity WebGL** (its tab is backgrounded → `requestAnimationFrame` is paused → the player loop never advances past the splash; no C# callback fires). So for every C# item below the loop is:
1. Claude edits C# in `Unity_RF_Simulator/…/Assets/Scripts/…`
2. **You** rebuild: `WebGLBuilder.BuildWebGL` (Unity 6 headless), then copy `Build/WebGL` → `misc/unity/Build/WebGL` (+ `misc/unity/unity_embed.html` if changed)
3. **You** verify the render in a real browser; report back / screenshot
Claude can still verify the **driving JS/postMessage** (stub the iframe, fire synthetic `unitySim` messages, assert commands) — done for Step 1.

---

## Step 1 — Verify the Outdoor‑3D‑Unity render (DONE in code; needs your eyes)
**Status: fixed, JS‑only, no rebuild needed.**
- Bug: `Frontend/osm3d/outdoor_view.html` re-sent `SetScene outdoor` on *every* Unity `ready`. Both scenes carry a `Bridge` that fires `ready` on each scene load (`WebGLBuilder.cs:112,148`) → infinite Outdoor scene reload → black canvas.
- Fix: one‑shot guard (`unityScene` state) — `SetScene` sent once; dimension + sector sync deferred to the Outdoor scene's own `ready`. Verified: 3 synthetic `ready` events → `SetScene` sent exactly **1×** (was 3×).
- **Your check:** reload the dashboard → Map Coverage → **Unity 3D (C#)**. Expect: NoMa buildings render, status → "Unity ready — NoMa 3D + C# path loss", then a solve status ("… solved 4,803,400 vox in … ms"). If still black, capture the iframe console + network (repeated `scene_outdoor_noma_grid.bin` fetches would mean a second reload source).

---

## Step 2 — OSM ↔ NoMa‑model alignment + make OSM visible
Two sub‑goals: (a) the OSM ground lines up with the extruded buildings; (b) you can actually see the OSM.

**What's already consistent (ruled out):** `scene_outdoor_noma_manifest.json` and the basemap source `city/NoMa_FCC_tile/manifest_3d.json` are identical (grid 470×28×365, origin/anchor/merc_scale match); `bake_osm_basemap.py` reproduces `voxelize_city.vox_to_lonlat` exactly; Unity UVs in `OutdoorNoMaView.BuildBasemap` map world XZ→(vx,vz) with no flip. So extent/flip is **not** the cause.

**Leading hypothesis:** buildings come from `Data/models/NoMa_DC/NoMa_DC_buildings.obj` (voxelized), while the ground comes from live OSM raster tiles. If the OBJ's registration (origin/rotation/scale into the Mercator frame) differs from the OSM tiles, footprints sit off the streets.

**Plan:**
1. **Diagnose from a real render** (you): Unity outdoor, heatmap OFF, screenshot. Measure offset — pure translation? rotation? scale? (a translation = anchor/origin offset; rotation = OBJ frame not aligned to grid‑north; scale = merc_scale not applied to the OBJ).
2. **Fix at the source of the mismatch:**
   - If translation/scale → correct the OBJ→voxel registration in `voxelize_city.py` (or re-bake the basemap from the *same* footprint source instead of OSM tiles, so both come from one geometry).
   - If rotation → the OBJ isn't grid‑north‑aligned; bake a rotation into the georef or pre‑rotate the OBJ.
   - Re-run `bake_unity_scene.py` (+ `bake_osm_basemap.py`) and re-copy StreamingAssets.
3. **Make OSM visible** (`OutdoorNoMaView`): raise the basemap quad from `y=-0.5` to `y≈0`; when the coverage cut‑plane/volumetric is on, drop its alpha or add a "ground" toggle so the map reads through; confirm `ShowHeatmap off` cleanly reveals the ground.

**Files:** `Assets/Scripts/Viz/OutdoorNoMaView.cs` (BuildBasemap, y/alpha), `Physics Engine/3D Map Physics/SIM V1 3D/voxelize_city.py`, `scripts/bake_osm_basemap.py`, `scripts/bake_unity_scene.py`. Rebuild + re-copy.

---

## Step 3 — Build the Indoor 3D Unity mode (new; biggest item)
Today only three.js has an indoor 3D coverage view; the Coverage (indoor) Unity scene exists but isn't wired as a Map‑Coverage option and has **no base stations** indoors. Goal: an Indoor Unity option showing **only the FCC 7th‑floor model**, with **fixed base stations georeferenced relative to the floor**, faint city clutter between BS and floor, and **C# path loss from BS → 7th‑floor Rx grid**.

**Architecture:**
1. **Wire the option** in `outdoor_view.html` (or a sibling indoor host): the `Model` selector already has "Indoor building (FCC 7th floor)". Add an "Indoor Unity" path that sends `SetScene indoor` (Coverage scene) instead of `outdoor`, reusing the Step‑1 one‑shot guard.
2. **Georeference base stations to the floor** — the core new piece. Needed inputs (design decisions, see below): the 7th‑floor model's **geo anchor** (lat/lon of a known model point) + **heading** (model +X vs true east) + floor **elevation** (≈7th storey height). Then map each `base_stations.json` site lon/lat → floor‑local metres via a small analog of `voxelize_city.lonlat_to_vox` built for the indoor frame. BS will land far outside the ~79×40 m floor footprint (they're 100s of m away) — represent them as directional markers at true bearing/distance + elevation, clamped to the scene edge with a distance label.
3. **Path loss BS → floor:** solve in C# from the (external, elevated) BS Tx to the floor Rx grid. Distance uses the true BS→floor geometry; the "city clutter in between" adds barrier crossings. Options: (a) reuse the outdoor NoMa voxel grid for the BS→façade segment + the indoor grid for façade→Rx (two‑segment link budget, O2I façade loss between); (b) a single combined coarse grid. Recommend (a) — matches the existing 3GPP O2I term already in `Environments`/`PathLossConfig`.
4. **Faint clutter:** render the NoMa buildings at low opacity/desaturated (or just the BS→floor corridor) so the floor stays the focus. "not shown" in the equation display, but included in the solve.

**Open design decisions (need your answers before building):**
- **Floor geo anchor + heading:** do we have the FCC 7th‑floor model's real‑world position + orientation? (Have: `fcc` lon −77.0074/lat 38.90359, but that's the *walk centre* in outdoor tile voxels, not the model's anchor/heading.)
- **Which base stations** indoors — all 6 sites, or the live/donor ones?
- **Clutter model** for the BS→floor path — reuse outdoor NoMa grid (recommended) vs a simplified between‑layer?
- **Tx here is Wi‑Fi‑only (Step 4)** — but do the *external base stations* stay cellular while the *indoor Tx* is Wi‑Fi? (i.e., two Tx concepts: donor macro BS vs an in‑floor Wi‑Fi AP.) This determines the UI.

**Files:** new/extended `Assets/Scripts/Viz/` indoor view (mirror `OutdoorNoMaView`), `WebGLBuilder.BuildCoverageScene` (add Bridge hooks for BS selection indoors), `Bridge.cs`, host HTML/JS. Rebuild + re-copy.

---

## Step 4 — Wi‑Fi‑only transmitter for the Indoor Unity sim
Make the indoor Tx offer **only Wi‑Fi bands** and make its path loss easy to see.
- **Bands:** restrict to Wi‑Fi = **2442 (2.4 GHz), 5500 (5 GHz), 6125 (6 GHz)**; drop cellular (619/1935/3500/…). In `Frontend/simulator/unity_bridge.js` `BANDS_MHZ` is `[619,1935,2442,3500,5500,6125]` → for indoor show only the Wi‑Fi subset; set the C# indoor default `fMhz` (`CoverageDemo.fMhz`, currently 3500) to 5500.
- **Tx visibility:** `CoverageDemo` has no Tx marker (only `OutdoorNoMaView` draws markers/trajectory). Add a visible Tx gizmo + a clear "Tx: Wi‑Fi <band>" label and (optionally) a Tx→Rx link line so the transmitter/path is obvious.
- Depends on Step 3's decision on whether the Wi‑Fi Tx is an in‑floor AP vs the external donor.

**Files:** `Frontend/simulator/unity_bridge.js` (+ indoor host band list), `Assets/Scripts/Viz/CoverageDemo.cs` (default band, Tx marker). Rebuild + re-copy.

---

### Recommended immediate next action
Step 1 is code‑complete — **reload and confirm the outdoor render**. In parallel I can start **Step 2** (raise/reveal the OSM ground + set up the alignment diagnosis), which only needs one heatmap‑off screenshot from you to pin the offset. Step 3 is gated on the four design decisions above.
