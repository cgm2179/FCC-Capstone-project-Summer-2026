# 3D Simulator — UI Roadmap

A usability roadmap for the browser 3D RF simulator (`Frontend/simulator/simulation3d.js`
+ the `#sim3dPanel` rail in `Frontend_Data_Display.html`). It maps ideas taken from the
Altair **WinProp / WRAP** capability set onto structure this project already has, and keeps
every proposal tied to the four project pillars:

- **A — Physics from Seybold** (six EM mechanisms; `PLAN_3D_SIM.md §1`).
- **B — Browser time-lapse** of the propagation mechanisms.
- **C — DL surrogate** as an *accelerator* over the deterministic engine (`PLAN_3D_SIM.md §0`).
- **D — Validation** against PCTEL G-flex scanner measurements (`PLAN_3D_SIM.md §6`).

The guiding idea borrowed from WinProp: it is really **two stages** — *wave propagation*
(a field in space, which we have) and *radio network planning* (turning that field into
decisions/KPIs, which we mostly do not expose yet). Most "make it more usable" work is
either exposing stage two, or making stage one **legible** and **editable**.

---

## Slice 1 (this roadmap's first build): Engine/fidelity selector + Run⇄Visualize unification + dead-control cleanup

Two small, low-risk, self-contained changes in `simulation3d.js` + the rail markup. They
pay off immediately in day-to-day use and set up the pattern (a single dispatch + an
explicit engine tier) that later slices build on.

Recommended sequencing inside the slice: **do the unification/cleanup first** (it
establishes one Run verb routed through the existing `runVizMode` dispatch), then **layer
the engine selector on top** (it hooks the same dispatch to re-run on change). Ship both in
one PR — they touch the same Run/rail region.

### Current behaviour (baseline, verified in code)

- The **Visualize** dropdown (`#viz3dMode`) already routes every view through a central
  dispatch `runVizMode(mode)` (`simulation3d.js`), and re-runs on `change`.
- The **Run** section has **two** buttons that ignore the Visualize selection:
  `#run3dStatic` → `runStaticField()` (coverage), `#run3dAnim` → `runAnimWave()` (a
  kinematic sphere "visual stand-in", not in the Visualize list at all).
- Step 2 "Waveform" has **dead** radios `name="wf3dSolver"` (Static/Time-lapse) — no JS
  reads them.
- Step 3 "Receiver" has a **dead** checkbox `#rx3dInterference` — no JS reads it.
- Tier resolution (`cached volume → DL surrogate → analytic`) is decided **silently**
  inside `runStaticField`/`runMechanismField`; `TIER` + `SURROGATE` state exist, and
  `window.SIM3D_TIER` is set, but the user cannot see or force the tier, and there is no
  timing.

### 1A. Run ⇄ Visualize unification + dead-control cleanup

**Goal:** the **Visualize** property is the single source of truth for *what* is shown;
**Run** is one verb that (re)computes/(re)starts the selected view.

Markup (`Frontend_Data_Display.html`, `#sim3dPanel`):
- **Remove** the dead `name="wf3dSolver"` radios from Step 2 (Static vs Time-lapse is now
  expressed by the Visualize property: `coverage`/mechanism views are static; `sweep`,
  `mechlapse`, `animwave` are time-lapse).
- **Remove** the dead `#rx3dInterference` checkbox from Step 3. (Receiver-linked
  interference physics is a later slice; the existing `interference` view is Tx-keyed.)
- **Replace** the two Run buttons with one primary `#run3dGo` plus the existing
  `#anim3dControls` block. Keep `#anim3dPlay` / `#anim3dScrub` unchanged.
- **Add** `animwave` as a Visualize option: `<option value="animwave">Animated wavefront
  (kinematic)</option>` under a sensible group, so the kinematic sphere lives with the
  other time-lapse views instead of on its own Run button.

JS (`simulation3d.js`):
- Add an `animwave` branch to `runVizMode`: `if (mode === 'animwave') { runAnimWave(); return; }`.
- Introduce `VIZ_IS_ANIMATED = new Set(['sweep', 'mechlapse', 'animwave'])`. In
  `runVizMode`, show `#anim3dControls` iff the mode is animated, and **hide it for static
  views** (fixes the current lingering-controls case where switching from a sweep to
  coverage leaves the scrub bar visible).
- Give `#run3dGo` a **dynamic label** derived from the active Visualize option, updated on
  `#viz3dMode` change: e.g. `Run coverage`, `Play wavefront sweep`, `Play mechanism
  time-lapse`, `Play animated wavefront`, `Render field`, `Show radiation pattern`,
  `Run interference`, or `Run <mechanism>`. A small `runLabelFor(mode)` map/helper.
- Rewire: `#run3dGo` → `runVizMode(vizMode.value)` (re-run/restart the current view).
  Remove the `#run3dStatic`/`#run3dAnim` listeners.
- Preserve the existing "auto-run on Tx placement when a viz is already selected" and the
  band-change re-run — both already route through `runVizMode`, so they keep working.

Keep the current behaviour that `#viz3dMode` **change** auto-runs the selected view (it
falls back to the analytic tier instantly, so it stays responsive); `Run` becomes the
explicit **re-run / start playback** control.

### 1B. Engine / fidelity selector with timing

**Goal:** make the three-tier resolution visible and forceable, and show how long each
solve took — the surrogate's whole reason to exist (pillar C).

Markup: add a new `<section class="sim3d-step">` **"Engine"** immediately **before** the
Run section:
- `select#engine3dTier`: `auto` (default — *Auto: cached → surrogate → analytic*),
  `cache` (*Full physics (cached volume)*), `surrogate` (*DL surrogate*),
  `analytic` (*Analytic (in-browser)*).
- `p#engine3dNote.sim3d-modenote` — one line explaining the active tier / why the surrogate
  is unavailable (reuses `SURROGATE.note`, e.g. *not trained yet (M4)*).
- `span#engine3dTiming` — last-solve readout, e.g. `cached volume · 42 ms` /
  `analytic · 8 ms`.

JS:
- Module state `let forcedTier = 'auto';` and `let lastSolve = null;`.
- Consume `forcedTier` where the tier is currently decided:
  - `analytic` → do **not** use `window.SIM3D_VOLUME`, do **not** call `tryLoadVolume`;
    render `marchPL` and label the readout `analytic (forced)`.
  - `cache` → require a matched volume; if none, keep showing the analytic floor but set an
    honest status (`requested cached — none within tolerance; showing analytic`) instead of
    silently substituting.
  - `surrogate` → `await tryLoadSurrogate()`; while untrained it reports `unavailable` and
    falls through to analytic with the reason shown (honest, matches `PLAN_3D_SIM.md §M3`).
  - `auto` → today's behaviour, unchanged.
- **Timing:** wrap the field computation (the sampling loop in `runStaticField`, and the
  equivalent in `runMechanismField`/`runVacuumField`) in `performance.now()` deltas; store
  `lastSolve = { tier, ms, mode, freqMHz }` and render it into `#engine3dTiming`. The tier
  label is already known at the point the status line is built.
- Re-run on change: `engine3dTier` `change` → set `forcedTier`, then
  `runVizMode(vizMode.value)`.
- Expose on the debug API: add `get forcedTier`, `get lastSolve` to `window.__sim3d`.

Honest edge cases to encode (all become clear status messages, not silent fallbacks):
- **Per-mechanism views** (`path_loss`/`reflection`/`diffraction`/`scattering`) have **no**
  analytic equivalent — with `analytic` forced, say so and point to Auto/Cached.
- **O2I** has no in-grid Tx, so `marchPL` can't run — with `analytic` forced in O2I, say
  the analytic tier is unavailable for O2I and to use Cached.
- **Vacuum** coverage stays the invariant-gate view (`runVacuumField`); timing still
  recorded.

### Combined rail after Slice 1

```
0  Propagation mode      (#mode3dSelect, #mode3dNote)          — unchanged
1  Transmitter           (#tx3dType, #tx3dPlace, #tx3dList)    — unchanged
2  Waveform              (#wf3dBand)                            — dead radios removed
3  Receiver              (#rx3dType, #rx3dPlace, #rx3dList)     — dead checkbox removed
4  Engine                (#engine3dTier, #engine3dNote,         — NEW
                          #engine3dTiming)
5  Run                   (#run3dGo, #anim3dControls)            — one context-aware button
6  Visualize             (#viz3dMode + animwave option)         — single source of truth
```

### Files touched
- `Frontend_Data_Display.html` — rail markup (remove 2 dead controls, replace Run buttons
  with `#run3dGo`, add Engine section, add `animwave` Visualize option).
- `Frontend/simulator/simulation3d.js` — `forcedTier`/`lastSolve`, timing, tier gating in
  the solve paths, `runVizMode` `animwave` branch + animated-controls visibility, `#run3dGo`
  label + wiring, `__sim3d` exposure.
- `Frontend/simulator/simulator.css` — minor styling for `#engine3dTiming` (can reuse
  `.sim3d-modenote`); no new layout.

No physics-engine (Python) changes. No new assets. No change to the volume format or cache.

### Test plan (browser, served over http)
1. Serve repo root (`python3 -m http.server 8777`), open `Frontend_Data_Display.html`,
   go Indoor → 3D → workspace → **Simulation**.
2. **Unification:** confirm only one Run button; its label tracks the Visualize property;
   `animwave` appears in Visualize and plays; the Static/Time-lapse radios and the
   Interference checkbox are gone; anim controls appear only for `sweep`/`mechlapse`/
   `animwave` and disappear for static views.
3. **Engine selector:** for **coverage**, cycle `auto`/`cache`/`analytic` and verify the
   status line + `#engine3dTiming` change accordingly (cached → *full-physics solve*;
   analytic → *Motley-Keenan*), each with a plausible ms.
4. **Surrogate:** select `surrogate`; verify it reports *unavailable (not trained yet, M4)*
   and falls through to analytic with the reason shown.
5. **Edge cases:** `analytic` + a per-mechanism view → honest "no analytic equivalent"
   message; `analytic` in **O2I** → honest "unavailable for O2I" message; **Vacuum**
   coverage still shows the invariant gate.
6. Capture a short screen recording of (2)+(3) for the PR.

### Acceptance criteria
- One Run control, label reflects the selected Visualize property; kinematic wave reachable
  from Visualize; two dead controls removed; anim controls scoped to animated views.
- Engine tier is user-selectable; forcing a tier changes what renders and the status text;
  `#engine3dTiming` shows tier + milliseconds for every solve.
- Surrogate/analytic/cache edge cases produce clear, honest status messages (never a silent
  wrong-tier substitution).
- No regression: Tx placement, band change, mode change, sweep/mechlapse playback, and the
  vacuum invariant gate all still work.

### Risk
Low. Confined to one JS module + rail markup + a few CSS lines; no Python, assets, or data
format changes. Headless WebGL software fallback already renders here (see `AGENTS.md`),
and the surrogate staying unavailable is the expected path.

---

## Backlog (prioritised, after Slice 1)

Ordered by fit to the four pillars. Each notes the WinProp/WRAP source, the Seybold
chapter that backs the physics, and whether the analysis already exists (mostly wiring) or
is new.

2. **Measurement-validation workspace (pillar D).** Import PCTEL/G-flex CSV, snap to the
   grid, overlay sim-vs-measured, live **RMSE / Spearman ρ / bias** per metric, and a
   **drive/walk-test trajectory** that animates a receiver along a path with Rx-power vs
   distance. *WinProp virtual drive test + WRAP compare-coverages.* Seybold Ch. 8–9,
   App. A. Analysis exists (`PLAN §6`, M5 `validate_scanner_3d.py`); this is its UI. Also
   retires the hardcoded `Affine RMSE: 0.00px` pill.
3. **Link / Fresnel-zone profile viewer (pillar A).** Pick Tx + Rx → vertical-plane cut
   with 1st/2nd Fresnel ellipsoids, LOS clearance, obstacle diffraction. *WRAP profile
   viewer.* Seybold §8.2.2 / §8.2.4. New view; geometry + eikonal `T` already computed.
4. **Materials editor (pillar A).** Make `#matSummary` editable: per-class εr, σ, thickness,
   loss; furniture/vegetation as lossy volumes; re-solve (the cache key already hashes
   materials). *WinProp WallMan.* Seybold Ch. 2.
5. **Multi-Tx network planning (stage two).** Per-Tx list (power/band/antenna/enable),
   best-server + SINR + superposed coverage, floor selector, repeaters. *WinProp indoor
   AP planning / heterogeneous nets.* Seybold Ch. 9. Today 3D uses `firstTx()` only.
6. **Antenna editor (pillar A).** Gain/beamwidth/tilt/azimuth/polarization; Rx pattern;
   pattern import (`.ffe`/`.msi`). *Feko→WinProp pattern import.* Seybold Ch. 3.
7. **Link-budget + KPI layer.** EIRP→cables→margin→SNR→throughput, MIMO order, reliability,
   coverage-% / CDF panel, A/B compare of two solves. *WinProp network planning + WRAP
   compare.* Seybold Ch. 4 / App. A.
8. **RF-exposure/safety overlay.** Threshold field vs FCC/ICNIRP limit, highlight
   exceedance. *WinProp EM-exposure app.* Seybold Ch. 12.
9. **Scenario chooser redesign** using WinProp's *Map data / Optional data / Model* matrix;
   **export everywhere** (PNG/CSV/GeoTIFF, report); **spectrum/interference** hookup to the
   simulator Tx set. Seybold Ch. 4/7.

Out of core scope (note only): FMCW **radar/ADAS** (Ch. 5), **satellite** (Ch. 11),
country-scale WRAP spectrum management.
