# 3D Simulation tab — bug/feature audit (PR #30)

A map of what the Simulation tab (`simulation3d.js`, ~2.8k lines) does and doesn't do, so the
follow-on Simulator PRs have a triage list. Written while fixing the "city renders blank" report.

## How it's built

- **3 propagation modes** (`mode3dSelect`): `vacuum` · `o2i` (indoor+outdoor plane wave) · `outdoor`
  (NoMa DC city). The built-in indoor 7th floor is the implicit default scene.
- **12 viz modes** (`viz3dMode`): field · radiation · interference · **coverage** · sweep · mechlapse ·
  path_loss · reflection · diffraction · scattering · refraction · absorption.
- **3-tier coverage solver** (`runStaticField`), reported in the status line:
  1. **cached volume** — full-physics PL/T volume from `precompute_volumes.py` (`SIM3D/web/volumes/`),
     matched to the Tx by `matchVolume` (mode + `tx_vox`).
  2. **DL surrogate** — `pl_unet3d.onnx` via onnxruntime-web.
  3. **analytic** — in-browser `marchPL` (Motley-Keenan multiwall + FSPL + satObs cap) — the same
     physics family as the validated `pathloss_*` work. The guaranteed floor; always available.
- Model import (`loadModelFile` → `voxelizeTriangles` → `setRuntimeScene`) runs on the analytic tier
  (a fresh grid can't match a cached volume).

## ✅ Works
Scene rendering (indoor CAD + outdoor city surface) · Tx/Rx placement (raycast) · band selection ·
indoor coverage (cached + analytic) · display controls (plane/height/EIRP/RSRP-classes/threshold) ·
mode switching · most viz modes for indoor · model import mechanics (load/voxelize/mount).

## 🐞 Was broken → fixed in PR #30
| symptom | root cause | fix |
|---|---|---|
| **City / imported coverage renders blank** | outdoor/runtime scenes have no Tx by default, so `runStaticField`'s source fell to scene-centre **inside the buildings** → `marchPL` all-loss → every sample dropped. And over `file://` the cached volume can't `fetch`, so there's no fallback. | `defaultRooftopTx()` — auto-place the source on the tallest roof near centre (a macro-cell) when nothing is placed on outdoor/imported scenes; status-noted. |
| **Coverage hidden behind the city** | outdoor buildings render ~0.9 opaque, occluding the street-level wash. | outdoor smooth surface capped to 0.35 opacity; prediction plane defaults to street level (~5% of scene height) instead of mid-air (45%). |
| **No Cellular option** | only `WIFI_BANDS` presets. | `CELLULAR_BANDS` (donor freqs 617–3710 MHz) + a **Coverage type: WiFi \| Cellular** selector that repopulates the band list. |
| **Cellular showed the 2.4 GHz city volume** | the tier used any Tx-matching cached volume regardless of frequency. | `volBandOk` — a cached volume is only used within ~25% of the requested frequency; else the analytic tier recomputes at the true freq. |
| **`refraction` / `absorption` silently did nothing** | `Refraction_3D.py` / `Absorption_3D.py` are stubs. | those two `viz3dMode` options are `disabled` + labelled "module stub". |

## ⏳ Still stubbed / deferred (roadmap for #31+)
- **Mechanism viz modes** (path_loss/reflection/diffraction/scattering) need a **cached volume** — they
  show an honest "nothing cached for this mode yet" for outdoor until a volume is precomputed for that Tx.
- **Cellular cached volumes** — only a 2.4 GHz outdoor volume exists (`tx_67-20-66`); cellular renders on
  the analytic tier. Precompute donor-band outdoor volumes (`precompute_volumes.py --mode outdoor`) to give
  cellular the full-physics tier too.
- **Georeferenced scene + real donors** — the outdoor scene is the old 129³ demo tile, not the georeferenced
  `NoMa_FCC_tile` (PR #29). #31 swaps it in and places the real base stations (lat/lon) as transmitters.
- **Validation overlay** — predicted-vs-measured RSRP (the 7/24 walk) is not shown in the Simulator yet.

## Notes
- **Serve over http** (`python3 -m http.server`): `file://` blocks `fetch`, so cached volumes + the volume
  index don't load — the analytic tier still works, which is why the default-rooftop-Tx fix matters most there.
- Debug handles: `window.__sim3d` exposes the scene, tier, volume index, and solver internals.
