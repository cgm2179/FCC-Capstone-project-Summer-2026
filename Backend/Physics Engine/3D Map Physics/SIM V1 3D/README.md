# SIM V1 3D — Volumetric RF Path-Loss Pipeline

A **true 3-D** propagation pipeline: it computes loss *through the real geometry of the OBJ* →
`PL(x, y, z)` plus a true eikonal arrival time `T(x, y, z)`, splits that into per-mechanism complex
fields, caches the result in the browser's volume format, and (M4) trains an ML surrogate so a
volume can be redrawn in one forward pass. This is the volumetric fork the 2-D skill spec documents
but defers — distinct from:

- **`SIM/`** — the deployed **2-D** pipeline (`PL(x,y)` on the floor grid), whose EM kernels
  (`physics_v2.py`) this stack imports verbatim rather than reimplementing, and
- **`viewer3d.js buildTerrain()`** — which renders a 2-D field as a 3-D *terrain* surface
  ("loss = height"). That is a render trick, not 3-D physics.

Roadmap and milestone status: **[PLAN_3D_SIM.md](PLAN_3D_SIM.md)**.
Physics scope and what is deliberately not modelled: **[MODEL_CARD_3D.md](MODEL_CARD_3D.md)**.

## Pipeline

| Stage | File | 2-D analog | Runs |
|---|---|---|---|
| A · voxelize | `voxelize.py` + `sandbox_material_map.py` | `phase_a.py` | local |
| — engine | `engine_3d.py` (`SceneV3`) | `engine_v2.py` | local |
| — mechanisms | `../Wave Behavior/Enivronmental Interaction/*_3D.py` | — | local / GPU |
| — combine | `Combine_3D.py` | — | local |
| — modes | `modes_3d.py` | `phase_a.bs_maps` | local |
| — one calc | `run_one_calc.py` | — | local |
| — export / cache | `export_pl_volume.py`, `precompute_volumes.py`, `cache_index.py` | `export_web_assets.py` | local / Colab |
| B · dataset | `phase_b3_dataset.ipynb` + `dataset_3d.py` | `phase_b_dataset.py` | Colab CPU |
| C · train | `phase_c3_train_colab.ipynb` | `phase_c_train_colab_v3.ipynb` | Colab GPU |
| config | `manifest_3d.json` | `manifest.json` | — |

## Quickstart (local)

```bash
pip install -r "SIM V1 3D/requirements_3d.txt"

python3 TESTS3D/run_all.py --full            # 129 tests
python3 TESTS3D/run_all.py --selftest        # + each module's own --test

python3 "SIM V1 3D/modes_3d.py"              # the four modes + availability
python3 "SIM V1 3D/cache_index.py" --stats   # what is cached
python3 "SIM V1 3D/cache_index.py" --verify  # are the .bin files intact
python3 "SIM V1 3D/dataset_3d.py" --budget   # how many Tx each mask can hold
```

Solve one transmitter (~27 s for all four mechanisms; the first diffraction run builds a relay
cache, ~74 s once):

```bash
python3 "SIM V1 3D/export_pl_volume.py" --tx 100 5 60 \
  --mechanisms path_loss,reflection,diffraction,scattering --mech-bands 2442,3500,5500
```

To browse the result, serve over HTTP and open `Frontend_Data_Display.html` — see
[`QUICKSTART.md`](../../../QUICKSTART.md). `file://` silently breaks every volume fetch.

## Colab — dataset → train → export (M4)

### What a clone already includes (gitignore does **not** strip these)
`material_grid.npy`, `inside_mask.npy`, `valid_tx_mask.npy`, `manifest_3d.json`,
both notebooks, `dataset_3d.py` / `engine_3d.py`, `web/pl_unet3d.json` + smoke
`web/pl_unet3d.onnx` (allow-listed), outdoor `city_demo/`, and cached `web/volumes/`.

Ignored on purpose (regenerate / keep on Drive): full `city/`, `cache/`,
`dataset/shard_*.npy`, `*.pt` checkpoints. After Colab, copy the trained
`web/pl_unet3d.onnx` + `pl_unet3d.json` back — git will accept that ONNX path.

### Drive layout (pick one)

**Recommended — whole repo on Drive** (paths match a normal clone / GitHub zip):
```text
MyDrive/indoor-walk-test-main/     ← GitHub “Download ZIP” or git clone
ROOT = "/content/drive/MyDrive/indoor-walk-test-main/Physics Engine/3D Map Physics/SIM V1 3D"
```
Confirm this file exists on Drive before running cells:
`…/indoor-walk-test-main/Physics Engine/2D/SIM/physics_v2.py`
(If it’s missing, the zip upload was incomplete — re-copy that folder, or place
`physics_v2.py` directly inside `SIM V1 3D/`.)

**Minimal — folder only:** upload `SIM V1 3D/` and also drop
`Physics Engine/2D/SIM/physics_v2.py` **into that same folder** (Colab fallback
search in `physics_3d.py`). For mechanism exports you also need the sibling
`Wave Behavior/` tree beside `SIM V1 3D/`.

### Run order

**B — `phase_b3_dataset.ipynb`** (CPU runtime; `SceneV3` is NumPy)

Set `ROOT`, leave `SMOKE = True` for a 16-position check, then set it `False` and Run All.
Writes resumable `dataset/shard_NNN_{pl,tau}.npy` + `_meta.npz`; re-run the generate cell to
resume after a disconnect. Roughly 4 s per position, so 1,000 positions is about an hour.

Preflight should print `preflight OK` (M2 `satObs` keeps clip fraction under 35%).
Do **not** raise `CLIP_LIMIT` if it fails — fix the scene / `use_satobs` instead.

**C — `phase_c3_train_colab.ipynb`** (GPU runtime — A100 preferred)

Set the same `ROOT`, Run All. Trains the 3-D UNet with FSPL-floor, causality, band-ordering and
reciprocity constraints; exports `web/pl_unet3d.onnx` behind a ≤ 0.1 dB parity gate plus
`web/pl_unet3d.json`. Resumable — checkpoints every 2 epochs; `MAX_HOURS` stops cleanly.

Copy both `web/pl_unet3d.*` files back into the repo (and optional
`checkpoints/train_report.json`), commit, push. Browser status then shows a real
`DL surrogate` instead of the smoke model.

Both notebooks import their tensor layout from `dataset_3d.py`. Do not redefine normalization,
channel order or the blob sigma anywhere else: a mismatch between trainer and browser does not
raise, it renders a confident wrong field.

## Status

**Built and verified** — `python3 TESTS3D/run_all.py --full` → 129 passed

- `voxelize.py` — OBJ → `262×17×132` @ 0.30 m from the 253-material mesh; ceiling sealed, floor
  slab concrete, ~40 people/prop materials excluded.
- `engine_3d.py` — per-crossing Fresnel/Airy transmission through the `physics_v2` `CrossingLUT`
  plus a real 3-D eikonal arrival time (`skfmm`). Not flat-dB Motley-Keenan, and not `d/c`.
- Four of six mechanisms: `Path_Loss_3D`, `Reflection_3D`, `Diffraction_3D`, `Scattering_3D`,
  combined by `Combine_3D` (coherent `E` + incoherent `p_incoh`, bandwidth-averaged).
- Four propagation modes (`modes_3d.py`), content-addressed volume cache (`cache_index.py`),
  mechanism channels and the browser time-lapse.
- `dataset_3d.py` — the surrogate input contract, with its own `--test`.

**Scaffolded, needs a Colab run**

- `phase_b3_dataset.ipynb` / `phase_c3_train_colab.ipynb` — both executed end to end locally at
  smoke scale (dataset → train → ONNX export at 0.0000 dB parity → contract written). No full run
  yet, so there is no trained model and the browser's surrogate tier reports its own absence.

**Not built** — see [PLAN_3D_SIM.md](PLAN_3D_SIM.md)

- `Refraction_3D.py`, `Absorption_3D.py` (0 bytes; both effects are inside the engine already,
  just not exposed as viewable channels).
- `Construct_Reciever_3D.py` (0 bytes) → no PL → RSRP/RSRQ/SINR, which blocks M5 validation.
- `validate_scanner_3d.py`, `Frontend/2d-3d/georef.js`, browser-side surrogate inference.
- The outdoor city grid — code ships, grid is gitignored; outdoor mode reports itself unavailable
  until `voxelize_city.py --mode 2.5d --cell 1` is run.
