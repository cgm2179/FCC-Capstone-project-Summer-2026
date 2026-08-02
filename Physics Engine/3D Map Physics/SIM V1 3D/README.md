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

Upload the whole `SIM V1 3D/` folder to Drive, plus the sibling `Physics Engine/2D/SIM/physics_v2.py`
that `physics_3d` imports. Then, in order:

**B — `phase_b3_dataset.ipynb`** (CPU runtime is correct; `SceneV3` is NumPy)

Set `ROOT`, leave `SMOKE = True` for a 16-position check, then set it `False` and Run All.
Writes resumable `dataset/shard_NNN_{pl,tau}.npy` + `_meta.npz`; re-run the generate cell to
resume after a disconnect. Roughly 4 s per position, so 1,000 positions is about an hour.

> **This notebook currently stops at its preflight gate, by design.** 79 % of interior voxels at
> 3500 MHz saturate the 170 dB normalization ceiling because `SceneV3.crossing_loss` has no
> saturating obstruction model. Training on that target teaches a network a flat plateau. Land
> Pre-M4 §P1 in [PLAN_3D_SIM.md](PLAN_3D_SIM.md) first; raising `CLIP_LIMIT` to get past the gate
> is the one change that guarantees a worthless model.

**C — `phase_c3_train_colab.ipynb`** (GPU runtime)

Set `ROOT`, Run All. Trains the 3-D UNet with FSPL-floor, causality, band-ordering and reciprocity
constraints; reports test RMSE/MAE in dB against FSPL and log-distance baselines; exports
`web/pl_unet3d.onnx` behind a ≤ 0.1 dB parity gate, plus the `web/pl_unet3d.json` input contract the
browser refuses to run without. Resumable — checkpoints to Drive every 2 epochs and honours a
`MAX_HOURS` budget, so re-running the training cell continues rather than restarts.

Both notebooks import their tensor layout from `dataset_3d.py`. Do not redefine normalization,
channel order or the blob sigma anywhere else: a mismatch between trainer and browser does not
raise, it renders a confident wrong field.

```bash
python3 "SIM V1 3D/export_web3.py"    # grid + masks + manifest -> web/sim_assets_3d.js
```

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
