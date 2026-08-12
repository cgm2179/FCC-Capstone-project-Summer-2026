# SIM V1.5 3D — the parameterized default path-loss engine

A **default physics engine for path loss** where every knob is explicit, in one config,
with sensible defaults. It reuses SIM V1 3D's ray-marched geometry (`SceneV3`) and makes
the *spreading* model **selectable by propagation mode** (indoor / outdoor), then adds the
link budget on top. The Triplanar UNet caches the expensive field; the link budget is a
cheap per-query add.

## The nine parameters, in two layers

The parameters split into two layers that behave differently — this is the whole design.

**Layer 1 — the PL(x,y,z) field** (what the physics engine computes and the surrogate caches):

| symbol | field | meaning |
|---|---|---|
| `n` | `ple_n` | path-loss exponent (near slope for dual-slope) |
| `σ` | `shadow_sigma_db` | shadow-fading std dev (enters as the fade margin, not the field) |
| `PL(d0)` | `pl_d0_db` | close-in reference (`None` → FSPL at `d0`, per band) |
| `(α,β,γ)` | `fi_alpha_db / fi_beta / fi_gamma` | floating-intercept (ABG) triple |
| `dbp` | `breakpoint_m` | dual-slope breakpoint (`None` → `4·ht·hr·f/c`) |
| `d0` | `d0_m` | reference distance |

**Layer 2 — the link budget** (`RSRP = EIRP − Lc − PL − M`, applied *after* PL):

| symbol | field | meaning |
|---|---|---|
| `EIRP = Pt+Gt` | `tx_power_dbm + tx_gain_dbi` | effective isotropic radiated power |
| `Lc` | `cable_loss_db` | cable / implementation loss |
| `M = zσ·σ` | `fade_margin_db` (derived) | fade margin; `zσ` from `reliability` (80/90/95 → 0.84/1.28/1.65) |

## Mode-selectable spreading

`PathLossConfig.indoor()` / `.outdoor()` pick a model and defaults:

- **indoor → `close_in`** — `PL(d0) + 10·n·log10(d/d0)` (log-distance / Motley-Keenan, high PLE).
- **outdoor → `dual_slope`** — slope `n` up to `dbp`, steeper `ple_far` beyond it (street-canyon).
- **`floating_intercept`** (ABG) — available in either mode as a calibration model.

`select_model(cfg)` resolves the explicit `cfg.model`, else the mode default.

## Modules

| file | role |
|---|---|
| `pathloss_config.py` | `PathLossConfig` — all 9 params, mode presets, JSON round-trip, derived EIRP / margin |
| `pathloss_models.py` | `close_in` / `floating_intercept` / `dual_slope` + `select_model` |
| `link_budget.py` | `received_power_dbm` / `rsrp_dbm` / `coverage` (Layer 2) |
| `engine.py` | `SceneV15` — wraps `SceneV3`; swaps its fixed spreading for the mode-selected model |
| `pathloss_config.json` | serialized indoor + outdoor presets (the single source of truth) |

## Usage

```python
import sys; sys.path.insert(0, "SIM V1.5 3D")
from engine import load_scenev15
from pathloss_config import PathLossConfig

# indoor default, reusing the SIM V1 3D scene geometry
scene15, man = load_scenev15("SIM V1 3D", mode="indoor")
pl = scene15.pathloss_maps((131, 8, 66))          # (nbands, X, Y, Z) dB
rsrp = scene15.rsrp_maps((131, 8, 66))            # EIRP − Lc − PL − margin

# outdoor, dual-slope, custom link budget
cfg = PathLossConfig.outdoor(breakpoint_m=30.0, tx_power_dbm=43, tx_gain_dbi=15,
                             cable_loss_db=2, reliability="95")
scene15o = load_scenev15("SIM V1 3D", cfg=cfg)[0]
cov = scene15o.coverage((131, 8, 66), 3500.0, threshold_dbm=-105, mask=man_inside)
```

## Reuse & parity

`SceneV15` reuses `SceneV3.crossing_loss` + `sat_obs` verbatim (the geometry is
identical) and only replaces the spreading term. **Indoor / `close_in` with
`ple_n = manifest.n_exp` reproduces `SceneV3.pathloss_maps` to 0.01 dB** — the parametric
layer is a strict generalization, so nothing regresses; outdoor selects dual-slope instead.

## How the surrogate fits

The Triplanar UNet (SIM V1 3D) caches **Layer 1** — the deterministic mean PL field this
engine produces. **Layer 2** (EIRP / Lc / fade margin) is analytic and applied to the
cached PL at query time, so changing tx power, reliability, or cable loss never requires a
re-solve or a retrain. Shadow `σ` is stochastic, so it is a *margin*, not a learned field.

## Tests

`pytest "SIM V1.5 3D/tests3d/test_pathloss_v15.py" -q` → 10 passed (config, all 3 models,
mode selection, link budget). Engine parity vs SceneV3 verified against the real scene.

`dataset/`, `checkpoints/`, `preview/` are the standard SIM scaffold for a V1.5 dataset +
surrogate run.
