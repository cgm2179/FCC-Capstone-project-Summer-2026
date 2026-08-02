# SIM V1 3D — Model Card

**What it is.** A volumetric indoor RF path-loss model for the 7th floor:
`PL(x, y, z)` computed through a voxelized version of the OBJ, plus a UNet
surrogate that learns `(Tx, frequency) → PL` volume over the **fixed** geometry.
This is the 2-D skill spec's deferred *volumetric* fork, scoped pragmatically for
a first version.

## Domain & inputs (`manifest_3d.json`)

| | |
|---|---|
| Grid | `262 × 11 × 118` voxels (X, Y-up, Z), cubic **0.30 m** |
| Scale | `m_per_unit ≈ 0.018307`, from registering the OBJ X-span to the floor-plan width (1150 px × 0.0679 m/px, QGIS GCP calibration in `SIM/manifest.json`) → ceiling ≈ **2.85 m** |
| Interior | façade-footprint fill; room band iy≈3–9; ~45 % of volume |
| Materials | 6 classes (air, drywall 4 dB, concrete 15, core 20, furniture 0.3 dB/m, glass 3) |
| Bands | 2442 / 3500 / 5500 / 6125 MHz, per-band loss multiplier |
| Norm | PL → `(PL − 40)/130` clamped to [0,1] |

## Physics (`engine_3d.py` `SceneV3`)

Two co-registered volumes per transmitter, in true (x, y, z).

**Path loss** — along the Tx→voxel ray, summed per wall crossing:

```
PL = fspl_1m(f) + 10·n·log10(max(d, d0))              # n = 2 (Motley-Keenan spread), d0 = 1 m
   + Σ CrossingLUT( material, incidence θ, thickness ) # per-crossing Fresnel/Airy
                                                       # transmission + Im(q) absorption
```

The per-crossing term is the **validated `physics_v2` EM kernel** (`CrossingLUT` —
angle- and thickness-dependent Fresnel/Airy slab transmission plus bulk absorption),
reused verbatim through `physics_3d.py`; it is **not** the flat per-class dB table.
Incidence angle comes from voxel wall normals; thickness from the along-ray run
length scaled to construction thickness.

**Arrival time** — a real 3-D **eikonal** solve (`skfmm.travel_time`, `|∇T| = 1/speed`)
on the slowness field `Re(√ε_r)/c` with opaque classes masked. Refraction (bending,
in-wall lag) and diffraction (corner wraparound) fall out of the single fast-march via
Fermat — **not** geometric `T = d/c` (that survives only as `geometric_time()`, the
lower-bound sanity check).

The flat `materials[].loss_db` table and `physics.fallback_obstruction_db` in
`manifest_3d.json` are the **browser display / in-browser analytic fallback** model
(`Frontend/simulator/simulation3d.js`, `landing.js` preprocess), not this engine.

## Material mapping (OBJ → class)

Tunable in `manifest_3d.json` (`obj_material_map`). Owner should sanity-check:

| OBJ material | → class | note |
|---|---|---|
| `Glass_Basic_01` | exterior_glass (3 dB) | façade + interior glass |
| `FrontColor` | drywall (4 dB) | **bulk** incl. the floor slab (iy 1–2) & partitions |
| `Steel_Brushed_Stainless` | core (20 dB) | stand-in for metal (no 30 dB class in the 6-class scheme) |
| `Blacktop / Formica` | concrete (15 dB) | hard surfaces |
| `Ty_*` (incl. `Ty_Skin`) | furniture (0.3 dB/m) | furniture / people clutter |

## Fidelity ladder

**Shipped (this engine):**

- **Per-material Fresnel/slab (S3–S7)** — angle+thickness `CrossingLUT`, replacing
  the old flat per-class dB.
- **Eikonal arrival time (T3c)** — 3-D fast-march routes the front around barriers
  and bends/lags it through dielectrics, instead of punching straight through.

**Still deferred** (added by later-stage modules, power-summed onto the direct field):

- **3-D UTD diffraction (S8)** — without it, shadows behind the metal core read
  over-deep (energy that would bend in doesn't); the direct field alone can exceed
  500 dB in the deep core shadow until diffraction fills it.
- **Reflection & scattering** — specular image sources + diffuse scattering
  (`Wave Behavior/Enivronmental Interaction/{Reflection,Scattering}_3D.py`, still stubs).
- **Correlated shadow field (S11)** — spatially-correlated large-scale fading.
- **Floor/ceiling material split** — `FrontColor` lumps the floor slab with walls
  as drywall; splitting horizontal (concrete slab) from vertical (partition)
  faces by normal would be more accurate for near-vertical rays.
- **Generalization** — v1 is single-building, fixed geometry (= "fixed boundary
  conditions"); the surrogate conditions on Tx+freq only.

## Known limitations

- **Ray "spokes."** Discrete per-voxel ray sampling produces faint radial streaks
  (same as the 2-D engine). Physical (wall shadows), and the surrogate smooths
  them; not a bug.
- **Thin vertical axis.** One ~2.85 m floor is ~7 room voxels tall at 0.30 m, so
  vertical detail is coarse — honest for a single floor. Lower `--cell` for more.
- **Interior mask is a heuristic** (façade-footprint fill); it drives display and
  Tx sampling only, not the PL values (PL is computed for every voxel).
- **Colab stages unrun here.** The training notebook was authored but not
  executed locally (no GPU); treat RMSE targets as TBD from the first real run.

## Validation (`run_one_calc.py`, 3500 MHz, interior-centroid ceiling Tx)

- PL over interior: **43.3 / 146.1 / 540.0 dB** (min/median/max). The high tail is
  the uncapped Fresnel/Airy loss straight through the metal core — physical, and
  power-filled once diffraction (deferred) is added.
- Excess over free space ≥ 0 everywhere (walls only add loss). ✓
- Eikonal `T`: max **144 ns**, and `T − d/c ≥ 0` within grid tolerance (the front
  never beats the straight line). ✓
- `corr(PL, log-distance) = 0.57` (loss rises with range). ✓
- Visual: radial gradient, wall shadowing, glass-façade leakage, vertical
  variation (`preview/pl_one_calc.png`).

## Provenance

Source OBJ: `Data/models/Indoor 7th floor v2 First Render.obj/…`. Scale
calibration inherited from `SIM/manifest.json`. Physics constants and the
6-class material scheme mirror the deployed 2-D model.
