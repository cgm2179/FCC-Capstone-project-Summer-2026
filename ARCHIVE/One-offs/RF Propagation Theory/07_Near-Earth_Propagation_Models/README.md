# Ch 7 — Near-Earth Propagation Models

**Verdict: Track-dependent** · **Pipeline tier 0 (empirical)** · the baseline for the
**outdoor city / voxelizer track**, not the indoor engine.

## Sections
- 7.2 Foliage Models (Weissberger, ITU vegetation) — **optional**; only if you model trees.
- 7.3 Terrain Modeling — **skim**. 7.3.1 Egli, 7.3.2 Longley–Rice (the classic
  irregular-terrain model), 7.3.3 ITU.
- **7.4 Propagation in Built-Up Areas — read for the outdoor track.** 7.4.2 Okumura,
  **7.4.3 Hata**, **7.4.4 COST-231** (Hata extended to 2 GHz), 7.4.5 Lee.
- 7.4.6 Comparison — **read**; tells you which urban model to trust where.

## Engine relevance
These are *empirical* macro models — regression fits, not physics. They're the cheap
baseline and the **validation reference** for your OSM-voxelized outdoor sim: run Hata /
COST-231 for a link and check your physics engine lands in the same ballpark. Not a
substitute for the GO/eikonal engine indoors.

## Reading progress — ✅ complete (2026-07-31) — the outdoor-track baseline
The empirical baseline + validation reference for the outdoor OSM-voxelized sim. All are
*median* path-loss fits, not physics — the cheap Tier-0 layer, not a GO/eikonal substitute.

## Equations encoded → [`../RF_Equations.ipynb`](../RF_Equations.ipynb) §14
- **Foliage:** `weissberger_foliage_db()` (eq 7.1), `itu_foliage_db()` (eq 7.2). *Ex 7.1 → 5.40 / 7.06 dB.*
- **Terrain:** `egli_pl_db()` (eq 7.9), `itu_terrain_diffraction_db()` + `fresnel_radius_terrain_m()`
  (= Ch 8 Fresnel radius). *Ex 7.3 → F1 47.4, Ad 9.7 dB.*
- **⚡ Urban macro-models:** `hata_pl_db()` (urban/suburban/open, eq 7.14), `cost231_pl_db()`
  (PCS 1500–2000 MHz, eq 7.19), `lee_pl_db()` + `LEE_PARAMS` (fittable, eq 7.20). *Ex 7.5 → Hata 137.1 dB.*
- ⚠ *Book slip #4 (Egli Ex 7.2): prints 112.4 dB but used hb·hm = 6 instead of 60; correct = 92.4 dB.*

## Model choice (Table 7.3)
Hata/COST-231 = mobile-telephony standard (base above rooftops). Lee = fit to local measurements.
Okumura = graphical parent of Hata. Longley–Rice = detailed terrain software (NTIA). Young = NYC power law.
