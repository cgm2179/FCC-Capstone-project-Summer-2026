# Ch 6 — Atmospheric Effects

**Verdict: Skip (indoor)** · outdoor / long-range only.

## Sections
- 6.2 Atmospheric Refraction — **skip indoors**. Revisit *only* if the outdoor city track
  goes long-haul: 6.2.1 radio horizon, 6.2.2 equivalent (4/3) earth radius, 6.2.3 ducting,
  6.2.4 atmospheric multipath.
- 6.3 Atmospheric Attenuation — **skip** (gaseous absorption; matters >10 GHz over km).
- 6.4 Loss from Moisture/Precipitation — **skip** (see Ch 10).

## Engine relevance
None for indoor. Distances are tens of meters — the atmosphere is effectively lossless
and non-refractive at that scale. Bookmark 6.2.1–6.2.2 (radio horizon, effective earth
radius) if you ever push the outdoor voxelizer to multi-km links.

## Reading progress — ✅ read (2026-07-31, skip indoor / future outdoor)
Negligible indoors; captured for the **outdoor city track** (long links) and **mmWave** absorption.

## Equations encoded → [`../RF_Equations.ipynb`](../RF_Equations.ipynb) §13
- Refraction / radio horizon: `refractivity()` (eq 6.4), `refractivity_gradient()`, `k_factor()`
  (eq 6.11), `radio_horizon_km()`. *Verified Ex 6.1 (N 394.6, k 1.370, horizon 29.5 km).*
  Ducting threshold dn/dh = −157×10⁻⁶/km (k → ∞).
- Gaseous absorption: `atmospheric_loss_db()` (`A = γd`, ×2 for radar). Lines: 22 GHz (H₂O),
  **60 GHz (O₂, ~15 dB/km)**; ~0.05 dB/km @1 GHz. Fog/clouds & ITU multipath noted, not encoded.
