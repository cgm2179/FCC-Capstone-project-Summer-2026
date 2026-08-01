# Ch 11 — Satellite Communications

**Verdict: Skip** · not applicable to indoor or terrestrial-city propagation.

## Sections
- 11.4 Satellite Path Free-Space Loss — already covered by FSPL (Ch 4).
- 11.5–11.7 Atmospheric / ionospheric / rain fades on slant paths — **skip**.
- 11.9 Noise Temperature — **skip** (link-budget detail for space links).

## Engine relevance
Mostly none — different geometry (single long slant path). **One idea transfers:** the hot-pad
formula shows an absorptive loss both attenuates the signal *and* raises the noise floor — so a
rain fade (or a lossy wall/foliage) hits SNR ~twice. A rigorous indoor SNR-level model would apply it.

## Reading progress — ✅ read (2026-08-01, skip-tier / final chapter)
Processed as the closing capstone; reuses FSPL + rain (§19) + noise (§10).

## §20 encoded → [`../RF_Equations.ipynb`](../RF_Equations.ipynb)
- **Geometry:** `sat_central_angle()`, `sat_slant_range_km()`, `sat_elevation_deg()` (eqs 11.4-11.6).
  *Verified Ex 11.1 → 36,314 km, 66.6°.*
- **ITU satellite rain:** `itu_sat_rain_atten_db()` (P.618 10-step). *Verified Ex 11.3 uplink → 38.5 / 14.8 dB.*
- **Noise:** `hotpad_noise_temp()` (eq 11.50), `rain_noise_temp()` (eq 11.51), `gt_db()`.
  *Verified Ex 11.5/11.6 → 222.5 K / 204.4 K.* ⚠ Book slips: Ex 11.5 "9.9 dB" → 11.9; "G/T 14.3 dB" is the ratio, not dB.
- Data sources: ITU-R P.618 (sat rain/atmos), P.839 (rain height).

## ✅ Completes the textbook — Ch 1–11 encoded & verified; Ch 12 (RF Safety) out of scope.
