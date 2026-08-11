# Ch 5 — Radar Systems

**Verdict: Skip** (mostly) · not your use case.

## Sections
- 5.2 Radar Range Equation — **skip** (two-way `1/R⁴`; not propagation-mapping).
- 5.3 Radar Measurements (range/Doppler/angle/signature) — **skip**.
- 5.4 Clutter — **optional skim only if you build the scattering effect**. 5.4.1 area
  clutter, 5.4.3 clutter statistics touch on rough-surface backscatter, which overlaps
  loosely with diffuse scattering off walls/ground.
- 5.5 Atmospheric Impairments — skip (see Ch 6).

## Engine relevance
Near zero. The only transferable idea is clutter/backscatter *statistics* informing how
rough surfaces scatter — and Ch 8.2.1 (surface roughness) covers that more directly for
your one-way problem.

## Reading progress — ✅ read (2026-07-31, skip-tier / future-feature)
Captured for a possible future **radar mode** or **scattering-reflectivity** feature; not on
the indoor one-way path.

## Equations encoded → [`../RF_Equations.ipynb`](../RF_Equations.ipynb) §12
- `radar_rx_power_dbw()` (eq 5.8), `radar_max_range_m()` (eq 5.9) — two-way ⇒ **1/R⁴**.
  *Verified Ex 5.1 (SNR 6.5 dB), Ex 5.2 (42.2 dB).*
- `rcs_sphere()`, `rcs_flat_plate()` — RCS as *electrical* reflectivity (Table 5.1).
- Clutter σ⁰ / η noted as the diffuse-scatter analog (ties to Ch 8 roughness).
