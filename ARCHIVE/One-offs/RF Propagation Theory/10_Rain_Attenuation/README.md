# Ch 10 — Rain Attenuation of Microwave & Millimeter Wave Signals

**Verdict: Skip** · mmWave / outdoor only; irrelevant indoors.

## Sections
- 10.3.1 Specific Attenuation `γ = k·Rᵅ` — reference if you ever go outdoor mmWave.
- 10.3.2 ITU Model, 10.3.3 Crane Global Model — **skip**.
- Everything else (slant paths, availability, cross-pol) — **skip**.

## Engine relevance
None for indoor sub-6 GHz. Rain matters only for outdoor links roughly >10 GHz over
hundreds of meters+. Bookmark the `γ = k·Rᵅ` specific-attenuation form (App 10A has the
k, α tables) if the outdoor track ever targets mmWave.

## Reading progress — ✅ read (2026-08-01, skip-tier / future outdoor mmWave)
Captured for a future outdoor mmWave link-budget feature; irrelevant to the indoor sub-6 GHz core.

## §19 encoded → [`../RF_Equations.ipynb`](../RF_Equations.ipynb)
- `rain_coeff_interp()`, `rain_coeffs()` (polarization combine, eqs 10.3-10.4), `rain_specific_attenuation()`
  (`γ = k·RR^α`). *Verified 38.6 GHz → 0.324/0.95; horizontal rains worse than vertical.*
- `itu_rain_attenuation_db()` (eqs 10.5-10.7), `itu_availability_adjust()` (eqs 10.8/10.9), `ITU_RAIN_RATE_001`
  (Table 10.2). *Verified Ex 10.1 → 23.8 / 34.3 dB.* Valid to 40 GHz / 60 km.
- `fog_attenuation_db()` (eq 10.15). *Verified Ex 10.4 → 4.7 dB.*
- ⚠ **Crane** eqs 10.11/10.13 garbled in OCR (couldn't reproduce Ex 10.2); only `crane_breakpoint_km()` + z
  shipped — use Crane 1996/2003. **Data sources: ITU‑R P.838** (coefficients) + **P.837** (rain rates).
