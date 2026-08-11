# Ch 8 — Fading & Multipath Characterization

**★ Verdict: Essential** · **Pipeline tier 2** · the source of your diffraction model.

## Sections
- 8.2 Ground-Bounce Multipath — **read** (two-ray; first phase/interference nulls).
- 8.2.1 Surface Roughness — **read**. Rayleigh criterion → specular vs diffuse; feeds the
  scattering effect.
- **8.2.2 Fresnel Zones — read.** Clearance test for "is this path really blocked?"
- **8.2.3 Diffraction & Huygens' Principle — read.** Why energy bends into shadow.
- **8.2.4 Quantifying Diffraction Loss — read closely.** Knife-edge loss `J(v)` → your
  diffraction effect (single-edge now, UTD for multi-edge later).
- 8.3 Large-Scale / Log-Normal Fading — **read** (shadowing `X_σ`).
- 8.4 Small-Scale Fading (delay spread, Doppler, channel models, statistics) — **skim**;
  advanced / validation, not core to a static coverage map.

## Engine relevance
Directly builds the **diffraction** effect and the **Fresnel-zone clearance** test that
decides when to invoke it; 8.2.1 feeds **scattering**; 8.3 gives the `X_σ` shadowing term
you can layer on any path-loss model.

## Reading progress — ✅ complete (2026-08-01)
- ✅ **Part 1 §8.1–8.2** — ground-bounce (two-ray), Rayleigh roughness, Fresnel zones, knife-edge
  diffraction. **Verified the blind-seeded `fresnel_zone_radius` & `knife_edge_v` vs Ex 8.2 & 8.3.**
- ✅ **Part 2 §8.3–8.5** — log-normal shadowing (`X_σ`), small-scale fading (Rayleigh/Ricean,
  delay & Doppler spread). *Verified Ex 8.5 (7.7/10.25 dB margin), 8.6, 8.7, 8.8 (0.031).*

## §16 encoded → [`../RF_Equations.ipynb`](../RF_Equations.ipynb)
- **Shadowing:** `shadowing_margin_db()` (z·σ_L), `edge_coverage_prob()` (Φ), `location_variability_okumura()`
  — self-contained normal CDF + Acklam Φ⁻¹.
- **Small-scale:** `rayleigh_fade_prob()`, `ricean_fade_prob()` (numerical; consistent w/ Rayleigh,
  book Ex 8.9's 0.018 doesn't reproduce — convention-dependent), `coherence_bandwidth_hz()`,
  `doppler_shift_hz()`, `coherence_time_s()`.

## Equations encoded → [`../RF_Equations.ipynb`](../RF_Equations.ipynb) §7, §8, §15
- §15 **two-ray:** `two_ray_pathloss_db()` (1/d⁴ vs FSL, eq 8.14), `two_ray_crossover_m()` (eq 8.15),
  `two_ray_phase_diff()` (eq 8.8). *Verified Ex 8.1 + Fig 8.5 crossover (12.6 km).*
- §15 **roughness:** `rayleigh_roughness_m()` (eq 8.16), `is_specular()` — the reflection/scatter trigger.
- §7 `fresnel_zone_radius()` (eq 8.20) 🟢; §8 `knife_edge_v()` (eq 8.19) + `knife_edge_loss_db()` (ITU) 🟢;
  §15 `knife_edge_loss_lee_db()` (Lee eqs 8.21). *All verified vs Ex 8.2 / 8.3.*
- ⬜ Log-normal `X_σ` (Part 2; `log_distance_pl()` already carries σ).

## Engine hooks
`diffraction_3D` (UTD; knife-edge = simple-geometry proxy), `scattering_3D` (Rayleigh trigger),
reflection (two-ray floor/ceiling bounce). 60% first-Fresnel-zone clearance = "is it really blocked?"
