# Ch 2 — Electromagnetics & RF Propagation

**★ Verdict: Essential** · **Pipeline tier 1** · the physics floor for
reflection / refraction / absorption.

## Sections
- 2.2 Electric Field — **read**. 2.2.1 permittivity, 2.2.2 conductivity = the material
  constants every effect consumes.
- 2.3 Magnetic Field — **skim**.
- 2.4 Electromagnetic Waves — **read**. 2.4.1 perfect dielectric, 2.4.2 lossy dielectric,
  2.4.3 conductor = how α (attenuation) and β (phase) change with the medium.
- 2.5 Wave Polarization — **read** (feeds antenna/PLF later).
- **2.6 Propagation at Material Boundaries — read closely. This is the payoff.**
  2.6.1 dielectric-dielectric, 2.6.2 dielectric-conductor, 2.6.3 dielectric-lossy =
  the Fresnel reflection/transmission coefficients.
- 2.7 Impairment, 2.8 Circular-pol ground effects — **skim**.

## Engine relevance
Γ(θ, material) from 2.6 is *exactly* the per-surface reflection coefficient the
reflection & refraction effects need; `T = 1 − |Γ|²` is the pass-through term the
absorption effect needs; the loss tangent from 2.2/2.4 sets how lossy a wall is.

## Reading progress — ✅ complete (2026-07-31)
- ✅ **Part 1 §2.1–2.3** static fields & materials (ε, σ, μr = 1 ⇒ n = √εr).
- ✅ **Part 2 §2.4–2.5** waves in matter (v, Z0, complex ε, α/β, skin depth) & polarization.
- ✅ **Part 3 §2.6–2.8** Fresnel boundaries, impairment taxonomy, polarization loss.

## §2.7 impairment taxonomy = your effect modules
| §2.7 phenomenon | Engine `_3D.py` | Physics |
|---|---|---|
| Reflection | reflection | Fresnel Γ |
| Refraction | refraction | Snell + τ |
| Absorption | absorption | α / complex ε |
| Scattering | scattering | Rayleigh (Ch 8) |
| Diffraction | diffraction | knife-edge / UTD (Ch 8) |
| Depolarization | — (cross-cutting) | PLF §2.8 |

**Judgment call the section forces:** exact lossy-boundary Fresnel is often impractical (rough
surfaces, poorly-known materials) — the book itself defaults to empirical data, naming *RF
modeling software* (= this engine) and *glass panes* as the exceptions. So material-data
quality (→ ITU‑R P.2040) is the binding constraint, not the math.

## Captured this pass
- **Material data:** [`02_Material_Properties.md`](02_Material_Properties.md) — Tables 2.2/2.3,
  (εr, σ) pairs, ITU‑R P.2040 pointer.
- **Notes:** [`../RF_Propagation_Notes.md`](../RF_Propagation_Notes.md) Ch 2 entry (Parts 1–3, full).

## Equations encoded → [`../RF_Equations.ipynb`](../RF_Equations.ipynb)
- §4 `refractive_index()`, `MATERIALS{}`, `efield_refraction_angle()` (**NOT** Snell) 🟢
- §5 `complex_permittivity()`, `loss_tangent()`, `phase_velocity()`, `intrinsic_impedance()`,
  `loss_regime()`, `wave_params_lossy()` (α, β), `skin_depth()` 🟢
- §6 `fresnel_coeffs()` (Γ_TE/TM), `brewster_grazing_deg()` / `brewster_normal_deg()` — with a
  verified grazing↔normal Brewster cross-check (23.6° + 66.4° = 90°) 🟢
- ⬜ `plf()` polarization-loss (only if the engine goes polarization-aware; with Ch 3.5.2)
