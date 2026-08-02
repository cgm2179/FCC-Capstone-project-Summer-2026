# §2.2–2.3 Material Properties (εr, σ) — reference

The engine's material model = **(εr, σ) per surface**, because the book assumes **μr = 1**
(non-magnetic, §2.3). So the refractive index is `n = √εr`, and reflection/refraction (Fresnel,
§2.6) + absorption (complex ε, §2.4.2) need nothing beyond these two numbers.

- **εr** (relative permittivity / dielectric constant) — the *real* part; sets how much a
  boundary reflects/refracts. `ε = εr·ε₀`, `ε₀ = 8.854×10⁻¹² F/m`.
- **σ** (conductivity, S/m) — feeds the *imaginary* part of the complex permittivity; sets loss.
  `ρ = 1/σ` (resistivity). Perfect dielectric ⇒ σ = 0; real materials have both.
- **μ₀ = 4π×10⁻⁷ H/m**; μr = 1 assumed.

## Relative permittivity εr (Table 2.2)
| Material | εr | | Material | εr |
|---|---|---|---|---|
| Vacuum | 1 | | Quartz | 5 |
| Air | 1.0006 | | Lead glass | 6 |
| Polystyrene | 2.7 | | Mica | 6 |
| Rubber | 3 | | Distilled water | 81 |
| Bakelite | 5 | | | |

## Conductivity σ, S/m (Table 2.3)
| Conductors | σ | | Insulators | σ |
|---|---|---|---|---|
| Silver | 6.1×10⁷ | | Wet earth | ~10⁻³ |
| Copper | 5.7×10⁷ | | Distilled water | ~10⁻⁴ |
| Gold | 4.1×10⁷ | | Dry earth | ~10⁻⁵ |
| Aluminum | 3.5×10⁷ | | Rock | ~10⁻⁶ |
| Tungsten | 1.8×10⁷ | | Bakelite | ~10⁻⁹ |
| Brass | 1.1×10⁷ | | Glass | ~10⁻¹² |
| Iron | ~10⁷ | | Rubber | ~10⁻¹⁵ |
| Mercury | ~10⁶ | | Mica | ~10⁻¹⁵ |
| Seawater | 4 | | Quartz | ~10⁻¹⁷ |

*~24 orders of magnitude, metals → quartz. Metals (σ ≈ 10⁷) behave as near-perfect
conductors ⇒ near-total reflection — the σ→∞ limit of the dielectric-to-conductor boundary
(§2.6.2).*

## Complete (εr, σ) pairs the book gives — engine-ready
Only materials the book lists in **both** tables make a usable engine entry:

| Material | εr | σ (S/m) | n = √εr |
|---|---|---|---|
| Distilled water | 81 | ~10⁻⁴ | 9.00 |
| Lead glass | 6 | ~10⁻¹² | 2.45 |
| Mica | 6 | ~10⁻¹⁵ | 2.45 |
| Quartz | 5 | ~10⁻¹⁷ | 2.24 |
| Rubber | 3 | ~10⁻¹⁵ | 1.73 |

Encoded as `MATERIALS{}` in [`../RF_Equations.ipynb`](../RF_Equations.ipynb) §4.

## ⚠ Gap for the engine — building materials aren't here
Seybold's tables are physics-lab materials. Your indoor/city engine needs **building**
materials — concrete, drywall/gypsum, brick, wood, glass panes, metal studs — which are
**not** in these tables, and whose εr *and* σ both **vary with frequency**. Authoritative
source: **ITU‑R P.2040** ("Effects of building materials and structures on radiowave
propagation"), which gives εr and σ as functions of frequency for exactly these materials.

**Plan:** populate the engine's material library from **ITU‑R P.2040**; use Seybold here for
the *physics* (why εr and σ matter) and for the σ-range intuition (conductor → insulator).
