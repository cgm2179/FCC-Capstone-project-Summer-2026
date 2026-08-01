# Ch 3 — Antenna Fundamentals

**Verdict: Selective** · **Pipeline tier 1** · read the marked sections — this is your
transmitter source model.

## Sections
- 3.2 Antenna Parameters — **read** 3.2.1 gain, 3.2.2 effective area, 3.2.3 radiation
  pattern, 3.2.4 polarization. Skim 3.2.5 impedance/VSWR (circuit-side, not propagation).
- 3.3 Radiation Regions — **read**. Near-field vs far-field boundary `2D²/λ` tells you
  the minimum radius where your far-field source assumption is valid.
- 3.4 Common Antennas (dipole, beam, horn, reflector, phased array) — **skim / reference**.
  Only matters if you model a specific antenna's pattern.
- 3.5 Antenna Polarization — **read**. 3.5.1 XPD, 3.5.2 Polarization Loss Factor.
- 3.6 Pointing Loss — **skim**.

## Engine relevance
The source term: EIRP + gain pattern `G(θ,φ)` is what launches the field before any
propagation. PLF (3.5.2) is a scalar loss when Tx/Rx polarizations mismatch. Far-field
distance (3.3) bounds where the sim's point-source model holds.

## Reading progress — ✅ complete (2026-07-31, selective)
Read §3.2 / §3.3 / §3.5; skimmed §3.4 (antenna zoo) & §3.6 (pointing loss) as reference.

## Key result — the Friis link is now closed
`Ae = Gλ²/4π` (eq 3.3 inverted) is the **receive effective area** — the piece that turns §1's
FSPL into a real link budget (Ch 4). And `d > 2D²/λ` (eq 3.9) is the **far-field boundary =
where Ch 2's ray theory is valid**.

## Equations encoded → [`../RF_Equations.ipynb`](../RF_Equations.ipynb) §9
- Gain/area: `antenna_gain_aperture()` (4πAe/λ²), `effective_area()` (Gλ²/4π = Friis Rx aperture),
  `effective_area_physical()` (ηAp), `gain_from_beamwidth()` (26000/θθ) 🟢
- Regions: `far_field_distance()` (2D²/λ), `reactive_nearfield_radius()` (λ/2π) 🟢
- Match: `reflection_coeff_z()`, `mismatch_loss()`, `vswr()` 🟢
- **Polarization (the deferred plf):** `plf_linear()` (cos²τ), `xpd_linear()` (sin²τ),
  `plf_elliptical()` (full eq 3.13) 🟢
- Verified vs book: Ex 3.1 (39.6 dBi), Ex 3.4 (0.341 m), Ex 3.5 (−0.35 / −0.01 dB, XPD −11.1 dB).
  ⚠ *Ex 3.3 has a slip — states 73 Ω but its 0.23 / 1.6 fit ~80 Ω; our functions give the correct
  0.187 / 1.46.*
