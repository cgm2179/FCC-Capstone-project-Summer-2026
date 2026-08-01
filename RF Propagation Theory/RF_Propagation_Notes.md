# RF Propagation — Running Notes

Study log for **EE 625 Radio Wave Propagation**. One entry per chapter (or session).
Keep it terse — equations, meaning, and the engine hook. Encode the math in
`RF_Equations.ipynb` and reference the function name here.

**Legend:** 🟢 encoded in notebook · 🟡 reading/deriving · ⚪ not started ·
🔧 wired into engine

---

## Entry template (copy this per chapter)

```
## Ch NN — <title>   [date]   status: ⚪/🟡/🟢

**Sections read:** 
**Key equations:**
- <name>:  <formula>            → notebook: `fn_name()`  <status>
**What it means for the engine:**
- 
**Open questions / to verify:**
- 
**Cross-check vs engine module:** <path or effect name>
```

---

# Log

## Ch 1 — Introduction   [2026-07-31]   status: 🟢 (bands + framing)
> Skim chapter. Value = §1.1 band vocabulary + §1.4-style model-selection framing.
> (Text pasted covered §1.1–§1.2; §1.3/§1.4 not yet pasted — framing below is
> reconstructed from the chapter intro + the note under Table 1.1.)

**§1.1 Frequency bands.** RF = 1 MHz–300 GHz; microwave 1–30 GHz; MMW 30–300 GHz. Two
naming systems overlap: IEEE spectrum (MF/HF/VHF/UHF/SHF/EHF, one decade each) and letter
bands (L/S/C/X/Ku/K/Ka…). Engine bands all sit **UHF–SHF**: 2.4 GHz = UHF/S, 5–6 GHz =
SHF/C, mmWave 24–40 = SHF/Ka. Full table → `01_Introduction/01_Frequency_Bands.md`;
code → `band_of()` in the notebook (§0).
- *Engine takeaway:* at 2.4–6 GHz, penetration + reflection + diffraction are **all**
  first-order (none optional). Higher f ⇒ less penetration, more reflect/diffract/scatter.

**Model-selection framing (the §1.4 payoff).** The book's worldview:
- Real environments have too many unknowns for a deterministic solve ⇒ most models are
  **statistical**: a mean/median path loss **+ a probabilistic margin** sized to an
  **availability** target (e.g. coverage at 99% of locations).
- Most models are **empirical** ("loosely based on physics"), fit to measurements.
- **Free space is the one clean deterministic case.**
- *Engine takeaway:* we're building the deterministic thing the book calls usually-
  impossible — and we can **because indoors we actually have the geometry + materials**,
  so the unknowns are bounded. That is the core justification for a physics engine over an
  empirical model. Still bolt on the statistical layer (log-normal `X_σ`, Ch 8.3 / 9) to
  cover furniture/people/clutter not in the floor plan — that's why `log_distance_pl()`
  carries a `sigma`. **Model selection = a fidelity/cost ladder (your Tier 0–3): use the
  cheapest model that answers the question.**

**Equations also in §1.2 (pasted) — encoded:**
- Power density `S = P/(4πd²)` (eq 1.1) — the 1/d² spreading FSPL is built on → `power_density()` 🟢
- Wavelength `λ = c/f` (eq 1.2) → `wavelength()` helper 🟢
- Radio horizon `d(mi) ≈ √(2·h_ft)`, 4/3-earth (eq 1.4) — **outdoor/LOS only**; reproduces
  Example 1.1 (100 ft + 50 ft → 24.1 mi) → `radio_horizon_mi()` 🟢

**Open:** paste §1.3 (Why Model Propagation) + §1.4 (Model Selection, incl. 1.4.1 Model
Sources) to capture the book's explicit model taxonomy / sources.

---

## Ch 2 — Electromagnetics & RF Propagation   status: 🟢 (complete, 2026-07-31)
> ★ Essential. The physics floor for reflection / refraction / absorption. Read in 3 parts:
> Part 1 §2.1–2.3 static fields/materials · Part 2 §2.4–2.5 waves in matter/polarization ·
> Part 3 §2.6–2.8 boundaries (Fresnel), impairment taxonomy, polarization loss.

### Part 1 — §2.1–2.3 Static fields & material properties   🟢
**The one idea that matters:** a material's EM behavior reduces to **(εr, σ)** because the
book assumes **μr = 1** (non-magnetic, §2.3). That is *why* `n = √εr` and *why* the Fresnel
coeffs take εr alone — it validates how `fresnel_coeffs()` is written.
- `E = D/ε` (constitutive); `εr` = ε/ε₀ = dielectric constant. `ε₀ = 8.854e-12 F/m`.
- `σ` (S/m) = conductivity, `ρ = 1/σ`. Perfect dielectric σ=0; real materials have both εr, σ;
  higher σ ⇒ lossier. Table 2.3 spans ~10⁷ (metals) → 10⁻¹⁷ (quartz), ~24 decades.
- **Forward hook:** εr + σ + f combine into a *complex* permittivity (§2.4.2) → the imaginary
  part is absorption. Already coded as `complex_permittivity()` (§5).
- Static **E-field** refraction at a boundary (eq 2.1): `tanφ₂ = (εr2/εr1)·tanφ₁`.
  ⚠ Bends the E-field **vector**, not the ray — **NOT Snell's law** (book footnotes this).
  Verified in notebook: E tilts *away* from normal (30°→73.9° into εr=6) while a ray bends
  *toward* it (30°→11.8°). Opposite senses — do **not** wire eq 2.1 into refraction. Snell
  comes from §2.6.
- `μ₀ = 4π·10⁻⁷ H/m`.

**Encoded (§4):** `refractive_index()`, `MATERIALS{}` (εr,σ table), `efield_refraction_angle()`
(labeled NOT-Snell). Material data + ITU‑R P.2040 pointer →
`02_Electromagnetics_and_RF_Propagation/02_Material_Properties.md`.

**Engine takeaway:** the material library needs (εr, σ) per surface class. Seybold's tables
are generic-physics materials; **building** materials (concrete, drywall, brick, wood) are
**not** here → pull those from **ITU‑R P.2040** (standard εr,σ-vs-frequency source).

### Part 2 — §2.4–2.5 Waves in matter & polarization   🟢
- **Velocity** `v = c/√(εr μr)` (eq 2.4); **wave number** `k = ω√(με) = 2π/λ` (eq 2.5);
  **intrinsic impedance** `Z0 = 377√(μr/εr) Ω` (eq 2.12). → `phase_velocity()`, `intrinsic_impedance()`.
- **Complex permittivity** `ε = ε'(1 − jσ/ωε')` (eq 2.6): imaginary part = loss; tanδ = σ/(ωε').
  **Low-loss** tanδ<0.1 (≈dielectric), **high-loss** tanδ>10 (≈conductor). → `loss_regime()`.
- **Conductor:** wave decays `e^(−αz)`; **α** (eq 2.9a), **β** (eq 2.9b); good-conductor
  **skin depth** `δ = 1/√(πfμσ)` (eq 2.11). → `wave_params_lossy()`, `skin_depth()`.
  *Copper δ: 67 µm @1 MHz → 1.4 µm @2.4 GHz → 0.27 µm @60 GHz — a metal sheet of microns is
  opaque ⇒ studs / foil-backed insulation / appliances = hard blocks.*
- **Polarization** (§2.5): orientation of E. Linear (V/H); circular/elliptical = two orthogonal
  linear waves 90° out of phase; axial ratio = major/minor, 0 dB = circular. Poynting `S = E×H`.

### Part 3 — §2.6–2.8 Boundaries, impairment, polarization loss   🟢
- **§2.6 Fresnel via transmission-line analogy:** ρ = (Z_L−Z_z1)/(Z_L+Z_z1) (eq 2.15),
  τ = 2Z_L/(Z_L+Z_z1) (eq 2.16) ⇒ **τ = 1+ρ** (field coeffs; τ can exceed 1). Angles by
  **Snell** cos θ2/cos θ1 = √(εr1/εr2) (eq 2.14). ⚠ **Book uses grazing angle (from surface);
  notebook uses angle-from-normal** — grazing = 90°−normal, so Fig 2.7's x-axis mirrors ours.
- **Critical/polarizing = Brewster** (eq 2.23, grazing) — TM only; ρ_TM→0 = total transmission.
  VERIFIED: eq 2.23 grazing (23.6°) + notebook Brewster arctan√εr (66.4°) = 90°, and |Γ_TM|
  independently bottoms at 66.4°. → `brewster_grazing_deg()`, `brewster_normal_deg()`.
- **§2.6.2 dielectric→conductor:** TM ρ=−1 (180° flip), TE ρ=+1, τ=0 ⇒ conductor bounce
  **flips circular handedness** (RHCP↔LHCP). At **grazing**, both pols → ρ=−1 (two-ray null). ✓
- **§2.6.3 lossy dielectric:** feed complex ε into the same coeffs → complex ρ,τ (mag+phase);
  angles unchanged, amplitude/phase modified. ✓ validates `complex_permittivity()`→`fresnel_coeffs()`.
- **§2.6.2 TIR** (eq 2.24): into a *lower*-ε medium (ε2/ε1<1), τ can hit 0 → total reflection.
  Indoors: a wave inside a wall hitting the wall→air face can TIR ⇒ **waveguiding along walls/corridors**.

**§2.7 impairment taxonomy = the engine's effect modules (1:1):**

| §2.7 phenomenon | Engine `_3D.py` | Physics source |
|---|---|---|
| Reflection (specular, φ=φ') | reflection | Fresnel Γ (§2.6) |
| Refraction (bends in, penetrates) | refraction | Snell + τ (eq 2.14) |
| Absorption (ohmic loss in material) | absorption | α / complex ε (§2.4.2–3) |
| Scattering (rough → many dirs) | scattering | Rayleigh criterion (Ch 8) |
| Diffraction (sharp edge fills shadow) | diffraction | knife-edge / UTD (Ch 8) |
| (aggregate) | path-loss | sum of the above |
| Depolarization (pol mismatch) | — (cross-cutting) | PLF §2.8 |

**Engineering-judgment takeaways:**
- Book: exact lossy-boundary analysis is "tedious" + material props "not well known" ⇒ RF prop
  is "generally treated using empirical data." **Exception it names: RF modeling software w/
  numerical surface techniques (= our engine) and glass panes (clean, known-property case).**
  ⇒ deterministic engine validated, but **material-data quality is the binding constraint**
  ([[rf-propagation-theory-study-track]] §1.4 + the ITU‑R P.2040 gap). Windows = cleanest surface.
- **Ray theory = far-field / plane-wave only**; invalid in near-field (ties to Ch 3 far-field
  distance 2D²/λ). Keep the Tx far enough that the point-source ray model holds.

**§2.8 PLF** (eqs 2.25–2.31): polarization-mismatch loss via Tx/Rx axial ratios. Deferred —
only if the engine becomes polarization-aware (revisit with Ch 3.5.2). → `plf()` ⚪

**Open:** which polarization does the engine assume (TE / TM / unpolarized avg)? decide + doc.
**Cross-check vs engine module:** reflection / refraction / absorption / scattering / diffraction `_3D.py`.

---

## Ch 3 — Antenna Fundamentals   status: 🟢 (Selective — Tx source model, 2026-07-31)
> Selective per triage: read §3.2 params, §3.3 regions, §3.5 polarization; §3.4 antenna zoo +
> §3.6 pointing loss = reference/skim. Reciprocity: Tx and Rx antenna params are identical.

**Tx source model (§3.2).** Engine source = **EIRP = Pt · G** + a gain pattern G(θ,φ).
- Gain `G = η·D` (η efficiency ~0.5–0.8; D = peak/mean power density; isotropic D=1). Units dBi
  (vs isotropic) or dBd (vs ½-wave dipole; 0 dBd = 2.14 dBi).
- Aperture: `Ae = η·Ap` (eq 3.2), `G = 4π·Ae/λ²` (eq 3.3); assume η≈0.6 from dimensions. Beamwidth
  rule `G ≈ 26000/(θ_az·θ_el)` (deg). *Verified Ex 3.1: 30 cm dish @39 GHz = 39.6 dBi.*
- **`Ae = Gλ²/4π` is the Friis receive aperture** — the missing piece that turns §1 FSPL into a
  real link budget (Ch 4). → `antenna_gain_aperture()`, `effective_area()`.
- Linear antennas use effective *height* he, not area; ½-wave dipole Ae=0.119λ², Rr=73Ω.
- Pattern (§3.2.3): 3-dB beamwidth, sidelobe level, front-to-back. Uniform aperture → −13 dB
  sidelobes/narrowest beam; taper → lower sidelobes + wider beam + taper loss. Pattern = FT of illumination.

**Far-field / regions (§3.3) — the ray-theory validity boundary.**
- **Far-field (Fraunhofer):** `d > 2D²/λ` (eq 3.9). Pattern fully formed, gain angle-only,
  wavefront ≈ planar, E⊥H. **This is exactly where Ch 2's ray theory holds.** → `far_field_distance()`.
- Reactive near-field `r < λ/2π` (eq 3.11); radiating near-field between. → `reactive_nearfield_radius()`.
- Caveat: electrically small antennas (D≪λ) → 2D²/λ can fall *inside* the near-field; use the
  near-field boundary instead. *Verified Ex 3.4: 140 MHz → reactive NF 0.341 m.*
- Engine: keep Tx ≥ far-field distance from the first voxel so the point-source ray model holds.
  Indoors @2.4 GHz with a small antenna that's ~cm–1 m → usually a non-issue.

**Impedance/VSWR (§3.2.5).** `ρ=(Z1−Z0)/(Z1+Z0)` (3.6), mismatch loss `1−ρ²` (3.7),
`VSWR=(1+ρ)/(1−ρ)` (3.8). → `reflection_coeff_z()`, `mismatch_loss()`, `vswr()`. System-side (not
propagation) → a fixed link-budget offset. ⚠ *Book Ex 3.3 slip: states 73 Ω → 0.23/1.6, but 73/50
gives 0.187/1.46; 0.23/1.6 fits ~80 Ω. Our functions give the correct 0.187/1.46.*

**Polarization + PLF (§3.5) — closes the deferred `plf()` from Ch 2.**
- Elliptical framework: E = x·E1 sin(ωt−βz) + y·E2 sin(ωt−βz+δ). δ=0 → linear; E1=E2 & δ=±90° →
  circular (IEEE: δ=−90° RHCP, +90° LHCP). Axial ratio AR = major/minor ≥ 1 (0 dB = circular).
- **XPD** linear = `sin²τ`; **PLF** linear = `cos²τ`; XPD = 1−PLF (τ = tilt between wave & antenna).
  → `plf_linear()`, `xpd_linear()`. Full elliptical PLF (eq 3.13 = Ch 2's eq 2.28) → `plf_elliptical()`.
- *Verified: both circular → PLF 1 (any tilt); large equal AR → cos²τ; circular↔linear = −3 dB;
  Ex 3.5 (AR 2 dB tx / 3 dB rx): worst −0.35, best −0.01 dB, XPDmin −11.1 dB.*
- **Circular↔linear ≈ 3 dB** rule of thumb (when Tx/Rx pol types differ or orientation unknown).

**Antenna zoo (§3.4) — reference only.** ½-wave dipole 2.14 dBi / Rr 73Ω; λ/4 monopole on ground
plane; Yagi (directional, linear); horn (aperture, gain standard); parabolic/Cassegrain; **phased
array** (electronic beam-steer + **null-steer** toward interferers = "smart antenna"). Matters only
if the engine models a specific pattern/beamforming. **§3.6 pointing loss:** link-budget misalignment
allowance (0.5–1 dB) — skim.

**Engine takeaways:**
- First cut: isotropic/omni Tx, G = const, single pol → PLF is a fixed offset.
- Upgrade: add a gain pattern G(θ,φ) (beamwidth rule or measured), and if the engine tracks
  depolarization ([[rf-propagation-theory-study-track]]; Ch 2 conductor bounces flip handedness),
  wire in `plf_elliptical()`.
- Feeds **Ch 4 link budget** next: EIRP → FSPL/PL(x,y) → Rx gain → SNR/margin.

**Open:** engine Tx-model fidelity — isotropic vs patterned; single-pol vs pol-aware?

---

## Ch 4 — Communication Systems & the Link Budget   status: 🟢 (complete, 2026-07-31)
> Important per triage. Read in 2 parts: **Part 1 §4.1–4.3** (link margin, path loss, noise) ·
> **Part 2 §4.4–4.6** (interference margin, detailed link budget, Eb/N0).

**The link budget = the "so what" that turns PL(x,y) into coverage.**
- **Link margin** `M = EIRP − L_path + G_Rx − TH_Rx` (all dB); M > 0 ⇒ link closes.
  → `link_margin_db()`. *Verified Ex 4.1: 100 m / 10 GHz / EIRP 25 dBm / TH −85 dBm → 22.6 dB.*
- **Coverage(x,y) = { M(x,y) > 0 } = { PL(x,y) < EIRP + G_Rx − TH_Rx }.** The engine produces
  PL(x,y); this is the thin post-step that turns it into a coverage boolean per cell.

**§4.2 Path loss (Friis).**
- `L = G_T G_R (λ/4πd)²` (eq 4.1) — Friis *including* antenna gains (a "gain" < 1). FSL without
  gains = `20log(4πd/λ)` = §1 `fspl_db` (reconfirmed). → `friis_received_power_dbm()`.
- The λ-dependence is only because Friis uses Rx *gain*, not effective area `Ae=Gλ²/4π` (Ch 3) —
  same physics, two bookkeepings.
- **Modified power law** (exponent ≠ 2): no longer FSL but a *median* path loss for lossy / near-
  earth environments → this is the log-distance exponent `n` (Ch 9). The engine's non-free-space cells.
- Misc losses to budget: Tx/Rx cable/waveguide, radome, pointing (Ch 3.6), polarization (Ch 3.5),
  atmospheric (Ch 6/10). "Path loss is the most significant term."

**§4.3 Noise — sets TH_Rx (the coverage cutoff).** *Receiver-system, not propagation — the engine
doesn't compute it, but coverage depends on it.*
- **Thermal (AWGN) floor** `N = kT₀B` (eq 4.4), k=1.38e-23, T₀=290 K. In dB:
  `N = −174 dBm/Hz + 10log₁₀B + NF` (eq 4.10b). **−174 dBm/Hz = kT₀ at 290 K** (memorize).
  → `thermal_noise_dbm()`. *Verified Ex 4.2: 10 MHz, Te 870 K → NF 6 dB, N = −98 dBm = −128 dBW.*
- **Noise figure** `F = 1 + Te/T₀` (eq 4.12) = SNRin/SNRout. → `noise_figure_db_from_temp()`.
- Noise-equiv bandwidth `B_N ≈ 1/T_S` (symbol rate); 3-dB BW is a common proxy.
- **Cascade (Friis):** `F_tot = F₁ + (F₂−1)/G₁ + (F₃−1)/(G₁G₂) + …` → first stage (LNA) sets the
  floor; **passive loss before the LNA adds dB-for-dB to NF.** → `cascade_noise_factor()`.
  *Verified Ex 4.3: 7 dB cable + 5 dB Rx = 12 dB.* ⚠ **Book slip:** printed eq 4.14 drops the `−1`;
  with it the cascade matches the book's own 12 dB, without it 13.2 dB. Implemented the correct −1 form.
- Only absorptive (resistive) losses count in NF; gain reductions (pointing) don't (noise referenced
  to Rx input). Non-AWGN sources (LO phase noise, AM-AM/PM, intermod, sync loss) → treated as extra
  loss / raised required SNR.

**Threshold synthesis (encoded):** `TH_Rx = (−174 + 10logB + NF) + SNR_req`; then
`max allowed PL = EIRP + G_Rx − TH_Rx`. The engine's PL(x,y) is thresholded against this.

**Engine takeaways:**
- Add a thin coverage post-step: `margin(x,y) = EIRP − PL(x,y) + G_Rx − TH_Rx`; render `margin > 0`.
  EIRP & G_Rx from Ch 3, PL from the engine, TH_Rx from §4.3.
- The path-loss *exponent* knob (modified power law) is the bridge to log-distance / ITU indoor (Ch 9).

### Part 2 — §4.4–4.6 Interference, detailed link budget, Eb/N0   🟢 (2026-07-31)
- **Interference margin:** a link-budget term for noise-floor rise from external interference.
  *Ex 4.4: a 1-dB margin ⇒ total interference must stay ≥ 5.9 dB below the noise floor* →
  `interference_for_margin_dbm()`. Types: co-channel (in-band, unfiltered), adjacent-channel
  (attenuated by filter skirts), ISI (spectral shaping; folded into implementation loss), intermod,
  de-sensing (strong nearby signal drives front-end AGC, cutting desired-signal gain).
- **Detailed link budget (§4.5, Fig 4.4)** — the per-cell coverage template:
  - EIRP = P_Tx + G_Tx − L_WG − L_radome;  Rx gain = G_Rx − L_radome − L_WG − L_pol − L_pt.
  - Total PL = FSL + fade margins (rain, at an *availability* prob) + misc (pointing, multipath, atmos).
  - RSL = EIRP − PL + Rx_gain;  SNR = RSL − N − M_int;  net margin = RSL − M_int − TH.
  → `link_budget()`. *Verified: reproduces all 9 totals of Fig 4.4 — EIRP 38.5, FSL 130.2, PL 148.4,
  RxG 26.8, RSL −83.1, N −93.0, SNR 8.9, TH −88.0, net margin 3.9 dB.*
- **Eb/N0 = SNR + 10log(B/Rb)** (use the *data* bit rate, not channel rate). → `eb_n0_db()`. The true
  "link closes" test for digital systems; required Eb/N0 comes from the modem/BER spec.
- ⚠ **Book slip #3:** the inline 4.5.2–4.5.5 numbers (PL 135 dB, EIRP 27 dBm, SNR 16 dB, 10log B = 60)
  are mutually inconsistent and mismatch Fig 4.4; only Fig 4.4 is self-consistent, so that's used.

**Engine takeaways (Part 2):**
- Fig 4.4 IS the engine's per-pixel coverage calc: PL(x,y) from the engine, everything else scalar.
  `link_budget()` with PL → PL(x,y) yields a `margin(x,y)` map.
- **Interference-limited coverage:** other transmitters = extra PL(x,y) maps summed as interference →
  SINR, not SNR (co-channel = the hard case; the interference map is the denominator). Bridge to a
  multi-source engine.
- Fade margins carry an *availability* probability (Ch 10 rain) — the statistical layer on top of the
  deterministic mean, echoing the §1.4 framing.

---

## Ch 5 — Radar Systems   status: 🟢 (Skip-tier; captured for a future radar/scattering mode, 2026-07-31)
> Skip per triage (indoor is one-way). Captured because reflections + RCS could seed a future
> **radar mode** or **scattering-reflectivity** feature.
- **Two-way ⇒ 1/R⁴** (doubling range = 12 dB): `Pr = Pt Gt Gr λ²σ/((4π)³R⁴)` (eq 5.8), `Rmax`
  (eq 5.9). → `radar_rx_power_dbw()`, `radar_max_range_m()`. *Verified Ex 5.1 (SNR 6.5), Ex 5.2 (42.2).*
- **RCS σ** (m²/dBsm) = *electrical* reflectivity, not physical size; varies w/ freq, aspect, pol.
  Shapes (Table 5.1): sphere πr², plate 4π(lw)²/λ². → `rcs_sphere()`, `rcs_flat_plate()`.
- **Clutter** (§5.4): area σ⁰ (RCS/m²), volume η (RCS/m³) — the diffuse-scatter analog; clutter
  falls off slower than a point target (illuminated area grows with R). Ties to Ch 8 Rayleigh roughness.

---

## Ch 6 — Atmospheric Effects   status: 🟢 (Skip indoor; captured for outdoor/mmWave, 2026-07-31)
> Skip for indoor. Relevant to the **outdoor city track** (long links) & **mmWave** absorption.
- **Refraction / radio horizon:** `N=(77.6/T)(P+4810e/T)` (eq 6.4), gradient `−(Ns/H)e^(−h/H)`,
  `k=1/(1+r·dn/dh)`, `d≈√(2krh)`. → `refractivity()`, `k_factor()`, `radio_horizon_km()`.
  *Verified Ex 6.1 (N 394.6, k 1.370, horizon 29.5 km).* **Ducting** at dn/dh = −157×10⁻⁶/km (k→∞).
- **Gaseous absorption:** `A=γd`; lines at **22 GHz (H₂O)** & **60 GHz (O₂, ~15 dB/km)**; ~0.05
  dB/km @1 GHz. → `atmospheric_loss_db()` (×2 for radar). Actual γ from ITU plots (formulas tedious).
- Fog/clouds (Kl·M), atmospheric multipath (ITU geoclimatic) = outdoor-microwave/satellite; noted, not encoded.

---

## Ch 7 — Near-Earth Propagation Models   status: 🟢 (Outdoor-track baseline, 2026-07-31)
> Track-dependent. **The empirical baseline + validation for the outdoor OSM-voxelized sim**
> ([[outdoor-city-voxelizer-track]]). All are *median* path-loss fits, not physics.
- **Foliage:** Weissberger (eq 7.1), early ITU (eq 7.2). → `weissberger_foliage_db()`,
  `itu_foliage_db()`. *Verified Ex 7.1 (5.40 / 7.06 dB).* Updated ITU caps loss at the diffraction path.
- **Terrain:** Egli 4th-power (eq 7.9) → `egli_pl_db()`; ITU terrain diffraction `Ad=−20 h/F1+10`
  with `F1=17.3√(d1d2/(fd))` = the **Ch 8 Fresnel radius** in km/GHz → `itu_terrain_diffraction_db()`.
  *Verified Ex 7.3 (F1 47.4, Ad 9.7).* Longley–Rice = detailed software model (no closed form; NTIA tool).
- **⚡ Urban macro-models (workhorses):** **Hata** 150–1500 MHz urban/suburban/open (eq 7.14) →
  `hata_pl_db()`; **COST-231** 1500–2000 MHz PCS (eq 7.19) → `cost231_pl_db()`; **Lee** fittable power
  law (eq 7.20, Table 7.2) → `lee_pl_db()`. *Verified Ex 7.5 (Hata 137.1 dB, a(hr) 2.69).* Okumura =
  graphical parent of Hata; Young = NYC power law (β clutter factor).
- ⚠ **Book slip #4 (Egli Ex 7.2):** prints 112.4 dB but used hb·hm = **6 instead of 60** (dropped ×10);
  correct value **92.4 dB**. Function uses the standard formula.
- *Engine:* run Hata/COST-231 for a link and confirm the physics engine lands in the same ballpark
  (validation). Hata/COST-231 need base *above* rooftops; Lee is the fittable choice with local data.
  These are the **cheap Tier-0 baseline** for the outdoor track — not a substitute for the GO/eikonal engine.

---

## Ch 8 — Fading & Multipath   status: 🟢 (complete, 2026-08-01)
> ★ Essential. Two-ray model + diffraction effect + the statistical (fading) layer. Read in 2 parts:
> **Part 1 §8.1–8.2** (ground-bounce, roughness, Fresnel, diffraction) ·
> **Part 2 §8.3–8.5** (log-normal shadowing, small-scale fading).

**§8.1 The 3-part path-loss framework** (the statistical layer, echoing §1.4):
1. **Median** path loss (Hata/Lee/FSL/ground-bounce) · 2. **Large-scale/shadowing** = log-normal
(slow, big geometry changes) · 3. **Small-scale fading** = Rayleigh/Rician (order of λ, multipath +
Doppler; delay spread = temporal, Doppler spread = spectral). Parts 2-3 are §8.3-8.4.

**§8.2 Ground-bounce (two-ray) — NEW, Tier-1 pipeline.**
- Smooth conductive ground at small grazing ⇒ **ρ ≈ −1** (180° flip, any polarization).
- E = Ed·2·|sin(Δθ/2)| (eq 8.6); exact phase Δθ (eq 8.8); Lmp = Lfsl·4sin²(Δθ/2) (eq 8.9).
- Far field ⇒ **received power ∝ 1/d⁴, independent of λ**: `Lmp ≈ (ht hr)²/d⁴` (eq 8.13; worst-case
  upper bound). → `two_ray_pathloss_db()` (min of FSL & 1/d⁴, eq 8.14).
- **Crossover** FSL→1/d⁴ at `dx = 4π ht hr/λ` (eq 8.15). → `two_ray_crossover_m()`.
  *Verified Ex 8.1 (dx 8.4 km; 4 km→FSL 110.5, 40 km→144.1 dB) + Fig 8.5 (dx 12.6 km).*
- In-phase ⇒ +6 dB; 180° out ⇒ full null. Engine: indoor floor/ceiling bounce is this effect.

**§8.2.1 Surface roughness — the scattering trigger.**
- **Rayleigh** `HR = λ/(8 sinθ)` (eq 8.16), sinθ ≈ (ht+hr)/d. Δh < HR ⇒ **smooth/specular
  (reflection)**; Δh ≫ HR ⇒ **rough/diffuse (scattering)**. → `rayleigh_roughness_m()`, `is_specular()`.
  *Exactly the reflection-vs-scattering decision `scattering_3D` needs.* @2.4 GHz, 5° grazing, HR ≈ 18 cm.

**§8.2.2 Fresnel zones — VERIFIED my blind seeds.**
- `hn = √(nλd1d2/(d1+d2))` (eq 8.20) = seeded `fresnel_zone_radius` ✓; `v = h√(2(d1+d2)/(λd1d2))`
  (eq 8.19) = seeded `knife_edge_v` ✓. *Verified Ex 8.2 (60% of 1st zone = 0.9 m).*
- **Odd Fresnel zones ⇒ destructive interference** (avoid reflectors there). **Keep 60% of the 1st
  zone clear** (v = −0.8 ⇒ 0 dB); v = 0 (50% blocked) ⇒ 6 dB.

**§8.2.3 Huygens / §8.2.4 diffraction loss — the diffraction effect.**
- Huygens: each wavefront point = secondary wavelet; a blockage unbalances them ⇒ energy radiates into
  the shadow. Why a knife edge *below* LOS still matters (3 shadow regions, Fig 8.11).
- **Knife-edge loss** two ways: Lee piecewise (eqs 8.21, matches book) → `knife_edge_loss_lee_db()`;
  ITU J(v) (seeded §8) → `knife_edge_loss_db()`. Agree ~0.2 dB. *Verified Ex 8.3 (v=−0.395 → Lee 2.55,
  ITU 2.76; book 2.6 dB).*
- Multi-edge & **rounded-surface** (`Lex = −11.7 α√(πr/λ)`, eq 8.22): advanced; Ex 8.4 double-hill →
  47.4 dB deep fade. Real corners → **UTD** (uniform theory of diffraction) = the `diffraction_3D`
  target (arbitrary shape/material); knife-edge is the simple-geometry proxy.
- H-pol diffracts slightly more than V-pol ⇒ circular pol becomes elliptical in the shadow.

**Cross-check vs engine module:** `diffraction_3D` (UTD; knife-edge proxy), `scattering_3D` (Rayleigh
trigger), reflection (two-ray floor/ceiling bounce).

### Part 2 — §8.3–8.5 Log-normal shadowing & small-scale fading   🟢 (2026-08-01)
**§8.3 Large-scale (log-normal) shadowing — the `X_σ` layer.** Many diffraction/reflection losses
multiply ⇒ add in dB ⇒ (CLT) **Gaussian in dB**. Characterized by location variability σ_L.
- Shadowing margin `L_S = z·σ_L`, `z = Φ⁻¹(coverage)`; edge coverage `= Φ(M/σ_L)`.
  → `shadowing_margin_db()`, `edge_coverage_prob()` (self-contained Φ / Acklam Φ⁻¹).
  *Verified Ex 8.5: 90% edge coverage → z=1.28, L_S=7.7 dB (σ=6) / 10.25 dB (σ=8).*
- Cell-area coverage = integrate over concentric rings (eq 8.23). σ_L(Okumura) = 0.65(log fc)² −
  1.3 log fc + A (A=5.2 urban / 6.2 suburban) → `location_variability_okumura()` (~7 dB @900 MHz).
- **Engine:** layer `X_σ ~ N(0, σ_L)` on any deterministic PL(x,y) → a coverage-*probability* map;
  `log_distance_pl()` already carries σ. All coverage curves cross at 50% (the median).

**§8.4 Small-scale fading — two independent axes.**
- **Amplitude pdf:** all reflections ⇒ **Rayleigh**; a dominant/LOS path ⇒ **Ricean** (factor
  K=10log(A²/2σ²)). → `rayleigh_fade_prob()` *(verified Ex 8.8: P(≥12 dB)=0.031)*, `ricean_fade_prob()`
  (numerical; consistent with Rayleigh as K→−∞, fewer deep fades for higher K; ⚠ book Ex 8.9's 0.018
  uses a different fade reference and doesn't reproduce — ours is internally consistent instead).
- **Spectral axis — delay spread** σ_τ ⇒ coherence BW `Bc ≈ 1/(5σ_τ)…1/(50σ_τ)` (eq 8.24). B<Bc ⇒
  flat; B>Bc ⇒ frequency-selective (needs equalizer/OFDM). → `coherence_bandwidth_hz()`.
  *Verified Ex 8.6 (Bc 50 kHz → σ_τ 4 µs), Ex 8.7 (35.2 m excess → 117 ns → max 852 ksps).*
- **Rate axis — Doppler** `fm = Δv/λ` (eq 8.26) ⇒ coherence time `Tc ≈ 1/fm` (eq 8.25). T_sym<Tc ⇒
  slow (AGC ok, QAM ok); else fast (use PSK/OFDM). → `doppler_shift_hz()`, `coherence_time_s()`.
  Spectral (flat/selective) and rate (fast/slow) are **independent** (Table 8.1).

**Engine takeaways:** the deterministic engine gives the *median* PL(x,y); Ch 8 wraps it in the
statistical envelope — `X_σ` shadowing → a coverage-*probability* map, and (for a channel-quality
layer) delay spread → coherence BW to flag where a link goes frequency-selective. The delay-spread
exponential impulse response is the Ch 9 indoor model.

---

## Ch 9 — Indoor Propagation Modeling   status: 🟢 (complete — book + references/exercises, 2026-08-01)
> ★ Essential AND the deployment use case. Book content §9.1–9.4 + reference content (bibliography +
> all 5 exercises worked) done. **Completes all three Essentials (Ch 2/8/9).**

**The framing (§9.1–9.3.2):** indoors, deterministic models are rare (layout/materials/people change
fast) → **site-general statistical models** fit to data. Site-specific ray-tracing (CAD) only for
large static environments — *exactly what the engine does, the exception the book flags* (cf. §1.4,
§2.6). Interference (§9.2) often matters more than propagation indoors (co-located Bluetooth/WiFi,
digital clocks, monitors, fluorescents) → check SIR ≥ required SNR; desensing/AGC. NLOS common →
reflection/diffraction/penetration dominate; scattering minor.

**§9.3.3 ITU indoor — VERIFIED my blind seed.** `PL = 20log f + N·log d + Lf(n) − 28` (eq 9.1) =
seeded `itu_indoor_pl_db` ✓. *Verified Ex 9.1: 5.2 GHz office, 100 m, N=31 → 108 dB same floor,
+16 dB/floor (→124 dB).* **N = 20 ⇔ free space** (FSPL with a tunable exponent; verified N=20 ≈ FSPL).
N rules: 18 corridor (channeling), 20 open, 40 walls/corners. Table 9.1 (N by band/env) + Table 9.2
(floor loss) captured → `ITU_N`, `itu_floor_loss_db()`. **Maps onto the engine's Motley–Keenan
distance + floor-count terms.**

**§9.3.4 Log-distance — VERIFIED my blind seed.** `PL = PL(d0) + N·log(d/d0) + Xσ` (eq 9.2) = seeded
`log_distance_pl_db` ✓ (my exponent `n` = book's N/10; book N=30 ⇒ n=3). Xσ ~ N(0,σ) = the Ch 8
shadowing. *Verified Ex 9.2: 1.5 GHz office, 100 m, 95% → 36 + 30log(100) + Xσ(11.5) = 107.5 dB
(FSL 76).* Table 9.4 (N,σ by building) → `LOGDIST_PARAMS`. **Rappaport: indoor σ ≈ 13 dB ⇒ ±26 dB
(2σ) is normal** — field measurements are the final word.

**§9.3.3 delay spread:** indoor impulse response `h(t) = e^(−t/S)`, 0<t<tmax (the exponential Ch 8.4
pointed to). → `indoor_impulse_response()`. Table 9.3: S ≈ 20–500 ns. Median office S=100 ns →
coherence BW ~2 MHz < a 20 MHz WiFi channel ⇒ **frequency-selective indoors** (why WiFi uses OFDM).

**Engine takeaways:**
- ITU/log-distance = the **Tier-0 baseline** for `PL(x,y)` and the sanity check on the GO/eikonal
  result. N (per environment) + Lf(n) are the empirical inputs; physics counterpart = Ch 2
  Fresnel/absorption + ITU-R P.2040 material data.
- The engine IS the *site-specific* deterministic model the book calls usually-impractical — viable
  because you supply the geometry ([[rf-propagation-theory-study-track]]). Layer Xσ (σ_L ~ 7–14 dB)
  for the coverage-probability map.
- Indoors is frequency-selective (delay spread) but delay/Doppler otherwise mild (short range, low
  speed) → the static PL(x,y) map is the right primary output.

**Cross-check vs engine module:** path-loss layer (Motley–Keenan ↔ ITU N/Lf), `X_σ` shadowing.

**Reference content (done):** bibliography + exercises → `09_Indoor_Propagation_Modeling/09_References.md`.
Authoritative sources: **ITU‑R P.1238** (indoor N/Lf/delay — pull real numbers here) and Rappaport
(log-distance + Table 9.4). All 5 exercises worked in notebook §18 with the encoded functions
(9‑1: 97.6 dB · 9‑3: 119 dB · 9‑4: [60.9, 96.9] dB · 9‑5: ≤400 ksps) — a working-tool check.
The book cites further empirical models (Ericsson multi-breakpoint, attenuation-factor); most indoor
models are modified-power-law variants, so the two encoded here cover the family.

---

## Ch 10 — Rain Attenuation   status: 🟢 (Skip-tier; captured for future outdoor mmWave, 2026-08-01)
> Skip for indoor. The dominant availability limiter for **outdoor links >~10 GHz** — relevant only if
> the outdoor city track goes mmWave. Rain-fade modeling isn't a mature field (huge year-to-year variance).
- **Specific attenuation** `γ = k·RR^α` dB/km; k,α frequency+polarization dependent (Table 10.1). Combine
  by polarization (τ=0 H / 45 circ / 90 V, eqs 10.3-10.4). **Horizontal rains worse than vertical**
  (elongated drops); circular between. → `rain_coeff_interp()`, `rain_coeffs()`, `rain_specific_attenuation()`.
  *Verified 38.6 GHz → k=0.324, α=0.95.*
- **ITU model** `A_0.01 = γ·d·r`, `r = 1/(1+d/d0)`, `d0 = 35·e^(−0.015 RR)` (eqs 10.5-10.7); availability
  scaling by latitude (eqs 10.8/10.9). Rain rate by region (Table 10.2). → `itu_rain_attenuation_db()`,
  `itu_availability_adjust()`, `ITU_RAIN_RATE_001`. *Verified Ex 10.1: Florida N, 38.6 GHz, 1.1 km →
  23.8 dB @99.99%, 34.3 dB @99.999%.* Valid to 40 GHz / 60 km. Rain is *nonlinear* with distance (the r factor).
- **Fog/cloud** `γc = Kl·M` (eq 10.15). → `fog_attenuation_db()`. *Verified Ex 10.4: 30 GHz heavy fog → 4.7 dB/15 km.*
- ⚠ **Crane global model** (two-segment, ≤22.5 km): eqs 10.11/10.13 garbled in OCR — couldn't reproduce
  Ex 10.2 (y=−0.189/43.9 dB). Only `crane_breakpoint_km()` (eq 10.12) + z (eq 10.14, verified) shipped.
  *Book slip: Ex 10.2's "0.0226" in z should be "0.026" (eq 10.14).* Use Crane 1996/2003 for the full model.
- **Cross-pol / other precip:** rain depolarizes (caps pol-diversity reuse at ~20-35 dB isolation); snow/sleet
  ≈ treat as rain (over-estimates). Availability: 4-nines ≈ 53 min/yr outage, 5-nines ≈ 5 min/yr.
- *Engine:* a future `A = γ(f,pol)·d·r` rain term on the outdoor mmWave link budget; **ITU-R P.838** (coeffs)
  + **P.837** (rain rates) are the live data sources ([[itu-recs-for-engine]]).

---

## Ch 11 — Satellite Communications   status: 🟢 (Skip-tier; final chapter, 2026-08-01)
> Skip for indoor. The final chapter — a capstone that reuses FSPL + rain + noise. Different geometry:
> one long slant path through the whole atmosphere.
- **Slant-range geometry:** central angle ψ from lat/long (eq 11.4), slant range rs (eq 11.5), elevation
  θ (eq 11.6). GEO ≈ 42,242 km from earth center; round-trip delay ~240 ms. → `sat_central_angle()`,
  `sat_slant_range_km()`, `sat_elevation_deg()`. *Verified Ex 11.1: 20° lat → rs 36,314 km, θ 66.6°.*
- **ITU satellite rain** (P.618, 10-step, eqs 11.9-11.25) = §19's terrestrial model + rain-cell-height
  geometry (hR = 4 km for |lat|<36, else 4−0.075(lat−36)). → `itu_sat_rain_atten_db()`. *Verified Ex 11.3
  uplink: NYC region K, 30 GHz, θ 40.9° → A0.01 38.5 dB, A0.1 14.8 dB (exact).* ⚠ Ex 11.3 downlink (43.25 dB)
  didn't reproduce with the uplink-validated function — likely a downlink OCR/input issue.
- **Noise — absorptive losses raise the noise floor** (the often-missed effect): hot-pad
  `TN=(Tin+(L−1)T)/L` (eq 11.50), rain temp `Tr=Tm(1−10^(−A/10))` (eq 11.51). → `hotpad_noise_temp()`,
  `rain_noise_temp()`. *Verified Ex 11.5/11.6: 6 dB fade → TN 222.5 K / Tr 204.4 K.* A rain fade hits SNR
  **twice**: signal down 6 dB AND noise up ~6 dB ≈ 12 dB total. G/T merit = `gt_db()`. *Book slips: Ex 11.5
  says "9.9 dB" SNR loss (should be 11.9, per Ex 11.6); "G/T 14.3 dB" is the linear ratio 1000/70, not dB (11.55 dB/K).*
- Ionospheric effects (Faraday rotation, scintillation, dispersion) only <10 GHz → **circular pol** used
  (insensitive to Faraday rotation). Sun outage: solar noise −183 dBW/Hz (~36,500 K) = +21 dB.
- *Engine:* the hot-pad insight **generalizes** — *any* absorptive loss (a lossy wall, foliage) both
  attenuates the signal AND raises the noise floor; a rigorous indoor SNR-level coverage model would apply it too.

---

## ✅ Corpus complete — Ch 1–11 read, encoded & verified (2026-08-01)
Every engine-relevant equation from EE 625 is a runnable, example-checked function in `RF_Equations.ipynb`
(§0–§20, 72 cells). Essentials Ch 2/8/9 drive the indoor engine; Ch 5-7/10-11 are the outdoor/future layer.
4 blind-seeded functions verified exact; ~7 book slips flagged (not reproduced). Ch 12 (RF Safety) = out of
scope for propagation.

---

<!-- Add Ch 3, 4, 7 entries when you get to the Tx-model / link-budget / outdoor work. -->
