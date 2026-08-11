# RF Propagation Theory — Study Track

Working through **EE 625 — Radio Wave Propagation** to sharpen the physics behind the
RF simulation engine (indoor GO/eikonal path-loss + material effects, plus the FDTD
sandbox and outdoor city tracks).

**Goal:** turn textbook equations into runnable code and make better engineering
decisions about *which* physics to model, in *what order*, from a basic 2D/3D
propagation sim up to the most advanced.

## What's in here

| File | Purpose |
|------|---------|
| `EE 625 Radio Wave Propagation.pdf` | The textbook. |
| `RF_Propagation_Notes.md` | **Running log** — per-chapter notes, equations, "what this means for the engine", open questions. Grows as I read. |
| `RF_Equations.ipynb` | **Code notebook** — each key equation encoded as a runnable Python function + a demo plot. |
| `01_…` … `12_…/` | One folder per chapter. Each has a `README.md` with a section-by-section triage and which equations to encode. Drop your own scratch notes / code per chapter here. |

---

## Chapter triage — what's worth your time

Ranked for **your engine**, not for a general RF course. Three tiers:
**Essential** (core physics of your stack) · **Selective/Important** (read the marked
sections) · **Skip** (not your use case — revisit only if a track demands it).

| Ch | Title | Verdict | Why it matters to the engine |
|----|-------|---------|------------------------------|
| 1 | Introduction | **Skim** | 1.1 frequency bands + 1.3–1.4 (why/what to model) — ~15 min of framing. |
| 2 | Electromagnetics & RF Propagation | **★ Essential** | The foundation. **2.6 propagation at material boundaries = Fresnel** → your reflection / refraction / absorption effects. |
| 3 | Antenna Fundamentals | **Selective** | 3.2 gain/eff-area/pattern, 3.3 near-vs-far field, 3.5 polarization = your **Tx source model**. Skip the antenna zoo (3.4). |
| 4 | Communication Systems & Link Budget | **Important** | 4.2 path loss + 4.5 link budget / EIRP / SNR = how `PL(x,y)` becomes usable coverage. Skim noise/interference (4.3–4.4). |
| 5 | Radar Systems | **Skip** | Not your use case. 5.4 clutter/RCS *loosely* informs the scattering stub — optional. |
| 6 | Atmospheric Effects | **Skip (indoor)** | Outdoor/long-range only. Revisit 6.2 refraction if the outdoor city track goes long-haul. |
| 7 | Near-Earth Propagation Models | **Track-dependent** | **7.4 Built-Up Areas (Okumura / Hata / COST-231)** = empirical baseline for the **outdoor city track**. 7.2 foliage optional. |
| 8 | Fading & Multipath | **★ Essential** | **8.2.2 Fresnel zones + 8.2.3–8.2.4 diffraction loss** → your **diffraction** effect. 8.3–8.4 fading = stochastic layer + validation. |
| 9 | Indoor Propagation Modeling | **★ Essential** | Literally your use case. **9.3.3 ITU indoor + 9.3.4 log-distance** = your path-loss layer (maps to Motley–Keenan). |
| 10 | Rain Attenuation | **Skip** | mmWave / outdoor only. |
| 11 | Satellite Communications | **Skip** | Not applicable. |
| 12 | RF Safety | **Skip (physics)** | Good general knowledge (FCC exposure), not engine physics. |
| App A | Probability Review | **Reference** | Pair with 8.3–8.4 if you add stochastic fading / shadowing. |

**If you read only three chapters: 2, 8, 9.** Those are the physics behind your
path-loss, reflection/refraction/absorption, and diffraction stack.

---

## Basic → advanced pipeline (and where the book helps)

A ladder from the cheapest analytic model to a full field solver. Each rung is a real
step you can ship and validate before climbing to the next. The book covers Tiers 0–2
well; Tiers 3 is numerical methods it does **not** cover (noted below).

**Tier 0 — Analytic baselines (no geometry)**
1. **FSPL** (Ch 4.2) — sanity floor. `PL = 20log₁₀d + 20log₁₀f + const`.
2. **Log-distance / ITU indoor** (Ch 9.3.3–9.3.4) — cheap `PL(x,y)` with wall/floor
   penalties. *= your path-loss layer, first cut.*

**Tier 1 — Single-interaction geometry**
3. **Two-ray ground bounce** (Ch 8.2) — first phase & interference nulls.
4. **Fresnel reflection/transmission at boundaries** (Ch 2.6) — *= reflection +
   refraction + absorption effects.*

**Tier 2 — Shadowing & corners**
5. **Fresnel zones + knife-edge / UTD diffraction** (Ch 8.2.2–8.2.4) — *= diffraction
   effect; fills the GO shadow zones your ray model leaves black.*

**Tier 3 — Field solvers (beyond this book)**
6. **Eikonal / fast-marching** arrival time `T(x,y)` — your wavefront sweep. *Numerical
   PDE — not in this book; see a comp-EM / level-set reference (e.g. Sethian, fast
   marching).*
7. **Full-wave FDTD** — the FDTD sandbox track; ground truth, most expensive. *Also
   outside this book; see Taflove & Hagness for FDTD.*

**Takeaway:** this textbook is the right reference for Tiers 0–2 (the physics + the
empirical/GO models) and for *validating* the engine against known models. For Tier 3
(eikonal solver, FDTD) reach for a numerical-methods text — the book gives you the
boundary/material physics those solvers consume, not the solvers themselves.

---

## How I'm using this

- Read a chapter → log the key equations + "what it means for the engine" in
  `RF_Propagation_Notes.md`.
- Encode each equation as a runnable cell in `RF_Equations.ipynb` (numpy + matplotlib,
  self-contained).
- Cross-check the encoded model against the engine's corresponding effect module.
