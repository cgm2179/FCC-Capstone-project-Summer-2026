# Ch 1 — Introduction

**Verdict: Skim (~15 min)** · framing only, no engine math.

## Sections
- 1.1 Frequency Designations — **skim** (handy band chart; keep as reference).
- 1.2 Modes of Propagation (LOS / radio horizon / non-LOS) — **read** (vocabulary).
- 1.2.3 Propagation Effects vs Frequency — **read** (why your effects scale with f).
- 1.3 Why Model Propagation? — **read** (framing).
- 1.4 Model Selection and Application — **read** — this is literally the meta-question of
  your whole track: *which* model for *which* job. 1.4.1 model sources = reference.

## Engine relevance
Low on math, high on framing. 1.4 mirrors your "basic → advanced pipeline" decision:
pick the cheapest model that captures the effect you care about.

**The framing that matters most (§1.4):** the book's models are mostly *statistical /
empirical* — mean path loss + a probabilistic margin sized to an *availability* target —
because real environments have too many unknowns to solve deterministically. Your engine is
the exception the book flags as usually-impossible: *deterministic*, and viable **only
because indoors you have the geometry + materials**. Keep both hats: run the deterministic
field solve, then add a log-normal `X_σ` margin for clutter you didn't model.

## Captured (this pass)
- **Band reference:** [`01_Frequency_Bands.md`](01_Frequency_Bands.md) — both naming
  systems, ITU Region-2 ranges, and where the engine's frequencies land.
- **Model-selection framing:** logged in [`../RF_Propagation_Notes.md`](../RF_Propagation_Notes.md)
  (Ch 1 entry) — deterministic vs. statistical/empirical.

## Equations encoded → [`../RF_Equations.ipynb`](../RF_Equations.ipynb) (§0 Foundations)
- `band_of(f)` — IEEE + letter band for a frequency (§1.1)
- `power_density(P,d)` — `S = P/(4πd²)`, the 1/d² spreading FSPL is built on (eq 1.1)
- `radio_horizon_mi(h)` — `d ≈ √(2h)`, 4/3-earth; reproduces Example 1.1 (**outdoor only**)
