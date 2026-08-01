# Ch 9 — Indoor Propagation Modeling

**★ Verdict: Essential** · **Pipeline tier 0** · literally your deployment use case.

## Sections
- 9.2 Interference — **skim**.
- 9.3 The Indoor Environment — **read**. 9.3.1 indoor propagation effects (what actually
  happens to the signal inside a building), 9.3.2 modeling overview.
- **9.3.3 ITU Indoor Path Loss Model — read.** `PL = 20log₁₀f + N·log₁₀d + Lf(n) − 28`.
- **9.3.4 Log-Distance Path Loss Model — read.** `PL(d) = PL(d₀) + 10n·log₁₀(d/d₀) + X_σ`.

## Engine relevance
The Tier-0 analytic baselines for your `PL(x,y)` layer and the cheapest sanity check on
the GO/eikonal result. The ITU coefficients map straight onto what you already have:
- `N` (distance power-loss coeff) ↔ the Motley–Keenan distance term,
- `Lf(n)` (floor-penetration) ↔ your floor-count penalty,
- and per-wall loss ↔ the Motley–Keenan wall-count sum.

Read this alongside Ch 2.6 (Fresnel): ITU/Motley–Keenan give the *empirical* per-wall dB;
Fresnel gives the *physical* per-wall dB. Cross-check them.

## Reading progress — ✅ complete (2026-08-01): book + references/exercises
**Verified both blind-seeded functions against the real text** (Ex 9.1 & 9.2 — exact match) — the
third seed confirmation after §7/§8. Bibliography + all 5 exercises → [`09_References.md`](09_References.md)
(worked in notebook §18). **Completes all three Essentials (Ch 2/8/9).**

## §17 encoded → [`../RF_Equations.ipynb`](../RF_Equations.ipynb)
- `itu_indoor_pl_db()` (eq 9.1) 🟢 — *Ex 9.1: 108 dB same floor, +16/floor.* **N=20 ⇔ free space.**
- `log_distance_pl_db()` (eq 9.2) 🟢 — *Ex 9.2: 107.5 dB @95%.* (my exponent n = book's N/10)
- **Data tables** (engine inputs): `ITU_N` (Table 9.1, N by band/env), `itu_floor_loss_db()` (Table 9.2),
  `LOGDIST_PARAMS` (Table 9.4, N & σ), `indoor_impulse_response()` (`h(t)=e^(−t/S)`), delay-spread table.
- Rappaport: indoor σ ≈ 13 dB ⇒ ±26 dB (2σ) — field measurements are the final word.

## Engine relevance (now data-backed)
Tier-0 baseline for `PL(x,y)` + sanity check on the GO/eikonal result. `N` ↔ Motley–Keenan distance
term, `Lf(n)` ↔ floor-count penalty. Read with Ch 2.6: ITU N/Lf = *empirical* per-wall dB; Fresnel =
*physical* per-wall dB — cross-check. Authoritative data sources: **ITU‑R P.1238** (indoor N/Lf/delay)
and **P.2040** (materials). The book cites further empirical models (Ericsson multi-breakpoint,
attenuation-factor) — all modified-power-law variants the two encoded models already cover.
