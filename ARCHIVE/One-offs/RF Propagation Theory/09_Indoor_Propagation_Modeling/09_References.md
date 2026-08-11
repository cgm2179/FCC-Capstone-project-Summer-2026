# Ch 9 — References & Exercises (reference content)

## Key references (source models)
| # | Reference | Relevance to the engine |
|---|-----------|--------------------------|
| 3 | **ITU‑R P.1238‑2** — Propagation data & prediction methods for indoor systems, 900 MHz–100 GHz (Geneva, 2001) | **The authoritative source** for the ITU indoor model (eq 9.1), the N & Lf tables, and the delay-spread table. Pull real numbers here — newer editions exist (P.1238‑11). |
| 4 | **Rappaport**, *Wireless Communications: Principles and Practice*, 2nd ed. (2002), pp. 161–166 | The log-distance model (eq 9.2) + Table 9.4 (N, σ by building type). The standard reference text. |
| 5 | Hashemi, "The indoor radio propagation channel," *Proc. IEEE* 81(7), 1993 | Classic survey of indoor channel characterization (delay spread, statistics). |
| 6 | Kivinen, Zhao, Vainikainen, "Empirical characterization of wideband indoor radio channel at 5.3 GHz," *IEEE T‑AP* 49(8), 2001 | Wideband 5 GHz measurements — directly relevant to the Wi‑Fi bands. |
| 2 | Dobkin, "Indoor propagation issues for wireless LANs," *RF Design*, Sep 2002 | Practitioner overview for WLANs. |
| 1 | Cripps, *RF Power Amplifiers for Wireless Communications* (1999), Ch 7 | Intermod / nonlinearity — the §9.2 interference context. |

**Two ITU recommendations the engine should consult for real numbers:**
- **P.1238** — indoor path-loss coefficients N, floor-penetration Lf, delay spread *(this chapter)*.
- **P.2040** — building-material εr, σ vs frequency *(the Ch 2 physics counterpart)*.

## Exercises — worked (code in notebook §18)
| # | Problem | Answer |
|---|---------|--------|
| 9‑1 | ITU median PL, 1.9 GHz office, 100 m (N=30) | **97.6 dB** |
| 9‑2 | PL vs probability of occurrence | plot: median 97.6 dB ± Φ⁻¹(p)·σ (σ≈8 dB assumed — ITU gives none) |
| 9‑3 | Log-distance, 1.9 GHz office **soft** partition, 98% @ 100 m (N=26, σ=14.1) | **119 dB** (median 90 + 29 dB shadow margin) |
| 9‑4 | Log-distance, 900 MHz office **hard** partition, 38 m, min/max @ 99% (N=30, σ=7) | median 78.9 dB; central‑99% **[60.9, 96.9] dB** (±18) |
| 9‑5 | Max symbol rate with no equalizer (Table 9.3) | worst-case S=500 ns → **≤ 400 ksps** (coherence-BW rule) or **≤ 200 ksps** (10%-of-symbol rule) |

**Notes / assumptions:**
- 9‑2: the ITU model specifies no σ, so a typical office value was borrowed for the log-normal spread.
- 9‑4: Table 9.4's only hard-partition entry is at 1500 MHz (N=30, σ=7); used here for the 900 MHz
  office-hard case (the table has no 900 MHz hard-partition row).
- 9‑5: without an equalizer the channel must stay flat (signal BW ≤ coherence BW), so pick the symbol
  period ≫ delay spread; using the *largest* tabulated S is the safe/conservative choice.
- The huge σ = 14.1 dB in 9‑3 (soft partition, 1.9 GHz) is why indoor links carry large fade margins.
