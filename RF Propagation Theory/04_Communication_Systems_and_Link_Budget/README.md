# Ch 4 — Communication Systems & the Link Budget

**Verdict: Important** · **Pipeline tier 0** · how `PL(x,y)` becomes usable coverage.

## Sections
- 4.2 Path Loss — **read** (Friis, free-space; sets the baseline your map rides on).
- 4.3 Noise — **skim** (thermal noise `kTB`, noise figure; needed only for SNR).
- 4.4 Interference — **skim**.
- 4.5 Detailed Link Budget — **read**. 4.5.1 EIRP, 4.5.2 path loss, 4.5.3 Rx gain,
  4.5.4 link margin, 4.5.5 SNR.

## Engine relevance
This is the "so what" layer. Your engine outputs `PL(x,y)`; the link budget converts it
to received power, SNR, and a coverage/no-coverage boundary. **EIRP is your source-strength
input**; link margin is the threshold that turns the loss surface into a coverage map.

## Reading progress — ✅ complete (2026-07-31)
- ✅ **Part 1 §4.1–4.3** — link margin, path loss (Friis), noise floor / noise figure.
- ✅ **Part 2 §4.4–4.6** — interference margin, detailed link budget (Fig 4.4), Eb/N0.

## Key result — coverage = margin > 0
`M(x,y) = EIRP − PL(x,y) + G_Rx − TH_Rx`. The engine makes `PL(x,y)`; EIRP & G_Rx come from Ch 3;
`TH_Rx = (−174 dBm/Hz + 10log B + NF) + SNR_req` comes from §4.3. Render `M > 0` for coverage.

## Equations encoded → [`../RF_Equations.ipynb`](../RF_Equations.ipynb) §10–11
- §10 `link_margin_db()`, `friis_received_power_dbm()` (Friis, eq 4.1) 🟢
- §10 `thermal_noise_dbm()` (−174 dBm/Hz + 10log B + NF), `noise_figure_db_from_temp()` (1 + Te/T₀) 🟢
- §10 `cascade_noise_factor()` — Friis cascade (correct −1 form) 🟢
- §11 `link_budget()` (full itemized — reproduces all 9 totals of Fig 4.4),
  `interference_for_margin_dbm()` (Ex 4.4 → 5.9 dB), `eb_n0_db()` (SNR + 10log B/Rb) 🟢
- Verified vs book: Ex 4.1 (22.6 dB margin), Ex 4.2 (6 dB NF, −128 dBW), Ex 4.3 (12 dB system NF),
  Ex 4.4 (5.9 dB), Fig 4.4 (all 9 totals).

### ⚠ Two book slips caught this chapter
- **eq 4.14 (noise cascade)** drops the `−1`; the book's own Ex 4.3 needs it (→ 12 dB, not 13.2).
- **inline §4.5.2–4.5.5 numbers** (PL 135, EIRP 27, SNR 16, 10log B = 60) are self-inconsistent and
  mismatch Fig 4.4. Both fixed — notebook uses the correct/consistent forms.
