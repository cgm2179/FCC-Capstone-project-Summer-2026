# §1.1 Frequency Band Designations — reference

**RF (this book):** 1 MHz – 300 GHz. **Microwave** 1–30 GHz · **millimeter-wave (MMW)**
30–300 GHz. (Common industry usage sometimes calls only 1 MHz–1 GHz "RF"; Seybold uses the
wider HF–EHF span.)

## IEEE / spectrum bands (Table 1.1)
| Band | Abbr | Range |
|------|------|-------|
| Extremely low | ELF | < 3 kHz |
| Very low | VLF | 3–30 kHz |
| Low | LF | 30–300 kHz |
| Medium | MF | 300 kHz – 3 MHz |
| High | HF | 3–30 MHz |
| Very high | VHF | 30–300 MHz |
| Ultra-high | UHF | 300 MHz – 3 GHz |
| Super-high | SHF | 3–30 GHz |
| Extra-high | EHF | 30–300 GHz |

(Each spectrum band spans exactly one decade of frequency.)

## Radar / microwave letter bands (Table 1.2, nominal)
| Label | Nominal range | ITU Region 2 (examples) |
|-------|---------------|-------------------------|
| HF | 3–30 MHz | — |
| VHF | 30–300 MHz | 138–145, 216–225 MHz |
| UHF | 300–1000 MHz | 420–450, 890–942 MHz |
| L | 1–2 GHz | 1215–1400 MHz |
| S | 2–4 GHz | 2.3–2.5, 2.7–3.7 GHz |
| C | 4–8 GHz | 5.25–5.925 GHz |
| X | 8–12 GHz | 8.5–10.68 GHz |
| Ku | 12–18 GHz | 13.4–14, 15.7–17.7 GHz |
| K | 18–27 GHz | 24.05–24.25 GHz |
| Ka | 27–40 GHz | 33.4–36 GHz |
| R / Q / V / W | 26.5–40 / 33–50 / 40–75 / 75–110 GHz | — |

> Note the overlap: the letter bands and the IEEE spectrum bands are two different naming
> systems for the same spectrum. A 2.4 GHz signal is **UHF** (spectrum) *and* **S-band** (letter).

## Where the engine's frequencies land
| System | Frequency | IEEE | Letter |
|--------|-----------|------|--------|
| Wi‑Fi 2.4 / BLE / Zigbee / Matter | 2.4 GHz | UHF | S |
| Wi‑Fi 5 / 6 | 5–6 GHz | SHF | C |
| Sub‑6 5G / LTE | 0.6–6 GHz | UHF–SHF | L / S / C |
| mmWave 5G / FWA | 24–40 GHz | SHF | K / Ka |

*(Confirm any value with `band_of(f_hz)` in `../RF_Equations.ipynb`, §0.)*

## Why this matters for the engine
The intro's rule of thumb (§1.2.2.1): **lower frequency penetrates** walls/foliage more
easily; **higher frequency reflects/scatters/diffracts more and penetrates less** — "above
UHF, indirect propagation becomes very inefficient," and once obstacle features are large
vs. λ they **reflect/diffract** rather than scatter.

Consequences for your effect stack:
- **2.4–6 GHz (your main indoor case):** transmission (wall penetration), reflection, and
  diffraction are **all** first-order. None is safe to drop.
- **mmWave (24–40 GHz):** penetration collapses → LOS + reflection + diffraction dominate;
  transmission through walls is nearly a hard block.

λ for reference: 2.4 GHz → 12.5 cm · 5.5 GHz → 5.5 cm · 28 GHz → 1.1 cm. Feature-vs‑λ
(the reflect/scatter threshold) shifts hard across your band range — the same wall texture
is "smooth" at 2.4 GHz and "rough" at 28 GHz (see Ch 8.2.1 roughness).
