# Ch 12 — RF Safety

**Verdict: Skip (for engine physics)** · useful general knowledge, not propagation modeling.

## Sections
- 12.2 Biological Effects — general awareness.
- 12.3 FCC Guidelines / 12.5 FCC Computations — power-density exposure limits and how to
  compute them (main-beam, omni, directivity).
- 12.6 Station Evaluations — practical compliance.

## Engine relevance
Not part of the propagation stack. The one point of contact: power density
`S = EIRP / (4πr²)` is the same inverse-square spreading that underlies FSPL, so a safety
calc is a Tier-0 field-strength check near the transmitter — but it doesn't feed the map.
Worth knowing if you ever deploy real hardware; skip for the simulation work.

## Equations to encode
- (Optional) Power density `S = EIRP/(4πr²)` W/m² → `power_density()` ⚪
