#!/usr/bin/env python3
"""Enrich Data/base_stations.js with the field-decoded PCI/band assignments per site.

The 7/24 blind-scan "band" field is noisy (a single PCI shows up across many bands), so the
decoded PCI/band lists below are the ground truth for WHICH operator owns WHICH PCIs at each
site. We build one sector per (carrier, PCI): the band is the entry from that carrier's stated
band list that the walk observed STRONGEST for that PCI (falling back to the first stated band),
network + rsrp_mean + n_points come from the walk. Coordinates refined from the site photos.

Rewrites Data/base_stations.js in place (window.BASE_STATIONS). Run bake_base_stations.py after
to regenerate the Unity StreamingAssets copy, and bake_outdoor_walk.py to refresh operator tags.
"""
import re, json, os, statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BS = os.path.join(ROOT, "Data/base_stations.js")
WALK = os.path.join(ROOT, "FCC_Walk_Outdoor_Indoor_Full/Outdoor_Walk_Test_7-24/records_data.js")

# --- field-decoded ground truth (from the user) ---------------------------------------------
# site_id -> {coords:(lat,lon) or None, note, carriers:{carrier:{net, pcis:[...], bands:[...]}}}
USER = {
    "BS2_900_nj_ave": {"carriers": {
        "Verizon": {"pcis": [291, 82], "bands": [2, 5, 13, 66]},
        "T-Mobile": {"pcis": [120, 179, 142, 42, 165, 5, 244, 316, 20, 231, 233, 453, 201, 232, 463, 2, 168, 150], "bands": [41, 4, 66, 2, 12, 71]},
    }},
    "BS4_55_m_st": {"coords": (38.90587, -77.01110), "carriers": {
        "T-Mobile": {"pcis": [17, 489, 172, 272, 429, 382, 194, 300, 56, 214], "bands": [41, 66]},
    }},
    "BS5_1140_n_capitol": {"carriers": {
        "AT&T": {"pcis": [67, 185, 171, 172, 173, 293], "bands": [2, 12, 14, 30, 66]},
    }},
    "BS6_1005_n_capitol": {"carriers": {
        "T-Mobile": {"pcis": [71, 169, 256, 199, 255, 54, 194, 193, 359, 140], "bands": [12, 71, 41, 2, 4, 66]},
        "AT&T": {"pcis": [146, 411], "bands": [2, 30]},
    }},
    # Forte Hall = Gonzaga College HS building; refine coords from the site photo. Keep its sectors.
    "BS7_forte_hall": {"coords": (38.90159, -77.01134), "name": "Forte Hall (Gonzaga HS)"},
    # 400 NY Ave belongs to a bankrupt experimental licensee (Starry Spectrum LLC) — no live cells.
    "BS3_400_ny_ave": {"note": "Starry Spectrum LLC (bankrupt experimental) — no live cells", "status": "na_bankrupt"},
}

# band -> downlink centre MHz (representative)
BAND_FREQ = {("lte", 2): 1960, ("lte", 4): 2115, ("lte", 5): 869, ("lte", 12): 729, ("lte", 13): 751,
             ("lte", 14): 763, ("lte", 30): 2350, ("lte", 41): 2593, ("lte", 66): 2155, ("lte", 71): 617,
             ("nr_fr1", 5): 850, ("nr_fr1", 41): 2593, ("nr_fr1", 66): 2130, ("nr_fr1", 71): 617,
             ("nr_fr1", 77): 3700, ("nr_fr1", 78): 3500, ("nr_fr1", 26): 859}

def parse_band(b):
    """messy walk band -> (network, band_number) or None."""
    b = str(b).strip()
    m = re.match(r"^([Bn])(\d+)", b)
    if not m:
        return None
    return ("nr_fr1" if m.group(1) == "n" else "lte", int(m.group(2)))

# --- walk: pci -> {(net,band): [rsrp]} -------------------------------------------------------
recs = json.loads(re.search(r"const records\s*=\s*(\[.*\])\s*;?\s*$", open(WALK).read(), re.S).group(1))
walk = defaultdict(lambda: defaultdict(list))
for r in recs:
    try:
        pci = int(float(r.get("pci")))
    except (TypeError, ValueError):
        continue
    nb = parse_band(r.get("band"))
    rs = r.get("avg_rsrp")
    if nb and rs is not None:
        walk[pci][nb].append(float(rs))

def sector_for(carrier, pci, bands):
    """best (net,band,rsrp,n) for this pci among the carrier's stated bands, from the walk."""
    cands = []
    for (net, bn), rs in walk.get(pci, {}).items():
        if bn in bands:
            cands.append((statistics.mean(rs), net, bn, len(rs)))
    if cands:
        rsrp, net, bn, n = max(cands)                    # strongest observed band in the stated list
    else:
        allobs = [(statistics.mean(rs), net, bn, len(rs)) for (net, bn), rs in walk.get(pci, {}).items()]
        if allobs:
            rsrp, net, bn, n = max(allobs); bn = bands[0] if bands else bn   # keep a stated band label
        else:
            return {"net": "lte", "band": bands[0] if bands else 0, "rsrp": None, "n": 0}   # never seen
    return {"net": net, "band": bn, "rsrp": round(rsrp, 1), "n": n}

# --- merge into the existing catalog ---------------------------------------------------------
sites = json.loads(re.search(r"window\.BASE_STATIONS\s*=\s*(\[.*\])\s*;?\s*$", open(BS).read(), re.S).group(1))
by_id = {s["site_id"]: s for s in sites}

for sid, u in USER.items():
    s = by_id.get(sid)
    if s is None:
        continue
    if u.get("coords"):
        s["lat"], s["lon"] = u["coords"][0], u["coords"][1]
    if u.get("name"):
        s["name"] = u["name"]
    if u.get("status"):
        s["status"] = u["status"]
    if u.get("note"):
        s["note"] = u["note"]
    if "carriers" in u:
        secs = []
        for carrier, cd in u["carriers"].items():
            for pci in dict.fromkeys(cd["pcis"]):          # dedupe, keep order
                info = sector_for(carrier, pci, cd["bands"])
                secs.append({"carrier": carrier, "network": info["net"], "band": info["band"], "pci": pci,
                             "azimuth_deg": None, "freq_mhz": BAND_FREQ.get((info["net"], info["band"]), 2000),
                             "rsrp_mean": info["rsrp"], "n_points": info["n"], "bands_all": cd["bands"]})
        s["sectors"] = secs
        s["status"] = s.get("status", "live") if s.get("status") != "na_bankrupt" else "na_bankrupt"

out = "// Generated by build_catalog.py, enriched by scripts/enrich_base_stations.py with the\n" \
      "// field-decoded PCI/band assignments per site (operator ground truth; bands/rsrp from the 7/24 walk).\n" \
      "window.BASE_STATIONS = " + json.dumps(sites, separators=(",", ":")) + ";"
open(BS, "w").write(out + "\n")

for s in sites:
    ns = len(s.get("sectors", []))
    print(f"  {s['site_id']:20s} {s.get('name','')[:26]:26s} sectors={ns}  ({s['lat']:.5f},{s['lon']:.5f})")
print("wrote", BS)
