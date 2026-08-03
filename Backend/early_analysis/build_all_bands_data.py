#!/usr/bin/env python3
"""Rebuild the Indoor dashboard data from EVERY band picked up in the walk test.

The original build_interactive_dashboard.py only ingested the NR "Top N Signal"
files, so records_data.js carried just 4 T-Mobile NR bands (2/25/41/71). This
generator instead reads the *Blind Scan* CSVs — which uniformly cover every LTE
E-UTRA band and every 5G-NR band the scanner detected — so the dashboard's
Network -> Band -> PCI -> Channel cascade and the coverage map include Band 13
(and all the rest), not just the featured NR carrier.

Outputs (consumed as-is by Frontend/2d-3d/dashboard.js):
  Data/records_data.js       const records = [...]                 (averaged per point)
  Data/timeseries_data.js    const timeseriesStartTime / Records   (per-sample, time-ordered)

Georeferencing (lon/lat -> px,py) uses the same 3 GCP similarity fit as the
floor-plan meta, verified to reproduce the existing px/py to 0.00 px.
"""
from __future__ import annotations
import csv, glob, json, os, re
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

ROOT = "/Users/cameronmickle/Documents/Indoor_Walk_Test_7-7"
CSV_DIR = os.path.join(ROOT, "Data/raw_walk_data/CSV")
OUT_RECORDS = os.path.join(ROOT, "Data/records_data.js")
OUT_TIMESERIES = os.path.join(ROOT, "Data/timeseries_data.js")

# --- georeference: 3 ground-control points from floorplan_meta.js -------------
GCPS = [(-77.0070535, 38.9036891, 974.0, 47.0),
        (-77.0077309, 38.9034977, 155.0, 468.0),
        (-77.0074321, 38.9035672, 557.0, 268.0)]
_A = np.array([[g[0], g[1], 1] for g in GCPS])
COEF_X = np.linalg.lstsq(_A, np.array([g[2] for g in GCPS]), rcond=None)[0]
COEF_Y = np.linalg.lstsq(_A, np.array([g[3] for g in GCPS]), rcond=None)[0]

def to_px(lon, lat):
    return (float(COEF_X[0]*lon + COEF_X[1]*lat + COEF_X[2]),
            float(COEF_Y[0]*lon + COEF_Y[1]*lat + COEF_Y[2]))

# --- LTE EARFCN -> DL centre frequency (MHz), 3GPP TS 36.101 ------------------
# band: (F_DL_low MHz, N_Offs-DL).  f = F_DL_low + 0.1*(EARFCN - N_Offs-DL)
LTE_BAND = {
    1:(2110,0), 2:(1930,600), 3:(1805,1200), 4:(2110,1950), 5:(869,2400),
    7:(2620,2750), 8:(925,3450), 10:(2110,4150), 12:(729,5010), 13:(746,5180),
    14:(758,5280), 17:(734,5730), 18:(860,5850), 19:(875,6000), 20:(791,6150),
    21:(1495.9,6450), 25:(1930,8040), 26:(859,8690), 27:(852,9040), 28:(758,9210),
    29:(717,9660), 30:(2350,9770), 31:(462.5,9870), 32:(1452,9920), 38:(2570,37750),
    39:(1880,38250), 40:(2300,38650), 41:(2496,39650), 46:(5150,46790), 48:(3550,55240),
    65:(2110,65536), 66:(2110,66436), 67:(738,67336), 68:(753,67536), 71:(617,68586),
    85:(728,70366), 252:(5150,255144), 255:(5725,260894),
}
def lte_freq_mhz(band, earfcn):
    try:
        earfcn = float(earfcn)
    except (TypeError, ValueError):
        return None
    if band in LTE_BAND:
        lo, noffs = LTE_BAND[band]
        return round(lo + 0.1*(earfcn - noffs), 2)
    return round(earfcn, 2)  # fallback: raw channel number

CARRIER = {(311,480):"Verizon", (310,260):"T-Mobile", (310,410):"AT&T",
           (310,280):"AT&T", (311,180):"AT&T", (312,530):"Dish"}

def band_network(fn):
    b = os.path.basename(fn)
    m = re.search(r"_LTE_EB\s+(\d+)", b)
    if m:
        return int(m.group(1)), "lte"
    m = re.search(r"[ _]n(\d+)(?=[ _]|$)", b)
    if m and "_NR_" in b:
        n = int(m.group(1))
        return n, ("nr_fr2" if n >= 257 else "nr_fr1")
    return None, None

def find_col(header, *names):
    low = [h.strip().lower() for h in header]
    for n in names:
        if n in low:
            return low.index(n)
    return None

def fnum(row, idx):
    if idx is None or idx >= len(row):
        return None
    s = row[idx].strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None

def parse_t(date_s, time_s):
    # Date "07/07/2026", Time "14:02:57:051" (HH:MM:SS:mmm), local clock treated as UTC
    try:
        hh, mm, ss, ms = time_s.strip().split(":")
        mo, dd, yy = date_s.strip().split("/")
        return datetime(int(yy), int(mo), int(dd), int(hh), int(mm), int(ss),
                        int(ms)*1000, tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None

def main():
    files = [f for f in sorted(glob.glob(os.path.join(CSV_DIR, "*.CSV")))
             if "Blind Scan" in os.path.basename(f)
             and "_Spectrum" not in os.path.basename(f)]

    samples = []  # every decoded-cell measurement
    for fn in files:
        band, network = band_network(fn)
        if band is None:
            continue
        with open(fn, newline="", encoding="latin-1") as fh:
            r = csv.reader(fh)
            header = next(r)
            pci_i = find_col(header, "cell id")
            if pci_i is None:
                continue
            lat_i = find_col(header, "latitude"); lon_i = find_col(header, "longitude")
            rsrp_i = find_col(header, "ssb - received power", "ref signal - received power")
            rsrq_i = find_col(header, "ssb - received quality", "ref signal - received quality")
            cinr_i = find_col(header, "ssb - cinr", "ref signal - cinr", "sync channel - cinr")
            rssi_i = find_col(header, "ssb rssi", "channel rssi")
            freq_i = find_col(header, "channel frequency")          # NR: MHz
            earfcn_i = find_col(header, "channel number")           # LTE: EARFCN
            mcc_i = find_col(header, "mcc"); mnc_i = find_col(header, "mnc")
            date_i = find_col(header, "date"); time_i = find_col(header, "time")
            for row in r:
                if pci_i >= len(row) or row[pci_i].strip() == "":
                    continue
                lat = fnum(row, lat_i); lon = fnum(row, lon_i)
                if lat is None or lon is None:
                    continue
                pci = fnum(row, pci_i)
                freq = fnum(row, freq_i)
                if freq is None:
                    freq = lte_freq_mhz(band, row[earfcn_i]) if earfcn_i is not None and earfcn_i < len(row) else None
                px, py = to_px(lon, lat)
                mcc = fnum(row, mcc_i); mnc = fnum(row, mnc_i)
                carrier = CARRIER.get((int(mcc), int(mnc))) if mcc and mnc else None
                samples.append({
                    "network": network, "band": band,
                    "pci": int(pci) if pci is not None else None,
                    "freq": freq,
                    "rsrp": fnum(row, rsrp_i), "rsrq": fnum(row, rsrq_i),
                    "cinr": fnum(row, cinr_i), "rssi": fnum(row, rssi_i),
                    "latitude": lat, "longitude": lon, "px": px, "py": py,
                    "mcc": int(mcc) if mcc else None, "mnc": int(mnc) if mnc else None,
                    "carrier": carrier,
                    "_t": parse_t(row[date_i] if date_i is not None else "",
                                  row[time_i] if time_i is not None else ""),
                })

    # ---- records: average metrics per (network, band, pci, coordinate) -------
    groups = defaultdict(list)
    for s in samples:
        groups[(s["network"], s["band"], s["pci"], round(s["latitude"], 6), round(s["longitude"], 6))].append(s)

    def avg(vals):
        v = [x for x in vals if x is not None]
        return round(sum(v)/len(v), 3) if v else None

    records = []
    for (network, band, pci, lat, lon), grp in groups.items():
        g0 = grp[0]
        records.append({
            "latitude": lat, "longitude": lon,
            "px": round(g0["px"], 4), "py": round(g0["py"], 4),
            "rsrp": avg([g["rsrp"] for g in grp]), "rsrq": avg([g["rsrq"] for g in grp]),
            "cinr": avg([g["cinr"] for g in grp]), "rssi": avg([g["rssi"] for g in grp]),
            "n_samples": len(grp), "pci": pci,
            "freq": g0["freq"], "network": network, "band": band,
            "mcc": g0["mcc"], "mnc": g0["mnc"], "carrier": g0["carrier"],
        })

    # ---- timeseries: per-sample, time-ordered --------------------------------
    timed = [s for s in samples if s["_t"] is not None]
    timed.sort(key=lambda s: s["_t"])
    start = timed[0]["_t"] if timed else None
    ts = []
    for s in timed:
        ts.append({
            "network": s["network"], "band": s["band"], "pci": s["pci"], "freq": s["freq"],
            "rsrp": s["rsrp"], "rsrq": s["rsrq"], "cinr": s["cinr"], "rssi": s["rssi"],
            "latitude": round(s["latitude"], 6), "longitude": round(s["longitude"], 6),
            "px": round(s["px"], 4), "py": round(s["py"], 4),
            "carrier": s["carrier"],
            "t_sec": round((s["_t"] - start).total_seconds(), 3),
        })

    with open(OUT_RECORDS, "w", encoding="utf-8") as f:
        f.write("    const records = " + json.dumps(records, ensure_ascii=True) + ";\n")
    with open(OUT_TIMESERIES, "w", encoding="utf-8") as f:
        f.write('const timeseriesStartTime = "%s";\n' % start.isoformat().replace("+00:00", "Z"))
        f.write("const timeseriesRecords = " + json.dumps(ts, ensure_ascii=True) + ";\n")

    # ---- report --------------------------------------------------------------
    from collections import Counter
    print(f"blind-scan files ingested : {len(files)}")
    print(f"decoded-cell samples      : {len(samples)}")
    print(f"records (averaged points) : {len(records)}")
    print(f"timeseries samples        : {len(ts)}")
    nb = Counter((r["network"], r["band"]) for r in records)
    print("\nnetwork / band -> points:")
    for (net, band), c in sorted(nb.items(), key=lambda x: (x[0][0], x[0][1])):
        print(f"   {net:7} band {str(band):>4} : {c}")
    car = Counter(r["carrier"] for r in records if r["carrier"])
    print("\ncarriers decoded (via MCC/MNC):", dict(car))
    b13 = [r for r in records if r["network"] == "lte" and r["band"] == 13]
    print(f"\nBand 13 (LTE) points: {len(b13)}  PCIs: {sorted(set(r['pci'] for r in b13))}")

if __name__ == "__main__":
    main()
