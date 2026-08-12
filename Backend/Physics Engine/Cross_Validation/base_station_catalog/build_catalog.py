#!/usr/bin/env python3
"""Build the base-station catalog (PR #24).

Joins the hand-authored site registry (sites_registry.json) to the measured
walk-scan RF stats and emits, in this folder:

  base_stations.csv              one row per (site, carrier, band, PCI) sector-band
  bs_<slug>.csv                  the same rows split per site
  unregistered_measured_cells.csv  measured (network,band,PCI) cells not in the registry
  osm_heights_cache.json         Overpass results cache (so re-runs are offline)

Each row carries the antenna's 3D position (lat/lon + OSM/manual building height),
its identification (carrier/network/band/PCI, decoded eNB-ID/sector), the measured
RSRP/RSRQ/SINR/RSSI stats attributed to that cell, and the site->floorplan geometry
(distance_m, bearing_deg) using the SAME local-ENU convention as
SIM V1 3D/modes_3d.forte_hall_geometry, so it drops straight into the O2I engine.

Reuses:
  ARCHIVE/EARLY Analysis/build_all_bands_data.py  (scan parsing + georeference)
  Physics Engine/3D Map Physics/SIM V1 3D/osm_building_height.py  (height resolver)

Usage:
  python build_catalog.py                # full build (queries OSM once, then caches)
  python build_catalog.py --no-network   # offline: manual/cached heights only
  python build_catalog.py --test         # quick self-test (geometry + registry parse)
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                                  # -> repo root
BACKEND = REPO / "ARCHIVE" / "EARLY Analysis"
SIM3D = REPO / "Physics Engine" / "3D Map Physics" / "SIM V1 3D"
REGISTRY = HERE / "sites_registry.json"
FLOORPLAN_META_JS = REPO / "Data" / "floorplan_meta.js"
CACHE_PATH = HERE / "osm_heights_cache.json"

sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(SIM3D))
import build_all_bands_data as bab      # noqa: E402  (scan parsing + georeference)
import osm_building_height as obh        # noqa: E402  (height resolver)

R_EARTH = 6371000.0

FIELDS = [
    "site_id", "name", "address", "lat", "lon",
    "building_height_m", "antenna_agl_m", "elev_source", "elev_confidence",
    "carrier", "network", "band", "channel", "freq_mhz", "pci",
    "antenna_id", "antenna_type", "azimuth_deg", "azimuth_source", "azimuth_confidence",
    "cellidentity", "enb_gnb_id", "sector_id",
    "n_points", "rsrp_min", "rsrp_max", "rsrp_mean", "rsrp_std",
    "rsrq_mean", "sinr_mean", "rssi_mean",
    "distance_m", "bearing_deg", "status", "carrier_match", "notes",
]


# ------------------------------------------------------------------ geometry
def load_floorplan_meta() -> dict:
    txt = FLOORPLAN_META_JS.read_text(encoding="utf-8")
    i, j = txt.index("{"), txt.rindex("}")
    return json.loads(txt[i:j + 1])


def _enu(lon, lat, lon0, lat0):
    x = math.radians(lon - lon0) * R_EARTH * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * R_EARTH
    return x, y


def site_geometry(lon, lat, meta) -> tuple[float, float]:
    """(distance_m, arrival_bearing_deg) from the floor-plan centre to (lon,lat).

    Mirrors SIM V1 3D/modes_3d.forte_hall_geometry exactly: local ENU about
    floorplan_meta.origin_lonlat, building centre from affine_px_to_local_m at the
    raster centre, arrival bearing = compass direction the wave arrives FROM.
    """
    lon0, lat0 = meta["origin_lonlat"]
    A = meta["affine_px_to_local_m"]
    H, W = meta["grid_shape"]
    cx, cy = W / 2.0, H / 2.0
    bx = A[0][0] * cx + A[0][1] * cy + A[0][2]
    by = A[1][0] * cx + A[1][1] * cy + A[1][2]
    tx_e, tx_n = _enu(lon, lat, lon0, lat0)
    dE, dN = tx_e - bx, tx_n - by
    dist = math.hypot(dE, dN)
    arrival = math.degrees(math.atan2(dE, dN)) % 360.0
    return round(dist, 1), round(arrival, 1)


# ------------------------------------------------------------------ scan parsing
def collect_samples() -> list[dict]:
    """Every decoded-cell measurement from the indoor blind-scan CSVs, plus the
    raw cellIdentity (ECI/NCI) and channel number that build_all_bands drops."""
    files = [f for f in sorted(glob.glob(os.path.join(bab.CSV_DIR, "*.CSV")))
             if "Blind Scan" in os.path.basename(f)
             and "_Spectrum" not in os.path.basename(f)]
    samples = []
    for fn in files:
        band, network = bab.band_network(fn)
        if band is None:
            continue
        with open(fn, newline="", encoding="latin-1") as fh:
            r = csv.reader(fh)
            header = next(r)
            pci_i = bab.find_col(header, "cell id")
            if pci_i is None:
                continue
            lat_i = bab.find_col(header, "latitude"); lon_i = bab.find_col(header, "longitude")
            rsrp_i = bab.find_col(header, "ssb - received power", "ref signal - received power")
            rsrq_i = bab.find_col(header, "ssb - received quality", "ref signal - received quality")
            cinr_i = bab.find_col(header, "ssb - cinr", "ref signal - cinr", "sync channel - cinr")
            rssi_i = bab.find_col(header, "ssb rssi", "channel rssi")
            freq_i = bab.find_col(header, "channel frequency")     # NR: MHz
            earfcn_i = bab.find_col(header, "channel number")      # LTE: EARFCN
            mcc_i = bab.find_col(header, "mcc"); mnc_i = bab.find_col(header, "mnc")
            eci_i = bab.find_col(header, "cellidentity", "cell identity")
            for row in r:
                if pci_i >= len(row) or row[pci_i].strip() == "":
                    continue
                lat = bab.fnum(row, lat_i); lon = bab.fnum(row, lon_i)
                if lat is None or lon is None:
                    continue
                pci = bab.fnum(row, pci_i)
                freq = bab.fnum(row, freq_i)
                channel = bab.fnum(row, earfcn_i)
                if freq is None and channel is not None:
                    freq = bab.lte_freq_mhz(band, channel)
                mcc = bab.fnum(row, mcc_i); mnc = bab.fnum(row, mnc_i)
                carrier = bab.CARRIER.get((int(mcc), int(mnc))) if mcc and mnc else None
                eci = bab.fnum(row, eci_i)
                samples.append({
                    "network": network, "band": band,
                    "pci": int(pci) if pci is not None else None,
                    "freq": freq, "channel": int(channel) if channel is not None else None,
                    "rsrp": bab.fnum(row, rsrp_i), "rsrq": bab.fnum(row, rsrq_i),
                    "cinr": bab.fnum(row, cinr_i), "rssi": bab.fnum(row, rssi_i),
                    "latitude": lat, "longitude": lon,
                    "carrier": carrier,
                    "cellidentity": int(eci) if eci is not None else None,
                })
    return samples


def _avg(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def build_records(samples) -> list[dict]:
    """Per-coordinate averaged records — identical grouping to build_all_bands_data,
    so counts match the dashboard's per-PCI point totals."""
    groups = defaultdict(list)
    for s in samples:
        key = (s["network"], s["band"], s["pci"],
               round(s["latitude"], 6), round(s["longitude"], 6))
        groups[key].append(s)
    records = []
    for (network, band, pci, lat, lon), grp in groups.items():
        records.append({
            "network": network, "band": band, "pci": pci,
            "latitude": lat, "longitude": lon,
            "rsrp": _avg([g["rsrp"] for g in grp]),
            "rsrq": _avg([g["rsrq"] for g in grp]),
            "cinr": _avg([g["cinr"] for g in grp]),
            "rssi": _avg([g["rssi"] for g in grp]),
            "freq": grp[0]["freq"], "channel": grp[0]["channel"],
            "carrier": grp[0]["carrier"], "cellidentity": grp[0]["cellidentity"],
        })
    return records


def _stats(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {"min": None, "max": None, "mean": None, "std": None}
    m = sum(v) / len(v)
    var = sum((x - m) ** 2 for x in v) / len(v) if len(v) > 1 else 0.0
    return {"min": round(min(v), 2), "max": round(max(v), 2),
            "mean": round(m, 2), "std": round(var ** 0.5, 2)}


def _modal(vals):
    v = [x for x in vals if x is not None]
    return Counter(v).most_common(1)[0][0] if v else None


def aggregate_by_cell(records) -> dict:
    """(network, band, pci) -> aggregate over per-coordinate records."""
    cells = defaultdict(list)
    for r in records:
        cells[(r["network"], r["band"], r["pci"])].append(r)
    out = {}
    for key, grp in cells.items():
        st = _stats([g["rsrp"] for g in grp])
        out[key] = {
            "n_points": len(grp),
            "rsrp": st,
            "rsrq_mean": _stats([g["rsrq"] for g in grp])["mean"],
            "sinr_mean": _stats([g["cinr"] for g in grp])["mean"],
            "rssi_mean": _stats([g["rssi"] for g in grp])["mean"],
            "freq": _modal([g["freq"] for g in grp]),
            "channel": _modal([g["channel"] for g in grp]),
            "carriers": Counter(g["carrier"] for g in grp if g["carrier"]),
            "cellidentity": _modal([g["cellidentity"] for g in grp]),
        }
    return out


def decode_cellid(network, cellidentity):
    """LTE ECI -> (eNB-ID, sector). NR NCI split is operator-specific -> leave None."""
    if cellidentity is None:
        return None, None
    if network == "lte":
        return cellidentity >> 8, cellidentity & 0xFF
    return None, None   # NR: NCI recorded in cellidentity, gNB/cell split unknown


# ------------------------------------------------------------------ build
def build(*, use_network=True):
    registry = json.loads(REGISTRY.read_text())
    meta = load_floorplan_meta()
    samples = collect_samples()
    records = build_records(samples)
    cells = aggregate_by_cell(records)
    matched_keys = set()

    cache = obh.load_cache(CACHE_PATH)
    rows = []

    for site in registry["sites"]:
        lat, lon = site.get("lat"), site.get("lon")
        status = site.get("status", "unknown")

        # geometry + height (per site)
        if lat is not None and lon is not None:
            dist, bearing = site_geometry(lon, lat, meta)
        else:
            dist, bearing = None, None
        h = site.get("height", {}) or {}
        if lat is not None and lon is not None:
            hres = obh.resolve_building_height(
                lon, lat, manual_m=h.get("manual_m"),
                prefer_manual=bool(h.get("prefer_manual")),
                use_network=use_network, cache=cache)
            building_h = hres["height_m"]
            elev_source = hres["source"]
            elev_conf = hres["confidence"]
        else:
            building_h, elev_source, elev_conf = None, "none", h.get("confidence", "none")

        base = {
            "site_id": site["site_id"], "name": site.get("name"),
            "address": site.get("address"), "lat": lat, "lon": lon,
            "building_height_m": building_h, "antenna_agl_m": building_h,  # rooftop mount
            "elev_source": elev_source, "elev_confidence": elev_conf,
            "distance_m": dist, "bearing_deg": bearing, "status": status,
        }

        for cg in site.get("carriers", []):
            carrier = cg.get("carrier")
            declared_bands = set(cg.get("bands", []))
            ant_by_pci = {a["pci"]: a for a in cg.get("antennas", [])}
            group_ant_type = cg.get("antenna_type", "unknown")

            for pci in cg.get("pcis", []):
                ant = ant_by_pci.get(pci, {})
                # candidate measured cells for this PCI (any network/band)
                cands = [(net, band, p) for (net, band, p) in cells if p == pci]
                emitted = False
                for key in cands:
                    net, band, _ = key
                    if declared_bands and band not in declared_bands:
                        continue
                    cell = cells[key]
                    seen = cell["carriers"]
                    if carrier in seen:
                        cmatch = "exact"
                    elif not seen:
                        cmatch = "pci_only"
                    else:
                        cmatch = "conflict"   # PCI reused by a different decoded carrier
                    matched_keys.add(key)
                    eci = cell["cellidentity"]
                    enb, sector = decode_cellid(net, eci)
                    rows.append({**base,
                        "carrier": carrier, "network": net, "band": band,
                        "channel": cell["channel"], "freq_mhz": cell["freq"], "pci": pci,
                        "antenna_id": ant.get("antenna_id"),
                        "antenna_type": ant.get("antenna_type", group_ant_type),
                        "azimuth_deg": ant.get("azimuth_deg"),
                        "azimuth_source": ant.get("azimuth_source"),
                        "azimuth_confidence": ant.get("azimuth_confidence"),
                        "cellidentity": eci, "enb_gnb_id": enb, "sector_id": sector,
                        "n_points": cell["n_points"],
                        "rsrp_min": cell["rsrp"]["min"], "rsrp_max": cell["rsrp"]["max"],
                        "rsrp_mean": cell["rsrp"]["mean"], "rsrp_std": cell["rsrp"]["std"],
                        "rsrq_mean": cell["rsrq_mean"], "sinr_mean": cell["sinr_mean"],
                        "rssi_mean": cell["rssi_mean"],
                        "carrier_match": cmatch, "notes": ant.get("note", ""),
                    })
                    emitted = True
                if not emitted:
                    # declared sector never observed in the walk — keep it, stats empty
                    rows.append({**base,
                        "carrier": carrier, "network": None, "band": None,
                        "channel": None, "freq_mhz": None, "pci": pci,
                        "antenna_id": ant.get("antenna_id"),
                        "antenna_type": ant.get("antenna_type", group_ant_type),
                        "azimuth_deg": ant.get("azimuth_deg"),
                        "azimuth_source": ant.get("azimuth_source"),
                        "azimuth_confidence": ant.get("azimuth_confidence"),
                        "cellidentity": None, "enb_gnb_id": None, "sector_id": None,
                        "n_points": 0, "rsrp_min": None, "rsrp_max": None,
                        "rsrp_mean": None, "rsrp_std": None, "rsrq_mean": None,
                        "sinr_mean": None, "rssi_mean": None,
                        "carrier_match": "none", "notes": ant.get("note", ""),
                    })

        if not site.get("carriers"):   # excluded site (na_tv / na_bankrupt) — one marker row
            rows.append({**base, "carrier": None, "network": None, "band": None,
                "channel": None, "freq_mhz": None, "pci": None, "antenna_id": None,
                "antenna_type": None, "azimuth_deg": None, "azimuth_source": None,
                "azimuth_confidence": None, "cellidentity": None, "enb_gnb_id": None,
                "sector_id": None, "n_points": 0, "rsrp_min": None, "rsrp_max": None,
                "rsrp_mean": None, "rsrp_std": None, "rsrq_mean": None, "sinr_mean": None,
                "rssi_mean": None, "carrier_match": "none",
                "notes": site.get("notes", "")})

    obh.save_cache(CACHE_PATH, cache)

    # unregistered measured cells (transparency)
    unreg = []
    for key, cell in cells.items():
        if key in matched_keys:
            continue
        net, band, pci = key
        unreg.append({
            "network": net, "band": band, "pci": pci,
            "freq_mhz": cell["freq"], "channel": cell["channel"],
            "carriers": "|".join(sorted(cell["carriers"])) or None,
            "n_points": cell["n_points"],
            "rsrp_mean": cell["rsrp"]["mean"], "rsrp_min": cell["rsrp"]["min"],
            "rsrp_max": cell["rsrp"]["max"], "cellidentity": cell["cellidentity"],
        })
    unreg.sort(key=lambda r: (-(r["n_points"] or 0), str(r["network"]), r["band"] or 0))
    return rows, unreg, cells


def _write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _slug(site_id):
    return site_id.split("_", 1)[-1] if "_" in site_id else site_id


def write_base_stations_js(rows, path):
    """Compact `window.BASE_STATIONS` sidecar for the frontend OSM map layer
    (pins + Forte Hall bearing line). A file:// page can't fetch the CSV, so this
    mirrors the records_data.js / floorplan_meta.js <script> convention."""
    by_site = {}
    for r in rows:
        s = by_site.setdefault(r["site_id"], {
            "site_id": r["site_id"], "name": r["name"], "lat": r["lat"], "lon": r["lon"],
            "status": r["status"], "distance_m": r["distance_m"],
            "bearing_deg": r["bearing_deg"], "building_height_m": r["building_height_m"],
            "sectors": [],
        })
        if r["pci"] is not None and r["network"] is not None:
            s["sectors"].append({
                "carrier": r["carrier"], "network": r["network"], "band": r["band"],
                "pci": r["pci"], "azimuth_deg": r["azimuth_deg"],
                "rsrp_mean": r["rsrp_mean"], "n_points": r["n_points"],
            })
    sites = [s for s in by_site.values() if s["lat"] is not None]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("// Generated by build_catalog.py — base-station pins for the OSM map layer.\n")
        f.write("window.BASE_STATIONS = " + json.dumps(sites, ensure_ascii=True) + ";\n")
    return len(sites)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-network", action="store_true",
                    help="skip Overpass; use cached/manual heights only")
    ap.add_argument("--test", action="store_true", help="quick self-test and exit")
    args = ap.parse_args()

    if args.test:
        return _selftest()

    rows, unreg, cells = build(use_network=not args.no_network)

    _write_csv(HERE / "base_stations.csv", FIELDS, rows)
    by_site = defaultdict(list)
    for r in rows:
        by_site[r["site_id"]].append(r)
    for site_id, srows in by_site.items():
        _write_csv(HERE / f"bs_{_slug(site_id)}.csv", FIELDS, srows)
    _write_csv(HERE / "unregistered_measured_cells.csv",
               ["network", "band", "pci", "freq_mhz", "channel", "carriers",
                "n_points", "rsrp_mean", "rsrp_min", "rsrp_max", "cellidentity"], unreg)
    n_js = write_base_stations_js(rows, REPO / "Data" / "base_stations.js")

    # -------- report / sanity gate --------
    print(f"catalog rows written      : {len(rows)}  -> base_stations.csv")
    print(f"base_stations.js sites    : {n_js}  -> Data/base_stations.js")
    print(f"per-site CSVs             : {len(by_site)}")
    print(f"unregistered measured cells: {len(unreg)}  -> unregistered_measured_cells.csv")
    print("\nForte Hall anchor (Verizon, LTE Band 13):")
    for pci in (396, 397, 398):
        c = cells.get(("lte", 13, pci))
        if c:
            print(f"   PCI {pci}: n={c['n_points']:>3}  rsrp_mean={c['rsrp']['mean']}"
                  f"  min/max={c['rsrp']['min']}/{c['rsrp']['max']}  f={c['freq']} MHz")
        else:
            print(f"   PCI {pci}: (not seen in indoor walk)")
    fh = next((r for r in rows if r["site_id"] == "BS7_forte_hall"), None)
    if fh:
        print(f"\nForte Hall geometry: distance={fh['distance_m']} m  "
              f"bearing={fh['bearing_deg']} deg  (expect ~415.9 m / ~237 deg)")
        print(f"Forte Hall height  : {fh['building_height_m']} m  "
              f"({fh['elev_source']}, {fh['elev_confidence']})")


def _selftest():
    """Offline sanity checks: geometry convention + registry integrity."""
    meta = load_floorplan_meta()
    d, b = site_geometry(-77.011420, 38.901550, meta)   # Forte Hall
    assert abs(d - 415.9) < 3.0, f"Forte Hall distance {d} != ~415.9"
    assert abs(b - 237.0) < 2.0, f"Forte Hall bearing {b} != ~237"
    reg = json.loads(REGISTRY.read_text())
    ids = [s["site_id"] for s in reg["sites"]]
    assert len(ids) == len(set(ids)), "duplicate site_id in registry"
    fh = next(s for s in reg["sites"] if s["site_id"] == "BS7_forte_hall")
    verizon = next(c for c in fh["carriers"] if c["carrier"] == "Verizon")
    assert set(verizon["pcis"]) == {396, 397, 398}, "Forte Hall PCIs wrong"
    assert decode_cellid("lte", 227951874) == (227951874 >> 8, 227951874 & 0xFF)
    print("SELFTEST OK: geometry ~415.9 m / ~237 deg; registry integrity; ECI decode.")


if __name__ == "__main__":
    main()
