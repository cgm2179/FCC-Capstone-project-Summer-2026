#!/usr/bin/env python3
"""
osm_building_height.py — resolve Forte Hall (known-Tx) antenna height for M5 / O2I.

Query order:
  1. Overpass (OSM) `height` / `building:levels` at the lon/lat
  2. Optional city / tile manifest `ceiling_height_m` (domain top — a weak upper bound)
  3. Fallback ~3 storeys ≈ 11 m (plan: 10–12 m)

usage:
  python osm_building_height.py
  python osm_building_height.py --lon -77.01142 --lat 38.90155
  from osm_building_height import resolve_tx_height_m
  h = resolve_tx_height_m()
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Known outdoor donor (modes_3d.FORTE_HALL_LONLAT)
FORTE_HALL_LON = -77.011420
FORTE_HALL_LAT = 38.901550

FALLBACK_HEIGHT_M = 11.0          # ~3 storeys
METERS_PER_LEVEL = 3.5
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
QUERY_RADIUS_M = 40.0


def _overpass_ql(lon: float, lat: float, radius_m: float = QUERY_RADIUS_M) -> str:
    return f"""
[out:json][timeout:25];
(
  way(around:{radius_m},{lat},{lon})["building"];
  relation(around:{radius_m},{lat},{lon})["building"];
);
out tags center;
""".strip()


def query_osm_building(lon: float, lat: float, *, timeout_s: float = 25.0) -> dict | None:
    """Return the nearest OSM building element tags+center, or None on failure/miss."""
    body = urllib.parse.urlencode({"data": _overpass_ql(lon, lat)}).encode()
    req = urllib.request.Request(
        OVERPASS_URL, data=body, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "User-Agent": "indoor-walk-test-m5/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        return {"_error": str(e)}

    els = payload.get("elements") or []
    if not els:
        return None

    def dist2(el):
        c = el.get("center") or {}
        if "lat" not in c or "lon" not in c:
            return 1e18
        return (c["lat"] - lat) ** 2 + (c["lon"] - lon) ** 2

    best = min(els, key=dist2)
    return best


def height_from_tags(tags: dict) -> tuple[float | None, str]:
    """Parse OSM tags → height metres. Prefer explicit height, else levels × 3.5 m."""
    if not tags:
        return None, "no tags"
    h = tags.get("height") or tags.get("building:height")
    if h is not None:
        try:
            # OSM heights are metres; sometimes "12 m"
            s = str(h).strip().lower().replace("m", "").strip()
            return float(s), "osm:height"
        except ValueError:
            pass
    levels = tags.get("building:levels") or tags.get("levels")
    if levels is not None:
        try:
            return float(levels) * METERS_PER_LEVEL, "osm:building:levels"
        except ValueError:
            pass
    return None, "osm:no height/levels"


def manifest_ceiling_m(paths: list[Path] | None = None) -> tuple[float | None, str]:
    """Weak anchor: tallest building / domain top from a city tile manifest."""
    candidates = paths or [
        HERE / "city" / "NoMa_DC_buildings" / "manifest_3d.json",
        HERE / "city_demo" / "NoMa_DC_tile" / "manifest_3d.json",
    ]
    for p in candidates:
        if not p.is_file():
            continue
        try:
            man = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        c = man.get("ceiling_height_m")
        if c is not None:
            return float(c), f"manifest:{p.name}:ceiling_height_m"
    return None, "no manifest ceiling"


def resolve_tx_height_m(lon: float = FORTE_HALL_LON, lat: float = FORTE_HALL_LAT,
                        *, use_network: bool = True,
                        fallback_m: float = FALLBACK_HEIGHT_M) -> dict:
    """Resolve antenna / building height at lon/lat.

    Returns dict with height_m, source, and diagnostics. Never raises for network misses.
    """
    out = {
        "lon": float(lon), "lat": float(lat),
        "height_m": float(fallback_m),
        "source": f"fallback:{fallback_m}m",
        "osm": None,
        "manifest_ceiling_m": None,
    }

    man_h, man_src = manifest_ceiling_m()
    if man_h is not None:
        out["manifest_ceiling_m"] = man_h
        out["manifest_source"] = man_src

    if use_network:
        el = query_osm_building(lon, lat)
        out["osm"] = el
        if el and "_error" not in el:
            h, src = height_from_tags(el.get("tags") or {})
            if h is not None and h > 0:
                out["height_m"] = float(h)
                out["source"] = src
                return out

    # Prefer a sane building-scale fallback over a city-domain ceiling (can be 60–70 m).
    # Expose the ceiling as an upper-bound note only.
    out["height_m"] = float(fallback_m)
    out["source"] = f"fallback:{fallback_m}m"
    if man_h is not None:
        out["note"] = (
            f"{man_src}={man_h} m is the voxel domain top (tallest nearby), "
            f"not the Forte Hall rooftop; using {fallback_m} m fallback for antenna height."
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lon", type=float, default=FORTE_HALL_LON)
    ap.add_argument("--lat", type=float, default=FORTE_HALL_LAT)
    ap.add_argument("--no-network", action="store_true",
                    help="Skip Overpass; fallback / manifest only (offline CI).")
    ap.add_argument("--fallback", type=float, default=FALLBACK_HEIGHT_M)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report = resolve_tx_height_m(
        args.lon, args.lat, use_network=not args.no_network, fallback_m=args.fallback)
    if args.json:
        # Drop bulky OSM element body for CLI readability unless useful
        slim = {k: v for k, v in report.items() if k != "osm"}
        if report.get("osm") and "_error" in (report["osm"] or {}):
            slim["osm_error"] = report["osm"]["_error"]
        elif report.get("osm"):
            slim["osm_tags"] = (report["osm"] or {}).get("tags")
        print(json.dumps(slim, indent=2))
    else:
        print(f"Forte Hall @ {report['lat']:.5f},{report['lon']:.5f}")
        print(f"  height_m = {report['height_m']:.1f}   source = {report['source']}")
        if report.get("note"):
            print(f"  note: {report['note']}")


if __name__ == "__main__":
    main()
