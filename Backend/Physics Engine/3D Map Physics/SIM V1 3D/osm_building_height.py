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


# --------------------------------------------------------------------------- #
# Generalized resolver with on-disk cache + per-site manual override.          #
# Added for the base-station catalog (Cross_Validation/base_station_catalog);  #
# resolve_tx_height_m above is unchanged so existing importers keep working.   #
# --------------------------------------------------------------------------- #
def _cache_key(lon: float, lat: float) -> str:
    return f"{round(float(lon), 6)},{round(float(lat), 6)}"


def load_cache(cache_path) -> dict:
    p = Path(cache_path)
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def save_cache(cache_path, cache: dict) -> None:
    Path(cache_path).write_text(json.dumps(cache, indent=2, sort_keys=True))


def resolve_building_height(lon: float, lat: float, *, manual_m: float | None = None,
                            prefer_manual: bool = False, use_network: bool = True,
                            fallback_m: float = FALLBACK_HEIGHT_M,
                            min_plausible_m: float = 6.0,
                            cache: dict | None = None, cache_path=None) -> dict:
    """Building height at (lon, lat) with a manual override and optional on-disk cache.

    Decision order:
      - prefer_manual=True and manual_m given -> manual   (source 'manual:prefer')
      - else a positive OSM height             -> OSM      (source 'osm:*')
      - else manual_m if given                 -> manual   (source 'manual:fallback')
      - else fallback_m                        -> fallback (source 'fallback:*')

    Never raises on a network miss. Pass a shared `cache` dict for batch runs, or a
    `cache_path` to load/save a JSON cache automatically (Overpass-friendly, offline
    after the first successful run).
    """
    key = _cache_key(lon, lat)
    own_cache = cache is None and cache_path is not None
    if own_cache:
        cache = load_cache(cache_path)

    osm_res = None
    if cache is not None and key in cache:
        osm_res = cache[key]
    elif use_network:
        res = resolve_tx_height_m(lon, lat, use_network=True, fallback_m=fallback_m)
        # strip the bulky OSM element body before caching / returning
        slim = {k: v for k, v in res.items() if k != "osm"}
        el = res.get("osm")
        if isinstance(el, dict):
            slim["osm_tags"] = el.get("tags")
            if "_error" in el:
                slim["osm_error"] = el["_error"]
        osm_res = slim
        if cache is not None:
            cache[key] = slim

    osm_height = None
    osm_source = None
    if osm_res and str(osm_res.get("source", "")).startswith("osm"):
        osm_height = osm_res.get("height_m")
        osm_source = osm_res.get("source")

    # Reject obvious OSM placeholders (e.g. the default height=3 on untagged buildings)
    # so a live re-fetch can't clobber a real survey/fallback with a stub value.
    osm_placeholder = osm_height is not None and float(osm_height) < min_plausible_m
    if osm_placeholder:
        osm_height, osm_source = None, None

    manual_ok = manual_m is not None and float(manual_m) > 0
    if manual_ok and prefer_manual:
        height, source, conf = float(manual_m), "manual:prefer", "manual"
    elif osm_height is not None and float(osm_height) > 0:
        height, source, conf = float(osm_height), osm_source, "osm"
    elif manual_ok:
        height, source, conf = float(manual_m), "manual:fallback", "manual"
    else:
        height, source, conf = float(fallback_m), f"fallback:{fallback_m}m", "fallback"

    if own_cache:
        save_cache(cache_path, cache)

    return {
        "lon": float(lon), "lat": float(lat),
        "height_m": round(height, 2), "source": source, "confidence": conf,
        "osm_height_m": osm_height, "manual_m": manual_m,
        "osm_placeholder": osm_placeholder,
        "osm_tags": (osm_res or {}).get("osm_tags"),
        "osm_error": (osm_res or {}).get("osm_error"),
    }


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
