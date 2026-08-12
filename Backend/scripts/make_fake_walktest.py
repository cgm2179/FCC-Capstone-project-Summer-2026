#!/usr/bin/env python3
"""make_fake_walktest.py — fabricate a SYNTHETIC walk-test project's raw inputs to stress the
Import/preprocess pipeline end-to-end with fake files (no real data):

  - a folder of "Blind Scan"-named CSVs carrying the exact columns build_all_bands_data.py reads
    (Cell ID, Latitude, Longitude, RSRP/RSRQ/CINR/RSSI, Channel Number/Frequency, MCC/MNC,
    Date, Time), filled with valid random values around a center;
  - a floor-plan PNG (PIL); and
  - a MapInfo .TAB with 3 valid GCPs in the format validate.parse_tab_gcps() expects.

    python scripts/make_fake_walktest.py --out /tmp/fake_walktest
Used by  python Backend/server/app.py --fake-project  to validate + create a project on fakes.
"""
from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

DEF_CENTER = (-77.00740, 38.90359)   # near the real FCC HQ (plausible, but synthetic)
PLAN_W, PLAN_H = 1150, 515

# (filename tag, band, network, freq-field name, freq value)
BANDS = [
    ("_LTE_EB 13  700 (Lower) DL_Blind Scan", 13, "lte", "Channel Number", 5230),
    ("_LTE_EB 66  AWS-3 DL_Blind Scan", 66, "lte", "Channel Number", 66486),
    ("_NR_FR1 TDD n41_Blind Scan SCS Autodetect", 41, "nr_fr1", "Channel Frequency", 2506.0),
    ("_NR_FR1 FDD n71 DL_Blind Scan SCS Autodetect", 71, "nr_fr1", "Channel Frequency", 619.35),
]


def _gcps(center):
    """3 non-collinear GCPs: (px,py) → (lon,lat), a small building-sized affine about the center."""
    lon0, lat0 = center
    return [(974.0, 47.0, lon0 + 0.00042, lat0 + 0.00068),
            (155.0, 468.0, lon0 - 0.00041, lat0 - 0.00052),
            (557.0, 268.0, lon0 + 0.00002, lat0 + 0.00010)]


def write_tab(path: Path, center) -> None:
    lines = ["!table", "!version 300", "!charset WindowsLatin1", "",
             "Definition Table", '  File "fake_floor.png"', '  Type "RASTER"']
    for i, (px, py, lon, lat) in enumerate(_gcps(center), 1):
        lines.append(f'  ({lon},{lat}) ({px},{py}) Label "Pt {i}",')
    lines += ["  CoordSys Earth Projection 1, 104"]
    path.write_text("\n".join(lines) + "\n")


def write_png(path: Path) -> None:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (PLAN_W, PLAN_H), "white")
    d = ImageDraw.Draw(im)
    for x in range(0, PLAN_W, 80):
        d.line([(x, 0), (x, PLAN_H)], fill="#dddddd")
    for y in range(0, PLAN_H, 80):
        d.line([(0, y), (PLAN_W, y)], fill="#dddddd")
    d.rectangle([2, 2, PLAN_W - 3, PLAN_H - 3], outline="black", width=3)
    im.save(path)


def write_csvs(csv_dir: Path, center, n_rows: int) -> None:
    lon0, lat0 = center
    t0 = datetime(2026, 7, 7, 14, 2, 1)
    for tag, band, net, freq_field, freq_val in BANDS:
        is_nr = net.startswith("nr")
        rsrp_h = "SSB - Received Power" if is_nr else "Ref Signal - Received Power"
        rsrq_h = "SSB - Received Quality" if is_nr else "Ref Signal - Received Quality"
        cinr_h = "SSB - CINR" if is_nr else "Ref Signal - CINR"
        rssi_h = "SSB RSSI" if is_nr else "Channel RSSI"
        header = ["Date", "Time", "Latitude", "Longitude", "Cell ID", "MCC", "MNC",
                  freq_field, rsrp_h, rsrq_h, cinr_h, rssi_h]
        with open(csv_dir / f"Fake Device 000000000{tag}.CSV", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(n_rows):
                t = t0 + timedelta(seconds=i * 0.9)
                lat = lat0 + random.uniform(-0.0006, 0.0006)
                lon = lon0 + random.uniform(-0.0006, 0.0006)
                w.writerow([
                    t.strftime("%m/%d/%Y"),
                    t.strftime("%H:%M:%S:") + f"{t.microsecond // 1000:03d}",
                    f"{lat:.6f}", f"{lon:.6f}", random.choice([396, 397, 398, 23, 186, 331]),
                    310, 260, freq_val, round(random.uniform(-120, -80), 2),
                    round(random.uniform(-20, -5), 2), round(random.uniform(0, 30), 2),
                    round(random.uniform(-90, -60), 2)])


def generate(out_dir, center=DEF_CENTER, n_rows: int = 200) -> dict:
    """Write the fixture; return {csv:[Path...], image:Path, tab:Path}."""
    out = Path(out_dir)
    csv_dir = out / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    write_csvs(csv_dir, center, n_rows)
    png, tab = out / "fake_floor.png", out / "fake_floor.tab"
    write_png(png)
    write_tab(tab, center)
    return {"csv": sorted(csv_dir.glob("*.CSV")), "image": png, "tab": tab}


def main() -> int:
    ap = argparse.ArgumentParser(description="fabricate a synthetic walk-test project's raw inputs")
    ap.add_argument("--out", default="/tmp/fake_walktest")
    ap.add_argument("--rows", type=int, default=200, help="rows per CSV")
    a = ap.parse_args()
    r = generate(a.out, n_rows=a.rows)
    print(f"wrote {len(r['csv'])} CSVs + {r['image'].name} + {r['tab'].name} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
