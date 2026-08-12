"""validate.py — required-info + checks for imported project files, per mode.

Powers POST /api/validate (live feedback in the Import "New project" builder) and gates
POST /api/projects (create). Returns a report:
    {required:[{role,ok,detail}], optional:[{role,ok,detail}], warnings:[...],
     grid:{cols,rows,cell_m,bbox_lonlat}, ok: bool}

Ports the column-detection heuristics from Frontend/landing/landing.js inspectCsv() so the
browser preview and the server agree on what a CSV provides.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

MODEL_EXTS = {".dae", ".obj", ".glb", ".gltf", ".fbx", ".3ds", ".stl", ".kmz"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# MapInfo raster TAB control-point line: (lon,lat)(px,py) Label "..."
_TAB_GCP = re.compile(
    r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)\s*"
    r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)\s*Label\s+\"([^\"]*)\"")


def parse_tab_gcps(text: str) -> list[tuple]:
    """[(lon, lat, px, py, label), ...] from a MapInfo raster .TAB."""
    return [(float(a), float(b), float(c), float(d), lbl)
            for a, b, c, d, lbl in _TAB_GCP.findall(text)]


def inspect_csv_text(text: str, pref: str = "auto") -> dict:
    """Detect coordinate system + column mapping + usable-row count for one CSV.
    Mirror of landing.js inspectCsv()."""
    lines = [ln for ln in text.lstrip("﻿").strip().splitlines() if ln]
    if len(lines) < 2:
        return {"ok": False, "error": "need a header row + at least one data row"}
    H = [s.strip().strip("\"'") for s in lines[0].split(",")]
    HL = [s.lower() for s in H]
    body = [ln.split(",") for ln in lines[1:]]

    def find(subs): return next((i for i, h in enumerate(HL) if any(s in h for s in subs)), -1)
    def exact(names): return next((i for i, h in enumerate(HL) if h in names), -1)

    iPx, iPy = exact(["px"]), exact(["py"])
    iLon = find(["lon", "lng", "long", "east"])
    iLat = find(["lat", "north"])
    iXp, iYp = exact(["x"]), exact(["y"])

    sys = pref
    if sys == "auto":
        sys = ("pixel" if iPx >= 0 and iPy >= 0 else
               "geographic" if iLon >= 0 and iLat >= 0 else
               "projected" if iXp >= 0 and iYp >= 0 else None)
    iX = iY = -1
    if sys == "pixel": iX, iY = iPx, iPy
    elif sys == "geographic": iX, iY = iLon, iLat
    elif sys == "projected": iX, iY = iXp, iYp

    iV = find(["rsrp", "rsrq", "cinr", "rssi", "dbm", "signal", "value"])
    if iV < 0:
        for k in range(len(H) - 1, -1, -1):
            if all(k < len(r) and r[k].strip() and _is_num(r[k]) for r in body):
                iV = k
                break

    def num_ok(r, i): return 0 <= i < len(r) and _is_num(r[i])
    nvalid = sum(1 for r in body if iX >= 0 and iY >= 0 and num_ok(r, iX) and num_ok(r, iY))
    return {
        "ok": iX >= 0 and iY >= 0, "sys": sys, "nRows": len(body), "nValid": nvalid,
        "valueMissing": iV < 0,
        "cols": {"x": H[iX] if iX >= 0 else None, "y": H[iY] if iY >= 0 else None,
                 "value": H[iV] if iV >= 0 else None},
        "lonlat": (iLon, iLat),
    }


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def check_csv_folder(paths: list[Path]) -> dict:
    """Aggregate a folder of measurement CSVs: file count, a sampled column mapping, and the
    lon/lat bounding box (used to crop the OSM basemap to the input grid)."""
    csvs = [p for p in paths if p.suffix.lower() == ".csv"]
    if not csvs:
        return {"ok": False, "detail": "no .csv files found", "n_files": 0}
    sample = inspect_csv_text(_read_text(csvs[0]))
    bbox = _lonlat_bbox(csvs)
    detail = f"{len(csvs)} CSV file(s)"
    if sample.get("cols", {}).get("value"):
        detail += f" · value col '{sample['cols']['value']}'"
    if sample.get("sys"):
        detail += f" · {sample['sys']} coords"
    return {"ok": True, "n_files": len(csvs), "sample": sample, "bbox_lonlat": bbox,
            "detail": detail}


def check_image(path: Path) -> dict:
    if path.suffix.lower() not in IMAGE_EXTS:
        return {"ok": False, "detail": f"unsupported image type {path.suffix}"}
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        return {"ok": True, "width": w, "height": h, "detail": f"{w}×{h} {path.suffix[1:].upper()}"}
    except Exception as e:  # pillow decode error
        return {"ok": False, "detail": f"cannot read image: {e}"}


def check_tab(path: Path) -> dict:
    try:
        gcps = parse_tab_gcps(_read_text(path))
    except Exception as e:
        return {"ok": False, "detail": f"cannot read TAB: {e}"}
    if len(gcps) < 3:
        return {"ok": False, "detail": f"need ≥3 GCPs, found {len(gcps)}"}
    lons = [g[0] for g in gcps]
    lats = [g[1] for g in gcps]
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    ctr = (sum(lons) / len(lons), sum(lats) / len(lats))
    return {"ok": True, "n_gcps": len(gcps), "gcps": gcps, "bbox_lonlat": bbox,
            "center_lonlat": ctr,
            "detail": f"{len(gcps)} GCPs · center {ctr[1]:.5f}, {ctr[0]:.5f}"}


def check_model(name: str) -> dict:
    ext = Path(name).suffix.lower()
    if ext not in MODEL_EXTS:
        return {"ok": False, "detail": f"unsupported model type {ext or '(none)'}"}
    warn = " — include the companion .mtl + textures" if ext == ".obj" else ""
    return {"ok": True, "ext": ext, "needs_mtl": ext == ".obj", "detail": f"{ext[1:].upper()} model{warn}"}


def _grid_from_bbox(bbox, cell_m):
    """cols/rows for a lon/lat bbox at cell_m metres (equirectangular)."""
    if not bbox or not cell_m or cell_m <= 0:
        return None
    lon0, lat0, lon1, lat1 = bbox
    latm = (lat0 + lat1) / 2
    w_m = abs(lon1 - lon0) * 111320 * math.cos(math.radians(latm))
    h_m = abs(lat1 - lat0) * 110540
    return {"cols": max(1, math.ceil(w_m / cell_m) + 1),
            "rows": max(1, math.ceil(h_m / cell_m) + 1),
            "cell_m": cell_m, "bbox_lonlat": bbox,
            "extent_m": [round(w_m, 1), round(h_m, 1)]}


def validate(mode: dict, files: dict, opts: dict) -> dict:
    """mode = {environment, dim}; files = {role: [Path,...]}; opts = {cell_m, elevation, ...}."""
    env, dim = mode.get("environment"), mode.get("dim")
    required, optional, warnings = [], [], []
    bbox = None
    cell_m = float(opts.get("cell_m") or (1.0 if env == "outdoor" else 0.3))

    def add(bucket, role, res):
        bucket.append({"role": role, "ok": bool(res.get("ok")),
                       "detail": res.get("detail", "")})
        return res

    # Measurement CSV(s) — required indoors & outdoors (except pure 3D-model imports).
    if files.get("csv"):
        r = add(required, "measurements (CSV)", check_csv_folder(files["csv"]))
        if r.get("bbox_lonlat"):
            bbox = r["bbox_lonlat"]
        if r.get("ok") and r.get("sample", {}).get("valueMissing"):
            warnings.append("No recognized value column (RSRP/RSRQ/CINR/RSSI) — coverage needs a metric.")
    elif dim == "2d":
        add(required, "measurements (CSV)", {"ok": False, "detail": "required — upload a CSV folder"})

    if dim == "2d" and env == "indoor":
        if files.get("image"):
            add(required, "floor-plan image", check_image(files["image"][0]))
        else:
            add(required, "floor-plan image", {"ok": False, "detail": "required — PNG/JPG of the plan"})
        if files.get("tab"):
            r = add(required, "TAB georeference", check_tab(files["tab"][0]))
            if r.get("ok"):
                bbox = r.get("bbox_lonlat", bbox)
        else:
            add(required, "TAB georeference", {"ok": False, "detail": "required — MapInfo .TAB with ≥3 GCPs"})
        add(optional, "OSM basemap", {"ok": True, "detail": "auto — cropped to the TAB bbox (live/offline)"})

    if dim == "3d":
        if files.get("model"):
            add(required, "3D model", check_model(files["model"][0].name))
        else:
            add(required, "3D model", {"ok": False, "detail": "required — OBJ/GLB/DAE/FBX/3DS/STL/KMZ"})
        elev = opts.get("elevation") or {}
        mode_e = elev.get("mode", "none")
        if mode_e == "none":
            add(required, "elevation", {"ok": True, "detail": "none — flat / city building heights"})
        else:
            fz, ch = elev.get("floor_z_m"), elev.get("ceiling_height_m")
            ok = _is_num(str(fz)) and _is_num(str(ch)) and float(ch) > 0
            add(required, "elevation", {"ok": ok,
                "detail": f"floor {fz} m · ceiling {ch} m" if ok else "choose floor_z_m + ceiling_height_m (m)"})
        if not files.get("tab"):
            add(optional, "TAB georeference", {"ok": True, "detail": "optional for 3D placement"})

    if env == "outdoor":
        add(optional, "OSM area / bbox", {"ok": bool(bbox),
            "detail": "cropped to the CSV/TAB extent" if bbox else "derive from measurements or set a bbox"})

    grid = _grid_from_bbox(bbox, cell_m) if bbox else None
    ok = all(x["ok"] for x in required)
    return {"ok": ok, "required": required, "optional": optional,
            "warnings": warnings, "grid": grid, "bbox_lonlat": bbox}


def _read_text(p: Path) -> str:
    return Path(p).read_text(encoding="latin-1", errors="replace")


def _lonlat_bbox(csvs: list[Path]):
    """Bounding box of lon/lat across a folder of CSVs (samples up to a few files)."""
    lons, lats = [], []
    for p in csvs[:6]:
        info = inspect_csv_text(_read_text(p))
        iLon, iLat = info.get("lonlat", (-1, -1))
        if iLon < 0 or iLat < 0:
            continue
        for ln in _read_text(p).splitlines()[1:]:
            c = ln.split(",")
            if iLon < len(c) and iLat < len(c) and _is_num(c[iLon]) and _is_num(c[iLat]):
                lo, la = float(c[iLon]), float(c[iLat])
                if -180 <= lo <= 180 and -90 <= la <= 90:
                    lons.append(lo)
                    lats.append(la)
    if not lons:
        return None
    return [min(lons), min(lats), max(lons), max(lats)]
