#!/usr/bin/env python3
"""Bake the penetrable NoMa city grid to a browser asset for the Outdoor 2D Full-Wave Studio.

Unlike bake_noma_clutter.py (a 1-bit building/air raster for the LEGACY path-loss sim), this
emits the SIM V3 penetrable material grid the surrogate (fw_bs.onnx) was trained on — concrete
core (2), foliage (4), glass facade (5), air (0) — as an 8-bit class raster, cropped to the
base-station window and lightly downsampled (priority glass>concrete>foliage so 1-cell facades
survive), georeferenced by corner lon/lat, with the material palette and the live base stations
(location, boresight, band, PCI) so the studio can place a panel Tx and solve a region.

Reads the penetrable OSM grid built by voxelize_gpkg.py (city/NoMa_DC_osm, terrain-following
y=2 slice). Output  Data/noma_city2d.js:
  window.NOMA_CITY_2D = { nx, nz, cell_m, corners{lon,lat}, grid_b64 (uint8 classes),
                          materials:[{id,name,color,loss}], stations:[{site,pci,band,f_mhz,
                          lon,lat,vx,vz,boresight}] }
"""
import os, sys, json, base64, csv, math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIMV3 = os.path.join(ROOT, "Physics Engine", "2D", "SIM V3")
SIM3D = os.path.join(ROOT, "Physics Engine", "3D Map Physics", "SIM V1 3D")
for p in (SIMV3, SIM3D):
    sys.path.insert(0, p)
import voxelize_city as VC          # vox_to_lonlat / lonlat_to_vox
import city_georef as CG            # FCC anchor + load_georef_city
import fw_bs_catalog as fbs         # BAND_TO_LABEL
from bands_v3 import get

OSM = os.path.join(SIM3D, "city", "NoMa_DC_osm")
OUT = os.path.join(ROOT, "Data", "noma_city2d.js")
TARGET_CELL = 1.5                   # browser raster resolution (m); source is 1 m
PAD_M = 450.0                       # pad around the station bbox (covers a coverage region)
PRIORITY = {0: 0, 4: 1, 2: 2, 5: 3}          # glass wins, then concrete, then foliage, then air
UNPRIORITY = {0: 0, 1: 4, 2: 2, 3: 5}
MATERIALS = [                        # legend ↔ map (matches physics_v2 / the penetrable model)
    {"id": 0, "name": "air / street", "color": "#eef2ef", "loss": "—"},
    {"id": 2, "name": "concrete core", "color": "#555555", "loss": "heavy (O2I)"},
    {"id": 4, "name": "foliage / trees", "color": "#2e9e3f", "loss": "light (dilute)"},
    {"id": 5, "name": "glass facade", "color": "#3a86ff", "loss": "light"},
]
PHYS_F_MHZ = 2000.0                  # representative freq for the browser-FDTD EM constants
PHYS_FALLBACK = {                    # measured from physics_3d.speed_field / damping_by_class @2 GHz
    0: {"n": 1.000, "gPerF": 0.0000, "rigid": False},   # air
    2: {"n": 2.294, "gPerF": 0.4281, "rigid": False},   # concrete_masonry
    3: {"n": 2.294, "gPerF": 0.0000, "rigid": True},    # core_service_area (antenna backplane)
    4: {"n": 1.411, "gPerF": 0.0029, "rigid": False},   # furniture_clutter (repurposed = foliage)
    5: {"n": 2.512, "gPerF": 0.0408, "rigid": False},   # exterior_glass_lowE
}


def em_constants():
    """Per-class {n=c/speed, gPerF=gamma/f_hz≈pi*tanδ, rigid} for the browser FDTD,
    straight from the production physics adapter so the fallback matches the FDTD the
    surrogate trained on. Falls back to measured constants if physics won't import."""
    try:
        import fullwave2d as FW, physics_3d as P3, physics_v2 as PV
        cls = np.arange(6, dtype=np.int64).reshape(1, 6)
        speed, rigid = P3.speed_field(cls, PHYS_F_MHZ, materials=PV.MATERIALS7, barrier_classes=(3,))
        gamma = np.atleast_1d(FW.damping_by_class(PV.MATERIALS7, PHYS_F_MHZ, barrier_classes=(3,)))
        C0 = 299792458.0
        out = {}
        for cid in (0, 2, 3, 4, 5):
            s = float(speed[0, cid])
            out[cid] = {"n": round(C0 / s, 4) if s > 0 else 2.29,
                        "gPerF": round(float(gamma[cid]) / (PHYS_F_MHZ * 1e6), 5),
                        "rigid": bool(rigid[0, cid])}
        return out
    except Exception as e:
        print(f"  (physics import failed: {e}; using measured EM constants)")
        return PHYS_FALLBACK


def main():
    man = CG.georef_manifest(json.loads(open(os.path.join(OSM, "manifest_3d.json")).read()))
    grid = np.load(os.path.join(OSM, "material_grid.npy"), mmap_mode="r")
    NX, NY, NZ = grid.shape
    cs = float(man["cell_size_m"])
    tvp = os.path.join(OSM, "terrain_v.npy")
    tv = np.load(tvp) if os.path.exists(tvp) else None

    # terrain-following 2 m slice (what bs_region cuts) -> (NX, NZ) class map
    if tv is not None:
        yy = np.clip(tv.astype(np.int64) + 2, 0, NY - 1)
        plane = np.take_along_axis(np.asarray(grid), yy[:, None, :], axis=1)[:, 0, :]
    else:
        plane = np.asarray(grid[:, 2, :])
    plane = plane.astype(np.uint8)

    # live base stations -> vox; crop window = their bbox + pad
    stations, vxs, vzs = [], [], []
    for r in csv.DictReader(open(CG.CATALOG)):
        if r["status"] != "live" or not r["lat"] or not r["freq_mhz"]:
            continue
        lab = (fbs.BAND_TO_LABEL.get(r["freq_mhz"].rstrip("0").rstrip("."))
               or fbs.BAND_TO_LABEL.get(str(int(float(r["freq_mhz"])))))
        if lab is None:
            continue
        vx, vz = VC.lonlat_to_vox(man, float(r["lon"]), float(r["lat"]))
        if not (0 <= vx < NX and 0 <= vz < NZ):
            continue
        stations.append(dict(site=r["site_id"], pci=r["pci"], band=lab, f_mhz=get(lab).f_mhz,
                             lon=float(r["lon"]), lat=float(r["lat"]), vx=float(vx), vz=float(vz),
                             boresight=float(r["bearing_deg"]) if r["bearing_deg"] else 0.0))
        vxs.append(vx); vzs.append(vz)
    # one representative (lowest-freq) sector per site for the picker
    best = {}
    for s in stations:
        if s["site"] not in best or s["f_mhz"] < best[s["site"]]["f_mhz"]:
            best[s["site"]] = s
    stations = list(best.values())

    pad = int(PAD_M / cs)
    x0, x1 = max(0, int(min(vxs)) - pad), min(NX, int(max(vxs)) + pad)
    z0, z1 = max(0, int(min(vzs)) - pad), min(NZ, int(max(vzs)) + pad)
    win = plane[x0:x1, z0:z1]
    wx, wz = win.shape

    # priority downsample to ~TARGET_CELL (keeps 1-cell glass facades)
    ds = max(1, int(round(TARGET_CELL / cs)))
    if ds > 1:
        px, pz = (wx // ds) * ds, (wz // ds) * ds
        blk = np.vectorize(PRIORITY.get)(win[:px, :pz]).reshape(px // ds, ds, pz // ds, ds)
        red = blk.max(axis=(1, 3))
        out = np.vectorize(UNPRIORITY.get)(red).astype(np.uint8)
    else:
        out = win.astype(np.uint8)
    ONX, ONZ = out.shape
    cell_m = cs * ds

    def to_lonlat(gx, gz):                              # window+downsample idx -> lon/lat
        return VC.vox_to_lonlat(man, x0 + gx * ds, z0 + gz * ds)
    corners = {"x0z0": to_lonlat(0, 0), "x1z0": to_lonlat(ONX - 1, 0),
               "x0z1": to_lonlat(0, ONZ - 1), "x1z1": to_lonlat(ONX - 1, ONZ - 1)}
    corners = {k: [round(v, 7) for v in xy] for k, xy in corners.items()}

    for s in stations:                                  # station idx in the baked raster
        s["vx"] = round((s["vx"] - x0) / ds, 2)
        s["vz"] = round((s["vz"] - z0) / ds, 2)

    payload = {"nx": ONX, "nz": ONZ, "cell_m": round(cell_m, 4), "corners": corners,
               "materials": MATERIALS, "physics": em_constants(), "phys_f_mhz": PHYS_F_MHZ,
               "stations": stations,
               "grid_b64": base64.b64encode(out.tobytes()).decode()}
    with open(OUT, "w") as f:
        f.write("// Generated by scripts/bake_noma_city2d.py — penetrable NoMa city material grid\n")
        f.write("// (SIM V3, concrete/glass/foliage) for the Outdoor 2D Full-Wave Studio. NOT the\n")
        f.write("// legacy clutter raster. uint8 classes, georeferenced, with the base stations.\n")
        f.write("window.NOMA_CITY_2D = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    u, c = np.unique(out, return_counts=True)
    print(f"raster {ONX}x{ONZ} @ {cell_m:.1f} m · classes {dict(zip(u.tolist(), c.tolist()))}")
    print(f"stations: {[s['site'].split('_')[0] for s in stations]}")
    print(f"wrote {OUT}  ({os.path.getsize(OUT)//1024} KB)")


if __name__ == "__main__":
    main()
