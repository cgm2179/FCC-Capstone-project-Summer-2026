#!/usr/bin/env python3
"""
Extrude the STEP_1 material grid into a 3D building model in several formats
for SketchUp / Google Earth.

Outputs (in --out, default STEP_1/model_3d):
  floor7_model.dxf   3DFACE quads, one layer per material  (SketchUp Pro/desktop)
  floor7_model.3ds   colored mesh, one object per material  (SketchUp Pro/desktop)
  floor7_model.dae   COLLADA, per-material color + glass     (SketchUp Pro/desktop)
  floor7_model.stl   binary STL, one solid                   (SketchUp Free web)
  floor7_model.kmz   DAE geo-placed at real lon/lat          (Google Earth)
  floor7_iso.png       isometric massing QA render
  floor7_split_map.png labeled plan of the 5ft/9ft orange split (for corrections)

Heights (feet), per the walked spec:
  exterior glass curtain wall  9 ft   (encloses the floor to the ceiling)
  solid masonry (gray)         9 ft
  core service area (red)      9 ft   (drywall)
  drywall wall (orange, room)  9 ft
  cubicle partition (orange)   5 ft   (open-plan zones)
  lunch-room glass (cyan)      9 ft
  furniture / desks (lt blue)  2.5 ft
  floor slab                   1 ft thick, extruded DOWN (z -1 -> 0)

The orange class merges real drywall walls and cubicle partitions (the drawing
can't separate them). They are split automatically: orange inside large dense-
furniture "open-plan" zones -> 5 ft cubicle; orange elsewhere -> 9 ft wall.
Refine by dropping rects into cubicle_zones.json / wall_zones.json (px coords).
"""
import argparse
import io
import json
import struct
import zipfile
from math import atan2, cos, degrees, radians, sin
from pathlib import Path

import numpy as np
from scipy import ndimage

FT_PER_M = 3.280839895013123

# split heuristic
SPLIT = dict(furn_min_area=25, dens_win=41, dens_thresh=0.14,
             close=15, open=9, zone_dilate=4)

# name -> dict(src, z0_ft, z1_ft, rgb, alpha, aci)
#   src: material id(s) in the grid, or a special tag resolved in build_masks()
PARTS_SPEC = [
    ("floor_slab",     dict(z=(-1.0, 0.0), rgb=(200, 195, 180), a=1.0, aci=8)),
    ("furniture_desk", dict(z=(0.0, 2.5),  rgb=(190, 210, 255), a=1.0, aci=151)),
    ("cubicle_5ft",    dict(z=(0.0, 5.0),  rgb=(245, 166, 35),  a=1.0, aci=30)),
    ("drywall_9ft",    dict(z=(0.0, 9.0),  rgb=(245, 208, 32),  a=1.0, aci=2)),
    ("masonry_9ft",    dict(z=(0.0, 9.0),  rgb=(64, 64, 64),    a=1.0, aci=8)),
    ("core_9ft",       dict(z=(0.0, 9.0),  rgb=(200, 30, 30),   a=1.0, aci=1)),
    ("lunch_glass_9ft",dict(z=(0.0, 9.0),  rgb=(0, 185, 205),   a=0.45, aci=4)),
    ("exterior_glass_9ft", dict(z=(0.0, 9.0), rgb=(40, 90, 190), a=0.50, aci=5)),
]

# --- box geometry ---------------------------------------------------------
_TRIS = ((4, 5, 6), (4, 6, 7), (0, 2, 1), (0, 3, 2), (0, 1, 5), (0, 5, 4),
         (3, 7, 6), (3, 6, 2), (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5))
# quads: top, bottom, -Y, +Y, -X, +X
_QUADS = ((4, 5, 6, 7), (0, 3, 2, 1), (0, 1, 5, 4),
          (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5))


def corners(b):
    x0, y0, z0, x1, y1, z1 = b
    return [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]


# --- greedy rectangle cover ----------------------------------------------
def greedy_rectangles(mask):
    mask = np.ascontiguousarray(mask, bool).copy()
    H, W = mask.shape
    rects = []
    for r in range(H):
        row = mask[r]
        if not row.any():
            continue
        c = 0
        while c < W:
            if not row[c]:
                c += 1
                continue
            c1 = c
            while c1 < W and row[c1]:
                c1 += 1
            r1 = r + 1
            while r1 < H and mask[r1, c:c1].all():
                r1 += 1
            rects.append((r, c, r1, c1))
            mask[r:r1, c:c1] = False
            c = c1
    return rects


def rects_to_boxes(rects, mpp, H, z0, z1):
    out = []
    for (r0, c0, r1, c1) in rects:
        out.append((c0 * mpp, (H - r1) * mpp, z0,
                    c1 * mpp, (H - r0) * mpp, z1))
    return out


# --- mask construction ----------------------------------------------------
def load_zone_rects(path):
    if not Path(path).exists():
        return []
    spec = json.loads(Path(path).read_text())
    return spec.get("rects", spec if isinstance(spec, list) else [])


def rects_mask(rects, H, W):
    m = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in rects:
        m[max(0, int(y0)):int(y1) + 1, max(0, int(x0)):int(x1) + 1] = True
    return m


def build_masks(grid, outdir):
    H, W = grid.shape
    orange = grid == 1

    # footprint slab: close the structural ring, fill interior
    solid = np.isin(grid, [1, 2, 3, 5, 6])
    footprint = ndimage.binary_fill_holes(
        ndimage.binary_closing(solid, np.ones((11, 11))))

    # furniture, speckle-filtered (drop text / dimension clutter)
    furn = grid == 4
    lbl, n = ndimage.label(furn, structure=np.ones((3, 3)))
    sz = ndimage.sum_labels(np.ones_like(lbl), lbl, np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool); keep[1:] = sz >= SPLIT["furn_min_area"]
    furn = keep[lbl]

    # open-plan zones -> cubicle vs wall split of the orange class
    dens = ndimage.uniform_filter(furn.astype(float), size=SPLIT["dens_win"])
    zone = dens > SPLIT["dens_thresh"]
    zone = ndimage.binary_closing(zone, np.ones((SPLIT["close"],) * 2))
    zone = ndimage.binary_opening(zone, np.ones((SPLIT["open"],) * 2))
    zone = ndimage.binary_dilation(zone, np.ones((2 * SPLIT["zone_dilate"] + 1,) * 2))

    # manual corrections win over the heuristic
    force_cub = rects_mask(load_zone_rects(outdir / "cubicle_zones.json"), H, W)
    force_wall = rects_mask(load_zone_rects(outdir / "wall_zones.json"), H, W)
    cub = orange & ((zone & ~force_wall) | force_cub)
    wall = orange & ~cub

    return dict(
        floor_slab=footprint,
        furniture_desk=furn,
        cubicle_5ft=cub,
        drywall_9ft=wall,
        masonry_9ft=grid == 2,
        core_9ft=grid == 3,
        lunch_glass_9ft=grid == 6,
        exterior_glass_9ft=grid == 5,
    ), zone


# --- writers --------------------------------------------------------------
def write_stl(path, parts, unit_note):
    tris = sum(len(b) * 12 for _, b, *_ in parts)
    with open(path, "wb") as f:
        f.write(unit_note.encode("ascii", "replace").ljust(80)[:80])
        f.write(struct.pack("<I", tris))
        for _, boxes, *_ in parts:
            for b in boxes:
                V = corners(b)
                for (ia, ib, ic) in _TRIS:
                    ax, ay, az = V[ia]; bx, by, bz = V[ib]; cx, cy, cz = V[ic]
                    ux, uy, uz = bx - ax, by - ay, bz - az
                    vx, vy, vz = cx - ax, cy - ay, cz - az
                    nx, ny, nz = uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx
                    nn = (nx*nx+ny*ny+nz*nz) ** .5 or 1.0
                    f.write(struct.pack("<12fH", nx/nn, ny/nn, nz/nn,
                                        ax, ay, az, bx, by, bz, cx, cy, cz, 0))
    return tris


def write_dxf(path, parts):
    L = ["0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1009\n0\nENDSEC",
         "0\nSECTION\n2\nTABLES\n0\nTABLE\n2\nLAYER\n70\n%d" % len(parts)]
    for name, _b, rgb, a, aci in parts:
        L.append("0\nLAYER\n2\n%s\n70\n0\n62\n%d\n6\nCONTINUOUS" % (name, aci))
    L.append("0\nENDTAB\n0\nENDSEC")
    L.append("0\nSECTION\n2\nENTITIES")
    for name, boxes, rgb, a, aci in parts:
        drop_bottom = name != "floor_slab"
        for b in boxes:
            V = corners(b)
            for qi, q in enumerate(_QUADS):
                if drop_bottom and qi == 1:
                    continue
                p = [V[i] for i in q]
                s = "0\n3DFACE\n8\n%s" % name
                for k, (x, y, z) in enumerate(p):
                    s += "\n%d\n%.4f\n%d\n%.4f\n%d\n%.4f" % (
                        10+k, x, 20+k, y, 30+k, z)
                L.append(s)
    L.append("0\nENDSEC\n0\nEOF\n")
    Path(path).write_text("\n".join(L))


def _c(cid, data):
    return struct.pack("<HI", cid, 6 + len(data)) + data


def write_3ds(path, parts):
    mats = b""
    for name, _b, rgb, a, aci in parts:
        col = _c(0x0011, bytes(rgb))
        mats += _c(0xAFFF,
                   _c(0xA000, name.encode()[:16] + b"\0") +
                   _c(0xA010, col) + _c(0xA020, col) + _c(0xA030, _c(0x0011, b"\0\0\0")))
    objs = b""
    for name, boxes, rgb, a, aci in parts:
        # split into <=5000 boxes/object (uint16 face-count limit)
        for ci in range(0, len(boxes), 5000):
            chunk_boxes = boxes[ci:ci + 5000]
            verts = []
            faces = []
            for bi, b in enumerate(chunk_boxes):
                base = bi * 8
                verts.extend(corners(b))
                for (ia, ib, ic) in _TRIS:
                    faces.append((base+ia, base+ib, base+ic))
            pa = struct.pack("<H", len(verts))
            for (x, y, z) in verts:
                pa += struct.pack("<3f", x, y, z)
            fa = struct.pack("<H", len(faces))
            for (i, j, k) in faces:
                fa += struct.pack("<4H", i, j, k, 7)
            mg = name.encode()[:16] + b"\0" + struct.pack("<H", len(faces))
            mg += b"".join(struct.pack("<H", fi) for fi in range(len(faces)))
            fa_chunk = _c(0x4120, fa + _c(0x4130, mg))
            tri = _c(0x4100, _c(0x4110, pa) + fa_chunk)
            oname = (name[:10] + str(ci // 5000 or "")).encode()[:16] + b"\0"
            objs += _c(0x4000, oname + tri)
    edit = _c(0x3D3D, _c(0x3D3E, struct.pack("<I", 3)) +
              _c(0x0100, struct.pack("<f", 1.0)) + mats + objs)
    main = _c(0x4D4D, _c(0x0002, struct.pack("<I", 3)) + edit)
    Path(path).write_bytes(main)


def write_dae(path, parts, unit_name, unit_meter):
    fx, mat, geo, nod = [], [], [], []
    for i, (name, boxes, rgb, a, aci) in enumerate(parts):
        cr, cg, cb = [v/255 for v in rgb]
        verts = []
        tris = []
        for bi, b in enumerate(boxes):
            base = bi*8
            verts.extend(corners(b))
            for t in _TRIS:
                tris.append((base+t[0], base+t[1], base+t[2]))
        pos = " ".join("%.4f" % c for v in verts for c in v)
        idx = " ".join("%d" % j for t in tris for j in t)
        g = "g%d" % i
        fx.append(f'<effect id="fx{i}"><profile_COMMON><technique sid="t"><lambert>'
                  f'<diffuse><color>{cr:.3f} {cg:.3f} {cb:.3f} {a:.2f}</color></diffuse>'
                  f'<transparent opaque="A_ONE"><color>1 1 1 1</color></transparent>'
                  f'<transparency><float>{a:.2f}</float></transparency>'
                  f'</lambert></technique></profile_COMMON></effect>')
        mat.append(f'<material id="m{i}" name="{name}"><instance_effect url="#fx{i}"/></material>')
        geo.append(f'<geometry id="{g}" name="{name}"><mesh>'
                   f'<source id="{g}p"><float_array id="{g}pa" count="{len(verts)*3}">{pos}</float_array>'
                   f'<technique_common><accessor source="#{g}pa" count="{len(verts)}" stride="3">'
                   f'<param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>'
                   f'</accessor></technique_common></source>'
                   f'<vertices id="{g}v"><input semantic="POSITION" source="#{g}p"/></vertices>'
                   f'<triangles count="{len(tris)}" material="s{i}">'
                   f'<input semantic="VERTEX" source="#{g}v" offset="0"/><p>{idx}</p>'
                   f'</triangles></mesh></geometry>')
        nod.append(f'<node id="n{i}" name="{name}"><instance_geometry url="#{g}">'
                   f'<bind_material><technique_common>'
                   f'<instance_material symbol="s{i}" target="#m{i}"/>'
                   f'</technique_common></bind_material></instance_geometry></node>')
    xml = (f'<?xml version="1.0" encoding="utf-8"?>\n'
           f'<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">'
           f'<asset><up_axis>Z_UP</up_axis><unit name="{unit_name}" meter="{unit_meter}"/>'
           f'<created>2026-01-01T00:00:00</created><modified>2026-01-01T00:00:00</modified></asset>'
           f'<library_effects>{"".join(fx)}</library_effects>'
           f'<library_materials>{"".join(mat)}</library_materials>'
           f'<library_geometries>{"".join(geo)}</library_geometries>'
           f'<library_visual_scenes><visual_scene id="s" name="s">{"".join(nod)}</visual_scene>'
           f'</library_visual_scenes><scene><instance_visual_scene url="#s"/></scene></COLLADA>')
    Path(path).write_text(xml)


def bearing(lon1, lat1, lon2, lat2):
    dl = radians(lon2 - lon1)
    y = sin(dl) * cos(radians(lat2))
    x = (cos(radians(lat1))*sin(radians(lat2)) -
         sin(radians(lat1))*cos(radians(lat2))*cos(dl))
    return (degrees(atan2(y, x)) + 360) % 360


def write_kmz(path, parts_m, meta, H):
    """DAE (meters) placed at the model origin's real lon/lat for Google Earth."""
    dae = io.StringIO()
    tmp = Path(path).with_suffix(".kmz.dae")
    write_dae(tmp, parts_m, "meter", 1.0)
    dae_bytes = tmp.read_bytes(); tmp.unlink()

    A = meta["affine_px_to_lonlat"]
    def ll(px, py):
        return (A[0][0]*px + A[0][1]*py + A[0][2],
                A[1][0]*px + A[1][1]*py + A[1][2])
    lon0, lat0 = ll(0, H)              # model origin = pixel (0, H)
    lon1, lat1 = ll(0, H - 1)         # +Y axis (one row up)
    head = bearing(lon0, lat0, lon1, lat1)

    kml = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
           f'<kml xmlns="http://www.opengis.net/kml/2.2"><Placemark>'
           f'<name>7th Floor</name><Model><altitudeMode>relativeToGround</altitudeMode>'
           f'<Location><longitude>{lon0:.8f}</longitude><latitude>{lat0:.8f}</latitude>'
           f'<altitude>0</altitude></Location>'
           f'<Orientation><heading>{head:.3f}</heading><tilt>0</tilt><roll>0</roll></Orientation>'
           f'<Scale><x>1</x><y>1</y><z>1</z></Scale>'
           f'<Link><href>models/floor7.dae</href></Link></Model></Placemark></kml>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)
        z.writestr("models/floor7.dae", dae_bytes)
    return lon0, lat0, head


# --- previews -------------------------------------------------------------
def render_iso(path, parts):
    from PIL import Image, ImageDraw
    d = np.array([1.0, -0.55, 0.72]); d /= np.linalg.norm(d)
    right = np.cross(d, [0, 0, 1.0]); right /= np.linalg.norm(right)
    sup = np.cross(right, d); sup /= np.linalg.norm(sup)
    light = np.array([0.35, -0.45, 0.82]); light /= np.linalg.norm(light)
    FACES = [((4, 5, 6, 7), (0, 0, 1)), ((0, 1, 5, 4), (0, -1, 0)),
             ((3, 2, 6, 7), (0, 1, 0)), ((0, 3, 7, 4), (-1, 0, 0)),
             ((1, 2, 6, 5), (1, 0, 0))]
    polys = []
    for name, boxes, rgb, a, aci in parts:
        rgb = np.array(rgb, float)
        for b in boxes:
            P = np.array(corners(b))
            for idx, n in FACES:
                n = np.array(n, float)
                if n @ d <= 0.01:
                    continue
                q = P[list(idx)]
                depth = q.mean(0) @ d
                sh = 0.55 + 0.45*max(0.0, float(n @ light))
                polys.append((depth, np.column_stack([q @ right, q @ sup]),
                              tuple((rgb*sh).astype(int))))
    polys.sort(key=lambda p: p[0])
    pts = np.vstack([p[1] for p in polys]); mn, mx = pts.min(0), pts.max(0)
    IMW = 1600; sc = (IMW-60)/(mx[0]-mn[0]); IMH = int((mx[1]-mn[1])*sc)+60
    img = Image.new("RGB", (IMW, IMH), (255, 255, 255)); dr = ImageDraw.Draw(img)
    for _, pp, col in polys:
        dr.polygon([((x-mn[0])*sc+30, IMH-30-(y-mn[1])*sc) for x, y in pp], fill=col)
    img.save(path)


def render_split_map(path, masks, zone, H, W):
    from PIL import Image, ImageDraw
    a = np.full((H, W, 3), 255, np.uint8)
    a[masks["floor_slab"]] = (238, 236, 228)
    a[zone & masks["floor_slab"]] = (232, 244, 214)
    a[masks["furniture_desk"]] = (205, 222, 255)
    a[masks["exterior_glass_9ft"]] = (40, 90, 190)
    a[masks["masonry_9ft"]] = (64, 64, 64)
    a[masks["core_9ft"]] = (200, 30, 30)
    a[masks["lunch_glass_9ft"]] = (0, 185, 205)
    a[masks["drywall_9ft"]] = (230, 120, 20)     # 9 ft
    a[masks["cubicle_5ft"]] = (255, 205, 70)     # 5 ft
    img = Image.fromarray(a); dr = ImageDraw.Draw(img)
    for x in range(0, W, 100):
        dr.line([(x, 0), (x, H)], fill=(0, 0, 0)); dr.text((x+2, 2), str(x), fill=(0, 0, 0))
    for y in range(0, H, 100):
        dr.line([(0, y), (W, y)], fill=(0, 0, 0)); dr.text((2, y+2), str(y), fill=(0, 0, 0))
    img.save(path)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--grid", default=here / "material_grid.npy")
    ap.add_argument("--meta", default=here / "floorplan_meta.json")
    ap.add_argument("--out", default=here / "model_3d")
    ap.add_argument("--unit", choices=["feet", "meters"], default="feet")
    args = ap.parse_args()

    grid = np.load(args.grid)
    H, W = grid.shape
    meta = json.loads(Path(args.meta).read_text())
    mpp_m = meta["meters_per_px"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    us = FT_PER_M if args.unit == "feet" else 1.0        # unit scale from meters
    u = "ft" if args.unit == "feet" else "m"
    mpp = mpp_m * us
    hs = 1.0 if args.unit == "feet" else 0.3048          # ft->unit for heights

    masks, zone = build_masks(grid, out)

    parts, parts_m = [], []
    for name, spec in PARTS_SPEC:
        rects = greedy_rectangles(masks[name])
        if not rects:
            continue
        z0, z1 = spec["z"]
        boxes = rects_to_boxes(rects, mpp, H, z0*hs, z1*hs)
        boxes_m = rects_to_boxes(rects, mpp_m, H, z0*0.3048, z1*0.3048)
        parts.append((name, boxes, spec["rgb"], spec["a"], spec["aci"]))
        parts_m.append((name, boxes_m, spec["rgb"], spec["a"], spec["aci"]))

    note = f"floor7 model, units={args.unit} (choose {u} on STL import)"
    tris = write_stl(out / "floor7_model.stl", parts, note)
    write_dxf(out / "floor7_model.dxf", parts)
    write_3ds(out / "floor7_model.3ds", parts)
    write_dae(out / "floor7_model.dae", parts, "foot" if u == "ft" else "meter",
              0.3048 if u == "ft" else 1.0)
    lon0, lat0, head = write_kmz(out / "floor7_model.kmz", parts_m, meta, H)
    render_iso(out / "floor7_iso.png", parts)
    render_split_map(out / "floor7_split_map.png", masks, zone, H, W)

    allv = np.array([c for _, b, *_ in parts for bx in b for c in corners(bx)])
    lo, hi = allv.min(0), allv.max(0)
    print(f"grid {W}x{H}  {mpp_m:.5f} m/px -> {mpp:.5f} {u}/px")
    for name, boxes, *_ in parts:
        print(f"  {name:<20} {len(boxes):>5} boxes")
    print(f"triangles {tris:,}")
    print(f"bbox {u}: X[{lo[0]:.1f},{hi[0]:.1f}] Y[{lo[1]:.1f},{hi[1]:.1f}] "
          f"Z[{lo[2]:.1f},{hi[2]:.1f}]  span {hi[0]-lo[0]:.1f}x{hi[1]-lo[1]:.1f}x{hi[2]-lo[2]:.1f}")
    print(f"KMZ placed at lon/lat ({lon0:.6f}, {lat0:.6f}) heading {head:.1f} deg")
    for f in ["dxf", "3ds", "dae", "stl", "kmz"]:
        p = out / f"floor7_model.{f}"
        print(f"  {p.name:<20} {p.stat().st_size/1e6:6.2f} MB")


if __name__ == "__main__":
    main()
