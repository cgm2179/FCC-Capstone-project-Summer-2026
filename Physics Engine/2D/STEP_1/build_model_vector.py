#!/usr/bin/env python3
"""Vector (contour) 3D building model — a higher-fidelity sibling of
build_model.py.

build_model.py extrudes the raster material grid into axis-aligned BOXES
(greedy rectangles): exact at pixel resolution but stair-stepped on every
diagonal and curve. This script instead VECTORIZES each material region
(marching-squares contours + Douglas-Peucker) into smooth polygons and
extrudes those into watertight, per-material solids. Same validated
classification (build_masks + PARTS_SPEC) -> identical materials/heights and
georeference, but true-polygon geometry (the curved lunch room, the round
conference core, angled offices come out smooth) and solid caps.

Outputs (in --out, default STEP_1/model_3d_vector):
  floor7v_model.stl   binary STL, watertight solids                (SketchUp Free web)
  floor7v_model.dxf   3DFACE triangles, one layer per material     (AutoCAD/SketchUp Pro)
  floor7v_model.3ds   colored mesh, one object per material         (SketchUp Pro/desktop)
  floor7v_model.dae   COLLADA, per-material color + glass alpha     (SketchUp; import->save .skp)
  floor7v_model.kmz   DAE (meters) geo-placed at real lon/lat       (Google Earth)
  floor7v_dem.tif/.tfw/.prj  top-of-structure height raster, meters (QGIS/ArcGIS, EPSG:3857)
  floor7v_dem.asc     ESRI ASCII grid of the same height field      (universal raster)
  floor7v_iso.png     isometric massing QA render

DWG / SKP: no faithful pure-Python writer exists (both are proprietary binary).
Use the accurate DXF above -> ODA File Converter (or AutoCAD/BricsCAD) for DWG,
and open the DAE (or DXF) in SketchUp and "Save As" for SKP. See README notes.

Heights, colors and the orange cubicle/wall split are inherited unchanged from
build_model.PARTS_SPEC / build_masks (edit those, or cubicle_zones.json /
wall_zones.json, to refine — this script picks the changes up automatically).
"""
import argparse
import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
from contourpy import contour_generator, FillType

import build_model as bm   # reuse build_masks, PARTS_SPEC, bearing, render_split_map

FT_PER_M = bm.FT_PER_M
SIMPLIFY_EPS_PX = 0.75      # DP tolerance in grid pixels (smooths stair-steps)


# ======================================================================
# 1.  Ear-clipping triangulation with holes  (self-contained, tested)
# ======================================================================
def _tri_area2(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)


def _signed(ring):
    s = 0.0
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]; x1, y1 = ring[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return s / 2.0


def _in_tri(px, py, ax, ay, bx, by, cx, cy, eps=1e-9):
    return (_tri_area2(ax, ay, bx, by, px, py) >= -eps and
            _tri_area2(bx, by, cx, cy, px, py) >= -eps and
            _tri_area2(cx, cy, ax, ay, px, py) >= -eps)


def _coincident(p, q, eps=1e-9):
    return abs(p[0] - q[0]) < eps and abs(p[1] - q[1]) < eps


def _dedup(ring):
    out = []
    for p in ring:
        p = (float(p[0]), float(p[1]))
        if not out or abs(out[-1][0] - p[0]) > 1e-9 or abs(out[-1][1] - p[1]) > 1e-9:
            out.append(p)
    if len(out) > 1 and _coincident(out[0], out[-1]):
        out.pop()
    return out


def _ccw(ring):
    return ring[::-1] if _signed(ring) < 0 else ring


def _seg_int(ax, ay, bx, by, cx, cy, dx, dy):
    def o(px, py, qx, qy, rx, ry):
        v = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)
    o1 = o(ax, ay, bx, by, cx, cy); o2 = o(ax, ay, bx, by, dx, dy)
    o3 = o(cx, cy, dx, dy, ax, ay); o4 = o(cx, cy, dx, dy, bx, by)
    return o1 != o2 and o3 != o4 and o1 and o2 and o3 and o4


def _visible(outer, oi, hx, hy):
    ox, oy = outer[oi]
    n = len(outer)
    for i in range(n):
        if i == oi or (i + 1) % n == oi:
            continue
        a = outer[i]; b = outer[(i + 1) % n]
        if _seg_int(ox, oy, hx, hy, a[0], a[1], b[0], b[1]):
            return False
    return True


def _eliminate_holes(outer, holes):
    outer = list(outer)
    holes = sorted((h for h in holes if len(h) >= 3),
                   key=lambda h: -max(p[0] for p in h))
    for hole in holes:
        hi = max(range(len(hole)), key=lambda i: hole[i][0])
        hx, hy = hole[hi]
        best, bestd = None, None
        for oi, (ox, oy) in enumerate(outer):
            d = (ox - hx) ** 2 + (oy - hy) ** 2
            if (bestd is None or d < bestd) and _visible(outer, oi, hx, hy):
                best, bestd = oi, d
        if best is None:
            best = min(range(len(outer)),
                       key=lambda oi: (outer[oi][0] - hx) ** 2 + (outer[oi][1] - hy) ** 2)
        hole_seq = hole[hi:] + hole[:hi] + [hole[hi]]
        outer = outer[:best + 1] + hole_seq + [outer[best]] + outer[best + 1:]
    return outer


def triangulate(outer, holes=None):
    """outer (any winding) + holes -> (verts[(x,y)], tris[(i,j,k)] CCW)."""
    outer = _ccw(_dedup(outer))
    if len(outer) < 3:
        return [], []
    holes = [_dedup(h) for h in (holes or [])]
    holes = [(h[::-1] if _signed(h) > 0 else h) for h in holes if len(h) >= 3]
    poly = _dedup(_eliminate_holes(outer, holes)) if holes else outer
    n = len(poly)
    if n < 3:
        return [], []
    verts = list(poly)

    def reflex_now(idx):
        m = len(idx)
        out = []
        for k in range(m):
            A = verts[idx[(k - 1) % m]]; B = verts[idx[k]]; C = verts[idx[(k + 1) % m]]
            if _tri_area2(A[0], A[1], B[0], B[1], C[0], C[1]) < 0:
                out.append(idx[k])
        return out

    idx = list(range(n))
    tris = []
    guard, limit = 0, 3 * n * n + 50
    while len(idx) > 3 and guard < limit:
        guard += 1
        m = len(idx)
        reflex = reflex_now(idx)
        clipped = False
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            A = verts[i0]; B = verts[i1]; C = verts[i2]
            if _tri_area2(A[0], A[1], B[0], B[1], C[0], C[1]) <= 1e-9:
                continue
            ear = True
            for j in reflex:
                if j in (i0, i1, i2):
                    continue
                P = verts[j]
                if _coincident(P, A) or _coincident(P, B) or _coincident(P, C):
                    continue
                if _in_tri(P[0], P[1], A[0], A[1], B[0], B[1], C[0], C[1]):
                    ear = False
                    break
            if ear:
                tris.append((i0, i1, i2))
                idx.pop(k)
                clipped = True
                break
        if not clipped:
            idx.pop(0)
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return verts, tris


# ======================================================================
# 2.  Vectorize a binary mask -> smooth polygons (outer + holes)
# ======================================================================
def _dp(pts, eps):
    if len(pts) < 3:
        return pts
    x0, y0 = pts[0]; x1, y1 = pts[-1]
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    dmax, idx = -1.0, 0
    for i in range(1, len(pts) - 1):
        px, py = pts[i]
        if L2 == 0:
            d = ((px - x0) ** 2 + (py - y0) ** 2) ** 0.5
        else:
            t = ((px - x0) * dx + (py - y0) * dy) / L2
            d = ((px - (x0 + t * dx)) ** 2 + (py - (y0 + t * dy)) ** 2) ** 0.5
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        return _dp(pts[:idx + 1], eps)[:-1] + _dp(pts[idx:], eps)
    return [pts[0], pts[-1]]


def _simplify_ring(ring, eps):
    ring = [tuple(p) for p in ring]
    if len(ring) > 1 and _coincident(ring[0], ring[-1]):
        ring = ring[:-1]
    n = len(ring)
    if n < 4:
        return ring
    i1 = max(range(n), key=lambda i: (ring[i][0] - ring[0][0]) ** 2 + (ring[i][1] - ring[0][1]) ** 2)
    out = _dp(ring[:i1 + 1], eps)[:-1] + _dp(ring[i1:] + [ring[0]], eps)[:-1]
    return out if len(out) >= 3 else ring


def vectorize_mask(mask, eps=SIMPLIFY_EPS_PX):
    z = mask.astype(np.float64)
    if z.max() == 0:
        return []
    cg = contour_generator(z=z, fill_type=FillType.ChunkCombinedOffsetOffset)
    points, offsets, outer_offsets = cg.filled(0.5, np.inf)
    polys = []
    for pts, offs, oouter in zip(points, offsets, outer_offsets):
        if pts is None:
            continue
        for pi in range(len(oouter) - 1):
            rings = [pts[offs[r]:offs[r + 1]] for r in range(oouter[pi], oouter[pi + 1])]
            outer = _simplify_ring(rings[0], eps)
            holes = [h for h in (_simplify_ring(hr, eps) for hr in rings[1:]) if len(h) >= 3]
            if len(outer) >= 3:
                polys.append({"outer": outer, "holes": holes})
    return polys


# ======================================================================
# 3.  Extrude polygons -> per-material triangle mesh (caps + sides)
# ======================================================================
def triangulate_polys(polys, H):
    """Triangulate caps ONCE, in flipped-pixel coords (X=col, Y=H-row) so the
    winding already matches the +Y-north world; only a positive scale (mpp) and
    Z remain per unit. Returns list of dict(outer, holes, cverts, ctris)."""
    pre = []
    for poly in polys:
        outer = [(x, H - y) for (x, y) in poly["outer"]]
        holes = [[(x, H - y) for (x, y) in h] for h in poly["holes"]]
        cverts, ctris = triangulate(outer, holes)
        if ctris:
            pre.append(dict(outer=outer, holes=holes, cverts=cverts, ctris=ctris))
    return pre


def extrude_pre(pre, z0, z1, mpp):
    """Scale pre-triangulated (flipped-pixel) polygons by mpp and extrude z0->z1.
    Returns (V float Nx3, T list of (i,j,k)); top+bottom caps + side walls."""
    V, T = [], []

    def av(x, y, z):
        V.append((x, y, z)); return len(V) - 1

    for poly in pre:
        top = [av(x * mpp, y * mpp, z1) for (x, y) in poly["cverts"]]
        bot = [av(x * mpp, y * mpp, z0) for (x, y) in poly["cverts"]]
        for (a, b, c) in poly["ctris"]:
            T.append((top[a], top[b], top[c]))          # top +Z
            T.append((bot[a], bot[c], bot[b]))          # bottom -Z
        for ring in [poly["outer"]] + poly["holes"]:    # side walls
            n = len(ring)
            for k in range(n):
                x0, y0 = ring[k]; x1, y1 = ring[(k + 1) % n]
                b0 = av(x0 * mpp, y0 * mpp, z0); b1 = av(x1 * mpp, y1 * mpp, z0)
                t0 = av(x0 * mpp, y0 * mpp, z1); t1 = av(x1 * mpp, y1 * mpp, z1)
                T.append((b0, b1, t1)); T.append((b0, t1, t0))
    return (np.array(V, float) if V else np.zeros((0, 3))), T


def build_specs(grid, out):
    """Vectorize + triangulate (once) each material part. Returns
    (specs[dict(name, pre, rgb, a, aci, z)], masks, zone)."""
    H, W = grid.shape
    masks, zone = bm.build_masks(grid, out)
    specs = []
    for name, spec in bm.PARTS_SPEC:
        pre = triangulate_polys(vectorize_mask(masks[name]), H)
        if not pre:
            continue
        specs.append(dict(name=name, pre=pre, rgb=spec["rgb"],
                          a=spec["a"], aci=spec["aci"], z=spec["z"]))
    return specs, masks, zone


# ======================================================================
# 4.  Writers  (triangle-mesh variants of build_model's writers)
# ======================================================================
def write_stl(path, meshes, note):
    tris = sum(len(m["T"]) for m in meshes)
    with open(path, "wb") as f:
        f.write(note.encode("ascii", "replace").ljust(80)[:80])
        f.write(struct.pack("<I", tris))
        for m in meshes:
            V = m["V"]
            for (ia, ib, ic) in m["T"]:
                ax, ay, az = V[ia]; bx, by, bz = V[ib]; cx, cy, cz = V[ic]
                ux, uy, uz = bx - ax, by - ay, bz - az
                vx, vy, vz = cx - ax, cy - ay, cz - az
                nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
                nn = (nx * nx + ny * ny + nz * nz) ** .5 or 1.0
                f.write(struct.pack("<12fH", nx / nn, ny / nn, nz / nn,
                                    ax, ay, az, bx, by, bz, cx, cy, cz, 0))
    return tris


def write_dxf(path, meshes):
    L = ["0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1009\n0\nENDSEC",
         "0\nSECTION\n2\nTABLES\n0\nTABLE\n2\nLAYER\n70\n%d" % len(meshes)]
    for m in meshes:
        L.append("0\nLAYER\n2\n%s\n70\n0\n62\n%d\n6\nCONTINUOUS" % (m["name"], m["aci"]))
    L.append("0\nENDTAB\n0\nENDSEC\n0\nSECTION\n2\nENTITIES")
    for m in meshes:
        V = m["V"]; name = m["name"]
        for (ia, ib, ic) in m["T"]:
            p = [V[ia], V[ib], V[ic], V[ic]]   # 3DFACE: repeat 3rd -> triangle
            s = "0\n3DFACE\n8\n%s" % name
            for k, (x, y, z) in enumerate(p):
                s += "\n%d\n%.4f\n%d\n%.4f\n%d\n%.4f" % (10 + k, x, 20 + k, y, 30 + k, z)
            L.append(s)
    L.append("0\nENDSEC\n0\nEOF\n")
    Path(path).write_text("\n".join(L))


def _c(cid, data):
    return struct.pack("<HI", cid, 6 + len(data)) + data


def _chunk_mesh(V, T, maxv=60000):
    """Yield (verts, tris) sub-meshes each with <= maxv unique verts (<=65535)."""
    remap = {}
    lv, lt = [], []
    for (a, b, c) in T:
        if len(lv) + 3 > maxv and remap:
            yield lv, lt
            remap = {}; lv, lt = [], []
        tri = []
        for vi in (a, b, c):
            if vi not in remap:
                remap[vi] = len(lv)
                lv.append(V[vi])
            tri.append(remap[vi])
        lt.append(tuple(tri))
    if lt:
        yield lv, lt


def write_3ds(path, meshes):
    mats = b""
    for m in meshes:
        col = _c(0x0011, bytes(m["rgb"]))
        mats += _c(0xAFFF, _c(0xA000, m["name"].encode()[:16] + b"\0") +
                   _c(0xA010, col) + _c(0xA020, col) +
                   _c(0xA030, _c(0x0011, b"\0\0\0")))
    objs = b""
    for m in meshes:
        for ci, (lv, lt) in enumerate(_chunk_mesh(m["V"], m["T"])):
            pa = struct.pack("<H", len(lv))
            for (x, y, z) in lv:
                pa += struct.pack("<3f", x, y, z)
            fa = struct.pack("<H", len(lt))
            for (i, j, k) in lt:
                fa += struct.pack("<4H", i, j, k, 7)
            mg = m["name"].encode()[:16] + b"\0" + struct.pack("<H", len(lt))
            mg += b"".join(struct.pack("<H", fi) for fi in range(len(lt)))
            tri = _c(0x4100, _c(0x4110, pa) + _c(0x4120, fa + _c(0x4130, mg)))
            oname = (m["name"][:10] + (str(ci) if ci else "")).encode()[:16] + b"\0"
            objs += _c(0x4000, oname + tri)
    edit = _c(0x3D3D, _c(0x3D3E, struct.pack("<I", 3)) +
              _c(0x0100, struct.pack("<f", 1.0)) + mats + objs)
    Path(path).write_bytes(_c(0x4D4D, _c(0x0002, struct.pack("<I", 3)) + edit))


def write_dae(path, meshes, unit_name, unit_meter):
    fx, mat, geo, nod = [], [], [], []
    for i, m in enumerate(meshes):
        cr, cg, cb = [v / 255 for v in m["rgb"]]; a = m["a"]
        V = m["V"]
        pos = " ".join("%.4f" % c for v in V for c in v)
        idx = " ".join("%d" % j for t in m["T"] for j in t)
        g = "g%d" % i
        fx.append(f'<effect id="fx{i}"><profile_COMMON><technique sid="t"><lambert>'
                  f'<diffuse><color>{cr:.3f} {cg:.3f} {cb:.3f} {a:.2f}</color></diffuse>'
                  f'<transparent opaque="A_ONE"><color>1 1 1 1</color></transparent>'
                  f'<transparency><float>{a:.2f}</float></transparency>'
                  f'</lambert></technique></profile_COMMON></effect>')
        mat.append(f'<material id="m{i}" name="{m["name"]}"><instance_effect url="#fx{i}"/></material>')
        geo.append(f'<geometry id="{g}" name="{m["name"]}"><mesh>'
                   f'<source id="{g}p"><float_array id="{g}pa" count="{len(V) * 3}">{pos}</float_array>'
                   f'<technique_common><accessor source="#{g}pa" count="{len(V)}" stride="3">'
                   f'<param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>'
                   f'</accessor></technique_common></source>'
                   f'<vertices id="{g}v"><input semantic="POSITION" source="#{g}p"/></vertices>'
                   f'<triangles count="{len(m["T"])}" material="s{i}">'
                   f'<input semantic="VERTEX" source="#{g}v" offset="0"/><p>{idx}</p>'
                   f'</triangles></mesh></geometry>')
        nod.append(f'<node id="n{i}" name="{m["name"]}"><instance_geometry url="#{g}">'
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


def write_kmz(path, meshes_m, meta, H):
    tmp = Path(path).with_suffix(".kmz.dae")
    write_dae(tmp, meshes_m, "meter", 1.0)
    dae_bytes = tmp.read_bytes(); tmp.unlink()
    A = meta["affine_px_to_lonlat"]

    def ll(px, py):
        return (A[0][0] * px + A[0][1] * py + A[0][2],
                A[1][0] * px + A[1][1] * py + A[1][2])
    lon0, lat0 = ll(0, H)
    lon1, lat1 = ll(0, H - 1)
    head = bm.bearing(lon0, lat0, lon1, lat1)
    kml = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
           f'<kml xmlns="http://www.opengis.net/kml/2.2"><Placemark>'
           f'<name>7th Floor (vector)</name><Model><altitudeMode>relativeToGround</altitudeMode>'
           f'<Location><longitude>{lon0:.8f}</longitude><latitude>{lat0:.8f}</latitude>'
           f'<altitude>0</altitude></Location>'
           f'<Orientation><heading>{head:.3f}</heading><tilt>0</tilt><roll>0</roll></Orientation>'
           f'<Scale><x>1</x><y>1</y><z>1</z></Scale>'
           f'<Link><href>models/floor7.dae</href></Link></Model></Placemark></kml>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)
        z.writestr("models/floor7.dae", dae_bytes)
    return lon0, lat0, head


# --- DEM: top-of-structure height field (meters), georeferenced -----------
def write_dem(base, masks, meta):
    """Raster (grid resolution) whose value is the tallest structure top (m) at
    each cell. GeoTIFF-adjacent: float TIFF + worldfile (EPSG:3857) + .prj, plus
    an ESRI ASCII grid in pixel space for universal readability."""
    from PIL import Image
    H, W = next(iter(masks.values())).shape
    hgt = np.zeros((H, W), np.float32)
    for name, spec in bm.PARTS_SPEC:
        if name == "floor_slab":
            continue
        z1_m = spec["z"][1] * 0.3048
        m = masks[name]
        hgt = np.where(m & (z1_m > hgt), np.float32(z1_m), hgt)

    # float TIFF (PIL mode 'F')
    Image.fromarray(hgt, mode="F").save(str(base) + ".tif")
    # worldfile from px->mercator affine (supports rotation): rows A,D,B,E,C,F
    a = meta["affine_px_to_mercator"]
    tfw = [a[0][0], a[1][0], a[0][1], a[1][1],
           a[0][2] + 0.5 * a[0][0] + 0.5 * a[0][1],
           a[1][2] + 0.5 * a[1][0] + 0.5 * a[1][1]]
    Path(str(base) + ".tfw").write_text("\n".join("%.10f" % v for v in tfw) + "\n")
    Path(str(base) + ".prj").write_text(
        'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433]],PROJECTION["Mercator_1SP"],'
        'PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],'
        'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
        'UNIT["metre",1],AXIS["X",EAST],AXIS["Y",NORTH],AUTHORITY["EPSG","3857"]]\n')
    # ESRI ASCII grid in pixel space (cellsize 1, y flipped to north-up)
    lines = [f"ncols {W}", f"nrows {H}", "xllcorner 0", "yllcorner 0",
             "cellsize 1", "NODATA_value -9999"]
    for r in range(H):
        lines.append(" ".join("%.3f" % v for v in hgt[r]))
    Path(str(base) + ".asc").write_text("\n".join(lines) + "\n")
    return float(hgt.max())


# --- iso QA render --------------------------------------------------------
def render_iso(path, meshes, imw=1700):
    from PIL import Image, ImageDraw
    d = np.array([1.0, -0.55, 0.72]); d /= np.linalg.norm(d)
    right = np.cross(d, [0, 0, 1.0]); right /= np.linalg.norm(right)
    sup = np.cross(right, d); sup /= np.linalg.norm(sup)
    light = np.array([0.35, -0.45, 0.82]); light /= np.linalg.norm(light)
    faces = []
    for m in meshes:
        V = m["V"]; rgb = np.array(m["rgb"], float)
        RS = np.column_stack([right, sup])
        for (a, b, c) in m["T"]:
            P = V[[a, b, c]]
            nrm = np.cross(P[1] - P[0], P[2] - P[0])
            ln = np.linalg.norm(nrm) or 1.0
            sh = 0.5 + 0.5 * max(0.0, abs(float((nrm / ln) @ light)))
            faces.append((P.mean(0) @ d, P @ RS, tuple(int(x * sh) for x in rgb)))
    faces.sort(key=lambda q: q[0])
    allp = np.vstack([f[1] for f in faces]); mn, mx = allp.min(0), allp.max(0)
    sc = (imw - 60) / (mx[0] - mn[0]); imh = int((mx[1] - mn[1]) * sc) + 60
    img = Image.new("RGB", (imw, imh), (255, 255, 255)); dr = ImageDraw.Draw(img)
    for _, scr, col in faces:
        dr.polygon([((x - mn[0]) * sc + 30, imh - 30 - (y - mn[1]) * sc)
                    for x, y in scr], fill=col)
    img.save(path)


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--grid", default=here / "material_grid.npy")
    ap.add_argument("--meta", default=here / "floorplan_meta.json")
    ap.add_argument("--out", default=here / "model_3d_vector")
    ap.add_argument("--unit", choices=["feet", "meters"], default="feet")
    args = ap.parse_args()

    grid = np.load(args.grid)
    H, W = grid.shape
    meta = json.loads(Path(args.meta).read_text())
    mpp_m = meta["meters_per_px"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    us = FT_PER_M if args.unit == "feet" else 1.0
    u = "ft" if args.unit == "feet" else "m"
    mpp = mpp_m * us
    hs = 1.0 if args.unit == "feet" else 0.3048

    specs, masks, zone = build_specs(grid, out)

    # extrude in the requested unit (meshes) and in meters (for KMZ); caps were
    # triangulated once in build_specs, so this is just scaling + side walls.
    meshes, meshes_m = [], []
    for s in specs:
        z0, z1 = s["z"]
        V, T = extrude_pre(s["pre"], z0 * hs, z1 * hs, mpp)
        Vm, Tm = extrude_pre(s["pre"], z0 * 0.3048, z1 * 0.3048, mpp_m)
        base = dict(name=s["name"], rgb=s["rgb"], a=s["a"], aci=s["aci"])
        meshes.append({**base, "V": V, "T": T})
        meshes_m.append({**base, "V": Vm, "T": Tm})

    note = f"floor7 vector model, units={args.unit} (choose {u} on STL import)"
    tris = write_stl(out / "floor7v_model.stl", meshes, note)
    write_dxf(out / "floor7v_model.dxf", meshes)
    write_3ds(out / "floor7v_model.3ds", meshes)
    write_dae(out / "floor7v_model.dae", meshes,
              "foot" if u == "ft" else "meter", 0.3048 if u == "ft" else 1.0)
    lon0, lat0, head = write_kmz(out / "floor7v_model.kmz", meshes_m, meta, H)
    dem_max = write_dem(out / "floor7v_dem", masks, meta)
    render_iso(out / "floor7v_iso.png", meshes)
    bm.render_split_map(out / "floor7v_split_map.png", masks, zone, H, W)

    allv = np.vstack([m["V"] for m in meshes if len(m["V"])])
    lo, hi = allv.min(0), allv.max(0)
    print(f"grid {W}x{H}  {mpp_m:.5f} m/px -> {mpp:.5f} {u}/px  (vector/contour build)")
    for m in meshes:
        print(f"  {m['name']:<20} {len(m['V']):>7} verts {len(m['T']):>7} tris")
    print(f"triangles {tris:,}")
    print(f"bbox {u}: X[{lo[0]:.1f},{hi[0]:.1f}] Y[{lo[1]:.1f},{hi[1]:.1f}] "
          f"Z[{lo[2]:.1f},{hi[2]:.1f}]  span {hi[0]-lo[0]:.1f}x{hi[1]-lo[1]:.1f}x{hi[2]-lo[2]:.1f}")
    print(f"DEM max height {dem_max:.2f} m")
    print(f"KMZ placed at lon/lat ({lon0:.6f}, {lat0:.6f}) heading {head:.1f} deg")
    for f in ["stl", "dxf", "3ds", "dae", "kmz"]:
        p = out / f"floor7v_model.{f}"
        print(f"  {p.name:<22} {p.stat().st_size/1e6:6.2f} MB")
    for f in ["tif", "asc"]:
        p = out / f"floor7v_dem.{f}"
        print(f"  {p.name:<22} {p.stat().st_size/1e6:6.2f} MB")


if __name__ == "__main__":
    main()
