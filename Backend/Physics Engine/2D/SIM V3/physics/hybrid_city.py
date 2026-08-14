#!/usr/bin/env python3
"""
hybrid_city.py — near-full-wave / far-analytic HYBRID coverage for the OUTDOOR city
(the km-scale engine). The outdoor sibling of hybrid_field.py (indoor).

Full-wave (FDTD or the fw_bs surrogate) is accurate but its cost scales with area,
so we run it ONLY on a near-box around the base station (where diffraction /
penetration / dense multipath genuinely need full-wave) and fill the rest of the
city with the material-aware SceneV3 far field (eikonal + Motley-Keenan direct-ray
crossing loss on the OSM buildings). The two are level-ANCHORED at the handoff
radius and crossfaded, so the map is seamless — the whole city, fast, with true
wave structure near the antenna:

  near  (r < R_in)   : full-wave loss (FDTD FullWaveScene, or the fw_bs surrogate)
  blend (R_in..R)    : distance crossfade, level-anchored to the far field
  far   (r > R)      : SceneV3 path loss on the city plane + near-field close-in floor

The antenna directivity shapes the map (panel etc.): the far field is isotropic and
gets the azimuth pattern added as loss; the near full-wave already carries the beam
(surrogate directivity channel / FDTD backplane), and the anchor keeps them on one
scale. Output is absolute **RSRP = EIRP − PL** (EIRP-anchored), or the raw PL map.

  # far field only (no model needed), one station:
  python hybrid_city.py --site BS7 --band TMO_B71_617 --near fdtd
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import _bootstrap as B
import antenna_patterns as AP
import engine_3d as E3
import fw_bs_catalog as fbs
import nearfield as NF
from bands_v3 import get

HERE = Path(__file__).resolve().parent
C0 = 299_792_458.0

# priority when downsampling the class plane for the far field: the heaviest-loss
# material in a block wins, so building cores survive the coarsening.
_FAR_PRIORITY = {0: 0, 4: 1, 5: 2, 2: 3, 3: 4}   # air < foliage < glass < concrete < barrier
_FAR_UNPRIORITY = {v: k for k, v in _FAR_PRIORITY.items()}


def city_plane(grid, man, ds=3):
    """Terrain-following 2 m class plane of the city, downsampled by `ds` (heaviest
    material wins). Returns (plane[int8] (NX2,NZ2), cell_m2, tx_scale=ds)."""
    NY = grid.shape[1]
    tv = man.get("_terrain_v")
    if tv is not None:
        yy = np.clip(np.asarray(tv, np.int64) + 2, 0, NY - 1)
        plane = np.take_along_axis(np.asarray(grid), yy[:, None, :], axis=1)[:, 0, :]
    else:
        plane = np.asarray(grid[:, 2, :])
    plane = plane.astype(np.int8)
    if ds <= 1:
        return plane, float(man["cell_size_m"]), 1
    NX, NZ = plane.shape
    px, pz = (NX // ds) * ds, (NZ // ds) * ds
    blk = np.vectorize(_FAR_PRIORITY.get)(plane[:px, :pz]).reshape(px // ds, ds, pz // ds, ds)
    red = blk.max(axis=(1, 3))
    out = np.vectorize(_FAR_UNPRIORITY.get)(red).astype(np.int8)
    return out, float(man["cell_size_m"]) * ds, ds


def far_scene(plane, cell_m2, manifest, barrier_classes=(2, 3, 5)):
    """A 2-D SceneV3 on the city plane for the far-field eikonal. Buildings
    (concrete=2, backplane=3, glass=5) are made OPAQUE barriers so the fast-march
    routes the front AROUND them at air speed — the far-field diffraction path —
    instead of crawling THROUGH them (the c/n slow-medium inflation that made a
    straight ray read as tens of km). Foliage (4) stays penetrable (light delay).
    SceneV3 needs ≥3 cells per axis for its wall-normal gradient, so the plane is
    stacked into 3 identical Y-layers (the field never moves in Y → still 2-D)."""
    M = np.repeat(plane[:, None, :], 3, axis=1).copy()
    man2 = dict(manifest); man2["cell_size_m"] = cell_m2
    return E3.SceneV3(M, man2, barrier_classes=barrier_classes), 1


def far_pl(scene, y_mid, tx_ds, f_mhz, cell_m2, plane, n_far=3.2, corner_excess_db=0.0):
    """Outdoor far-field path loss (dB) from the SceneV3 **eikonal**.

    SceneV3.pathloss_map uses a DIRECT-ray crossing loss — an indoor penetration
    model that explodes outdoors (a straight Tx→cell ray punches through many
    concrete blocks → 200 dB). Real outdoor signals DIFFRACT around/over buildings,
    which is exactly what the eikonal fast-march does: the front routes around
    barriers, so its travel time T gives the effective (go-around) path length
    d_eff = c·T. Path loss is then FSPL over that diffracted path — NLOS cells get
    a longer path (more loss), bounded, no penetration runaway. `corner_excess_db`
    optionally adds loss ∝ how much longer the diffracted path is than line-of-sight
    (a cheap knife-edge proxy). tx_ds = (x, z) station cell in the downsampled frame."""
    tx = np.array([tx_ds[0], y_mid, tx_ds[1]], float)
    T = scene.arrival_time(tx, f_mhz)[:, y_mid, :].astype(np.float64)    # eikonal, routes around barriers
    NX, NZ = T.shape
    ii, jj = np.mgrid[0:NX, 0:NZ]
    d_geo = np.hypot(ii - tx_ds[0], jj - tx_ds[1]) * cell_m2            # line-of-sight distance
    d_eff = np.clip(T * C0, cell_m2, None)                              # diffracted path length (∞ if unreachable)
    # log-distance path loss with an URBAN exponent over the diffracted path. n=2 is
    # free-space; cities are n≈3–3.8 (COST-231/3GPP UMa), which the plain FSPL of the
    # go-around path underestimates by 20–40 dB — n_far restores realistic decay.
    PL = 20.0 * np.log10(f_mhz) - 27.55 + 10.0 * n_far * np.log10(np.maximum(d_eff, 1.0))
    if corner_excess_db:                                               # extra NLOS loss ∝ detour ratio
        detour = np.clip(d_eff / np.maximum(d_geo, cell_m2) - 1.0, 0.0, None)
        PL = PL + corner_excess_db * np.log1p(detour)
    PL = NF.apply_close_in(PL, d_geo, f_mhz)
    return PL, d_geo


def _near_loss_fine(grid, man, st, npw, region_m, near="fdtd", model=None, dev="cpu"):
    """Full-wave loss on the fine near-box: FDTD (FullWaveScene, no model) or the
    fw_bs surrogate (needs `model`). Returns (near_loss_fine dB, Plane p)."""
    band = get(st["band"])
    if near == "surrogate":
        if model is None:
            raise ValueError("near='surrogate' needs a trained model (pass model=…)")
        p = fbs.bs_region(grid, man, st, npw=npw, region_m=region_m)
        U = fbs._tiled_predict_bs(model, p.classes, p.tx_idx, p.h_m, band.f_mhz,
                                  st["boresight"], st.get("kind", "panel"))
    else:
        p, U = fbs.bs_field(grid, man, st, npw=npw, region_m=region_m)
    amp = np.abs(U)
    ref = np.percentile(amp[amp > 0], 99.5) if np.any(amp > 0) else 1.0
    near_loss = -20.0 * np.log10(np.maximum(amp, ref * 1e-6) / ref)      # dB loss (relative)
    return near_loss, p


def hybrid_pl(grid, man, st, plane, cell_m2, ds, far_pl_map, d_m, *,
              near="fdtd", model=None, dev="cpu", near_radius_m=None, npw=6.0):
    """Full hybrid PL map (dB) on the city plane: far SceneV3 + pattern, with the
    near full-wave box anchored + crossfaded in. st carries vx,vz (native cells),
    boresight, band, kind."""
    band = get(st["band"])
    lam = band.wavelength_m
    R = near_radius_m or fbs.band_region_m(band.f_mhz) * 0.5           # near-box radius (m)
    NX, NZ = plane.shape
    tx_ds = (st["vx"] / ds, st["vz"] / ds)

    # far field + antenna azimuth pattern (isotropic far -> directional)
    ii, jj = np.mgrid[0:NX, 0:NZ]
    bearing = np.degrees(np.arctan2(-(ii - tx_ds[0]), (jj - tx_ds[1])))
    patt = AP.pattern_atten_db(st.get("kind", "panel"), bearing, st["boresight"])
    far = far_pl_map + patt

    # near full-wave box -> loss on the fine grid, downsampled onto the coarse plane
    near_loss_fine, p = _near_loss_fine(grid, man, st, npw, 2.0 * R, near, model, dev)
    # the near box spans p.extent_m around the Tx; map it onto coarse cells
    ex_x, ex_z = p.extent_m
    x0 = int(round(tx_ds[0] - (ex_x / 2) / cell_m2)); x1 = x0 + int(round(ex_x / cell_m2))
    z0 = int(round(tx_ds[1] - (ex_z / 2) / cell_m2)); z1 = z0 + int(round(ex_z / cell_m2))
    x0c, x1c = max(0, x0), min(NX, x1); z0c, z1c = max(0, z0), min(NZ, z1)
    near_loss = np.full((NX, NZ), np.nan)
    if x1c > x0c and z1c > z0c:
        ds_near = ndimage.zoom(near_loss_fine, ((x1 - x0) / near_loss_fine.shape[0],
                                                (z1 - z0) / near_loss_fine.shape[1]), order=1)
        near_loss[x0c:x1c, z0c:z1c] = ds_near[x0c - x0:x1c - x0, z0c - z0:z1c - z0]

    # anchor near->far at the handoff annulus, then distance-crossfade
    R_in = 0.7 * R
    air = (plane == 0)
    annulus = air & (d_m > R_in) & (d_m < R) & np.isfinite(near_loss)
    offset = float(np.median(far[annulus] - near_loss[annulus])) if annulus.any() else 0.0
    near_anchored = near_loss + offset
    w = np.clip((d_m - R_in) / max(R - R_in, 1e-6), 0.0, 1.0)
    have = np.isfinite(near_anchored)
    hybrid = np.where(have, (1 - w) * np.where(have, near_anchored, 0.0) + w * far, far)
    return dict(pl=hybrid, far=far, near_anchored=near_anchored, d_m=d_m,
                R=R, R_in=R_in, offset=offset, plane=plane, cell_m=cell_m2, tx_ds=tx_ds)


def rsrp_map(pl, eirp_dbm):
    """Absolute RSRP (dBm) = EIRP − PL."""
    return eirp_dbm - pl


def coverage(grid, man, st, *, ds=3, near="fdtd", model=None, dev="cpu",
             eirp_dbm=None, near_radius_m=None, npw=6.0, n_far=3.2, manifest=None):
    """One-call hybrid RSRP (or PL) coverage over the whole city for a station+band.
    Returns the hybrid_pl dict plus `rsrp` (if eirp_dbm given) and `plane`."""
    manifest = manifest or json.loads(B.MANIFEST.read_text())
    plane, cell_m2, ds_ = city_plane(grid, man, ds)
    scene, ymid = far_scene(plane, cell_m2, manifest)
    band = get(st["band"])
    tx_ds = (st["vx"] / ds_, st["vz"] / ds_)
    fpl, d_m = far_pl(scene, ymid, tx_ds, band.f_mhz, cell_m2, plane, n_far=n_far)
    out = hybrid_pl(grid, man, st, plane, cell_m2, ds_, fpl, d_m,
                    near=near, model=model, dev=dev, near_radius_m=near_radius_m, npw=npw)
    if eirp_dbm is not None:
        out["rsrp"] = rsrp_map(out["pl"], eirp_dbm)
    out["eirp_dbm"] = eirp_dbm
    return out


def main():
    ap = argparse.ArgumentParser(description="hybrid near-full-wave / far-SceneV3 city coverage")
    ap.add_argument("--site", default="BS7")
    ap.add_argument("--band", default="TMO_B71_617")
    ap.add_argument("--near", choices=["fdtd", "surrogate"], default="fdtd")
    ap.add_argument("--ds", type=int, default=3, help="far-field plane downsample (native 1 m)")
    ap.add_argument("--eirp", type=float, default=63.0, help="EIRP dBm (Ptx+Gmax); RSRP=EIRP-PL")
    ap.add_argument("--near-radius-m", type=float, default=None)
    ap.add_argument("--npw", type=float, default=6.0, help="near-box cells per wavelength")
    ap.add_argument("--n-far", type=float, default=3.2, help="urban far-field path-loss exponent")
    ap.add_argument("--city-dir", default=None,
                    help="city grid dir (default SIM V1 3D/city/NoMa_DC_osm)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    city_dir = args.city_dir or str(B.SIM3D / "city" / "NoMa_DC_osm")
    grid, man, sts = fbs.load_stations([args.band], city_dir=city_dir)
    st = next((s for s in fbs._dedupe_by_site(sts)
               if s["site"].split("_")[0] == args.site and s["band"] == args.band), None)
    if st is None:
        raise SystemExit(f"no live sector for {args.site} / {args.band}")
    st = {**st, "kind": st.get("kind", "panel")}
    print(f"{st['site']} {args.band} f={get(args.band).f_mhz:.0f} MHz  near={args.near}  ds={args.ds}")
    out = coverage(grid, man, st, ds=args.ds, near=args.near, eirp_dbm=args.eirp,
                   near_radius_m=args.near_radius_m, npw=args.npw, n_far=args.n_far)
    pl, rsrp, plane = out["pl"], out.get("rsrp"), out["plane"]
    air = plane == 0
    print(f"anchor offset {out['offset']:+.1f} dB · near R={out['R']:.0f} m")
    print(f"PL(air)   median {np.nanmedian(pl[air]):.0f} dB  range [{np.nanmin(pl[air]):.0f}, {np.nanmax(pl[air]):.0f}]")
    if rsrp is not None:
        print(f"RSRP(air) median {np.nanmedian(rsrp[air]):.0f} dBm range [{np.nanmin(rsrp[air]):.0f}, {np.nanmax(rsrp[air]):.0f}]")
    if args.out:
        np.savez(args.out, pl=pl.astype(np.float32),
                 rsrp=(rsrp.astype(np.float32) if rsrp is not None else np.zeros(0)),
                 plane=plane, cell_m=out["cell_m"])
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
