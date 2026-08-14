#!/usr/bin/env python3
"""
hybrid_field_3d.py — near full-wave + far analytic, stitched in 3-D.

The 3-D port of `2D/SIM V3/hybrid_field.py`. Full-wave everywhere is infeasible, so:
  * NEAR (r < R): a fine 3-D FDTD sub-volume around the Tx (`fullwave3d.solve3d`) →
    relative-dB loss from the RMS envelope (true multipath / interference).
  * FAR  (r > R): the analytic 3-D path-loss engine (`far_field_3d`, SceneV3) → dB.
Handoff on the COARSE scene grid: level-anchor the (relative) near map to the
(absolute) far map by the median offset over a spherical shell R_in..R, then a
radial crossfade `w = clip((r-R_in)/(R-R_in))`. near r<R_in, blend R_in..R, far r>R.

  from hybrid_field_3d import hybrid_pathloss
  out = hybrid_pathloss("LTE_B71_617", tx_coarse=(131,8,66), R_m=6.0)
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy import ndimage

import bootstrap  # noqa: F401
import far_field_3d as FF
import fullwave3d as F3
import solid_extract as SE
from bands_v3 import get


def hybrid_pathloss(band_label="LTE_B71_617", tx_coarse=None, R_m=6.0, R_in_frac=0.7,
                    npw=8.0, crossings=1.2, max_cells=2_000_000, dtype="float32"):
    band = get(band_label)
    scene, man = FF.load()
    cs = float(man["cell_size_m"])
    NX, NY, NZ = man["grid_shape"]
    if tx_coarse is None:
        tx_coarse = (NX // 2, NY // 2, NZ // 2)
    tx_coarse = tuple(int(v) for v in tx_coarse)

    # ---- far analytic on the coarse grid ----
    far = FF.far_pathloss(scene, tx_coarse, band.f_mhz).astype(float)   # (NX,NY,NZ) dB

    # ---- near full-wave on a fine box around Tx ----
    txm = np.array(tx_coarse, float) * cs
    crop = (txm[0] - R_m, txm[0] + R_m, max(0.0, txm[1] - R_m), txm[1] + R_m,
            txm[2] - R_m, txm[2] + R_m)
    s = SE.extract_solid(band, n_per_wavelength=npw, crop_m=crop, tx_m=tuple(txm),
                         max_cells=max_cells)
    res = F3.solve3d(s.classes, s.h_m, band.f_mhz, s.tx_idx, crossings=crossings,
                     extract_phasor=False, dtype=dtype)
    amp = np.asarray(res["rms"])
    ref = np.percentile(amp[amp > 0], 99.0) or 1.0
    near_rel = -20.0 * np.log10(np.maximum(amp, ref * 1e-6) / ref)      # 0 = strongest

    # ---- resample near (fine) → coarse box, place into a full-grid near map ----
    o = np.array(s.meta["crop_origin_cells"], int)                     # coarse origin
    coarse_shape = np.maximum(np.round(np.array(near_rel.shape) * (s.h_m / cs)).astype(int), 1)
    near_c = ndimage.zoom(near_rel, tuple(coarse_shape / np.array(near_rel.shape)), order=1)
    near_full = np.full_like(far, np.nan)
    x1 = min(o[0] + near_c.shape[0], NX); y1 = min(o[1] + near_c.shape[1], NY); z1 = min(o[2] + near_c.shape[2], NZ)
    near_full[o[0]:x1, o[1]:y1, o[2]:z1] = near_c[:x1 - o[0], :y1 - o[1], :z1 - o[2]]
    near_mask = np.isfinite(near_full)

    # ---- radius (coarse cells → metres), anchor, crossfade ----
    ii, jj, kk = np.mgrid[0:NX, 0:NY, 0:NZ]
    r = np.sqrt((ii - tx_coarse[0]) ** 2 + (jj - tx_coarse[1]) ** 2 +
                (kk - tx_coarse[2]) ** 2) * cs
    R_in = R_in_frac * R_m
    shell = near_mask & (r >= R_in) & (r <= R_m) & np.isfinite(far)
    offset = float(np.median(far[shell] - near_full[shell])) if shell.any() else 0.0
    near_abs = near_full + offset

    out = far.copy()
    w = np.clip((r - R_in) / max(R_m - R_in, 1e-6), 0.0, 1.0)          # 0 near → 1 far
    blend = near_mask & (r <= R_m)
    out[blend] = (1 - w[blend]) * near_abs[blend] + w[blend] * far[blend]
    inner = near_mask & (r < R_in)
    out[inner] = near_abs[inner]

    return dict(pl=out, far=far, near_abs=near_abs, mask=near_mask, r=r,
                R_m=R_m, R_in=R_in, offset=offset, solid=s, near_seconds=res["dt_run"])


def continuity_db(out) -> float:
    """Mean |hybrid − far| in a thin shell just outside R (should be small)."""
    r, R = out["r"], out["R_m"]
    band = (r >= R) & (r <= R + out["solid"].meta["cell_size_coarse_m"] * 3) & np.isfinite(out["far"])
    return float(np.nanmean(np.abs(out["pl"][band] - out["far"][band]))) if band.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--R-m", type=float, default=6.0)
    ap.add_argument("--npw", type=float, default=8.0)
    a = ap.parse_args()
    out = hybrid_pathloss(a.band, R_m=a.R_m, npw=a.npw)
    print(f"hybrid PL {out['pl'].shape}  anchor offset={out['offset']:.1f} dB  "
          f"near {out['near_seconds']:.1f}s  handoff continuity={continuity_db(out):.2f} dB")


if __name__ == "__main__":
    main()
