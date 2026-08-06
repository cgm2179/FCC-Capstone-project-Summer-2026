#!/usr/bin/env python3
"""
hybrid_field.py — near-FDTD / far-analytic hybrid coverage (the fast 2-D engine).

Full-wave FDTD is accurate everywhere but its cost scales with domain area, so we
run it ONLY on a small near-box around the transmitter (where the reactive /
Fresnel / dense-multipath field genuinely needs full-wave), and fill the rest of
the floor with the material-aware eikonal path loss (SceneV3: Motley-Keenan
spreading + CrossingLUT Fresnel/Airy transmission + Im(q) absorption — the "old
physics"). The two are level-ANCHORED at the handoff radius and crossfaded, so
the map is seamless. Result: the whole floor, fast, with true wave structure near
the source.

  near  (r < R_in)      : FDTD RMS envelope (FullWaveScene)
  blend (R_in..R)       : distance crossfade
  far   (r > R)         : SceneV3 path loss + near-field close-in floor

Convention: everything is expressed as path LOSS in dB (higher = weaker), so the
FDTD strength is negated and offset to sit on the SceneV3 scale.

  python hybrid_field.py --band WiFi_2G4                 # indoor WLAN, whole floor
  python hybrid_field.py --band WiFi_2G4 --near-radius-m 8 --quick
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import _bootstrap as B
import engine_3d as E3
import fullwave2d as FW
import nearfield as NF
import perf_v3
import plane_extract as PE
from bands_v3 import get
from fullwave2d import FullWaveScene

HERE = Path(__file__).resolve().parent
C0 = 299_792_458.0


def _fit(a, shape):
    """Crop / edge-pad `a` to exactly `shape` (zoom rounding is +/-1 cell)."""
    out = np.full(shape, np.nan)
    nx, nz = min(a.shape[0], shape[0]), min(a.shape[1], shape[1])
    out[:nx, :nz] = a[:nx, :nz]
    if nx < shape[0]:
        out[nx:, :nz] = out[nx - 1, :nz]
    if nz < shape[1]:
        out[:, nz:] = out[:, nz - 1][:, None]
    return out


def _tx_on_plane(grid, valid_tx, y):
    src = valid_tx[:, y, :] if valid_tx[:, y, :].any() else (grid[:, y, :] == 0)
    cx, cz = np.argwhere(src).mean(axis=0)
    # snap to a real valid cell
    v = np.argwhere(src)
    j = np.argmin(np.abs(v[:, 0] - cx) + np.abs(v[:, 1] - cz))
    return int(v[j, 0]), int(v[j, 1])


def main():
    ap = argparse.ArgumentParser(description="near-FDTD / far-analytic hybrid")
    ap.add_argument("--band", default="WiFi_2G4")
    ap.add_argument("--height-m", type=float, default=None)
    ap.add_argument("--height-frac", type=float, default=0.5)
    ap.add_argument("--near-radius-m", type=float, default=None,
                    help="FDTD near-box half-size (default max(8*lambda, 6 m))")
    ap.add_argument("--n-per-wavelength", type=float, default=10.0)
    ap.add_argument("--crossings", type=float, default=2.0)
    ap.add_argument("--record-frames", type=int, default=60)
    ap.add_argument("--warmup-frac", type=float, default=0.6)
    ap.add_argument("--D-m", type=float, default=0.15, help="antenna max dim")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    band = get(args.band)
    lam = band.wavelength_m
    out = Path(args.out) if args.out else HERE / "out" / f"hybrid_{band.label}"
    out.mkdir(parents=True, exist_ok=True)

    # ---- scene + transmitter (coarse native grid) ------------------------
    grid, valid_tx, manifest = PE.load_scene()
    cs = float(manifest["cell_size_m"])
    ny = grid.shape[1]
    ceiling = float(manifest.get("ceiling_height_m", ny * cs))
    height = args.height_m if args.height_m is not None else args.height_frac * ceiling
    y = max(0, min(ny - 1, int(round(height / cs))))
    icx, icz = _tx_on_plane(grid, valid_tx, y)
    tx_x, tx_z = icx * cs, icz * cs
    R = args.near_radius_m or max(8.0 * lam, 6.0)
    z = NF.zones(band.f_mhz, args.D_m)
    print(f"band {band.label} f={band.f_mhz:.0f} MHz  lambda={lam*100:.1f} cm  "
          f"far-field>{z['fraunhofer_m']*100:.0f} cm (antenna); near-box R={R:.1f} m")
    print(f"tx plane y={y} ({y*cs:.2f} m)  tx=({icx},{icz}) cell = "
          f"({tx_x:.1f},{tx_z:.1f}) m")

    # ---- far field: material-aware SceneV3 path loss on the plane --------
    scene, _ = E3.load_scene()
    PL = scene.pathloss_map(np.array([icx, y, icz], float), band.f_mhz)
    PL_far = PL[:, y, :].astype(np.float64)                 # (NX, NZ) dB loss
    NX, NZ = PL_far.shape
    xx, zz = np.meshgrid(np.arange(NX), np.arange(NZ), indexing="ij")
    r_m = np.hypot(xx - icx, zz - icz) * cs
    PL_far = NF.apply_close_in(PL_far, r_m, band.f_mhz)     # fix sub-lambda FSPL
    air = (grid[:, y, :] == 0)
    try:
        air &= np.load(B.SCENE_DIR / "inside_mask.npy")[:, y, :]
    except FileNotFoundError:
        pass

    # ---- near field: FDTD on an R-box around the Tx ----------------------
    cx0 = max(0, int((tx_x - R) / cs)); cx1 = min(NX, int(np.ceil((tx_x + R) / cs)))
    cz0 = max(0, int((tx_z - R) / cs)); cz1 = min(NZ, int(np.ceil((tx_z + R) / cs)))
    crop_m = (cx0 * cs, cx1 * cs, cz0 * cs, cz1 * cs)
    npw = min(args.n_per_wavelength, 6.0) if args.quick else args.n_per_wavelength
    p = PE.extract_plane(band, height_m=height, n_per_wavelength=npw,
                         crop_m=crop_m, tx_m=(tx_x, tx_z))
    sim = FullWaveScene(p.classes, p.h_m, band.f_mhz, p.tx_idx, source="cw")
    t_cross = 2.0 * R / C0
    steps = int(round(args.crossings * t_cross / sim.dt))
    if args.quick:
        steps = min(steps, 2500)
    mcps = perf_v3.calibrate_throughput()
    print(f"near-box {p.classes.shape} h={p.h_m*100:.2f} cm  steps={steps}  "
          f"CFL={sim.cfl():.3f}  est ~{perf_v3.fmt_eta(perf_v3.estimate_seconds(p.classes.size, steps, mcps))}")
    prog = perf_v3.ProgressWriter(out / "progress.json", steps, p.classes.size,
                                  stage="fdtd-near")
    res = FW.simulate(sim, steps, warmup_frac=args.warmup_frac,
                      record_frames=args.record_frames, progress=prog,
                      show_tqdm=True, desc=f"near {band.label}")
    if not res["finite"]:
        raise FloatingPointError("near-field FDTD blew up — CFL too high")
    rms = res["rms"]
    ref = np.percentile(rms[rms > 0], 99.5) if np.any(rms > 0) else 1.0
    near_loss_fine = -20.0 * np.log10(np.maximum(rms, ref * 1e-6) / ref)  # dB loss

    # downsample near loss to coarse cells, place into a full-coarse array
    ds = ndimage.zoom(near_loss_fine, ((cx1 - cx0) / near_loss_fine.shape[0],
                                       (cz1 - cz0) / near_loss_fine.shape[1]),
                      order=1)
    ds = _fit(ds, (cx1 - cx0, cz1 - cz0))
    near_loss = np.full((NX, NZ), np.nan)
    near_loss[cx0:cx1, cz0:cz1] = ds

    # ---- anchor near->far at the handoff, then crossfade ----------------
    R_in = 0.7 * R
    annulus = air & (r_m > R_in) & (r_m < R) & np.isfinite(near_loss)
    offset = float(np.median(PL_far[annulus] - near_loss[annulus])) if annulus.any() else 0.0
    near_anchored = near_loss + offset
    w = np.clip((r_m - R_in) / max(R - R_in, 1e-6), 0.0, 1.0)
    have_near = np.isfinite(near_anchored)
    hybrid = np.where(have_near, (1 - w) * np.where(have_near, near_anchored, 0.0)
                      + w * PL_far, PL_far)
    print(f"anchor offset {offset:+.1f} dB over {int(annulus.sum())} annulus cells")

    # ---- save + render ---------------------------------------------------
    np.save(out / "hybrid_pl.npy", hybrid.astype(np.float32))
    np.save(out / "far_pl.npy", PL_far.astype(np.float32))
    meta = dict(band=band.label, f_mhz=band.f_mhz, tx_cell=[icx, y, icz],
                near_radius_m=R, anchor_offset_db=offset, near_grid=list(p.classes.shape),
                near_h_m=p.h_m, near_steps=steps, near_run_s=res["dt_run"],
                coarse_grid=[NX, NZ], cell_m=cs, zones=z)
    (out / "hybrid_meta.json").write_text(json.dumps(meta, indent=2))

    _plot(out, PL_far, hybrid, near_anchored, air, r_m, R, cs, band, icx, icz)
    # reuse the SIM V3 renderer for the near-zone wave animation
    if not args.no_gif and res["frames"]:
        try:
            import run_wave2d
            near_db = 20.0 * np.log10(np.maximum(rms, ref * 1e-6) / ref)
            run_wave2d._render(out, res["frames"], res["times"], near_db, p, band,
                               do_gif=True)
        except Exception as e:
            print(f"  (near GIF skipped: {e})")
    print(f"outputs -> {out}")


def _plot(out, PL_far, hybrid, near_anchored, air, r_m, R, cs, band, icx, icz):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ex = [0, PL_far.shape[0] * cs, 0, PL_far.shape[1] * cs]
    vmin, vmax = np.nanpercentile(PL_far[air], [2, 98])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for a, M, ttl in [(ax[0], PL_far, "far only (SceneV3)"),
                      (ax[1], np.where(np.isfinite(near_anchored), near_anchored, np.nan),
                       "near only (FDTD, anchored)"),
                      (ax[2], hybrid, "HYBRID (near-FDTD + far-analytic)")]:
        show = np.where(air, M, np.nan)
        im = a.imshow(show.T, origin="lower", cmap="magma_r", vmin=vmin, vmax=vmax,
                      extent=ex, aspect="equal")
        a.add_patch(plt.Circle((icx * cs, icz * cs), R, fill=False, ec="cyan", lw=0.8))
        a.plot(icx * cs, icz * cs, "c*", ms=8)
        a.set_title(f"{ttl}  {band.label}")
        a.set_xlabel("x (m)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="path loss (dB)")
    fig.savefig(out / "hybrid_coverage.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
