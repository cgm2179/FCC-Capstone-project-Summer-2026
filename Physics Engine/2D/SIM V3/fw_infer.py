#!/usr/bin/env python3
"""
fw_infer.py — tiled inference for the full-wave surrogate + the g2 validation gate.

The model predicts the phase-reduced field Ũ within a box. To cover a whole scene
we tile it (overlapping, Hann-feathered), blend on the SMOOTH Ũ, then reapply the
GLOBAL free-space ramp once: U = Ũ·e^{-j k d}. Because the ramp is shared, the
stitched field stays phase-coherent.

g2 gate: predict the whole 7th floor at 617 MHz for a held-out Tx and compare to a
fresh FDTD solve of the same scene (envelope dB RMSE, Spearman, complex coherence).

  python fw_infer.py --ckpt fw_unet2d.pt --band LTE_B71_617 --validate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap as B
import dataset_3d as D3
import fullwave2d as FW
import plane_extract as PE
from bands_v3 import get
from fullwave2d import FullWaveScene
from fw_dataset import _featurize
from fw_unet2d import load_model, pick_device

HERE = Path(__file__).resolve().parent


def tiled_predict(model, classes, tx_ij, h, f_mhz, box=128, stride=96, dev=None):
    """Predict U(x) over `classes` by tiling. Returns complex U (full grid)."""
    import torch
    dev = dev or pick_device()
    norm = D3.load_norm(json.loads(B.MANIFEST.read_text()))
    ff = norm.freq_feature(f_mhz)
    k = 2.0 * np.pi / (299_792_458.0 / (f_mhz * 1e6))
    H, W = classes.shape
    win = (np.hanning(box)[:, None] * np.hanning(box)[None, :] + 1e-3).astype(np.float32)

    acc = np.zeros((2, H, W), np.float64)
    wsum = np.zeros((H, W), np.float64)
    xs = list(range(0, max(1, H - box + 1), stride)) + [max(0, H - box)]
    ys = list(range(0, max(1, W - box + 1), stride)) + [max(0, W - box)]
    for x0 in sorted(set(xs)):
        for y0 in sorted(set(ys)):
            b = min(box, H, W)
            cls = classes[x0:x0 + b, y0:y0 + b]
            ii, jj = np.mgrid[x0:x0 + b, y0:y0 + b]
            d_m = np.hypot(ii - tx_ij[0], jj - tx_ij[1]) * h
            xin = _featurize(cls, (tx_ij[0] - x0, tx_ij[1] - y0), d_m, ff)
            with torch.no_grad():
                p = model(torch.from_numpy(xin[None]).to(dev)).cpu().numpy()[0]
            w = win[:b, :b]
            acc[:, x0:x0 + b, y0:y0 + b] += p * w
            wsum[x0:x0 + b, y0:y0 + b] += w
    Utilde = (acc[0] + 1j * acc[1]) / np.maximum(wsum, 1e-9)
    ii, jj = np.mgrid[0:H, 0:W]
    d_global = np.hypot(ii - tx_ij[0], jj - tx_ij[1]) * h
    return Utilde * np.exp(-1j * k * d_global)


def _db(a, ref):
    return 20.0 * np.log10(np.maximum(a, ref * 1e-6) / ref)


def validate(model, band_label="LTE_B71_617", n_per_wavelength=8.0, seed=99):
    """g2 gate: tiled prediction vs a fresh FDTD solve on the same held-out scene."""
    from scipy.stats import spearmanr
    band = get(band_label)
    manifest = json.loads(B.MANIFEST.read_text())
    cs = float(manifest["cell_size_m"])
    valid_tx = np.load(B.VALID_TX_MASK)
    txc = np.argwhere(valid_tx.any(axis=1))
    tc = txc[np.random.default_rng(seed).integers(len(txc))]      # held-out Tx
    tx_m = (float(tc[0]) * cs, float(tc[1]) * cs)

    p, Utrue = _fdtd_field(band, tx_m, n_per_wavelength)
    Upred = tiled_predict(model, p.classes, p.tx_idx, p.h_m, band.f_mhz)

    air = (p.classes == 0)
    at = np.abs(Utrue); ap = np.abs(Upred)
    reft = np.percentile(at[air][at[air] > 0], 99.0)
    refp = np.percentile(ap[air][ap[air] > 0], 99.0)
    et, ep = _db(at, reft), _db(ap, refp)                          # relative dB
    m = air & (et > -60)
    rmse = float(np.sqrt(np.mean((et[m] - ep[m]) ** 2)))
    rho, _ = spearmanr(et[m], ep[m])
    # complex coherence |<Up,Ut>| / (||Up|| ||Ut||)
    a, b = Upred[m], Utrue[m]
    coh = float(np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"g2 @ {band.label}  tx=({tc[0]},{tc[1]})  grid {p.classes.shape}")
    print(f"   envelope dB RMSE = {rmse:.2f} dB   Spearman = {rho:+.3f}   "
          f"complex coherence = {coh:.3f}")
    _plot(HERE / "out" / "fw_validate", et, ep, m, p, band, rmse, rho, coh)
    return dict(rmse_db=rmse, spearman=float(rho), coherence=coh)


def validate_outdoor(model, band_label="NR_n41_2506", region_m=40.0, npw=8.0,
                     seed=7, max_cells=2_000_000):
    """Outdoor g2: tiled prediction vs a fresh FDTD solve on a held-out NoMa
    REGION (citywide full-wave is infeasible). Streets = air, buildings = barrier."""
    from scipy.stats import spearmanr
    import fw_dataset
    band = get(band_label)
    city = np.load(B.CITY_DIR / "material_grid.npy")
    cs = 1.0
    air = np.argwhere(city[:, 2, :] == 0)
    tc = air[np.random.default_rng(seed).integers(len(air))]      # held-out tower
    p, Utrue = fw_dataset._outdoor_field(city, cs, band, (float(tc[0]) * cs,
                                         float(tc[1]) * cs), 2.0, npw, region_m,
                                         2.5, max_cells)
    Upred = tiled_predict(model, p.classes, p.tx_idx, p.h_m, band.f_mhz)

    served = (p.classes == 0)                                     # streets/open
    at, ap = np.abs(Utrue), np.abs(Upred)
    reft = np.percentile(at[served][at[served] > 0], 99.0)
    refp = np.percentile(ap[served][ap[served] > 0], 99.0)
    et, ep = _db(at, reft), _db(ap, refp)
    m = served & (et > -60)
    rmse = float(np.sqrt(np.mean((et[m] - ep[m]) ** 2)))
    rho, _ = spearmanr(et[m], ep[m])
    a, b = Upred[m], Utrue[m]
    coh = float(np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"outdoor g2 @ {band.label}  region {p.classes.shape} "
          f"({p.extent_m[0]:.0f}x{p.extent_m[1]:.0f} m)")
    print(f"   envelope dB RMSE = {rmse:.2f}   Spearman = {rho:+.3f}   coherence = {coh:.3f}")
    _plot(HERE / "out" / f"fw_validate_out_{band.label}", et, ep, m, p, band,
          rmse, rho, coh)
    return dict(rmse_db=rmse, spearman=float(rho), coherence=coh)


def _fdtd_field(band, tx_m, n_per_wavelength):
    p = PE.extract_plane(band, height_frac=0.5, n_per_wavelength=n_per_wavelength,
                         tx_m=tx_m, max_cells=1_200_000)
    sim = FullWaveScene(p.classes, p.h_m, band.f_mhz, p.tx_idx, source="cw")
    steps = int(round(2.5 * max(p.extent_m) / 299_792_458.0 / sim.dt))
    res = FW.simulate(sim, steps, warmup_frac=0.6, extract_phasor=True,
                      show_tqdm=True, desc=f"fdtd {band.label}")
    return p, res["phasor"]


def _plot(out, et, ep, m, p, band, rmse, rho, coh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    ex = [0, p.extent_m[0], 0, p.extent_m[1]]
    et_s = np.where(m, et, np.nan); ep_s = np.where(m, ep, np.nan)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for a, M, t in [(ax[0], et_s, "FDTD (ground truth)"),
                    (ax[1], ep_s, "U-Net surrogate (tiled)"),
                    (ax[2], np.where(m, ep - et, np.nan), "error (dB)")]:
        cm = "viridis" if a is not ax[2] else "coolwarm"
        vm = (-40, 0) if a is not ax[2] else (-15, 15)
        im = a.imshow(M.T, origin="lower", cmap=cm, vmin=vm[0], vmax=vm[1],
                      extent=ex, aspect="equal")
        a.plot(p.tx_idx[0] * p.h_m, p.tx_idx[1] * p.h_m, "r*", ms=8)
        a.set_title(t); fig.colorbar(im, ax=a, shrink=0.7)
    ax[1].set_xlabel(f"RMSE {rmse:.1f} dB · ρ {rho:+.2f} · coherence {coh:.2f}")
    fig.suptitle(f"Full-wave surrogate vs FDTD — {band.label}")
    fig.savefig(out / f"g2_{band.label}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"   wrote {out / f'g2_{band.label}.png'}")


def render_prediction(model, band_label="LTE_B71_617", n_per_wavelength=8.0,
                      seed=99, out=None, frames=32):
    """Whole-floor surrogate prediction → timed still (envelope) + wave GIF.

    Because the net predicts the COMPLEX field U, the map is animatable as
    Re{U·e^{jωt}} — a real wavefront sweep, not just a static heatmap.
    """
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    band = get(band_label)
    cs = float(json.loads(B.MANIFEST.read_text())["cell_size_m"])
    txc = np.argwhere(np.load(B.VALID_TX_MASK).any(axis=1))
    tc = txc[np.random.default_rng(seed).integers(len(txc))]
    p = PE.extract_plane(band, height_frac=0.5, n_per_wavelength=n_per_wavelength,
                         tx_m=(float(tc[0]) * cs, float(tc[1]) * cs), max_cells=1_200_000)
    t0 = time.time()
    U = tiled_predict(model, p.classes, p.tx_idx, p.h_m, band.f_mhz)
    dt = time.time() - t0

    out = Path(out) if out else HERE / "out" / "fw_predict"
    out.mkdir(parents=True, exist_ok=True)
    air = (p.classes == 0); solids = (p.classes > 0)
    ex = [0, p.extent_m[0], 0, p.extent_m[1]]
    amp = np.abs(U)
    ref = np.percentile(amp[air][amp[air] > 0], 99.5) if np.any(amp[air] > 0) else 1.0

    fig, ax = plt.subplots(figsize=(9, 4.6))
    im = ax.imshow(np.where(air, _db(amp, ref), np.nan).T, origin="lower",
                   cmap="viridis", vmin=-40, vmax=0, extent=ex, aspect="equal")
    ax.contour(solids.T, levels=[0.5], colors="w", linewidths=0.3, extent=ex)
    ax.plot(p.tx_idx[0] * p.h_m, p.tx_idx[1] * p.h_m, "r*", ms=9)
    ax.set_title(f"U-Net surrogate envelope — {band.label} whole floor "
                 f"(solve {dt*1000:.0f} ms, {p.classes.size/1e6:.2f} M cells)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="dB")
    fig.tight_layout(); fig.savefig(out / "surrogate_envelope.png", dpi=120); plt.close(fig)

    m = float(np.percentile(np.abs(U.real), 99.0)) or 1.0
    eps = max(m / 300.0, 1e-12); vlim = float(np.arcsinh(m / eps))
    comp = lambda a: np.arcsinh(a.astype(np.float32) / eps)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    img = ax.imshow(comp(U.real).T, cmap="RdBu", vmin=-vlim, vmax=vlim,
                    origin="lower", extent=ex, aspect="equal")
    ax.contour(solids.T, levels=[0.5], colors="k", linewidths=0.35, extent=ex)
    ax.plot(p.tx_idx[0] * p.h_m, p.tx_idx[1] * p.h_m, "k*", ms=8)
    ax.set_title(f"U-Net surrogate field (animated) — {band.label}")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")

    def _upd(i):
        img.set_data(comp((U * np.exp(1j * 2 * np.pi * i / frames)).real).T)
        return (img,)

    FuncAnimation(fig, _upd, frames=frames, blit=False).save(
        out / "surrogate_wave.gif", writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"solve {dt*1000:.0f} ms  grid {p.classes.shape}  outputs -> {out}")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="fw_unet2d.pt")
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--validate", action="store_true", help="indoor g2 gate")
    ap.add_argument("--outdoor", action="store_true", help="outdoor g2 (NoMa region)")
    ap.add_argument("--predict", action="store_true", help="whole-floor still + wave GIF")
    ap.add_argument("--region-m", type=float, default=40.0)
    ap.add_argument("--max-cells", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    model = load_model(HERE / args.ckpt)
    if args.validate:
        validate(model, args.band, seed=args.seed)
    if args.outdoor:
        validate_outdoor(model, args.band, region_m=args.region_m,
                         seed=args.seed, max_cells=args.max_cells)
    if args.predict:
        render_prediction(model, args.band, seed=args.seed)
    if not (args.validate or args.outdoor or args.predict):
        print("use --validate / --outdoor (g2 gates) or --predict (still + GIF).")


if __name__ == "__main__":
    main()
