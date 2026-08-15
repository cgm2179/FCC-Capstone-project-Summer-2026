#!/usr/bin/env python3
"""
run_outdoor2d.py — outdoor 2-D cellular RF coverage on the NoMa DC city plane.

Same 2-D-plane methodology as indoor, but **far-field-analytic primary**: a
horizontal slice of the city voxels (buildings = barrier class 3, streets = air)
fed to the 2-D Motley-Keenan ray-march (`STEP_2/motley_keenan.compute_pathloss`),
which carries the old physics — per-crossing building attenuation + a saturating
obstruction term (corridor/street waveguiding proxy) — over the whole area, fast.

Full-wave over a city is infeasible (2768 m at cm cells = billions of cells), so
FDTD is offered only as an OPTIONAL local "microscope" patch (`--fdtd-patch`)
using the same `FullWaveScene` as indoors.

  python run_outdoor2d.py                              # 800 m coverage @ 2506 MHz
  python run_outdoor2d.py --tx 1400 900 --crop-m 1000  # tower at (1400,900) m
  python run_outdoor2d.py --fdtd-patch 1400 900 60     # + a 60 m FDTD patch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # engine root, for _bootstrap
import _bootstrap as B
import motley_keenan as MK

HERE = Path(__file__).resolve().parent.parent


def _load_city():
    grid = np.load(B.require(B.CITY_DIR / "material_grid.npy"))
    manifest = json.loads((B.CITY_DIR / "manifest_3d.json").read_text())
    return grid, manifest


def _snap_air(plane, ix, iz):
    if plane[ix, iz] == 0:
        return ix, iz
    air = np.argwhere(plane == 0)
    j = np.argmin(np.abs(air[:, 0] - ix) + np.abs(air[:, 1] - iz))
    return int(air[j, 0]), int(air[j, 1])


def main():
    ap = argparse.ArgumentParser(description="outdoor 2-D cellular coverage")
    ap.add_argument("--height-m", type=float, default=2.0, help="receiver plane height")
    ap.add_argument("--tx", type=float, nargs=2, default=None, metavar=("X", "Z"),
                    help="tower location in plane metres (default: area centre)")
    ap.add_argument("--crop-m", type=float, default=800.0,
                    help="half-not: full box side in metres around the Tx (0 = whole city)")
    ap.add_argument("--freq-mhz", type=float, default=2506.0, help="carrier (n41)")
    ap.add_argument("--n", type=float, default=2.2, dest="n_exp")
    ap.add_argument("--eirp-dbm", type=float, default=46.0, help="macro cell EIRP")
    ap.add_argument("--building-loss-db", type=float, default=None,
                    help="per-crossing building loss (default from manifest class 3)")
    ap.add_argument("--wall-sat-db", type=float, default=50.0)
    ap.add_argument("--fdtd-patch", type=float, nargs=3, default=None,
                    metavar=("X", "Z", "SIZE"), help="local FDTD microscope (metres)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else HERE / "out" / "outdoor_noma"
    out.mkdir(parents=True, exist_ok=True)

    grid, manifest = _load_city()
    cs = float(manifest["cell_size_m"])                       # 1.0 m
    NX, NY, NZ = grid.shape
    y = max(0, min(NY - 1, int(round(args.height_m / cs))))
    plane = grid[:, y, :]                                     # (X, Z) int8

    # transmitter (tower) in plane cells
    if args.tx is not None:
        ix, iz = int(args.tx[0] / cs), int(args.tx[1] / cs)
    else:
        ix, iz = NX // 2, NZ // 2
    ix, iz = _snap_air(plane, np.clip(ix, 1, NX - 2), np.clip(iz, 1, NZ - 2))

    # crop to an area of interest around the tower (full city is slow)
    if args.crop_m and args.crop_m > 0:
        half = int(args.crop_m / cs / 2)
        x0, x1 = max(0, ix - half), min(NX, ix + half)
        z0, z1 = max(0, iz - half), min(NZ, iz + half)
    else:
        x0, x1, z0, z1 = 0, NX, 0, NZ
    pc = plane[x0:x1, z0:z1]
    txl = (iz - z0, ix - x0)               # compute_pathloss tx = (col, row) = (z, x)
    print(f"NoMa plane y={y} ({y*cs:.0f} m)  full={NX}x{NZ}  crop={pc.shape} "
          f"@ {cs:.0f} m  tx=({ix},{iz}) cell = ({ix*cs:.0f},{iz*cs:.0f}) m")

    # building loss lookup (class 3 = building barrier)
    bl = args.building_loss_db
    if bl is None:
        mats = {m["id"]: m for m in manifest.get("materials", [])}
        bl = float(mats.get(3, {}).get("loss_db", 18.0))
    loss_db = np.zeros(256, np.float32); loss_db[3] = bl
    loss_per_m = np.zeros(256, np.float32)
    print(f"building loss {bl:.0f} dB/crossing (class 3), sat {args.wall_sat_db:.0f} dB, "
          f"f={args.freq_mhz:.0f} MHz, n={args.n_exp}, EIRP {args.eirp_dbm:.0f} dBm")

    # ---- far-field analytic coverage (the primary result) ----------------
    PL = MK.compute_pathloss(pc, loss_db, loss_per_m, txl, cs,
                             freq_mhz=args.freq_mhz, n_exp=args.n_exp,
                             wall_sat_db=args.wall_sat_db)
    prx = args.eirp_dbm - PL
    np.save(out / "outdoor_pl.npy", PL.astype(np.float32))
    np.save(out / "outdoor_prx.npy", prx.astype(np.float32))
    buildings = (pc == 3)
    served = prx[~buildings]
    print(f"coverage: median {np.median(served):.1f} dBm, p10 {np.percentile(served,10):.1f}, "
          f">= -100 dBm over {100*(served>=-100).mean():.1f}% of open area")

    _plot(out, prx, buildings, cs, ix - x0, iz - z0, args)

    meta = dict(scene="NoMa_DC_buildings", height_m=y * cs, cell_m=cs,
                tx_cell=[int(ix), int(iz)], crop_cells=[int(x0), int(x1), int(z0), int(z1)],
                freq_mhz=args.freq_mhz, n_exp=args.n_exp, eirp_dbm=args.eirp_dbm,
                building_loss_db=float(bl), wall_sat_db=args.wall_sat_db,
                grid=[int(pc.shape[0]), int(pc.shape[1])])
    (out / "outdoor_meta.json").write_text(json.dumps(meta, indent=2))

    if args.fdtd_patch is not None:
        _fdtd_patch(out, plane, cs, x0, z0, args)
    print(f"outputs -> {out}")


def _plot(out, prx, buildings, cs, txx, txz, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ex = [0, prx.shape[0] * cs, 0, prx.shape[1] * cs]
    shown = np.where(buildings, np.nan, prx)
    cmap = plt.get_cmap("turbo").copy(); cmap.set_bad((0.75, 0.75, 0.75))
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(shown.T, origin="lower", cmap=cmap, vmin=-120, vmax=-40,
                   extent=ex, aspect="equal")
    ax.plot(txx * cs, txz * cs, "kv", ms=11, mfc="white", mew=1.5)
    ax.set_title(f"NoMa DC outdoor coverage — {args.freq_mhz:.0f} MHz, "
                 f"EIRP {args.eirp_dbm:.0f} dBm (buildings grey)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="received power (dBm)")
    fig.savefig(out / "outdoor_coverage.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _fdtd_patch(out, plane, cs, x0, z0, args):
    """Optional local full-wave microscope around (x,z,size) metres."""
    import perf_v3
    import fullwave2d as FW
    from scipy import ndimage
    from bands_v3 import Band
    from fullwave2d import FullWaveScene

    px, pz, size = args.fdtd_patch
    band = Band("cell", args.freq_mhz, "cellular", "outdoor patch")
    ix, iz = int(px / cs), int(pz / cs)
    half = int(size / cs / 2)
    sub = plane[ix - half:ix + half, iz - half:iz + half]
    zoom = cs / band.cell_size_m(8.0)                     # N=8 for the patch
    fine = ndimage.zoom(sub, zoom, order=0).astype(np.int8)
    txp = (fine.shape[0] // 2, fine.shape[1] // 2)
    sim = FullWaveScene(fine, band.cell_size_m(8.0), band.f_mhz, txp, source="cw")
    steps = min(2500, int(2.0 * (size / 299792458.0) / sim.dt))
    print(f"FDTD patch {fine.shape} h={band.cell_size_m(8.0)*100:.2f} cm steps={steps}")
    prog = perf_v3.ProgressWriter(out / "progress.json", steps, fine.size, stage="fdtd-patch")
    res = FW.simulate(sim, steps, warmup_frac=0.6, record_frames=50,
                      progress=prog, show_tqdm=True, desc="patch")
    from plane_extract import Plane
    p = Plane(fine, band.cell_size_m(8.0), (fine.shape[0]*band.cell_size_m(8.0),
              fine.shape[1]*band.cell_size_m(8.0)), txp, args.height_m, 8.0, {})
    rms = res["rms"]; ref = np.percentile(rms[rms > 0], 99.5) if np.any(rms > 0) else 1.0
    near_db = 20*np.log10(np.maximum(rms, ref*1e-6)/ref)
    import run_wave2d
    run_wave2d._render(out, res["frames"], res["times"], near_db, p, band, do_gif=True)


if __name__ == "__main__":
    main()
