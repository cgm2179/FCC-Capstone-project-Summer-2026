#!/usr/bin/env python3
"""
export_pl_volume.py — export the real SceneV3 volumes for the browser.

Computes, for one transmitter, the full-physics PL(x,y,z) per band and the
eikonal T(x,y,z), and serializes them to compact float16 binaries the 3-D
Simulation tab fetches from SIM3D/web/volumes/. These drive the browser's two
cheap production views: the "coverage field" (PL) and the "wavefront sweep" (T,
scrubbed by an arrival-time threshold). A full multi-Tx sweep is the same script
called with --tx in a loop (heavy → Colab).

Outputs (SIM3D/web/volumes/):
  pl_volume_<txid>.bin   float16, C-order (nband, NX, NY, NZ), dB
  t_volume_<txid>.bin    float16, C-order (NX, NY, NZ), nanoseconds
                         (barrier / unreachable voxels → 65504 sentinel)
  index.json             list of sidecars {txid, tx_vox, grid_shape, bands,
                         t_max_ns, files, …} — the browser's volume catalog

usage:
  python "SIM V1 3D/export_pl_volume.py"                 # auto Tx (interior centroid, ceiling)
  python "SIM V1 3D/export_pl_volume.py" --tx 131 9 59   # explicit voxel
  python "SIM V1 3D/export_pl_volume.py" --t-freq 2442   # eikonal T at a chosen band
"""
import argparse
import json
from pathlib import Path

import numpy as np

from engine_3d import load_scene

HERE = Path(__file__).resolve().parent
T_UNREACHABLE_NS = 65504.0   # float16 max — sentinel for barrier / unreachable voxels


def pick_tx(valid, want=(0.5, 0.85, 0.5)):
    """Nearest valid-Tx voxel to a fractional target (mirrors run_one_calc)."""
    ijk = np.argwhere(valid)
    target = np.array([want[i] * valid.shape[i] for i in range(3)])
    return ijk[np.argmin(((ijk - target) ** 2).sum(1))].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tx", type=float, nargs=3, default=None)
    ap.add_argument("--t-freq", type=float, default=3500.0,
                    help="frequency (MHz) for the eikonal T solve")
    args = ap.parse_args()

    scene, man = load_scene()
    valid = np.load(HERE / "valid_tx_mask.npy")
    tx = np.array(args.tx, float) if args.tx else pick_tx(valid)
    txi = [int(round(v)) for v in tx]
    nx, ny, nz = scene.M.shape
    bands = [float(f) for f in man["freqs_mhz"]]

    print(f"grid {nx}x{ny}x{nz}  tx(voxel)={txi}  bands={bands}  T@{args.t_freq:.0f} MHz")
    PL = scene.pathloss_maps(tx)                              # (nf, nx, ny, nz) float32 dB
    T = scene.arrival_time(tx, args.t_freq)                   # (nx, ny, nz) seconds
    reach = np.isfinite(T)
    T_ns = np.where(reach, T * 1e9, T_UNREACHABLE_NS).astype(np.float16)
    t_max_ns = float(T[reach].max() * 1e9) if reach.any() else 0.0

    txid = f"tx_{txi[0]}-{txi[1]}-{txi[2]}"
    vdir = HERE / "web" / "volumes"
    vdir.mkdir(parents=True, exist_ok=True)
    pl_file, t_file = f"pl_volume_{txid}.bin", f"t_volume_{txid}.bin"
    np.ascontiguousarray(PL.astype(np.float16)).tofile(vdir / pl_file)
    np.ascontiguousarray(T_ns).tofile(vdir / t_file)

    sidecar = dict(
        txid=txid, tx_vox=txi, grid_shape=[nx, ny, nz], bands=bands,
        t_freq_mhz=args.t_freq, t_max_ns=round(t_max_ns, 2),
        t_unreachable_ns=T_UNREACHABLE_NS, dtype="float16",
        pl_layout="band,x,y,z", t_layout="x,y,z",
        pl_file=pl_file, t_file=t_file, norm=man.get("norm"),
    )
    idxp = vdir / "index.json"
    idx = json.loads(idxp.read_text()) if idxp.exists() else {"volumes": []}
    idx["volumes"] = [v for v in idx["volumes"] if v.get("txid") != txid] + [sidecar]
    idxp.write_text(json.dumps(idx, indent=2))

    pin = PL[:, scene.inside] if scene.inside is not None else PL.reshape(len(bands), -1)
    print(f"wrote {vdir / pl_file} ({(vdir / pl_file).stat().st_size / 1e6:.1f} MB), "
          f"{t_file} ({(vdir / t_file).stat().st_size / 1e6:.1f} MB), index.json ({len(idx['volumes'])} vols)")
    print(f"  PL band0 min/median/max = {pin[0].min():.1f}/{np.median(pin[0]):.1f}/{pin[0].max():.1f} dB · "
          f"T_max = {t_max_ns:.1f} ns")


if __name__ == "__main__":
    main()
