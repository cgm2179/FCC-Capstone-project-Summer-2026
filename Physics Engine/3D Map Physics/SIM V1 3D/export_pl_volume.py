#!/usr/bin/env python3
"""
export_pl_volume.py — export the real SceneV3 volumes for the browser.

Computes, for one transmitter, the full-physics PL(x,y,z) per band and the
eikonal T(x,y,z), and serializes them to compact float16 binaries the 3-D
Simulation tab fetches from SIM3D/web/volumes/. These drive the browser's two
cheap production views: the "coverage field" (PL) and the "wavefront sweep" (T,
scrubbed by an arrival-time threshold).

This is the ONE-transmitter front door; `precompute_volumes.py` is the batch
(Colab A100) path. Both write through the same `write_volumes` / `merge_index`
so a single-Tx export and a batch export produce byte-identical formats — they
drifted apart once already, and the browser silently could not read the batch
output for it.

Outputs (SIM3D/web/volumes/):
  pl_volume_<txid>.bin    float16, C-order (nband, NX, NY, NZ), dB
  t_volume_<txid>.bin     float16, C-order (NX, NY, NZ), nanoseconds
                          (barrier / unreachable voxels → 65504 sentinel)
  m_<mech>_<txid>.bin     float16 (nband, NX, NY, NZ) dB — one mechanism alone
  tau_<mech>_<txid>.bin   float16 (NX, NY, NZ) ns — that mechanism's first arrival
  index.json              the browser's volume catalog

With --mechanisms the export runs the full mechanism stack through Combine_3D
instead of SceneV3 alone, which is what feeds the browser's per-mechanism views
and the mechanism time-lapse. Without it the output is exactly the path-loss-only
volume this script has always written.

usage:
  python "SIM V1 3D/export_pl_volume.py"                 # auto Tx (interior centroid, ceiling)
  python "SIM V1 3D/export_pl_volume.py" --tx 131 9 59   # explicit voxel
  python "SIM V1 3D/export_pl_volume.py" --t-freq 2442   # eikonal T at a chosen band
  python "SIM V1 3D/export_pl_volume.py" --mechanisms path_loss,reflection,diffraction,scattering
"""
import argparse
import json
from pathlib import Path

import numpy as np

import precompute_volumes as PV
from engine_3d import load_scene

HERE = Path(__file__).resolve().parent
T_UNREACHABLE_NS = PV.F16_MAX   # float16 max — sentinel for barrier / unreachable voxels


def pick_tx(valid, want=(0.5, 0.85, 0.5)):
    """Nearest valid-Tx voxel to a fractional target (mirrors run_one_calc)."""
    ijk = np.argwhere(valid)
    target = np.array([want[i] * valid.shape[i] for i in range(3)])
    return ijk[np.argmin(((ijk - target) ** 2).sum(1))].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tx", type=float, nargs=3, default=None)
    ap.add_argument("--t-freq", type=float, default=3500.0,
                    help="frequency (MHz) for the eikonal T solve (path-loss-only mode)")
    ap.add_argument("--mechanisms", default=None,
                    help="comma-separated mechanism stack (path_loss,reflection,"
                         "diffraction,scattering); omit for the path-loss-only volume")
    ap.add_argument("--bands", default=None, help="comma-separated MHz (default: all)")
    ap.add_argument("--mech-bands", default=None,
                    help="band subset for the per-mechanism channels only")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "torch"])
    ap.add_argument("--n-edges", type=int, default=16)
    ap.add_argument("--mode", default=None,
                    help="vacuum | indoor | o2i | outdoor (default: the manifest's `mode`)")
    ap.add_argument("--out", default=str(HERE / "web" / "volumes"))
    args = ap.parse_args()

    import modes_3d as MD
    man0 = json.loads((HERE / "manifest_3d.json").read_text())
    mode = args.mode or man0.get("mode", MD.DEFAULT_MODE)
    info = MD.describe(mode)
    if not info["available"]:
        raise SystemExit(f"mode {mode!r} unavailable: {info['unavailable_reason']}")

    scene, man = MD.build_scene(mode)
    if mode == "o2i":
        # no Tx voxel exists: the source is 416 m outside the grid
        txi = list(MD.bs_field_3d(scene, MD.forte_hall_geometry()["arrival_bearing_deg"],
                                  bands=[float(scene.freqs[0])]).tx_vox)
        tx = np.array(txi, float)
    else:
        vt = HERE / "valid_tx_mask.npy"
        valid = np.load(vt) if vt.exists() else np.ones(scene.M.shape, bool)
        tx = np.array(args.tx, float) if args.tx else pick_tx(valid)
        txi = [int(round(v)) for v in tx]
    nx, ny, nz = scene.M.shape
    bands = ([float(x) for x in args.bands.split(",")] if args.bands
             else [float(f) for f in man["freqs_mhz"]])
    out_dir = Path(args.out).expanduser().resolve()

    band_sel = None
    if args.mech_bands:
        want = [float(x) for x in args.mech_bands.split(",")]
        band_sel = [min(range(len(bands)), key=lambda i: abs(bands[i] - w)) for w in want]

    # An explicit --mechanisms wins; otherwise the mode supplies its own stack, which is
    # the only way `o2i` ever gets its facade source.
    mechs = ([m.strip() for m in args.mechanisms.split(",")] if args.mechanisms
             else (list(info["mechanisms"]) if mode != "indoor" else []))
    print(f"mode {mode} — {info['label']}")
    print(f"grid {nx}x{ny}x{nz}  tx(voxel)={txi}  bands={bands}  "
          + (f"mechanisms={mechs}" if mechs else f"path-loss only, T@{args.t_freq:.0f} MHz"))

    cf = contribs = None
    if mechs:
        edges, relay = PV._prepare_diffraction(scene, mechs, args.n_edges, None)
        PL, T, cf, contribs = PV.solve_one(scene, tx, bands, mechs, args.backend,
                                           relay=relay, edges=edges,
                                           n_edges=args.n_edges, mode=mode)
    else:
        idx = [scene.band_index(f) for f in bands]
        PL = np.ascontiguousarray(scene.pathloss_maps(tx)[idx])   # (nf,nx,ny,nz) dB
        T = scene.arrival_time(tx, args.t_freq)                   # (nx,ny,nz) seconds

    txid, nbytes, t_max_ns = PV.write_volumes(out_dir, txi, PL, T)
    mech_files = {}
    if contribs:
        mech_files, mbytes = PV.write_mechanism_channels(out_dir, txi, cf, contribs,
                                                         band_sel=band_sel, scene=scene)
        nbytes += mbytes

    mech_bands = [bands[i] for i in band_sel] if band_sel else bands
    # Same content key the batch path computes, so a volume written here counts as a cache
    # hit there (and vice versa) instead of being silently resolved by filename.
    key = PV.content_key(scene_sha=PV.scene_sha(scene), mode=mode, tx_vox=txi,
                         bands=bands, mechanisms=mechs or ["path_loss"],
                         engine_ver=PV.engine_version(), mech_bands=mech_bands)
    entry = {
        "txid": txid, "mode": mode, "key": key,
        "scene_sha": PV.scene_sha(scene), "engine_ver": PV.engine_version(),
        "tx_vox": txi, "grid_shape": [nx, ny, nz], "bands": bands,
        "t_max_ns": t_max_ns, "t_unreachable_ns": T_UNREACHABLE_NS,
        "pl_file": f"pl_volume_{txid}.bin", "t_file": f"t_volume_{txid}.bin",
        "mechanisms": mech_files,
        "mech_bands": mech_bands,
        "mechanism_names": mechs or ["path_loss"],
        "t_freq_mhz": None if mechs else args.t_freq,
        "norm": man.get("norm"), "bytes": int(nbytes),
    }
    doc = PV.merge_index(out_dir, [entry], man, mechs or ["path_loss"], bands, mode=mode)

    pin = PL[:, scene.inside] if scene.inside is not None else PL.reshape(len(bands), -1)
    print(f"wrote {out_dir}/{entry['pl_file']}, {entry['t_file']}"
          + (f", {len(mech_files)} mechanism channel pairs" if mech_files else "")
          + f" ({nbytes/1e6:.1f} MB total), index.json ({doc['count']} vols)")
    print(f"  PL band0 min/median/max = {pin[0].min():.1f}/{np.median(pin[0]):.1f}/"
          f"{pin[0].max():.1f} dB · T_max = {t_max_ns:.1f} ns")
    for m, f in mech_files.items():
        print(f"  {m:12s} T_max {f['t_max_ns']:8.1f} ns  {f['bytes']/1e6:5.1f} MB  "
              f"({f['combine_as']})")


if __name__ == "__main__":
    main()
