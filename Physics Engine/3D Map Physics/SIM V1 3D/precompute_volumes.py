#!/usr/bin/env python3
"""
Multi-Tx volume precompute — the offload+cache stage.

The browser must stay real-time while the physics stays honest, so the resolution order is
  cached volume  ->  DL surrogate  ->  analytic JS fallback.
This script fills the first tier: it solves many transmitter positions offline (Colab A100)
and writes them in the exact float16 format `Frontend/simulator/simulation3d.js` already
loads, so no exporter or frontend change is needed.

Per Tx it runs the mechanism stack through `Combine_3D`, then `to_legacy_volumes()`:
    pl_volume_<txid>.bin   float16, layout (band, x, y, z), dB
    t_volume_<txid>.bin    float16, layout (x, y, z), NANOSECONDS (sentinel 65504)
plus a merged `index.json` describing every cached Tx.

Resumable by design: an existing pair for a txid is skipped, so a dropped Colab session
costs only the Tx in flight. Mirrors the shard-level skip the 2D dataset stage uses.

usage
    python precompute_volumes.py --list 24 --stratify          # pick Tx positions
    python precompute_volumes.py --n-tx 24 --mechanisms path_loss,reflection,diffraction \
           --backend torch --out web/volumes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MECH_DIR = HERE.parent / "Wave Behavior" / "Enivronmental Interaction"
for _p in (HERE, MECH_DIR, HERE.parent.parent / "2D" / "SIM"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import engine_3d  # noqa: E402

F16_MAX = 65504.0


def tx_id(tx) -> str:
    return f"tx_{int(tx[0])}-{int(tx[1])}-{int(tx[2])}"


def choose_tx(scene, n, *, stratify=True, seed=0, y_fixed=None):
    """Pick n transmitter positions from valid_tx_mask.

    Stratified over an 8-way XZ octant split so the cache covers the floor rather than
    clustering in one room — the same idea as the 2D dataset's octant stratification.
    """
    vt = np.load(HERE / "valid_tx_mask.npy")
    if y_fixed is not None:
        m = np.zeros_like(vt)
        m[:, y_fixed, :] = vt[:, y_fixed, :]
        vt = m
    xs, ys, zs = np.nonzero(vt)
    if len(xs) == 0:
        raise SystemExit("valid_tx_mask is empty")
    rng = np.random.default_rng(seed)
    if not stratify:
        sel = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
        return [(int(xs[i]), int(ys[i]), int(zs[i])) for i in sel]

    NX, NZ = scene.NX, scene.NZ
    oct_id = (xs > NX / 2).astype(int) * 2 + (zs > NZ / 2).astype(int)
    quad = (xs > NX / 4).astype(int) + (xs > 3 * NX / 4).astype(int)
    key = oct_id * 3 + quad
    out = []
    for k in np.unique(key):
        idx = np.nonzero(key == k)[0]
        take = max(1, round(n * len(idx) / len(xs)))
        pick = rng.choice(idx, size=min(take, len(idx)), replace=False)
        out += [(int(xs[i]), int(ys[i]), int(zs[i])) for i in pick]
    rng.shuffle(out)
    return out[:n]


def solve_one(scene, tx, bands, mechanisms, backend, bandwidth_hz=None):
    """Run the requested mechanisms and return (PL (nf,NX,NY,NZ) dB, T (NX,NY,NZ) s)."""
    import Combine_3D as CMB
    contribs = []
    if "path_loss" in mechanisms:
        import Path_Loss_3D as PLM
        contribs.append(PLM.solve(scene, tx, bands=bands))
    if "reflection" in mechanisms:
        import Reflection_3D as RFL
        contribs.append(RFL.solve(scene, tx, bands=bands, backend=backend))
    if "diffraction" in mechanisms:
        import Diffraction_3D as DIF
        contribs.append(DIF.solve(scene, tx, bands=bands, n_edges=16))
    if "scattering" in mechanisms:
        import Scattering_3D as SCT
        contribs.append(SCT.solve(scene, tx, bands=bands, backend=backend,
                                  patch_cells=3, rx_downsample=2))
    if not contribs:
        raise SystemExit("no mechanisms selected")
    cf = CMB.combine(contribs, bandwidth_hz=bandwidth_hz, floor_at_fspl=True,
                     scene=scene, tx=tx)
    return CMB.to_legacy_volumes(cf)


def write_volumes(out_dir: Path, tx, pl_db, t_s):
    """float16 .bin pair in the layout simulation3d.js already decodes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tid = tx_id(tx)
    pl = np.nan_to_num(np.asarray(pl_db, np.float32), nan=F16_MAX,
                       posinf=F16_MAX, neginf=F16_MAX)
    pl = np.clip(pl, 0.0, F16_MAX).astype(np.float16)
    t_ns = np.asarray(t_s, np.float64) * 1e9
    t_ns = np.where(np.isfinite(t_ns), np.clip(t_ns, 0.0, F16_MAX - 1.0), F16_MAX)
    t_ns = t_ns.astype(np.float16)
    (out_dir / f"pl_volume_{tid}.bin").write_bytes(pl.tobytes())
    (out_dir / f"t_volume_{tid}.bin").write_bytes(t_ns.tobytes())
    return tid, pl.nbytes + t_ns.nbytes


def merge_index(out_dir: Path, entries, manifest, mechanisms, bands):
    idx_path = out_dir / "index.json"
    old = {}
    if idx_path.exists():
        try:
            prev = json.loads(idx_path.read_text())
            old = {e["tx_id"]: e for e in prev.get("volumes", prev.get("entries", []))}
        except Exception:
            pass
    for e in entries:
        old[e["tx_id"]] = e
    doc = {
        "version": "vol3d-v2",
        "grid_shape": manifest["grid_shape"],
        "cell_size_m": manifest["cell_size_m"],
        "bands_mhz": list(bands),
        "mechanisms": list(mechanisms),
        "pl_layout": "band,x,y,z  float16 dB",
        "t_layout": "x,y,z  float16 NANOSECONDS (sentinel 65504 = unreachable)",
        "count": len(old),
        "volumes": sorted(old.values(), key=lambda e: e["tx_id"]),
    }
    idx_path.write_text(json.dumps(doc, indent=1))
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--n-tx", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "web" / "volumes"))
    ap.add_argument("--mechanisms", default="path_loss,reflection,diffraction")
    ap.add_argument("--bands", default=None, help="comma-separated MHz (default: all)")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "torch"])
    ap.add_argument("--bandwidth-mhz", type=float, default=None)
    ap.add_argument("--y", type=int, default=None, help="fix the Tx height (voxels)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stratify", action="store_true", default=True)
    ap.add_argument("--list", type=int, default=0, help="only print chosen Tx positions")
    ap.add_argument("--force", action="store_true", help="recompute existing volumes")
    a = ap.parse_args(argv)

    scene, manifest = engine_3d.load_scene(HERE)
    bands = ([float(x) for x in a.bands.split(",")] if a.bands else list(scene.freqs))
    mechs = [m.strip() for m in a.mechanisms.split(",") if m.strip()]
    out_dir = Path(a.out).expanduser().resolve()

    n = a.list or a.n_tx
    txs = choose_tx(scene, n, stratify=a.stratify, seed=a.seed, y_fixed=a.y)
    if a.list:
        for t in txs:
            print(tx_id(t), t)
        return 0

    print(f"scene {scene.M.shape}  bands {len(bands)}  mechanisms {mechs}  "
          f"backend {a.backend}")
    print(f"out {out_dir}\n")

    entries, t0, done_bytes = [], time.time(), 0
    for i, tx in enumerate(txs, 1):
        tid = tx_id(tx)
        if not a.force and (out_dir / f"pl_volume_{tid}.bin").exists():
            print(f"[{i}/{len(txs)}] {tid}  SKIP (cached)")
            continue
        t1 = time.time()
        pl, t = solve_one(scene, tx, bands, mechs, a.backend,
                          bandwidth_hz=(a.bandwidth_mhz * 1e6) if a.bandwidth_mhz else None)
        tid, nbytes = write_volumes(out_dir, tx, pl, t)
        done_bytes += nbytes
        fin = np.isfinite(pl) & (pl < 400)
        entries.append({"tx_id": tid, "tx_vox": list(map(int, tx)),
                        "median_pl_db": round(float(np.median(pl[fin])), 2) if fin.any() else None,
                        "mechanisms": mechs, "bands_mhz": bands,
                        "bytes": int(nbytes)})
        print(f"[{i}/{len(txs)}] {tid}  {time.time()-t1:5.1f}s  "
              f"median PL {entries[-1]['median_pl_db']} dB  {nbytes/1e6:.1f} MB")

    doc = merge_index(out_dir, entries, manifest, mechs, bands)
    print(f"\nwrote {len(entries)} new volumes ({done_bytes/1e6:.1f} MB) "
          f"in {time.time()-t0:.1f}s; cache now holds {doc['count']} Tx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
