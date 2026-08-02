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

MECHANISM CHANNELS (--mech-channels, on by default)
    m_<mech>_<txid>.bin    float16, layout (band, x, y, z), dB   -- that mechanism alone
    tau_<mech>_<txid>.bin  float16, layout (x, y, z), NANOSECONDS -- its first arrival
These are what let the browser show "how much of the signal here is reflected?" and, more
importantly, play the MECHANISM TIME-LAPSE: each mechanism sweeps on its own tau channel
against a shared clock, so the direct front arrives first, the reflected after it, the
diffracted after that, and the diffuse halo last. Storing per-mechanism dB (rather than a
precomputed share) keeps the file self-describing: the browser derives the contribution
share against the total it already has, as 10**((PL_total - PL_mech)/10).

    Size: one band-set costs (1 + n_mech) * 2 bytes/voxel/band for dB plus a small tau
    channel. On the production grid that is ~11.8 MB per mechanism at 10 bands, so
    --mech-bands trims the export when only a display band or two is wanted.

Resumable by design: an existing pair for a txid is skipped, so a dropped Colab session
costs only the Tx in flight. Mirrors the shard-level skip the 2D dataset stage uses.

usage
    python precompute_volumes.py --list 24 --stratify          # pick Tx positions
    python precompute_volumes.py --n-tx 24 --mechanisms path_loss,reflection,diffraction \
           --backend torch --out web/volumes
    python precompute_volumes.py --n-tx 1 --mech-bands 2442,3500   # small viz export
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


def solve_one(scene, tx, bands, mechanisms, backend, bandwidth_hz=None,
              relay=None, edges=None, n_edges=16):
    """Run the requested mechanisms for one Tx.

    Returns (PL (nf,NX,NY,NZ) dB, T (NX,NY,NZ) s, CombinedField, contribs) — the last two
    carry the per-mechanism breakdown the channel exporter needs. The (PL, T) pair alone
    is exactly what the legacy single-channel path wrote, so callers that only want the
    total can ignore the rest.
    """
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
        if relay is not None and edges is None:
            raise ValueError("relay cache given without its edge list — relay[i] is "
                             "positional, so the two must travel together")
        contribs.append(DIF.solve(scene, tx, bands=bands, edges=edges,
                                  n_edges=n_edges if edges is None else len(edges),
                                  relay=relay))
    if "scattering" in mechanisms:
        import Scattering_3D as SCT
        contribs.append(SCT.solve(scene, tx, bands=bands, backend=backend,
                                  patch_cells=3, rx_downsample=2))
    if not contribs:
        raise SystemExit("no mechanisms selected")
    cf = CMB.combine(contribs, bandwidth_hz=bandwidth_hz, floor_at_fspl=True,
                     scene=scene, tx=tx)
    pl, t = CMB.to_legacy_volumes(cf)
    return pl, t, cf, contribs


def _f16_db(a):
    """dB volume -> float16, NaN/inf parked on the sentinel (never silently zero)."""
    x = np.nan_to_num(np.asarray(a, np.float32), nan=F16_MAX,
                      posinf=F16_MAX, neginf=F16_MAX)
    return np.clip(x, 0.0, F16_MAX).astype(np.float16)


def _f16_ns(t_s):
    """seconds -> float16 nanoseconds, unreachable on the 65504 sentinel."""
    ns = np.asarray(t_s, np.float64) * 1e9
    ns = np.where(np.isfinite(ns), np.clip(ns, 0.0, F16_MAX - 1.0), F16_MAX)
    return ns.astype(np.float16)


def _t_max_ns(t_ns) -> float:
    """Largest reachable arrival time — the browser's sweep uses it as the clock end."""
    live = np.asarray(t_ns, np.float32)
    live = live[live < F16_MAX - 1.0]
    return round(float(live.max()), 2) if live.size else 0.0


def write_volumes(out_dir: Path, tx, pl_db, t_s):
    """float16 .bin pair in the layout simulation3d.js already decodes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tid = tx_id(tx)
    pl = _f16_db(pl_db)
    t_ns = _f16_ns(t_s)
    (out_dir / f"pl_volume_{tid}.bin").write_bytes(pl.tobytes())
    (out_dir / f"t_volume_{tid}.bin").write_bytes(t_ns.tobytes())
    return tid, pl.nbytes + t_ns.nbytes, _t_max_ns(t_ns)


def write_mechanism_channels(out_dir: Path, tx, cf, contribs, *, band_sel=None, scene=None):
    """Per-mechanism dB + first-arrival channels beside the total volume.

    band_sel  indices into the solved band list to export (None = all). Trimming bands is
              the only lever that meaningfully shrinks these files.
    scene     if given, also writes the geometric (vacuum d/c) arrival for path_loss —
              see TWO CLOCKS below.

    TWO CLOCKS, ON PURPOSE
    ----------------------
    The mechanisms do not share an arrival-time convention, and comparing them naively
    puts the fronts in the wrong order:

        path_loss    tau = the EIKONAL — includes in-wall slowdown (~+6 ns median here)
        reflection   tau = geometric image-source path length / c   (vacuum speed)
        diffraction  tau = (s' + s) / c                             (vacuum speed)
        scattering   tau = (|tx-p| + |p-rx|) / c                    (vacuum speed)

    Each is right on its own terms, but only the direct path is charged for the walls it
    slows down in, so it arrives LAST in 99% of interior voxels — the reflected front
    appears to beat the line-of-sight field, which is nonsense as a sequence.

    So path_loss carries both: `tau_file` is the eikonal (what the wavefront sweep shows,
    physically the true first arrival) and `tau_geom_file` is d/c. The time-lapse uses the
    geometric one for every layer, making it an honest PATH-LENGTH comparison in which
    direct <= reflected <= diffracted holds by construction. Charging the other three for
    in-wall slowdown is the real fix and is a physics change, not an export change.

    Returns ({mech: {pl_file, tau_file, t_max_ns, ...}}, bytes_written).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tid = tx_id(tx)
    out, total = {}, 0
    for c in contribs:
        mech = c.mechanism
        # power -> dB with the same reference as the total volume, so the browser can take
        # the ratio directly. float64 inside: deep-shadow power is ~1e-54 and underflows
        # float32 to zero, which would report PL = +inf for a live voxel.
        p = np.asarray(cf.per_mech_lin[mech], np.float64)
        with np.errstate(divide="ignore"):
            pl_db = -10.0 * np.log10(np.maximum(p, 1e-300))

        # First arrival for this mechanism alone: min over bands, matching the semantic
        # Combine_3D.T_first uses for the total.
        tau = np.asarray(c.tau_first, np.float32)
        tau = np.nanmin(tau, axis=0) if tau.ndim == 4 else tau

        if band_sel is not None:
            pl_db = pl_db[band_sel]

        pl16, tau16 = _f16_db(pl_db), _f16_ns(tau)
        pl_name, tau_name = f"m_{mech}_{tid}.bin", f"tau_{mech}_{tid}.bin"
        (out_dir / pl_name).write_bytes(pl16.tobytes())
        (out_dir / tau_name).write_bytes(tau16.tobytes())
        nbytes = pl16.nbytes + tau16.nbytes
        info = {"pl_file": pl_name, "tau_file": tau_name, "t_max_ns": _t_max_ns(tau16),
                "tau_convention": "eikonal" if mech == "path_loss" else "geometric",
                "combine_as": c.combine_as}

        # the second clock: vacuum d/c for the direct path, so the time-lapse can compare
        # every mechanism on the same convention (see the docstring)
        if mech == "path_loss" and scene is not None:
            geom16 = _f16_ns(scene.geometric_time(tx))
            geom_name = f"tau_geom_{mech}_{tid}.bin"
            (out_dir / geom_name).write_bytes(geom16.tobytes())
            nbytes += geom16.nbytes
            info["tau_geom_file"] = geom_name
            info["tau_geom_max_ns"] = _t_max_ns(geom16)

        info["bytes"] = int(nbytes)
        total += nbytes
        out[mech] = info
    return out, total


def _entry_id(e):
    """Tx id of an index entry, tolerating the pre-v3 `tx_id` spelling."""
    return e.get("txid") or e.get("tx_id")


def merge_index(out_dir: Path, entries, manifest, mechanisms, bands):
    """Rewrite index.json, preserving entries for transmitters not in this run.

    Schema note (v3): entries carry `txid`, `pl_file` and `t_file` because that is what
    `simulation3d.js` reads. v2 wrote `tx_id` and no file names at all, so the browser
    could not load anything the batch precompute produced; entries in that shape are
    migrated on read rather than dropped.
    """
    idx_path = out_dir / "index.json"
    old = {}
    if idx_path.exists():
        try:
            prev = json.loads(idx_path.read_text())
            for e in prev.get("volumes", prev.get("entries", [])):
                tid = _entry_id(e)
                if tid:
                    old[tid] = migrate_entry(e, out_dir, manifest)
        except Exception as exc:                      # a corrupt index must not lose data
            print(f"  warning: could not read existing index.json ({exc}); "
                  f"entries not in this run will be dropped")
    for e in entries:
        old[_entry_id(e)] = e
    doc = {
        "version": "vol3d-v3",
        "grid_shape": manifest["grid_shape"],
        "cell_size_m": manifest["cell_size_m"],
        "bands_mhz": list(bands),
        "mechanisms": list(mechanisms),
        "pl_layout": "band,x,y,z  float16 dB",
        "t_layout": "x,y,z  float16 NANOSECONDS (sentinel 65504 = unreachable)",
        "mech_layout": "m_<mech>_<txid>.bin = band,x,y,z float16 dB; "
                       "tau_<mech>_<txid>.bin = x,y,z float16 ns",
        "count": len(old),
        "volumes": sorted(old.values(), key=lambda e: _entry_id(e)),
    }
    idx_path.write_text(json.dumps(doc, indent=1))
    return doc


def migrate_entry(e: dict, out_dir: Path, manifest: dict) -> dict:
    """Bring a pre-v3 entry up to the schema the browser reads, in place of dropping it.

    Fills the fields v2 omitted (`txid`, `pl_file`, `t_file`, `grid_shape`, `bands`) and
    recovers `t_max_ns` from the stored T volume, since the sweep clock needs it.
    """
    tid = _entry_id(e)
    if not tid:
        return e
    e = dict(e)
    e["txid"] = tid
    e.pop("tx_id", None)
    e.setdefault("pl_file", f"pl_volume_{tid}.bin")
    e.setdefault("t_file", f"t_volume_{tid}.bin")
    e.setdefault("grid_shape", list(manifest["grid_shape"]))
    if "bands" not in e:
        e["bands"] = e.pop("bands_mhz", list(manifest.get("freqs_mhz", [])))
    if isinstance(e.get("mechanisms"), list):
        e["mechanism_names"] = e.pop("mechanisms")
    if "t_max_ns" not in e:
        tp = out_dir / e["t_file"]
        e["t_max_ns"] = (_t_max_ns(np.fromfile(tp, dtype=np.float16))
                         if tp.exists() else 0.0)
    e.setdefault("t_unreachable_ns", F16_MAX)
    return e


def _prepare_diffraction(scene, mechs, n_edges, cache_path):
    """(edges, relay) for the whole run — built once, reused by every transmitter.

    An edge's second leg (edge anchor -> every voxel) is fixed geometry, so solving it per
    Tx is 96% of the per-Tx cost for nothing. Building it once turns n_edges solves PER TX
    into n_edges solves TOTAL. `select_edges` is transmitter-independent, so the same
    ordered edge list must be handed to `solve()` for `relay[ei]` to line up with it.
    """
    if "diffraction" not in mechs:
        return None, None
    import Diffraction_3D as DIF

    path = Path(cache_path) if cache_path else (HERE / "cache" / f"relay_{n_edges}.npy")
    relay, edges = DIF.load_relay_cache(path)
    if relay is not None and len(edges) == n_edges:
        print(f"diffraction: relay cache hit ({path.name}, {len(edges)} edges, "
              f"{relay.nbytes/1e6:.0f} MB)")
        return edges, relay

    inside = scene.inside if scene.inside is not None else np.ones(scene.M.shape, bool)
    all_edges = DIF.find_diffracting_edges_3d(scene.M, inside, y_band=(1, scene.NY - 1))
    edges = DIF.select_edges(all_edges, None, n_edges=n_edges)
    print(f"diffraction: building relay cache for {len(edges)} edges "
          f"(once for the whole run) -> {path}")
    t0 = time.time()
    relay = DIF.build_relay_cache(scene, edges, path=path, progress=True)
    print(f"diffraction: relay cache built in {time.time()-t0:.1f}s "
          f"({relay.nbytes/1e6:.0f} MB)")
    return edges, relay


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
    ap.add_argument("--mech-channels", dest="mech_channels", action="store_true",
                    default=True, help="also write per-mechanism dB + tau channels")
    ap.add_argument("--no-mech-channels", dest="mech_channels", action="store_false")
    ap.add_argument("--mech-bands", default=None,
                    help="comma-separated MHz subset for the mechanism channels only "
                         "(default: every solved band)")
    ap.add_argument("--n-edges", type=int, default=16, help="diffracting edges to solve")
    ap.add_argument("--relay-cache", default=None,
                    help="path for the diffraction relay cache (.npy); built if missing. "
                         "Default: cache/relay_<n_edges>.npy beside this script")
    a = ap.parse_args(argv)

    scene, manifest = engine_3d.load_scene(HERE)
    bands = ([float(x) for x in a.bands.split(",")] if a.bands else list(scene.freqs))
    mechs = [m.strip() for m in a.mechanisms.split(",") if m.strip()]
    out_dir = Path(a.out).expanduser().resolve()

    band_sel = None
    if a.mech_bands:
        want = [float(x) for x in a.mech_bands.split(",")]
        band_sel = [min(range(len(bands)), key=lambda i: abs(bands[i] - w)) for w in want]

    n = a.list or a.n_tx
    txs = choose_tx(scene, n, stratify=a.stratify, seed=a.seed, y_fixed=a.y)
    if a.list:
        for t in txs:
            print(tx_id(t), t)
        return 0

    print(f"scene {scene.M.shape}  bands {len(bands)}  mechanisms {mechs}  "
          f"backend {a.backend}")
    print(f"out {out_dir}"
          + (f"  mech-channels on ({len(band_sel) if band_sel else len(bands)} bands)"
             if a.mech_channels else "  mech-channels off") + "\n")

    edges, relay = _prepare_diffraction(scene, mechs, a.n_edges, a.relay_cache)

    mech_bands = [bands[i] for i in band_sel] if band_sel else list(bands)
    entries, t0, done_bytes = [], time.time(), 0
    for i, tx in enumerate(txs, 1):
        tid = tx_id(tx)
        if not a.force and (out_dir / f"pl_volume_{tid}.bin").exists():
            print(f"[{i}/{len(txs)}] {tid}  SKIP (cached)")
            continue
        t1 = time.time()
        pl, t, cf, contribs = solve_one(
            scene, tx, bands, mechs, a.backend, relay=relay, edges=edges,
            n_edges=a.n_edges,
            bandwidth_hz=(a.bandwidth_mhz * 1e6) if a.bandwidth_mhz else None)
        tid, nbytes, tmax = write_volumes(out_dir, tx, pl, t)
        mech_files = {}
        if a.mech_channels:
            mech_files, mbytes = write_mechanism_channels(out_dir, tx, cf, contribs,
                                                          band_sel=band_sel, scene=scene)
            nbytes += mbytes
        done_bytes += nbytes
        fin = np.isfinite(pl) & (pl < 400)
        entries.append({"txid": tid, "tx_vox": list(map(int, tx)),
                        "grid_shape": list(map(int, scene.M.shape)),
                        "bands": bands, "t_max_ns": tmax, "t_unreachable_ns": F16_MAX,
                        "pl_file": f"pl_volume_{tid}.bin", "t_file": f"t_volume_{tid}.bin",
                        "mechanisms": mech_files, "mech_bands": mech_bands,
                        "mechanism_names": mechs,
                        "median_pl_db": round(float(np.median(pl[fin])), 2) if fin.any() else None,
                        "bytes": int(nbytes)})
        print(f"[{i}/{len(txs)}] {tid}  {time.time()-t1:5.1f}s  "
              f"median PL {entries[-1]['median_pl_db']} dB  T_max {tmax:.0f} ns  "
              f"{nbytes/1e6:.1f} MB"
              + (f"  +{len(mech_files)} mech channels" if mech_files else ""))

    doc = merge_index(out_dir, entries, manifest, mechs, bands)
    print(f"\nwrote {len(entries)} new volumes ({done_bytes/1e6:.1f} MB) "
          f"in {time.time()-t0:.1f}s; cache now holds {doc['count']} Tx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
