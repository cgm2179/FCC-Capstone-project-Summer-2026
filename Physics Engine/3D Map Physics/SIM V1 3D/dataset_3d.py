#!/usr/bin/env python3
"""
dataset_3d.py — the single source of truth for surrogate featurization (M4).

Three consumers have to agree on exactly one thing or the surrogate is worse than
useless: **what the input tensor means**. `phase_b3_dataset.ipynb` writes targets,
`phase_c3_train_colab.ipynb` trains against them, and the browser
(`simulation3d.js`) builds the same tensor at inference time. A disagreement in
channel order, blob sigma or the log-distance divisor does not raise — it renders
a confident, wrong field. So the layout lives here, once, and the trainer copies
it verbatim into `web/pl_unet3d.json` for the browser to read back.

WHAT THE SURROGATE PREDICTS
---------------------------
Two channels, not one:

    pl_norm   (PL_db - pl_min) / pl_range          clipped to [0, 1]
    tau_norm  tau_ns / tau_max_ns                  clipped to [0, 1]

`tau` is carried because the browser needs it. The wavefront sweep and the
mechanism time-lapse read a T volume, so a PL-only surrogate could accelerate the
coverage view and nothing else. It also makes the causality constraint
(`tau >= d/c`) expressible as a training loss instead of an aspiration.

WHY tau IS STORED PER POSITION, NOT PER BAND
--------------------------------------------
`SceneV3.arrival_time` takes a frequency, but the slowness field is
`Re(sqrt(eps_r))/c` and P.2040 permittivity moves only a few percent across
600-6200 MHz. Storing one tau per Tx instead of one per (Tx, band) costs
nothing in fidelity and divides that half of the dataset by `bands_per_pos`.

SAMPLING DOMAIN — inside_mask, NOT valid_tx_mask
------------------------------------------------
`valid_tx_mask` (2,510 voxels) is `inside_mask` eroded for physical AP clearance;
it exists so the placement UI cannot drop an antenna inside a wall. A *training*
transmitter has no such constraint, and min-spacing sampling over it saturates at
**370 positions** — which is why the committed `dataset/splits.json` asking for
1,500 was unreachable. Sampling `inside_mask` (69,432 voxels) yields 1,592
positions at 3-voxel spacing. See `max_positions()`.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

SPEC_VERSION = "pl-unet3d-v1"

# Canonical input channel order. The browser rebuilds this list from the contract
# sidecar, so renaming anything here without re-exporting the model breaks inference.
STATIC_CHANNELS = tuple(f"material_onehot_{i}" for i in range(6))
DYNAMIC_CHANNELS = ("tx_blob", "freq_feat", "log_distance")
INPUT_CHANNELS = STATIC_CHANNELS + DYNAMIC_CHANNELS
OUTPUT_CHANNELS = ("pl_norm", "tau_norm")

# Featurization constants. TX_SIGMA_CELLS and LOGDIST_DIVISOR carry over from the
# 2-D pipeline (phase_c_train_colab_v3) so the two surrogates stay comparable.
TX_SIGMA_CELLS = 2.0
LOGDIST_DIVISOR = 3.0
TAU_MAX_NS = 400.0        # interior tau tops out near 260 ns on this scene

C_LIGHT = 299_792_458.0


# --------------------------------------------------------------------------- norm
@dataclass(frozen=True)
class Norm:
    """Normalization window, read from `manifest_3d.json` so it cannot drift."""
    pl_min_db: float
    pl_range_db: float
    freq_log_lo_mhz: float
    freq_log_hi_mhz: float
    tau_max_ns: float = TAU_MAX_NS

    @property
    def pl_max_db(self) -> float:
        return self.pl_min_db + self.pl_range_db

    def pl_to_norm(self, pl_db):
        return np.clip((np.asarray(pl_db, np.float32) - self.pl_min_db)
                       / self.pl_range_db, 0.0, 1.0)

    def norm_to_pl(self, v):
        return np.asarray(v, np.float32) * self.pl_range_db + self.pl_min_db

    def tau_to_norm(self, tau_s):
        ns = np.asarray(tau_s, np.float64) * 1e9
        return np.clip(np.nan_to_num(ns, posinf=self.tau_max_ns), 0.0,
                       self.tau_max_ns).astype(np.float32) / self.tau_max_ns

    def norm_to_tau_ns(self, v):
        return np.asarray(v, np.float32) * self.tau_max_ns

    def freq_feature(self, f_mhz):
        lo, hi = np.log10(self.freq_log_lo_mhz), np.log10(self.freq_log_hi_mhz)
        return float((np.log10(float(f_mhz)) - lo) / (hi - lo))

    def feature_to_freq_mhz(self, ff):
        lo, hi = np.log10(self.freq_log_lo_mhz), np.log10(self.freq_log_hi_mhz)
        return float(10.0 ** (lo + float(ff) * (hi - lo)))


def load_norm(manifest: dict) -> Norm:
    n = manifest["norm"]
    return Norm(pl_min_db=float(n["pl_min_db"]),
                pl_range_db=float(n["pl_range_db"]),
                freq_log_lo_mhz=float(n["freq_log_lo_mhz"]),
                freq_log_hi_mhz=float(n["freq_log_hi_mhz"]))


# --------------------------------------------------------------------------- featurize
def static_channels(M: np.ndarray, n_classes: int = 6) -> np.ndarray:
    """Material one-hot (n_classes, NX, NY, NZ) float32 — the FIXED geometry.

    Constant across the whole dataset, so it is built once and broadcast rather
    than stored per sample. That is the whole reason a shard holds only
    `(tx, freq)` and a target volume.
    """
    return np.stack([(M == c) for c in range(n_classes)], 0).astype(np.float32)


def voxel_coords(shape) -> np.ndarray:
    nx, ny, nz = shape
    gx, gy, gz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    return np.stack([gx, gy, gz], 0).astype(np.float32)


def dynamic_channels(tx, freq_feat: float, coords: np.ndarray, cell_m: float,
                     sigma_cells: float = TX_SIGMA_CELLS,
                     logdist_divisor: float = LOGDIST_DIVISOR) -> np.ndarray:
    """(3, NX, NY, NZ): Tx Gaussian blob, broadcast frequency feature, log-distance.

    The blob localizes the source for a convolutional net that has no other way to
    know where it is; log-distance hands the network the smooth part of the answer
    so its capacity goes to the structure walls impose, not to relearning 1/r.
    """
    d2 = ((coords - np.asarray(tx, np.float32)[:, None, None, None]) ** 2).sum(0)
    blob = np.exp(-d2 / (2.0 * sigma_cells ** 2)).astype(np.float32)
    d_m = np.maximum(np.sqrt(d2) * float(cell_m), 1.0)
    logd = (np.log10(d_m) / logdist_divisor).astype(np.float32)
    ff = np.full(blob.shape, float(freq_feat), np.float32)
    return np.stack([blob, ff, logd], 0)


def make_input(static: np.ndarray, tx, freq_feat: float, coords: np.ndarray,
               cell_m: float) -> np.ndarray:
    """Full (9, NX, NY, NZ) input tensor in the canonical channel order."""
    return np.concatenate([static, dynamic_channels(tx, freq_feat, coords, cell_m)], 0)


def build_input_volume(M: np.ndarray, tx, freq_mhz: float, norm: Norm,
                       cell_m: float, n_classes: int = 6) -> np.ndarray:
    """Convenience wrapper for one full surrogate input volume.

    This is intentionally small glue around the lower-level helpers so the
    browser can mirror one named reference: material one-hot channels first,
    then Tx blob, frequency feature, and log-distance.
    """
    static = static_channels(M, n_classes)
    coords = voxel_coords(M.shape)
    return make_input(static, tx, norm.freq_feature(freq_mhz), coords, cell_m)


def distance_m(tx, coords: np.ndarray, cell_m: float) -> np.ndarray:
    d2 = ((coords - np.asarray(tx, np.float32)[:, None, None, None]) ** 2).sum(0)
    return (np.sqrt(d2) * float(cell_m)).astype(np.float32)


def fspl_db(d_m, f_mhz: float, manifest: dict) -> np.ndarray:
    """Free-space reference used by both the FSPL-floor loss and the baseline.

    Matches `SceneV3.pathloss_maps` exactly: same n_exp, d0 and fspl_const_db, so
    "the surrogate must not beat free space" means the same thing in both places.
    """
    p = manifest["physics"]
    d = np.maximum(np.asarray(d_m, np.float32), float(p["d0_m"]))
    return (20.0 * np.log10(float(f_mhz)) + float(p["fspl_const_db"])
            + 10.0 * float(p["n_exp"]) * np.log10(d)).astype(np.float32)


# --------------------------------------------------------------------------- sampling
def sample_tx_positions(mask: np.ndarray, n: int, spacing: float, seed: int = 0):
    """Blue-noise-ish Tx sampling: shuffle candidates, keep one per spacing cube.

    Returns fewer than `n` when the mask cannot hold that many — check the length.
    Silently returning a short array is how `splits.json` came to claim 1,500
    positions that were never generated.
    """
    cells = np.argwhere(mask)
    rng = np.random.default_rng(seed)
    rng.shuffle(cells)
    taken = np.zeros(mask.shape, bool)
    r = int(np.ceil(spacing))
    out = []
    for x, y, z in cells:
        sl = (slice(max(0, x - r), x + r + 1),
              slice(max(0, y - r), y + r + 1),
              slice(max(0, z - r), z + r + 1))
        if taken[sl].any():
            continue
        taken[x, y, z] = True
        out.append((int(x), int(y), int(z)))
        if len(out) == n:
            break
    return np.array(out, np.int16).reshape(-1, 3)


def max_positions(mask: np.ndarray, spacing: float, seed: int = 0) -> int:
    """How many Tx the mask can actually hold at this spacing. Call before asking."""
    return len(sample_tx_positions(mask, 10 ** 9, spacing, seed))


def make_splits(pos: np.ndarray, shape, seed: int = 1,
                frac=(0.80, 0.10, 0.10)) -> dict:
    """Split BY POSITION, stratified over X-Z octants.

    Splitting by sample would put the same transmitter at 619 MHz in train and at
    3500 MHz in test, and the reported RMSE would be measuring interpolation
    across frequency rather than generalization to an unseen source location.
    """
    nx, _, nz = shape
    ox = np.clip(pos[:, 0].astype(int) * 4 // nx, 0, 3)
    oz = np.clip(pos[:, 2].astype(int) * 4 // nz, 0, 3)
    octant = ox * 4 + oz
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for o in np.unique(octant):
        idx = np.nonzero(octant == o)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * frac[0]))
        n_va = int(round(n * frac[1]))
        train += idx[:n_tr].tolist()
        val += idx[n_tr:n_tr + n_va].tolist()
        test += idx[n_tr + n_va:].tolist()
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def scene_sha(M: np.ndarray) -> str:
    """Same convention as `cache_index.scene_sha` — ties a dataset to a geometry."""
    return hashlib.sha256(np.ascontiguousarray(M).tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- shard IO
def shard_paths(out_dir, s: int) -> dict:
    d = str(out_dir)
    return {"pl": f"{d}/shard_{s:03d}_pl.npy",
            "tau": f"{d}/shard_{s:03d}_tau.npy",
            "meta": f"{d}/shard_{s:03d}_meta.npz"}


def shard_complete(out_dir, s: int, *, expect_pos=None, bands=None, scene_sha=None) -> bool:
    """True if shard `s` is present AND (when expectations are given) matches this run.

    The original check only asked whether the three files existed, which let a
    leftover *smoke* shard (8 positions, written with SHARD_POS=8) look finished. A
    later `full` run (SHARD_POS=25) then skipped that slot and never generated the
    positions it was meant to hold — a silent hole in the position coverage that no
    downstream `pos_id`-uniqueness check can catch. Passing the expected geometry
    makes a mismatched shard count as *incomplete*, so the generator regenerates it.

    With no expectations this is the old files-exist behaviour, so existing callers
    are unaffected.
    """
    p = shard_paths(out_dir, s)
    if not all(os.path.exists(f) for f in p.values()):
        return False
    if expect_pos is None and bands is None and scene_sha is None:
        return True
    try:
        with np.load(p["meta"], allow_pickle=False) as z:
            meta = {k: z[k] for k in z.files}
    except Exception:
        return False
    if scene_sha is not None:
        got = meta.get("scene_sha")
        if got is None or str(np.asarray(got).reshape(-1)[0]) != str(scene_sha):
            return False
    if bands is not None:
        got = [round(float(b), 3) for b in np.ravel(meta.get("bands_mhz", ()))]
        if got != [round(float(b), 3) for b in bands]:
            return False
    if expect_pos is not None:
        pid = meta.get("pos_id")
        if pid is None or int(np.unique(pid).size) != int(expect_pos):
            return False
    return True


def audit_shards(out_dir, *, shard_pos, n_positions, bands=None, scene_sha=None):
    """Census of the shard slots needed to cover positions [0, n_positions).

    Returns `(ok, rows)`. `ok` is True only when every slot is present and matches
    the current run — i.e. no gaps and no smoke leftovers. Each row is
    {shard, positions, expect, got, status} with status in {"ok","missing",
    "mismatch"}, ready to print in the generator's preflight.
    """
    n_shards = int(np.ceil(n_positions / shard_pos))
    rows, ok = [], True
    for s in range(n_shards):
        p0 = s * shard_pos
        p1 = min((s + 1) * shard_pos, n_positions)
        expect = p1 - p0
        meta_path = shard_paths(out_dir, s)["meta"]
        present = os.path.exists(meta_path)
        matches = shard_complete(out_dir, s, expect_pos=expect, bands=bands,
                                 scene_sha=scene_sha)
        got = None
        if present:
            try:
                with np.load(meta_path, allow_pickle=False) as z:
                    got = int(np.unique(z["pos_id"]).size)
            except Exception:
                got = -1
        status = "ok" if matches else ("missing" if not present else "mismatch")
        ok = ok and matches
        rows.append(dict(shard=s, positions=f"{p0}-{p1 - 1}", expect=expect,
                         got=got, status=status))
    return ok, rows


def write_shard(out_dir, s: int, pl: np.ndarray, tau: np.ndarray, meta: dict) -> dict:
    """Raw .npy for the volumes, .npz only for the small metadata.

    Deliberately NOT `savez_compressed` for the volumes: `np.load(mmap_mode='r')`
    cannot memory-map a compressed member, so a compressed shard forces the whole
    dataset into RAM at import. That is the fp32-at-load bug's real root — the
    dataset is 8 GB and Colab has 12.7.
    """
    p = shard_paths(out_dir, s)
    # Write through file handles, not names: np.save/np.savez APPEND .npy/.npz to a
    # path that lacks the suffix, so a ".tmp" name silently lands somewhere else.
    tmp = {k: v + ".part" for k, v in p.items()}
    for key, arr in (("pl", pl), ("tau", tau)):
        with open(tmp[key], "wb") as fh:
            np.save(fh, np.asarray(arr, np.float16))
    with open(tmp["meta"], "wb") as fh:
        np.savez(fh, **meta)
    for k in p:                       # rename last: a killed cell leaves no half shard
        os.replace(tmp[k], p[k])
    return p


def list_shards(out_dir) -> list:
    return sorted(glob.glob(f"{out_dir}/shard_*_meta.npz"))


def open_shard(out_dir, s: int):
    """(pl_memmap, tau_memmap, meta_dict). Volumes stay on disk until indexed."""
    p = shard_paths(out_dir, s)
    pl = np.load(p["pl"], mmap_mode="r")
    tau = np.load(p["tau"], mmap_mode="r")
    meta = dict(np.load(p["meta"]))
    return pl, tau, meta


# --------------------------------------------------------------------------- preflight
def clip_report(scene, manifest: dict, mask: np.ndarray, norm: Norm,
                n_probe: int = 4, bands=None, seed: int = 0) -> dict:
    """Fraction of interior voxels that saturate the normalization ceiling.

    Phase B's preflight refuses to generate when `worst_clipped_fraction` > 0.35 —
    above that the target is mostly the constant 1.0 and a surrogate learns a flat
    plateau. Pre-M4 (M2 satObs): `SceneV3.pathloss_maps` now caps direct crossing
    loss via `sat_obs`, so production-scene clip fractions drop from ~80% to well
    under the 35% gate. Keep this report — it is the regression tripwire if
    `use_satobs` is turned off or the cap regresses.
    """
    rng = np.random.default_rng(seed)
    cells = np.argwhere(mask)
    freqs = list(bands if bands is not None else manifest["freqs_mhz"])
    per_band = {f: [] for f in freqs}
    med = {f: [] for f in freqs}
    for _ in range(int(n_probe)):
        tx = tuple(float(v) for v in cells[rng.integers(len(cells))])
        PL = scene.pathloss_maps(tx)
        for f in freqs:
            p = PL[scene.band_index(f)][mask]
            per_band[f].append(float((p > norm.pl_max_db).mean()))
            med[f].append(float(np.median(p)))
    out = {"pl_max_db": norm.pl_max_db, "n_probe": int(n_probe),
           "clipped_fraction": {float(f): float(np.mean(v)) for f, v in per_band.items()},
           "median_pl_db": {float(f): float(np.mean(v)) for f, v in med.items()}}
    out["worst_clipped_fraction"] = max(out["clipped_fraction"].values())
    return out


# --------------------------------------------------------------------------- contract
def surrogate_contract(manifest: dict, M: np.ndarray, *, bands, mechanisms, mode,
                       metrics: dict | None = None, extra: dict | None = None) -> dict:
    """The `web/pl_unet3d.json` sidecar the browser refuses to run without.

    `simulation3d.js:527` fetches this before touching the .onnx, precisely so a
    model can never be paired with the wrong input layout.
    """
    norm = load_norm(manifest)
    return {
        "spec_version": SPEC_VERSION,
        "model_file": "pl_unet3d.onnx",
        "mode": mode,
        "mechanisms": list(mechanisms),
        "grid_shape": [int(v) for v in M.shape],
        "cell_size_m": float(manifest["cell_size_m"]),
        "scene_sha": scene_sha(M),
        "bands_mhz": [float(b) for b in bands],
        "input": {
            "name": "x",
            "layout": "N,C,X,Y,Z",
            "dtype": "float32",
            "channels": list(INPUT_CHANNELS),
            "material_classes": len(STATIC_CHANNELS),
            "tx_blob_sigma_cells": TX_SIGMA_CELLS,
            "logdist_divisor": LOGDIST_DIVISOR,
            "logdist_min_m": 1.0,
            "freq_log_lo_mhz": norm.freq_log_lo_mhz,
            "freq_log_hi_mhz": norm.freq_log_hi_mhz,
        },
        "output": {
            "name": "y",
            "layout": "N,C,X,Y,Z",
            "channels": list(OUTPUT_CHANNELS),
            "pl_min_db": norm.pl_min_db,
            "pl_range_db": norm.pl_range_db,
            "tau_max_ns": norm.tau_max_ns,
        },
        "metrics": metrics or {},
        **(extra or {}),
    }


def write_surrogate_contract(path, contract: dict) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(contract, indent=1))
    return str(path)


# --------------------------------------------------------------------------- selftest
def _selftest() -> int:
    ok, fails = 0, []

    def check(name, cond, msg=""):
        nonlocal ok
        if cond:
            ok += 1
            print(f"  PASS  {name}")
        else:
            fails.append(name)
            print(f"  FAIL  {name}  {msg}")

    manifest = json.loads((HERE / "manifest_3d.json").read_text())
    M = np.load(HERE / "material_grid.npy")
    inside = np.load(HERE / "inside_mask.npy")
    norm = load_norm(manifest)

    check("channel count is 9", len(INPUT_CHANNELS) == 9, str(INPUT_CHANNELS))
    check("output is (pl, tau)", OUTPUT_CHANNELS == ("pl_norm", "tau_norm"))

    # normalization round-trips
    pl = np.array([40.0, 105.0, 170.0], np.float32)
    check("pl round-trip", np.allclose(norm.norm_to_pl(norm.pl_to_norm(pl)), pl, atol=1e-4))
    check("pl clips above the ceiling", float(norm.pl_to_norm(np.array([900.0]))[0]) == 1.0)
    check("tau round-trip",
          abs(float(norm.norm_to_tau_ns(norm.tau_to_norm(np.array([120e-9])))[0]) - 120.0) < 1e-3)
    check("tau inf -> ceiling", float(norm.tau_to_norm(np.array([np.inf]))[0]) == 1.0)
    for f in manifest["freqs_mhz"]:
        if abs(norm.feature_to_freq_mhz(norm.freq_feature(f)) - f) > 1e-3:
            fails.append("freq round-trip")
            break
    else:
        ok += 1
        print("  PASS  freq feature round-trips on every manifest band")

    # featurization shapes and semantics
    static = static_channels(M, len(manifest["materials"]))
    coords = voxel_coords(M.shape)
    tx = (60, 5, 63)
    x = make_input(static, tx, norm.freq_feature(3500.0), coords, manifest["cell_size_m"])
    check("input shape", x.shape == (9,) + M.shape, str(x.shape))
    check("one-hot sums to 1", np.allclose(x[:6].sum(0), 1.0))
    check("blob peaks at tx", int(np.argmax(x[6])) == np.ravel_multi_index(tx, M.shape))
    check("freq channel constant", float(x[7].max() - x[7].min()) == 0.0)
    check("log-distance rises with range", x[8][60, 5, 63] < x[8][200, 5, 63])

    # FSPL reference must agree with the engine's own free-space term
    d = distance_m(tx, coords, manifest["cell_size_m"])
    fs = fspl_db(d, 3500.0, manifest)
    check("fspl matches manifest anchor",
          abs(float(fs[61, 5, 63]) - (20 * np.log10(3500.0)
                                      + manifest["physics"]["fspl_const_db"])) < 0.6,
          f"{float(fs[61, 5, 63]):.2f} dB at ~1 cell")

    # sampling: the finding that invalidated the committed splits
    valid = np.load(HERE / "valid_tx_mask.npy")
    n_valid = max_positions(valid, 3.0)
    n_inside = max_positions(inside, 3.0)
    check("valid_tx cannot hold 1500 positions", n_valid < 400, f"{n_valid}")
    check("inside_mask can hold >= 1500", n_inside >= 1500, f"{n_inside}")

    pos = sample_tx_positions(inside, 200, 3.0, seed=0)
    check("sampled positions are interior",
          bool(inside[pos[:, 0], pos[:, 1], pos[:, 2]].all()))
    sp = make_splits(pos, M.shape)
    allidx = sorted(sp["train"] + sp["val"] + sp["test"])
    check("splits partition exactly", allidx == list(range(len(pos))))
    check("splits do not overlap",
          not (set(sp["train"]) & set(sp["val"]) or set(sp["val"]) & set(sp["test"])))

    contract = surrogate_contract(manifest, M, bands=manifest["freqs_mhz"],
                                  mechanisms=["path_loss"], mode="indoor")
    check("contract carries the channel order",
          contract["input"]["channels"] == list(INPUT_CHANNELS))
    check("contract grid matches the scene", contract["grid_shape"] == list(M.shape))

    print(f"\n{ok} passed, {len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="surrogate dataset featurization (M4)")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--budget", action="store_true",
                    help="how many Tx positions each mask can hold")
    a = ap.parse_args()
    if a.budget:
        M = np.load(HERE / "material_grid.npy")
        for name in ("inside_mask", "valid_tx_mask"):
            m = np.load(HERE / f"{name}.npy")
            row = "  ".join(f"sp={s}: {max_positions(m, s):5d}" for s in (1.0, 2.0, 3.0))
            print(f"{name:16s} {int(m.sum()):6d} voxels   {row}")
        print(f"grid {M.shape}  scene_sha {scene_sha(M)}")
        raise SystemExit(0)
    raise SystemExit(_selftest() if a.test else (ap.print_help() or 0))
