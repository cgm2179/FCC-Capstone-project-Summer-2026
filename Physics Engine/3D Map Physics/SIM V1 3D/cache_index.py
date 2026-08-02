#!/usr/bin/env python3
"""
cache_index.py — the content-addressed volume cache (M3, offload + cache).

The browser must stay real-time while the physics stays honest, so heavy solves run
offline (Colab A100) and land here. This module owns the catalog: what is cached, whether
it is still valid, and what to delete first when the disk fills.

WHY CONTENT ADDRESSING, AND NOT FILENAMES
`precompute_volumes.py` used to resume by asking "does pl_volume_tx_66-5-54.bin exist?".
That is wrong in a way that is silent and expensive: change --mechanisms, widen --bands,
or edit a physics module, and the old file still exists, so the run skips it and you keep
a stale volume while believing you recomputed it. A cache entry is only valid for the
exact inputs that produced it:

    key = H(scene_sha, mode, tx_vox, bands, mechanisms, engine_ver)

`engine_ver` is a hash of the PHYSICS SOURCE itself, so editing `Reflection_3D.py` or
`engine_3d.py` invalidates every affected volume automatically. Nobody has to remember to
bump a version — forgetting is the failure this is designed around.

ONE INDEX, NOT TWO
`web/volumes/index.json` is what the browser fetches, so the cache metadata lives IN it
rather than in a parallel sidecar that can drift. Every entry keeps the fields
`simulation3d.js` reads plus the cache's own bookkeeping (`key`, `scene_sha`,
`engine_ver`, `created`, `last_used`, `hits`, `bytes`). The browser ignores what it does
not need.

LRU ON DISK
Volumes are ~13-33 MB each and a full Tx sweep is hundreds. `gc(max_bytes)` evicts whole
transmitters, least-recently-used first, deleting every file the entry owns. Eviction is
by entry, never by file: half a volume is worse than none, because the browser would fetch
a 404 mid-render instead of falling through to the analytic tier.

usage
    python cache_index.py --stats
    python cache_index.py --verify              # files present and correctly sized?
    python cache_index.py --gc --max-gb 2       # evict LRU until under budget
    python cache_index.py --prune-stale         # drop entries the current engine invalidates
    python cache_index.py --test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MECH_DIR = HERE.parent / "Wave Behavior" / "Enivronmental Interaction"
SIM2D = HERE.parent.parent / "2D" / "SIM"
DEFAULT_DIR = HERE / "web" / "volumes"

# Bump only for changes that alter results WITHOUT touching these files (a new numpy with
# different rounding, say). Ordinary physics edits invalidate on their own via the hash.
CACHE_EPOCH = 1

# Files whose contents define "the engine". A change to any of them means every cached
# volume that used it is stale.
PHYSICS_SOURCES = [
    HERE / "engine_3d.py",
    HERE / "physics_3d.py",
    HERE / "contracts.py",
    HERE / "modes_3d.py",
    MECH_DIR / "Combine_3D.py",
    MECH_DIR / "Path_Loss_3D.py",
    MECH_DIR / "Reflection_3D.py",
    MECH_DIR / "Diffraction_3D.py",
    MECH_DIR / "Scattering_3D.py",
    MECH_DIR / "Refraction_3D.py",
    MECH_DIR / "Absorption_3D.py",
    SIM2D / "physics_v2.py",
]

# Keys `simulation3d.js` dereferences on every entry. Losing one is a TypeError in the
# browser, not a graceful fallback, so the cache refuses to write an entry without them.
BROWSER_KEYS = ("txid", "tx_vox", "grid_shape", "bands", "t_max_ns", "pl_file", "t_file")


# --------------------------------------------------------------------------- hashing
def _sha(*parts, n=16) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else repr(p).encode())
    return h.hexdigest()[:n]


def engine_version(sources=None) -> str:
    """Hash of the physics source. Editing a solver invalidates its volumes, silently
    and correctly, without anyone remembering to bump a number."""
    h = hashlib.sha256()
    h.update(f"epoch={CACHE_EPOCH}".encode())
    for p in sorted(sources or PHYSICS_SOURCES, key=lambda x: x.name):
        h.update(p.name.encode())
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()[:16]


def scene_sha(scene_or_grid) -> str:
    """Hash of the material grid — the scene identity the volumes were solved against."""
    M = getattr(scene_or_grid, "M", scene_or_grid)
    return _sha(np.ascontiguousarray(M).tobytes())


def content_key(*, scene_sha: str, mode: str, tx_vox, bands, mechanisms,
                engine_ver: str, mech_bands=None) -> str:
    """The cache key. Every argument is something that changes the answer.

    Bands and mechanisms are normalized (sorted, rounded) so that `2442,3500` and
    `3500,2442` are one entry rather than two copies of the same solve.
    """
    return _sha(
        scene_sha, str(mode),
        tuple(int(v) for v in tx_vox),
        tuple(sorted(round(float(b), 3) for b in bands)),
        tuple(sorted(str(m) for m in mechanisms)),
        tuple(sorted(round(float(b), 3) for b in (mech_bands or []))),
        engine_ver,
    )


# --------------------------------------------------------------------------- index
class CacheIndex:
    """Read/modify/write `web/volumes/index.json`, the browser's volume catalog."""

    SCHEMA = "vol3d-v3"

    def __init__(self, directory=DEFAULT_DIR, manifest=None):
        self.dir = Path(directory)
        self.manifest = manifest or {}
        self.doc = {}
        self.entries: dict[str, dict] = {}      # txid -> entry
        self.load()

    # ---- persistence ------------------------------------------------------
    def load(self):
        p = self.dir / "index.json"
        self.doc, self.entries = {}, {}
        if not p.exists():
            return self
        try:
            self.doc = json.loads(p.read_text())
        except Exception as exc:                # noqa: BLE001
            print(f"warning: {p} is unreadable ({exc}); starting a fresh index")
            self.doc = {}
            return self
        for e in self.doc.get("volumes", self.doc.get("entries", [])):
            e = self.migrate(e)
            tid = e.get("txid")
            if tid:
                self.entries[tid] = e
        return self

    def save(self, mechanisms=None, bands=None, mode=None):
        self.dir.mkdir(parents=True, exist_ok=True)
        vols = sorted(self.entries.values(), key=lambda e: (e.get("mode", "indoor"),
                                                            e["txid"]))
        doc = {
            "version": self.SCHEMA,
            "modes": sorted({e.get("mode", "indoor") for e in vols}
                            | ({mode} if mode else set())),
            "grid_shape": self.manifest.get("grid_shape", self.doc.get("grid_shape")),
            "cell_size_m": self.manifest.get("cell_size_m", self.doc.get("cell_size_m")),
            "bands_mhz": list(bands) if bands else self.doc.get("bands_mhz"),
            "mechanisms": list(mechanisms) if mechanisms else self.doc.get("mechanisms"),
            "engine_ver": engine_version(),
            "pl_layout": "band,x,y,z  float16 dB",
            "t_layout": "x,y,z  float16 NANOSECONDS (sentinel 65504 = unreachable)",
            "mech_layout": "m_<mech>_<txid>.bin = band,x,y,z float16 dB; "
                           "tau_<mech>_<txid>.bin = x,y,z float16 ns",
            "count": len(vols),
            "bytes": int(sum(int(e.get("bytes") or 0) for e in vols)),
            "volumes": vols,
        }
        (self.dir / "index.json").write_text(json.dumps(doc, indent=1))
        self.doc = doc
        return doc

    # ---- schema -----------------------------------------------------------
    def migrate(self, e: dict) -> dict:
        """Bring an older entry up to schema instead of dropping it.

        v2 keyed entries `tx_id` and wrote no file names at all, which is why the browser
        could not load anything the batch precompute produced. Entries in that shape are
        upgraded on read.
        """
        e = dict(e)
        tid = e.get("txid") or e.get("tx_id")
        if not tid:
            return e
        e["txid"] = tid
        e.pop("tx_id", None)
        e.setdefault("mode", "indoor")           # everything pre-M3 was the indoor floor
        e.setdefault("pl_file", f"pl_volume_{tid}.bin")
        e.setdefault("t_file", f"t_volume_{tid}.bin")
        if "grid_shape" not in e and self.manifest.get("grid_shape"):
            e["grid_shape"] = list(self.manifest["grid_shape"])
        if "bands" not in e:
            e["bands"] = e.pop("bands_mhz", list(self.manifest.get("freqs_mhz", [])))
        if isinstance(e.get("mechanisms"), list):
            e["mechanism_names"] = e.pop("mechanisms")
        if "t_max_ns" not in e:
            tp = self.dir / e["t_file"]
            if tp.exists():
                t = np.fromfile(tp, dtype=np.float16).astype(np.float32)
                live = t[t < 65504.0 - 1]
                e["t_max_ns"] = round(float(live.max()), 2) if live.size else 0.0
            else:
                e["t_max_ns"] = 0.0
        e.setdefault("t_unreachable_ns", 65504.0)
        return e

    # ---- lookup -----------------------------------------------------------
    def by_key(self, key: str) -> dict | None:
        for e in self.entries.values():
            if e.get("key") == key:
                return e
        return None

    def is_fresh(self, key: str) -> bool:
        """True only if this exact solve is cached AND all its files are on disk."""
        e = self.by_key(key)
        return bool(e) and not self.missing_files(e)

    def by_mode(self, mode: str) -> list[dict]:
        return [e for e in self.entries.values() if e.get("mode", "indoor") == mode]

    def nearest(self, mode: str, tx_vox, tol=None) -> dict | None:
        pool = self.by_mode(mode)
        if not pool or tx_vox is None:
            return pool[0] if pool and tx_vox is None else None
        best, bd = None, float("inf")
        for e in pool:
            d = float(np.linalg.norm(np.asarray(e["tx_vox"], float)
                                     - np.asarray(tx_vox, float)))
            if d < bd:
                bd, best = d, e
        return best if (tol is None or bd <= tol) else None

    # ---- mutation ---------------------------------------------------------
    def put(self, entry: dict, *, key=None) -> dict:
        missing = [k for k in BROWSER_KEYS if k not in entry]
        if missing:
            raise ValueError(
                f"refusing to cache an entry the browser cannot read: missing {missing}")
        e = dict(entry)
        now = time.time()
        prev = self.entries.get(e["txid"], {})
        if key:
            e["key"] = key
        e.setdefault("scene_sha", None)
        e["engine_ver"] = e.get("engine_ver") or engine_version()
        e["created"] = prev.get("created", now)
        e["last_used"] = now
        e["hits"] = int(prev.get("hits", 0))
        e["bytes"] = int(e.get("bytes") or self.entry_bytes(e))
        self.entries[e["txid"]] = e
        return e

    def touch(self, txid: str):
        e = self.entries.get(txid)
        if e:
            e["last_used"] = time.time()
            e["hits"] = int(e.get("hits", 0)) + 1
        return e

    def files_of(self, e: dict) -> list[str]:
        out = [e.get("pl_file"), e.get("t_file")]
        m = e.get("mechanisms")
        if isinstance(m, dict):
            for info in m.values():
                out += [info.get("pl_file"), info.get("tau_file"),
                        info.get("tau_geom_file")]
        return [f for f in out if f]

    def entry_bytes(self, e: dict) -> int:
        return int(sum((self.dir / f).stat().st_size
                       for f in self.files_of(e) if (self.dir / f).exists()))

    def missing_files(self, e: dict) -> list[str]:
        return [f for f in self.files_of(e) if not (self.dir / f).exists()]

    def remove(self, txid: str, *, delete_files=True) -> int:
        e = self.entries.pop(txid, None)
        if not e:
            return 0
        freed = 0
        if delete_files:
            for f in self.files_of(e):
                p = self.dir / f
                if p.exists():
                    freed += p.stat().st_size
                    p.unlink()
        return freed

    # ---- maintenance ------------------------------------------------------
    def verify(self) -> list[dict]:
        """Every cached volume must be present and exactly the size its layout implies.

        A truncated .bin is the nastiest failure here: the JS decoder reads past the end,
        gets undefined, and paints a plausible-looking but wrong field. Better to catch it
        with arithmetic than with eyes.
        """
        problems = []
        for tid, e in sorted(self.entries.items()):
            shape = e.get("grid_shape")
            nvox = int(np.prod(shape)) if shape else None
            for f in self.files_of(e):
                p = self.dir / f
                if not p.exists():
                    problems.append({"txid": tid, "file": f, "problem": "missing"})
                    continue
                if not nvox:
                    continue
                if f.startswith("pl_volume_"):
                    want = len(e.get("bands", [])) * nvox * 2
                elif f.startswith("m_"):
                    want = len(e.get("mech_bands", e.get("bands", []))) * nvox * 2
                else:                                     # t_ / tau_ are single-band
                    want = nvox * 2
                got = p.stat().st_size
                if want and got != want:
                    problems.append({"txid": tid, "file": f, "problem": "wrong size",
                                     "bytes": got, "expected": want})
        return problems

    def stale(self, engine_ver=None) -> list[str]:
        """txids whose engine_ver no longer matches the current physics source."""
        cur = engine_ver or engine_version()
        return [t for t, e in self.entries.items()
                if e.get("engine_ver") and e["engine_ver"] != cur]

    def total_bytes(self) -> int:
        return int(sum(int(e.get("bytes") or 0) for e in self.entries.values()))

    def gc(self, max_bytes: int, *, protect=()) -> dict:
        """Evict whole transmitters, least-recently-used first, until under budget.

        Whole entries, never individual files: a half-present volume makes the browser
        fetch a 404 mid-render rather than fall through to the analytic tier, which is a
        worse failure than not having the volume at all.
        """
        protect = set(protect)
        freed, evicted = 0, []
        order = sorted(self.entries.values(),
                       key=lambda e: (e.get("last_used", 0), e.get("created", 0)))
        for e in order:
            if self.total_bytes() <= max_bytes:
                break
            if e["txid"] in protect:
                continue
            freed += self.remove(e["txid"])
            evicted.append(e["txid"])
        return {"evicted": evicted, "freed_bytes": freed,
                "remaining_bytes": self.total_bytes(), "budget_bytes": int(max_bytes)}

    def stats(self) -> dict:
        by_mode = {}
        for e in self.entries.values():
            by_mode.setdefault(e.get("mode", "indoor"), 0)
            by_mode[e.get("mode", "indoor")] += 1
        return {"count": len(self.entries), "bytes": self.total_bytes(),
                "by_mode": by_mode, "engine_ver": engine_version(),
                "stale": len(self.stale()), "dir": str(self.dir)}


# --------------------------------------------------------------------------- self-test
def _selftest() -> int:
    import shutil
    import tempfile

    ok, fails = 0, []

    def check(name, cond, msg=""):
        nonlocal ok
        if cond:
            ok += 1
            print(f"  PASS  {name}")
        else:
            fails.append(name)
            print(f"  FAIL  {name}  {msg}")

    base = dict(scene_sha="abc", mode="indoor", tx_vox=(1, 2, 3),
                bands=[2442.0, 3500.0], mechanisms=["path_loss", "reflection"],
                engine_ver="v1")
    k = content_key(**base)
    check("key is stable", k == content_key(**base))
    check("band order does not matter",
          k == content_key(**{**base, "bands": [3500.0, 2442.0]}))
    check("mechanism order does not matter",
          k == content_key(**{**base, "mechanisms": ["reflection", "path_loss"]}))
    for field, val in [("scene_sha", "def"), ("mode", "o2i"), ("tx_vox", (1, 2, 4)),
                       ("bands", [2442.0]), ("mechanisms", ["path_loss"]),
                       ("engine_ver", "v2")]:
        check(f"key changes with {field}", content_key(**{**base, field: val}) != k)

    ev = engine_version()
    check("engine_version is a 16-hex digest", len(ev) == 16 and int(ev, 16) >= 0)
    check("engine_version is deterministic", ev == engine_version())

    tmp = Path(tempfile.mkdtemp())
    try:
        shape = (4, 3, 5)
        nvox = int(np.prod(shape))

        def write(tid, nb=2, mech=True):
            (tmp / f"pl_volume_{tid}.bin").write_bytes(
                np.zeros(nb * nvox, np.float16).tobytes())
            (tmp / f"t_volume_{tid}.bin").write_bytes(
                np.zeros(nvox, np.float16).tobytes())
            mechs = {}
            if mech:
                (tmp / f"m_path_loss_{tid}.bin").write_bytes(
                    np.zeros(nb * nvox, np.float16).tobytes())
                (tmp / f"tau_path_loss_{tid}.bin").write_bytes(
                    np.zeros(nvox, np.float16).tobytes())
                mechs = {"path_loss": {"pl_file": f"m_path_loss_{tid}.bin",
                                       "tau_file": f"tau_path_loss_{tid}.bin"}}
            return {"txid": tid, "mode": "indoor", "tx_vox": [1, 1, 1],
                    "grid_shape": list(shape), "bands": [2442.0, 3500.0],
                    "mech_bands": [2442.0, 3500.0], "t_max_ns": 10.0,
                    "pl_file": f"pl_volume_{tid}.bin", "t_file": f"t_volume_{tid}.bin",
                    "mechanisms": mechs}

        ci = CacheIndex(tmp, manifest={"grid_shape": list(shape), "cell_size_m": 0.3})
        e1 = ci.put(write("tx_1-1-1"), key="k1")
        check("put computes size from the files on disk", e1["bytes"] > 0)
        check("is_fresh on a complete entry", ci.is_fresh("k1"))
        check("is_fresh is False for an unknown key", not ci.is_fresh("nope"))

        try:
            ci.put({"txid": "bad"})
            check("put rejects browser-unreadable entries", False, "no error raised")
        except ValueError:
            check("put rejects browser-unreadable entries", True)

        check("verify is clean", ci.verify() == [], str(ci.verify()))
        (tmp / "pl_volume_tx_1-1-1.bin").write_bytes(b"\x00" * 8)     # truncate
        v = ci.verify()
        check("verify catches truncation", any(p["problem"] == "wrong size" for p in v),
              str(v))
        (tmp / "pl_volume_tx_1-1-1.bin").write_bytes(
            np.zeros(2 * nvox, np.float16).tobytes())

        e2 = ci.put(write("tx_2-2-2"), key="k2")
        e2["tx_vox"] = [9, 9, 9]
        ci.entries["tx_1-1-1"]["last_used"] = 1.0                    # oldest
        ci.entries["tx_2-2-2"]["last_used"] = 2.0
        before = ci.total_bytes()
        r = ci.gc(max_bytes=before - 1)
        check("gc evicts the least recently used", r["evicted"] == ["tx_1-1-1"],
              str(r["evicted"]))
        check("gc deletes the evicted files",
              not (tmp / "pl_volume_tx_1-1-1.bin").exists())
        check("gc leaves the survivor whole", (tmp / "pl_volume_tx_2-2-2.bin").exists())

        ci.put(write("tx_3-3-3"), key="k3")
        ci.entries["tx_3-3-3"]["last_used"] = 0.0
        r2 = ci.gc(max_bytes=0, protect=["tx_3-3-3"])
        check("gc honours protect", "tx_3-3-3" not in r2["evicted"])

        doc = ci.save(mechanisms=["path_loss"], bands=[2442.0], mode="indoor")
        check("save writes the browser schema", doc["version"] == CacheIndex.SCHEMA)
        check("save records engine_ver", doc["engine_ver"] == engine_version())
        for e in doc["volumes"]:
            miss = [k for k in BROWSER_KEYS if k not in e]
            check(f"saved entry {e['txid']} is browser-readable", not miss, str(miss))

        ci2 = CacheIndex(tmp, manifest={"grid_shape": list(shape)})
        check("round-trips through disk", set(ci2.entries) == set(ci.entries))
        check("stats reports totals", ci2.stats()["count"] == len(ci.entries))

        # a pre-v3 entry must survive the migration, not vanish
        (tmp / "index.json").write_text(json.dumps({
            "version": "vol3d-v2",
            "volumes": [{"tx_id": "tx_7-7-7", "tx_vox": [7, 7, 7],
                         "mechanisms": ["path_loss"], "bands_mhz": [2442.0]}]}))
        (tmp / "pl_volume_tx_7-7-7.bin").write_bytes(
            np.zeros(nvox, np.float16).tobytes())
        (tmp / "t_volume_tx_7-7-7.bin").write_bytes(
            np.zeros(nvox, np.float16).tobytes())
        ci3 = CacheIndex(tmp, manifest={"grid_shape": list(shape),
                                        "freqs_mhz": [2442.0]})
        e = ci3.entries.get("tx_7-7-7")
        check("v2 entry is migrated, not dropped", e is not None)
        if e:
            check("migrated entry is browser-readable",
                  not [k for k in BROWSER_KEYS if k not in e])
            check("migrated entry defaults to indoor mode", e["mode"] == "indoor")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{ok} passed, {len(fails)} failed")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--gc", action="store_true", help="evict LRU until under --max-gb")
    ap.add_argument("--max-gb", type=float, default=2.0)
    ap.add_argument("--prune-stale", action="store_true",
                    help="drop entries the current physics source invalidates")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args(argv)

    if a.test:
        return _selftest()

    man_p = HERE / "manifest_3d.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    ci = CacheIndex(a.dir, manifest=man)

    if a.verify:
        probs = ci.verify()
        for p in probs:
            print(f"  {p['txid']:16s} {p['file']:34s} {p['problem']}"
                  + (f" ({p['bytes']} vs {p['expected']})" if "bytes" in p else ""))
        print(f"{len(probs)} problem(s) across {len(ci.entries)} entries")
        return 1 if probs else 0

    if a.prune_stale:
        st = ci.stale()
        print(f"engine_ver {engine_version()} — {len(st)} stale entr(ies): {st}")
        if st and not a.dry_run:
            freed = sum(ci.remove(t) for t in st)
            ci.save()
            print(f"removed, freed {freed/1e6:.1f} MB")
        return 0

    if a.gc:
        budget = int(a.max_gb * 1e9)
        if a.dry_run:
            total = ci.total_bytes()
            if total <= budget:
                print(f"{total/1e9:.2f} GB / {a.max_gb} GB — under budget, nothing to evict")
                return 0
            # replay the eviction order without touching disk
            order = sorted(ci.entries.values(), key=lambda e: (e.get("last_used", 0),
                                                               e.get("created", 0)))
            running, would = total, []
            for e in order:
                if running <= budget:
                    break
                running -= int(e.get("bytes") or 0)
                would.append(e["txid"])
            print(f"would evict {len(would)} of {len(ci.entries)} to go from "
                  f"{total/1e9:.2f} GB to {running/1e9:.2f} GB (budget {a.max_gb} GB), "
                  f"oldest first: {would}")
            return 0
        r = ci.gc(budget)
        ci.save()
        print(f"evicted {len(r['evicted'])} — freed {r['freed_bytes']/1e6:.1f} MB, "
              f"now {r['remaining_bytes']/1e9:.2f} GB / {a.max_gb} GB")
        return 0

    s = ci.stats()
    unkeyed = [t for t, e in ci.entries.items() if not e.get("key")]
    print(f"cache {s['dir']}")
    print(f"  entries    {s['count']}  ({s['bytes']/1e6:.1f} MB)")
    print(f"  by mode    {s['by_mode']}")
    print(f"  engine_ver {s['engine_ver']}   stale entries: {s['stale']}")
    if unkeyed:
        # Safe direction: no key means no proof of what produced it, so a re-run
        # recomputes rather than trusting it. Worth saying out loud.
        print(f"  unkeyed    {len(unkeyed)} entr(ies) predate content addressing — "
              f"still served to the browser, but a re-solve will not treat them as hits")
    for e in sorted(ci.entries.values(), key=lambda x: (x.get("mode", ""), x["txid"])):
        mech = e.get("mechanisms")
        nm = len(mech) if isinstance(mech, dict) else 0
        print(f"    {e.get('mode','indoor'):8s} {e['txid']:16s} "
              f"{(e.get('bytes') or 0)/1e6:6.1f} MB  {nm} mech channels  "
              f"hits={e.get('hits', 0)}  key={str(e.get('key'))[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
