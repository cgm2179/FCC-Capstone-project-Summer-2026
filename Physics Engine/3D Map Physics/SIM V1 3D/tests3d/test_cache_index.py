"""
M3 — the offload/cache layer.

The failure this file exists to prevent is a SILENT one. Before content addressing,
`precompute_volumes.py` resumed by asking "does pl_volume_<tid>.bin exist?", so changing
--mechanisms, widening --bands, or editing a physics module left the old file in place,
the run skipped it, and you kept a stale volume while believing you had recomputed. No
error, no warning, just wrong numbers that look exactly like right ones.

So the tests here are mostly about INVALIDATION: for every input that changes the answer,
the key must change. A test that only checked "the same inputs give the same key" would
pass on a key that ignored half its arguments.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import cache_index as CI
from cache_index import CacheIndex, content_key, engine_version, scene_sha

SHAPE = (4, 3, 5)
NVOX = int(np.prod(SHAPE))
BASE = dict(scene_sha="sceneA", mode="indoor", tx_vox=(1, 2, 3),
            bands=[2442.0, 3500.0], mechanisms=["path_loss", "reflection"],
            engine_ver="engA", mech_bands=[2442.0])


def _write_volume(d, tid, *, nb=2, mech=True, mech_nb=2):
    (d / f"pl_volume_{tid}.bin").write_bytes(np.zeros(nb * NVOX, np.float16).tobytes())
    (d / f"t_volume_{tid}.bin").write_bytes(np.zeros(NVOX, np.float16).tobytes())
    mechs = {}
    if mech:
        (d / f"m_path_loss_{tid}.bin").write_bytes(
            np.zeros(mech_nb * NVOX, np.float16).tobytes())
        (d / f"tau_path_loss_{tid}.bin").write_bytes(
            np.zeros(NVOX, np.float16).tobytes())
        mechs = {"path_loss": {"pl_file": f"m_path_loss_{tid}.bin",
                               "tau_file": f"tau_path_loss_{tid}.bin"}}
    return {"txid": tid, "mode": "indoor", "tx_vox": [1, 1, 1],
            "grid_shape": list(SHAPE), "bands": [2442.0, 3500.0],
            "mech_bands": [2442.0, 3500.0][:mech_nb], "t_max_ns": 10.0,
            "pl_file": f"pl_volume_{tid}.bin", "t_file": f"t_volume_{tid}.bin",
            "mechanisms": mechs}


@pytest.fixture
def cache(tmp_path):
    return CacheIndex(tmp_path, manifest={"grid_shape": list(SHAPE), "cell_size_m": 0.3,
                                          "freqs_mhz": [2442.0, 3500.0]})


# --------------------------------------------------------------------------- the key
def test_key_is_stable_across_calls():
    assert content_key(**BASE) == content_key(**BASE)


@pytest.mark.parametrize("field,value", [
    ("scene_sha", "sceneB"),
    ("mode", "o2i"),
    ("tx_vox", (1, 2, 4)),
    ("bands", [2442.0]),
    ("mechanisms", ["path_loss"]),
    ("engine_ver", "engB"),
    ("mech_bands", [3500.0]),
])
def test_key_changes_for_every_input_that_changes_the_answer(field, value):
    """The whole point. A key that ignored one of these would resume from a stale volume."""
    assert content_key(**{**BASE, field: value}) != content_key(**BASE), \
        f"changing {field} did not change the key — stale volumes would be reused"


@pytest.mark.parametrize("field,reordered", [
    ("bands", [3500.0, 2442.0]),
    ("mechanisms", ["reflection", "path_loss"]),
])
def test_key_ignores_argument_order(field, reordered):
    """`--bands 2442,3500` and `--bands 3500,2442` are one solve, not two."""
    assert content_key(**{**BASE, field: reordered}) == content_key(**BASE)


def test_engine_version_tracks_the_physics_source(tmp_path):
    """Editing a solver must invalidate its volumes with nobody remembering to say so."""
    a = tmp_path / "phys_a.py"
    a.write_text("SPEED = 1\n")
    v1 = engine_version([a])
    a.write_text("SPEED = 2\n")
    v2 = engine_version([a])
    assert v1 != v2, "a physics edit did not change engine_version"
    assert len(v1) == 16 and int(v1, 16) >= 0


def test_engine_version_is_deterministic():
    assert engine_version() == engine_version()


def test_scene_sha_follows_the_grid():
    a = np.zeros(SHAPE, np.int8)
    b = a.copy()
    b[0, 0, 0] = 3
    assert scene_sha(a) == scene_sha(a.copy())
    assert scene_sha(a) != scene_sha(b), "a re-voxelized scene must invalidate its volumes"


def test_real_physics_sources_exist():
    """A missing source hashes as '<missing>', which would quietly weaken invalidation."""
    absent = [p.name for p in CI.PHYSICS_SOURCES if not p.exists()]
    assert not absent, f"cache_index tracks files that do not exist: {absent}"


# --------------------------------------------------------------------------- index
def test_put_refuses_entries_the_browser_cannot_read(cache):
    """Missing pl_file/grid_shape is a TypeError in loadVolume, not a graceful fallback."""
    with pytest.raises(ValueError, match="browser cannot read"):
        cache.put({"txid": "tx_0-0-0"})


def test_saved_entries_keep_every_browser_key(cache, tmp_path):
    cache.put(_write_volume(tmp_path, "tx_1-1-1"), key="k1")
    doc = cache.save(mechanisms=["path_loss"], bands=[2442.0], mode="indoor")
    for e in doc["volumes"]:
        missing = [k for k in CI.BROWSER_KEYS if k not in e]
        assert not missing, f"{e['txid']} would break simulation3d.js: missing {missing}"


def test_is_fresh_requires_the_files_to_be_present(cache, tmp_path):
    cache.put(_write_volume(tmp_path, "tx_1-1-1"), key="k1")
    assert cache.is_fresh("k1")
    (tmp_path / "pl_volume_tx_1-1-1.bin").unlink()
    assert not cache.is_fresh("k1"), \
        "a key match with a missing file must not count as cached"


def test_round_trips_through_disk(cache, tmp_path):
    cache.put(_write_volume(tmp_path, "tx_1-1-1"), key="k1")
    cache.save()
    reloaded = CacheIndex(tmp_path, manifest={"grid_shape": list(SHAPE)})
    assert set(reloaded.entries) == {"tx_1-1-1"}
    assert reloaded.by_key("k1") is not None


def test_a_corrupt_index_does_not_take_the_files_with_it(tmp_path):
    (tmp_path / "index.json").write_text("{not json")
    ci = CacheIndex(tmp_path, manifest={"grid_shape": list(SHAPE)})
    assert ci.entries == {}          # starts fresh rather than raising


# --------------------------------------------------------------------------- verify
def test_verify_is_clean_on_a_good_cache(cache, tmp_path):
    cache.put(_write_volume(tmp_path, "tx_1-1-1"), key="k1")
    assert cache.verify() == []


def test_verify_catches_truncation(cache, tmp_path):
    """A short .bin makes the JS decoder read undefined and paint a plausible wrong field."""
    cache.put(_write_volume(tmp_path, "tx_1-1-1"), key="k1")
    (tmp_path / "pl_volume_tx_1-1-1.bin").write_bytes(b"\x00" * 8)
    probs = cache.verify()
    assert any(p["problem"] == "wrong size" for p in probs), probs


def test_verify_catches_a_missing_file(cache, tmp_path):
    cache.put(_write_volume(tmp_path, "tx_1-1-1"), key="k1")
    (tmp_path / "tau_path_loss_tx_1-1-1.bin").unlink()
    assert any(p["problem"] == "missing" for p in cache.verify())


# --------------------------------------------------------------------------- LRU
def test_gc_evicts_least_recently_used_first(cache, tmp_path):
    cache.put(_write_volume(tmp_path, "tx_1-1-1"), key="k1")
    cache.put(_write_volume(tmp_path, "tx_2-2-2"), key="k2")
    cache.entries["tx_1-1-1"]["last_used"] = 1.0
    cache.entries["tx_2-2-2"]["last_used"] = 2.0
    r = cache.gc(max_bytes=cache.total_bytes() - 1)
    assert r["evicted"] == ["tx_1-1-1"]
    assert r["freed_bytes"] > 0


def test_gc_deletes_whole_entries_not_stray_files(cache, tmp_path):
    """Half a volume is worse than none: the browser 404s mid-render instead of
    falling through to the analytic tier."""
    cache.put(_write_volume(tmp_path, "tx_1-1-1"), key="k1")
    cache.put(_write_volume(tmp_path, "tx_2-2-2"), key="k2")
    cache.entries["tx_1-1-1"]["last_used"] = 1.0
    cache.entries["tx_2-2-2"]["last_used"] = 2.0
    cache.gc(max_bytes=cache.total_bytes() - 1)
    for f in ("pl_volume", "t_volume", "m_path_loss", "tau_path_loss"):
        assert not (tmp_path / f"{f}_tx_1-1-1.bin").exists(), f"{f} left behind"
        assert (tmp_path / f"{f}_tx_2-2-2.bin").exists(), f"{f} wrongly evicted"


def test_gc_honours_protect(cache, tmp_path):
    cache.put(_write_volume(tmp_path, "tx_1-1-1"), key="k1")
    cache.entries["tx_1-1-1"]["last_used"] = 0.0
    r = cache.gc(max_bytes=0, protect=["tx_1-1-1"])
    assert r["evicted"] == []
    assert (tmp_path / "pl_volume_tx_1-1-1.bin").exists()


def test_touch_updates_lru_and_hit_count(cache, tmp_path):
    cache.put(_write_volume(tmp_path, "tx_1-1-1"), key="k1")
    before = cache.entries["tx_1-1-1"]["last_used"]
    cache.entries["tx_1-1-1"]["last_used"] = 0.0
    cache.touch("tx_1-1-1")
    assert cache.entries["tx_1-1-1"]["last_used"] > 0.0
    assert cache.entries["tx_1-1-1"]["hits"] == 1
    assert before > 0


def test_stale_reports_entries_from_a_different_engine(cache, tmp_path):
    cache.put(_write_volume(tmp_path, "tx_1-1-1"), key="k1")
    cache.entries["tx_1-1-1"]["engine_ver"] = "somethingelse"
    assert cache.stale() == ["tx_1-1-1"]
    cache.entries["tx_1-1-1"]["engine_ver"] = engine_version()
    assert cache.stale() == []


# --------------------------------------------------------------------------- lookup
def test_nearest_respects_the_mode_partition(cache, tmp_path):
    a = _write_volume(tmp_path, "tx_1-1-1"); a["tx_vox"] = [10, 5, 10]
    b = _write_volume(tmp_path, "tx_2-2-2"); b["tx_vox"] = [11, 5, 10]; b["mode"] = "o2i"
    cache.put(a, key="k1")
    cache.put(b, key="k2")
    hit = cache.nearest("indoor", [10, 5, 10], tol=5)
    assert hit["txid"] == "tx_1-1-1", "a volume from another mode must never match"
    assert cache.nearest("outdoor", [10, 5, 10], tol=5) is None


def test_nearest_enforces_the_tolerance(cache, tmp_path):
    e = _write_volume(tmp_path, "tx_1-1-1"); e["tx_vox"] = [0, 0, 0]
    cache.put(e, key="k1")
    assert cache.nearest("indoor", [100, 0, 0], tol=20) is None
    assert cache.nearest("indoor", [5, 0, 0], tol=20)["txid"] == "tx_1-1-1"


# --------------------------------------------------------------------------- migration
def test_pre_v3_entries_are_migrated_not_dropped(tmp_path):
    """v2 keyed on `tx_id` and wrote no file names — the bug that made the A100 cache
    invisible to the browser."""
    tid = "tx_7-7-7"
    _write_volume(tmp_path, tid, mech=False)
    (tmp_path / "index.json").write_text(json.dumps({
        "version": "vol3d-v2",
        "volumes": [{"tx_id": tid, "tx_vox": [7, 7, 7], "mechanisms": ["path_loss"],
                     "bands_mhz": [2442.0, 3500.0], "median_pl_db": 100.0}]}))
    ci = CacheIndex(tmp_path, manifest={"grid_shape": list(SHAPE),
                                        "freqs_mhz": [2442.0, 3500.0]})
    e = ci.entries.get(tid)
    assert e is not None, "a pre-v3 entry was dropped"
    assert not [k for k in CI.BROWSER_KEYS if k not in e]
    assert e["mode"] == "indoor"
    assert e["t_max_ns"] >= 0.0


def test_the_live_cache_is_readable_and_verifies(sim3d_dir, manifest):
    """The real web/volumes must stay loadable by the browser at all times."""
    d = sim3d_dir / "web" / "volumes"
    if not (d / "index.json").exists():
        pytest.skip("no volume cache built")
    ci = CacheIndex(d, manifest=manifest)
    assert ci.entries, "index.json has no volumes"
    for tid, e in ci.entries.items():
        missing = [k for k in CI.BROWSER_KEYS if k not in e]
        assert not missing, f"{tid} is unreadable by simulation3d.js: missing {missing}"
    assert ci.verify() == [], f"cache is corrupt: {ci.verify()}"
