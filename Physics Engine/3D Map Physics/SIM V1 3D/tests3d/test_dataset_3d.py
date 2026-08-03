"""
Surrogate featurization contract (M4).

The failure this suite exists to prevent is silent: `dataset_3d` defines the input
tensor that Phase B writes against, Phase C trains against, and the browser rebuilds
at inference time. A drift in channel order, blob sigma or the normalization window
does not raise anywhere — it renders a confident, wrong field. So the layout is
asserted here rather than trusted.

Also pins the two facts that invalidated the previous dataset:
  * `valid_tx_mask` cannot supply the 1,500 positions `splits.json` used to claim;
  * the committed splits must match the scene they will be trained on.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import dataset_3d as D


# --------------------------------------------------------------------------- layout
def test_channel_layout_is_nine_in_two_out():
    assert len(D.INPUT_CHANNELS) == 9
    assert D.INPUT_CHANNELS[:6] == D.STATIC_CHANNELS
    assert D.INPUT_CHANNELS[6:] == ("tx_blob", "freq_feat", "log_distance")
    assert D.OUTPUT_CHANNELS == ("pl_norm", "tau_norm")


def test_normalization_round_trips(manifest):
    norm = D.load_norm(manifest)
    pl = np.array([40.0, 77.5, 105.0, 170.0], np.float32)
    assert np.allclose(norm.norm_to_pl(norm.pl_to_norm(pl)), pl, atol=1e-4)
    # the clip is the point: the scene produces PL far above the window
    assert float(norm.pl_to_norm(np.array([1800.0]))[0]) == 1.0
    assert float(norm.pl_to_norm(np.array([0.0]))[0]) == 0.0
    tau = np.array([0.0, 120e-9, 260e-9], np.float64)
    assert np.allclose(norm.norm_to_tau_ns(norm.tau_to_norm(tau)), tau * 1e9, atol=1e-3)
    assert float(norm.tau_to_norm(np.array([np.inf]))[0]) == 1.0


def test_freq_feature_round_trips_every_band(manifest):
    norm = D.load_norm(manifest)
    for f in manifest["freqs_mhz"]:
        assert abs(norm.feature_to_freq_mhz(norm.freq_feature(f)) - f) < 1e-3
    # the widened window has to actually contain the measured cellular set
    assert norm.freq_log_lo_mhz <= min(manifest["freqs_mhz"])
    assert norm.freq_log_hi_mhz >= max(manifest["freqs_mhz"])
    assert 0.0 <= norm.freq_feature(min(manifest["freqs_mhz"])) <= 1.0
    assert 0.0 <= norm.freq_feature(max(manifest["freqs_mhz"])) <= 1.0


def test_input_tensor_semantics(grid, manifest):
    norm = D.load_norm(manifest)
    static = D.static_channels(grid, len(manifest["materials"]))
    coords = D.voxel_coords(grid.shape)
    tx = (60, 5, 63)
    x = D.make_input(static, tx, norm.freq_feature(3500.0), coords, manifest["cell_size_m"])

    assert x.shape == (9,) + grid.shape
    assert x.dtype == np.float32
    assert np.allclose(x[:6].sum(0), 1.0), "material one-hot must partition every voxel"
    assert int(np.argmax(x[6])) == int(np.ravel_multi_index(tx, grid.shape))
    assert float(x[7].max() - x[7].min()) == 0.0, "frequency channel is a constant plane"
    assert x[8][tx] < x[8][250, 5, 63], "log-distance must rise with range"


def test_fspl_reference_matches_the_engine(scene, grid, manifest):
    """The FSPL-floor loss and the engine must mean the same thing by 'free space'."""
    coords = D.voxel_coords(grid.shape)
    tx = (60, 5, 63)
    d = D.distance_m(tx, coords, manifest["cell_size_m"])
    mine = D.fspl_db(d, 3500.0, manifest)
    # SceneV3 path loss is FSPL + crossing loss, so free space is its lower bound
    theirs = scene.pathloss_maps(tuple(float(v) for v in tx))[scene.band_index(3500.0)]
    assert float((theirs - mine).min()) > -1e-2, "walls may only add loss"
    los = (61, 5, 63)
    assert abs(float(theirs[los] - mine[los])) < 1.0, "an adjacent voxel should be near free space"


# --------------------------------------------------------------------------- sampling
def test_valid_tx_mask_cannot_supply_1500_positions(sim3d_dir):
    """Why the previous splits.json was fiction.

    `valid_tx_mask` is `inside_mask` eroded for physical AP clearance. Min-spacing
    sampling over its 2,510 voxels saturates in the hundreds, so a file asking for
    1,500 positions was never generatable — and 1,486 of its entries did not land on
    a valid voxel at all.
    """
    valid = np.load(sim3d_dir / "valid_tx_mask.npy")
    assert D.max_positions(valid, 3.0) < 1500


def test_inside_mask_supplies_the_training_domain(inside_mask):
    assert D.max_positions(inside_mask, 3.0) >= 1500


def test_sampled_positions_are_interior_and_spaced(inside_mask):
    pos = D.sample_tx_positions(inside_mask, 150, 3.0, seed=0)
    assert len(pos) == 150
    assert inside_mask[pos[:, 0], pos[:, 1], pos[:, 2]].all()
    d = np.abs(pos[:, None, :].astype(int) - pos[None, :, :].astype(int)).max(-1)
    np.fill_diagonal(d, 99)
    assert d.min() >= 3, "min-spacing violated"


def test_sampling_is_seeded(inside_mask):
    a = D.sample_tx_positions(inside_mask, 40, 3.0, seed=7)
    b = D.sample_tx_positions(inside_mask, 40, 3.0, seed=7)
    c = D.sample_tx_positions(inside_mask, 40, 3.0, seed=8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_splits_partition_by_position(inside_mask, grid):
    pos = D.sample_tx_positions(inside_mask, 200, 3.0, seed=0)
    sp = D.make_splits(pos, grid.shape, seed=1)
    allidx = sorted(sp["train"] + sp["val"] + sp["test"])
    assert allidx == list(range(len(pos)))
    assert not set(sp["train"]) & set(sp["val"])
    assert not set(sp["val"]) & set(sp["test"])
    assert not set(sp["train"]) & set(sp["test"])
    assert len(sp["val"]) and len(sp["test"])


def test_committed_splits_match_this_scene(sim3d_dir, grid, inside_mask):
    """A dataset generated against a different geometry is silently wrong, not broken."""
    p = sim3d_dir / "dataset" / "splits.json"
    if not p.exists():
        pytest.skip("dataset/splits.json not generated")
    sp = json.loads(p.read_text())
    assert sp.get("scene_sha") == D.scene_sha(grid), "splits.json predates this voxelization"
    assert sp.get("grid_shape") == list(grid.shape)
    pos = np.array(sp["positions"], int)
    assert len(pos) == sp["n_positions"]
    assert inside_mask[pos[:, 0], pos[:, 1], pos[:, 2]].all()


# --------------------------------------------------------------------------- shards
def test_shard_round_trip_is_memmappable(tmp_path, manifest):
    """The volumes must come back as a memmap, not a RAM copy.

    An 8 GB dataset on a 12.7 GB Colab runtime only works if `np.load(mmap_mode='r')`
    can reach the array, which rules out compressed members.
    """
    n, shape = 4, (6, 5, 7)
    pl = np.random.default_rng(0).random((n,) + shape).astype(np.float16)
    tau = np.random.default_rng(1).random((2,) + shape).astype(np.float16)
    meta = dict(tx=np.zeros((n, 3), np.int16), freq_mhz=np.full(n, 3500.0, np.float32),
                freq_feat=np.full(n, 0.5, np.float32), pos_id=np.arange(n, dtype=np.int32),
                tau_row=np.array([0, 0, 1, 1], np.int32))

    assert not D.shard_complete(tmp_path, 0)
    D.write_shard(tmp_path, 0, pl, tau, meta)
    assert D.shard_complete(tmp_path, 0)
    assert len(D.list_shards(tmp_path)) == 1

    rpl, rtau, rmeta = D.open_shard(tmp_path, 0)
    assert isinstance(rpl, np.memmap) and isinstance(rtau, np.memmap)
    assert rpl.dtype == np.float16 and rpl.shape == (n,) + shape
    assert np.array_equal(np.asarray(rpl), pl)
    assert np.array_equal(np.asarray(rtau), tau)
    assert np.array_equal(rmeta["pos_id"], meta["pos_id"])


def test_no_partial_shard_survives_a_crash(tmp_path):
    """Files land under .part and are renamed last, so a killed cell leaves nothing
    that a resumed run would mistake for finished work."""
    D.write_shard(tmp_path, 3, np.zeros((1, 2, 2, 2), np.float16),
                  np.zeros((1, 2, 2, 2), np.float16), dict(pos_id=np.zeros(1, np.int32)))
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".part")]
    assert not leftovers, f"temp files not cleaned up: {leftovers}"


# ------------------------------------------------ shard_complete: the smoke-gap fix
_BANDS = [619.0, 1935.0, 2442.0, 3500.0, 5500.0, 6125.0]
_SHA = "deadbeefdeadbeef"


def _write(tmp_path, s, pos_ids, *, bands=_BANDS, sha=_SHA, shape=(2, 2, 2)):
    """Write a shard covering `pos_ids` at `bands`, exactly as phase_b3's loop does."""
    pos_ids = list(pos_ids)
    npos, nb = len(pos_ids), len(bands)
    pl = np.zeros((npos * nb,) + shape, np.float16)
    tau = np.zeros((npos,) + shape, np.float16)
    meta = dict(
        tx=np.zeros((npos * nb, 3), np.int16),
        freq_mhz=np.array(bands * npos, np.float32),
        freq_feat=np.zeros(npos * nb, np.float32),
        pos_id=np.repeat(pos_ids, nb).astype(np.int32),
        tau_row=np.repeat(np.arange(npos), nb).astype(np.int32),
        scene_sha=sha, bands_mhz=np.array(bands, np.float32))
    D.write_shard(tmp_path, s, pl, tau, meta)


def test_shard_complete_rejects_smoke_leftover(tmp_path):
    """A SHARD_POS=8 smoke shard must not satisfy a SHARD_POS=25 full run.

    Reproduces the gap that shipped in the 716/750 dataset: a smoke run wrote
    8-position shards into slots 0-1, and the old files-exist shard_complete let the
    full run skip them, so positions 16-49 were never generated. The strengthened
    check catches every way a shard can fail to match the current run.
    """
    _write(tmp_path, 0, range(0, 8))                       # smoke: 8 positions

    # old behaviour preserved: with no expectations, files-exist == complete
    assert D.shard_complete(tmp_path, 0) is True

    # full run expects 25 positions here -> incomplete -> the loop regenerates it
    assert D.shard_complete(tmp_path, 0, expect_pos=25, bands=_BANDS, scene_sha=_SHA) is False

    _write(tmp_path, 2, range(50, 75))                     # a genuine 25-position shard
    assert D.shard_complete(tmp_path, 2, expect_pos=25, bands=_BANDS, scene_sha=_SHA) is True
    # a different scene or band set also invalidates a present shard
    assert D.shard_complete(tmp_path, 2, expect_pos=25, bands=_BANDS, scene_sha="0" * 16) is False
    assert D.shard_complete(tmp_path, 2, expect_pos=25, bands=_BANDS[:5], scene_sha=_SHA) is False


def test_audit_shards_finds_then_clears_the_position_gap(tmp_path):
    """audit_shards reports the smoke gap, and regenerating the slots clears it."""
    # smoke left slots 0-1 (8 pos each: 0-7, 8-15); the full run filled 2-5 (25 each)
    _write(tmp_path, 0, range(0, 8))
    _write(tmp_path, 1, range(8, 16))
    for s in range(2, 6):
        _write(tmp_path, s, range(s * 25, s * 25 + 25))

    ok, rows = D.audit_shards(tmp_path, shard_pos=25, n_positions=150,
                              bands=_BANDS, scene_sha=_SHA)
    assert ok is False
    by_shard = {r["shard"]: r for r in rows}
    assert by_shard[0]["status"] == "mismatch" and by_shard[0]["got"] == 8
    assert by_shard[1]["status"] == "mismatch"
    assert by_shard[2]["status"] == "ok"

    # regenerate the two contaminated slots at the full SHARD_POS -> gap closes
    _write(tmp_path, 0, range(0, 25))
    _write(tmp_path, 1, range(25, 50))
    ok2, rows2 = D.audit_shards(tmp_path, shard_pos=25, n_positions=150,
                                bands=_BANDS, scene_sha=_SHA)
    assert ok2 is True, [r for r in rows2 if r["status"] != "ok"]


# --------------------------------------------------------------------------- contract
def test_surrogate_contract_is_what_the_browser_reads(grid, manifest):
    c = D.surrogate_contract(manifest, grid, bands=[2442.0, 3500.0],
                             mechanisms=["path_loss"], mode="indoor")
    assert c["spec_version"] == D.SPEC_VERSION
    assert c["model_file"] == "pl_unet3d.onnx"
    assert c["grid_shape"] == list(grid.shape)
    assert c["scene_sha"] == D.scene_sha(grid)
    assert c["input"]["channels"] == list(D.INPUT_CHANNELS)
    assert c["output"]["channels"] == list(D.OUTPUT_CHANNELS)
    # every constant the browser needs to rebuild the input must be present
    for k in ("tx_blob_sigma_cells", "logdist_divisor", "logdist_min_m",
              "freq_log_lo_mhz", "freq_log_hi_mhz", "material_classes"):
        assert k in c["input"], f"contract is missing {k}"
    assert c["input"]["tx_blob_sigma_cells"] == D.TX_SIGMA_CELLS
    assert c["input"]["logdist_divisor"] == D.LOGDIST_DIVISOR
    assert json.loads(json.dumps(c)) == c, "contract must be JSON-serializable as-is"


def test_contract_writes_and_reloads(tmp_path, grid, manifest):
    c = D.surrogate_contract(manifest, grid, bands=[3500.0], mechanisms=["path_loss"],
                             mode="indoor", metrics={"rmse_pl_db": 4.2})
    p = tmp_path / "pl_unet3d.json"
    D.write_surrogate_contract(p, c)
    assert json.loads(p.read_text())["metrics"]["rmse_pl_db"] == 4.2


# --------------------------------------------------------------------------- the gate
@pytest.mark.slow
def test_clip_report_under_norm_ceiling(scene, manifest, inside_mask):
    """Pre-M4 saturating-obstruction fix landed (M2 satObs) — targets stay in-window.

    Before satObs, direct PL ran past 1,700 dB and ~80% of interior voxels pinned to the
    170 dB norm ceiling. SceneV3.pathloss_maps now applies sat_obs to the direct crossing
    loss, so Phase B's clip_report preflight (limit 0.35) passes and dataset generation
    is unblocked. This test guards the fix: if it regresses, flip back to the old gate.
    """
    norm = D.load_norm(manifest)
    rep = D.clip_report(scene, manifest, inside_mask, norm, n_probe=1,
                        bands=[619.0, 3500.0], seed=0)
    assert set(rep["clipped_fraction"]) == {619.0, 3500.0}
    assert 0.0 <= rep["worst_clipped_fraction"] <= 1.0
    assert rep["clipped_fraction"][3500.0] >= rep["clipped_fraction"][619.0], \
        "higher band must clip at least as much"
    assert rep["worst_clipped_fraction"] < 0.35, (
        "saturating-obstruction regress — Phase B preflight would refuse to generate; "
        "check SceneV3.use_satobs / sat_obs")
