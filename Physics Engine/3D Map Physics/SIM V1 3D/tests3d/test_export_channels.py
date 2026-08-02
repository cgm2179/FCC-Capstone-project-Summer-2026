"""
The export contract between the physics and the browser.

Everything the 3-D Simulation tab draws comes out of `web/volumes/`, so the failure mode
this file guards against is not "wrong physics" but "correct physics the browser cannot
read": a layout the JS decoder addresses differently, an index entry missing the keys
`simulation3d.js` looks up, or a mechanism whose arrival-time channel is empty so its
front never appears in the time-lapse. That last one is not hypothetical — Scattering_3D
shipped with `tau_first` left at inf.

Reference for the addressing asserted here (simulation3d.js):
    plAt/chanAt   idx = ((band * NX + x) * NY + y) * NZ + z      C-order (band,x,y,z)
    tAt/tauAt     idx = (x * NY + y) * NZ + z                    C-order (x,y,z), ns
    sentinel      65504.0 (float16 max) = unreachable
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import contracts
import engine_3d
import precompute_volumes as PV
import synth3d as S

F16_MAX = 65504.0
MECHS = ["path_loss", "reflection", "diffraction", "scattering"]

# Keys simulation3d.js reads off an index entry. loadVolume() dereferences pl_file/t_file
# and grid_shape directly, so a missing one is a TypeError in the browser, not a fallback.
BROWSER_KEYS = ("txid", "tx_vox", "grid_shape", "bands", "t_max_ns", "pl_file", "t_file")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """A small but real export: solve, write channels, merge an index."""
    M, man = S.two_plate_scene_3d(n=(30, 14, 22), gap_cells=8, mat=2)
    man = dict(man)
    man.setdefault("grid_shape", list(M.shape))
    man.setdefault("cell_size_m", man.get("cell_size_m", 0.3))
    sc = engine_3d.SceneV3(M, man)
    sc.inside = (M == 0)
    tx = (4, 7, 11)
    bands = [float(f) for f in man["freqs_mhz"][:2]]
    mechs = ["path_loss", "reflection", "scattering"]

    out = tmp_path_factory.mktemp("volumes")
    pl, t, cf, contribs = PV.solve_one(sc, tx, bands, mechs, "numpy")
    txid, nbytes, tmax = PV.write_volumes(out, tx, pl, t)
    mech_files, _mb = PV.write_mechanism_channels(out, tx, cf, contribs, scene=sc)
    entry = {"txid": txid, "tx_vox": list(tx), "grid_shape": list(M.shape), "bands": bands,
             "t_max_ns": tmax, "t_unreachable_ns": F16_MAX,
             "pl_file": f"pl_volume_{txid}.bin", "t_file": f"t_volume_{txid}.bin",
             "mechanisms": mech_files, "mech_bands": bands}
    doc = PV.merge_index(out, [entry], man, mechs, bands)
    return dict(dir=out, doc=doc, entry=entry, cf=cf, contribs=contribs, scene=sc,
                tx=tx, bands=bands, shape=M.shape, pl=pl, t=t)


# --------------------------------------------------------------------------- index
def test_index_entry_has_every_key_the_browser_reads(exported):
    e = exported["doc"]["volumes"][0]
    missing = [k for k in BROWSER_KEYS if k not in e]
    assert not missing, f"simulation3d.js would fail on a volume missing {missing}"
    assert e["pl_file"].endswith(".bin") and e["t_file"].endswith(".bin")


def test_index_migrates_pre_v3_entries_instead_of_dropping_them(exported, tmp_path):
    """v2 wrote `tx_id` and no file names; merge_index must upgrade, not discard.

    This is the bug that made the A100-computed multi-Tx cache invisible to the browser.
    """
    out = tmp_path / "vols"
    out.mkdir()
    src = exported["dir"]
    old_id = "tx_9-9-9"
    for kind in ("pl_volume", "t_volume"):
        (out / f"{kind}_{old_id}.bin").write_bytes(
            (src / f"{kind}_{exported['entry']['txid']}.bin").read_bytes())
    (out / "index.json").write_text(json.dumps({
        "version": "vol3d-v2",
        "volumes": [{"tx_id": old_id, "tx_vox": [9, 9, 9], "mechanisms": ["path_loss"],
                     "bands_mhz": exported["bands"], "median_pl_db": 100.0}]}))

    man = {"grid_shape": list(exported["shape"]), "cell_size_m": 0.3,
           "freqs_mhz": exported["bands"]}
    doc = PV.merge_index(out, [], man, ["path_loss"], exported["bands"])

    kept = [v for v in doc["volumes"] if v.get("txid") == old_id]
    assert kept, "a pre-v3 entry was dropped on merge"
    e = kept[0]
    assert all(k in e for k in BROWSER_KEYS), "migrated entry is still unreadable"
    assert e["t_max_ns"] > 0, "t_max_ns must be recovered from the T volume — the sweep " \
                              "clock ends at 0 without it"


# --------------------------------------------------------------------------- layout
def test_pl_volume_layout_matches_the_js_decoder(exported):
    """Round-trip through the exact index arithmetic plAt() uses."""
    e, shape = exported["entry"], exported["shape"]
    NX, NY, NZ = shape
    nb = len(exported["bands"])
    raw = np.fromfile(exported["dir"] / e["pl_file"], dtype=np.float16).astype(np.float32)
    assert raw.size == nb * NX * NY * NZ

    ref = np.asarray(exported["pl"], np.float32)
    rng = np.random.default_rng(0)
    for _ in range(64):
        b = int(rng.integers(nb))
        x, y, z = (int(rng.integers(n)) for n in shape)
        js = raw[((b * NX + x) * NY + y) * NZ + z]
        assert np.isclose(js, ref[b, x, y, z], atol=0.5), \
            f"band {b} voxel {(x, y, z)}: browser reads {js}, engine says {ref[b, x, y, z]}"


def test_t_volume_is_nanoseconds_with_the_sentinel(exported):
    e, shape = exported["entry"], exported["shape"]
    NX, NY, NZ = shape
    raw = np.fromfile(exported["dir"] / e["t_file"], dtype=np.float16).astype(np.float32)
    assert raw.size == NX * NY * NZ
    live = raw[raw < F16_MAX - 1]
    assert live.size, "no reachable voxels in the T volume"
    ref = np.asarray(exported["t"], np.float64) * 1e9
    fin = np.isfinite(ref.ravel())
    assert np.allclose(raw[fin], ref.ravel()[fin], atol=0.5), "T is not in nanoseconds"
    assert e["t_max_ns"] == pytest.approx(float(live.max()), abs=0.01)


# --------------------------------------------------------------------------- channels
def test_mechanism_channel_reproduces_the_per_mechanism_power(exported):
    """m_<mech>.bin must equal -10log10(per_mech_lin) to float16 precision.

    The browser derives the contribution share from this against the total volume, so a
    drift here is a wrong percentage on screen, not a crash.
    """
    cf, e = exported["cf"], exported["entry"]
    shape, nb = exported["shape"], len(exported["bands"])
    for mech, info in e["mechanisms"].items():
        raw = np.fromfile(exported["dir"] / info["pl_file"],
                          dtype=np.float16).astype(np.float32).reshape((nb,) + shape)
        with np.errstate(divide="ignore"):
            ref = -10.0 * np.log10(np.maximum(np.asarray(cf.per_mech_lin[mech], np.float64),
                                              1e-300))
        live = np.isfinite(ref) & (ref < 400)
        err = float(np.max(np.abs(raw[live] - ref[live]))) if live.any() else 0.0
        assert err < 0.1, f"{mech}: channel drifted {err:.3f} dB from per_mech_lin"


def test_contribution_share_is_recoverable_in_the_browser(exported):
    """share = 10**((PL_total - PL_mech)/10) recovers the mechanism's power ratio.

    That identity is the whole reason the channels store dB rather than a baked-in
    percentage — the browser computes the share itself, against the total it already has.

    The denominator is the STORED total, which carries the FSPL floor. Where the floor
    binds (constructive interference beating free space) the stored total is deliberately
    not -10log10(P_lin), so the browser's share is "share of the level actually displayed"
    — which is the honest reading of the on-screen number, and is asserted separately from
    the unfloored physics ratio below.
    """
    cf, e = exported["cf"], exported["entry"]
    shape, nb = exported["shape"], len(exported["bands"])
    total = np.fromfile(exported["dir"] / e["pl_file"],
                        dtype=np.float16).astype(np.float32).reshape((nb,) + shape)
    p_stored = np.power(10.0, -np.asarray(cf.PL_db, np.float64) / 10.0)
    for mech, info in e["mechanisms"].items():
        mch = np.fromfile(exported["dir"] / info["pl_file"],
                          dtype=np.float16).astype(np.float32).reshape((nb,) + shape)
        share_js = np.power(10.0, (total - mch) / 10.0)
        share_ref = np.asarray(cf.per_mech_lin[mech], np.float64) / np.maximum(p_stored, 1e-300)
        m = np.isfinite(share_js) & (share_ref > 1e-6) & (mch < 400) & (total < 400)
        assert m.any(), f"{mech}: no voxel where the share is testable"
        rel = np.abs(share_js[m] - share_ref[m]) / np.maximum(share_ref[m], 1e-9)
        assert float(np.median(rel)) < 0.05, \
            f"{mech}: browser-side share differs from the engine by {np.median(rel):.1%}"


def test_share_equals_the_physics_ratio_where_the_fspl_floor_does_not_bind(exported):
    """Off the floor, the displayed share IS cf.contribution_fraction()."""
    cf, e = exported["cf"], exported["entry"]
    shape, nb = exported["shape"], len(exported["bands"])
    total = np.fromfile(exported["dir"] / e["pl_file"],
                        dtype=np.float16).astype(np.float32).reshape((nb,) + shape)
    with np.errstate(divide="ignore"):
        pl_raw = -10.0 * np.log10(np.maximum(np.asarray(cf.P_lin, np.float64), 1e-300))
    unfloored = np.abs(np.asarray(cf.PL_db, np.float64) - pl_raw) < 0.01
    for mech, info in e["mechanisms"].items():
        mch = np.fromfile(exported["dir"] / info["pl_file"],
                          dtype=np.float16).astype(np.float32).reshape((nb,) + shape)
        share_js = np.power(10.0, (total - mch) / 10.0)
        share_py = cf.contribution_fraction(mech)
        m = unfloored & np.isfinite(share_js) & (share_py > 1e-6) & (mch < 400) & (total < 400)
        if not m.any():
            continue
        rel = np.abs(share_js[m] - share_py[m]) / np.maximum(share_py[m], 1e-9)
        assert float(np.median(rel)) < 0.05, \
            f"{mech}: off-floor share drifted {np.median(rel):.1%} from the physics ratio"


def test_every_mechanism_has_a_usable_arrival_channel(exported):
    """A mechanism whose tau is all-sentinel never appears in the time-lapse."""
    e = exported["entry"]
    shape = exported["shape"]
    for mech, info in e["mechanisms"].items():
        tau = np.fromfile(exported["dir"] / info["tau_file"],
                          dtype=np.float16).astype(np.float32)
        assert tau.size == int(np.prod(shape)), f"{mech}: tau channel is the wrong size"
        live = tau[tau < F16_MAX - 1]
        assert live.size, (f"{mech}: arrival-time channel is entirely the unreachable "
                           f"sentinel — its front would never appear in the time-lapse")
        assert info["t_max_ns"] > 0


def test_mechanism_arrivals_are_causal(exported):
    """No mechanism may arrive before the straight-line time from the transmitter."""
    e, sc, tx = exported["entry"], exported["scene"], exported["tx"]
    shape = exported["shape"]
    gx, gy, gz = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    d = np.sqrt((gx - tx[0]) ** 2 + (gy - tx[1]) ** 2 + (gz - tx[2]) ** 2) * sc.cell
    t_direct_ns = (d / contracts.C0) * 1e9
    for mech, info in e["mechanisms"].items():
        tau = np.fromfile(exported["dir"] / info["tau_file"],
                          dtype=np.float16).astype(np.float32).reshape(shape)
        live = tau < F16_MAX - 1
        # one cell of slack for float16 and for the diffuse channel's coarse lattice
        slack = (sc.cell * 2.0 / contracts.C0) * 1e9 + 0.5
        worst = float(np.min(tau[live] - t_direct_ns[live]))
        assert worst >= -slack, f"{mech}: arrives {-worst:.2f} ns before the direct path"


def test_direct_path_carries_both_clocks(exported):
    """path_loss must ship the geometric arrival alongside the eikonal one.

    Without it the time-lapse compares an eikonal direct front (charged for in-wall
    slowdown) against three vacuum-speed fronts, and the direct field lands LAST in ~99%
    of interior voxels — a sequence that reads as physically backwards.
    """
    info = exported["entry"]["mechanisms"]["path_loss"]
    assert info.get("tau_convention") == "eikonal"
    assert "tau_geom_file" in info, "no geometric clock — the time-lapse order is not comparable"
    shape = exported["shape"]
    geom = np.fromfile(exported["dir"] / info["tau_geom_file"],
                       dtype=np.float16).astype(np.float32).reshape(shape)
    eik = np.fromfile(exported["dir"] / info["tau_file"],
                      dtype=np.float16).astype(np.float32).reshape(shape)
    m = (geom < F16_MAX - 1) & (eik < F16_MAX - 1)
    assert m.any()
    # The eikonal can only be slower: it travels the same geometry at <= c. The tolerance
    # is the fast-marching discretization, not physics slack — skfmm puts the zero contour
    # on the source CELL, so travel times start about half a cell early, and the
    # first-order scheme adds a fraction of a cell more. One cell is ~1 ns at 0.3 m.
    slack = 1.5 * (exported["scene"].cell / contracts.C0) * 1e9
    worst = float(np.min(eik[m] - geom[m]))
    assert worst >= -slack, \
        f"eikonal beats vacuum d/c by {-worst:.2f} ns, past the {slack:.2f} ns grid tolerance"
    # and it must actually be slower somewhere — that lag IS the refraction the sweep shows
    assert float(np.max(eik[m] - geom[m])) > 0


def test_time_lapse_order_is_direct_first_on_the_shared_clock(exported):
    """On one convention, no mechanism may precede the direct path.

    This is the ordering the time-lapse exists to show. Reflection travels an image-source
    path, diffraction goes via an edge, and diffuse takes two legs — all >= the straight
    line by construction, so the direct front must arrive first everywhere.
    """
    e, shape = exported["entry"], exported["shape"]
    mechs = e["mechanisms"]
    if "path_loss" not in mechs:
        pytest.skip("needs the path_loss channel")
    rd = lambda f: np.fromfile(exported["dir"] / f, dtype=np.float16).astype(np.float32).reshape(shape)
    direct = rd(mechs["path_loss"]["tau_geom_file"])
    for mech, info in mechs.items():
        if mech == "path_loss":
            continue
        other = rd(info["tau_file"])
        m = (direct < F16_MAX - 1) & (other < F16_MAX - 1)
        if not m.any():
            continue
        early = float(np.min(other[m] - direct[m]))
        assert early >= -0.5, f"{mech} arrives {-early:.2f} ns before the direct path"
        assert float(np.median(other[m] - direct[m])) >= 0, \
            f"{mech} typically precedes the direct field — the sequence is inverted"
