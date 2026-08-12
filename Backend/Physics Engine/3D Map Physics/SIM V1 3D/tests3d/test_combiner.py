"""
M1 — the complex-field spine: Path_Loss_3D + Combine_3D.

The load-bearing test here is the SUPERSET GATE: combining path-loss-only must reproduce
`SceneV3.pathloss_maps` to < 0.01 dB. If that holds, the complex architecture is provably
a strict superset of the working engine — the browser and dataset stages keep consuming
the same (PL, T) arrays, and every later mechanism is a pure addition rather than a
replacement that might silently drift.
"""
from __future__ import annotations

import numpy as np
import pytest

import Combine_3D as CMB
import contracts
import engine_3d
import invariants as inv
import Path_Loss_3D as PLM
import synth3d as S


# --------------------------------------------------------------------------- gate
@pytest.mark.parametrize("name,builder", [
    ("free_space", lambda: S.free_space_scene(n=(24, 24, 24))),
    ("slab", lambda: S.slab_scene_3d(n=(32, 16, 16), mat=2, x0=14, x1=17)),
    ("two_plate", lambda: S.two_plate_scene_3d(n=(32, 16, 24), gap_cells=8, mat=2)),
])
def test_superset_gate(name, builder):
    """to_legacy_volumes(combine([path_loss])) == SceneV3.pathloss_maps, < 0.01 dB."""
    M, man = builder()
    sc = engine_3d.SceneV3(M, man)
    tx = (4, M.shape[1] // 2, M.shape[2] // 2)
    bands = man["freqs_mhz"][:2]

    fg = PLM.solve(sc, tx, bands=bands)
    pl_new, _t = CMB.to_legacy_volumes(CMB.combine([fg], floor_at_fspl=False))
    pl_engine = PLM.direct_pl_db(sc, tx, bands=bands)

    err = float(np.nanmax(np.abs(pl_new - pl_engine)))
    assert err < 0.01, f"{name}: complex spine drifted from the engine by {err:.5f} dB"


def test_superset_gate_real_scene(scene, manifest):
    """Same gate on the production voxel scene, not just synthetic ones."""
    ins = np.load(__import__("conftest").SIM3D / "valid_tx_mask.npy")
    xs, ys, zs = np.nonzero(ins)
    i = len(xs) // 2
    tx = (int(xs[i]), int(ys[i]), int(zs[i]))
    bands = list(scene.freqs[:2])

    fg = PLM.solve(scene, tx, bands=bands)
    pl_new, _ = CMB.to_legacy_volumes(CMB.combine([fg], floor_at_fspl=False))
    err = float(np.nanmax(np.abs(pl_new - PLM.direct_pl_db(scene, tx, bands=bands))))
    assert err < 0.01, f"real scene: drift {err:.5f} dB at tx={tx}"


# --------------------------------------------------------------------------- contract
def test_field_matches_power():
    """-10 log10 sum_pol |E|^2 == PL. The invariant the whole spine rests on."""
    M, man = S.slab_scene_3d(n=(28, 14, 14), mat=2, x0=12, x1=15)
    sc = engine_3d.SceneV3(M, man)
    tx = (3, 7, 7)
    bands = man["freqs_mhz"][:2]
    fg = PLM.solve(sc, tx, bands=bands)
    inv.assert_field_matches_power(fg.E, PLM.direct_pl_db(sc, tx, bands=bands), tol_db=0.01)


def test_phase_is_carried():
    """A power-only pipeline would lose this: the field must actually rotate with range."""
    M, man = S.free_space_scene(n=(24, 24, 24))
    sc = engine_3d.SceneV3(M, man)
    fg = PLM.solve(sc, S.centre(M.shape), bands=man["freqs_mhz"][:1])
    live = np.abs(fg.E[0, 0]) > 0
    assert float(np.ptp(np.angle(fg.E[0, 0][live]))) > 1.0, "phase is not being carried"


# --------------------------------------------------------------------------- combining
def test_coherent_vs_incoherent_addition():
    """The distinction the whole design rests on: correlated paths add as FIELDS (+6 dB
    for two equal contributions), uncorrelated ones add as POWER (+3 dB)."""
    M, man = S.free_space_scene(n=(20, 20, 20))
    sc = engine_3d.SceneV3(M, man)
    tx = S.centre(M.shape)
    bands = man["freqs_mhz"][:1]

    a = PLM.solve(sc, tx, bands=bands)
    one = CMB.combine([a], floor_at_fspl=False)

    b = PLM.solve(sc, tx, bands=bands)
    b.mechanism = "copy"
    coh = CMB.combine([a, b], floor_at_fspl=False)
    assert abs(float(np.nanmedian(one.PL_db - coh.PL_db)) - 6.0206) < 0.01

    c = PLM.solve(sc, tx, bands=bands)
    c.mechanism = "diffuse"
    c.p_incoh = a.power()
    c.E = np.zeros_like(c.E)
    c.combine_as = contracts.INCOHERENT
    incoh = CMB.combine([a, c], floor_at_fspl=False)
    assert abs(float(np.nanmedian(one.PL_db - incoh.PL_db)) - 3.0103) < 0.01


def test_bandwidth_averaging_is_noop_for_single_path():
    """With one contribution there is no interference to average away."""
    M, man = S.free_space_scene(n=(20, 20, 20))
    sc = engine_3d.SceneV3(M, man)
    fg = PLM.solve(sc, S.centre(M.shape), bands=man["freqs_mhz"][:1])
    a = CMB.combine([fg], floor_at_fspl=False)
    b = CMB.combine([fg], bandwidth_hz=20e6, n_sub=8, floor_at_fspl=False)
    assert float(np.nanmax(np.abs(a.PL_db - b.PL_db))) < 1e-3


def test_bandwidth_averaging_softens_nulls():
    """Two spatially separated sources make a real interference pattern; averaging across
    a channel must SHRINK its dynamic range. This is why the knob exists: a single-tone
    sum shows nulls no 20/100 MHz receiver ever observes, and a sim that renders them
    would fail against the scanner for the wrong reason.
    """
    M, man = S.free_space_scene(n=(24, 24, 24))
    sc = engine_3d.SceneV3(M, man)
    bands = man["freqs_mhz"][:1]

    # Two genuinely separated sources -> a distance-dependent phase difference, i.e. a
    # true spatial fringe pattern (a constant phase offset would produce none).
    a = PLM.solve(sc, (6, 12, 12), bands=bands)
    b = PLM.solve(sc, (18, 12, 12), bands=bands)
    b.mechanism = "second_source"

    tone = CMB.combine([a, b], floor_at_fspl=False)
    wide = CMB.combine([a, b], bandwidth_hz=200e6, n_sub=16, floor_at_fspl=False)

    def dyn_range(cf):
        v = cf.PL_db[np.isfinite(cf.PL_db)]
        return float(np.percentile(v, 99) - np.percentile(v, 1))

    assert dyn_range(wide) < dyn_range(tone), (
        f"bandwidth averaging did not soften the fringes: "
        f"tone {dyn_range(tone):.2f} dB -> wide {dyn_range(wide):.2f} dB")


# --------------------------------------------------------------------------- outputs
def test_fspl_floor_enforced():
    M, man = S.slab_scene_3d(n=(28, 14, 14), mat=2, x0=12, x1=15)
    sc = engine_3d.SceneV3(M, man)
    tx = (3, 7, 7)
    bands = man["freqs_mhz"][:1]
    cf = CMB.combine([PLM.solve(sc, tx, bands=bands)], floor_at_fspl=True, scene=sc, tx=tx)
    fs = PLM.free_space_field(sc, tx, bands=bands).pathloss_db()
    assert (cf.PL_db >= fs - 1e-3).all(), "a passive channel beat free space"


def test_legacy_volume_shapes(manifest):
    """The exporter contract: PL (NF,NX,NY,NZ) and T (NX,NY,NZ)."""
    M, man = S.free_space_scene(n=(16, 16, 16))
    sc = engine_3d.SceneV3(M, man)
    bands = man["freqs_mhz"][:3]
    cf = CMB.combine([PLM.solve(sc, S.centre(M.shape), bands=bands)], floor_at_fspl=False)
    pl, t = CMB.to_legacy_volumes(cf)
    assert pl.shape == (len(bands),) + M.shape and pl.dtype == np.float32
    assert t.shape == M.shape and t.dtype == np.float32


def test_delay_spread_zero_for_single_path():
    """One path has no spread; the metric must not invent one."""
    M, man = S.free_space_scene(n=(16, 16, 16))
    sc = engine_3d.SceneV3(M, man)
    bands = man["freqs_mhz"][:1]
    fg = PLM.solve(sc, S.centre(M.shape), bands=bands)
    cf = CMB.combine([fg], floor_at_fspl=False)
    ds = CMB.rms_delay_spread(cf, [fg])
    assert float(np.nanmax(ds)) < 1e-12, f"single path should have zero delay spread, got {float(np.nanmax(ds)):.3e} s"
