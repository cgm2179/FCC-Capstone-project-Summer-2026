#!/usr/bin/env python3
"""
Diffraction_3D — UTD wedge diffraction on the voxel grid.

The mechanism that fills geometric shadow. Without it a ray/eikonal map leaves the region
behind a corner at essentially zero field, which never happens physically — so along with
diffuse scattering this is what separates a deterministic map from something that
resembles measured coverage.

PHYSICS IS REUSED VERBATIM from the 2D engine, which already implements the full
Kouyoumjian-Pathak coefficient with the scipy Fresnel-integral transition function:

    physics_v2.utd_coefficient   -> COMPLEX D (sqrt m)   <- phase lives here
    physics_v2.utd_pathloss_db   -> magnitude, referenced exactly like FSPL
    physics_v2.knife_edge_j_db   -> independent cross-check (must agree at n=2)

We call BOTH: `utd_pathloss_db` for the magnitude (already referenced consistently with
`SceneV3.pathloss_maps`, so the two are directly summable) and `utd_coefficient` for
arg(D). Routing only through `utd_pathloss_db` would take |D| and throw the phase away,
which is exactly what makes a combiner unable to produce interference.

    E_d = 10**(-PL_utd/20) * exp(j*arg(D)) * exp(-j*k*(s' + s))

3-D EDGE FINDING
The 2D `find_diffracting_edges` measures the wedge parameter n directly from geometry --
it samples occupancy on a circle and takes n = (largest air arc)/pi, so the geometry
measurement and the UTD parameter are the SAME number rather than assuming n=1.5. That
idea ports slice-by-slice: run it on each XZ slice through the room band to find vertical
edges (door jambs, wall ends, column corners), then cluster the hits along Y into 3-D
edge segments. Horizontal edges (door headers, wall/ceiling junctions) come from XY
slices when `modes` includes them.

Two subtleties carried over from the 2D implementation, both of which silently disable
diffraction if got wrong:
  * face0 is where the air arc BEGINS, so phi increases THROUGH the air. Taking the other
    end gives the n-face and pushes every phi' outside the valid range.
  * The corner voxel is itself STRUCTURE, so tracing a leg from it charges the ray for the
    very wall it is bending around (~15 dB for concrete). The relay is anchored a few
    voxels out along the bisector of the air arc instead; the offset is ~1 m against legs
    of tens of metres, so the UTD geometry still references the true edge.

OBLIQUE INCIDENCE
`utd_coefficient` takes beta0, the Keller-cone angle between the incident ray and the edge
direction. In 2D it is always pi/2; in 3D it is not, and passing the true value is most of
what makes this a 3-D solver rather than a stack of 2-D ones.

usage:
    python Diffraction_3D.py --test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

from _bootstrap import SIM3D, contracts, engine_3d, physics_3d  # noqa: F401

sys.path.insert(0, str(__import__("_bootstrap").SIM2D))
import physics_v2 as P2  # noqa: E402

MECHANISM = "diffraction"
DEFAULTS = {"radius": 3, "n_angles": 72, "n_min": 1.15, "min_sep": 6,
            "n_edges": 24, "modes": ("vertical",)}
C0 = contracts.C0


# --------------------------------------------------------------------------- edges
def _longest_circular_run(b):
    """Longest contiguous run of True in a circular boolean array -> (start, length)."""
    n = len(b)
    if b.all():
        return 0, n
    d = np.concatenate([b, b])
    best_len = best_start = 0
    cur_len = 0
    for i, v in enumerate(d):
        cur_len = cur_len + 1 if v else 0
        if cur_len > best_len:
            best_len, best_start = cur_len, i - cur_len + 1
    return best_start % n, min(best_len, n)


def _edges_in_slice(occ2d, air_ok, radius, n_angles, n_min):
    """2-D corner finder on one slice. Returns (u, v, n_wedge, face0, au, av)."""
    bnd = occ2d & ~ndimage.binary_erosion(occ2d, np.ones((3, 3), bool))
    us, vs = np.nonzero(bnd)
    if not len(us):
        return []
    ang = np.arange(n_angles) * (2 * np.pi / n_angles)
    du = np.round(radius * np.cos(ang)).astype(int)
    dv = np.round(radius * np.sin(ang)).astype(int)
    NU, NV = occ2d.shape
    out = []
    for u, v in zip(us, vs):
        uu = np.clip(u + du, 0, NU - 1)
        vv = np.clip(v + dv, 0, NV - 1)
        air = ~occ2d[uu, vv]
        if air.all() or not air.any():
            continue
        start, length = _longest_circular_run(air)
        n_wedge = length * (2 * np.pi / n_angles) / np.pi
        if n_wedge < n_min or n_wedge > 2.0:
            continue
        face0 = ang[start % n_angles]
        # anchor the relay OUT of the structure, along the bisector of the air arc
        bis = face0 + 0.5 * n_wedge * np.pi
        au = av = None
        for off in (2, 3, 4, 5):
            cu = int(round(u + off * np.cos(bis)))
            cv = int(round(v + off * np.sin(bis)))
            if 0 <= cu < NU and 0 <= cv < NV and not occ2d[cu, cv] and air_ok[cu, cv]:
                au, av = cu, cv
                break
        if au is None:
            continue
        out.append((int(u), int(v), float(min(n_wedge, 2.0)), float(face0),
                    int(au), int(av)))
    return out


def find_diffracting_edges_3d(M, inside, *, radius=3, n_angles=72, n_min=1.15,
                              min_sep=6, modes=("vertical",), y_band=None,
                              y_step=2) -> list[dict]:
    """Locate 3-D diffracting edges and measure the wedge parameter n at each.

    Returns dicts with: x, y, z (edge voxel), n, face0 (rad, in the edge's normal plane),
    edge_dir (unit 3-vector), ax, ay, az (relay anchor, in air), mode.
    """
    occ = M > 0
    NX, NY, NZ = M.shape
    ys = range(*(y_band or (1, NY - 1)), y_step)
    cands: list[dict] = []

    if "vertical" in modes:
        # XZ slices -> edges running along +Y (door jambs, wall ends, columns)
        for y in ys:
            for (x, z, n_w, f0, ax, az) in _edges_in_slice(
                    occ[:, y, :], inside[:, y, :], radius, n_angles, n_min):
                cands.append(dict(x=x, y=int(y), z=z, n=n_w, face0=f0,
                                  edge_dir=(0.0, 1.0, 0.0),
                                  ax=ax, ay=int(y), az=az, mode="vertical"))
    if "horizontal" in modes:
        # XY slices -> edges running along +Z (door headers, wall/ceiling junctions)
        for z in range(1, NZ - 1, max(2 * y_step, 4)):
            for (x, y, n_w, f0, ax, ay) in _edges_in_slice(
                    occ[:, :, z], inside[:, :, z], radius, n_angles, n_min):
                cands.append(dict(x=x, y=y, z=int(z), n=n_w, face0=f0,
                                  edge_dir=(0.0, 0.0, 1.0),
                                  ax=ax, ay=ay, az=int(z), mode="horizontal"))

    # thin: keep the sharpest corner in each min_sep neighbourhood
    cands.sort(key=lambda c: -c["n"])
    kept: list[dict] = []
    for c in cands:
        if all((c["x"] - k["x"]) ** 2 + (c["y"] - k["y"]) ** 2 + (c["z"] - k["z"]) ** 2
               >= min_sep ** 2 for k in kept):
            kept.append(c)
    return kept


def build_relay_cache(scene, edges, *, path=None, dtype=np.float32, progress=False):
    """Precompute crossing_loss from every edge anchor ONCE, for reuse across all Tx.

    This is the single biggest speedup in the stack, and it exists because of a profile
    rather than a guess. Per transmitter the solver calls `crossing_loss` 17 times -- once
    for the direct field and once per diffracting edge -- and at ~2.9 s each that is 96% of
    the ~52 s per-Tx cost (the eikonal, by comparison, is 4%).

    But an edge's second leg depends only on the EDGE, not on the transmitter: the map
    "wall loss from this edge anchor to every voxel" is fixed geometry. So it can be solved
    once and reused for every Tx forever, turning 16 solves PER TX into 16 solves TOTAL.
    Over a 24-Tx precompute that is 384 solves reduced to 16.

    Mirrors the relay cache in the 2D engine_v2. Returns (n_edges, nf, NX, NY, NZ).

    dtype is float32, NOT float16: these are dB values reaching ~500, where float16 spacing
    is ~0.5 dB, and the error compounds across the two legs -- measured at 2.32 dB against
    the uncached path, far past the 0.01 dB parity bar. float32 costs 2x memory
    (~380 MB at 16 edges x 10 bands on a 262x17x132 grid) and reproduces exactly.
    """
    n = len(edges)
    out = np.empty((n, len(scene.freqs), scene.NX, scene.NY, scene.NZ), dtype)
    for i, e in enumerate(edges):
        out[i] = scene.crossing_loss((e["ax"], e["ay"], e["az"])).astype(dtype)
        if progress:
            print(f"  relay {i+1}/{n}", flush=True)
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, out)
        (p.with_suffix(".edges.json")).write_text(json.dumps(
            [{k: (list(v) if isinstance(v, tuple) else v) for k, v in e.items()}
             for e in edges], indent=1))
    return out


def load_relay_cache(path):
    """(cache array, edges) written by build_relay_cache, or (None, None)."""
    p = Path(path)
    if not p.exists():
        return None, None
    edges = json.loads(p.with_suffix(".edges.json").read_text())
    for e in edges:
        e["edge_dir"] = tuple(e["edge_dir"])
    return np.load(p), edges


def select_edges(edges, n_edges=24):
    """Pick n_edges edges: seed on the sharpest, then farthest-point sample so they spread
    over the scene instead of clustering on one doorway.

    DELIBERATELY TRANSMITTER-INDEPENDENT, unlike the 2D `SceneV2.select_edges`, which ranks
    relays by how cheaply THIS transmitter reaches each anchor. Two reasons:

      * n_edges is the RELAY CACHE budget, and the cache is what makes diffraction
        affordable at all (96% of the per-Tx cost -- see build_relay_cache). Transmitters
        rank the edges almost disjointly: on the production floor, two transmitters' top-16
        by the 2D rule overlap by 4%, so 8 transmitters would need 109 of the 240 candidate
        relays (2.6 GB, 317 s) instead of 16 (0.38 GB, 47 s, built once).
      * Spending that same budget on MORE shared edges buys the same accuracy for every
        transmitter at once. Measured on a four-room scene against an all-edges reference,
        as delivered shadow power per relay cached: Tx-ranked n_edges=8 reaches 88% but
        needs 22 relays for 5 transmitters, where the spread selection reaches 85% at 20
        relays and 100% at 24 — for any transmitter, including ones not yet placed.

    Transmitter relevance is not lost, it is charged in the physics instead: `solve` adds
    the Tx->anchor crossing loss to every edge, so an edge this transmitter cannot see
    contributes negligible power rather than a wrong one.

    Order is part of the contract: `relay[ei]` is positional, so the same edge list must
    come back in the same order for a cached relay to line up with its edge.
    """
    if len(edges) <= n_edges:
        return list(edges)
    picked = [max(edges, key=lambda e: e["n"])]
    rest = [e for e in edges if e is not picked[0]]
    while len(picked) < n_edges and rest:
        far = max(rest, key=lambda e: min((e["x"] - p["x"]) ** 2 + (e["y"] - p["y"]) ** 2
                                          + (e["z"] - p["z"]) ** 2 for p in picked))
        picked.append(far)
        rest.remove(far)
    return picked


# --------------------------------------------------------------------------- solve
def _angles_in_edge_frame(vec, edge_dir, face0):
    """(phi from face0 through the air, beta0 Keller-cone angle) for vectors `vec`
    (..., 3) relative to an edge with direction `edge_dir`."""
    e = np.asarray(edge_dir, float)
    v = np.asarray(vec, float)
    L = np.linalg.norm(v, axis=-1)
    Lc = np.maximum(L, 1e-9)
    # component along the edge -> Keller cone angle
    along = v @ e
    beta0 = np.arccos(np.clip(along / Lc, -1.0, 1.0))
    # project into the plane normal to the edge, then measure azimuth from face0
    perp = v - along[..., None] * e
    if abs(e[1]) > 0.5:            # vertical edge: plane is XZ
        a = np.arctan2(perp[..., 2], perp[..., 0])
    else:                          # horizontal edge along Z: plane is XY
        a = np.arctan2(perp[..., 1], perp[..., 0])
    return np.mod(a - face0, 2 * np.pi), beta0


def solve(scene, tx, *, bands=None, edges=None, n_edges=24, soft=True,
          modes=("vertical",), backend="numpy", max_edges_solve=None,
          relay=None, progress=False) -> "contracts.FieldGrid":
    """Diffracted coherent field for one Tx.

    Each edge contributes a Tx->edge->Rx ray. Legs are charged for the walls they cross
    using the engine's own crossing loss: leg 1 (Tx->edge) is read for free off
    `crossing_loss(tx)` at the edge anchor, and leg 2 (edge->Rx) is one crossing_loss
    solve per edge.
    """
    if backend != "numpy":
        raise NotImplementedError(f"backend {backend!r} not implemented yet (M2 GPU port)")

    freqs = np.asarray(scene.freqs if bands is None else bands, float)
    bidx = [scene.band_index(f) for f in freqs]
    NX, NY, NZ = scene.NX, scene.NY, scene.NZ
    cell = scene.cell
    inside = scene.inside if scene.inside is not None else np.ones(scene.M.shape, bool)

    if edges is None:
        edges = find_diffracting_edges_3d(scene.M, inside, modes=modes,
                                          y_band=(1, NY - 1))
    edges = select_edges(edges, n_edges=n_edges)
    if max_edges_solve:
        edges = edges[:max_edges_solve]

    gx, gy, gz = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing="ij")
    P = np.stack([gx, gy, gz], -1).astype(np.float32)          # (NX,NY,NZ,3) voxels

    obs_tx = scene.crossing_loss(tx)                            # (nf_scene, NX,NY,NZ) dB
    E = np.zeros((contracts.NPOL, len(freqs), NX, NY, NZ), np.complex64)
    tau_min = np.full((len(freqs), NX, NY, NZ), np.inf, np.float32)

    for ei, e in enumerate(edges):
        ep = np.array([e["x"], e["y"], e["z"]], float)
        ap = (e["ax"], e["ay"], e["az"])
        # Second leg from a PRECOMPUTED relay cache when available: it depends only on the
        # edge, never on the transmitter, so solving it per-Tx is pure waste (96% of the
        # per-Tx cost -- see build_relay_cache).

        v_tx = (np.asarray(tx, float) - ep)
        s_p = np.linalg.norm(v_tx) * cell
        if s_p < cell:
            continue
        phi_p, beta_p = _angles_in_edge_frame(v_tx, e["edge_dir"], e["face0"])

        v_rx = P - ep.astype(np.float32)
        s = np.linalg.norm(v_rx, axis=-1) * cell
        phi, _b = _angles_in_edge_frame(v_rx, e["edge_dir"], e["face0"])

        obs_edge = (relay[ei].astype(np.float32) if relay is not None
                    else scene.crossing_loss(ap))          # leg 2, per band
        live = (s > cell) & inside
        if not live.any():
            continue

        for i, (f, bi) in enumerate(zip(freqs, bidx)):
            lam = C0 / (f * 1e6)
            k = 2.0 * np.pi / lam
            pl = np.full((NX, NY, NZ), np.inf, np.float32)
            pl[live] = P2.utd_pathloss_db(phi[live], float(phi_p), s[live], float(s_p),
                                          float(f), n=e["n"], soft=soft).astype(np.float32)
            # charge both legs for the walls they cross (engine's own CrossingLUT)
            leg1 = float(obs_tx[bi][ap[0], ap[1], ap[2]])
            pl[live] += np.float32(leg1) + obs_edge[bi][live].astype(np.float32)

            D = P2.utd_coefficient(phi[live], float(phi_p), s[live], float(s_p), k,
                                   n=e["n"], beta0=float(beta_p), soft=soft)
            tau = ((s_p + s[live]) / C0).astype(np.float64)
            ph = np.angle(D) - 2.0 * np.pi * (f * 1e6) * tau
            amp = np.power(10.0, -pl[live] / 20.0)

            contrib = np.zeros((NX, NY, NZ), np.complex64)
            contrib[live] = (amp * np.exp(1j * ph)).astype(np.complex64)
            E[0, i] += contrib

            t = np.full((NX, NY, NZ), np.inf, np.float32)
            t[live] = tau.astype(np.float32)
            tau_min[i] = np.minimum(tau_min[i], t)

        if progress:
            print(f"  edge {ei+1}/{len(edges)}  n={e['n']:.2f}  {e['mode']}")

    fg = contracts.FieldGrid(
        E=E, tau_first=tau_min, mechanism=MECHANISM, freqs_mhz=freqs,
        tx_vox=tuple(int(v) for v in tx),
        meta={"n_edges": len(edges), "modes": list(modes), "soft": bool(soft),
              "physics": "physics_v2.utd_coefficient (complex D) + utd_pathloss_db",
              "note": "phase = arg(D) - k(s'+s); legs charged via SceneV3.crossing_loss"})
    fg.combine_as = contracts.COHERENT
    return fg


# --------------------------------------------------------------------------- self-test
def _selftest() -> int:
    sys.path.insert(0, str(SIM3D / "tests3d"))
    import synth3d as S

    ok, fails = 0, []

    def check(name, cond, msg=""):
        nonlocal ok
        if cond:
            ok += 1
            print(f"  PASS  {name}")
        else:
            fails.append(name)
            print(f"  FAIL  {name}  {msg}")

    # --- the physics we reuse must still hold (guards the 2D dependency) ---
    v = np.linspace(0.5, 3.0, 12)
    j = P2.knife_edge_j_db(v)
    check("knife-edge monotone in v", bool(np.all(np.diff(j) > 0)))
    D = P2.utd_coefficient(np.pi, np.pi / 2, 10.0, 10.0, 2 * np.pi / 0.0857, n=2.0)
    check("utd_coefficient returns complex", np.iscomplexobj(D), str(type(D)))
    check("utd D non-zero", abs(complex(np.asarray(D).ravel()[0])) > 0)

    # --- edge finding on a corner scene ---
    M, man = S.corner_scene_3d(n=(40, 12, 40), mat=2)
    sc = engine_3d.SceneV3(M, man)
    sc.inside = M == 0
    edges = find_diffracting_edges_3d(M, sc.inside, y_band=(2, 10), y_step=2)
    check("found edges on a corner scene", len(edges) > 0, f"{len(edges)} edges")
    if edges:
        ns = [e["n"] for e in edges]
        check("wedge n in [1.15, 2]", all(1.15 <= x <= 2.0 for x in ns),
              f"range {min(ns):.2f}..{max(ns):.2f}")
        check("anchors are in air",
              all(M[e["ax"], e["ay"], e["az"]] == 0 for e in edges))

    # --- solve: must put energy into the geometric shadow ---
    tx = (8, 6, 8)
    bands = man["freqs_mhz"][:1]
    fg = solve(sc, tx, bands=bands, n_edges=6, modes=("vertical",))
    check("field shape", fg.E.shape == (2, 1, 40, 12, 40), str(fg.E.shape))
    check("diffracted field is finite", bool(np.isfinite(np.abs(fg.E)).all()))

    pl_d = fg.pathloss_db()[0]
    shadow = (M == 0) & (np.arange(40)[:, None, None] > 22) & (np.arange(40)[None, None, :] > 22)
    lit = np.isfinite(pl_d) & (pl_d < 300)
    check("shadow receives diffracted energy", bool((lit & shadow).sum() > 0),
          f"{int((lit & shadow).sum())} shadow voxels lit")

    check("causal (tau >= d/c)",
          bool(np.nanmin(np.where(np.isfinite(fg.tau_first), fg.tau_first, np.inf)) >= 0))

    # phase must be carried, else the combiner cannot interfere
    live = np.abs(fg.E[0, 0]) > 0
    check("phase carried", float(np.ptp(np.angle(fg.E[0, 0][live]))) > 1.0 if live.any() else False)

    print(f"\n{ok} passed, {len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="UTD wedge diffraction on the voxel grid")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    raise SystemExit(_selftest() if a.test else (ap.print_help() or 0))
