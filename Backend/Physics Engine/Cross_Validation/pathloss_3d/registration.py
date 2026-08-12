#!/usr/bin/env python3
"""PR #27 follow-up — derive the CORRECT px→voxel registration (Commit 1).

Why this exists
---------------
`validate_3d.py` sampled the 3-D field through `SIM V1 3D/registration_3d.json`, which stores
a **scale only** (`vx = px·0.2264, vz = py·0.2264`, no flip/offset). That mapping put every
measured point in the wrong voxel: PCI 397 scored Spearman ρ = **0.048** vs the 2-D model's
0.643 on the *same* points. The earlier `fit_registration.py` PCA replacement reached IoU 0.39
(no better) and was never wired in.

What the georeference actually says
-----------------------------------
The voxel grid is aligned to the floor-plan RASTER (manifest_3d.json: "the OBJ X-span registers
to the floor-plan width, 1150 px"), so px→vox is **scale + per-axis sign flip + translation, no
rotation, no transpose** (X↔width, Z↔height). Anchored scale s = meters_per_px / cell_size_m =
0.226421. Only 4 candidate orientations exist.

Each flip's translation is parametrized as a small OFFSET from its NATURAL extent-aligning value
(`v = s·px + off` for a non-flipped axis; `v = s·(px_max − px) + off` for a flipped axis), which
keeps the mapping **non-degenerate** — a plain ±offset search collapses a flipped axis onto one
grid edge and then "correlates" for the wrong reason.

Disambiguation (empirically decisive, see registration_v2 in the JSON)
----------------------------------------------------------------------
Two independent signals both pick flip **(−1, +1)** (voxel +X runs opposite floor-plan px_x):
  * footprint recall (2-D inside-air mapped onto the 3-D air footprint): 0.83 vs 0.72–0.79,
  * the **2-D-field oracle** (dense correlation of the 3-D field vs the known-good 2-D field,
    ~6 k inside px): +0.43 for (−1,+1), while the other X sign gives +0.41 and both Z-flips go
    negative. Measured ρ confirms: +0.30 at (−1,+1) vs −0.03 / −0.25 / −0.12.
Fixing the flip lifts PCI 397 from 0.048 → ~0.30; the residual gap to the 2-D 0.64 is the
façade-sum physics, which Commit 2 replaces with the explicit directional model.

Output: writes `affine_px_to_vox = [[a,0,c],[0,e,f]]` into `SIM V1 3D/registration_3d.json`
(read by `validate_3d.py`) plus the full 4-orientation diagnostic.

usage:
  python registration.py                 # fit (oracle = PCI 397), write, print the table
  python registration.py --pci 396       # use a different PCI's fields as the oracle
  python registration.py --dry-run
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SIM3D = REPO / "Physics Engine" / "3D Map Physics" / "SIM V1 3D"
STEP1 = REPO / "Physics Engine" / "2D" / "STEP_1"
STEP2 = REPO / "Physics Engine" / "2D" / "STEP_2"
PL2D_OUT = REPO / "Physics Engine" / "Cross_Validation" / "pathloss_2d" / "out"
PL3D_OUT = HERE / "out"
REG3D = SIM3D / "registration_3d.json"
RECORDS_JS = REPO / "Data" / "records_data.js"


# --------------------------------------------------------------------------- helpers
def anchored_scale():
    meta = json.loads((STEP1 / "floorplan_meta.json").read_text())
    man = json.loads((SIM3D / "manifest_3d.json").read_text())
    return float(meta["meters_per_px"]) / float(man["cell_size_m"])


def footprints():
    """2-D inside-air (H,W) and 3-D interior-air footprint (NX,NZ)."""
    mat2 = np.load(STEP1 / "material_grid.npy")
    out2 = np.load(STEP2 / "outside_mask.npy")
    ins2 = (mat2 == 0) & (~out2)
    foot3 = np.load(SIM3D / "inside_mask.npy").any(axis=1)
    return ins2, foot3


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if len(a) < 8:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean(); d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def sample2(grid, a, b):
    """Bilinear sample of grid at (a=axis0, b=axis1)."""
    A0, A1 = grid.shape
    a = np.clip(np.asarray(a, float), 0, A0 - 1.001)
    b = np.clip(np.asarray(b, float), 0, A1 - 1.001)
    a0, b0 = np.floor(a).astype(int), np.floor(b).astype(int)
    fa, fb = a - a0, b - b0
    return (grid[a0, b0] * (1 - fa) * (1 - fb) + grid[a0 + 1, b0] * fa * (1 - fb)
            + grid[a0, b0 + 1] * (1 - fa) * fb + grid[a0 + 1, b0 + 1] * fa * fb)


def load_oracle(pci):
    """(rsrp2d (H,W), rsrp3d slice (NX,NZ)) or (None, None) if the fields aren't built."""
    c2 = sorted(PL2D_OUT.glob(f"metrics_bs_{pci}_*MHz.npz"))
    c3 = sorted(PL3D_OUT.glob(f"metrics3d_bs_{pci}_*MHz.npz"))
    if not c2 or not c3:
        return None, None
    d2 = np.load(c2[0], allow_pickle=True)
    d3 = np.load(c3[0], allow_pickle=True)
    return np.asarray(d2["rsrp"], float), np.asarray(d3["rsrp"], float)[:, int(d3["rx_y_vox"]), :]


def measured_397():
    """(px, py, rsrp) for LTE B13 PCI 397 — the confirmation set. Empty arrays if unavailable."""
    try:
        t = RECORDS_JS.read_text(encoding="utf-8")
        recs = json.loads(t[t.index("["):t.rindex("]") + 1])
    except Exception:                                                # noqa: BLE001
        return np.array([]), np.array([]), np.array([])
    pts = [r for r in recs if r.get("network") == "lte" and r.get("band") == 13
           and r.get("pci") == 397 and r.get("rsrp") is not None]
    return (np.array([r["px"] for r in pts], float), np.array([r["py"] for r in pts], float),
            np.array([r["rsrp"] for r in pts], float))


# --------------------------------------------------------------------------- the fit
def mapxy(sx, sz, ox, oz, px, py, s, pxmax_x, pxmax_y):
    vx = (s * px if sx > 0 else s * (pxmax_x - px)) + ox
    vz = (s * py if sz > 0 else s * (pxmax_y - py)) + oz
    return vx, vz


def fit(pci=397, verbose=True):
    ins2, foot3 = footprints()
    H, W = ins2.shape
    NX, NZ = foot3.shape
    s = anchored_scale()
    pxmax_x, pxmax_y = W - 1, H - 1
    rsrp2d, rsrp3d = load_oracle(pci)
    mpx, mpy, meas = measured_397()

    ys, xs = np.nonzero(ins2)
    Pall = np.stack([xs.astype(float), ys.astype(float)], 1)
    Pd = Pall[::max(1, len(Pall) // 6000)]
    val2 = sample2(rsrp2d, Pd[:, 1], Pd[:, 0]) if rsrp2d is not None else None

    def recall(sx, sz, ox, oz):
        vx, vz = mapxy(sx, sz, ox, oz, Pall[:, 0], Pall[:, 1], s, pxmax_x, pxmax_y)
        ix = np.round(vx).astype(int); iz = np.round(vz).astype(int)
        k = (ix >= 0) & (ix < NX) & (iz >= 0) & (iz < NZ)
        m = np.zeros((NX, NZ), bool); m[ix[k], iz[k]] = True
        return float((m & foot3).sum()) / int(foot3.sum())

    rows = []
    for sx in (1, -1):
        for sz in (1, -1):
            # small offset polished by the oracle when available, else by recall
            best = (-9.0, 0, 0)
            for ox in range(-14, 15, 2):
                for oz in range(-14, 15, 2):
                    if val2 is not None:
                        score = spearman(val2, sample2(rsrp3d, *mapxy(sx, sz, ox, oz, Pd[:, 0],
                                                                      Pd[:, 1], s, pxmax_x, pxmax_y)))
                    else:
                        score = recall(sx, sz, ox, oz)
                    if score > best[0]:
                        best = (score, ox, oz)
            _, ox, oz = best
            rec = recall(sx, sz, ox, oz)
            dense = (spearman(val2, sample2(rsrp3d, *mapxy(sx, sz, ox, oz, Pd[:, 0], Pd[:, 1],
                                                           s, pxmax_x, pxmax_y)))
                     if val2 is not None else float("nan"))
            m_rho = (spearman(meas, sample2(rsrp3d, *mapxy(sx, sz, ox, oz, mpx, mpy,
                                                           s, pxmax_x, pxmax_y)))
                     if (len(meas) and rsrp3d is not None) else float("nan"))
            a = (-s if sx < 0 else s)
            c = (s * pxmax_x + ox) if sx < 0 else float(ox)
            e = (-s if sz < 0 else s)
            f = (s * pxmax_y + oz) if sz < 0 else float(oz)
            rows.append(dict(flips=[sx, sz], offset=[ox, oz],
                             affine=[[a, 0.0, float(c)], [0.0, e, float(f)]],
                             recall=round(rec, 3),
                             dense_oracle_rho=None if np.isnan(dense) else round(dense, 3),
                             measured_rho=None if np.isnan(m_rho) else round(m_rho, 3)))

    # rank: dense oracle first (tied to validation), then recall
    def key(r):
        return (r["dense_oracle_rho"] if r["dense_oracle_rho"] is not None else -9, r["recall"])
    winner = max(rows, key=key)

    if verbose:
        print(f"anchored scale s = {s:.6f}   grid (NX,NZ)=({NX},{NZ})   frame (H,W)=({H},{W})")
        print(f"natural flip offsets: X {s*pxmax_x:.1f}  Z {s*pxmax_y:.1f}")
        print(f"oracle = PCI {pci} dense 2-D-vs-3-D field correlation\n")
        print(f"  {'flips':10}{'recall':>8}{'dense_rho':>11}{'measured_rho':>14}")
        for r in sorted(rows, key=key, reverse=True):
            d = "  n/a" if r["dense_oracle_rho"] is None else f"{r['dense_oracle_rho']:.3f}"
            m = "  n/a" if r["measured_rho"] is None else f"{r['measured_rho']:.3f}"
            mark = "  <-- chosen" if r is winner else ""
            print(f"  {str(r['flips']):10}{r['recall']:>8.3f}{d:>11}{m:>14}{mark}")
        print(f"\nchosen affine px->vox = {winner['affine']}")
    return winner, rows, s


def write(pci=397, dry_run=False):
    winner, rows, s = fit(pci=pci, verbose=True)
    reg = json.loads(REG3D.read_text())
    reg["affine_px_to_vox"] = winner["affine"]
    reg["registration_v2"] = {
        "method": "anchored-scale (no rotation/transpose), 4-flip with natural extent-aligning "
                  "offset (non-degenerate), flip disambiguated by footprint recall + the "
                  "dense 2-D-field oracle",
        "anchored_scale": round(s, 6), "chosen_flips": winner["flips"],
        "offset": winner["offset"], "footprint_recall": winner["recall"],
        "oracle_pci": pci, "dense_oracle_rho": winner["dense_oracle_rho"],
        "measured_rho_pci397": winner["measured_rho"], "candidates": rows,
        "note": "px(1150x515 records frame) -> voxel: vx=a*px_x+c (axis0=X, floor-plan width; "
                "a<0 means voxel +X runs opposite px_x = west), vz=e*px_y+f (axis1=Z, height). "
                "Supersedes px_to_vox_scale (scale-only) which scrambled validation (rho 0.048).",
    }
    if dry_run:
        print("\n[dry-run] registration_3d.json not modified")
        return winner
    REG3D.write_text(json.dumps(reg, indent=2))
    print(f"\n-> wrote affine_px_to_vox + registration_v2 into {REG3D}")
    return winner


def _selftest():
    ok, fails = 0, []

    def check(name, cond, msg=""):
        nonlocal ok
        if cond:
            ok += 1; print(f"  PASS  {name}")
        else:
            fails.append(name); print(f"  FAIL  {name}  {msg}")

    winner, rows, s = fit(pci=397, verbose=False)
    check("anchored scale ~0.2264", abs(s - 0.226421) < 1e-4, f"{s:.6f}")
    check("no transpose: 4 orientations", len(rows) == 4)
    check("winner flip is (-1,+1)", winner["flips"] == [-1, 1], str(winner["flips"]))
    check("winner footprint recall > 0.8", winner["recall"] > 0.8, f"{winner['recall']}")
    if winner["measured_rho"] is not None:
        check("PCI 397 measured rho recovers > 0.2 (was 0.048)",
              winner["measured_rho"] > 0.2, f"{winner['measured_rho']}")
    both_z_flips_negative = all(
        (r["dense_oracle_rho"] is None) or (r["dense_oracle_rho"] < 0)
        for r in rows if r["flips"][1] == -1)
    check("both Z-flips lose the oracle (Z orientation is decisive)", both_z_flips_negative)
    print(f"\n{ok} passed, {len(fails)} failed")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pci", type=int, default=397, help="PCI whose 2-D/3-D fields disambiguate the flip")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", action="store_true", help="run the self-test (no writes)")
    args = ap.parse_args()
    if args.test:
        raise SystemExit(_selftest())
    write(pci=args.pci, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
