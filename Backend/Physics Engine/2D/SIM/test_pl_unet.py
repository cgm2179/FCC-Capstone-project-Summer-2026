#!/usr/bin/env python3
"""
Local test harness for the pl_unet path-loss surrogate (SIM V1).

Loads the ONNX model + the real scene assets (SIM/web/sim_assets.js), builds the
exact 9-channel input the browser uses (SIM/web/simulator_tab.js: pathlossOnnx),
runs inference locally, sanity-checks the predicted path-loss map, and writes a PNG.

Input  x      : float32 [1, 9, H, W]   (6 material one-hot + Tx gaussian + freq + log-dist)
Output pl_norm : float32 [1, H, W]      (0..1; PL_dB = pl_norm*pl_range_db + pl_min_db)

Works on the shipped model or any freshly trained/downloaded model with the same contract:
    python3 test_pl_unet.py                         # default: web/pl_unet.onnx, 2400 MHz, centre Tx
    python3 test_pl_unet.py --model /path/new.onnx --freq 3500 --tx-x 224 --tx-y 128
"""
import argparse, base64, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_MODEL = os.path.join(HERE, "web", "pl_unet.onnx")
DEF_ASSETS = os.path.join(HERE, "web", "sim_assets.js")


def load_assets(path):
    txt = re.sub(r"^window\.SIM_ASSETS\s*=\s*", "", open(path).read().strip()).rstrip(";").strip()
    A = json.loads(txt)
    m = A["manifest"]
    H, W = m["grid_shape"]
    grid = np.frombuffer(base64.b64decode(A["grid_b64"]), dtype=np.uint8).reshape(H, W).astype(np.int64)
    fold = {int(k): int(v) for k, v in m.get("material_fold_8to6", {}).items()}
    if fold:
        grid = np.vectorize(lambda c: fold.get(int(c), int(c)))(grid)
    inside = None
    if "inside_b64" in A:
        inside = np.frombuffer(base64.b64decode(A["inside_b64"]), dtype=np.uint8).reshape(H, W).astype(bool)
    return A, m, grid, inside


def build_input(grid, m, tx_xy, freq_mhz):
    """Reproduce SIM/web/simulator_tab.js pathlossOnnx() exactly."""
    H, W = grid.shape
    n = H * W
    cell = m["cell_size_m"]
    sigma = m["tx_blob_sigma_cells"]
    dnorm = m["dist_channel_norm"]
    nrm = m["norm"]
    tx_x, tx_y = tx_xy

    x = np.zeros((9, H, W), dtype=np.float32)
    for c in range(6):                                   # channels 0-5: material one-hot
        x[c] = (grid == c)
    yy, xx = np.mgrid[0:H, 0:W]
    d2 = (xx - tx_x) ** 2 + (yy - tx_y) ** 2              # squared distance in cells
    x[6] = np.exp(-d2 / (2 * sigma * sigma))             # Tx gaussian blob
    ff = (np.log10(freq_mhz) - np.log10(nrm["freq_log_lo_mhz"])) / \
         (np.log10(nrm["freq_log_hi_mhz"]) - np.log10(nrm["freq_log_lo_mhz"]))
    x[7] = ff                                            # frequency feature (constant)
    x[8] = np.log10(np.maximum(np.sqrt(d2) * cell, 1.0)) / dnorm  # log-distance
    return x[None], d2, ff                               # add batch dim -> [1,9,H,W]


def main():
    ap = argparse.ArgumentParser(description="Local test for the pl_unet ONNX surrogate.")
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--assets", default=DEF_ASSETS)
    ap.add_argument("--freq", type=float, default=2400.0, help="frequency in MHz")
    ap.add_argument("--tx-x", type=int, default=None, help="Tx column (default: scene centroid)")
    ap.add_argument("--tx-y", type=int, default=None, help="Tx row (default: scene centroid)")
    ap.add_argument("--out", default=os.path.join(HERE, "pl_unet_test.png"))
    ap.add_argument("--cmap", default="jet", help="matplotlib colormap for the PL map")
    args = ap.parse_args()

    import onnxruntime as ort
    A, m, grid, inside = load_assets(args.assets)
    H, W = grid.shape
    nrm = m["norm"]

    # default Tx = centroid of the building interior (else grid centre)
    if args.tx_x is None or args.tx_y is None:
        if inside is not None and inside.any():
            ys, xs = np.where(inside)
            tx = (int(xs.mean()), int(ys.mean()))
        else:
            tx = (W // 2, H // 2)
    else:
        tx = (args.tx_x, args.tx_y)

    x, d2, ff = build_input(grid, m, tx, args.freq)
    print(f"model : {args.model}")
    print(f"scene : {H}x{W}  cell={m['cell_size_m']:.4f} m   Tx={tx}   f={args.freq:.0f} MHz  (freq_feat={ff:.3f})")

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name
    pl_norm = sess.run([oname], {iname: x})[0]           # [1,H,W]
    pl_norm = np.squeeze(pl_norm)
    pl_db = np.clip(pl_norm, 0, 1) * nrm["pl_range_db"] + nrm["pl_min_db"]

    # ---- sanity checks ----
    dist_m = np.sqrt(d2) * m["cell_size_m"]
    corr = float(np.corrcoef(dist_m.ravel(), pl_db.ravel())[0, 1])
    checks = {
        "output finite": bool(np.isfinite(pl_db).all()),
        "raw pl_norm in ~[0,1]": bool(pl_norm.min() > -0.05 and pl_norm.max() < 1.05),
        "PL grows with distance (corr>0.3)": corr > 0.3,
        "PL span sensible (>20 dB)": (pl_db.max() - pl_db.min()) > 20,
    }
    print(f"\noutput: {oname} {pl_norm.shape}  raw[min,max]=[{pl_norm.min():.3f},{pl_norm.max():.3f}]")
    print(f"PL(dB): min={pl_db.min():.1f}  max={pl_db.max():.1f}  mean={pl_db.mean():.1f}  "
          f"@Tx={pl_db[tx[1], tx[0]]:.1f}")
    print(f"corr(PL, distance) = {corr:+.3f}")
    print("\nsanity checks:")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(checks.values())

    # ---- render PNG ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        ax[0].imshow(grid, cmap="tab10", vmin=0, vmax=9); ax[0].set_title("materials (6 classes)")
        im = ax[1].imshow(pl_db, cmap=args.cmap); ax[1].set_title(f"predicted path loss (dB) @ {args.freq:.0f} MHz")
        ax[1].plot(tx[0], tx[1], "wo", ms=7, mec="k")
        fig.colorbar(im, ax=ax[1], fraction=0.03, pad=0.02, label="dB")
        for a in ax: a.set_xticks([]); a.set_yticks([])
        fig.tight_layout(); fig.savefig(args.out, dpi=110)
        print(f"\nsaved: {args.out}")
    except Exception as e:
        print(f"\n(plot skipped: {e})")

    print("\nRESULT:", "OK — model runs and output is sane." if ok else "CHECK FAILED — see above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
