#!/usr/bin/env python3
"""
unet3d_train_jax.py — Phase C (train) + Phase D (eval) for the pure-JAX 3-D U-Net.

From-scratch JAX training (no Flax/Equinox) with optax. Uses **GroupNorm** instead
of BatchNorm — batch-size-independent and train==eval (no running-stat bookkeeping),
which is the clean choice for the small 3-D batches here. Reuses the hand-written
conv/pool/upsample kernels from `unet3d_jax`. Trains on the Phase-B shards
(`fw_dataset3d`), saves JAX params (.npz), and validates the whole-volume tiled
prediction against a FRESH 3-D FDTD (envelope-dB RMSE / Spearman / coherence).

ONNX export still goes through PyTorch (`fw_export`): the weights are portable
(parity proven), so train in JAX, copy into UNet3DField, export there.

  python unet3d_train_jax.py --data fw_data3d/LTE_B71_617 --epochs 40 --base 16
  python unet3d_train_jax.py --validate --params unet3d_jax.npz --band LTE_B71_617
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

import bootstrap  # noqa: F401
import unet3d_jax as UJ

HERE = Path(__file__).resolve().parent
GROUPS = 8


# --------------------------------------------------------------------------- model (GroupNorm)
def _dims(base, cin, cout):
    b = base
    return dict(e1=(cin, b), e2=(b, 2 * b), e3=(2 * b, 4 * b), bk=(4 * b, 8 * b),
                d3=(8 * b + 4 * b, 4 * b), d2=(4 * b + 2 * b, 2 * b), d1=(2 * b + b, b),
                out=(b, cout))


def init_params(key, cin=9, cout=2, base=16):
    import jax
    dims = _dims(base, cin, cout)
    P = {}
    def he(k, ci, co, kk=3):
        fan = ci * kk ** 3
        return jax.random.normal(k, (co, ci, kk, kk, kk)) * np.sqrt(2.0 / fan)
    for blk in ("e1", "e2", "e3", "bk", "d3", "d2", "d1"):
        ci, co = dims[blk]
        k1, k2, key = jax.random.split(key, 3)
        P[blk] = dict(c1=he(k1, ci, co), c2=he(k2, co, co),
                      g1s=np.ones(co), g1b=np.zeros(co),
                      g2s=np.ones(co), g2b=np.zeros(co))
    ci, co = dims["out"]
    ko, key = jax.random.split(key)
    P["out"] = dict(w=he(ko, ci, co, kk=1) * 0.1, b=np.zeros(co))
    return jax.tree_util.tree_map(lambda a: np.asarray(a, np.float32), P)


def _make_net(jax, jnp, lax):
    def pool(x):
        # differentiable MaxPool3d((2,1,2)) via reshape-max (D,W even at every level:
        # 24→12→6). reduce_window(max,-inf) breaks reverse-mode autodiff.
        N, C, D, H, W = x.shape
        x = x.reshape(N, C, D // 2, 2, H, W // 2, 2)
        return x.max(axis=(3, 6))

    def gn(x, s, b, groups=GROUPS, eps=1e-5):
        N, C = x.shape[:2]
        xr = x.reshape(N, groups, C // groups, *x.shape[2:])
        m = xr.mean(axis=(2, 3, 4, 5), keepdims=True)
        v = xr.var(axis=(2, 3, 4, 5), keepdims=True)
        xr = (xr - m) / jnp.sqrt(v + eps)
        x = xr.reshape(N, C, *x.shape[2:])
        return x * s[None, :, None, None, None] + b[None, :, None, None, None]

    def dconv(p, x):
        x = UJ.conv3d(lax, x, p["c1"]); x = gn(x, p["g1s"], p["g1b"]); x = jnp.maximum(x, 0)
        x = UJ.conv3d(lax, x, p["c2"]); x = gn(x, p["g2s"], p["g2b"]); x = jnp.maximum(x, 0)
        return x

    def forward(P, x):
        e1 = dconv(P["e1"], x)
        e2 = dconv(P["e2"], pool(e1))
        e3 = dconv(P["e3"], pool(e2))
        bk = dconv(P["bk"], pool(e3))
        d3 = dconv(P["d3"], UJ.up_concat(jax, jnp, bk, e3))
        d2 = dconv(P["d2"], UJ.up_concat(jax, jnp, d3, e2))
        d1 = dconv(P["d1"], UJ.up_concat(jax, jnp, d2, e1))
        return UJ.conv3d(lax, d1, P["out"]["w"], P["out"]["b"], pad=0)
    return forward


# --------------------------------------------------------------------------- data
def load_shards(data_dir):
    # accept a single band dir (shard_*.npz) OR a multiband root (<band>/shard_*.npz)
    files = sorted(glob.glob(str(Path(data_dir) / "shard_*.npz"))) or \
        sorted(glob.glob(str(Path(data_dir) / "*" / "shard_*.npz")))
    if not files:
        raise FileNotFoundError(f"no shards in {data_dir} — run fw_dataset3d first")
    xs = [np.load(f)["x"] for f in files]; ys = [np.load(f)["y"] for f in files]
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


# --------------------------------------------------------------------------- train (Phase C)
def train(data_dir, epochs=40, base=16, bs=4, lr=1e-3, seed=0, out=None):
    import jax, jax.numpy as jnp, optax
    from jax import lax
    x, y = load_shards(data_dir)
    cin, cout = x.shape[1], y.shape[1]
    print(f"data x{x.shape} y{y.shape}  cin={cin} cout={cout}  device={jax.default_backend()}")
    key = jax.random.PRNGKey(seed)
    P = init_params(key, cin=cin, cout=cout, base=base)
    P = jax.tree_util.tree_map(jnp.asarray, P)
    forward = _make_net(jax, jnp, lax)
    opt = optax.adam(lr); state = opt.init(P)

    def loss_fn(P, xb, yb):
        return jnp.mean((forward(P, xb) - yb) ** 2)

    @jax.jit
    def step(P, state, xb, yb):
        l, g = jax.value_and_grad(loss_fn)(P, xb, yb)
        upd, state = opt.update(g, state, P)
        return optax.apply_updates(P, upd), state, l

    n = len(x); rng = np.random.default_rng(seed)
    xj, yj = jnp.asarray(x), jnp.asarray(y)
    for ep in range(epochs):
        idx = rng.permutation(n); tot = 0.0
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            P, state, l = step(P, state, xj[b], yj[b]); tot += float(l) * len(b)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d}  mse={tot/n:.5f}")
    out = Path(out) if out else HERE / "unet3d_jax.npz"
    flat = {f"{k}.{kk}": np.asarray(vv) for k, v in P.items()
            for kk, vv in (v.items() if isinstance(v, dict) else [("_", v)])}
    np.savez(out, base=base, cin=cin, cout=cout, **flat)
    print(f"saved {out}")
    return P


# V3/V4 input-channel layout (see fw_dataset3d.INPUT_CHANNELS_V4): 0-5 material one-hot,
# 6 tx_blob, 7 freq, 8 log_distance, 9 epsr_eff, 10 sigma, 11 sdf, 12-14 normals, 15-17 antenna J
CH_LOGDIST, CH_EPSR = 8, 9
EPSR_NORM = 6.0                      # must match fw_dataset3d.EPSR_NORM (de-normalize εr)


def _load_shards_phys(data_dir):
    """load_shards + per-box (k0, h, phys_mask) for the Helmholtz-residual loss. phys_mask=1 for
    isotropic shards (the field satisfies ∇²E+k²E=0), 0 for directional shards (masked → skip)."""
    from bands_v3 import get
    files = sorted(glob.glob(str(Path(data_dir) / "shard_*.npz"))) or \
        sorted(glob.glob(str(Path(data_dir) / "*" / "shard_*.npz")))
    if not files:
        raise FileNotFoundError(f"no shards in {data_dir} — run fw_dataset3d first")
    C0 = 299_792_458.0
    xs, ys, ks, hs, ms = [], [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        x, y = d["x"], d["y"]; n = len(x)
        k0 = 2.0 * np.pi * get(str(d["band"])).f_mhz * 1e6 / C0
        directional = bool(d["directional"]) if "directional" in d.files else False
        xs.append(x); ys.append(y)
        ks.append(np.full(n, k0, np.float32)); hs.append(np.full(n, float(d["h_m"]), np.float32))
        ms.append(np.full(n, 0.0 if directional else 1.0, np.float32))
    return (np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32),
            np.concatenate(ks), np.concatenate(hs), np.concatenate(ms))


def train_resumable(data_dir, total_epochs=100, base=48, bs=8, lr=1e-3, seed=0,
                    ckpt="unet3d_ckpt.pkl", ckpt_every=5, out=None, log_every=1, phys_weight=0.05):
    """Checkpointed Phase C that SURVIVES Colab timeouts. Saves (params + optax state + epoch + rng)
    to `ckpt` every `ckpt_every` epochs; on restart it reloads that and resumes from the saved epoch.
    Point `ckpt`/`out` at Google Drive so the state persists across sessions. `data_dir` may be one
    band or a multiband root.

    Loss = MSE + `phys_weight`·Helmholtz-residual (the scalar vector-wave equation ∇²E+k²εr·E=0,
    the field reconstructed from the phase-reduced envelope; applied only to isotropic samples, since
    the directional teacher is a masked field). Set phys_weight=0 for pure MSE."""
    import jax, jax.numpy as jnp, optax, pickle, time, datetime
    from jax import lax
    import dataset_3d as D3
    x, y, kbox, hbox, physmask = _load_shards_phys(data_dir)
    cin, cout = x.shape[1], y.shape[1]
    n = len(x)
    print(f"data x{x.shape} y{y.shape}  cin={cin} cout={cout}  base={base}  "
          f"device={jax.default_backend()}  boxes={n}  phys_weight={phys_weight}  "
          f"iso_frac={physmask.mean():.2f}")
    forward = _make_net(jax, jnp, lax)
    opt = optax.adam(lr)

    LOGDIST_DIVISOR = float(D3.LOGDIST_DIVISOR)
    LAP = np.zeros((3, 3, 3), np.float32); LAP[1, 1, 1] = -6.0
    for a in range(3):
        for o in (0, 2):
            ix = [1, 1, 1]; ix[a] = o; LAP[tuple(ix)] = 1.0
    LAPj = jnp.asarray(LAP)[None, None]                     # (1,1,3,3,3) 7-point Laplacian

    def helm_res_box(pred, xb, kb, hb):
        """Per-box mean |∇²E/k² + εr·E|² (normalized, ~O(1)); E from the phase-reduced envelope."""
        d = 10.0 ** (xb[:, CH_LOGDIST] * LOGDIST_DIVISOR)           # metres (N,D,H,W)
        epsr = xb[:, CH_EPSR] * EPSR_NORM + 1.0
        ph = kb[:, None, None, None] * d
        c, s = jnp.cos(ph), jnp.sin(ph)
        Ar, Ai = pred[:, 0], pred[:, 1]
        Er = Ar * c + Ai * s; Ei = Ai * c - Ar * s                 # E = A·exp(-i k d)

        def lap(fld):
            L = lax.conv_general_dilated(fld[:, None], LAPj, (1, 1, 1), [(1, 1)] * 3,
                                         dimension_numbers=("NCDHW", "OIDHW", "NCDHW"))[:, 0]
            return L / (hb[:, None, None, None] ** 2)
        invk2 = 1.0 / (kb[:, None, None, None] ** 2)
        Rr = lap(Er) * invk2 + epsr * Er
        Ri = lap(Ei) * invk2 + epsr * Ei
        r2 = (Rr ** 2 + Ri ** 2)[:, 1:-1, 1:-1, 1:-1]              # drop the conv border
        return r2.mean(axis=(1, 2, 3))                             # (N,)

    def loss_fn(P, xb, yb, kb, hb, mb):
        pred = forward(P, xb)
        mse = jnp.mean((pred - yb) ** 2)
        if phys_weight > 0:
            r2 = helm_res_box(pred, xb, kb, hb)
            lphys = jnp.sum(mb * r2) / (jnp.sum(mb) + 1e-6)         # isotropic samples only
        else:
            lphys = jnp.asarray(0.0)
        return mse + phys_weight * lphys, (mse, lphys)

    @jax.jit
    def step(P, state, xb, yb, kb, hb, mb):
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(P, xb, yb, kb, hb, mb)
        upd, state = opt.update(g, state, P)
        return optax.apply_updates(P, upd), state, aux[0], aux[1]

    ckpt = Path(ckpt)
    if ckpt.exists():
        d = pickle.load(open(ckpt, "rb"))
        P = jax.tree_util.tree_map(jnp.asarray, d["params"])
        state = jax.tree_util.tree_map(jnp.asarray, d["opt"])
        start, rng_state = d["epoch"], d.get("rng")
        rng = np.random.default_rng(); rng.bit_generator.state = rng_state if rng_state else \
            np.random.default_rng(seed).bit_generator.state
        assert d["cin"] == cin, f"ckpt cin={d['cin']} != data cin={cin} (delete {ckpt} to restart)"
        print(f"[resume] {ckpt.name} @ epoch {start}/{total_epochs}")
    else:
        P = jax.tree_util.tree_map(jnp.asarray, init_params(jax.random.PRNGKey(seed), cin, cout, base))
        state = opt.init(P); start = 0; rng = np.random.default_rng(seed)
        print(f"[fresh ] training 0/{total_epochs}")

    def save_ckpt(ep):
        tmp = ckpt.with_suffix(".tmp")
        pickle.dump(dict(params=jax.tree_util.tree_map(np.asarray, P),
                         opt=jax.tree_util.tree_map(np.asarray, state), epoch=ep,
                         rng=rng.bit_generator.state, cin=cin, cout=cout, base=base),
                    open(tmp, "wb"))
        tmp.replace(ckpt)   # atomic: a timeout mid-write can't corrupt the checkpoint

    def _fmt(s):                                          # human "time left"
        return f"{s:.0f}s" if s < 90 else (f"{s/60:.1f} min" if s < 5400 else f"{s/3600:.2f} h")

    # Keep the corpus on the HOST (numpy); upload only each batch to the device. A big corpus
    # (tens of GB) can't live on the GPU at once — the old jnp.asarray(x) OOM'd instantly.
    ep_dt = []
    for ep in range(start, total_epochs):
        t0 = time.time(); idx = rng.permutation(n); tmse = tphys = 0.0
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            P, state, mse_b, phys_b = step(P, state, jnp.asarray(x[b]), jnp.asarray(y[b]),
                                           jnp.asarray(kbox[b]), jnp.asarray(hbox[b]), jnp.asarray(physmask[b]))
            tmse += float(mse_b) * len(b); tphys += float(phys_b) * len(b)
        dt = time.time() - t0; ep_dt.append(dt)
        ref = ep_dt[1:] or ep_dt                          # drop epoch-0 (JIT compile) once we can
        left = total_epochs - ep - 1
        eta = (sum(ref) / len(ref)) * left
        finish = (datetime.datetime.now() + datetime.timedelta(seconds=eta)).strftime('%a %H:%M')
        if ep % log_every == 0 or ep == total_epochs - 1:
            print(f"  ep{ep:3d}/{total_epochs}  mse={tmse/n:.5f}  phys={tphys/n:.5f}  {dt:.1f}s/ep  "
                  f"·  {left} epochs left  ·  ETA {_fmt(eta)} (≈{finish})")
        if (ep + 1) % ckpt_every == 0 or ep == total_epochs - 1:
            save_ckpt(ep + 1); print(f"    [ckpt] epoch {ep+1} -> {ckpt.name}")

    out = Path(out) if out else HERE / "unet3d_jax.npz"
    flat = {f"{k}.{kk}": np.asarray(vv) for k, v in P.items()
            for kk, vv in (v.items() if isinstance(v, dict) else [("_", v)])}
    np.savez(out, base=base, cin=cin, cout=cout, **flat)
    print(f"saved {out}  (ckpt {ckpt} kept for resume)")
    return P


def load_params(path):
    import jax.numpy as jnp
    d = np.load(path)
    meta = dict(base=int(d["base"]), cin=int(d["cin"]), cout=int(d["cout"]))
    P = {}
    for key in d.files:
        if key in ("base", "cin", "cout"):
            continue
        blk, sub = key.split(".", 1)
        P.setdefault(blk, {})[sub] = jnp.asarray(d[key])
    return P, meta


# --------------------------------------------------------------------------- predict + validate (Phase D)
def jax_tiled_predict(P, forward, classes, tx, h, f_mhz, box=24, stride=18):
    """Whole-volume complex U via the trained JAX net (overlap-tiled)."""
    import jax.numpy as jnp
    import dataset_3d as D3
    from fw_dataset3d import featurize3d
    norm = D3.load_norm(json.loads(bootstrap.MANIFEST.read_text()))
    ff = norm.freq_feature(f_mhz); k = 2 * np.pi / (299_792_458.0 / (f_mhz * 1e6))
    Xd, Yd, Zd = classes.shape
    acc = np.zeros((2, Xd, Yd, Zd)); wsum = np.zeros((Xd, Yd, Zd))
    ax = lambda D: sorted(set(list(range(0, max(1, D - box + 1), stride)) + [max(0, D - box)]))
    for x0 in ax(Xd):
        for y0 in ax(Yd):
            for z0 in ax(Zd):
                bx, by, bz = min(box, Xd), min(box, Yd), min(box, Zd)
                sl = (slice(x0, x0 + bx), slice(y0, y0 + by), slice(z0, z0 + bz))
                ii, jj, kk = np.mgrid[x0:x0 + bx, y0:y0 + by, z0:z0 + bz]
                d_m = np.sqrt((ii - tx[0]) ** 2 + (jj - tx[1]) ** 2 + (kk - tx[2]) ** 2) * h
                xin = featurize3d(classes[sl], (tx[0] - x0, tx[1] - y0, tx[2] - z0), d_m, ff)
                pr = np.asarray(forward(P, jnp.asarray(xin[None])))[0]
                acc[(slice(None),) + sl] += pr; wsum[sl] += 1
    Ut = (acc[0] + 1j * acc[1]) / np.maximum(wsum, 1e-9)
    ii, jj, kk = np.mgrid[0:Xd, 0:Yd, 0:Zd]
    dg = np.sqrt((ii - tx[0]) ** 2 + (jj - tx[1]) ** 2 + (kk - tx[2]) ** 2) * h
    return Ut * np.exp(-1j * k * dg)


def validate(params="unet3d_jax.npz", band_label="LTE_B71_617", npw=6.0,
             region_m=8.0, seed=99, max_cells=1_500_000):
    """Phase D g2 gate: trained JAX net vs a fresh 3-D FDTD on a held-out sub-volume."""
    import jax, jax.numpy as jnp
    from jax import lax
    from scipy.stats import spearmanr
    import fullwave3d as F3, solid_extract as SE
    from bands_v3 import get
    band = get(band_label)
    P, meta = load_params(HERE / params if not Path(params).is_absolute() else params)
    forward = jax.jit(_make_net(jax, jnp, lax))

    man = json.loads(bootstrap.MANIFEST.read_text()); cs = float(man["cell_size_m"])
    txc = np.argwhere(np.load(bootstrap.VALID_TX_MASK).any(axis=1))
    tc = txc[np.random.default_rng(seed).integers(len(txc))]
    txm = (float(tc[0]) * cs, float(tc[1]) * cs)
    crop = (txm[0] - region_m, txm[0] + region_m, 0.0, man["grid_shape"][1] * cs,
            txm[1] - region_m, txm[1] + region_m)
    s = SE.extract_solid(band, n_per_wavelength=npw, crop_m=crop,
                         tx_m=(txm[0], (man["grid_shape"][1] // 2) * cs, txm[1]), max_cells=max_cells)
    res = F3.solve3d(s.classes, s.h_m, band.f_mhz, s.tx_idx, crossings=2.0,
                     extract_phasor=True, dtype="float64")
    Utrue = res["phasor"]
    Upred = jax_tiled_predict(P, forward, s.classes, s.tx_idx, s.h_m, band.f_mhz)

    air = (s.classes == 0)
    _db = lambda a, r: 20.0 * np.log10(np.maximum(a, r * 1e-6) / r)
    at, ap = np.abs(Utrue), np.abs(Upred)
    et = _db(at, np.percentile(at[air][at[air] > 0], 99.0))
    ep = _db(ap, np.percentile(ap[air][ap[air] > 0], 99.0) or 1.0)
    m = air & (et > -60)
    rmse = float(np.sqrt(np.mean((et[m] - ep[m]) ** 2)))
    rho, _ = spearmanr(et[m], ep[m])
    a, b = Upred[m], Utrue[m]
    coh = float(np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"g2 3-D (JAX) @ {band.label}  grid {s.classes.shape}")
    print(f"   envelope dB RMSE = {rmse:.2f} dB   Spearman = {rho:+.3f}   coherence = {coh:.3f}")
    return dict(rmse_db=rmse, spearman=float(rho), coherence=coh)


def validate_conformal(params="unet3d_jax.npz", band_label="LTE_B71_617", n_tx=3, boxes_per=48,
                       npw=10.0, region_m=None, super="auto", seed=999, grids_dir=None,
                       directional_frac=0.5, out_png=None):
    """Phase D for the 18-channel conformal surrogate. Builds a HELD-OUT set with a fresh `seed`
    (transmitters + antennas the model never trained on), runs the trained net, and reports
    envelope-dB RMSE, Spearman rank-corr, and complex coherence vs the FDTD ground truth. Uses the
    SAME pipeline as training (subvol_field_em → featurize3d). `grids_dir` = frozen-geometry grids
    (fast, no OBJ); else per-Tx mesh bake (matches a mesh-baked training set). Optional pred-vs-truth
    slice PNG at `out_png`."""
    import jax, jax.numpy as jnp
    from jax import lax
    from scipy.stats import spearmanr
    from bands_v3 import get
    import fw_dataset3d as FD

    band = get(band_label); k = 2 * np.pi / band.wavelength_m
    reg = region_m if region_m is not None else FD.BAND_PLAN.get(band_label, (4.0, 10.0))[0]
    cs = float(json.loads(bootstrap.MANIFEST.read_text())["cell_size_m"])
    ff = FD.D3.load_norm(json.loads(bootstrap.MANIFEST.read_text())).freq_feature(band.f_mhz)
    baked = None
    if grids_dir:
        gd = Path(grids_dir)
        baked = (np.load(gd / "epsr_eff.npy"), np.load(gd / "sigma_eff.npy"),
                 np.load(gd / "metal_frac.npy"), np.load(gd / "material_grid.npy"), np.load(gd / "sdf.npy"))
    P, meta = load_params(HERE / params if not Path(params).is_absolute() else params)
    forward = jax.jit(_make_net(jax, jnp, lax))
    print(f"model: cin={meta['cin']} base={meta['base']}  ·  device={jax.default_backend()}")

    rng = np.random.default_rng(seed)
    txc = np.argwhere(np.load(bootstrap.VALID_TX_MASK).any(axis=1))
    xs, ys = [], []
    for _ in range(n_tx):
        tc = txc[rng.integers(len(txc))]
        directional = directional_frac > 0 and rng.random() < directional_frac
        az = float(rng.uniform(0, 360)) if directional else None
        tilt = float(rng.uniform(-15, 5)) if directional else 0.0
        kind = "panel" if directional else "iso"
        bore = FD.AP3.boresight_vec_from_angles(az, tilt) if directional else np.ones(3) / np.sqrt(3.0)
        try:
            cg, epsr, sigma, sdf, normals, U, tx, h = FD.subvol_field_em(
                band, (tc[0] * cs, tc[1] * cs), npw=npw, region_m=reg, super=super, baked=baked,
                base_cell=cs, az_deg=az, tilt_deg=tilt, kind=kind)
        except Exception as e:
            print(f"  [skip] held-out Tx: {type(e).__name__}: {e}"); continue
        Dx, Dy, Dz = U.shape
        if min(Dx, Dy, Dz) < 24:
            continue
        txf = np.array(tx, float); ref = np.percentile(np.abs(U), 99.0) or 1.0
        ii, jj, kk = np.mgrid[0:Dx, 0:Dy, 0:Dz]
        blob = np.exp(-(((ii - txf[0]) ** 2 + (jj - txf[1]) ** 2 + (kk - txf[2]) ** 2)
                        / (2.0 * FD.D3.TX_SIGMA_CELLS ** 2)))
        Jfull = (np.abs(bore)[:, None, None, None] * blob[None]).astype(np.float32)
        for x0, y0, z0 in FD._stratified_origins(rng, (Dx, Dy, Dz), 24, boxes_per):
            x0, y0, z0 = int(x0), int(y0), int(z0)
            sl = (slice(x0, x0 + 24), slice(y0, y0 + 24), slice(z0, z0 + 24))
            ii2, jj2, kk2 = np.mgrid[x0:x0 + 24, y0:y0 + 24, z0:z0 + 24]
            d_m = np.sqrt((ii2 - txf[0]) ** 2 + (jj2 - txf[1]) ** 2 + (kk2 - txf[2]) ** 2) * h
            Ut = U[sl] * np.exp(1j * k * d_m)
            ys.append((np.stack([Ut.real, Ut.imag]) / ref).astype(np.float32))
            xs.append(FD.featurize3d(cg[sl], (txf[0] - x0, txf[1] - y0, txf[2] - z0), d_m, ff,
                                     epsr=epsr[sl], sigma=sigma[sl], sdf=sdf[sl],
                                     normals=normals[(slice(None),) + sl],
                                     antenna=Jfull[(slice(None),) + sl]))
    if not xs:
        print("no held-out boxes produced"); return None
    X, Yt = np.stack(xs), np.stack(ys)

    Yp = np.concatenate([np.asarray(forward(P, jnp.asarray(X[i:i + 16]))) for i in range(0, len(X), 16)])
    mse = float(np.mean((Yp - Yt) ** 2))
    pr, tr = Yp[:, 0] + 1j * Yp[:, 1], Yt[:, 0] + 1j * Yt[:, 1]
    ap, at = np.abs(pr).ravel(), np.abs(tr).ravel()
    r = np.percentile(at[at > 0], 99.0) or 1.0
    dbp = 20 * np.log10(np.maximum(ap, r * 1e-6) / r); dbt = 20 * np.log10(np.maximum(at, r * 1e-6) / r)
    m = dbt > -60
    rmse = float(np.sqrt(np.mean((dbp[m] - dbt[m]) ** 2)))
    rho, _ = spearmanr(dbp[m], dbt[m])
    coh = float(np.abs(np.vdot(pr.ravel(), tr.ravel())) / (np.linalg.norm(pr) * np.linalg.norm(tr) + 1e-12))
    print(f"Phase D · {band.label} · {len(X)} held-out boxes (seed {seed})")
    print(f"   MSE={mse:.5f}   envelope dB RMSE={rmse:.2f} dB   Spearman={rho:+.3f}   coherence={coh:.3f}")

    if out_png:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ym = 12; ta = np.abs(tr[0, :, ym, :]); pa = np.abs(pr[0, :, ym, :])
        dB = lambda a: 20 * np.log10(np.maximum(a, (a.max() or 1) * 1e-3) / (a.max() or 1))
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(dB(ta), cmap="turbo", vmin=-40, vmax=0); ax[0].set_title("FDTD truth  |U| dB")
        ax[1].imshow(dB(pa), cmap="turbo", vmin=-40, vmax=0); ax[1].set_title("surrogate  |U| dB")
        for a in ax: a.axis("off")
        fig.suptitle(f"{band.label} · held-out box · dB RMSE {rmse:.1f} · ρ {rho:+.2f} · coh {coh:.2f}")
        fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
        print(f"   wrote {out_png}")
    return dict(mse=mse, rmse_db=rmse, spearman=float(rho), coherence=coh, n=len(X))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="fw_data3d/LTE_B71_617")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--validate", action="store_true", help="legacy 9-ch validate")
    ap.add_argument("--test", action="store_true", help="Phase D held-out test for the 18-ch conformal model")
    ap.add_argument("--grids-dir", default=None, help="frozen-geometry grids for --test (fast, no OBJ)")
    ap.add_argument("--out-png", default=None, help="pred-vs-truth slice PNG for --test")
    ap.add_argument("--params", default="unet3d_jax.npz")
    ap.add_argument("--band", default="LTE_B71_617")
    ap.add_argument("--resume", action="store_true", help="checkpointed, timeout-safe training")
    ap.add_argument("--ckpt", default="unet3d_ckpt.pkl")
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--phys-weight", type=float, default=0.05, help="Helmholtz-residual loss weight")
    a = ap.parse_args()
    if a.test:
        validate_conformal(a.params, a.band, grids_dir=a.grids_dir, out_png=a.out_png)
    elif a.validate:
        validate(a.params, a.band)
    elif a.resume:
        train_resumable(a.data, total_epochs=a.epochs, base=a.base, bs=a.bs,
                        ckpt=a.ckpt, ckpt_every=a.ckpt_every, phys_weight=a.phys_weight)
    else:
        train(a.data, epochs=a.epochs, base=a.base, bs=a.bs)


if __name__ == "__main__":
    main()
