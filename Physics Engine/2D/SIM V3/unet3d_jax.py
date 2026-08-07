#!/usr/bin/env python3
"""
unet3d_jax.py — the 3-D complex-field U-Net (UNet3DField) in PURE JAX.

Phase C, from scratch: no Flax/Equinox/Haiku. Every layer is hand-written on
`jax.lax` / `jax.numpy` — Conv3d via `lax.conv_general_dilated`, the anisotropic
MaxPool3d((2,1,2)) via `lax.reduce_window`, trilinear upsample via
`jax.image.resize`, BatchNorm3d as an affine (eval) using the checkpoint's running
stats. The forward exactly mirrors models_3d.UNet3D / fw_unet3d.UNet3DField
(9→2 channels, no sigmoid), so:

  * weights port 1:1 from the trained PyTorch checkpoint (fw_unet3d.pt), and
  * `--parity` proves this pure-JAX net matches the Torch reference before we
    train anything or export ONNX.

Training (optax, bf16) and the ONNX export live in the SIM V2 Phase-C notebook;
this module owns the architecture so both share one definition.

  python unet3d_jax.py --parity          # port fw_unet3d.pt, compare vs PyTorch
  python unet3d_jax.py --bench           # jitted forward-pass latency (a 24^3 tile)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BN_EPS = 1e-5   # torch BatchNorm3d default


# --------------------------------------------------------------------------- pure-jax layers
def _jax(x64=False):
    import jax
    if x64:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import lax
    return jax, jnp, lax


def conv3d(lax, x, w, b=None, pad=1):
    """x:(N,C,D,H,W)  w:(O,I,kd,kh,kw)  — 'same' conv when pad matches (k-1)//2."""
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1, 1),
        padding=[(pad, pad), (pad, pad), (pad, pad)],
        dimension_numbers=("NCDHW", "OIDHW", "NCDHW"))
    if b is not None:
        y = y + b[None, :, None, None, None]
    return y


def bn_eval(jnp, x, w, b, rm, rv):
    """BatchNorm3d in eval mode: affine from running stats, per channel (axis 1)."""
    scale = w / jnp.sqrt(rv + BN_EPS)
    shift = b - rm * scale
    return x * scale[None, :, None, None, None] + shift[None, :, None, None, None]


def relu(jnp, x):
    return jnp.maximum(x, 0)


def maxpool_212(lax, jnp, x):
    """MaxPool3d((2,1,2)): halve D and W (floor), leave the thin H axis intact."""
    neg = jnp.asarray(-jnp.inf, x.dtype)
    return lax.reduce_window(x, neg, lax.max,
                             window_dimensions=(1, 1, 2, 1, 2),
                             window_strides=(1, 1, 2, 1, 2), padding="VALID")


def up_concat(jax, jnp, x, skip):
    """Trilinear upsample x to skip's spatial size (align_corners=False ≈ jax half-pixel)
    then concat on channels — the U-Net skip connection."""
    N, C = x.shape[:2]
    tgt = (N, C) + tuple(skip.shape[2:])
    xr = jax.image.resize(x, tgt, method="linear", antialias=False)
    return jnp.concatenate([xr, skip], axis=1)


# --------------------------------------------------------------------------- forward
def _dconv(lax, jnp, p, x, pfx):
    """Two (Conv3d-BN-ReLU) blocks — the models_3d.dconv double-conv."""
    x = conv3d(lax, x, p[f"{pfx}.0.weight"])                                   # conv (bias=False)
    x = bn_eval(jnp, x, p[f"{pfx}.1.weight"], p[f"{pfx}.1.bias"],
                p[f"{pfx}.1.running_mean"], p[f"{pfx}.1.running_var"]); x = relu(jnp, x)
    x = conv3d(lax, x, p[f"{pfx}.3.weight"])
    x = bn_eval(jnp, x, p[f"{pfx}.4.weight"], p[f"{pfx}.4.bias"],
                p[f"{pfx}.4.running_mean"], p[f"{pfx}.4.running_var"]); x = relu(jnp, x)
    return x


def make_forward(jax, jnp, lax):
    def forward(p, x):
        e1 = _dconv(lax, jnp, p, x, "e1")
        e2 = _dconv(lax, jnp, p, maxpool_212(lax, jnp, e1), "e2")
        e3 = _dconv(lax, jnp, p, maxpool_212(lax, jnp, e2), "e3")
        b = _dconv(lax, jnp, p, maxpool_212(lax, jnp, e3), "b")
        d3 = _dconv(lax, jnp, p, up_concat(jax, jnp, b, e3), "d3")
        d2 = _dconv(lax, jnp, p, up_concat(jax, jnp, d3, e2), "d2")
        d1 = _dconv(lax, jnp, p, up_concat(jax, jnp, d2, e1), "d1")
        return conv3d(lax, d1, p["out.weight"], p["out.bias"], pad=0)          # 1x1x1 head
    return forward


# --------------------------------------------------------------------------- weight port
def load_params_from_torch(ckpt="fw_unet3d.pt", jnp=None, dtype=None):
    """Port a trained PyTorch UNet3DField state_dict into a flat {name: array} pytree.
    Conv/BN weight layouts already match (OIDHW / per-channel), so it's a straight cast."""
    import torch
    ck = torch.load(HERE / ckpt if not Path(ckpt).is_absolute() else ckpt, map_location="cpu")
    state = ck["state"]
    meta = dict(base=ck["base"], in_ch=ck["in_ch"], out_ch=ck["out_ch"])
    params = {}
    for k, v in state.items():
        if k.endswith("num_batches_tracked"):
            continue
        a = v.detach().cpu().numpy()
        params[k] = jnp.asarray(a, dtype) if jnp is not None else a
    return params, meta


# --------------------------------------------------------------------------- parity + bench
def parity(ckpt="fw_unet3d.pt", shape=(1, 9, 24, 16, 24)):
    """Load the Torch reference + this pure-JAX port, run both on one input (float64),
    report max-abs / max-rel diff. A correct port matches to ~1e-6 (residual = the
    trilinear-resize convention)."""
    import torch
    jax, jnp, lax = _jax(x64=True)
    from fw_unet3d import UNet3DField

    ck = torch.load(HERE / ckpt, map_location="cpu")
    tm = UNet3DField(cin=ck["in_ch"], cout=ck["out_ch"], base=ck["base"])
    tm.load_state_dict(ck["state"]); tm.eval(); tm.double()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float64)

    with torch.no_grad():
        yt = tm(torch.from_numpy(x)).numpy()

    params, meta = load_params_from_torch(ckpt, jnp=jnp, dtype=jnp.float64)
    forward = make_forward(jax, jnp, lax)
    yj = np.asarray(forward(params, jnp.asarray(x)))

    d = np.abs(yt - yj)
    denom = np.abs(yt).max() + 1e-30
    print(f"pure-JAX UNet3D parity  base={meta['base']} {meta['in_ch']}->{meta['out_ch']}  "
          f"in{tuple(shape)} -> out{tuple(yj.shape)}")
    print(f"  torch vs jax:  max-abs={d.max():.3e}  max-rel={d.max()/denom:.3e}  "
          f"corr={np.corrcoef(yt.ravel(), yj.ravel())[0,1]:.8f}")
    ok = d.max() / denom < 1e-4
    print("  PARITY", "OK ✓" if ok else "MISMATCH ✗")
    return ok


def bench(ckpt="fw_unet3d.pt", shape=(1, 9, 24, 24, 24), repeats=20):
    """Jitted pure-JAX forward-pass latency for one box (float32)."""
    jax, jnp, lax = _jax(x64=False)
    params, meta = load_params_from_torch(ckpt, jnp=jnp, dtype=jnp.float32)
    forward = jax.jit(make_forward(jax, jnp, lax))
    x = jnp.asarray(np.random.default_rng(0).standard_normal(shape), jnp.float32)
    y = forward(params, x); y.block_until_ready()                        # compile
    t0 = time.perf_counter()
    for _ in range(repeats):
        y = forward(params, x); y.block_until_ready()
    dt = (time.perf_counter() - t0) / repeats
    print(f"pure-JAX UNet3D forward  in{tuple(shape)} -> out{tuple(y.shape)}  "
          f"[{jax.default_backend()}]  {dt*1e3:.2f} ms/box ({1/dt:.0f} box/s)")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--ckpt", default="fw_unet3d.pt")
    a = ap.parse_args()
    if a.parity:
        parity(a.ckpt)
    if a.bench:
        bench(a.ckpt)
    if not (a.parity or a.bench):
        print("use --parity (vs PyTorch) or --bench (forward latency)")


if __name__ == "__main__":
    main()
