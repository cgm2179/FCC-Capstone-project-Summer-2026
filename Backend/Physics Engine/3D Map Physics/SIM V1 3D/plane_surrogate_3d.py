#!/usr/bin/env python3
"""Plane-surrogate scaffold for the M4 speed strategy.

The proposed path is to train 2-D UNets on axis-aligned slices rather than one
large 3-D network: xy slices across z, yz slices across x, and zx slices across
y. Each 2-D model sees geometry and Tx/frequency features on its own plane; the
three predicted stacks are then composed back into a 3-D PL/tau volume by
averaging where planes overlap. That gives a route to arbitrary/imported scenes
without a per-scene cached volume, while keeping every learner in tractable 2-D.

This module only defines the data movement and CLI plan. It intentionally does
not train, solve physics, or claim accuracy.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Mapping

import numpy as np

AXES = ("xy", "yz", "zx")


@dataclass(frozen=True)
class PlaneSpec:
    """How axis-aligned plane predictions are represented and recomposed."""

    axes: tuple[str, ...] = AXES
    compose: str = "average"
    fill_value: float = 0.0

    def __post_init__(self):
        unknown = tuple(a for a in self.axes if a not in AXES)
        if unknown:
            raise ValueError(f"unknown plane axes: {unknown}")
        if self.compose != "average":
            raise ValueError("only average composition is scaffolded")


def extract_planes(volume: np.ndarray, spec: PlaneSpec | None = None) -> dict[str, np.ndarray]:
    """Return axis-aligned slice stacks from a `(X,Y,Z)` volume.

    Shapes are:
      * `xy`: `(Z,X,Y)` so each item is an xy plane;
      * `yz`: `(X,Y,Z)` so each item is a yz plane;
      * `zx`: `(Y,Z,X)` so each item is a zx plane.
    """
    spec = spec or PlaneSpec()
    v = np.asarray(volume)
    if v.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got shape {v.shape}")
    out: dict[str, np.ndarray] = {}
    if "xy" in spec.axes:
        out["xy"] = np.moveaxis(v, 2, 0).copy()
    if "yz" in spec.axes:
        out["yz"] = v.copy()
    if "zx" in spec.axes:
        out["zx"] = np.transpose(v, (1, 2, 0)).copy()
    return out


def _plane_to_volume(axis: str, plane_stack: np.ndarray) -> np.ndarray:
    if axis == "xy":
        return np.moveaxis(plane_stack, 0, 2)
    if axis == "yz":
        return np.asarray(plane_stack)
    if axis == "zx":
        return np.transpose(plane_stack, (2, 0, 1))
    raise ValueError(f"unknown plane axis {axis!r}")


def compose_planes(
    planes: Mapping[str, np.ndarray],
    shape: tuple[int, int, int] | None = None,
    spec: PlaneSpec | None = None,
) -> np.ndarray:
    """Average plane-stack predictions back into a `(X,Y,Z)` volume."""
    spec = spec or PlaneSpec()
    if not planes:
        if shape is None:
            raise ValueError("shape is required when no planes are supplied")
        return np.full(shape, spec.fill_value, np.float32)

    acc = None if shape is None else np.zeros(shape, np.float32)
    cnt = None if shape is None else np.zeros(shape, np.float32)
    for axis in spec.axes:
        if axis not in planes:
            continue
        vol = _plane_to_volume(axis, np.asarray(planes[axis])).astype(np.float32, copy=False)
        if acc is None:
            shape = tuple(int(v) for v in vol.shape)
            acc = np.zeros(shape, np.float32)
            cnt = np.zeros(shape, np.float32)
        if tuple(vol.shape) != tuple(shape):
            raise ValueError(f"{axis} planes compose to {vol.shape}, expected {shape}")
        acc += vol
        cnt += 1.0

    if acc is None or cnt is None:
        raise ValueError("none of the requested axes were present")
    out = np.full(tuple(shape), spec.fill_value, np.float32)
    np.divide(acc, cnt, out=out, where=cnt > 0)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="M4 plane-surrogate scaffold: train xy/yz/zx 2-D UNets, then compose.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Plan:\n"
            "  1. Extract xy/yz/zx plane stacks from material/input/target volumes.\n"
            "  2. Train one or more 2-D UNets over those plane samples.\n"
            "  3. Predict plane stacks for a new scene and average them back to 3-D.\n"
            "  4. Validate against held-out full-physics volumes before enabling in UI."
        ),
    )
    ap.add_argument("--demo", action="store_true",
                    help="run a synthetic extract->compose roundtrip")
    args = ap.parse_args(argv)
    if args.demo:
        v = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
        r = compose_planes(extract_planes(v), v.shape)
        print(f"roundtrip max abs error {float(np.max(np.abs(r - v))):.3g}")
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
