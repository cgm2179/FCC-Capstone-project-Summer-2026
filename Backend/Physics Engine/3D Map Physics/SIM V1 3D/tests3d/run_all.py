#!/usr/bin/env python3
"""
Test runner for the 3D simulation harness.

    python TESTS3D/run_all.py --fast     # analytic + scene checks, no GPU, no big solves
    python TESTS3D/run_all.py --full     # everything except GPU
    python TESTS3D/run_all.py --gpu      # include CUDA/Colab tests
    python TESTS3D/run_all.py --selftest # run each mechanism module's own --test

Exit code is 0 only if every selected test passed (xfail does not count as failure —
the scene-fidelity tests are expected-fail until M0.4).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM3D = HERE.parent
MECH = SIM3D.parent / "Wave Behavior" / "Enivronmental Interaction"

MECHANISMS = ["Path_Loss_3D", "Reflection_3D", "Refraction_3D",
              "Diffraction_3D", "Absorption_3D", "Scattering_3D", "Combine_3D"]

# Modules that keep the same `--test` convention but live in SIM V1 3D rather than
# beside the mechanisms.
ENGINE_SELFTESTS = ["dataset_3d"]


def run_pytest(args: list[str]) -> int:
    cmd = [sys.executable, "-m", "pytest", str(HERE), "-q", "--no-header", *args]
    print(f"$ {' '.join(cmd[-4:])}\n")
    return subprocess.call(cmd, cwd=str(HERE))


def run_selftests() -> int:
    """Each mechanism keeps a dependency-free `python X_3D.py --test` (physics_v2 convention)."""
    rc = 0
    for name in MECHANISMS:
        p = MECH / f"{name}.py"
        if not p.exists() or p.stat().st_size == 0:
            print(f"  {name:18s} SKIP (not implemented yet)")
            continue
        r = subprocess.call([sys.executable, str(p), "--test"], cwd=str(MECH))
        print(f"  {name:18s} {'OK' if r == 0 else 'FAIL'}")
        rc |= r
    for name in ENGINE_SELFTESTS:
        p = SIM3D / f"{name}.py"
        if not p.exists() or p.stat().st_size == 0:
            print(f"  {name:18s} SKIP (not implemented yet)")
            continue
        r = subprocess.call([sys.executable, str(p), "--test"], cwd=str(SIM3D))
        print(f"  {name:18s} {'OK' if r == 0 else 'FAIL'}")
        rc |= r
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fast", action="store_true", help="skip slow and gpu tests (default)")
    g.add_argument("--full", action="store_true", help="everything except gpu")
    g.add_argument("--gpu", action="store_true", help="include gpu tests")
    ap.add_argument("--selftest", action="store_true", help="also run mechanism self-tests")
    ap.add_argument("-k", dest="expr", default=None, help="pytest -k expression")
    a = ap.parse_args(argv)

    args = []
    if a.gpu:
        pass
    elif a.full:
        args += ["-m", "not gpu"]
    else:
        args += ["-m", "not gpu and not slow"]
    if a.expr:
        args += ["-k", a.expr]

    rc = run_pytest(args)
    if a.selftest:
        print("\nmechanism self-tests:")
        rc |= run_selftests()

    print("\n" + ("PASS" if rc == 0 else "FAIL"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
