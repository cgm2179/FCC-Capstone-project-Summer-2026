#!/usr/bin/env python3
"""
engine.py — SceneV15: the parameterized default path-loss engine.

Wraps SIM V1 3D's `SceneV3` (the ray-marched geometry + per-crossing Fresnel/Airy
material loss) and swaps its fixed close-in spreading for the mode-selected, fully
parameterized Layer-1 model. Everything geometric is reused verbatim; only the spreading
term (and the Layer-2 link budget) become configurable:

    PL(x,y,z) = spreading(d ; cfg, mode)  +  sat_obs( crossing_loss(geometry) )
    RSRP      = EIRP - Lc - PL - fade_margin

Indoor + close_in with `ple_n = manifest.n_exp` reproduces SceneV3 to ~0.01 dB (the
parametric layer is a strict generalization); outdoor selects the dual-slope breakpoint
model instead. The Triplanar UNet caches PL(x,y,z); the link budget is a per-query add.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import link_budget as LB
import pathloss_models as PM
from pathloss_config import PathLossConfig

# reuse SceneV3 + sat_obs from the sibling SIM V1 3D folder
_V1 = Path(__file__).resolve().parent.parent / "SIM V1 3D"
if str(_V1) not in sys.path:
    sys.path.insert(0, str(_V1))


class SceneV15:
    """Parameterized path loss over a reused SceneV3 geometry."""

    def __init__(self, scene_v3, cfg: PathLossConfig):
        self.s = scene_v3
        self.cfg = cfg

    # ---------- Layer 1: the PL field ----------
    def distance_m(self, tx):
        tx = np.asarray(tx, np.float32)
        return np.maximum(np.linalg.norm(self.s.P - tx, axis=1) * self.s.cell, self.cfg.d0_m)

    def spreading_maps(self, tx):
        """Mode-selected spreading PL(d) per band -> (nf, NX, NY, NZ)."""
        d = self.distance_m(tx)
        out = np.stack([PM.pathloss_db(d, float(f), self.cfg) for f in self.s.freqs], 0)
        return out.reshape(len(self.s.freqs), self.s.NX, self.s.NY, self.s.NZ).astype(np.float32)

    def pathloss_maps(self, tx):
        """Full PL field: mode-selected spreading + (sat-capped) geometry crossing loss."""
        from engine_3d import sat_obs
        spread = self.spreading_maps(tx)
        cl = self.s.crossing_loss(np.asarray(tx, np.float32))
        if getattr(self.s, "use_satobs", False):
            cl = sat_obs(cl, self.s.obs_T, self.s.obs_S).astype(np.float32)
        return spread + cl

    def pathloss_map(self, tx, f_mhz):
        return self.pathloss_maps(tx)[self.s.band_index(f_mhz)]

    # ---------- Layer 2: link budget ----------
    def rsrp_maps(self, tx, *, apply_margin: bool = True):
        return LB.rsrp_dbm(self.pathloss_maps(tx), self.cfg, apply_margin=apply_margin)

    def coverage(self, tx, f_mhz: float, *, threshold_dbm: float = -85.0, mask=None) -> float:
        """Coverage at one band (mask must match the 3-D grid)."""
        return LB.coverage(self.pathloss_map(tx, f_mhz), self.cfg,
                           threshold_dbm=threshold_dbm, mask=mask)


def load_scenev15(sim_v1_dir, cfg: PathLossConfig | None = None, *, mode: str = "indoor"):
    """Build a SceneV3 from a SIM V1 3D folder and wrap it as SceneV15.

    Pass an explicit `cfg`, or a `mode` ("indoor"/"outdoor") to take the mode preset.
    """
    d = str(Path(sim_v1_dir))
    if d not in sys.path:
        sys.path.insert(0, d)
    from engine_3d import load_scene
    scene, man = load_scene(d)
    if cfg is None:
        cfg = PathLossConfig.for_mode(mode)
    return SceneV15(scene, cfg), man
