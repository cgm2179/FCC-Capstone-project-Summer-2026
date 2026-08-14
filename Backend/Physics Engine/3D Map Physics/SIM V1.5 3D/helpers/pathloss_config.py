#!/usr/bin/env python3
"""
pathloss_config.py — the single parameterized source of truth for path loss (SIM V1.5).

Every path-loss knob lives here, split into the two layers that actually behave
differently. Layer 1 is the PL(x,y,z) *field* the physics engine computes and the
Triplanar UNet caches; Layer 2 is the *link budget* applied afterwards (cheap, per
query, nothing to learn):

  Layer 1 — PL field
    * ple_n              path-loss exponent n            (near slope for dual-slope)
    * ple_far            far-slope exponent past dbp     (dual-slope only)
    * d0_m               reference distance d0
    * pl_d0_db           close-in reference PL(d0)        (None -> FSPL at d0, per band)
    * shadow_sigma_db    shadow-fading std dev sigma
    * breakpoint_m       dual-slope breakpoint dbp        (None -> 4*ht*hr*f/c)
    * fi_alpha/beta/gamma  floating-intercept triple (alpha,beta,gamma) — ABG model

  Layer 2 — link budget   (RSRP = EIRP - Lc - PL - M)
    * tx_power_dbm  Pt   +  tx_gain_dbi  Gt   ->  EIRP = Pt + Gt
    * cable_loss_db Lc   cable / implementation loss
    * reliability   ->   z_sigma   ->   fade margin  M = z_sigma * sigma

The Layer-1 spreading model is **selectable and mode-aware**:
    indoor  -> "close_in"   (log-distance / Motley-Keenan spreading; high PLE; P.1238-ish)
    outdoor -> "dual_slope" (breakpoint / street-canyon; low near-PLE, steep far-PLE)
"floating_intercept" (ABG) is available in either mode as a calibration model.

Defaults come from PathLossConfig.indoor() / .outdoor(); override any field by keyword.
The whole thing round-trips to JSON so the engine, dataset generator, surrogate and
receiver all read one config (mirrors how manifest_3d.json anchors SIM V1 3D).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

C_MPS = 299_792_458.0

# One-sided normal reliability factors z_sigma: fraction of shadow-faded locations that
# stay above threshold. Matches the 2-D optimizer (phase_e_optimizer.py / phase_a.py).
RELIABILITY_Z = {"50": 0.00, "80": 0.84, "90": 1.28, "95": 1.65, "99": 2.33}

VALID_MODES = ("indoor", "outdoor")
VALID_MODELS = ("close_in", "floating_intercept", "dual_slope")


def fspl_db(f_mhz: float, d_m: float = 1.0) -> float:
    """Free-space path loss (dB): 20log10(4*pi*d*f/c). MHz + metres convention."""
    return 20.0 * math.log10(max(f_mhz, 1e-9)) + 20.0 * math.log10(max(d_m, 1e-9)) - 27.55


@dataclass
class PathLossConfig:
    mode: str = "indoor"
    model: str = "close_in"

    # --- Layer 1: PL field ---
    ple_n: float = 3.0
    ple_far: float = 4.0
    d0_m: float = 1.0
    pl_d0_db: float | None = None
    shadow_sigma_db: float = 8.0
    breakpoint_m: float | None = None
    fi_alpha_db: float = 32.4
    fi_beta: float = 2.0
    fi_gamma: float = 2.0

    # --- Layer 2: link budget ---
    tx_power_dbm: float = 20.0
    tx_gain_dbi: float = 3.0
    cable_loss_db: float = 0.0
    reliability: str = "90"

    # --- geometry (for the auto breakpoint) ---
    tx_height_m: float = 2.5
    rx_height_m: float = 1.5

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        if self.model not in VALID_MODELS:
            raise ValueError(f"model must be one of {VALID_MODELS}, got {self.model!r}")
        if str(self.reliability) not in RELIABILITY_Z:
            raise ValueError(f"reliability must be one of {tuple(RELIABILITY_Z)}")
        self.reliability = str(self.reliability)

    # ---- Layer-2 derived quantities ----
    @property
    def eirp_dbm(self) -> float:
        """EIRP = Pt + Gt."""
        return self.tx_power_dbm + self.tx_gain_dbi

    @property
    def z_sigma(self) -> float:
        return RELIABILITY_Z[self.reliability]

    @property
    def fade_margin_db(self) -> float:
        """Fade margin M = z_sigma * sigma."""
        return self.z_sigma * self.shadow_sigma_db

    # ---- Layer-1 helpers ----
    def pl_ref_db(self, f_mhz: float) -> float:
        """Close-in reference PL(d0): explicit pl_d0_db if given, else FSPL at d0."""
        return self.pl_d0_db if self.pl_d0_db is not None else fspl_db(f_mhz, self.d0_m)

    def breakpoint(self, f_mhz: float) -> float:
        """Breakpoint distance dbp: explicit, else the two-ray break 4*ht*hr*f/c."""
        if self.breakpoint_m is not None:
            return float(self.breakpoint_m)
        return 4.0 * self.tx_height_m * self.rx_height_m * (f_mhz * 1e6) / C_MPS

    # ---- mode presets ----
    @classmethod
    def indoor(cls, **kw) -> "PathLossConfig":
        base = dict(mode="indoor", model="close_in", ple_n=3.0, shadow_sigma_db=8.0,
                    tx_height_m=2.5, rx_height_m=1.5)
        base.update(kw)
        return cls(**base)

    @classmethod
    def outdoor(cls, **kw) -> "PathLossConfig":
        base = dict(mode="outdoor", model="dual_slope", ple_n=2.0, ple_far=4.0,
                    shadow_sigma_db=6.0, tx_height_m=10.0, rx_height_m=1.5)
        base.update(kw)
        return cls(**base)

    @classmethod
    def for_mode(cls, mode: str, **kw) -> "PathLossConfig":
        return cls.outdoor(**kw) if mode == "outdoor" else cls.indoor(**kw)

    # ---- IO ----
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PathLossConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def summary(self) -> dict:
        """Human-readable roll-up of the derived link-budget terms."""
        return dict(mode=self.mode, model=self.model, ple_n=self.ple_n,
                    shadow_sigma_db=self.shadow_sigma_db, eirp_dbm=self.eirp_dbm,
                    cable_loss_db=self.cable_loss_db, reliability=self.reliability,
                    z_sigma=self.z_sigma, fade_margin_db=round(self.fade_margin_db, 2))
