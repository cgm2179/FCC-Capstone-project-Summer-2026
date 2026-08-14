#!/usr/bin/env python3
"""
Data contracts for the 3-D propagation stack (M1 — the complex-field spine).

Every mechanism module emits a `FieldGrid`; `Combine_3D.combine()` sums them into a
`CombinedField`; `to_legacy_volumes()` converts that back into the exact (PL, T) arrays
`export_pl_volume.py` already serializes, so the browser and dataset stages need no changes.

WHY COMPLEX FIELDS
------------------
Mechanisms must emit amplitude AND phase, not power. The moment a module returns dB, the
phase is gone and the combiner can only add powers — which erases the interference
structure (standing-wave peaks and nulls) that makes a coverage map read as measured
rather than rendered. Diffuse scattering is the one exception: its contributions are
mutually uncorrelated, so it fills `p_incoh` instead of `E`.

AMPLITUDE AND PHASE REFERENCE
-----------------------------
    E = 10**(-PL/20) * exp(-j*2*pi*f*tau)        =>   -10*log10(sum_pol |E|^2) == PL_dB

with PL referenced exactly as `SceneV3.pathloss_maps` defines it (fspl_1m(f) + spreading
+ per-crossing slab loss). That identity is asserted by `invariants.assert_field_matches_power`
and is what makes the complex spine a *strict superset* of the working engine rather than
a parallel implementation that might drift.

The key reuse: amplitude comes from the calibrated `CrossingLUT` (correct magnitude,
already decohered), and phase comes from the 3-D eikonal `T` that `SceneV3.arrival_time`
already solves — that IS the excess path length including in-wall slowdown and Fermat
corner-wrap. So complex fields cost no new solver, and we never fabricate Fabry-Perot
phase that a 0.3 m raster cannot resolve.

POLARIZATION
------------
Axis 0 of `E` is (TE, TM) resolved in the local plane of incidence at each interaction.
Modules that do not track polarization put the field in TE and leave TM zero; the
combiner sums |E|^2 over the axis either way, so a scalar module stays correct.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

C0 = 299_792_458.0
NPOL = 2                      # (TE, TM)
POL_NAMES = ("te", "tm")

COHERENT = "coherent"         # sums as complex field  (specular, direct, diffracted)
INCOHERENT = "incoherent"     # sums as power          (diffuse scattering)
DIAGNOSTIC = "diagnostic"     # NOT summed into received power — a decomposition of the
                              # direct path for visualization (refraction, absorption).
                              # The eikonal already bends (refraction) and the CrossingLUT
                              # already includes Im(q) (absorption), so adding these as
                              # contributions would double-count. Recorded per-mechanism
                              # (for the viz) but excluded from the power sum and T_first.


# --------------------------------------------------------------------------- FieldGrid
@dataclass
class FieldGrid:
    """One mechanism's contribution, for one Tx, over all bands.

    Shapes (NF = bands, NX/NY/NZ = grid):
        E         complex64 (NPOL, NF, NX, NY, NZ)   phasor; |E|^2 is linear power
        tau_first float32   (NF, NX, NY, NZ)         earliest arrival via this mechanism, s
        tau_rms   float32   (NF, NX, NY, NZ)         delay spread WITHIN this mechanism, s
        p_incoh   float32   (NF, NX, NY, NZ)         linear power that must add incoherently
    """
    E: np.ndarray
    tau_first: np.ndarray
    mechanism: str
    freqs_mhz: np.ndarray
    tx_vox: tuple
    tau_rms: np.ndarray | None = None
    p_incoh: np.ndarray | None = None
    combine_as: str = COHERENT
    meta: dict = field(default_factory=dict)

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_pl_tau(cls, pl_db, tau_s, freqs_mhz, mechanism, tx_vox,
                    pol="te", p_incoh=None, tau_rms=None, meta=None) -> "FieldGrid":
        """Build the complex field from a path-loss volume and an arrival-time volume.

        This is the bridge from the existing engine: magnitude from PL (CrossingLUT),
        phase from tau (eikonal). `tau_s` may be (NF,...) or a single (...) broadcast
        across bands.
        """
        pl = np.asarray(pl_db, np.float32)
        f = np.asarray(freqs_mhz, float)
        nf = pl.shape[0]
        tau = np.asarray(tau_s, np.float32)
        if tau.ndim == pl.ndim - 1:
            tau = np.broadcast_to(tau, pl.shape).copy()

        amp = np.power(10.0, -pl / 20.0).astype(np.float32)     # |E|
        # exp(-j 2 pi f tau); tau may be inf where unreachable -> zero field there.
        ph = np.zeros(pl.shape, np.float32)
        good = np.isfinite(tau) & np.isfinite(pl)
        w = (2.0 * np.pi * (f * 1e6)).astype(np.float64)        # rad/s, per band
        for i in range(nf):
            gi = good[i]
            ph[i][gi] = np.float32(np.mod(w[i] * tau[i][gi].astype(np.float64), 2 * np.pi))
        amp = np.where(good, amp, 0.0).astype(np.float32)

        E = np.zeros((NPOL,) + pl.shape, np.complex64)
        E[POL_NAMES.index(pol)] = (amp * np.exp(-1j * ph)).astype(np.complex64)
        return cls(E=E, tau_first=np.where(good, tau, np.inf).astype(np.float32),
                   mechanism=mechanism, freqs_mhz=f, tx_vox=tuple(int(v) for v in tx_vox),
                   tau_rms=tau_rms, p_incoh=p_incoh, meta=meta or {})

    # ---- views ------------------------------------------------------------
    @property
    def shape(self):
        return self.E.shape[2:]

    @property
    def n_bands(self):
        return self.E.shape[1]

    def power(self) -> np.ndarray:
        """Linear power (NF, NX, NY, NZ), float64: coherent |E|^2 summed over
        polarization, plus any incoherent part.

        float64 is REQUIRED, not defensive. Deep-shadow voxels in the production scene
        reach ~540 dB, so |E| ~ 1e-27 and |E|^2 ~ 1e-54 -- which underflows float32
        (min ~1e-38) to exactly zero and reports PL = +inf. The dB value itself is
        perfectly representable; only the linear intermediate is not.
        """
        p = np.sum(np.abs(self.E).astype(np.float64) ** 2, axis=0)
        if self.p_incoh is not None:
            p = p + np.asarray(self.p_incoh, np.float64)
        return p

    def pathloss_db(self) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return (-10.0 * np.log10(np.maximum(self.power(), 1e-300))).astype(np.float32)

    def __repr__(self):
        return (f"FieldGrid({self.mechanism!r}, bands={self.n_bands}, grid={self.shape}, "
                f"tx={self.tx_vox}, combine_as={self.combine_as})")


# --------------------------------------------------------------------------- Combined
@dataclass
class CombinedField:
    """Result of coherently+incoherently summing several FieldGrids."""
    E_total: np.ndarray            # complex64 (NPOL, NF, NX, NY, NZ)
    P_lin: np.ndarray              # float32   (NF, NX, NY, NZ)  total linear power
    PL_db: np.ndarray              # float32   (NF, NX, NY, NZ)
    T_first: np.ndarray            # float32   (NX, NY, NZ)      seconds
    per_mech_lin: dict             # mechanism -> float32 (NF, NX, NY, NZ) linear power
    freqs_mhz: np.ndarray
    tx_vox: tuple
    tau_rms: np.ndarray | None = None
    k_factor_db: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def contribution_fraction(self, mechanism: str) -> np.ndarray:
        """Share of total power from one mechanism (0..1) — the frontend colour scale
        for the five non-path-loss mechanisms, since 'reflection in dB' alone is not
        interpretable."""
        num = self.per_mech_lin[mechanism]
        return (num / np.maximum(self.P_lin, 1e-300)).astype(np.float32)

    def __repr__(self):
        return (f"CombinedField(mechs={list(self.per_mech_lin)}, "
                f"bands={len(self.freqs_mhz)}, grid={self.PL_db.shape[1:]})")


# --------------------------------------------------------------------------- PathSet
@dataclass
class PathSet:
    """Optional per-receiver multipath list (CIR / PDP). Gated behind a flag — a per-voxel
    top-K path list is ~43 MB/band at K=8, fine on an A100 and borderline locally."""
    amp: np.ndarray                # complex64 (NPOL, NF, N_RX, K)
    tau: np.ndarray                # float32   (NF, N_RX, K) seconds
    mech_id: np.ndarray            # int8      (N_RX, K)
    rx_vox: np.ndarray             # int16     (N_RX, 3)
    mech_names: Sequence[str]
    freqs_mhz: np.ndarray
