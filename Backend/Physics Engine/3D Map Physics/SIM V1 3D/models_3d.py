#!/usr/bin/env python3
"""Importable neural network modules for the 3-D surrogate.

The Colab trainer keeps the training loop in the notebook, but the architecture
needs to exist in normal Python for smoke exports, tests, and future CLI tooling.
This is the same UNet3D used by `phase_c3_train_colab.ipynb`; smoke exports pass
a smaller `base` width while preserving the `(N,9,X,Y,Z) -> (N,2,X,Y,Z)` contract.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised only before ML deps install
    raise ImportError("models_3d requires torch; install the CPU wheel for smoke export") from exc


def dconv(ci: int, co: int) -> nn.Sequential:
    """Two Conv3d-BatchNorm-ReLU blocks, matching the trainer notebook."""
    return nn.Sequential(
        nn.Conv3d(ci, co, 3, padding=1, bias=False),
        nn.BatchNorm3d(co),
        nn.ReLU(inplace=True),
        nn.Conv3d(co, co, 3, padding=1, bias=False),
        nn.BatchNorm3d(co),
        nn.ReLU(inplace=True),
    )


class UNet3D(nn.Module):
    """Compact 3-level 3-D UNet used for PL/tau surrogate inference."""

    def __init__(self, cin: int = 9, cout: int = 2, base: int = 24):
        super().__init__()
        self.pool = nn.MaxPool3d((2, 1, 2))
        self.e1 = dconv(cin, base)
        self.e2 = dconv(base, base * 2)
        self.e3 = dconv(base * 2, base * 4)
        self.b = dconv(base * 4, base * 8)
        self.d3 = dconv(base * 8 + base * 4, base * 4)
        self.d2 = dconv(base * 4 + base * 2, base * 2)
        self.d1 = dconv(base * 2 + base, base)
        self.out = nn.Conv3d(base, cout, 1)

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return torch.cat([x, skip], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        d3 = self.d3(self._up(b, e3))
        d2 = self.d2(self._up(d3, e2))
        d1 = self.d1(self._up(d2, e1))
        return torch.sigmoid(self.out(d1))
