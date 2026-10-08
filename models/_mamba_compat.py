# SPDX-License-Identifier: Apache-2.0
"""
Compatibility shim for the Mamba block.

Uses the official ``mamba-ssm`` package when it is importable (CUDA-only;
fails to build on CPU/ROCm). Otherwise falls back to a pure-PyTorch Mamba
implementation so the models run (slower) on CPU-only targets — this is the
bare-metal i5 path (D-STRAT-3). The fallback keeps the same constructor
signature: ``Mamba(d_model, d_state=16, d_conv=4, expand=2)`` and maps
``(B, L, d_model) -> (B, L, d_model)``.

Do NOT re-add CUDA-only kwargs here; see C-2.1 in TODO.md.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - depends on environment
    from mamba_ssm import Mamba  # type: ignore

    MAMBA_BACKEND = "mamba-ssm"
except ImportError:  # CPU / ROCm / no-CUDA environments

    MAMBA_BACKEND = "pure-torch"

    class Mamba(nn.Module):
        """Pure-PyTorch Mamba-1 (selective scan), API-compatible subset.

        Implements the standard Mamba block (input projection, depthwise
        causal conv, input-dependent dt/B/C, discretized selective scan,
        gated output). A sequential scan over time in Python ops — correct but
        O(L) loop; fine for the short frame sequences produced by the STFT
        front-ends here, slow for very long sequences.
        """

        def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            **_ignored,
        ):
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            self.d_conv = d_conv
            self.expand = expand
            self.d_inner = expand * d_model
            self.dt_rank = math.ceil(d_model / 16)

            self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
            self.conv1d = nn.Conv1d(
                self.d_inner,
                self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
            )
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)

            # A (HiPPO init) and D skip connection
            A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(self.d_inner))

            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """(B, L, D) -> (B, L, D)"""
            B, L, _ = x.shape
            x_and_z = self.in_proj(x)  # (B, L, 2*d_inner)
            x_in, z = x_and_z.chunk(2, dim=-1)  # (B, L, d_inner) each

            # Causal depthwise conv + SiLU
            xc = self.conv1d(x_in.transpose(1, 2))[..., :L]  # (B, d_inner, L)
            xc = F.silu(xc).transpose(1, 2)  # (B, L, d_inner)

            x_dbl = self.x_proj(xc)  # (B, L, dt_rank + 2*N)
            dt_raw, Bmat, Cmat = torch.split(
                x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt = F.softplus(self.dt_proj(dt_raw))  # (B, L, d_inner)
            A = -torch.exp(self.A_log)  # (d_inner, N)

            # Discretize: dA = exp(dt * A), dB*x for the scan
            dA = torch.exp(dt.unsqueeze(-1) * A)  # (B, L, d_inner, N)
            dBx = dt.unsqueeze(-1) * xc.unsqueeze(-1) * Bmat.unsqueeze(2)  # (B, L, d_inner, N)

            # Sequential selective scan (reference implementation)
            h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
            ys = []
            for t in range(L):
                h = dA[:, t] * h + dBx[:, t]
                ys.append(torch.einsum("bdn,bn->bd", h, Cmat[:, t]))
            y = torch.stack(ys, dim=1)  # (B, L, d_inner)

            y = y + self.D.unsqueeze(0).unsqueeze(0) * xc  # skip connection
            y = y * F.silu(z)  # gate
            return self.out_proj(y)
