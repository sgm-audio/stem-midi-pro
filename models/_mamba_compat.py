# SPDX-License-Identifier: Apache-2.0
"""
Compatibility shim for the Mamba block.

Uses the official ``mamba-ssm`` package when it is importable (CUDA-only;
fails to build on CPU/ROCm). Otherwise falls back to a pure-PyTorch Mamba
implementation so the models run (slower) on CPU-only targets - this is the
bare-metal i5 path (D-STRAT-3). The fallback keeps the same constructor
signature: ``Mamba(d_model, d_state=16, d_conv=4, expand=2)`` and maps
``(B, L, d_model) -> (B, L, d_model)``.

Strict mode: set ``STEM_MIDI_REQUIRE_MAMBA_SSM=1`` in the environment to make
a missing ``mamba_ssm`` package a hard import error instead of a silent
fallback (M-7). ``MAMBA_BACKEND`` records which implementation is live and is
surfaced in logs, ``/model-info``, and the processing report.

Do NOT re-add CUDA-only kwargs here; see C-2.1 in TODO.md.

Known deviations of the pure-torch fallback from upstream mamba-ssm (documented,
not hidden): no fused selective-scan kernel (sequential scan, O(L) python loop),
no fast path, and no Triton layer-norm. The special ``dt_proj`` bias
initialization IS ported from upstream so discretization steps match
``softplus(bias) ∈ [dt_min, dt_max]`` at init (H-2).
"""

import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:  # pragma: no cover - depends on environment
    from mamba_ssm import Mamba  # type: ignore

    MAMBA_BACKEND = "mamba-ssm"
except ImportError:  # CPU / ROCm / no-CUDA environments
    if os.environ.get("STEM_MIDI_REQUIRE_MAMBA_SSM", "").strip() in {"1", "true", "yes"}:
        raise RuntimeError(
            "STEM_MIDI_REQUIRE_MAMBA_SSM=1 but the 'mamba_ssm' package is not "
            "installed. Install it on a CUDA machine with: pip install mamba-ssm "
            "(it requires CUDA/nvcc to build), or unset the variable to use the "
            "pure-PyTorch fallback."
        ) from None

    MAMBA_BACKEND = "pure-torch"
    logger.warning(
        "mamba-ssm not importable; using the pure-PyTorch Mamba fallback "
        "(models/_mamba_compat.py). This is the CPU path: correct, but slower "
        "than the upstream CUDA kernels."
    )

    class Mamba(nn.Module):
        """Pure-PyTorch Mamba-1 (selective scan), API-compatible subset.

        Implements the standard Mamba block (input projection, depthwise
        causal conv, input-dependent dt/B/C, discretized selective scan,
        gated output).

        Memory notes:
          * The selective scan runs in chunks of ``scan_chunk`` frames and
            never materializes the full ``(B, L, d_inner, N)`` discretization
            tensors, bounding peak allocation to O(scan_chunk) per block
            (C-1). Previously a 10-minute song needed >10 GB; now the scan
            temporaries stay in the tens of MB.
          * If ``gradient_checkpointing=True`` (plumbed from
            ``cfg["training"]["gradient_checkpointing"]``), each block is
            recomputed on backward via ``torch.utils.checkpoint``, bounding
            *training* memory to one block of activations.
        """

        # Frames per scan chunk. Bounded temporaries: chunk * d_inner * N * 4 B.
        SCAN_CHUNK = 256

        def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            gradient_checkpointing: bool = False,
            **_ignored,
        ):
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            self.d_conv = d_conv
            self.expand = expand
            self.d_inner = expand * d_model
            self.dt_rank = math.ceil(d_model / 16)
            self.gradient_checkpointing = gradient_checkpointing

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

            # A (S4D-real init) and D skip connection
            A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(self.d_inner))

            # dt bias init ported from upstream mamba_ssm (H-2): guarantees
            # softplus(dt_proj.bias) is spread across [dt_min, dt_max] at init,
            # so the discretized decay exp(dt * A) has a sane memory horizon.
            dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
            with torch.no_grad():
                self.dt_proj.bias.copy_(inv_dt)

            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """(B, L, D) -> (B, L, D)"""
            if self.gradient_checkpointing and self.training and x.requires_grad:
                return torch.utils.checkpoint.checkpoint(
                    self._block_forward, x, use_reentrant=False
                )
            return self._block_forward(x)

        def _block_forward(self, x: torch.Tensor) -> torch.Tensor:
            _, L, _ = x.shape
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

            y = self._selective_scan(xc, dt, Bmat, Cmat)  # (B, L, d_inner)

            y = y + self.D.unsqueeze(0).unsqueeze(0) * xc  # skip connection
            y = y * F.silu(z)  # gate
            return self.out_proj(y)

        def _selective_scan(
            self, xc: torch.Tensor, dt: torch.Tensor, Bmat: torch.Tensor, Cmat: torch.Tensor
        ) -> torch.Tensor:
            """Chunked sequential selective scan.

            The recurrence h_t = exp(dt_t * A) * h_{t-1} + dt_t * x_t * B_t is
            inherently sequential; chunking bounds the size of the
            materialized discretization tensors to (B, chunk, d_inner, N)
            instead of (B, L, d_inner, N). C-1.

            Each step is a single fused ``torch.addcmul(..., out=state_history[t])``
            into a preallocated buffer - zero per-step allocation. The naive
            version allocated ~3 tensors per frame, and CPU allocator churn
            inflated peak RSS severalx above what the math requires.
            """
            B, L, _ = xc.shape
            A = -torch.exp(self.A_log)  # (d_inner, N), all entries < 0

            # State carries ACROSS chunks - it is initialized exactly once
            # here, never inside the chunk loop (a per-chunk reset silently
            # truncates the SSM memory to SCAN_CHUNK frames).
            h = xc.new_zeros(B, self.d_inner, self.d_state)
            ys: list[torch.Tensor] = []
            chunk = self.SCAN_CHUNK
            for start in range(0, L, chunk):
                end = min(start + chunk, L)
                seg = end - start
                dt_c = dt[:, start:end]  # (B, l, d_inner)
                dA_c = torch.exp(dt_c.unsqueeze(-1) * A)  # (B, l, d_inner, N)
                dBx_c = (
                    dt_c.unsqueeze(-1)
                    * xc[:, start:end].unsqueeze(-1)
                    * Bmat[:, start:end].unsqueeze(2)
                )  # (B, l, d_inner, N)

                if torch.is_grad_enabled():
                    # Autograd path: out= ops don't support autograd, so chain
                    # out-of-place (2 allocs/frame); training memory is bounded
                    # by gradient checkpointing, not this loop.
                    hs_list: list[torch.Tensor] = []
                    for t in range(seg):
                        h = torch.addcmul(dBx_c[:, t], dA_c[:, t], h)
                        hs_list.append(h)
                    hs = torch.stack(hs_list, dim=1)
                else:
                    # Inference path: preallocated state history, zero per-step
                    # allocation.
                    hs = xc.new_zeros(B, seg, self.d_inner, self.d_state)
                    for t in range(seg):
                        # h_t = dBx_t + dA_t * h_{t-1}, written to its own row
                        h = torch.addcmul(dBx_c[:, t], dA_c[:, t], h, out=hs[:, t])

                # y_t = <h_t, C_t> for the whole chunk in one einsum
                ys.append(torch.einsum("bijn,bin->bij", hs, Cmat[:, start:end]))

            return torch.cat(ys, dim=1)  # (B, L, d_inner)
