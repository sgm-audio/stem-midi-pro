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
import torch.utils.checkpoint  # must follow `import torch`

# `import torch` alone does NOT bind torch.utils.checkpoint: the submodule only
# appears on the parent package once something imports it. MambaBackbone.forward
# (CFG-4) calls torch.utils.checkpoint.checkpoint(...) under
# training.gradient_checkpointing, so importing it here is what makes the
# documented CPU memory mitigation reachable at all.

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

        # Parameters the real mamba-ssm ``Mamba.__init__`` accepts (see
        # research/mamba-ssm-reference/mamba/mamba_ssm/modules/mamba_simple.py).
        # This set is what the fallback accepts too -- anything else is a
        # TypeError, exactly as on a CUDA box (SHIM-1).
        _UPSTREAM_KWARGS = frozenset(
            {
                "d_state",
                "d_conv",
                "expand",
                "dt_rank",
                "dt_min",
                "dt_max",
                "dt_init",
                "dt_scale",
                "dt_init_floor",
                "conv_bias",
                "bias",
                "use_fast_path",
                "layer_idx",
                "device",
                "dtype",
            }
        )

        def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            dt_rank: str | int = "auto",
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init: str = "random",
            dt_scale: float = 1.0,
            dt_init_floor: float = 1e-4,
            conv_bias: bool = True,
            bias: bool = False,
            use_fast_path: bool = True,
            layer_idx: int | None = None,
            device=None,
            dtype=None,
            gradient_checkpointing: bool = False,
            **kwargs,
        ):
            if kwargs:
                unknown = ", ".join(sorted(kwargs))
                raise TypeError(
                    f"Mamba.__init__() got unexpected keyword argument(s): {unknown}. "
                    "Accepted: "
                    + ", ".join(sorted(self._UPSTREAM_KWARGS | {"gradient_checkpointing"}))
                )
            # Accepted for signature compatibility; the pure-PyTorch path has no
            # fused kernels, so these are recorded and unused. `dt_max` is the
            # exception: it clamps the discretisation step below, because an
            # unclamped dt turns the recurrence into a pure integrator.
            self.use_fast_path = use_fast_path
            self.layer_idx = layer_idx
            super().__init__()
            if not 0 < dt_max <= 1.0:
                raise ValueError(f"dt_max must be in (0, 1], got {dt_max!r}")
            self.d_model = d_model
            self.d_state = d_state
            self.d_conv = d_conv
            self.expand = expand
            self.d_inner = expand * d_model
            self.dt_rank = math.ceil(d_model / 16)
            self.gradient_checkpointing = gradient_checkpointing
            # Runtime ceiling on the discretisation step. The recurrence is
            # stable for any dt > 0 (dA = exp(dt*A) < 1), but a large dt makes
            # the state saturate to a pure integrator of the input, which is how
            # a bounded-input model can still produce a huge DC gain. The init
            # already lands in [dt_min, dt_max]; this also enforces it when a
            # data path drives large dt inputs (see AUDIT_REPORT.md, "dt
            # clamping").
            self.dt_max = float(dt_max)

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

        def forward(self, x: torch.Tensor, state: dict | None = None, return_state: bool = False):
            """(B, L, D) -> (B, L, D), optionally carrying SSM state across calls.

            ``state``/``return_state`` implement the streaming path (C-2.5): a
            caller that feeds consecutive pieces of one sequence can thread the
            returned state into the next call and get the same result as one
            long call, instead of restarting the recurrence at every boundary.
            The state is ``{"h": (B, d_inner, d_state), "conv": (B, d_conv-1, d_inner)}``
            — the SSM accumulator plus the causal convolution's history.
            """
            if state is None and self.gradient_checkpointing and self.training and x.requires_grad:
                out = torch.utils.checkpoint.checkpoint(self._block_forward, x, use_reentrant=False)
                return (out, self.initial_state(x)) if return_state else out
            out, new_state = self._block_forward(x, state=state, return_state=True)
            return (out, new_state) if return_state else out

        def initial_state(self, x: torch.Tensor) -> dict:
            """Zero state for the first call of a streaming session (C-2.5)."""
            B = x.shape[0]
            return {
                "h": torch.zeros(
                    B, self.d_inner, self.d_state, device=x.device, dtype=torch.float32
                ),
                "conv": torch.zeros(
                    B, self.d_conv - 1, self.d_inner, device=x.device, dtype=x.dtype
                ),
            }

        def _block_forward(
            self,
            x: torch.Tensor,
            state: dict | None = None,
            return_state: bool = False,
        ):
            """(B, L, D) -> (B, L, D).

            Fused chunking: projections, the causal conv, the scan and the
            gated output all run in windows of ``SCAN_CHUNK`` frames, so no
            full-length ``(B, L, 2*d_inner)`` tensor (or the ``(B, d_inner, L)``
            conv im2col) is ever alive. Those full-length allocations were the
            dominant memory cost on long files — 121 MB for ``in_proj`` alone
            at L=10,336, roughly doubling transiently inside ``conv1d``
            (review §1).

            Equivalence to the whole-sequence form: the causal depthwise conv
            is windowed with ``d_conv - 1`` frames of history prepended (a
            depthwise kernel reaches back exactly that far), and the SSM state
            ``h`` is threaded through every window rather than reset.
            """
            B, L, _ = x.shape
            hist = self.d_conv - 1
            out = torch.empty(B, L, self.d_model, device=x.device, dtype=x.dtype)
            A = -torch.exp(self.A_log)  # (d_inner, N), negative => contraction
            if state is None:
                h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=torch.float32)
                # Convolution history in *pre-conv* space: the depthwise kernel
                # reaches back exactly d_conv-1 frames, and zeros reproduce the
                # left padding the whole-sequence form uses at the start.
                conv_tail = torch.zeros(B, hist, self.d_inner, device=x.device, dtype=x.dtype)
            else:
                h = state["h"].to(device=x.device, dtype=torch.float32)
                conv_tail = state["conv"].to(device=x.device, dtype=x.dtype)

            for start in range(0, L, self.SCAN_CHUNK):
                stop = min(start + self.SCAN_CHUNK, L)

                # --- projections over this window only -------------------
                xz = self.in_proj(x[:, start:stop])  # (B, win, 2*d_inner)
                x_in, z = xz.chunk(2, dim=-1)

                # --- causal depthwise conv, fed with the carried history ---
                # Equivalent to convolving the whole sequence: the previous
                # window's last d_conv-1 frames ARE this window's lookback.
                conv_in = torch.cat([conv_tail, x_in], dim=1)  # (B, hist+win, d_inner)
                conv_tail = conv_in[:, -hist:, :] if hist else conv_in[:, :0, :]
                xc = F.conv1d(
                    conv_in.transpose(1, 2),
                    self.conv1d.weight,
                    self.conv1d.bias,
                    groups=self.d_inner,
                )  # (B, d_inner, win)
                xc = F.silu(xc).transpose(1, 2)  # (B, win, d_inner)

                x_dbl = self.x_proj(xc)  # (B, win, dt_rank + 2*N)
                dt_raw, Bmat, Cmat = torch.split(
                    x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
                )
                # dt > 0 and A < 0 => dA = exp(dt*A) < 1, so the recurrence
                # always forgets. The remaining risk (a large DC gain on a
                # *constant* input) is removed by the per-block LayerNorm in
                # the models plus the bounded log-mel input; see review §3.
                dt = F.softplus(self.dt_proj(dt_raw))
                if self.dt_max < float("inf"):
                    dt = dt.clamp(max=self.dt_max)

                y, h = self._selective_scan(xc, dt, Bmat, Cmat, A=A, h=h, return_state=True)
                y = y + self.D.unsqueeze(0).unsqueeze(0) * xc  # skip connection
                y = y * F.silu(z)  # gate
                out[:, start:stop] = self.out_proj(y)

            if return_state:
                return out, {"h": h, "conv": conv_tail}
            return out

        def _selective_scan(
            self,
            xc: torch.Tensor,
            dt: torch.Tensor,
            Bmat: torch.Tensor,
            Cmat: torch.Tensor,
            A: torch.Tensor | None = None,
            h: torch.Tensor | None = None,
            return_state: bool = False,
        ):
            """Selective scan over one window; returns (y, carried state).

            h(t) = exp(dt_t * A) * h(t-1) + dt_t * xc_t * B_t
            y(t) = h(t) . C_t

            The state is threaded in and out so callers can window the
            sequence without breaking the recurrence. Peak transient memory is
            O(window), never O(L).

            Signature compatibility: `A` defaults to ``-exp(A_log)`` and the
            carried state is optional, so the 4-argument positional form used
            elsewhere (and by the test-suite) keeps working; pass
            ``return_state=True`` to also get ``h`` back.
            """
            if A is None:
                A = -torch.exp(self.A_log)  # (d_inner, N), negative => contraction
            B, L, _ = xc.shape
            if h is None:
                h = torch.zeros(
                    B, self.d_inner, self.d_state, device=xc.device, dtype=torch.float32
                )
            if xc.dtype != torch.float32:  # `.float()` is a no-op copy otherwise
                xc, dt, Bmat, Cmat = (t.float() for t in (xc, dt, Bmat, Cmat))

            outs = []
            for start in range(0, L, self.SCAN_CHUNK):
                stop = min(start + self.SCAN_CHUNK, L)
                dt_c = dt[:, start:stop]
                xc_c = xc[:, start:stop]
                B_c = Bmat[:, start:stop]
                C_c = Cmat[:, start:stop]

                # (B, c, d_inner, N) — bounded by _SCAN_CHUNK, not by L.
                dA = torch.exp(dt_c.unsqueeze(-1) * A)
                dBx = dt_c.unsqueeze(-1) * xc_c.unsqueeze(-1) * B_c.unsqueeze(2)

                y_c = torch.empty(
                    B, stop - start, self.d_inner, device=xc.device, dtype=torch.float32
                )
                for t in range(stop - start):
                    h = dA[:, t] * h + dBx[:, t]
                    y_c[:, t] = torch.einsum("bdn,bn->bd", h, C_c[:, t])
                outs.append(y_c)

            y = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
            y = y.to(xc.dtype)
            return (y, h) if return_state else y
