# SPDX-License-Identifier: Apache-2.0
"""Mamba backends using the native module and its CPU fallback.

These wrappers do not require the optional ``mamba_ssm`` CUDA package on the
CPU target; the pure-PyTorch fallback in ``models/_mamba_compat.py`` is used
when it is unavailable. Native Mamba weights still require a project
checkpoint before file/API inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.backends.base import SeparateOut, TranscribeOut


class MambaSeparatorBackend(nn.Module):
    """Wraps :class:`MambaSeparator` in the backend protocol."""

    def __init__(self, cfg: dict):
        super().__init__()
        from models.mamba_separator import MambaSeparator

        self.module = MambaSeparator(cfg)
        self.weights_loaded = False

    def separate(self, audio: torch.Tensor) -> SeparateOut:
        guitar, bass, residual, _cache, metrics = self.module(audio)
        return {
            "guitar": guitar,
            "bass": bass,
            "residual": residual,
            "metrics": metrics,
        }

    def forward(self, audio: torch.Tensor) -> SeparateOut:
        return self.separate(audio)


class MambaTranscriberBackend(nn.Module):
    """Wraps :class:`MambaTranscriber` in the backend protocol."""

    def __init__(self, cfg: dict):
        super().__init__()
        from models.mamba_transcriber import MambaTranscriber

        self.module = MambaTranscriber(cfg)
        self.weights_loaded = False

    def set_instrument(self, name: str) -> None:
        pass

    def transcribe(self, stem: torch.Tensor) -> TranscribeOut:
        onset, pitch, velocity, expression, confidence = self.module(stem)
        return {
            "onset_logits": onset,
            "pitch_logits": pitch,
            "velocity": velocity,
            "expression": expression,
            "confidence": confidence,
        }

    def forward(self, stem: torch.Tensor) -> TranscribeOut:
        return self.transcribe(stem)
