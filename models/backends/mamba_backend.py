# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Mamba-SSM backends (requires mamba_ssm)."""
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
