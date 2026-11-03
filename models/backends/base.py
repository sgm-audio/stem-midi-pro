# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Backend contracts for stem separation and MIDI transcription."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, TypedDict, runtime_checkable

import torch


class SeparateOut(TypedDict):
    """Separator output tensors. Audio shapes are (B, 1, T)."""

    guitar: torch.Tensor
    bass: torch.Tensor
    residual: torch.Tensor
    # (B, 2) — [quality_proxy, phase_coherence]; quality_proxy fills ProcessingReport.si_sdr
    metrics: torch.Tensor


class TranscribeOut(TypedDict, total=False):
    """
    Transcriber output.

    Either provide frame-level logits (Mamba path) *or* pre-built note events
    (Basic Pitch path). StemMidiModel prefers ``events`` when present.
    """

    onset_logits: torch.Tensor  # (B, T_frames, 1)
    pitch_logits: torch.Tensor  # (B, T_frames, 128)
    velocity: torch.Tensor  # (B, T_frames, 1) in [0, 1]
    expression: torch.Tensor  # (B, T_frames, E)
    confidence: torch.Tensor  # (B, T_frames)
    # Optional: list-per-batch of event dicts {note, onset_frame, velocity, confidence, ...}
    events: List[List[Dict[str, Any]]]


@runtime_checkable
class SeparatorBackend(Protocol):
    def separate(self, audio: torch.Tensor) -> SeparateOut:
        """Separate mono mix ``audio`` (B, 1, T) into guitar/bass/residual."""
        ...


@runtime_checkable
class TranscriberBackend(Protocol):
    def transcribe(self, stem: torch.Tensor) -> TranscribeOut:
        """Transcribe mono ``stem`` (B, 1, T) to logits and/or note events."""
        ...


def empty_metrics(batch: int, device: torch.device) -> torch.Tensor:
    """Default metrics tensor when a backend has no quality estimate."""
    return torch.zeros(batch, 2, device=device)


def device_from_cfg(cfg: dict) -> torch.device:
    """Resolve inference device from config / env."""
    import os

    name = (
        (cfg.get("backends") or {}).get("device")
        or os.getenv("INFERENCE_DEVICE")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    return torch.device(name)
