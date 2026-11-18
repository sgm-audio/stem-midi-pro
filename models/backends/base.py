# SPDX-License-Identifier: Apache-2.0
"""Backend contracts for stem separation and MIDI transcription."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict, runtime_checkable

import torch


class SeparateOut(TypedDict):
    """Separator output tensors. Audio shapes are (B, 1, T)."""

    guitar: torch.Tensor
    bass: torch.Tensor
    residual: torch.Tensor
    # (B, 2) — [separation_proxy, phase_coherence]. The first value is a
    # heuristic score, not reference-based SI-SDR or a dB certification.
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
    events: list[list[dict[str, Any]]]


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
    """Resolve the inference device from config / env, defaulting to CPU.

    Auto-selecting CUDA whenever a GPU happens to be visible contradicted the
    project's stated CPU-only target (and started a model on a device the rest
    of the pipeline was not sized for). A device is used only when it is asked
    for explicitly; a request that cannot be honoured raises instead of silently
    falling back, because a silent fallback is how "it ran on CPU" surprises
    show up in production.
    """
    import logging
    import os

    name = (cfg.get("backends") or {}).get("device") or os.getenv("INFERENCE_DEVICE") or "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device {device} was requested but CUDA is not available. "
            "Set backends.device / INFERENCE_DEVICE to 'cpu'."
        )
    if device.type != "cpu":
        logging.getLogger(__name__).warning(
            "Inference device %s requested; this project is certified on CPU only.", device
        )
    return device
