# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Deterministic fake backends for unit tests (no mamba_ssm / demucs / basic-pitch)."""
from __future__ import annotations

import torch

from models.backends.base import SeparateOut, TranscribeOut, empty_metrics


class FakeSeparatorBackend:
    """Splits mix into high/low filtered proxies (not musical — shape-correct only)."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def separate(self, audio: torch.Tensor) -> SeparateOut:
        B, C, T = audio.shape
        # Crude spectral split via simple gain so guitar != bass
        guitar = audio * 0.6
        bass = audio * 0.4
        residual = audio - guitar - bass
        metrics = empty_metrics(B, audio.device)
        metrics[:, 0] = 12.0  # fake SI-SDR proxy
        metrics[:, 1] = 0.5
        return {
            "guitar": guitar,
            "bass": bass,
            "residual": residual,
            "metrics": metrics,
        }


class FakeTranscriberBackend:
    """Emits a single middle-C event at frame 10 for non-empty audio."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        audio_cfg = (cfg or {}).get("audio") or {}
        tr_cfg = (cfg or {}).get("transcriber") or {}
        self.hop = int(audio_cfg.get("hop_length", 512))
        self.n_pitch = int(tr_cfg.get("pitch_vocab_size", 128))

    def set_instrument(self, name: str) -> None:
        pass

    def transcribe(self, stem: torch.Tensor) -> TranscribeOut:
        B, _, T = stem.shape
        n_frames = max(1, T // self.hop)
        onset = torch.full((B, n_frames, 1), -10.0, device=stem.device)
        pitch = torch.zeros(B, n_frames, self.n_pitch, device=stem.device)
        vel = torch.zeros(B, n_frames, 1, device=stem.device)
        expr = torch.zeros(B, n_frames, 3, device=stem.device)
        conf = torch.zeros(B, n_frames, device=stem.device)

        frame = min(10, n_frames - 1)
        note = 60
        events = []
        for b in range(B):
            onset[b, frame, 0] = 10.0
            pitch[b, frame, note] = 10.0
            vel[b, frame, 0] = 0.8
            conf[b, frame] = 0.9
            events.append(
                [
                    {
                        "note": note,
                        "onset_frame": frame,
                        "duration_frames": 4,
                        "velocity": 100,
                        "confidence": 0.9,
                    }
                ]
            )

        return {
            "onset_logits": onset,
            "pitch_logits": pitch,
            "velocity": vel,
            "expression": expr,
            "confidence": conf,
            "events": events,
        }
