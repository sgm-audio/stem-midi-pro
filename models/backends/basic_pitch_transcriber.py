# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Basic Pitch (Spotify) transcription backend."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from models.backends.base import TranscribeOut

log = logging.getLogger(__name__)


class BasicPitchTranscriberBackend:
    """
    Transcribe a mono stem with Spotify Basic Pitch.

    Returns pre-built ``events`` (preferred by StemMidiModel) plus sparse
    frame tensors for loss/compat paths.
    """

    def __init__(self, cfg: dict):
        try:
            from basic_pitch.inference import predict  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "basic-pitch is required for transcriber backend 'basic_pitch'. "
                "Install with: pip install 'stem-midi-pro[pretrained]' "
                "or pip install basic-pitch"
            ) from e

        self.cfg = cfg
        audio_cfg = cfg.get("audio") or {}
        tr_cfg = cfg.get("transcriber") or {}
        self.sample_rate = int(audio_cfg.get("sample_rate", 44100))
        self.hop_length = int(audio_cfg.get("hop_length", 512))
        self.pitch_vocab = int(tr_cfg.get("pitch_vocab_size", 128))
        # Guitar-ish band by default; bass uses lower max via instrument hint
        self.min_freq = tr_cfg.get("basic_pitch_min_freq")  # Hz or None
        self.max_freq = tr_cfg.get("basic_pitch_max_freq")
        self._instrument = "guitar"  # overridden per call via set_instrument

    def set_instrument(self, name: str) -> None:
        """Hint frequency range: 'guitar' or 'bass'."""
        self._instrument = name

    def transcribe(self, stem: torch.Tensor) -> TranscribeOut:
        if stem.dim() != 3 or stem.shape[1] != 1:
            raise ValueError(f"Expected stem (B, 1, T), got {tuple(stem.shape)}")

        from basic_pitch.inference import predict

        B, _, T = stem.shape
        n_frames = max(1, T // self.hop_length)
        min_f, max_f = self._freq_range()

        all_events: List[List[Dict[str, Any]]] = []
        onset_logits = torch.full((B, n_frames, 1), -10.0, device=stem.device)
        pitch_logits = torch.zeros(B, n_frames, self.pitch_vocab, device=stem.device)
        velocity = torch.zeros(B, n_frames, 1, device=stem.device)
        expression = torch.zeros(B, n_frames, 3, device=stem.device)
        confidence = torch.zeros(B, n_frames, device=stem.device)

        for b in range(B):
            mono = stem[b, 0].detach().cpu().numpy().astype(np.float32)
            events_b = self._predict_file(predict, mono, min_f, max_f)
            all_events.append(events_b)

            for ev in events_b:
                f = int(ev["onset_frame"])
                if f < 0 or f >= n_frames:
                    continue
                note = int(ev["note"])
                conf = float(ev["confidence"])
                vel = float(ev["velocity"]) / 127.0
                onset_logits[b, f, 0] = 10.0  # strong positive logit
                if 0 <= note < self.pitch_vocab:
                    pitch_logits[b, f, :] = -10.0
                    pitch_logits[b, f, note] = 10.0
                velocity[b, f, 0] = vel
                confidence[b, f] = conf

        return {
            "onset_logits": onset_logits,
            "pitch_logits": pitch_logits,
            "velocity": velocity,
            "expression": expression,
            "confidence": confidence,
            "events": all_events,
        }

    def _freq_range(self):
        if self.min_freq is not None or self.max_freq is not None:
            return self.min_freq, self.max_freq
        if self._instrument == "bass":
            return 30.0, 400.0
        # guitar
        return 70.0, 1200.0

    def _predict_file(self, predict, mono: np.ndarray, min_f, max_f) -> List[Dict[str, Any]]:
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
            sf.write(path, mono, self.sample_rate)

        try:
            kwargs = {}
            if min_f is not None:
                kwargs["minimum_frequency"] = float(min_f)
            if max_f is not None:
                kwargs["maximum_frequency"] = float(max_f)
            _model_out, _midi, note_events = predict(path, **kwargs)
        finally:
            Path(path).unlink(missing_ok=True)

        return self._notes_to_events(note_events)

    def _notes_to_events(self, note_events) -> List[Dict[str, Any]]:
        """
        Basic Pitch note_events: list of
        (start_time_s, end_time_s, pitch_midi, amplitude, ...)
        """
        events: List[Dict[str, Any]] = []
        for item in note_events or []:
            try:
                start_s = float(item[0])
                end_s = float(item[1])
                pitch = int(item[2])
                amp = float(item[3]) if len(item) > 3 else 0.8
            except (TypeError, ValueError, IndexError):
                continue
            onset_frame = int(round(start_s * self.sample_rate / self.hop_length))
            duration_frames = max(
                1, int(round((end_s - start_s) * self.sample_rate / self.hop_length))
            )
            vel = int(np.clip(amp * 127.0, 1, 127))
            conf = float(np.clip(amp, 0.0, 1.0))
            events.append(
                {
                    "note": pitch,
                    "onset_frame": onset_frame,
                    "duration_frames": duration_frames,
                    "velocity": vel,
                    "confidence": conf,
                }
            )
        return events
