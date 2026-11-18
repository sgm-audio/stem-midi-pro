# SPDX-License-Identifier: Apache-2.0
"""Deterministic fake backends for unit tests (no mamba_ssm / demucs / basic-pitch)."""

from __future__ import annotations

import torch

from models.backends.base import SeparateOut, TranscribeOut, empty_metrics


class FakeSeparatorBackend:
    """Splits mix into high/low filtered proxies (not musical — shape-correct only)."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.weights_loaded = False

    def separate(self, audio: torch.Tensor) -> SeparateOut:
        B, _C, _T = audio.shape
        # Crude spectral split via simple gain so guitar != bass
        guitar = audio * 0.6
        bass = audio * 0.4
        residual = audio - guitar - bass
        # FAKE-1: these used to be the constants 12.0 / 0.5 for ANY input, so a
        # test that ran a quality decision through this backend was asserting a
        # fixture, not behaviour -- and 12.0 is below the shipped
        # `min_centroid_separation_db: 20.0`, which pinned the fake path to
        # COMPLEX and made the studio/draft branches untestable.
        #
        # Compute them from the signal instead. Still crude and still not a
        # musical quality estimate (that is the real backend's job), but it is a
        # function of the input, so the gate sees something that moves.
        metrics = empty_metrics(B, audio.device)
        flat_g = guitar.reshape(B, -1)
        flat_b = bass.reshape(B, -1)
        eps = 1e-12
        # Spectral "centroid" stand-in: energy-weighted mean of the two stems.
        centroid_like = 20.0 * torch.log10(
            (flat_g.pow(2).mean(dim=-1) + eps) / (flat_b.pow(2).mean(dim=-1) + eps)
        )
        metrics[:, 0] = 25.0 + centroid_like.clamp(-5.0, 5.0)
        # Phase coherence stand-in: normalised correlation of the two stems.
        denom = flat_g.norm(dim=-1) * flat_b.norm(dim=-1) + eps
        metrics[:, 1] = (flat_g * flat_b).sum(dim=-1).div(denom).abs().clamp(0.0, 1.0)
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
        self._instrument = "guitar"
        self.weights_loaded = False

    def set_instrument(self, name: str) -> None:
        if name not in {"guitar", "bass"}:
            raise ValueError(f"Unsupported fake instrument {name!r}")
        self._instrument = name

    def transcribe(self, stem: torch.Tensor) -> TranscribeOut:
        B, _, T = stem.shape
        n_frames = max(1, T // self.hop)
        onset = torch.full((B, n_frames, 1), -10.0, device=stem.device)
        pitch = torch.zeros(B, n_frames, self.n_pitch, device=stem.device)
        vel = torch.zeros(B, n_frames, 1, device=stem.device)
        expr = torch.zeros(B, n_frames, 3, device=stem.device)
        conf = torch.zeros(B, n_frames, device=stem.device)

        frame = min(10, n_frames - 1)
        note = 60 if self._instrument == "guitar" else 36
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
