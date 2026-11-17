# SPDX-License-Identifier: Apache-2.0
"""Test doubles that are injectable into ``StemMidiModel``.

``nn.Module`` refuses to let you assign a non-Module over a registered child
(``TypeError: cannot assign 'model.separator' as child module``), which is why
the suite has no stub-driven pipeline tests: people stub ``api.model`` instead
and the real wiring never gets checked. ``inject`` deletes the child first, then
writes the instance ``__dict__`` so normal attribute lookup finds the double.
"""

from __future__ import annotations

import contextlib
from typing import ClassVar


def inject(obj, name: str, value) -> None:
    """Replace a registered nn.Module child with a test double (restores nothing)."""
    import torch.nn as nn

    assert isinstance(obj, nn.Module), "inject() targets nn.Module instances"
    with contextlib.suppress(AttributeError):
        delattr(obj, name)  # pops from _modules
    object.__setattr__(obj, name, value)


class StubSeparator:
    """Deterministic separator: fixed stems + fixed (proxy, phase) metrics."""

    def __init__(self, guitar, bass, metrics=(25.0, 0.9), residual=None):
        import torch

        self.torch = torch
        self.guitar, self.bass = guitar, bass
        self.residual = residual
        self.metrics = metrics
        self.calls = []

    def __call__(self, audio):
        torch = self.torch
        residual = self.residual if self.residual is not None else torch.zeros_like(audio)
        metrics = (
            self.metrics
            if isinstance(self.metrics, torch.Tensor)
            else torch.tensor([list(self.metrics)], dtype=torch.float32).expand(audio.shape[0], 2)
        )
        self.calls.append(tuple(audio.shape))
        return self.guitar, self.bass, residual, None, metrics

    def separate(self, audio):
        g, b, r, _c, m = self(audio)
        return {"guitar": g, "bass": b, "residual": r, "metrics": m}


class StubTranscriber:
    """Frame-level logits with caller-controlled onsets / pitches / confidence."""

    def __init__(
        self,
        onsets,
        pitches,
        n_frames,
        confidence=0.95,
        velocity=0.7,
        expression=None,
        batch=1,
        durations=None,
        events=None,
    ):
        self.onsets = list(onsets)
        self.pitches = list(pitches)
        self.n_frames = n_frames
        self.confidence = confidence
        self.velocity = velocity
        self.expression = expression
        self.batch = batch
        self.durations = durations
        self.events = events

    def __call__(self, stem):
        import torch

        n, B = self.n_frames, self.batch
        onset = torch.full((B, n, 1), -10.0)
        pitch = torch.zeros((B, n, 128))
        vel = torch.full((B, n, 1), float(self.velocity))
        expr = torch.zeros((B, n, 3)) if self.expression is None else self.expression
        if isinstance(self.confidence, torch.Tensor):
            conf = self.confidence
        else:
            conf = torch.full((B, n), float(self.confidence))
        for f, p in zip(self.onsets, self.pitches, strict=False):
            onset[:, f, 0] = 10.0
            pitch[:, f, p] = 10.0
        return onset, pitch, vel, expr, conf

    def transcribe(self, stem):
        onset, pitch, vel, expr, conf = self(stem)
        out = {
            "onset_logits": onset,
            "pitch_logits": pitch,
            "velocity": vel,
            "expression": expr,
            "confidence": conf,
        }
        if self.events is not None:
            out["events"] = self.events
        return out


def frame_count(n_samples: int, hop: int) -> int:
    """torch.stft with center=True on (n_samples,) → 1 + floor(n/hop)."""
    return 1 + n_samples // hop


class StubReport:
    """A ProcessingReport-shaped stub for API-level packaging tests."""

    si_sdr: ClassVar[float] = 25.0
    phase_coherence: ClassVar[float] = 0.9
    avg_confidence: ClassVar[float] = 0.9
    artifact_flags: ClassVar[list[str]] = ["none"]
    low_confidence_notes: ClassVar[int] = 0

    @property
    def quality_tier(self):
        from utils.quality_gates import QualityTier

        return QualityTier.STUDIO

    @property
    def config(self):
        return {}
