# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Unit tests for pluggable backends (fake path — no heavy deps)."""
from __future__ import annotations

import torch

from models.backends.factory import build_separator, build_transcriber
from models.backends.fake_backend import FakeSeparatorBackend, FakeTranscriberBackend


def test_factory_builds_fake_backends(mock_config):
    sep = build_separator(mock_config)
    tr = build_transcriber(mock_config)
    assert isinstance(sep, FakeSeparatorBackend)
    assert isinstance(tr, FakeTranscriberBackend)


def test_fake_separator_shapes(mock_config):
    sep = FakeSeparatorBackend(mock_config)
    audio = torch.randn(2, 1, 8192)
    out = sep.separate(audio)
    assert out["guitar"].shape == audio.shape
    assert out["bass"].shape == audio.shape
    assert out["residual"].shape == audio.shape
    assert out["metrics"].shape == (2, 2)


def test_fake_transcriber_events(mock_config):
    tr = FakeTranscriberBackend(mock_config)
    stem = torch.randn(1, 1, 44100)
    out = tr.transcribe(stem)
    assert "events" in out
    assert len(out["events"][0]) >= 1
    assert out["onset_logits"].dim() == 3
    assert out["confidence"].dim() == 2


def test_stem_midi_model_with_fake_backends(mock_config):
    from main import StemMidiModel

    model = StemMidiModel(mock_config)
    audio = torch.randn(1, 1, 8192)
    with torch.inference_mode():
        out = model.forward(audio)
    assert "guitar_stem" in out
    assert "bass_stem" in out
    assert "midi_metadata" in out
    assert "guitar" in out["midi_metadata"]
    assert "bass" in out["midi_metadata"]
    assert out["processing_report"] is not None
    assert out["routing_decision"]["action"]


def test_process_audio_file_fake(mock_config, temp_audio_file):
    from main import StemMidiModel

    model = StemMidiModel(mock_config)
    result = model.process_audio_file(temp_audio_file)
    assert "stems" in result
    assert "midi_guitar" in result
    assert "midi_bass" in result
    assert result["sample_rate"] == 44100


def test_unknown_backend_raises(mock_config):
    bad = {**mock_config, "backends": {"separator": "nope", "transcriber": "fake"}}
    try:
        build_separator(bad)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "nope" in str(e)
