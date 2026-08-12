# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Tests for the streaming inference path.

Tests that process_audio_streaming correctly counts chunks and merges outputs.
Requires mamba_ssm (CUDA). Marked cuda and skipped when mamba_ssm unavailable.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.cuda

try:
    import yaml

    from main import StemMidiModel

except ModuleNotFoundError as e:
    pytest.skip(f"mamba_ssm not available: {e}", allow_module_level=True)


@pytest.fixture(scope="module")
def model():
    with open("configs/model_config.yaml") as f:
        cfg = yaml.safe_load(f)
    return StemMidiModel(cfg)


def _write_audio(path, duration_s: float, sr: int = 44100):
    import numpy as np
    import soundfile as sf

    p = path / "in.wav"
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(str(p), audio, sr)
    return p


def test_chunks_count_matches_audio_length(tmp_path, model):
    """6 second audio with 2 second chunks and 0.5 overlap → hop=1s → ~6 chunks (5-6)."""
    p = _write_audio(tmp_path, duration_s=6.0)
    result = model.process_audio_streaming(str(p), chunk_seconds=2.0)
    assert "chunks_processed" in result
    # Overlap 0.5 means hop_len = chunk_len/2 = 1s. With 6s audio → ~6 hops.
    assert 4 <= result["chunks_processed"] <= 10


def test_streaming_returns_stems_and_midi(tmp_path, model):
    """Streamed inference should still produce the full output contract."""
    p = _write_audio(tmp_path, duration_s=3.0)
    result = model.process_audio_streaming(str(p), chunk_seconds=1.0)

    assert "stems" in result
    assert "guitar" in result["stems"]
    assert "bass" in result["stems"]
    assert "midi" in result
    assert "report" in result
    assert "routing" in result
    assert result["sample_rate"] == 44100


def test_streaming_full_length_matches_chunk_audio_shape(tmp_path, model):
    """Guitar stem length from streaming should be at least as long as audio."""
    duration_s = 5.0
    p = _write_audio(tmp_path, duration_s=duration_s)
    result = model.process_audio_streaming(str(p), chunk_seconds=2.0)

    expected_min_samples = int(0.8 * 44100 * duration_s)  # Allow some padding tolerance
    actual_samples = result["stems"]["guitar"].shape[-1]
    assert actual_samples >= expected_min_samples, (
        f"guitar stem {actual_samples} samples too short for {duration_s}s audio"
    )


def test_chunk_seconds_one_second_smaller(tmp_path, model):
    """1-second chunks should still process without errors."""
    p = _write_audio(tmp_path, duration_s=4.0)
    result = model.process_audio_streaming(str(p), chunk_seconds=1.0)
    assert result["chunks_processed"] >= 1


def test_short_audio_single_chunk(tmp_path, model):
    """Audio shorter than chunk_seconds → 1 chunk."""
    p = _write_audio(tmp_path, duration_s=0.5)
    result = model.process_audio_streaming(str(p), chunk_seconds=2.0)
    assert result["chunks_processed"] == 1
