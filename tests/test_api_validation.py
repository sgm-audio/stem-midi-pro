"""Tests for API audio validation."""
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException

from api import validate_audio_file


def test_validate_audio_valid_file(tmp_path):
    """A supported WAV file should pass validation."""
    audio_path = tmp_path / "valid.wav"
    sf.write(audio_path, np.zeros(44100 * 5, dtype=np.float32), 44100)

    info = validate_audio_file(str(audio_path))

    assert info["duration"] == pytest.approx(5.0, rel=0.1)
    assert info["sample_rate"] == 44100


def test_validate_audio_too_long(monkeypatch):
    """Audio over the duration limit should be rejected without a huge fixture."""
    monkeypatch.setattr(
        "api.sf.info",
        lambda _path: SimpleNamespace(
            duration=700.0,
            samplerate=44100,
            subtype="PCM_16",
            channels=1,
            format="WAV",
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        validate_audio_file("unused.wav")

    assert exc_info.value.status_code == 400


def test_validate_audio_unsupported_sample_rate(tmp_path):
    """Unsupported sample rates should be rejected."""
    audio_path = tmp_path / "unsupported-rate.wav"
    sf.write(audio_path, np.zeros(22050, dtype=np.float32), 22050)

    with pytest.raises(HTTPException) as exc_info:
        validate_audio_file(str(audio_path))

    assert exc_info.value.status_code == 400
