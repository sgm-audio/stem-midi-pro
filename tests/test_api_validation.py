# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Tests for API audio validation."""
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException


def _write_audio(suffix: str, audio: np.ndarray, sr: int) -> str:
    """Write audio to a temp file and return the path, ensuring it's fully closed."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    sf.write(path, audio, sr)
    return path


def test_validate_audio_valid_file():
    """Valid WAV file should pass validation."""
    from api import validate_audio_file

    path = _write_audio(".wav", np.random.randn(44100 * 5).astype(np.float32), 44100)
    try:
        info = validate_audio_file(path)
        assert info["duration"] == pytest.approx(5.0, rel=0.01)
        assert info["sample_rate"] == 44100
        assert info["channels"] == 1
    finally:
        os.unlink(path)


def test_validate_audio_too_long():
    """Audio exceeding MAX_DURATION should raise HTTPException(400)."""
    from api import MAX_DURATION_SECONDS, validate_audio_file

    n_samples = 44100 * (MAX_DURATION_SECONDS + 5)
    path = _write_audio(".wav", np.zeros(n_samples, dtype=np.float32), 44100)
    try:
        with pytest.raises(HTTPException) as exc:
            validate_audio_file(path)
        assert exc.value.status_code == 400
    finally:
        os.unlink(path)


def test_validate_audio_unsupported_sr():
    """Unsupported sample rates should raise HTTPException(400)."""
    from api import SUPPORTED_SAMPLE_RATES, validate_audio_file

    bad_sr = 22050
    assert bad_sr not in SUPPORTED_SAMPLE_RATES
    path = _write_audio(".wav", np.random.randn(bad_sr * 5).astype(np.float32), bad_sr)
    try:
        with pytest.raises(HTTPException) as exc:
            validate_audio_file(path)
        assert exc.value.status_code == 400
    finally:
        os.unlink(path)


def test_validate_audio_supported_sr_48k():
    """48kHz should pass — both 44.1kHz and 48kHz are supported."""
    from api import validate_audio_file

    path = _write_audio(".wav", np.random.randn(48000 * 2).astype(np.float32), 48000)
    try:
        info = validate_audio_file(path)
        assert info["sample_rate"] == 48000
    finally:
        os.unlink(path)


def test_validate_audio_stereo():
    """Stereo audio file should report 2 channels."""
    from api import validate_audio_file

    path = _write_audio(".wav", np.random.randn(44100 * 2, 2).astype(np.float32), 44100)
    try:
        info = validate_audio_file(path)
        assert info["channels"] == 2
    finally:
        os.unlink(path)


def test_validate_audio_flac():
    """FLAC format should validate if supported by soundfile."""
    from api import validate_audio_file

    path = _write_audio(".flac", np.random.randn(44100 * 2).astype(np.float32), 44100)
    try:
        info = validate_audio_file(path)
        assert info["sample_rate"] == 44100
    finally:
        os.unlink(path)
