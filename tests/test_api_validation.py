# SPDX-License-Identifier: Apache-2.0
"""Tests for API audio validation (T-8.2.2 — no skip markers).

`api.py` imports StemMidiModel at module level, which pulls in torch/nemo/
mamba_ssm. T-8.2.1 (lazy import) will decouple this; until then each test
imports api lazily and skips if the chain is unavailable.
"""

import pytest


def _import_validate_audio_file():
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    try:
        from api import validate_audio_file
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"Cannot import api.py (see T-8.2.1): {exc}")
    return validate_audio_file


def test_validate_audio_valid_wav(temp_audio_file):
    """Valid 44.1 kHz WAV file should pass validation."""
    validate_audio_file = _import_validate_audio_file()
    pytest.importorskip("soundfile")

    info = validate_audio_file(temp_audio_file)
    assert info["duration"] == pytest.approx(1.0, rel=0.05)
    assert info["sample_rate"] == 44100


def test_validate_audio_too_long(tmp_path):
    """Audio exceeding MAX_DURATION should raise HTTPException."""
    validate_audio_file = _import_validate_audio_file()
    sf = pytest.importorskip("soundfile")
    np = pytest.importorskip("numpy")
    from fastapi import HTTPException

    path = tmp_path / "too_long.wav"
    sf.write(str(path), np.zeros(44100 * 601, dtype=np.float32), 44100)

    with pytest.raises(HTTPException) as exc_info:
        validate_audio_file(str(path))
    assert exc_info.value.status_code == 400
    assert "duration" in str(exc_info.value.detail).lower()


def test_validate_audio_unsupported_sample_rate(tmp_path):
    """22.05 kHz audio should be rejected (only 44.1k/48k supported)."""
    validate_audio_file = _import_validate_audio_file()
    sf = pytest.importorskip("soundfile")
    np = pytest.importorskip("numpy")
    from fastapi import HTTPException

    path = tmp_path / "bad_sr.wav"
    sf.write(str(path), np.zeros(22050, dtype=np.float32), 22050)

    with pytest.raises(HTTPException) as exc_info:
        validate_audio_file(str(path))
    assert exc_info.value.status_code == 400


def test_validate_audio_stereo(temp_stereo_audio_file):
    """Stereo WAV is valid (downmix happens at load, not validation)."""
    validate_audio_file = _import_validate_audio_file()

    info = validate_audio_file(temp_stereo_audio_file)
    assert info["channels"] == 2
    assert info["sample_rate"] == 44100


def test_validate_audio_flac(temp_flac_file):
    """Valid FLAC file should pass validation."""
    validate_audio_file = _import_validate_audio_file()

    info = validate_audio_file(temp_flac_file)
    assert info["sample_rate"] == 44100
    assert info["format"] == "FLAC"


def test_validate_audio_mp3(temp_mp3_file):
    """Valid MP3 file should pass validation (skipped if no encoder)."""
    validate_audio_file = _import_validate_audio_file()

    info = validate_audio_file(temp_mp3_file)
    assert info["sample_rate"] == 44100


def test_validate_audio_invalid_file(tmp_path):
    """Garbage bytes should raise HTTPException(400), not crash."""
    validate_audio_file = _import_validate_audio_file()
    from fastapi import HTTPException

    path = tmp_path / "not_audio.wav"
    path.write_bytes(b"this is not audio data")

    with pytest.raises(HTTPException) as exc_info:
        validate_audio_file(str(path))
    assert exc_info.value.status_code == 400
