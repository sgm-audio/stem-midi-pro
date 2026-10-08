# SPDX-License-Identifier: Apache-2.0
"""Tests for audio loading via main.StemMidiModel._load_audio (T-8.2.3).

Loading instantiation of StemMidiModel requires torch/nemo/mamba_ssm; each
test guards with importorskip so collection works without them.
"""

import os

import pytest


def _import_model_cls():
    pytest.importorskip("torch")
    try:
        from main import StemMidiModel
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"Cannot import main.py: {exc}")
    return StemMidiModel


def test_load_audio_mono(mock_config, temp_audio_file):
    """_load_audio should load a mono WAV file as float32."""
    np = pytest.importorskip("numpy")
    StemMidiModel = _import_model_cls()

    model = StemMidiModel(mock_config)
    audio, sr = model._load_audio(temp_audio_file)

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert sr == 44100
    assert audio.ndim == 1
    assert len(audio) == 44100


def test_load_audio_stereo_to_mono(mock_config, temp_stereo_audio_file):
    """Stereo audio should be downmixed to mono."""
    pytest.importorskip("numpy")
    StemMidiModel = _import_model_cls()

    model = StemMidiModel(mock_config)
    audio, sr = model._load_audio(temp_stereo_audio_file)

    assert audio.ndim == 1, f"Expected mono, got shape {audio.shape}"
    assert sr == 44100


def test_load_audio_file_not_found(mock_config):
    """Missing file paths should raise FileNotFoundError."""
    StemMidiModel = _import_model_cls()

    model = StemMidiModel(mock_config)
    assert not os.path.exists("/nonexistent/file.wav")
    with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
        model._load_audio("/nonexistent/file.wav")


def test_load_audio_22k_resample_path(mock_config, tmp_path):
    """A 22.05 kHz file should load at its native rate; process_audio_file
    resamples it to the configured sample rate.

    # expects fix per TODO C-2.12 (torch.tensor -> torch.from_numpy) for the
    # full process_audio_file path to run cleanly.
    """
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    StemMidiModel = _import_model_cls()

    path = tmp_path / "low_sr.wav"
    sf.write(str(path), np.zeros(22050, dtype=np.float32), 22050)

    model = StemMidiModel(mock_config)
    audio, sr = model._load_audio(str(path))
    assert sr == 22050
    assert len(audio) == 22050
