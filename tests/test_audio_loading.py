# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Tests for audio loading functionality."""
import pytest
import numpy as np
import tempfile
import soundfile as sf

pytestmark = pytest.mark.cuda


def test_load_audio_mono():
    """_load_audio should load a mono WAV file correctly."""
    from main import StemMidiModel
    import yaml

    with open("configs/model_config.yaml") as f:
        config = yaml.safe_load(f)
    model = StemMidiModel(config)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        dummy = np.random.randn(44100).astype(np.float32)
        sf.write(f.name, dummy, 44100)
        audio, sr = model._load_audio(f.name)

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert sr == 44100
    assert len(audio) == 44100


def test_load_audio_stereo_to_mono():
    """Stereo audio should be converted to mono."""
    from main import StemMidiModel
    import yaml

    with open("configs/model_config.yaml") as f:
        config = yaml.safe_load(f)
    model = StemMidiModel(config)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        dummy = np.random.randn(44100, 2).astype(np.float32)
        sf.write(f.name, dummy, 44100)
        audio, sr = model._load_audio(f.name)

    assert audio.ndim == 1, f"Expected mono, got shape {audio.shape}"
    assert sr == 44100


def test_load_audio_file_not_found():
    """Should raise (librosa raises FileNotFoundError or IOError) for missing file."""
    from main import StemMidiModel
    import yaml

    with open("configs/model_config.yaml") as f:
        config = yaml.safe_load(f)
    model = StemMidiModel(config)

    with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
        model._load_audio("/nonexistent/file.wav")
