# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for Stem+MIDI Pro (T-8.1.1).

All heavy imports (torch, numpy, soundfile, nemo, mamba_ssm) happen INSIDE
fixtures/tests so that `pytest --collect-only` works in environments without
those dependencies installed.
"""

import struct
import wave

import pytest

# ---------------------------------------------------------------------------
# Audio tensor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def audio():
    """1-second 44.1 kHz mono audio tensor of shape (1, 1, 44100)."""
    torch = pytest.importorskip("torch")
    sr = 44100
    t = torch.linspace(0, 1, sr)
    mono = 0.3 * torch.sin(2 * torch.pi * 220 * t) + 0.2 * torch.sin(2 * torch.pi * 110 * t)
    return mono.reshape(1, 1, sr).float()


@pytest.fixture
def stereo_audio():
    """1-second 44.1 kHz stereo audio tensor of shape (1, 2, 44100)."""
    torch = pytest.importorskip("torch")
    sr = 44100
    t = torch.linspace(0, 1, sr)
    left = 0.3 * torch.sin(2 * torch.pi * 220 * t)
    right = 0.3 * torch.sin(2 * torch.pi * 330 * t)
    return torch.stack([left, right]).unsqueeze(0).float()  # (1, 2, 44100)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Full config dict mirroring configs/model_config.yaml, downsized for CPU.

    Same keys as the real config, but tiny dims so tests run in seconds on
    CPU (d_model=32, n_layers=2, small n_fft so (1,1,8192) forwards are fast).
    """
    return {
        "name": "stem_midi_mamba_test",
        "restore_from_path": None,
        "audio": {
            "sample_rate": 44100,
            "n_fft": 512,
            "hop_length": 128,
            "n_mels": 32,
            "chunk_duration_sec": 2.0,
            "overlap_ratio": 0.5,
        },
        "separator": {
            "d_model": 32,
            "n_layers": 2,
            "d_state": 8,
            "d_conv": 4,
            "expand": 2,
        },
        "transcriber": {
            "d_model": 32,
            "n_layers": 2,
            "d_state": 8,
            "onset_head_dim": 16,
            "pitch_vocab_size": 128,
            "velocity_bins": 128,
            "expression_heads": ["bend", "vibrato", "slide"],
            "onset_threshold": 0.5,
        },
        "training": {
            "precision": "fp32",
            "gradient_checkpointing": False,
        },
        "loss": {
            "mr_stft_weight": 1.0,
            "spectral_flatness_weight": 0.1,
            "crest_factor_weight": 0.05,
            "onset_f1_weight": 1.0,
            "pitch_ce_weight": 0.8,
            "velocity_mae_weight": 0.3,
            "cross_modal_alignment_weight": 0.5,
        },
        "quality_gates": {
            "studio_confidence_threshold": 0.85,
            "draft_confidence_threshold": 0.70,
            "min_confidence": 0.6,
            "min_si_sdr": 20.0,
        },
    }


# ---------------------------------------------------------------------------
# Temporary audio files
# ---------------------------------------------------------------------------


def _sine_seconds(seconds, sr=44100, freq=220.0):
    """Pure-numpy sine wave, float32 in [-1, 1)."""
    import math

    n = int(seconds * sr)
    return [0.3 * math.sin(2 * math.pi * freq * i / sr) for i in range(n)]


def _write_wav_wave_module(path, samples, sr):
    """Fallback WAV writer using the stdlib `wave` module (16-bit PCM)."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples)
        w.writeframes(frames)


@pytest.fixture
def temp_audio_file(tmp_path):
    """1-second 44.1 kHz mono WAV file (via soundfile, else stdlib wave)."""
    path = tmp_path / "test_audio.wav"
    sr = 44100
    samples = _sine_seconds(1.0, sr)
    try:
        import numpy as np
        import soundfile as sf

        sf.write(str(path), np.asarray(samples, dtype=np.float32), sr)
    except ImportError:
        _write_wav_wave_module(path, samples, sr)
    return str(path)


@pytest.fixture
def temp_stereo_audio_file(tmp_path):
    """1-second 44.1 kHz stereo WAV file."""
    sf = pytest.importorskip("soundfile")
    np = pytest.importorskip("numpy")
    path = tmp_path / "test_stereo.wav"
    sr = 44100
    t = np.linspace(0, 1, sr)
    stereo = np.stack(
        [0.3 * np.sin(2 * np.pi * 220 * t), 0.3 * np.sin(2 * np.pi * 330 * t)],
        axis=1,
    ).astype(np.float32)
    sf.write(str(path), stereo, sr)
    return str(path)


@pytest.fixture
def temp_flac_file(tmp_path):
    """1-second 44.1 kHz mono FLAC file (requires soundfile FLAC support)."""
    sf = pytest.importorskip("soundfile")
    np = pytest.importorskip("numpy")
    path = tmp_path / "test_audio.flac"
    sr = 44100
    samples = np.asarray(_sine_seconds(1.0, sr), dtype=np.float32)
    try:
        sf.write(str(path), samples, sr, format="FLAC")
    except Exception as exc:
        pytest.skip(f"soundfile lacks FLAC support: {exc}")
    return str(path)


@pytest.fixture
def temp_mp3_file(tmp_path):
    """1-second 44.1 kHz mono MP3 file; skipped if no MP3 encoder available."""
    sf = pytest.importorskip("soundfile")
    np = pytest.importorskip("numpy")
    path = tmp_path / "test_audio.mp3"
    sr = 44100
    samples = np.asarray(_sine_seconds(1.0, sr), dtype=np.float32)
    try:
        sf.write(str(path), samples, sr, format="MP3", subtype="MPEG_LAYER_III")
    except Exception as exc:
        pytest.skip(f"soundfile lacks MP3 encoder support: {exc}")
    return str(path)


# ---------------------------------------------------------------------------
# Model fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model(mock_config):
    """StemMidiModel built from mock_config (downsized for CPU).

    Skips if heavyweight deps (torch) are unavailable. The pure-PyTorch
    Mamba fallback in models/_mamba_compat.py means mamba_ssm and nemo are
    NOT required (both were dropped; see DEP-4.1.3 / DEP-4.1.5).
    """
    pytest.importorskip("torch")
    from main import StemMidiModel

    model = StemMidiModel(mock_config)
    model.eval()
    return model
