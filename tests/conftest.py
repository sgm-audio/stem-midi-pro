# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Pytest configuration and fixtures for Stem+MIDI Pro tests."""
from __future__ import annotations

import pytest
import torch
import numpy as np
import tempfile
import os


@pytest.fixture
def audio():
    """1-second 44.1kHz mono tensor."""
    sr = 44100
    t = torch.linspace(0, 1, sr)
    return torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).unsqueeze(0)


@pytest.fixture
def stereo_audio():
    """2-channel 1-second audio."""
    sr = 44100
    t = torch.linspace(0, 1, sr)
    left = torch.sin(2 * torch.pi * 440 * t)
    right = torch.sin(2 * torch.pi * 330 * t)
    return torch.stack([left, right], dim=0).unsqueeze(0)


@pytest.fixture
def mock_config():
    """Full YAML config dict for testing."""
    return {
        "name": "test_model",
        "audio": {
            "sample_rate": 44100,
            "n_fft": 2048,
            "hop_length": 512,
            "n_mels": 80,
            "chunk_duration_sec": 2.0,
            "overlap_ratio": 0.5,
        },
        "separator": {
            "d_model": 256,
            "n_layer": 6,
            "d_state": 12,
            "d_conv": 4,
            "expand": 2,
        },
        "transcriber": {
            "d_model": 256,
            "n_layer": 4,
            "d_state": 12,
            "onset_head_dim": 64,
            "pitch_vocab_size": 128,
            "velocity_bins": 128,
            "expression_heads": ["bend", "vibrato", "slide"],
            "onset_threshold": 0.5,
        },
        "training": {"precision": "fp32", "gradient_checkpointing": True},
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
        # Default test backends: no mamba_ssm / demucs / basic-pitch required
        "backends": {
            "separator": "fake",
            "transcriber": "fake",
            "device": "cpu",
        },
    }


@pytest.fixture
def temp_audio_file():
    """WAV file in tmp directory."""
    import soundfile as sf
    sr = 44100
    t = np.linspace(0, 1, sr)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_flac_file():
    """FLAC file in tmp directory."""
    import soundfile as sf
    sr = 44100
    t = np.linspace(0, 1, sr)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
        sf.write(f.name, audio, sr)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_model(mock_config):
    """StemMidiModel with small config for testing."""
    from main import StemMidiModel
    return StemMidiModel(mock_config)
