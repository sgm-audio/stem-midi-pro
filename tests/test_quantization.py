# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Tests for INT8 dynamic quantization on the inference path.

Requires mamba_ssm (CUDA) — the model construction imports MambaSeparator
which imports mamba_ssm. Skipped on CPU-only environments.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.cuda

try:
    import yaml

    with open("configs/model_config.yaml") as f:
        _cfg = yaml.safe_load(f)
    from main import StemMidiModel

    _model = None  # Lazy
except ModuleNotFoundError as e:
    pytest.skip(f"model deps unavailable: {e}", allow_module_level=True)


@pytest.fixture(scope="module")
def model():
    """Build a small StemMidiModel for quantization tests."""
    with open("configs/model_config.yaml") as f:
        config = yaml.safe_load(f)
    return StemMidiModel(config)


def test_process_audio_file_quantize_returns_same_shapes(tmp_path, model):
    """process_audio_file(quantize=True) should produce same output shapes as quantize=False."""
    import numpy as np
    import soundfile as sf

    sr = 44100
    p = tmp_path / "in.wav"
    sf.write(str(p), np.zeros(sr, dtype=np.float32), sr)

    out_q = model.process_audio_file(str(p), quantize=True)
    out_nq = model.process_audio_file(str(p), quantize=False)

    assert out_q["stems"]["guitar"].shape == out_nq["stems"]["guitar"].shape
    assert out_q["stems"]["bass"].shape == out_nq["stems"]["bass"].shape
    assert out_q["sample_rate"] == out_nq["sample_rate"]


def test_quantized_mask_within_tolerance_of_unquantized(model, tmp_path):
    """Masks from quantized and unquantized paths should be close (within 5%).

    INT8 dynamic quantization should preserve mask values very closely on CPU.
    """
    import numpy as np
    import soundfile as sf

    sr = 44100
    p = tmp_path / "in.wav"
    # Use a non-trivial input
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(str(p), audio, sr)

    out_q = model.process_audio_file(str(p), quantize=True)
    out_nq = model.process_audio_file(str(p), quantize=False)

    guitar_q = out_q["stems"]["guitar"]
    guitar_nq = out_nq["stems"]["guitar"]
    # Reasonable tolerance for INT8 dynamic quantization
    diff = np.abs(guitar_q - guitar_nq).max()
    assert diff < 0.05, f"max mask diff {diff} too large"


def test_quantization_flag_runs_without_error(model, tmp_path):
    """Calling process_audio_file with quantize=True should not raise."""
    import numpy as np
    import soundfile as sf

    p = tmp_path / "in.wav"
    sf.write(str(p), np.zeros(44100, dtype=np.float32), 44100)
    out = model.process_audio_file(str(p), quantize=True)
    assert "stems" in out
    assert "midi" in out
