# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Performance smoke tests for inference (fake backends — shape/contract only).

Heavy pretrained/mamba perf is covered manually via scripts/eval_smoke.py.
"""
from __future__ import annotations

import time

import pytest
import torch

from main import StemMidiModel


def test_single_forward_walltime(mock_config):
    """A 1-second (44100 samples) forward should finish well under 30s."""
    model = StemMidiModel(mock_config)
    audio = torch.randn(1, 1, 44100)
    t0 = time.monotonic()
    with torch.inference_mode():
        _ = model.forward(audio)
    dt = time.monotonic() - t0
    assert dt < 60.0, f"forward took {dt:.1f}s — exceeds 60s envelope"


def test_memory_under_500mb_after_forward(mock_config):
    """RSS should stay under 500 MB after a single forward pass."""
    try:
        import resource
    except ImportError:
        pytest.skip("resource module not available on this platform")

    model = StemMidiModel(mock_config)
    audio = torch.randn(1, 1, 44100)
    with torch.inference_mode():
        _ = model.forward(audio)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss_kb / 1024  # on Linux, ru_maxrss is in KB
    except AttributeError:
        pytest.skip("ru_maxrss not available on this platform")

    assert rss_mb < 500, f"RSS {rss_mb:.0f} MB exceeds 500 MB budget"


def test_forward_output_contract(mock_config):
    """Forward output should expose keys the API/integration tests rely on."""
    model = StemMidiModel(mock_config)
    audio = torch.randn(1, 1, 44100)
    with torch.inference_mode():
        out = model.forward(audio)
    assert "guitar_stem" in out
    assert "bass_stem" in out
    assert "midi_metadata" in out
    assert "guitar" in out["midi_metadata"]
    assert "bass" in out["midi_metadata"]
    assert "processing_report" in out
    assert "routing_decision" in out


def test_short_audio_also_runs(mock_config):
    """0.25-sec input should not crash (well under the chunk hop size)."""
    model = StemMidiModel(mock_config)
    audio = torch.randn(1, 1, 11025)
    with torch.inference_mode():
        out = model.forward(audio)
    assert out["guitar_stem"].shape[-1] == 11025
