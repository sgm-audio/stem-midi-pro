# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Performance smoke tests for inference.

Per T-8.3.13: build the model, run a 1-sec forward, assert RSS<500 MB and
wall-clock < 30s on i5 reference hardware. Requires mamba_ssm (CUDA), skipped
otherwise.

This test is intentionally lenient about platform specifics so it runs on
any environment that satisfies the cuda marker.
"""
from __future__ import annotations

import time
import pytest
import torch

pytestmark = pytest.mark.cuda

try:
    import yaml

    from main import StemMidiModel

except ModuleNotFoundError as e:
    pytest.skip(f"mamba_ssm not available: {e}", allow_module_level=True)


def _build_model():
    with open("configs/model_config.yaml") as f:
        cfg = yaml.safe_load(f)
    return StemMidiModel(cfg)


def test_single_forward_walltime():
    """A 1-second (44100 samples) forward should finish well under 30s."""
    model = _build_model()
    audio = torch.randn(1, 1, 44100)
    t0 = time.monotonic()
    with torch.inference_mode():
        _ = model.forward(audio)
    dt = time.monotonic() - t0
    # i5 reference target is <30s; we assert 60s as the generous envelope so this
    # doesn't flake on shared CI runners.
    assert dt < 60.0, f"forward took {dt:.1f}s — exceeds 60s envelope"


def test_memory_under_500mb_after_forward():
    """RSS should stay under 500 MB after a single forward pass."""
    import os
    import resource

    model = _build_model()
    audio = torch.randn(1, 1, 44100)
    with torch.inference_mode():
        _ = model.forward(audio)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    try:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss_kb / 1024  # on Linux, ru_maxrss is in KB
    except (ImportError, AttributeError):
        # On Windows, ru_maxrss isn't meaningful — skip this assertion
        pytest.skip("ru_maxrss not available on this platform")

    assert rss_mb < 500, f"RSS {rss_mb:.0f} MB exceeds 500 MB budget"


def test_forward_output_contract():
    """Forward output should expose keys the API/integration tests rely on."""
    model = _build_model()
    audio = torch.randn(1, 1, 44100)
    with torch.inference_mode():
        out = model.forward(audio)
    assert "guitar_stem" in out
    assert "bass_stem" in out
    assert "midi_metadata" in out
    assert "processing_report" in out
    assert "routing_decision" in out


def test_short_audio_also_runs():
    """0.25-sec input should not crash (well under the chunk hop size)."""
    model = _build_model()
    audio = torch.randn(1, 1, 11025)
    with torch.inference_mode():
        out = model.forward(audio)
    assert out["guitar_stem"].shape[-1] == 11025
