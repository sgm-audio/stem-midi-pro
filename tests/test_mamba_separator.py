# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Tests for MambaSeparator model."""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.cuda

try:
    from models.mamba_separator import MambaSeparator
except ModuleNotFoundError as e:
    pytest.skip(f"mamba_ssm not available: {e}", allow_module_level=True)


class TestMambaSeparator:
    """Test MambaSeparator construction and forward pass."""

    @pytest.fixture
    def small_cfg(self):
        return {
            "audio": {"hop_length": 512, "n_fft": 2048, "sample_rate": 44100},
            "separator": {
                "d_model": 128,
                "n_layer": 2,
                "d_state": 8,
                "d_conv": 4,
                "expand": 2,
            },
            "quality_gates": {"min_confidence": 0.6, "min_si_sdr": 20.0},
        }

    def test_construct_from_config(self, small_cfg):
        """C-2.1: Model should construct without invalid kwargs."""
        model = MambaSeparator(small_cfg)
        assert isinstance(model, torch.nn.Module)
        assert model.mamba_blocks is not None
        assert len(model.mamba_blocks) == 2

    def test_forward_shape(self, small_cfg):
        """Forward pass should produce correct output shapes."""
        model = MambaSeparator(small_cfg)
        audio = torch.randn(1, 1, 8192)

        guitar, bass, residual, state_cache, metrics = model(audio)

        assert guitar.shape == (1, 1, 8192), f"got {guitar.shape}"
        assert bass.shape == (1, 1, 8192), f"got {bass.shape}"
        assert residual.shape == (1, 1, 8192), f"got {residual.shape}"
        assert state_cache is None  # Streaming not yet implemented
        assert metrics.shape == (1, 2), f"got {metrics.shape}"

    def test_no_nan(self, small_cfg):
        """Output should contain no NaN values."""
        model = MambaSeparator(small_cfg)
        audio = torch.randn(1, 1, 8192)

        with torch.inference_mode():
            guitar, bass, residual, _, _ = model(audio)

        assert not torch.isnan(guitar).any(), "guitar contains NaN"
        assert not torch.isnan(bass).any(), "bass contains NaN"
        assert not torch.isnan(residual).any(), "residual contains NaN"

    def test_stems_differ_on_synthetic(self, small_cfg):
        """Guitar and bass stems should be different on synthetic input."""
        model = MambaSeparator(small_cfg)
        audio = torch.randn(1, 1, 8192)

        with torch.inference_mode():
            guitar, bass, _, _, _ = model(audio)

        # They should not be bit-identical
        assert not torch.allclose(guitar, bass, atol=1e-4), "stems are identical"

    def test_centroid_metric_range(self, small_cfg):
        """Centroid separation metric should be in valid range [0, 30]."""
        model = MambaSeparator(small_cfg)
        audio = torch.randn(1, 1, 8192)

        with torch.inference_mode():
            _, _, _, _, metrics = model(audio)

        centroid = metrics[0, 0].item()
        phase = metrics[0, 1].item()
        assert 0 <= centroid <= 30, f"centroid={centroid} out of range"
        assert 0 <= phase <= 1, f"phase={phase} out of range"

    def test_short_input_rejected(self, small_cfg):
        """Input shorter than n_fft should raise assertion error."""
        model = MambaSeparator(small_cfg)
        audio = torch.randn(1, 1, 512)  # shorter than n_fft=2048

        with pytest.raises(AssertionError):
            model(audio)

    def test_stereo_input_rejected(self, small_cfg):
        """Stereo input (C=2) should raise assertion error."""
        model = MambaSeparator(small_cfg)
        audio = torch.randn(1, 2, 8192)

        with pytest.raises(AssertionError):
            model(audio)

    def test_batch_size_2(self, small_cfg):
        """Model should handle batch_size > 1."""
        model = MambaSeparator(small_cfg)
        audio = torch.randn(2, 1, 8192)

        with torch.inference_mode():
            guitar, bass, residual, _, _ = model(audio)

        assert guitar.shape == (2, 1, 8192)
        assert bass.shape == (2, 1, 8192)
