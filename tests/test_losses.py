# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Tests for PerceptualAudioLoss."""
from __future__ import annotations

import pytest
import torch

from models.losses import PerceptualAudioLoss


class TestPerceptualAudioLoss:
    """Test loss function correctness."""

    @pytest.fixture
    def cfg(self):
        return {
            "loss": {
                "mr_stft_weight": 1.0,
                "spectral_flatness_weight": 0.1,
                "crest_factor_weight": 0.05,
                "onset_f1_weight": 1.0,
                "pitch_ce_weight": 0.8,
                "velocity_mae_weight": 0.3,
                "cross_modal_alignment_weight": 0.5,
            }
        }

    @pytest.fixture
    def loss_fn(self, cfg):
        return PerceptualAudioLoss(cfg)

    def test_mr_stft_returns_scalar(self, loss_fn):
        """MR-STFT loss should return a scalar."""
        pred = torch.randn(1, 1, 4096)
        target = torch.randn(1, 1, 4096)

        total, components = loss_fn(
            pred_stems=[pred],
            target_stems=[target],
        )

        assert total.dim() == 0, f"total loss dim={total.dim()}"
        assert total.item() > 0, "loss should be positive"

    def test_components_sum(self, loss_fn):
        """Loss components dict should have expected keys."""
        pred_guitar = torch.randn(1, 1, 4096)
        pred_bass = torch.randn(1, 1, 4096)
        target_guitar = torch.randn(1, 1, 4096)
        target_bass = torch.randn(1, 1, 4096)

        _, components = loss_fn(
            pred_stems=[pred_guitar, pred_bass],
            target_stems=[target_guitar, target_bass],
        )

        assert "spectral" in components
        assert "flatness" in components
        assert "crest" in components
        assert "alignment" in components

    def test_finite_values(self, loss_fn):
        """All loss components should be finite."""
        pred = torch.randn(1, 1, 4096)
        target = torch.randn(1, 1, 4096)

        total, components = loss_fn(
            pred_stems=[pred],
            target_stems=[target],
        )

        assert torch.isfinite(total).all(), "total loss is not finite"
        for k, v in components.items():
            assert abs(v) < 1e6, f"{k} = {v} is too large"

    def test_zero_loss_on_identical(self, loss_fn):
        """Loss should be near-zero for identical pred/target."""
        target = torch.randn(1, 1, 4096)
        pred = target.clone()

        total, _ = loss_fn(
            pred_stems=[pred],
            target_stems=[target],
        )

        assert total.item() < 1.0, f"loss={total.item()} should be near zero"

    def test_loss_increases_with_noise(self, loss_fn):
        """Adding noise should increase loss."""
        target = torch.randn(1, 1, 4096)
        pred_clean = target.clone()
        pred_noisy = target + 0.1 * torch.randn_like(target)

        total_clean, _ = loss_fn([pred_clean], [target])
        total_noisy, _ = loss_fn([pred_noisy], [target])

        assert total_noisy > total_clean, "noisy loss should be higher"
