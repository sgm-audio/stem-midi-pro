# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
Tests for models/losses.py — MR-STFT, composite PerceptualAudioLoss,
and the transcription losses: onset F1, pitch CE, velocity MAE, duration IoU.
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch", reason="requires torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.losses import (  # noqa: E402
    MultiResolutionSTFTLoss,
    PerceptualAudioLoss,
    duration_iou_loss_fn,
    onset_f1_loss_fn,
    pitch_ce_loss_fn,
    velocity_mae_loss_fn,
)


def make_cfg():
    return {
        "loss": {
            "mr_stft_weight": 1.0,
            "spectral_flatness_weight": 0.1,
            "crest_factor_weight": 0.05,
            "onset_f1_weight": 1.0,
            "pitch_ce_weight": 0.8,
            "velocity_mae_weight": 0.3,
            "duration_iou_weight": 0.5,
            "cross_modal_alignment_weight": 0.5,
        }
    }


def make_audio(batch=2, channels=1, length=8192, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, channels, length)


class TestMRSTFT:
    def test_returns_scalar(self):
        loss_fn = MultiResolutionSTFTLoss()
        x, y = make_audio(), make_audio(seed=1)
        loss = loss_fn(x, y)
        assert loss.dim() == 0, "MR-STFT loss must be a scalar"

    def test_finite(self):
        loss_fn = MultiResolutionSTFTLoss()
        x, y = make_audio(), make_audio(seed=1)
        loss = loss_fn(x, y)
        assert torch.isfinite(loss)

    def test_zero_for_identical(self):
        loss_fn = MultiResolutionSTFTLoss()
        x = make_audio()
        assert loss_fn(x, x).item() == pytest.approx(0.0, abs=1e-5)


class TestPerceptualAudioLoss:
    def test_components_sum_and_finite(self):
        loss_fn = PerceptualAudioLoss(make_cfg())
        pred = [make_audio(), make_audio(seed=1)]
        target = [make_audio(seed=2), make_audio(seed=3)]
        total, components = loss_fn(pred, target)
        assert total.dim() == 0
        assert torch.isfinite(total)
        for key in ("spectral", "flatness", "crest", "alignment"):
            assert key in components
            assert components[key] == pytest.approx(components[key])  # not NaN

    def test_alignment_with_onsets(self):
        loss_fn = PerceptualAudioLoss(make_cfg())
        pred = [make_audio(), make_audio(seed=1)]
        target = [make_audio(seed=2), make_audio(seed=3)]
        pred_onsets = torch.rand(2, 8192)
        target_onsets = (torch.rand(2, 8192) > 0.95).float()
        total, components = loss_fn(pred, target, pred_onsets, target_onsets)
        assert torch.isfinite(total)
        assert components["alignment"] == pytest.approx(components["alignment"])


class TestOnsetF1:
    def test_range(self):
        pred = torch.rand(4, 100)
        target = (torch.rand(4, 100) > 0.9).float()
        loss = onset_f1_loss_fn(pred, target)
        assert 0.0 <= loss.item() <= 1.0

    def test_perfect_prediction_near_zero(self):
        target = (torch.rand(2, 50) > 0.5).float()
        loss = onset_f1_loss_fn(target, target)
        assert loss.item() < 0.01


class TestPitchCE:
    def test_scalar_finite(self):
        logits = torch.randn(2, 16, 128)
        targets = torch.randint(0, 128, (2, 16))
        loss = pitch_ce_loss_fn(logits, targets)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_perfect_low(self):
        targets = torch.randint(0, 128, (2, 16))
        logits = torch.full((2, 16, 128), -10.0)
        logits.scatter_(2, targets.unsqueeze(-1), 10.0)
        loss = pitch_ce_loss_fn(logits, targets)
        assert loss.item() < 0.01


class TestVelocityMAE:
    def test_non_negative(self):
        pred = torch.rand(2, 10)
        target = torch.rand(2, 10)
        loss = velocity_mae_loss_fn(pred, target)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_zero_for_identical(self):
        target = torch.rand(2, 10)
        assert velocity_mae_loss_fn(target, target).item() == pytest.approx(0.0)


class TestDurationIoU:
    def test_range(self):
        pred = torch.rand(2, 10) * 100
        target = torch.rand(2, 10) * 100
        loss = duration_iou_loss_fn(pred, target)
        assert 0.0 <= loss.item() <= 1.0

    def test_zero_for_identical(self):
        target = torch.rand(2, 10) * 100
        assert duration_iou_loss_fn(target, target).item() < 1e-5
