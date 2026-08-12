"""Tests for Mamba-3 loss functions (research only)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_track_processor_loss_creation():
    """TrackProcessorLoss should initialize without error."""
    from losses import TrackProcessorLoss

    cfg = type('MockCfg', (), {})()
    loss_fn = TrackProcessorLoss(cfg, sample_rate=44100)
    assert loss_fn is not None
    assert loss_fn.global_step == 0


def test_ntxent_loss_shape():
    """NTXentLoss should work with batch embeddings."""
    from losses import NTXentLoss

    loss_fn = NTXentLoss(temperature=0.1)
    z1 = torch.randn(4, 256)
    z2 = torch.randn(4, 256)

    loss = loss_fn(z1, z2)
    assert loss.ndim == 0
    assert loss > 0


def test_augment_for_contrastive():
    """augment_for_contrastive should return same-shape tensor."""
    from losses import augment_for_contrastive

    wav = torch.randn(44100)
    aug = augment_for_contrastive(wav)
    assert aug.shape == wav.shape
    assert aug.dtype == wav.dtype
