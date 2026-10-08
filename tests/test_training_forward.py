# SPDX-License-Identifier: Apache-2.0
"""Training-path forward tests for StemMidiModel (regression for the
dataset-onset-target shape mismatch found by the train.py smoke run).

Dataset items carry target_onsets shaped (1, n_frames, 1) — after DataLoader
collation (B, 1, T', 1) — with a frame count from a hardcoded hop and no STFT
center padding. The model must normalize these onto its (B, T, 1) frame grid
before computing losses.
"""

import pytest

torch = pytest.importorskip("torch")


def test_prepare_onset_target_flattens_and_pads():
    """(B, 1, T', 1) targets are coerced to (B, T_pred), zero-padded."""
    from main import StemMidiModel

    target = torch.zeros(2, 1, 5, 1)  # collated dataset shape, T'=5
    target[0, 0, 0, 0] = 1.0
    logits = torch.zeros(2, 8, 1)  # model frame grid, T_pred=8

    out = StemMidiModel._prepare_onset_target(target, logits)

    assert out.shape == (2, 8)
    assert out[0, 0] == 1.0  # content preserved
    assert out[:, 5:].sum() == 0  # padding = no onset
    assert out[1].sum() == 0


def test_prepare_onset_target_truncates():
    """Overlong targets are truncated to the predicted frame count."""
    from main import StemMidiModel

    target = torch.ones(1, 1, 20, 1)
    logits = torch.zeros(1, 8, 1)

    out = StemMidiModel._prepare_onset_target(target, logits)

    assert out.shape == (1, 8)
    assert out.sum() == 8


def test_prepare_onset_target_none_passthrough():
    from main import StemMidiModel

    assert StemMidiModel._prepare_onset_target(None, torch.zeros(1, 4, 1)) is None


def test_training_forward_with_dataset_shaped_targets(mock_config):
    """Full training-mode forward: mismatched 4-D onset targets still yield a
    finite loss (the exact failure mode of the train.py smoke run)."""
    from main import StemMidiModel

    model = StemMidiModel(mock_config)
    model.train()

    B, T = 1, 8192
    audio = torch.randn(B, 1, T)
    # Mimic DataLoader-collated dataset items: spurious dims + stale frame count
    n_frames_stale = T // 512  # dataset hardcodes hop=512
    target = torch.zeros(B, 1, n_frames_stale, 1)
    target[0, 0, 0, 0] = 1.0

    outputs = model(
        audio,
        target_guitar=torch.randn(B, 1, T),
        target_bass=torch.randn(B, 1, T),
        target_onsets=target,
        target_pitch=torch.zeros(B, 1, n_frames_stale, 128),
    )

    assert "loss" in outputs
    assert torch.isfinite(outputs["loss"])
    assert all(torch.isfinite(torch.as_tensor(v)) for v in outputs["loss_dict"].values())
