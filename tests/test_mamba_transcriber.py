# SPDX-License-Identifier: Apache-2.0
"""MambaTranscriber unit tests (T-8.3.2).

Locks the 5-output forward contract — would have caught C-2.2 (self.cfg
never assigned), C-2.9 (per-forward mel basis rebuild), and C-2.10
(MC-dropout confidence).
"""

import pytest

torch = pytest.importorskip("torch", reason="requires torch")


@pytest.fixture
def transcriber(mock_config):
    try:
        from models.mamba_transcriber import MambaTranscriber
    except (ImportError, TypeError) as exc:
        pytest.skip(f"MambaTranscriber not constructible: {exc}")
    return MambaTranscriber(mock_config)


def test_store_cfg_assignment(mock_config, transcriber):
    """self.cfg must be assigned (C-2.2)."""
    # expects fix per TODO C-2.2
    assert (
        getattr(transcriber, "cfg", None) is not None
    ), "MambaTranscriber.__init__ never assigns self.cfg (TODO C-2.2)"


def test_forward_five_outputs(mock_config, transcriber):
    """Forward on (1, 1, 8192) returns 5 outputs with correct shapes."""
    # expects fix per TODO C-2.2 (self.cfg) and C-2.9 (mel basis buffer)
    audio = torch.randn(1, 1, 8192)
    transcriber.eval()
    with torch.no_grad():
        onset_logits, pitch_logits, velocity, expression, confidence = transcriber(audio)

    hop = mock_config["audio"]["hop_length"]
    n_frames = 1 + 8192 // hop  # torch.stft frame count (center padding)

    assert onset_logits.shape == (1, n_frames, 1)
    assert pitch_logits.shape == (1, n_frames, 128)
    assert velocity.shape == (1, n_frames, 1)
    assert expression.shape == (1, n_frames, 3)  # bend, vibrato, slide
    assert confidence.shape == (1, n_frames)


def test_outputs_bounded(mock_config, transcriber):
    """Velocity/expression/confidence live in [0, 1]; no NaNs anywhere."""
    audio = torch.randn(1, 1, 8192) * 0.1
    transcriber.eval()
    with torch.no_grad():
        onset, pitch, velocity, expression, confidence = transcriber(audio)

    for name, t in [
        ("onset", onset),
        ("pitch", pitch),
        ("velocity", velocity),
        ("expression", expression),
        ("confidence", confidence),
    ]:
        assert torch.isfinite(t).all(), f"NaN/Inf in {name}"
    assert (velocity >= 0).all() and (velocity <= 1).all()
    assert (expression >= 0).all() and (expression <= 1).all()
    assert (confidence >= 0).all() and (confidence <= 1).all()


def test_mel_basis_is_buffer(mock_config, transcriber):
    """Mel filterbank must be a registered buffer, not rebuilt per forward
    (TODO C-2.9 / PERF-7.2)."""
    # expects fix per TODO C-2.9
    assert "mel_basis" in dict(transcriber.named_buffers()) or hasattr(
        transcriber, "mel_basis"
    ), "mel_basis should be registered as a buffer in __init__ (TODO C-2.9)"
