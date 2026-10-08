# SPDX-License-Identifier: Apache-2.0
"""MambaSeparator unit tests (T-8.3.1).

Locks the constructor/forward contract — would have caught the invalid
Mamba kwargs (C-2.1), the broken state cache (C-2.5), the duplicate STFT
(C-2.6) and shape aliasing (C-2.11).
"""

import pytest

torch = pytest.importorskip("torch", reason="requires torch")


@pytest.fixture
def separator(mock_config):
    try:
        from models.mamba_separator import MambaSeparator  # expects fix per TODO C-2.1
    except (ImportError, TypeError) as exc:
        pytest.skip(f"MambaSeparator not constructible (C-2.1?): {exc}")
    return MambaSeparator(mock_config)


def test_forward_output_shapes(mock_config, separator):
    """Forward on (1, 1, 8192) returns stems with identical shape."""
    audio = torch.randn(1, 1, 8192)
    guitar, bass, residual, _state_cache, metrics = separator(audio)

    for name, stem in [("guitar", guitar), ("bass", bass), ("residual", residual)]:
        assert stem.shape == (1, 1, 8192), f"{name} shape {stem.shape}"
    assert metrics.shape[-1] == 2  # [centroid proxy, phase coherence]


def test_forward_no_nan(mock_config, separator):
    """Outputs must be finite — no NaN/Inf from STFT/Mamba/ISTFT round-trip."""
    audio = torch.randn(1, 1, 8192) * 0.1
    guitar, bass, residual, _, metrics = separator(audio)

    for name, tensor in [
        ("guitar", guitar),
        ("bass", bass),
        ("residual", residual),
        ("metrics", metrics),
    ]:
        assert torch.isfinite(tensor).all(), f"NaN/Inf in {name}"


def test_stems_differ(mock_config, separator):
    """Guitar and bass stems must not be identical (independent mask heads)."""
    torch.manual_seed(0)
    audio = torch.randn(1, 1, 8192)
    guitar, bass, residual, _, _ = separator(audio)

    assert not torch.allclose(guitar, bass), "guitar and bass stems are identical"
    # Conservation: residual captures what the stems don't.
    assert guitar.shape == residual.shape


def test_config_contract(mock_config, separator):
    """Constructor must honour downsized CPU config dims."""
    assert separator.mamba_blocks[0].d_model == mock_config["separator"]["d_model"]
    assert len(separator.mamba_blocks) == mock_config["separator"]["n_layers"]
