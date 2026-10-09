"""Model submodule tests requiring the optional PyTorch/NeMo/Mamba stack."""

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("nemo")
pytest.importorskip("mamba_ssm")

from models.mamba_separator import MambaSeparator  # noqa: E402
from models.mamba_transcriber import MambaTranscriber  # noqa: E402
from main import StemMidiModel  # noqa: E402


def _small_config():
    """Return a tiny config that avoids constructing Mamba blocks."""
    return {
        "audio": {
            "sample_rate": 16000,
            "n_fft": 64,
            "hop_length": 16,
            "n_mels": 8,
        },
        "separator": {
            "d_model": 8,
            "n_layer": 0,
            "d_state": 2,
            "d_conv": 2,
            "expand": 2,
        },
        "transcriber": {
            "d_model": 8,
            "n_layer": 0,
            "d_state": 2,
            "pitch_vocab_size": 128,
            "expression_heads": ["bend"],
            "onset_threshold": 0.5,
        },
    }


def test_nemo_submodules_construct_with_small_config():
    """The submodules use NeMo's exported NeuralModule base correctly."""
    config = _small_config()

    separator = MambaSeparator(config)
    transcriber = MambaTranscriber(config)

    assert "metrics" in separator.output_types
    assert separator.output_types["new_state_cache"].optional
    assert transcriber.cfg is config
    assert transcriber.mel_basis.shape == (8, 33)


def test_training_targets_are_optional_for_inference():
    """NeMo type-checking accepts omitted targets on inference calls."""
    input_types = StemMidiModel.input_types.fget(None)

    assert all(
        input_types[name].optional
        for name in ("target_guitar", "target_bass", "target_onsets", "target_pitch")
    )


def test_separator_forward_supports_an_empty_mamba_stack():
    """Small test configs with zero Mamba layers still produce valid outputs."""
    separator = MambaSeparator(_small_config())
    audio = torch.randn(1, 1, 1024)

    guitar, bass, residual, state, metrics = separator(audio)

    assert guitar.shape == audio.shape
    assert bass.shape == audio.shape
    assert residual.shape == audio.shape
    assert state is None
    assert metrics.shape == (1, 2)


def test_transcriber_mel_spectrogram_uses_cached_filter_bank():
    """Mel construction handles scalar config and produces finite features."""
    transcriber = MambaTranscriber(_small_config())
    audio = torch.randn(1, 1024)

    mel = transcriber._mel_spectrogram(audio)

    assert mel.shape[0] == 1
    assert mel.shape[1] == 8
    assert torch.isfinite(mel).all()


def test_phase_cancellation_flag_requires_negative_correlation():
    """Positive correlation is not labeled phase cancellation."""
    guitar = torch.full((1, 1, 32), 0.5)
    in_phase_bass = torch.full((1, 1, 32), 0.5)
    anti_phase_bass = -in_phase_bass

    in_phase_flags = StemMidiModel._detect_artifacts(None, guitar, in_phase_bass)
    anti_phase_flags = StemMidiModel._detect_artifacts(None, guitar, anti_phase_bass)

    assert "phase_cancellation" not in in_phase_flags
    assert "phase_cancellation" in anti_phase_flags
