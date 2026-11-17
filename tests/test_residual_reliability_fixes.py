# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the residual reliability findings.

PRs #11 and #14 already fixed most of `docs/reviews/2026-11-17-reliability-review.md`.
This file pins the five findings that were still live on `ee8dd62`, each of which
was reproduced on that commit before being fixed (see
`docs/reviews/2026-11-17-upstream-status-audit.md` for the measurements).

  §1/§2  peak memory must not scale with file length (block + inverse STFT)
  §3     log-mel bounded on BOTH sides; silence must not crash
  §4     non-finite metrics must not reach a happy tier
  §9     the centroid proxy must not gate STUDIO
  §4b    a silent render must not ship fabricated MIDI
"""

import math

import pytest

torch = pytest.importorskip("torch", reason="requires torch")


# ---------------------------------------------------------------------------
# §1/§2 — memory
# ---------------------------------------------------------------------------


def test_chunked_istft_is_bit_exact():
    """The windowed inverse STFT must match whole-file `torch.istft`.

    The chunked reconstruction normalises by a per-window window sum. Without
    ``pad`` frames of context on each side of a window the samples near the
    seam are normalised by a truncated sum and the reconstruction develops
    ~3e-3 peak steps (measured), which is audible on transients. This test
    forces several small window sizes so a regression cannot hide behind a
    chunk size that happens to cover the whole signal.
    """
    import models.mamba_separator as module
    from models.mamba_separator import MambaSeparator

    cfg = {
        "separator": {"d_model": 64, "n_layers": 2, "d_state": 16, "d_conv": 4, "expand": 2},
        "audio": {"sample_rate": 44100, "n_fft": 1024, "hop_length": 256},
    }
    sep = MambaSeparator(cfg).eval()
    n_fft, hop = sep.n_fft, sep.hop_length
    samples = hop * 40
    torch.manual_seed(0)
    audio = torch.randn(1, samples)

    with torch.inference_mode():
        spec = torch.stft(
            audio, n_fft=n_fft, hop_length=hop, window=sep.window, return_complex=True
        )
        reference = torch.istft(
            spec, n_fft=n_fft, hop_length=hop, window=sep.window, length=samples
        )
        original = module._ISTFT_CHUNK_FRAMES
        try:
            for chunk in (4, 9, 32):
                module._ISTFT_CHUNK_FRAMES = chunk
                got = sep._istft_chunked(spec, length=samples)
                max_diff = (got - reference).abs().max().item()
                assert max_diff < 1e-5, (
                    f"chunk={chunk} frames: windowed istft differs from whole-file "
                    f"istft by {max_diff:.3e} (seam normalisation bug)"
                )
        finally:
            module._ISTFT_CHUNK_FRAMES = original


def test_block_memory_is_independent_of_sequence_length():
    """A Mamba block must not allocate full-sequence intermediates.

    `in_proj`, `conv1d`, `silu`, `x_proj` and the gated output all used to run
    on the whole (B, L, ...) sequence. At 600 s (L=51,680) one block allocated
    2538 MB on its own. The whole block is now windowed.
    """
    from models._mamba_compat import Mamba

    block = Mamba(128, d_state=16, d_conv=4, expand=2).eval()
    d_inner = block.d_inner

    def peak_mb(fn):
        import resource

        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        fn()
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 - before

    short = torch.randn(1, 512, 128) * 0.1
    long = torch.randn(1, 512 * 16, 128) * 0.1

    with torch.inference_mode():
        block(short)  # warm allocator / import paths
        # One full-sequence intermediate would be L * d_inner * 4 bytes:
        # 8192 * 256 * 4 = 8 MB. The windowed block must stay far below that.
        full_seq_mb = (512 * 16) * d_inner * 4 / 2**20
        growth = peak_mb(lambda: block(long))

    assert growth < full_seq_mb, (
        f"one block grew RSS by {growth:.1f} MB for L=8192, where a single "
        f"full-sequence intermediate would be {full_seq_mb:.1f} MB — the "
        "block is not windowed"
    )


# ---------------------------------------------------------------------------
# §3 — the log-mel must be bounded on both sides
# ---------------------------------------------------------------------------


def test_log_mel_is_bounded_both_sides(mock_config):
    """Extreme inputs must stay finite and capped.

    With only a floor (`clamp(min=1e-8)`), a loud/DC stem reached log-mel
    19.36 and the SSM overflowed fp32 to NaN. That NaN then hit the
    finiteness guard, so a *silent* WAV raised RuntimeError instead of
    returning an empty transcription.
    """
    from models.mamba_transcriber import MambaTranscriber

    tr = MambaTranscriber(mock_config).eval()
    for audio in (
        torch.zeros(1, 4096),
        torch.full((1, 4096), 1e6),
        torch.full((1, 4096), -1e6),
        1e5 * torch.sin(torch.arange(4096) * 0.1).view(1, -1),
    ):
        with torch.inference_mode():
            mel = tr._mel_spectrogram(audio)
        assert torch.isfinite(mel).all(), "log-mel produced NaN/Inf"
        assert mel.max() <= math.log(1e3) + 1e-3, (
            f"log-mel max {mel.max().item():.3f} exceeds the log(1e3) ceiling"
        )


def test_silent_input_does_not_raise(mock_config):
    """A silent file is a legitimate input, not a corrupt one."""
    from main import StemMidiModel

    cfg = {**mock_config}
    cfg.pop("backends", None)
    model = StemMidiModel(cfg).eval()

    silent = torch.zeros(1, 1, 44100)
    with torch.inference_mode():
        out = model.forward(silent)

    assert out["guitar_stem"].abs().max() < 1e-6
    assert torch.isfinite(out["guitar_stem"]).all()


# ---------------------------------------------------------------------------
# §4 — non-finite metrics must not reach a happy tier
# ---------------------------------------------------------------------------


def test_nan_metrics_are_complex_not_draft():
    """`NaN >= x` is False, so NaN used to fall through to DRAFT."""
    from utils.quality_gates import ProcessingReport, QualityTier, route_by_quality

    report = ProcessingReport(
        centroid_separation_db=float("nan"),
        phase_coherence=0.5,
        avg_confidence=0.75,  # would have been DRAFT
        artifact_flags=["none"],
        low_confidence_notes=0,
    )
    assert report.invalid
    assert report.quality_tier == QualityTier.COMPLEX
    assert "non_finite" in report.artifact_flags
    assert route_by_quality(report)["action"] != "editor_launch"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_every_non_finite_metric_is_caught(value):
    from utils.quality_gates import ProcessingReport, QualityTier

    for field in ("centroid_separation_db", "phase_coherence", "avg_confidence"):
        kwargs = {
            "centroid_separation_db": 25.0,
            "phase_coherence": 0.9,
            "avg_confidence": 0.95,
            "artifact_flags": ["none"],
            "low_confidence_notes": 0,
        }
        kwargs[field] = value
        report = ProcessingReport(**kwargs)
        assert report.invalid, f"{field}={value} not flagged"
        assert report.quality_tier == QualityTier.COMPLEX


def test_unknown_artifact_flag_cannot_bypass_the_check():
    from utils.quality_gates import ProcessingReport, QualityTier

    report = ProcessingReport(
        centroid_separation_db=30.0,
        phase_coherence=0.99,
        avg_confidence=0.99,
        artifact_flags=["some_new_flag"],
        low_confidence_notes=0,
    )
    assert report.artifact_flags == ["unknown:some_new_flag"]
    assert report.quality_tier == QualityTier.COMPLEX


def test_report_as_dict_is_json_safe():
    """Bare NaN in json.dumps is invalid RFC 8259 and throws in JSON.parse."""
    import json

    from utils.quality_gates import ProcessingReport

    report = ProcessingReport(
        centroid_separation_db=float("nan"),
        phase_coherence=0.5,
        avg_confidence=0.75,
        artifact_flags=["none"],
        low_confidence_notes=0,
    )
    payload = report.as_dict()
    assert payload["centroid_separation_db"] is None
    assert payload["invalid"] is True
    # allow_nan=False turns any surviving NaN into a hard error.
    json.dumps(payload, allow_nan=False)


# ---------------------------------------------------------------------------
# §9 — the proxy must not gate STUDIO
# ---------------------------------------------------------------------------


def test_one_bin_of_separation_does_not_decide_studio():
    """One FFT bin reported exactly 20.00 dB and cleared `min_si_sdr: 20`."""
    from utils.quality_gates import ProcessingReport, QualityTier

    one_bin = 10 * math.log10(1.0 + 1e-8) + 20
    assert one_bin >= 20.0  # cleared the old gate

    # A saturated proxy cannot rescue a render whose confidence is below the
    # studio bar: the proxy is not consulted at all.
    report = ProcessingReport(
        centroid_separation_db=30.0,
        phase_coherence=0.99,
        avg_confidence=0.80,
        artifact_flags=["none"],
        low_confidence_notes=0,
    )
    assert report.quality_tier == QualityTier.DRAFT


def test_low_confidence_notes_block_studio():
    from utils.quality_gates import ProcessingReport, QualityTier

    report = ProcessingReport(
        centroid_separation_db=30.0,
        phase_coherence=0.99,
        avg_confidence=0.95,
        artifact_flags=["none"],
        low_confidence_notes=3,
    )
    assert report.quality_tier == QualityTier.DRAFT


def test_legacy_min_si_sdr_key_warns_and_does_not_gate():
    import warnings

    from utils.quality_gates import ProcessingReport

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = ProcessingReport(
            centroid_separation_db=22.0,
            phase_coherence=0.9,
            avg_confidence=0.9,
            artifact_flags=["none"],
            low_confidence_notes=0,
            config={"quality_gates": {"min_si_sdr": 25.0}},
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    # 22 < the legacy 25 threshold, but the proxy no longer decides the tier.
    assert report.quality_tier.value == "studio"


# ---------------------------------------------------------------------------
# §4b — a silent render must not ship fabricated MIDI
# ---------------------------------------------------------------------------


def test_silence_predicate_is_shared_by_gate_and_midi(mock_config):
    """The artifact flag and the emitted events must agree on "silent"."""
    from main import StemMidiModel

    cfg = {**mock_config}
    cfg.pop("backends", None)
    model = StemMidiModel(cfg).eval()

    silent = torch.zeros(1, 1, 44100)
    assert StemMidiModel._stems_are_silent(silent, silent)
    assert model._detect_artifacts(silent, silent) == ["silent_output"]

    nan_stem = torch.full((1, 1, 64), float("nan"))
    assert StemMidiModel._stem_is_silent(nan_stem)


def test_real_audio_is_not_treated_as_silent():
    from main import StemMidiModel

    loud = 0.2 * torch.sin(2 * math.pi * 196 * torch.arange(44100) / 44100)
    loud = loud.view(1, 1, -1)
    assert not StemMidiModel._stem_is_silent(loud)
    assert not StemMidiModel._stems_are_silent(loud, loud * 0.5)


def test_silent_stem_short_circuits_to_zero_events(mock_config):
    """`_enhance` is the choke point for BOTH stems.

    A guitar-only guard left `midi_bass` shipping ~129 fabricated notes, so
    this asserts on the helper rather than on the guitar path alone.
    """
    from main import StemMidiModel

    cfg = {**mock_config}
    cfg.pop("backends", None)
    model = StemMidiModel(cfg).eval()

    silent = torch.zeros(1, 1, 4096)
    with torch.inference_mode():
        tr = model._transcribe(silent)
        meta, _conf = model._enhance(tr, silent)

    assert meta["midi_events"] == []
