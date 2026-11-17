# SPDX-License-Identifier: Apache-2.0
"""Are the quality gates making decisions about *this* audio, or about nothing?

``tests/test_quality_gates.py`` covers the tier arithmetic with hand-built
``ProcessingReport`` objects — good unit coverage of a pure function, but it
never asks whether the numbers fed into that function mean anything. These
tests close the loop: they drive real inference, then check the gate inputs and
the review flags that reach the user.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires torch")

from models.confidence_injector import ConfidenceInjector  # noqa: E402
from utils.quality_gates import (  # noqa: E402
    ProcessingReport,
    QualityTier,
    route_by_quality,
)

SR = 44100


def _sine(seconds: float = 1.0, amp: float = 0.3) -> torch.Tensor:
    t = torch.arange(int(seconds * SR)) / SR
    return (amp * torch.sin(2 * torch.pi * 220 * t)).reshape(1, 1, -1)


def _events(n: int = 4) -> list[dict]:
    return [
        {"note": 52 + i, "onset_frame": i * 20, "velocity": 90, "confidence": 0.99}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# The gate is fed by real inference (the missing integration test)
# ---------------------------------------------------------------------------


def test_gate_inputs_come_from_real_inference(mock_model):
    """A report built by the model must be internally consistent and finite.

    This is the integration hop the existing gate tests skip entirely: they
    construct ``ProcessingReport`` by hand, so a pipeline that fed the gate
    NaN, zeros, or the wrong tensor would still pass all of them.
    """
    with torch.inference_mode():
        out = mock_model(_sine())
    report = out["processing_report"]
    routing = out["routing_decision"]

    assert isinstance(report.quality_tier, QualityTier)
    assert 0.0 <= report.avg_confidence <= 1.0
    assert 0.0 <= report.phase_coherence <= 1.0
    assert np.isfinite(report.centroid_separation_db)
    assert routing["action"] in {"direct_download", "editor_launch", "fallback_options"}
    expected = {
        QualityTier.STUDIO: "direct_download",
        QualityTier.DRAFT: "editor_launch",
        QualityTier.COMPLEX: "fallback_options",
    }[report.quality_tier]
    assert routing["action"] == expected


def test_random_weights_never_reach_studio(mock_model):
    """An untrained model must not be routed as studio quality.

    The repo ships no checkpoints (checkpoints/README.md), and ``api.load_model``
    falls back to ``StemMidiModel(config)`` with random weights; the gate is the
    only thing standing between that and a "direct download" badge.
    """
    with torch.inference_mode():
        report = mock_model(_sine())["processing_report"]
    assert report.quality_tier is not QualityTier.STUDIO


def test_nan_metrics_cannot_be_routed_as_studio():
    """Comparisons with NaN are False, so a poisoned report must degrade safely."""
    report = ProcessingReport(
        centroid_separation_db=float("nan"),
        phase_coherence=float("nan"),
        avg_confidence=float("nan"),
        artifact_flags=["none"],
        low_confidence_notes=0,
        config={"quality_gates": {"studio_confidence_threshold": 0.85, "min_si_sdr": 20.0}},
    )
    assert report.quality_tier is not QualityTier.STUDIO


def test_gate_thresholds_are_config_driven():
    """The same numbers must tier differently under different configs."""
    kwargs = dict(
        centroid_separation_db=25.0,
        phase_coherence=0.95,
        avg_confidence=0.80,
        artifact_flags=["none"],
        low_confidence_notes=0,
    )
    lenient = ProcessingReport(
        config={"quality_gates": {"studio_confidence_threshold": 0.75}}, **kwargs
    )
    strict = ProcessingReport(
        config={"quality_gates": {"studio_confidence_threshold": 0.90}}, **kwargs
    )
    assert lenient.quality_tier is QualityTier.STUDIO
    assert strict.quality_tier is not QualityTier.STUDIO
    assert route_by_quality(strict)["editor_launch"] is True


def test_any_artifact_flag_blocks_studio():
    report = ProcessingReport(
        centroid_separation_db=30.0,
        phase_coherence=0.99,
        avg_confidence=0.99,
        artifact_flags=["clipping"],
        low_confidence_notes=0,
        config={"quality_gates": {"studio_confidence_threshold": 0.85, "min_si_sdr": 20.0}},
    )
    assert report.quality_tier is QualityTier.COMPLEX


# ---------------------------------------------------------------------------
# Per-note review flags: driven by confidence, or by loudness?
# ---------------------------------------------------------------------------


def test_review_flag_is_independent_of_output_gain():
    """Scaling a stem up or down must not change which notes need review.

    ``alignment_score = confidence * sigmoid(energy * 10)`` mixes two unrelated
    quantities: at energy 1e-4 the factor is ~0.5, so a perfectly confident note
    in a quiet passage is flagged for review, while the identical note in a loud
    passage is not. Review should track confidence; loudness is already reported
    separately as ``stem_energy``.
    """
    injector = ConfidenceInjector(min_confidence=0.6)
    confidence = torch.full((64,), 0.99)
    events = _events(4)

    quiet = injector.inject_midi_metadata(events, confidence, torch.full((64,), 1e-4))
    loud = injector.inject_midi_metadata(events, confidence, torch.full((64,), 1.0))

    quiet_flags = [e["needs_review"] for e in quiet["midi_events"]]
    loud_flags = [e["needs_review"] for e in loud["midi_events"]]
    assert (
        quiet_flags == loud_flags
    ), f"review flags changed with gain alone: quiet={quiet_flags} loud={loud_flags}"


def test_low_confidence_notes_are_flagged():
    injector = ConfidenceInjector(min_confidence=0.6)
    out = injector.inject_midi_metadata(_events(2), torch.full((64,), 0.05), torch.full((64,), 1.0))
    assert all(e["needs_review"] for e in out["midi_events"])


def test_low_confidence_count_counts_notes_not_frames():
    """``ProcessingReport.low_confidence_notes`` must count notes.

    With 2 events over a 64-frame confidence grid the summary reports 64, so
    the UI can claim more low-confidence notes than the MIDI contains.
    """
    injector = ConfidenceInjector(min_confidence=0.6)
    out = injector.inject_midi_metadata(_events(2), torch.full((64,), 0.05), torch.full((64,), 1.0))
    assert out["summary"]["low_confidence_count"] == 2


def test_high_confidence_loud_notes_are_not_flagged():
    injector = ConfidenceInjector(min_confidence=0.6)
    out = injector.inject_midi_metadata(_events(2), torch.full((64,), 0.99), torch.full((64,), 1.0))
    assert not any(e["needs_review"] for e in out["midi_events"])


def test_cc127_reflects_confidence_not_alignment():
    """The DAW-visible confidence cue must move with confidence alone."""
    injector = ConfidenceInjector(min_confidence=0.6)
    events = _events(1)
    quiet = injector.inject_midi_metadata(events, torch.full((64,), 0.9), torch.full((64,), 1e-6))
    loud = injector.inject_midi_metadata(events, torch.full((64,), 0.9), torch.full((64,), 1.0))
    assert quiet["midi_events"][0]["cc_127_value"] == loud["midi_events"][0]["cc_127_value"]


def test_events_outside_the_confidence_grid_are_handled_safely():
    """An onset frame past the model's frame count must not raise or NaN."""
    injector = ConfidenceInjector(min_confidence=0.6)
    events = [{"note": 60, "onset_frame": 10_000, "velocity": 90, "confidence": 0.99}]
    out = injector.inject_midi_metadata(events, torch.full((8,), 0.9), torch.full((8,), 0.5))
    (event,) = out["midi_events"]
    assert np.isfinite(event["alignment_score"])
    assert event["needs_review"] is True  # zero confidence for an unseen frame


# ---------------------------------------------------------------------------
# Provenance: does anything tell the user the model was never trained?
# ---------------------------------------------------------------------------


def test_report_records_weights_provenance(mock_model):
    """The delivered report must state whether real weights were loaded.

    ``checkpoints/`` contains only a README, and ``api.load_model`` constructs a
    ``StemMidiModel`` with random weights when ``MODEL_CHECKPOINT_PATH`` is
    unset. Nothing in ``processing_report.json``, ``/model-info`` or the logs
    distinguishes that from a trained model, so a downstream operator cannot
    tell "no trained model" from "trained model doing badly".
    """
    with torch.inference_mode():
        report = mock_model(_sine())["processing_report"]
    assert hasattr(report, "weights_loaded") and report.weights_loaded is False
