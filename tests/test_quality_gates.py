# SPDX-License-Identifier: Apache-2.0
"""Tests for quality gating and routing."""

import pytest

from utils.quality_gates import ProcessingReport, QualityTier, route_by_quality


def test_studio_tier():
    """High confidence + high separation proxy should pass the heuristic gate."""
    report = ProcessingReport(
        centroid_separation_db=25.0,
        phase_coherence=0.9,
        avg_confidence=0.95,
        artifact_flags=["none"],
        low_confidence_notes=0,
    )
    assert report.quality_tier == QualityTier.STUDIO


def test_draft_tier():
    """Medium confidence should route to draft."""
    report = ProcessingReport(
        centroid_separation_db=15.0,
        phase_coherence=0.7,
        avg_confidence=0.75,
        artifact_flags=["none"],
        low_confidence_notes=5,
    )
    assert report.quality_tier == QualityTier.DRAFT


def test_complex_tier():
    """Low confidence + artifacts should route to complex."""
    report = ProcessingReport(
        centroid_separation_db=5.0,
        phase_coherence=0.3,
        avg_confidence=0.5,
        artifact_flags=["clipping", "noise_floor"],
        low_confidence_notes=20,
    )
    assert report.quality_tier == QualityTier.COMPLEX


def test_config_overrides_thresholds():
    """Config dict should override default thresholds."""
    config = {
        "quality_gates": {"studio_confidence_threshold": 0.90, "min_centroid_separation_db": 25.0}
    }
    report = ProcessingReport(
        centroid_separation_db=22.0,
        phase_coherence=0.9,
        avg_confidence=0.88,
        artifact_flags=["none"],
        low_confidence_notes=0,
        config=config,
    )
    assert report.quality_tier != QualityTier.STUDIO  # Doesn't meet the 25.0 proxy threshold


def test_route_by_quality_studio():
    """Studio report should produce direct_download action."""
    report = ProcessingReport(
        centroid_separation_db=25.0,
        phase_coherence=0.9,
        avg_confidence=0.95,
        artifact_flags=["none"],
        low_confidence_notes=0,
    )
    decision = route_by_quality(report)
    assert decision["action"] == "direct_download"


def test_route_by_quality_draft():
    """Draft report should request manual DAW review."""
    report = ProcessingReport(
        centroid_separation_db=15.0,
        phase_coherence=0.7,
        avg_confidence=0.75,
        artifact_flags=["none"],
        low_confidence_notes=5,
    )
    decision = route_by_quality(report)
    assert decision["action"] == "manual_review"
    assert decision["review_in_daw"]
    assert not decision["editor_launch"]


def test_route_by_quality_complex():
    """Complex report should request manual DAW review options."""
    report = ProcessingReport(
        centroid_separation_db=5.0,
        phase_coherence=0.3,
        avg_confidence=0.5,
        artifact_flags=["clipping"],
        low_confidence_notes=20,
    )
    decision = route_by_quality(report)
    assert decision["action"] == "manual_review"
    assert decision["review_in_daw"]
    assert len(decision["options"]) == 3


# ---------------------------------------------------------------------------
# T-8.3.9: artifact flags force COMPLEX regardless of confidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [["clipping"], ["noise_floor"], ["phase_cancellation"], ["clipping", "noise_floor"]],
)
def test_artifact_flags_force_complex_tier(flags):
    """Artifact flags force COMPLEX tier regardless of confidence (T-8.3.9).

    Even with a stellar proxy and confidence, real artifacts mean the output
    cannot be studio or draft quality.
    """
    report = ProcessingReport(
        centroid_separation_db=30.0,
        phase_coherence=0.99,
        avg_confidence=0.99,
        artifact_flags=flags,
        low_confidence_notes=0,
    )
    assert report.quality_tier == QualityTier.COMPLEX
    decision = route_by_quality(report)
    assert decision["action"] == "manual_review"
