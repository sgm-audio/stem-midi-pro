"""Tests for quality gating and routing."""
import pytest
from utils.quality_gates import ProcessingReport, QualityTier, route_by_quality

def test_studio_tier():
    """High confidence + high SI-SDR should route to studio."""
    report = ProcessingReport(
        si_sdr=25.0, phase_coherence=0.9, avg_confidence=0.95,
        artifact_flags=["none"], low_confidence_notes=0,
    )
    assert report.quality_tier == QualityTier.STUDIO

def test_draft_tier():
    """Medium confidence should route to draft."""
    report = ProcessingReport(
        si_sdr=15.0, phase_coherence=0.7, avg_confidence=0.75,
        artifact_flags=["none"], low_confidence_notes=5,
    )
    assert report.quality_tier == QualityTier.DRAFT

def test_complex_tier():
    """Low confidence + artifacts should route to complex."""
    report = ProcessingReport(
        si_sdr=5.0, phase_coherence=0.3, avg_confidence=0.5,
        artifact_flags=["clipping", "noise_floor"], low_confidence_notes=20,
    )
    assert report.quality_tier == QualityTier.COMPLEX

def test_config_overrides_thresholds():
    """Config dict should override default thresholds."""
    config = {'quality_gates': {'studio_confidence_threshold': 0.90, 'min_si_sdr': 25.0}}
    report = ProcessingReport(
        si_sdr=22.0, phase_coherence=0.9, avg_confidence=0.88,
        artifact_flags=["none"], low_confidence_notes=0,
        config=config,
    )
    assert report.quality_tier != QualityTier.STUDIO  # Doesn't meet 25.0 SI-SDR

def test_route_by_quality_studio():
    """Studio report should produce direct_download action."""
    report = ProcessingReport(
        si_sdr=25.0, phase_coherence=0.9, avg_confidence=0.95,
        artifact_flags=["none"], low_confidence_notes=0,
    )
    decision = route_by_quality(report)
    assert decision['action'] == 'direct_download'

def test_route_by_quality_draft():
    """Draft report should produce editor_launch action."""
    report = ProcessingReport(
        si_sdr=15.0, phase_coherence=0.7, avg_confidence=0.75,
        artifact_flags=["none"], low_confidence_notes=5,
    )
    decision = route_by_quality(report)
    assert decision['action'] == 'editor_launch'
    assert decision['editor_launch'] == True

def test_route_by_quality_complex():
    """Complex report should produce fallback_options."""
    report = ProcessingReport(
        si_sdr=5.0, phase_coherence=0.3, avg_confidence=0.5,
        artifact_flags=["clipping"], low_confidence_notes=20,
    )
    decision = route_by_quality(report)
    assert decision['action'] == 'fallback_options'
    assert len(decision['options']) == 3
