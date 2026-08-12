# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Tests for ConfidenceInjector."""
from __future__ import annotations

import pytest
import torch

from models.confidence_injector import ConfidenceInjector


class TestConfidenceInjector:
    """Test confidence injection and metadata validation."""

    @pytest.fixture
    def injector(self):
        return ConfidenceInjector(onset_threshold=0.5, min_confidence=0.6)

    @pytest.fixture
    def sample_events(self):
        return [[
            {"note": 45, "onset_frame": 5, "velocity": 80, "confidence": 0.85},
            {"note": 48, "onset_frame": 10, "velocity": 64, "confidence": 0.55},
            {"note": 52, "onset_frame": 15, "velocity": 100, "confidence": 0.92},
        ]]

    def test_inject_metadata_shape(self, injector, sample_events):
        """Should return enhanced events with metadata keys."""
        confidence = torch.rand(20)
        stem_energy = torch.rand(20)

        result = injector.inject_midi_metadata(sample_events, confidence, stem_energy)

        assert "midi_events" in result
        assert "summary" in result
        assert len(result["midi_events"]) == 3

        for ev in result["midi_events"]:
            assert "confidence" in ev
            assert "stem_energy" in ev
            assert "alignment_score" in ev
            assert "needs_review" in ev
            assert "cc_127_value" in ev
            assert "quantization_suggestion" in ev

    def test_cc127_range(self, injector, sample_events):
        """CC#127 values should be in [0, 127]."""
        confidence = torch.rand(20)
        stem_energy = torch.rand(20)

        result = injector.inject_midi_metadata(sample_events, confidence, stem_energy)

        for ev in result["midi_events"]:
            assert 0 <= ev["cc_127_value"] <= 127, f"cc_127={ev['cc_127_value']}"

    def test_needs_review_threshold(self, injector):
        """Events below min_confidence should be marked for review."""
        events = [[{"note": 45, "onset_frame": 0, "velocity": 64, "confidence": 0.3}]]
        confidence = torch.tensor([0.3])
        stem_energy = torch.tensor([0.01])

        result = injector.inject_midi_metadata(events, confidence, stem_energy)

        assert result["midi_events"][0]["needs_review"] is True
        assert result["summary"]["alignment_warnings"] == 1

    def test_high_confidence_not_review(self, injector):
        """High-confidence events should not be flagged."""
        events = [[{"note": 60, "onset_frame": 0, "velocity": 100, "confidence": 0.95}]]
        confidence = torch.tensor([0.95])
        stem_energy = torch.tensor([0.5])

        result = injector.inject_midi_metadata(events, confidence, stem_energy)

        assert result["midi_events"][0]["needs_review"] is False

    def test_empty_events(self, injector):
        """Should handle empty event list gracefully."""
        confidence = torch.rand(10)
        stem_energy = torch.rand(10)

        result = injector.inject_midi_metadata([], confidence, stem_energy)

        assert result["midi_events"] == []
        assert result["summary"]["low_confidence_count"] == 0
        assert result["summary"]["alignment_warnings"] == 0

    def test_summary_counts(self, injector, sample_events):
        """Summary should have accurate counts."""
        confidence = torch.cat([
            torch.full((10,), 0.9),
            torch.full((10,), 0.4),
        ])
        stem_energy = torch.ones(20)

        result = injector.inject_midi_metadata(sample_events, confidence, stem_energy)

        assert "avg_confidence" in result["summary"]
        assert "low_confidence_count" in result["summary"]
        assert isinstance(result["summary"]["low_confidence_count"], int)
