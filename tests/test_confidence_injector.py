# SPDX-License-Identifier: Apache-2.0
"""ConfidenceInjector unit tests (T-8.3.3)."""

import pytest

torch = pytest.importorskip("torch", reason="requires torch")

from models.confidence_injector import ConfidenceInjector  # noqa: E402


@pytest.fixture
def injector():
    return ConfidenceInjector(onset_threshold=0.5, min_confidence=0.6)


def _make_events(frames=(10, 50)):
    return [{"note": 60 + i, "onset_frame": f, "velocity": 80} for i, f in enumerate(frames)]


def test_alignment_score_bounded(injector):
    """Alignment score = confidence * sigmoid(energy*10) ∈ (0, confidence]."""
    events = _make_events()
    confidence = torch.linspace(0.01, 0.99, 100)
    energy = torch.linspace(0.0, 1.0, 100)

    out = injector.inject_midi_metadata(events, confidence, energy)
    for ev in out["midi_events"]:
        assert 0.0 < ev["alignment_score"] <= ev["confidence"] + 1e-6


def test_needs_review_threshold(injector):
    """Events below min_confidence alignment must be flagged needs_review."""
    # frame 0: low confidence (0.1) → alignment << 0.6 → needs_review
    # frame 1: high confidence (0.99) * sigmoid(hot energy) > 0.6 → not flagged
    events = _make_events(frames=(0, 1))
    confidence = torch.tensor([0.1, 0.99])
    energy = torch.tensor([1.0, 1.0])  # sigmoid(10) ≈ 1.0

    out = injector.inject_midi_metadata(events, confidence, energy)
    ev0, ev1 = out["midi_events"]
    assert ev0["needs_review"] is True
    assert ev1["needs_review"] is False


def test_cc127_mapping(injector):
    """cc_127_value must equal int(confidence * 127)."""
    events = _make_events(frames=(0, 1))
    confidence = torch.tensor([0.5, 1.0])
    energy = torch.ones(2)

    out = injector.inject_midi_metadata(events, confidence, energy)
    assert out["midi_events"][0]["cc_127_value"] == int(0.5 * 127)
    assert out["midi_events"][1]["cc_127_value"] == 127


def test_summary_counts(injector):
    """Summary aggregates: avg, low-confidence count, alignment warnings."""
    events = _make_events(frames=(0, 1, 2))
    confidence = torch.tensor([0.3, 0.7, 0.9])
    energy = torch.ones(3)

    out = injector.inject_midi_metadata(events, confidence, energy)
    summary = out["summary"]

    assert summary["avg_confidence"] == pytest.approx((0.3 + 0.7 + 0.9) / 3)
    assert summary["low_confidence_count"] == 1  # only 0.3 below 0.6
    assert summary["alignment_warnings"] == sum(1 for e in out["midi_events"] if e["needs_review"])


def test_empty_events(injector):
    """No events should produce empty list with valid summary."""
    confidence = torch.tensor([0.5])
    energy = torch.tensor([0.5])

    out = injector.inject_midi_metadata([], confidence, energy)
    assert out["midi_events"] == []
    assert out["summary"]["alignment_warnings"] == 0
