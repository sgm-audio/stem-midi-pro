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
    """Alignment score = confidence * energy_score ∈ (0, confidence].

    energy_score is sigmoid(2*(logE - mean(logE))) ∈ (0, 1), so it can never
    exceed 1 and can never be NaN (the old sigmoid(energy*10) saturated to exactly
    1.0 for any energy > 0.5 and produced NaN for NaN input).
    """
    events = _make_events()
    confidence = torch.linspace(0.01, 0.99, 100)
    energy = torch.linspace(0.0, 1.0, 100)

    out = injector.inject_midi_metadata(events, confidence, energy)
    for ev in out["midi_events"]:
        assert 0.0 < ev["alignment_score"] <= ev["confidence"] + 1e-6


def test_needs_review_threshold(injector):
    """Low-confidence or near-silent notes must be flagged needs_review."""
    # frame 0: confidence 0.1 < 0.6 → needs_review
    # frame 1: confidence 0.99 on typical energy → not flagged
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


def test_energy_grid_mismatch_raises(injector):
    """Per-sample energy passed as frame energy must fail loudly (H-3)."""
    events = _make_events()
    confidence = torch.linspace(0.0, 1.0, 100)
    with pytest.raises(ValueError, match="frame grid"):
        injector.inject_midi_metadata(events, confidence, torch.randn(44100))


def test_non_finite_confidence_is_rejected(injector):
    """NaN/Inf metadata must fail loudly, not be written out (API-6.9/H-6)."""
    events = _make_events(frames=(0, 1))
    energy = torch.ones(2)
    with pytest.raises(ValueError, match="non-finite"):
        injector.inject_midi_metadata(events, torch.tensor([float("nan"), 0.9]), energy)
    with pytest.raises(ValueError, match="non-finite"):
        injector.inject_midi_metadata(
            events, torch.tensor([0.4, 0.9]), torch.tensor([1.0, float("inf")])
        )


def test_review_flag_is_confidence_not_energy(injector):
    """`needs_review` follows confidence; energy stays a diagnostic.

    Deriving the flag from `alignment_score` (conf * sigmoid(energy)) meant a
    confident pianissimo passage was always flagged and the same events flipped
    on gain alone, and it disagreed with `low_confidence_count`.
    """
    events = _make_events(frames=(0, 1))
    confidence = torch.tensor([0.9, 0.9])  # same confidence, different energy
    energy = torch.tensor([1e-9, 1.0])  # note 0 sits on near-silence
    out = injector.inject_midi_metadata(events, confidence, energy)
    silent, loud = out["midi_events"]

    assert silent["alignment_score"] < loud["alignment_score"]  # diagnostic reacts to energy
    assert silent["needs_review"] is False  # ...but confidence decides the flag
    assert loud["needs_review"] is False

    quiet = injector.inject_midi_metadata(
        _make_events(frames=(0, 1)), torch.tensor([0.9, 0.30]), torch.ones(2)
    )
    assert quiet["midi_events"][1]["needs_review"] is True
    assert quiet["summary"]["low_confidence_count"] == 1
    assert quiet["summary"]["total_notes"] == 2
