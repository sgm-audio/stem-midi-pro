# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Tests for MIDI generation from metadata."""
import io

import mido
import pytest
from mido import MidiFile

from api import build_midi_from_events


def test_create_midi_from_events():
    """Should produce a valid MIDI file with note events."""
    midi_data = {
        "midi_events": [
            {
                "note": 60, "onset_frame": 0, "velocity": 100, "confidence": 0.9,
                "stem_energy": 0.5, "alignment_score": 0.85, "needs_review": False,
                "cc_127_value": 114, "quantization_suggestion": "grid",
            },
            {
                "note": 64, "onset_frame": 86, "velocity": 80, "confidence": 0.8,
                "stem_energy": 0.4, "alignment_score": 0.75, "needs_review": False,
                "cc_127_value": 102, "quantization_suggestion": "human",
            },
        ],
        "summary": {"avg_confidence": 0.85, "low_confidence_count": 0, "alignment_warnings": 0},
    }

    midi_bytes = build_midi_from_events(midi_data, "guitar")
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

    assert len(mid.tracks) > 0
    assert len(mid.tracks[0]) > 1  # has events beyond header

    # Check for note_on messages
    has_note = any(msg.type == "note_on" for msg in mid.tracks[0])
    assert has_note, "MIDI file should contain note_on messages"


def test_create_midi_empty_events():
    """Should produce a valid (but empty of notes) MIDI file."""
    midi_data = {
        "midi_events": [],
        "summary": {"avg_confidence": 0.0, "low_confidence_count": 0, "alignment_warnings": 0},
    }

    midi_bytes = build_midi_from_events(midi_data, "bass")
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

    assert len(mid.tracks) == 1
    # Should have meta messages but no note_on
    note_ons = [m for m in mid.tracks[0] if m.type == "note_on"]
    assert len(note_ons) == 0


def test_midi_parses_round_trip():
    """MIDI bytes should parse and re-save successfully."""
    midi_data = {
        "midi_events": [
            {"note": 60, "onset_frame": 100, "velocity": 100, "confidence": 0.9},
        ],
        "summary": {"avg_confidence": 0.9, "low_confidence_count": 0, "alignment_warnings": 0},
    }
    midi_bytes = build_midi_from_events(midi_data, "guitar")
    mid1 = MidiFile(file=io.BytesIO(midi_bytes))
    buf = io.BytesIO()
    mid1.save(file=buf)
    mid2 = MidiFile(file=io.BytesIO(buf.getvalue()))
    assert len(mid1.tracks) == len(mid2.tracks)
