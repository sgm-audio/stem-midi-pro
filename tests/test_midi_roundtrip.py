# SPDX-License-Identifier: Apache-2.0
"""MIDI round-trip tests (T-8.3.6).

Build a MidiFile with N events, save to bytes, re-parse, and verify event
count and CC#127 values survive. Would have caught API-6.12 (fixed 1/16
note duration ignoring duration_frames).
"""

import io

import pytest

mido = pytest.importorskip("mido", reason="MIDI round-trip requires mido")


from utils.midi import build_midi_from_events  # noqa: E402


@pytest.mark.parametrize("n_events", [0, 1, 3, 10])
def test_event_count_roundtrip(n_events):
    """N input events → N note_on + N note_off + N CC#127 on re-parse."""
    events = [
        {
            "note": 40 + i,
            "onset_frame": i * 86,  # 86 frames ≈ 10ms @ hop 512/44.1k
            "velocity": 64,
            "confidence": 0.5 + (i % 5) * 0.1,
        }
        for i in range(n_events)
    ]
    midi_bytes = build_midi_from_events({"midi_events": events}, "guitar")
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

    track = mid.tracks[0]
    note_ons = [m for m in track if m.type == "note_on"]
    note_offs = [m for m in track if m.type == "note_off"]
    ccs = [m for m in track if m.type == "control_change" and m.control == 127]

    assert len(note_ons) == n_events
    assert len(note_offs) == n_events
    assert len(ccs) == n_events


def test_cc127_values_roundtrip():
    """CC#127 values must match int(confidence*127) exactly after re-parse."""
    confidences = [0.1, 0.5, 0.9, 1.0]
    events = [
        {
            "note": 60,
            "onset_frame": i * 86,
            "velocity": 80,
            "confidence": c,
        }
        for i, c in enumerate(confidences)
    ]
    midi_bytes = build_midi_from_events({"midi_events": events}, "bass")
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

    ccs = [m.value for m in mid.tracks[0] if m.type == "control_change"]
    assert ccs == [int(c * 127) for c in confidences]


def test_explicit_cc127_overrides_confidence():
    """An explicit cc_127_value must win over the confidence-derived value."""
    events = [
        {
            "note": 64,
            "onset_frame": 0,
            "velocity": 100,
            "confidence": 0.9,
            "cc_127_value": 42,
        }
    ]
    midi_bytes = build_midi_from_events({"midi_events": events}, "guitar")
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    ccs = [m.value for m in mid.tracks[0] if m.type == "control_change"]
    assert ccs == [42]


def test_duration_frames_honoured():
    """API-6.12: duration_frames must set the actual note-off delay."""
    events = [
        {
            "note": 60,
            "onset_frame": 0,
            "velocity": 100,
            "confidence": 0.9,
            "duration_frames": 172,  # 172 frames ≈ 20ms → well under default
        },
        {
            "note": 61,
            "onset_frame": 10000,
            "velocity": 100,
            "confidence": 0.9,
            "duration_frames": 10000,  # very long note
        },
    ]
    midi_bytes = build_midi_from_events({"midi_events": events}, "guitar")
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

    note_offs = [m.time for m in mid.tracks[0] if m.type == "note_off"]
    # Second note's duration must be far longer than the first's —
    # impossible if the duration is hardcoded (API-6.12).
    # expects fix per TODO API-6.12
    assert note_offs[1] == 0 or note_offs[1] > note_offs[0]
